import os
import mmcv
import torch
import numpy as np
import os.path as osp
from mmdet3d.datasets.builder import PIPELINES
from numpy.linalg import inv
from mmcv.runner import get_dist_info
from mmdet3d.core.points import BasePoints
from ..utils import compose_ego2img


@PIPELINES.register_module()
class LoadOcc3DFromFile:

    def __init__(self, occ_root, ignore_class_names=[]):
        self.occ_root = occ_root
        self.ignore_class_names = ignore_class_names
        self.occ_class_names = [
            'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
            'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
            'driveable_surface', 'other_flat', 'sidewalk',
            'terrain', 'manmade', 'vegetation', 'free'
        ]

    def __call__(self, results):
        scene_name, sample_token = results['scene_name'], results['sample_token']
        occ_file = osp.join(self.occ_root, scene_name, sample_token, 'labels.npz')
        # load lidar and camera visible label
        occ_labels = np.load(occ_file)
        mask_lidar = occ_labels['mask_lidar'].astype(np.bool_)  # [200, 200, 16]
        mask_camera = occ_labels['mask_camera'].astype(np.bool_)  # [200, 200, 16]
        results['mask_lidar'] = mask_lidar
        results['mask_camera'] = mask_camera

        semantics = occ_labels['semantics']  # [200, 200, 16]
        for class_id in range(len(self.occ_class_names) - 1):
            mask = semantics == class_id
            if mask.sum() == 0:
                continue
            if self.occ_class_names[class_id] in self.ignore_class_names:
                semantics[mask] = self.num_classes - 1
        results['voxel_semantics'] = semantics
        return results


@PIPELINES.register_module()
class LoadOccupancyFromFile:

    def __init__(self, occ_root, ignore_class_names=['noise']):
        self.occ_root = occ_root
        self.ignore_class_names = ignore_class_names
        self.pc_range = np.array([-51.2, -51.2, -5.0, 51.2, 51.2, 3])
        self.voxel_size = np.array([0.2, 0.2, 0.2])
        self.occ_class_names = [
            'noise', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
            'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
            'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation'
        ]
    
    def __call__(self, results):
        scene_token, lidar_token = results['scene_token'], results['lidar_token']
        occ_file = osp.join(self.occ_root, f'scene_{scene_token}', 'occupancy', f'{lidar_token}.npy')
        # load lidar and camera visible label
        occ_labels = np.load(occ_file)
        coors, labels = occ_labels[:, :3], occ_labels[:, 3]

        curr_class_names = [n for n in self.occ_class_names if n not in self.ignore_class_names]
        empty_labels = len(curr_class_names)
        label_mapper = [curr_class_names.index(n) if n in curr_class_names else empty_labels
                        for n in self.occ_class_names]
        label_mapper = np.array(label_mapper)
        labels = label_mapper[labels]

        scene_size = self.pc_range[3:] - self.pc_range[:3]
        voxel_num = (scene_size / self.voxel_size).astype(np.int64)
        semantics = np.full(voxel_num, empty_labels, dtype=np.uint8)
        semantics[coors[:, 2], coors[:, 1], coors[:, 0]] = labels
        results['voxel_semantics'] = np.ascontiguousarray(semantics)
        return results


@PIPELINES.register_module()
class LoadMultiViewImageFromMultiSweeps:
    def __init__(self,
                 sweeps_num=5,
                 color_type='color',
                 test_mode=False,
                 train_interval=[4, 8],
                 only_keyframe=False,
                 test_interval=6,
                 force_offline=False):
        self.sweeps_num = sweeps_num
        self.color_type = color_type
        self.test_mode = test_mode
        self.force_offline = force_offline

        self.train_interval = train_interval
        self.test_interval = test_interval
        self.only_keyframe = only_keyframe

        try:
            mmcv.use_backend('turbojpeg')
        except ImportError:
            mmcv.use_backend('cv2')

    def load_offline(self, results):
        cam_types = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]

        if len(results['cam_sweeps']['prev']) == 0:
            for _ in range(self.sweeps_num):
                for j in range(len(cam_types)):
                    results['img'].append(results['img'][j])
                    results['img_timestamp'].append(results['img_timestamp'][j])
                    results['filename'].append(results['filename'][j])
                    results['ego2img'].append(np.copy(results['ego2img'][j]))
        else:
            if self.only_keyframe:
                sweeps = results['cam_sweeps']['prev']
                index = [i for i in range(len(sweeps)) if sweeps[i]['CAM_FRONT']['is_key_frame']]
                if len(index) < self.sweeps_num:
                    index = index + [index[-1]] * (self.sweeps_num - len(index))
                choice = index[:self.sweeps_num]
            elif self.test_mode:
                interval = self.test_interval
                choices = [(k + 1) * interval - 1 for k in range(self.sweeps_num)]
            elif len(results['cam_sweeps']['prev']) <= self.sweeps_num:
                pad_len = self.sweeps_num - len(results['cam_sweeps']['prev'])
                choices = list(range(len(results['cam_sweeps']['prev']))) + \
                    [len(results['cam_sweeps']['prev']) - 1] * pad_len
            else:
                max_interval = len(results['cam_sweeps']['prev']) // self.sweeps_num
                max_interval = min(max_interval, self.train_interval[1])
                min_interval = min(max_interval, self.train_interval[0])
                interval = np.random.randint(min_interval, max_interval + 1)
                choices = [(k + 1) * interval - 1 for k in range(self.sweeps_num)]

            for idx in sorted(list(choices)):
                sweep_idx = min(idx, len(results['cam_sweeps']['prev']) - 1)
                sweep = results['cam_sweeps']['prev'][sweep_idx]

                if len(sweep.keys()) < len(cam_types):
                    sweep = results['cam_sweeps']['prev'][sweep_idx - 1]

                for sensor in cam_types:
                    results['img'].append(mmcv.imread(sweep[sensor]['data_path'], self.color_type))
                    results['img_timestamp'].append(sweep[sensor]['timestamp'] / 1e6)
                    results['filename'].append(os.path.relpath(sweep[sensor]['data_path']))
                    results['ego2img'].append(compose_ego2img(
                        results['ego2global_translation'],
                        results['ego2global_rotation'],
                        sweep[sensor]['sensor2global_translation'],
                        sweep[sensor]['sensor2global_rotation'],
                        sweep[sensor]['cam_intrinsic'],
                    ))

        return results

    def load_online(self, results):
        # only used when measuring FPS
        assert self.test_mode
        assert self.test_interval % 6 == 0

        cam_types = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]

        if len(results['cam_sweeps']['prev']) == 0:
            for _ in range(self.sweeps_num):
                for j in range(len(cam_types)):
                    results['img_timestamp'].append(results['img_timestamp'][j])
                    results['filename'].append(results['filename'][j])
                    results['ego2img'].append(np.copy(results['ego2img'][j]))
        else:
            if self.only_keyframe:
                sweeps = results['cam_sweeps']['prev']
                index = [i for i in range(len(sweeps)) if sweeps[i]['CAM_FRONT']['is_key_frame']]
                if len(index) < self.sweeps_num:
                    index = index + [index[-1]] * (self.sweeps_num - len(index))
                choice = index[:self.sweeps_num]
            else:
                interval = self.test_interval
                choices = [(k + 1) * interval - 1 for k in range(self.sweeps_num)]

            for idx in sorted(list(choices)):
                sweep_idx = min(idx, len(results['cam_sweeps']['prev']) - 1)
                sweep = results['cam_sweeps']['prev'][sweep_idx]

                if len(sweep.keys()) < len(cam_types):
                    sweep = results['cam_sweeps']['prev'][sweep_idx - 1]

                for sensor in cam_types:
                    # skip loading history frames
                    results['img_timestamp'].append(sweep[sensor]['timestamp'] / 1e6)
                    results['filename'].append(os.path.relpath(sweep[sensor]['data_path']))
                    results['ego2img'].append(compose_ego2img(
                        results['ego2global_translation'],
                        results['ego2global_rotation'],
                        sweep[sensor]['sensor2global_translation'],
                        sweep[sensor]['sensor2global_rotation'],
                        sweep[sensor]['cam_intrinsic'],
                    ))

        return results

    def __call__(self, results):
        if self.sweeps_num == 0:
            return results

        world_size = get_dist_info()[1]
        if world_size == 1 and self.test_mode and (not self.force_offline):
            return self.load_online(results)
        else:
            return self.load_offline(results)


@PIPELINES.register_module()
class LoadMultiViewImageFromMultiSweepsFuture:
    def __init__(self,
                 prev_sweeps_num=5,
                 next_sweeps_num=5,
                 color_type='color',
                 test_mode=False):
        self.prev_sweeps_num = prev_sweeps_num
        self.next_sweeps_num = next_sweeps_num
        self.color_type = color_type
        self.test_mode = test_mode

        assert prev_sweeps_num == next_sweeps_num

        self.train_interval = [4, 8]
        self.test_interval = 6

        try:
            mmcv.use_backend('turbojpeg')
        except ImportError:
            mmcv.use_backend('cv2')

    def __call__(self, results):
        if self.prev_sweeps_num == 0 and self.next_sweeps_num == 0:
            return results

        cam_types = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]

        if self.test_mode:
            interval = self.test_interval
        else:
            interval = np.random.randint(self.train_interval[0], self.train_interval[1] + 1)

        # previous sweeps
        if len(results['cam_sweeps']['prev']) == 0:
            for _ in range(self.prev_sweeps_num):
                for j in range(len(cam_types)):
                    results['img'].append(results['img'][j])
                    results['img_timestamp'].append(results['img_timestamp'][j])
                    results['filename'].append(results['filename'][j])
                    results['ego2img'].append(np.copy(results['ego2img'][j]))
        else:
            choices = [(k + 1) * interval - 1 for k in range(self.prev_sweeps_num)]

            for idx in sorted(list(choices)):
                sweep_idx = min(idx, len(results['cam_sweeps']['prev']) - 1)
                sweep = results['cam_sweeps']['prev'][sweep_idx]

                if len(sweep.keys()) < len(cam_types):
                    sweep = results['cam_sweeps']['prev'][sweep_idx - 1]

                for sensor in cam_types:
                    results['img'].append(mmcv.imread(sweep[sensor]['data_path'], self.color_type))
                    results['img_timestamp'].append(sweep[sensor]['timestamp'] / 1e6)
                    results['filename'].append(sweep[sensor]['data_path'])
                    results['ego2img'].append(compose_ego2img(
                        results['ego2global_translation'],
                        results['ego2global_rotation'],
                        sweep[sensor]['sensor2global_translation'],
                        sweep[sensor]['sensor2global_rotation'],
                        sweep[sensor]['cam_intrinsic'],
                    ))

        # future sweeps
        if len(results['cam_sweeps']['next']) == 0:
            for _ in range(self.next_sweeps_num):
                for j in range(len(cam_types)):
                    results['img'].append(results['img'][j])
                    results['img_timestamp'].append(results['img_timestamp'][j])
                    results['filename'].append(results['filename'][j])
                    results['ego2img'].append(np.copy(results['ego2img'][j]))
        else:
            choices = [(k + 1) * interval - 1 for k in range(self.next_sweeps_num)]

            for idx in sorted(list(choices)):
                sweep_idx = min(idx, len(results['cam_sweeps']['next']) - 1)
                sweep = results['cam_sweeps']['next'][sweep_idx]

                if len(sweep.keys()) < len(cam_types):
                    sweep = results['cam_sweeps']['next'][sweep_idx - 1]

                for sensor in cam_types:
                    results['img'].append(mmcv.imread(sweep[sensor]['data_path'], self.color_type))
                    results['img_timestamp'].append(sweep[sensor]['timestamp'] / 1e6)
                    results['filename'].append(sweep[sensor]['data_path'])
                    results['ego2img'].append(compose_ego2img(
                        results['ego2global_translation'],
                        results['ego2global_rotation'],
                        sweep[sensor]['sensor2global_translation'],
                        sweep[sensor]['sensor2global_rotation'],
                        sweep[sensor]['cam_intrinsic'],
                    ))

        return results


'''
This func loads previous and future frames in interleaved order, 
e.g. curr, prev1, next1, prev2, next2, prev3, next3...
'''
@PIPELINES.register_module()
class LoadMultiViewImageFromMultiSweepsFutureInterleave:
    def __init__(self,
                 prev_sweeps_num=5,
                 next_sweeps_num=5,
                 color_type='color',
                 test_mode=False):
        self.prev_sweeps_num = prev_sweeps_num
        self.next_sweeps_num = next_sweeps_num
        self.color_type = color_type
        self.test_mode = test_mode

        assert prev_sweeps_num == next_sweeps_num

        self.train_interval = [4, 8]
        self.test_interval = 6

        try:
            mmcv.use_backend('turbojpeg')
        except ImportError:
            mmcv.use_backend('cv2')

    def __call__(self, results):
        if self.prev_sweeps_num == 0 and self.next_sweeps_num == 0:
            return results

        cam_types = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]

        if self.test_mode:
            interval = self.test_interval
        else:
            interval = np.random.randint(self.train_interval[0], self.train_interval[1] + 1)

        results_prev = dict(
            img=[],
            img_timestamp=[],
            filename=[],
            ego2img=[],
        )
        results_next = dict(
            img=[],
            img_timestamp=[],
            filename=[],
            ego2img=[],
        )

        if len(results['sweeps']['prev']) == 0:
            for _ in range(self.prev_sweeps_num):
                for j in range(len(cam_types)):
                    results_prev['img'].append(results['img'][j])
                    results_prev['img_timestamp'].append(results['img_timestamp'][j])
                    results_prev['filename'].append(results['filename'][j])
                    results_prev['ego2img'].append(np.copy(results['ego2img'][j]))
        else:
            choices = [(k + 1) * interval - 1 for k in range(self.prev_sweeps_num)]

            for idx in sorted(list(choices)):
                sweep_idx = min(idx, len(results['sweeps']['prev']) - 1)
                sweep = results['sweeps']['prev'][sweep_idx]

                if len(sweep.keys()) < len(cam_types):
                    sweep = results['sweeps']['prev'][sweep_idx - 1]

                for sensor in cam_types:
                    results_prev['img'].append(mmcv.imread(sweep[sensor]['data_path'], self.color_type))
                    results_prev['img_timestamp'].append(sweep[sensor]['timestamp'] / 1e6)
                    results_prev['filename'].append(os.path.relpath(sweep[sensor]['data_path']))
                    results['ego2img'].append(compose_ego2img(
                        results['ego2global_translation'],
                        results['ego2global_rotation'],
                        sweep[sensor]['sensor2global_translation'],
                        sweep[sensor]['sensor2global_rotation'],
                        sweep[sensor]['cam_intrinsic'],
                    ))

        if len(results['sweeps']['next']) == 0:
            print(1, len(results_next['img']) )
            for _ in range(self.next_sweeps_num):
                for j in range(len(cam_types)):
                    results_next['img'].append(results['img'][j])
                    results_next['img_timestamp'].append(results['img_timestamp'][j])
                    results_next['filename'].append(results['filename'][j])
                    results_next['ego2img'].append(np.copy(results['ego2img'][j]))
        else:
            choices = [(k + 1) * interval - 1 for k in range(self.next_sweeps_num)]

            for idx in sorted(list(choices)):
                sweep_idx = min(idx, len(results['sweeps']['next']) - 1)
                sweep = results['sweeps']['next'][sweep_idx]

                if len(sweep.keys()) < len(cam_types):
                    sweep = results['sweeps']['next'][sweep_idx - 1]

                for sensor in cam_types:
                    results_next['img'].append(mmcv.imread(sweep[sensor]['data_path'], self.color_type))
                    results_next['img_timestamp'].append(sweep[sensor]['timestamp'] / 1e6)
                    results_next['filename'].append(os.path.relpath(sweep[sensor]['data_path']))
                    results['ego2img'].append(compose_ego2img(
                        results['ego2global_translation'],
                        results['ego2global_rotation'],
                        sweep[sensor]['sensor2global_translation'],
                        sweep[sensor]['sensor2global_rotation'],
                        sweep[sensor]['cam_intrinsic'],
                    ))

        assert len(results_prev['img']) % 6 == 0
        assert len(results_next['img']) % 6 == 0

        for i in range(len(results_prev['img']) // 6):
            for j in range(6):
                results['img'].append(results_prev['img'][i * 6 + j])
                results['img_timestamp'].append(results_prev['img_timestamp'][i * 6 + j])
                results['filename'].append(results_prev['filename'][i * 6 + j])
                results['ego2img'].append(results_prev['ego2img'][i * 6 + j])

            for j in range(6):
                results['img'].append(results_next['img'][i * 6 + j])
                results['img_timestamp'].append(results_next['img_timestamp'][i * 6 + j])
                results['filename'].append(results_next['filename'][i * 6 + j])
                results['ego2img'].append(results_next['ego2img'][i * 6 + j])

        return results


# Repalce LoadPointsFromMultiSweeps in mmdet3d to adapt sparsebev data
@PIPELINES.register_module(force=True)
class LoadPointsFromMultiSweeps:
    """Load points from multiple sweeps.

    This is usually used for nuScenes dataset to utilize previous sweeps.

    Args:
        sweeps_num (int, optional): Number of sweeps. Defaults to 10.
        load_dim (int, optional): Dimension number of the loaded points.
            Defaults to 5.
        use_dim (list[int], optional): Which dimension to use.
            Defaults to [0, 1, 2, 4].
        time_dim (int, optional): Which dimension to represent the timestamps
            of each points. Defaults to 4.
        file_client_args (dict, optional): Config dict of file clients,
            refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
        pad_empty_sweeps (bool, optional): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool, optional): Whether to remove close points.
            Defaults to False.
        test_mode (bool, optional): If `test_mode=True`, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """

    def __init__(self,
                 sweeps_num=10,
                 load_dim=5,
                 use_dim=[0, 1, 2, 4],
                 time_dim=4,
                 file_client_args=dict(backend='disk'),
                 pad_empty_sweeps=False,
                 remove_close=False,
                 test_mode=False):
        self.load_dim = load_dim
        self.sweeps_num = sweeps_num
        self.use_dim = use_dim
        self.time_dim = time_dim
        assert time_dim < load_dim, \
            f'Expect the timestamp dimension < {load_dim}, got {time_dim}'
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)
        return points

    def _remove_close(self, points, radius=1.0):
        """Removes point too close within a certain radius from origin.

        Args:
            points (np.ndarray | :obj:`BasePoints`): Sweep points.
            radius (float, optional): Radius below which points are removed.
                Defaults to 1.0.

        Returns:
            np.ndarray: Points after removing.
        """
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.

        Args:
            results (dict): Result dict containing multi-sweep point cloud
                filenames.

        Returns:
            dict: The result dict containing the multi-sweep points data.
                Added key and value are described below.

                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point
                    cloud arrays.
        """
        points = results['points']
        points.tensor[:, self.time_dim] = 0
        sweep_points_list = [points]
        ts = results['timestamp']
        sweeps = results['lidar_sweeps']['prev']
        l2g_r = results['lidar2global_rotation']
        l2g_t = results['lidar2global_translation']

        if self.pad_empty_sweeps and len(sweeps) == 0:
            for i in range(self.sweeps_num):
                if self.remove_close:
                    sweep_points_list.append(self._remove_close(points))
                else:
                    sweep_points_list.append(points)
        else:
            if len(sweeps) <= self.sweeps_num:
                choices = np.arange(len(sweeps))
            else:
                choices = np.arange(self.sweeps_num)

            for idx in choices:
                sweep = sweeps[idx]
                points_sweep = self._load_points(sweep['data_path'])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)
                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep)

                s2g_r = sweep['sensor2global_rotation']
                s2g_t = sweep['sensor2global_translation']
                s2l_r = l2g_r.T @ s2g_r
                s2l_t = (s2g_t - l2g_t) @ l2g_r
                points_sweep[:, :3] = points_sweep[:, :3] @ s2l_r.T + s2l_t

                sweep_ts = sweep['timestamp'] / 1e6
                points_sweep[:, self.time_dim] = ts - sweep_ts
                points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)

        points = points.cat(sweep_points_list)
        points = points[:, self.use_dim]
        results['points'] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f'{self.__class__.__name__}(sweeps_num={self.sweeps_num})'


@PIPELINES.register_module()
class LiDARToOccSpace:
    
    def __call__(self, results):
        points = results['points']
        ego2lidar, ego2occ = results['ego2lidar'], results['ego2occ']

        lidar2ego = torch.tensor(inv(ego2lidar)).float()
        lidar2occ = torch.tensor(ego2occ @ lidar2ego.numpy()).float()
        ones = torch.ones_like(points.tensor[..., :1])
        pts = torch.cat([points.tensor[..., :3], ones], dim=1).transpose(0, 1)
        pts = torch.matmul(lidar2occ, pts).transpose(0, 1)[...,:3]

        points.tensor = torch.cat([pts, points.tensor[..., 3:]], dim=1)
        results['points'] = points
        results['ego2lidar'] = ego2occ.copy()
        return results


@PIPELINES.register_module()
class ObjectToOccSpace:
    
    def __call__(self, results):
        ego2occ, ego2obj = results['ego2occ'], results['ego2obj']
        if np.array_equal(ego2occ, ego2obj):
            del results['ego2obj']
            return results
        
        boxes = results['gt_bboxes_3d']
        matrix = torch.from_numpy(ego2occ @ inv(ego2obj)).float()

        ctr, dims, yaw = boxes.center, boxes.dims, boxes.yaw
        velo = boxes.tensor[:, 7:9] if boxes.box_dim > 7 else torch.zeros_like(ctr[:, :2])

        ones = torch.ones_like(ctr[:, :1])
        ctr_ = torch.cat([ctr, ones], dim=-1)
        ctr_ = (matrix @ ctr_.unsqueeze(-1)).squeeze(-1)[:, :3]

        rot_matrix = matrix[:3, :3]
        yaw_ = torch.stack([torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)], dim=-1)
        yaw_ = (rot_matrix @ yaw_.unsqueeze(-1)).squeeze(-1)
        yaw_ = torch.atan2(yaw_[:, [1]], yaw_[:, [0]])

        velo_ = torch.cat([velo, torch.zeros_like(velo[:, :1])], dim=-1)
        velo_ = (rot_matrix @ velo_.unsqueeze(-1)).squeeze(-1)[:, :2]

        box_tensor = torch.cat([ctr_, dims, yaw_, velo_], dim=-1)
        boxes = type(boxes)(box_tensor, box_dim=boxes.box_dim, with_yaw=boxes.with_yaw)
        results['gt_bboxes_3d'] = boxes
        del results['ego2obj'] # objects and occupancy are in the same space now
        return results