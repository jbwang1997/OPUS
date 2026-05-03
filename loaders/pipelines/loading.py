import os
import mmcv
import torch
import numpy as np
import os.path as osp
from mmdet.datasets.builder import PIPELINES
from numpy.linalg import inv
from mmcv.runner import get_dist_info


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
class LoadMVImageWithSweeps:
    def __init__(self,
                 prev_sweeps_num=0,
                 next_sweeps_num=0,
                 color_type='color',
                 interval=[4, 8],
                 only_keyframe=False,
                 order='temporal',
                 test_mode=False,
                 force_offline=False):
        self.prev_sweeps_num = prev_sweeps_num
        self.next_sweeps_num = next_sweeps_num
        self.color_type = color_type
        self.interval = interval if isinstance(interval, list) else [interval, interval]
        self.only_keyframe = only_keyframe
        self.force_offline = force_offline
        self.test_mode = test_mode

        self.order = order
        assert self.order in ['temporal', 'interleave']
        if self.order == 'interleave':
            assert self.prev_sweeps_num == self.next_sweeps_num

        self.cam_types = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]

        try:
            mmcv.use_backend('turbojpeg')
        except ImportError:
            mmcv.use_backend('cv2')
    
    def compose_ego2img(self, results, info):
        ego2global_t = results['ego2global_translation']
        ego2global_r = results['ego2global_rotation']
        sensor2global_t = info['sensor2global_translation']
        sensor2global_r = info['sensor2global_rotation']
        intrinsic = info['cam_intrinsic']

        R = np.linalg.inv(sensor2global_r) @ ego2global_r
        T = (ego2global_t - sensor2global_t) @ sensor2global_r

        ego2cam_rt = np.eye(4)
        ego2cam_rt[:3, :3] = R
        ego2cam_rt[:3, 3] = T.T

        viewpad = np.eye(4)
        viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
        ego2img = (viewpad @ ego2cam_rt).astype(np.float32)

        return ego2img
    
    def extract_info(self, results, data):
        filename, img_timestamp, ego2img = [], [], []
        for cam_type in self.cam_types:
            filename.append(data[cam_type]['data_path'])
            img_timestamp.append(data[cam_type]['timestamp'] / 1e6)
            ego2img.append(self.compose_ego2img(results, data[cam_type]))
        return dict(filename=filename, img_timestamp=img_timestamp, ego2img=ego2img)
    
    def padding(self, index, tgt_len):
        if len(index) == 0:
            return [-1 for _ in range(tgt_len)]
        else:
            return index + [index[-1]] * (tgt_len - len(index))
    
    def collect_infos(self, curr, prevs, nexts):
        if self.order == 'temporal':
            infos = nexts[::-1] + [curr] + prevs
            latest = 0
        else:
            infos = [curr]
            for prev, next in zip(prevs, nexts):
                infos.extend([prev, next])
            latest = len(infos) - 1

        data = dict(img=[], filename=[], img_timestamp=[], ego2img=[])
        world_size = get_dist_info()[1]
        online = self.test_mode and not self.force_offline and world_size == 1
        for i, info in enumerate(infos):
            data['filename'].extend(info['filename'])
            data['img_timestamp'].extend(info['img_timestamp'])
            data['ego2img'].extend(info['ego2img'])
            if online and i != latest:
                continue
            else:
                for filename in info['filename']:
                    data['img'].append(mmcv.imread(filename, self.color_type))
        
        data['img_shape'] = data['img'][0].shape
        data['ori_shape'] = data['img'][0].shape
        data['pad_shape'] = data['img'][0].shape
        num_channels = 1 if len(data['img'][0].shape) < 3 else data['img'][0].shape[2]
        data['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return data
    
    def __call__(self, results):
        prev_sweeps = results['cam_sweeps']['prev']
        next_sweeps = results['cam_sweeps']['next']
        if self.test_mode or self.only_keyframe:
            prev_index = [i for i in range(len(prev_sweeps)) 
                          if prev_sweeps[i]['CAM_FRONT']['is_key_frame']]
            prev_index = prev_index[:self.prev_sweeps_num]
            next_index = [i for i in range(len(next_sweeps))
                          if next_sweeps[i]['CAM_FRONT']['is_key_frame']]
            next_index = next_index[:self.next_sweeps_num]
        else:
            interval = np.random.randint(self.interval[0], self.interval[1] + 1)
            prev_index = [(k + 1) * interval - 1 for k in range(self.prev_sweeps_num)
                          if (k + 1) * interval - 1 < len(prev_sweeps)]
            next_index = [(k + 1) * interval - 1 for k in range(self.next_sweeps_num)
                          if (k + 1) * interval - 1 < len(next_sweeps)]
        
        prev_index = self.padding(prev_index, self.prev_sweeps_num)
        next_index = self.padding(next_index, self.next_sweeps_num)
        
        curr_info = self.extract_info(results, results['cam_info'])
        prev_infos = [self.extract_info(results, prev_sweeps[i]) if i != -1 \
                      else curr_info for i in prev_index]
        next_infos = [self.extract_info(results, next_sweeps[i]) if i != -1 \
                      else curr_info for i in next_index]
        data = self.collect_infos(curr_info, prev_infos, next_infos)
        results.update(data)
        return results


@PIPELINES.register_module()
class LoadPointsWithSweeps:

    def __init__(self,
                 prev_sweeps_num=0,
                 next_sweeps_num=0,
                 load_dim=5,
                 use_dim=[0, 1, 2, 4],
                 time_dim=4,
                 tgt_coord_system='occ',
                 file_client_args=dict(backend='disk'),
                 pad_empty_sweeps=False,
                 remove_close=False,
                 test_mode=False):
        self.prev_sweeps_num = prev_sweeps_num
        self.next_sweeps_num = next_sweeps_num

        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert time_dim < load_dim, \
            f'Expect the timestamp dimension < {load_dim}, got {time_dim}'
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.time_dim = time_dim

        assert tgt_coord_system in ['ego', 'lidar', 'occ', 'obj']
        self.tgt_coord_system = tgt_coord_system

        self.file_client = None
        self.file_client_args = file_client_args.copy()
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close

    def load_points(self, pts_filename):
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
        return points.copy().reshape(-1, self.load_dim)

    def _remove_close(self, points, radius=1.0):
        x_filt = np.abs(points[:, 0]) < radius
        y_filt = np.abs(points[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]
    
    def padding(self, index, tgt_len):
        if len(index) == 0:
            return [-1 for _ in range(tgt_len)]
        else:
            return index + [index[-1]] * (tgt_len - len(index))
    
    def load_sweep(self, sweep, l2g_r, l2g_t, ts):
        points_sweep = self.load_points(sweep['data_path'])
        if self.remove_close:
            points_sweep = self._remove_close(points_sweep)

        s2g_r = sweep['sensor2global_rotation']
        s2g_t = sweep['sensor2global_translation']
        s2l_r = l2g_r.T @ s2g_r
        s2l_t = (s2g_t - l2g_t) @ l2g_r
        points_sweep[:, :3] = points_sweep[:, :3] @ s2l_r.T + s2l_t

        sweep_ts = sweep['timestamp'] / 1e6
        points_sweep[:, self.time_dim] = ts - sweep_ts
        return points_sweep

    def __call__(self, results):
        prev_sweeps = results['lidar_sweeps']['prev']
        next_sweeps = results['lidar_sweeps']['next']
        l2g_r = results['lidar_info']['sensor2global_rotation']
        l2g_t = results['lidar_info']['sensor2global_translation']

        curr_points = self.load_points(results['lidar_info']['data_path'])
        curr_points[:, self.time_dim] = 0
        ts = results['timestamp']

        prev_index = list(range(self.prev_sweeps_num))[:len(prev_sweeps)]
        if self.pad_empty_sweeps:
            prev_index = self.padding(prev_index, self.prev_sweeps_num)
        next_index = list(range(self.next_sweeps_num))[:len(next_sweeps)]
        if self.pad_empty_sweeps:
            next_index = self.padding(next_index, self.next_sweeps_num)
        
        prev_points = []
        for i in prev_index:
            if i == -1:
                points = curr_points if not self.remove_close else \
                    self._remove_close(curr_points)
            else:
                points = self.load_sweep(prev_sweeps[i], l2g_r, l2g_t, ts)
            prev_points.append(points)

        next_points = []
        for i in next_index:
            if i == -1:
                points = curr_points if not self.remove_close else \
                    self._remove_close(curr_points)
            else:
                points = self.load_sweep(next_sweeps[i], l2g_r, l2g_t, ts)
            next_points.append(points)
        
        points = [curr_points] + prev_points + next_points
        points = np.concatenate(points, axis=0)
        points = points[:, self.use_dim]

        lidar2ego = np.linalg.inv(results['ego2lidar'])
        if self.tgt_coord_system == 'ego':
            ego2tgt = np.eye(4)
        elif self.tgt_coord_system == 'lidar':
            ego2tgt = results['ego2lidar']
        elif self.tgt_coord_system == 'occ':
            ego2tgt = results['ego2occ']
        elif self.tgt_coord_system == 'obj':
            ego2tgt = results['ego2obj']
        mat = ego2tgt @ lidar2ego
        coors = np.concatenate(
            [points[:, :3], np.ones((points.shape[0], 1))], axis=1)
        coors = (mat @ coors.T).T
        points[:, :3] = coors[:, :3]

        results['points'] = points
        results['ego2lidar'] = ego2tgt.copy()
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