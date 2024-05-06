import os
import mmcv
import numpy as np
import os.path as osp
from mmdet.datasets.builder import PIPELINES
from numpy.linalg import inv
from mmcv.runner import get_dist_info


def compose_lidar2img(ego2global_translation_curr,
                      ego2global_rotation_curr,
                      lidar2ego_translation_curr,
                      lidar2ego_rotation_curr,
                      sensor2global_translation_past,
                      sensor2global_rotation_past,
                      cam_intrinsic_past):
    
    R = sensor2global_rotation_past @ (inv(ego2global_rotation_curr).T @ inv(lidar2ego_rotation_curr).T)
    T = sensor2global_translation_past @ (inv(ego2global_rotation_curr).T @ inv(lidar2ego_rotation_curr).T)
    T -= ego2global_translation_curr @ (inv(ego2global_rotation_curr).T @ inv(lidar2ego_rotation_curr).T) + lidar2ego_translation_curr @ inv(lidar2ego_rotation_curr).T

    lidar2cam_r = inv(R.T)
    lidar2cam_t = T @ lidar2cam_r.T

    lidar2cam_rt = np.eye(4)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t

    viewpad = np.eye(4)
    viewpad[:cam_intrinsic_past.shape[0], :cam_intrinsic_past.shape[1]] = cam_intrinsic_past
    lidar2img = (viewpad @ lidar2cam_rt.T).astype(np.float32)

    return lidar2img


@PIPELINES.register_module()
class LoadOccFromFile:

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
        scene_name, sample_idx = results['scene_name'], results['sample_idx']
        occ_file = osp.join(self.occ_root, scene_name, sample_idx, 'labels.npz')
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
class LoadLidartoDepthMap:

    def __init__(self,
                 downsample=2,
                 load_dim=5, 
                 depth_range=[2.0, 42.0, 0.5],
                 dtype='float32',
                 file_client_args=dict(backend='disk')):
        self.downsample = downsample
        self.load_dim = load_dim
        self.depth_range = depth_range
        if dtype=='float32':
            self.dtype = np.float32
        elif dtype== 'float16':
            self.dtype = np.float16
        else:
            raise NotImplementedError
        self.file_client = mmcv.FileClient(**file_client_args)
    
    def points2depthmap(self, points, height, width):
        height, width = height // self.downsample, width // self.downsample
        depth_map = np.zeros((height, width), dtype=np.float32)

        coor = np.round(points[:, :2] / self.downsample)
        depth = points[:, 2]
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
            coor[:, 1] >= 0) & (coor[:, 1] < height) & (
                depth < self.depth_range[1]) & (
                    depth >= self.depth_range[0])
        coor, depth = coor[kept1], depth[kept1]

        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks + depth / 100.).argsort()
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]

        kept2 = np.ones(coor.shape[0], dtype=np.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, depth = coor[kept2], depth[kept2]

        coor = coor.astype(np.int64)
        depth_map[coor[:, 1], coor[:, 0]] = depth
        return depth_map
    
    def load_points(self, results):
        if 'points' in results:
            return results['points']

        pts_filename = results['pts_filename']
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=self.dtype)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=self.dtype)

        return points.reshape(-1, self.load_dim)[:, :3]
    
    def __call__(self, results):
        points = self.load_points(results)
        lidar2img, imgs = results['lidar2img'], results['img']

        depth_maps = []
        for mat, img in zip(lidar2img, imgs):
            ones = np.ones((points.shape[0], 1))
            points_ = np.concatenate([points, ones], axis=1)
            points_ = np.matmul(mat, points_.T).T

            points_ = np.concatenate(
                [points_[:, :2] / points_[:, 2:3], points_[:, 2:3]], 1)
            depth_map = self.points2depthmap(points_, img.shape[0], img.shape[1])  

            # import cv2
            # cv2.imwrite('original_img.png', img)
            # depth_map = ((depth_map / depth_map.max()) * 255).astype(np.uint8)
            # cv2.imwrite('depth_map.png', depth_map)
            # import pdb; pdb.set_trace()

            depth_maps.append(depth_map)
        
        results['depth_map'] = np.stack(depth_maps, axis=0)
        return results


@PIPELINES.register_module()
class LoadOcctoDepthMap:

    def __init__(self, pc_range, downsample=2, empty_label=17, depth_range=[2.0, 42.0]):
        self.pc_range = pc_range
        self.downsample = downsample
        self.empty_label = empty_label
        self.depth_range = depth_range
    
    def points2depthmap(self, points, labels, height, width):
        height, width = height // self.downsample, width // self.downsample
        depth_map = np.zeros((height, width), dtype=np.float32)
        proj_label = np.full((height, width), 17, dtype=np.int64)

        coor = np.round(points[:, :2] / self.downsample)
        depth = points[:, 2]
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
            coor[:, 1] >= 0) & (coor[:, 1] < height) & (
                depth < self.depth_range[1]) & (
                    depth >= self.depth_range[0])
        coor, depth, labels = coor[kept1], depth[kept1], labels[kept1]

        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks + depth / 100.).argsort()
        coor, depth = coor[sort], depth[sort]
        ranks, labels = ranks[sort], labels[sort]

        kept2 = np.ones(coor.shape[0], dtype=np.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, depth, labels = coor[kept2], depth[kept2], labels[kept2]

        coor = coor.astype(np.int64)
        depth_map[coor[:, 1], coor[:, 0]] = depth
        proj_label[coor[:, 1], coor[:, 0]] = labels
        return depth_map, proj_label
    
    def get_occ_center(self, voxel_semantics, mask_camera):
        ratio = 5
        W_, H_, Z_ = voxel_semantics.shape
        W, H, Z = W_ * ratio, H_ * ratio, Z_ * ratio
        voxel_semantics = voxel_semantics[:, None, :, None, :, None]
        voxel_semantics = voxel_semantics.repeat(ratio, axis=1)
        voxel_semantics = voxel_semantics.repeat(ratio, axis=3)
        voxel_semantics = voxel_semantics.repeat(ratio, axis=5)
        voxel_semantics = voxel_semantics.reshape(W, H, Z)

        pc_range = np.array(self.pc_range)
        scene_size = pc_range[3:] - pc_range[:3]

        x = (np.arange(0, W) + 0.5) / W * scene_size[0] + pc_range[0]
        y = (np.arange(0, H) + 0.5) / H * scene_size[1] + pc_range[1]
        z = (np.arange(0, Z) + 0.5) / Z * scene_size[2] + pc_range[2]
        xx = x[:, None, None].repeat(H, axis=1).repeat(Z, axis=2)
        yy = y[None, :, None].repeat(W, axis=0).repeat(Z, axis=2)
        zz = z[None, None, :].repeat(W, axis=0).repeat(H, axis=1)
        coors = np.stack([xx, yy, zz], axis=-1)

        mask = (voxel_semantics != self.empty_label)
        points, labels = coors[mask], voxel_semantics[mask]

        return points, labels
    
    def __call__(self, results):
        voxel_semantics = results['voxel_semantics']
        mask_camera = results['mask_camera']
        points, labels = self.get_occ_center(voxel_semantics, mask_camera)

        lidar2img = results['lidar2img']
        ego2lidar = results['ego2lidar']
        images = results['img']

        depth_maps, proj_labels = [], []
        for l2i, e2l, img in zip(lidar2img[:6], ego2lidar[:6], images[:6]):
            ones = np.ones((points.shape[0], 1))
            mat = np.matmul(l2i, e2l)

            points_ = np.concatenate([points, ones], axis=1)
            points_ = np.matmul(mat, points_.T).T

            points_ = np.concatenate(
                [points_[:, :2] / points_[:, 2:3], points_[:, 2:3]], 1)
            depth_map, proj_label = self.points2depthmap(
                points_, labels, img.shape[0], img.shape[1])  

            depth_maps.append(depth_map)
            proj_labels.append(proj_label)

            # import matplotlib.pyplot as plt
            # color_map = plt.get_cmap('tab20b')
            # import cv2
            # cv2.imwrite('original_img.png', img)
            # depth_map = ((depth_map / depth_map.max()) * 255).astype(np.uint8)
            # cv2.imwrite('depth_map.png', depth_map)
            # proj_label_show = color_map(proj_label)[..., :3]
            # proj_label_show[proj_label==17] = 0
            # proj_label_show = (proj_label_show * 255).astype(np.uint8)
            # cv2.imwrite('proj_label.png', proj_label_show)
            # import pdb; pdb.set_trace()

        results['depth_map'] = np.stack(depth_maps, axis=0)
        results['proj_label'] = np.stack(proj_labels, axis=0)
        return results


@PIPELINES.register_module()
class LoadMultiViewImageFromMultiSweeps(object):
    def __init__(self,
                 sweeps_num=5,
                 color_type='color',
                 test_mode=False,
                 train_interval=[4, 8],
                 test_interval=6):
        self.sweeps_num = sweeps_num
        self.color_type = color_type
        self.test_mode = test_mode

        self.train_interval = train_interval
        self.test_interval = test_interval

        try:
            mmcv.use_backend('turbojpeg')
        except ImportError:
            mmcv.use_backend('cv2')

    def load_offline(self, results):
        cam_types = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]

        if len(results['sweeps']['prev']) == 0:
            for _ in range(self.sweeps_num):
                for j in range(len(cam_types)):
                    results['img'].append(results['img'][j])
                    results['img_timestamp'].append(results['img_timestamp'][j])
                    results['filename'].append(results['filename'][j])
                    results['lidar2img'].append(np.copy(results['lidar2img'][j]))
                    if 'ego2lidar' in results:
                        results['ego2lidar'].append(results['ego2lidar'][0])
        else:
            if self.test_mode:
                interval = self.test_interval
                choices = [(k + 1) * interval - 1 for k in range(self.sweeps_num)]
            elif len(results['sweeps']['prev']) <= self.sweeps_num:
                pad_len = self.sweeps_num - len(results['sweeps']['prev'])
                choices = list(range(len(results['sweeps']['prev']))) + [len(results['sweeps']['prev']) - 1] * pad_len
            else:
                max_interval = len(results['sweeps']['prev']) // self.sweeps_num
                max_interval = min(max_interval, self.train_interval[1])
                min_interval = min(max_interval, self.train_interval[0])
                interval = np.random.randint(min_interval, max_interval + 1)
                choices = [(k + 1) * interval - 1 for k in range(self.sweeps_num)]

            for idx in sorted(list(choices)):
                sweep_idx = min(idx, len(results['sweeps']['prev']) - 1)
                sweep = results['sweeps']['prev'][sweep_idx]

                if len(sweep.keys()) < len(cam_types):
                    sweep = results['sweeps']['prev'][sweep_idx - 1]

                for sensor in cam_types:
                    results['img'].append(mmcv.imread(sweep[sensor]['data_path'], self.color_type))
                    results['img_timestamp'].append(sweep[sensor]['timestamp'] / 1e6)
                    results['filename'].append(os.path.relpath(sweep[sensor]['data_path']))
                    results['lidar2img'].append(compose_lidar2img(
                        results['ego2global_translation'],
                        results['ego2global_rotation'],
                        results['lidar2ego_translation'],
                        results['lidar2ego_rotation'],
                        sweep[sensor]['sensor2global_translation'],
                        sweep[sensor]['sensor2global_rotation'],
                        sweep[sensor]['cam_intrinsic'],
                    ))
                    if 'ego2lidar' in results:
                        results['ego2lidar'].append(results['ego2lidar'][0])

        return results

    def load_online(self, results):
        # only used when measuring FPS
        assert self.test_mode
        assert self.test_interval % 6 == 0

        cam_types = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]

        if len(results['sweeps']['prev']) == 0:
            for _ in range(self.sweeps_num):
                for j in range(len(cam_types)):
                    results['img_timestamp'].append(results['img_timestamp'][j])
                    results['filename'].append(results['filename'][j])
                    results['lidar2img'].append(np.copy(results['lidar2img'][j]))
                    if 'ego2lidar' in results:
                        results['ego2lidar'].append(results['ego2lidar'][0])
        else:
            interval = self.test_interval
            choices = [(k + 1) * interval - 1 for k in range(self.sweeps_num)]

            for idx in sorted(list(choices)):
                sweep_idx = min(idx, len(results['sweeps']['prev']) - 1)
                sweep = results['sweeps']['prev'][sweep_idx]

                if len(sweep.keys()) < len(cam_types):
                    sweep = results['sweeps']['prev'][sweep_idx - 1]

                for sensor in cam_types:
                    # skip loading history frames
                    results['img_timestamp'].append(sweep[sensor]['timestamp'] / 1e6)
                    results['filename'].append(os.path.relpath(sweep[sensor]['data_path']))
                    results['lidar2img'].append(compose_lidar2img(
                        results['ego2global_translation'],
                        results['ego2global_rotation'],
                        results['lidar2ego_translation'],
                        results['lidar2ego_rotation'],
                        sweep[sensor]['sensor2global_translation'],
                        sweep[sensor]['sensor2global_rotation'],
                        sweep[sensor]['cam_intrinsic'],
                    ))
                    if 'ego2lidar' in results:
                        results['ego2lidar'].append(results['ego2lidar'][0])

        return results

    def __call__(self, results):
        if self.sweeps_num == 0:
            return results

        world_size = get_dist_info()[1]
        if world_size == 1 and self.test_mode:
            return self.load_online(results)
        else:
            return self.load_offline(results)


@PIPELINES.register_module()
class LoadMultiViewImageFromMultiSweepsFuture(object):
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
        if len(results['sweeps']['prev']) == 0:
            for _ in range(self.prev_sweeps_num):
                for j in range(len(cam_types)):
                    results['img'].append(results['img'][j])
                    results['img_timestamp'].append(results['img_timestamp'][j])
                    results['filename'].append(results['filename'][j])
                    results['lidar2img'].append(np.copy(results['lidar2img'][j]))
        else:
            choices = [(k + 1) * interval - 1 for k in range(self.prev_sweeps_num)]

            for idx in sorted(list(choices)):
                sweep_idx = min(idx, len(results['sweeps']['prev']) - 1)
                sweep = results['sweeps']['prev'][sweep_idx]

                if len(sweep.keys()) < len(cam_types):
                    sweep = results['sweeps']['prev'][sweep_idx - 1]

                for sensor in cam_types:
                    results['img'].append(mmcv.imread(sweep[sensor]['data_path'], self.color_type))
                    results['img_timestamp'].append(sweep[sensor]['timestamp'] / 1e6)
                    results['filename'].append(sweep[sensor]['data_path'])
                    results['lidar2img'].append(compose_lidar2img(
                        results['ego2global_translation'],
                        results['ego2global_rotation'],
                        results['lidar2ego_translation'],
                        results['lidar2ego_rotation'],
                        sweep[sensor]['sensor2global_translation'],
                        sweep[sensor]['sensor2global_rotation'],
                        sweep[sensor]['cam_intrinsic'],
                    ))

        # future sweeps
        if len(results['sweeps']['next']) == 0:
            for _ in range(self.next_sweeps_num):
                for j in range(len(cam_types)):
                    results['img'].append(results['img'][j])
                    results['img_timestamp'].append(results['img_timestamp'][j])
                    results['filename'].append(results['filename'][j])
                    results['lidar2img'].append(np.copy(results['lidar2img'][j]))
        else:
            choices = [(k + 1) * interval - 1 for k in range(self.next_sweeps_num)]

            for idx in sorted(list(choices)):
                sweep_idx = min(idx, len(results['sweeps']['next']) - 1)
                sweep = results['sweeps']['next'][sweep_idx]

                if len(sweep.keys()) < len(cam_types):
                    sweep = results['sweeps']['next'][sweep_idx - 1]

                for sensor in cam_types:
                    results['img'].append(mmcv.imread(sweep[sensor]['data_path'], self.color_type))
                    results['img_timestamp'].append(sweep[sensor]['timestamp'] / 1e6)
                    results['filename'].append(sweep[sensor]['data_path'])
                    results['lidar2img'].append(compose_lidar2img(
                        results['ego2global_translation'],
                        results['ego2global_rotation'],
                        results['lidar2ego_translation'],
                        results['lidar2ego_rotation'],
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
class LoadMultiViewImageFromMultiSweepsFutureInterleave(object):
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
            lidar2img=[]
        )
        results_next = dict(
            img=[],
            img_timestamp=[],
            filename=[],
            lidar2img=[]
        )

        if len(results['sweeps']['prev']) == 0:
            for _ in range(self.prev_sweeps_num):
                for j in range(len(cam_types)):
                    results_prev['img'].append(results['img'][j])
                    results_prev['img_timestamp'].append(results['img_timestamp'][j])
                    results_prev['filename'].append(results['filename'][j])
                    results_prev['lidar2img'].append(np.copy(results['lidar2img'][j]))
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
                    results_prev['lidar2img'].append(compose_lidar2img(
                        results['ego2global_translation'],
                        results['ego2global_rotation'],
                        results['lidar2ego_translation'],
                        results['lidar2ego_rotation'],
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
                    results_next['lidar2img'].append(np.copy(results['lidar2img'][j]))
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
                    results_next['lidar2img'].append(compose_lidar2img(
                        results['ego2global_translation'],
                        results['ego2global_rotation'],
                        results['lidar2ego_translation'],
                        results['lidar2ego_rotation'],
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
                results['lidar2img'].append(results_prev['lidar2img'][i * 6 + j])

            for j in range(6):
                results['img'].append(results_next['img'][i * 6 + j])
                results['img_timestamp'].append(results_next['img_timestamp'][i * 6 + j])
                results['filename'].append(results_next['filename'][i * 6 + j])
                results['lidar2img'].append(results_next['lidar2img'][i * 6 + j])

        return results
