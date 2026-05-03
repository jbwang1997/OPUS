import os
import mmcv
import numpy as np
import torch
import pickle
import os.path as osp
from tqdm import tqdm
from mmdet.datasets import DATASETS
from nuscenes.eval.common.utils import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix
from torch.utils.data import DataLoader
from models.utils import sparse2dense
from .old_metrics import Metric_mIoU_Occ3D

from mmdet.datasets.pipelines import Compose
from .custom import CustomDataset


@DATASETS.register_module()
class NuScenesOccupancyDataset(CustomDataset):    

    def __init__(self,
                 ann_file,
                 pipeline=None,
                 data_root=None,
                 classes=None,
                 load_interval=1,
                 with_velocity=True,
                 modality=None,
                 filter_empty_gt=False,
                 test_mode=False,
                 use_valid_flag=False):
        super().__init__()
        self.ann_file = ann_file
        self.data_root = data_root
        self.load_interval = load_interval
        self.with_velocity = with_velocity
        self.filter_empty_gt = filter_empty_gt
        self.test_mode = test_mode
        self.use_valid_flag = use_valid_flag

        if modality is None:
            modality = dict(use_camera=False,
                            use_lidar=True,
                            use_radar=False,
                            use_map=False,
                            use_external=False)
        self.modality = modality

        self.CLASSES = self.get_classes(classes)
        self.data_infos = self.load_annotations(self.ann_file)

        # process pipeline
        if pipeline is not None:
            self.pipeline = Compose(pipeline)

        # set group flag for the samplers
        if not self.test_mode:
            self._set_group_flag()

    def load_annotations(self, ann_file):
        data = mmcv.load(ann_file, file_format='pkl')
        data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        data_infos = data_infos[::self.load_interval]
        return data_infos

    def collect_cam_sweeps(self, index, into_past=75, into_future=75):
        scene_name = self.data_infos[index]['scene_name']

        all_sweeps_prev = []
        curr_index = index - 1
        all_sweeps_prev.extend(self.data_infos[index]['cam_sweeps'])
        while len(all_sweeps_prev) < into_past and curr_index >= 0:
            if self.data_infos[curr_index]['scene_name'] != scene_name:
                break
            all_sweeps_prev.append(self.data_infos[curr_index]['cams'])
            all_sweeps_prev.extend(self.data_infos[curr_index]['cam_sweeps'])
            curr_index = curr_index - 1
        
        all_sweeps_next = []
        curr_index = index + 1
        while len(all_sweeps_next) < into_future and curr_index < len(self.data_infos):
            if self.data_infos[curr_index]['scene_name'] != scene_name:
                break
            all_sweeps_next.extend(self.data_infos[curr_index]['cam_sweeps'][::-1])
            all_sweeps_next.append(self.data_infos[curr_index]['cams'])
            curr_index = curr_index + 1

        return all_sweeps_prev, all_sweeps_next

    def collect_lidar_sweeps(self, index, into_past=20, into_future=20):
        scene_name = self.data_infos[index]['scene_name']

        all_sweeps_prev = []
        curr_index = index - 1
        all_sweeps_prev.extend(self.data_infos[index]['lidar_sweeps'])
        while len(all_sweeps_prev) < into_past and curr_index >= 0:
            if self.data_infos[curr_index]['scene_name'] != scene_name:
                break
            all_sweeps_prev.append(self.data_infos[curr_index]['lidar'])
            all_sweeps_prev.extend(self.data_infos[curr_index]['lidar_sweeps'])
            curr_index = curr_index - 1
        
        all_sweeps_next = []
        curr_index = index + 1
        while len(all_sweeps_next) < into_future and curr_index < len(self.data_infos):
            if self.data_infos[curr_index]['scene_name'] != scene_name:
                break
            all_sweeps_next.extend(self.data_infos[curr_index]['lidar_sweeps'][::-1])
            all_sweeps_next.append(self.data_infos[curr_index]['lidar'])
            curr_index = curr_index + 1

        return all_sweeps_prev, all_sweeps_next

    def get_ann_info(self, index):
        info = self.data_infos[index]
        # filter out bbox containing no points
        if self.use_valid_flag:
            mask = info['valid_flag']
        else:
            mask = info['num_lidar_pts'] > 0
        gt_bboxes_3d = info['gt_boxes'][mask]
        gt_names_3d = info['gt_names'][mask]

        if self.with_velocity:
            gt_velocity = info['gt_velocity'][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        return dict(gt_bboxes_3d=gt_bboxes_3d, gt_names=gt_names_3d)

    def get_data_info(self, index):
        info = self.data_infos[index]

        ego2global_translation = info['ego2global_translation']
        ego2global_rotation = info['ego2global_rotation']
        lidar2ego_translation = info['lidar2ego_translation']
        lidar2ego_rotation = info['lidar2ego_rotation']
        ego2global_rotation_mat = Quaternion(ego2global_rotation).rotation_matrix
        lidar2ego_rotation_mat = Quaternion(lidar2ego_rotation).rotation_matrix
        ego2lidar = transform_matrix(
            lidar2ego_translation, Quaternion(lidar2ego_rotation), inverse=True)

        input_dict = dict(
            sample_token=info['token'],
            scene_name=info['scene_name'],
            scene_token=info['scene_token'],
            lidar_token=info['lidar_token'],
            timestamp=info['timestamp'] / 1e6,
            ego2lidar=ego2lidar,
            ego2obj=ego2lidar,
            ego2occ=ego2lidar,
            ego2global_translation=np.array(ego2global_translation),
            ego2global_rotation=ego2global_rotation_mat,
            lidar2ego_translation=np.array(lidar2ego_translation),
            lidar2ego_rotation=lidar2ego_rotation_mat,
        )

        if self.modality['use_lidar']:
            lidar_sweeps_prev, lidar_sweeps_next = self.collect_lidar_sweeps(index)
            input_dict.update(dict(
                lidar_info=info['lidar'],
                lidar_sweeps={'prev': lidar_sweeps_prev, 'next': lidar_sweeps_next},
            ))

        if self.modality['use_camera']:
            cam_sweeps_prev, cam_sweeps_next = self.collect_cam_sweeps(index)
            input_dict.update(dict(
                cam_info=info['cams'],
                cam_sweeps={'prev': cam_sweeps_prev, 'next': cam_sweeps_next},
            ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict
    
    def evaluate(self, occ_results, runner=None, show_dir=None, **eval_kwargs):
        occ_gts = []
        occ_preds = []
        lidar_origins = []

        print('\nStarting Evaluation...')
        metric = Metric_mIoU_Occupancy()

        occ_class_names = [
            'noise', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
            'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
            'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation'
        ]
        ignore_class_names=['noise']
        pc_range = np.array([-51.2, -51.2, -5.0, 51.2, 51.2, 3])
        voxel_size = np.array([0.2, 0.2, 0.2])
        voxel_num = ((pc_range[3:] - pc_range[:3]) / voxel_size).astype(np.int64)

        from tqdm import tqdm
        for i in tqdm(range(len(occ_results))):
            result_dict = occ_results[i]
            info = self.get_data_info(i)

            scene_token, lidar_token = info['scene_token'], info['lidar_token']
            occ_root = 'data/nuscenes/openoccupancy/'
            occ_file = osp.join(occ_root, f'scene_{scene_token}', 'occupancy', f'{lidar_token}.npy')
            # load lidar and camera visible label
            occ_labels = np.load(occ_file)
            coors, labels = occ_labels[:, :3], occ_labels[:, 3]
            occ_labels, _ = sparse2dense(coors[:, ::-1], labels, voxel_num, empty_value=len(occ_class_names))
            mask = occ_labels != 0 # ignore noise

            curr_class_names = [n for n in occ_class_names if n not in ignore_class_names]
            curr_bg_class_idx = len(curr_class_names) # 16
            label_mapper = [curr_class_names.index(n) if n in curr_class_names else 16
                            for n in occ_class_names] + [curr_bg_class_idx]
            label_mapper = np.array(label_mapper)
            occ_labels = label_mapper[occ_labels]

            occ_pred, _ = sparse2dense(result_dict['occ_loc'], result_dict['sem_pred'], voxel_num, 16)
            metric.add_batch(occ_pred, occ_labels, mask)

        mIoU, IoU = metric.count_miou()
        return {'mIoU': mIoU, 'IoU': IoU}

    def format_results(self, occ_results, submission_prefix, **kwargs):
        if submission_prefix is not None:
            mmcv.mkdir_or_exist(submission_prefix)

        for index, occ_pred in enumerate(tqdm(occ_results)):
            info = self.data_infos[index]
            sample_token = info['token']
            save_path=os.path.join(submission_prefix, '{}.npz'.format(sample_token))
            np.savez_compressed(save_path,occ_pred.astype(np.uint8))
        print('\nFinished.')