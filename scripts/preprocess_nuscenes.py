import os.path as osp
import sys
path = osp.abspath(osp.join(osp.dirname(__file__), '..'))
sys.path.insert(0, path)

import pickle
import argparse
import numpy as np
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion

name_mapper = {
    'movable_object.barrier': 'barrier',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.car': 'car',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.motorcycle': 'motorcycle',
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'human.pedestrian.police_officer': 'pedestrian',
    'movable_object.trafficcone': 'traffic_cone',
    'vehicle.trailer': 'trailer',
    'vehicle.truck': 'truck'
}


def create_nuscenes_infos(root_path):
    nusc = NuScenes(version='v1.0-trainval', dataroot=root_path, verbose=True)
    scene_token_mapper = {s['name']: s['token'] for s in nusc.scene}
    train_scenes = [scene_token_mapper[n] for n in splits.train]
    val_scenes = [scene_token_mapper[n] for n in splits.val]
    print('train scene: {}, val scene: {}'.format(len(train_scenes), len(val_scenes)))

    # start preprocess sample
    train_infos, val_infos = [], []
    for sample in tqdm(nusc.sample):
        scene = nusc.get('scene', sample['scene_token'])
        info = dict(token=sample['token'],
                    scene_name=scene['name'],
                    scene_token=scene['token'])

        fill_lidar_info(info, nusc, sample)
        fill_cam_info(info, nusc, sample)
        fill_ann_info(info, nusc, sample)

        if sample['scene_token'] in train_scenes:
            train_infos.append(info)
        else:
            val_infos.append(info)

    print('train sample: {}, val sample: {}'.format(len(train_infos), len(val_infos)))
    metadata = dict(version='v1.0_trainval')

    train_data = dict(infos=train_infos, metadata=metadata)
    info_path = osp.join(root_path, 'nuscenes_infos_train_sweep.pkl')
    with open(info_path, 'wb') as f:
        pickle.dump(train_data, f)

    val_data = dict(infos=val_infos, metadata=metadata)
    info_path = osp.join(root_path, 'nuscenes_infos_val_sweep.pkl')
    with open(info_path, 'wb') as f:
        pickle.dump(val_data, f)


def fill_lidar_info(info, nusc, sample):
    lidar_token = sample['data']['LIDAR_TOP']
    sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    lidar_path = str(nusc.get_sample_data_path(sd_rec['token']))

    info['lidar2ego_rotation'] = cs_record['rotation']
    info['lidar2ego_translation'] = cs_record['translation']
    info['ego2global_rotation'] = pose_record['rotation']
    info['ego2global_translation'] = pose_record['translation']
    info['timestamp'] = sample['timestamp']

    lidar_sweep = get_lidar_sweep(nusc, sd_rec)
    lidar_sweep['is_key_frame'] = True
    info['lidar'] = lidar_sweep

    sweeps = []
    while sd_rec['prev'] != '':
        sd_rec = nusc.get('sample_data', sd_rec['prev'])
        if sd_rec['is_key_frame']:
            break
        lidar_sweep = get_lidar_sweep(nusc, sd_rec)
        lidar_sweep['is_key_frame'] = False
        sweeps.append(lidar_sweep)
    info['lidar_sweeps'] = sweeps


def fill_cam_info(info, nusc, sample):
    cam_types = [
        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
    ]

    info['cams'] = dict()
    for cam in cam_types:
        sample_data = nusc.get('sample_data', sample['data'][cam])
        sweep_cam = get_cam_sweep(nusc, sample_data, cam)
        sweep_cam['is_key_frame'] = True
        info['cams'][cam] = sweep_cam

    curr_cams = dict()
    for cam in cam_types:
        curr_cams[cam] = nusc.get('sample_data', sample['data'][cam])
    
    sweep_infos = []
    if sample['prev'] != '':  # add sweep frame between two key frame
        for _ in range(5):
            sweep_info = dict()
            for cam in cam_types: 
                if curr_cams[cam]['prev'] == '':    
                    sweep_info = sweep_infos[-1] 
                    break
                sample_data = nusc.get('sample_data', curr_cams[cam]['prev'])
                sweep_cam = get_cam_sweep(nusc, sample_data, cam)
                sweep_cam['is_key_frame'] = False
                curr_cams[cam] = sample_data
                sweep_info[cam] = sweep_cam
            sweep_infos.append(sweep_info)
    info['cam_sweeps'] = sweep_infos


def fill_ann_info(info, nusc, sample):
    lidar_token = sample['data']['LIDAR_TOP']
    _, boxes, _ = nusc.get_sample_data(sample['data']['LIDAR_TOP'])
    l2e_r_mat = Quaternion(info['lidar2ego_rotation']).rotation_matrix
    e2g_r_mat = Quaternion(info['ego2global_rotation']).rotation_matrix

    # get boxes
    locs = np.array([b.center for b in boxes]).reshape(-1, 3)
    dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
    rots = np.array([b.orientation.yaw_pitch_roll[0]
                        for b in boxes]).reshape(-1, 1)
    # Adhere to mmdet3dv1.0.0rc6's definition of the 3D box
    gt_boxes = np.concatenate([locs, dims[:, [1, 0, 2]], rots], axis=1)

    # get velocity
    velocity = np.array([nusc.box_velocity(token)[:2] for token in sample['anns']])
    # convert velo from global to lidar
    for i in range(len(boxes)):
        velo = np.array([*velocity[i], 0.0])
        velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
        velocity[i] = velo[:2]
    
    # get names
    names = [b.name for b in boxes]
    for i in range(len(names)):
        if names[i] in name_mapper:
            names[i] = name_mapper[names[i]]
    names = np.array(names, dtype=np.dtype('<U20'))

    # get other annotation information
    annotations = [nusc.get('sample_annotation', token) for token in sample['anns']]
    valid_flag = np.array([(anno['num_lidar_pts'] + anno['num_radar_pts']) > 0
                           for anno in annotations], dtype=bool).reshape(-1)
    identity = np.array([anno['instance_token'] for anno in annotations],
                        dtype=np.dtype('<U32'))

    assert len(gt_boxes) == len(annotations), f'{len(gt_boxes)}, {len(annotations)}'
    info['gt_boxes'] = gt_boxes
    info['gt_names'] = names
    info['gt_velocity'] = velocity.reshape(-1, 2)
    info['num_lidar_pts'] = np.array([a['num_lidar_pts'] for a in annotations])
    info['num_radar_pts'] = np.array([a['num_radar_pts'] for a in annotations])
    info['valid_flag'] = valid_flag
    info['identity'] = identity


def get_lidar_sweep(nusc, sd_rec):
    cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    data_path = str(nusc.get_sample_data_path(sd_rec['token']))

    sensor2ego_translation = cs_record['translation']
    ego2global_translation = pose_record['translation']
    sensor2ego_rotation = Quaternion(cs_record['rotation']).rotation_matrix
    ego2global_rotation = Quaternion(pose_record['rotation']).rotation_matrix

    sensor2global_rotation = ego2global_rotation @ sensor2ego_rotation
    sensor2global_translation = sensor2ego_translation @ ego2global_rotation.T + ego2global_translation

    return {
        'type': 'LIDAR_TOP',
        'data_path': data_path,
        'sample_data_token': sd_rec['token'],
        'sensor2global_rotation': sensor2global_rotation,
        'sensor2global_translation': sensor2global_translation,
        'timestamp': sd_rec['timestamp']
    }


def get_cam_sweep(nusc, sample_data, cam_type):
    pose_record = nusc.get('ego_pose', sample_data['ego_pose_token'])
    cs_record = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
    
    sensor2ego_translation = cs_record['translation']
    ego2global_translation = pose_record['translation']
    sensor2ego_rotation = Quaternion(cs_record['rotation']).rotation_matrix
    ego2global_rotation = Quaternion(pose_record['rotation']).rotation_matrix
    cam_intrinsic = np.array(cs_record['camera_intrinsic'])

    sensor2global_rotation = ego2global_rotation @ sensor2ego_rotation
    sensor2global_translation = sensor2ego_translation @ ego2global_rotation.T + ego2global_translation

    return {
        'type': cam_type,
        'data_path': nusc.get_sample_data_path(sample_data['token']),
        'sensor2global_rotation': sensor2global_rotation,
        'sensor2global_translation': sensor2global_translation,
        'cam_intrinsic': cam_intrinsic,
        'timestamp': sample_data['timestamp'],
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='data/nuscenes')
    args = parser.parse_args()
    create_nuscenes_infos(args.data_root)
