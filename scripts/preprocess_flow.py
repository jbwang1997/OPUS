import sys
import os
import os.path as osp
path = osp.abspath(osp.join(osp.dirname(__file__), '..'))
sys.path.insert(0, path)

import pickle
import argparse
import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm
from nuscenes.eval.common.utils import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix
from mmcv.ops import points_in_boxes_all

DEVICE = 'cuda:0'
DYNAMIC_CLASSES = np.array(['bicycle', 'bus', 'car', 'construction_vehicle',
                            'motorcycle', 'pedestrian', 'trailer', 'truck'],
                            dtype=np.dtype('<U20'))


def occ3d_loader(args, data):
    scene_name, sample_token = data['scene_name'], data['token']
    occ_file = osp.join(args.occ_root, scene_name, sample_token, 'labels.npz')
    occ = np.load(occ_file)['semantics']
    name_mapper = np.array([
        'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
        'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
        'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
        'vegetation'], dtype=np.dtype('<U20'))

    occ = torch.tensor(occ, device=DEVICE, dtype=torch.int64)
    pc_range = torch.tensor([-40.0, -40.0, -1.0, 40.0, 40.0, 5.4],
                            device=DEVICE, dtype=torch.float32)
    voxel_size = torch.tensor([0.4, 0.4, 0.4], device=DEVICE, dtype=torch.float32)
    scene_size = pc_range[3:] - pc_range[:3]

    device = occ.device
    W, H, Z = occ.shape
    x = torch.arange(0, W, dtype=torch.int64, device=DEVICE)
    cx = (x.float() + 0.5) / W * scene_size[0] + pc_range[0]
    y = torch.arange(0, H, dtype=torch.int64, device=DEVICE)
    cy = (y.float() + 0.5) / H * scene_size[1] + pc_range[1]
    z = torch.arange(0, Z, dtype=torch.int64, device=DEVICE)
    cz = (z.float() + 0.5) / Z * scene_size[2] + pc_range[2]

    xx = x[:, None, None].expand(W, H, Z)
    yy = y[None, :, None].expand(W, H, Z)
    zz = z[None, None, :].expand(W, W, Z)
    index = torch.stack([xx, yy, zz], dim=-1)

    cxx = cx[:, None, None].expand(W, H, Z)
    cyy = cy[None, :, None].expand(W, H, Z)
    czz = cz[None, None, :].expand(W, W, Z)
    coors = torch.stack([cxx, cyy, czz], dim=-1)

    index = index[occ != 17]
    coors = coors[occ != 17]
    names = name_mapper[occ[occ != 17].cpu().numpy()]

    dyn_mask = np.isin(names, DYNAMIC_CLASSES)
    index = index[dyn_mask]
    coors = coors[dyn_mask]
    names = names[dyn_mask]

    return index, coors, names, torch.eye(4, device=DEVICE, dtype=torch.float32)

def occ3d_saver(args, data, index, flow):
    scene_name, sample_token = data['scene_name'], data['token']
    save_root = args.save_root if args.save_root is not None \
        else osp.join(args.data_root, 'occ3d_flow')
    occ_file = osp.join(save_root, scene_name, sample_token, 'flow.npz')
    os.makedirs(osp.dirname(occ_file), exist_ok=True)
    np.savez(occ_file, index=index, flow=flow, curr=args.n_his)


def occupancy_loader(data_root):
    raise NotImplementedError


def occupancy_saver():
    raise NotImplementedError


def quick_stop(args, data, saver):
    index = np.zeros((0, 3), dtype=np.int64)
    flow = np.zeros((0, args.n_his + args.n_fut + 1, 3), dtype=np.float32)
    saver(args, data, index, flow)


def collect_boxes(other, base):
    if other is None:
        return dict(
            gt_boxes=torch.zeros((0, 7), device=DEVICE, dtype=torch.float32),
            gt_names=np.zeros((0, ), dtype=np.dtype('<U20')),
            identity=np.zeros((0, ), dtype=np.dtype('<U32')),
            mat=torch.eye(4, device=DEVICE, dtype=torch.float32)
        )
    
    if other is not base:
        other_l2e_t = other['lidar2ego_translation']
        other_l2e_r = other['lidar2ego_rotation']
        other_l2e_mat = transform_matrix(other_l2e_t, Quaternion(other_l2e_r))
        other_e2g_t = other['ego2global_translation']
        other_e2g_r = other['ego2global_rotation']
        other_e2g_mat = transform_matrix(other_e2g_t, Quaternion(other_e2g_r))
        other_l2g_mat = other_e2g_mat @ other_l2e_mat

        base_l2e_t = base['lidar2ego_translation']
        base_l2e_r = base['lidar2ego_rotation']
        base_e2l_mat = transform_matrix(base_l2e_t, Quaternion(base_l2e_r), inverse=True)
        base_e2g_t = base['ego2global_translation']
        base_e2g_r = base['ego2global_rotation']
        base_g2e_mat = transform_matrix(base_e2g_t, Quaternion(base_e2g_r), inverse=True)
        base_g2l_mat = base_e2l_mat @ base_g2e_mat

        other2base = base_g2l_mat @ other_l2g_mat
        other2base = torch.tensor(other2base, device=DEVICE, dtype=torch.float32)
    else:
        other2base = torch.eye(4, device=DEVICE, dtype=torch.float32)
    
    gt_boxes, gt_names, identity = other['gt_boxes'], other['gt_names'], other['identity']
    dyn_mask = np.isin(gt_names, DYNAMIC_CLASSES)
    gt_boxes = torch.tensor(gt_boxes[dyn_mask], device=DEVICE, dtype=torch.float32)
    gt_names = gt_names[dyn_mask]
    identity = identity[dyn_mask]

    return dict(gt_boxes=gt_boxes, gt_names=gt_names, identity=identity, mat=other2base)


def transform_boxes(info, lidar2occ):
    if info['gt_boxes'].shape[0] == 0:
        return dict(gt_boxes=info['gt_boxes'],
                    gt_names=info['gt_names'],
                    identity=info['identity'])
    
    gt_boxes = info['gt_boxes']
    gt_names = info['gt_names']
    identity = info['identity']
    mat = lidar2occ @ info['mat']

    centers = torch.cat([gt_boxes[:, :3], torch.ones_like(gt_boxes[:, :1])], dim=-1)
    centers = (mat @ centers.T).T[:, :3]
    theta_pts = torch.stack([gt_boxes[:, 0] + torch.cos(gt_boxes[:, 6]),
                             gt_boxes[:, 1] + torch.sin(gt_boxes[:, 6]),
                             gt_boxes[:, 2], 
                             torch.ones_like(gt_boxes[:, 1])], dim=-1)
    theta_pts = (mat @ theta_pts.T).T[:, :3] - centers

    gt_boxes[:, :3] = centers
    gt_boxes[:, 6] = torch.atan2(theta_pts[:, 1], theta_pts[:, 0])

    return dict(gt_boxes=gt_boxes,
                gt_names=gt_names,
                identity=identity)


def cal_points_in_boxes(points, pt_names, info):
    boxes, box_names, identity = info['gt_boxes'], info['gt_names'], info['identity']
    bottom_boxes = boxes.clone()

    expand_rate = torch.ones_like(bottom_boxes[:, 3:6])
    expand_rate[:, 0] = 1.3 # expand a bit in x-y plane to cover more points
    expand_rate[:, 1] = 1.5 # 
    expand_rate[:, 2] = 1.6 # expand more in z-axis to cover more points
    expand_rate[box_names == 'pedestrian', :2] = 2 # pedestrians are too small, we need to expand more

    bottom_boxes[:, 3:6] *= expand_rate
    bottom_boxes[:, 2] -= bottom_boxes[:, 5] / 2
    in_box_mask = points_in_boxes_all(coors[None, ...], bottom_boxes[None, ...])[0].bool()
    name_match_mask = torch.tensor(pt_names[:, None] == box_names[None, :], device=DEVICE)

    mask = in_box_mask & name_match_mask
    value, box_index = mask.float().max(dim=1)
    box_index[value == 0] = -1
    return box_index


def match_boxes(box_infos, curr_boxes, curr_ids):
    prev_boxes_list = []
    for box_info in box_infos:
        prev_boxes = curr_boxes.clone()
        boxes, identity = box_info['gt_boxes'], box_info['identity']
        if boxes.shape[0] != 0:
            match = boxes.new_tensor(curr_ids[:, None] == identity[None, :])
            value, index = match.float().max(dim=1)
            prev_boxes[value == 1] = boxes[index[value == 1]]
        prev_boxes_list.append(prev_boxes)
        curr_boxes = prev_boxes
    return prev_boxes_list



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize results')
    parser.add_argument('--dataset', choices=['occ3d', 'occupancy'], default='occ3d')
    parser.add_argument('--data_root', default='data/nuscenes/', type=str, help='Path to config file')
    parser.add_argument('--occ_root', default='data/nuscenes/gts', type=str, help='Path to config file')
    parser.add_argument('--save_root', default=None, type=str, help='Path to config file')
    parser.add_argument('--ann_files', nargs='+', type=str,
                        default=['nuscenes_infos_train_sweep2.pkl', 'nuscenes_infos_val_sweep2.pkl'])
    parser.add_argument('--n_his', default=7, type=int, help='Path to config file')
    parser.add_argument('--n_fut', default=6, type=int, help='Path to config file')
    args = parser.parse_args()

    data = []
    for ann_file in args.ann_files:
        res = pickle.load(open(osp.join(args.data_root, ann_file), 'rb'))
        data.extend(res['infos'])

    if args.dataset == 'occ3d':
        loader, saver = occ3d_loader, occ3d_saver
    elif args.dataset == 'occupancy':
        loader, saver = occupancy_loader, occupancy_saver
    else:
        raise NotImplementedError
    
    scene_collector = defaultdict(list)
    for d in data:
        scene_collector[d['scene_name']].append(d)
    for value in scene_collector.values():
        value.sort(key=lambda x: x['timestamp'])
    
    for name, values in tqdm(scene_collector.items(), leave=True):
        nseq = len(values)
        for i in tqdm(range(len(values)), leave=False):
            infos = []
            for j in range(-args.n_his, args.n_fut+1):
                if i + j < 0 or i + j > nseq - 1:
                    infos.append(collect_boxes(None, values[i]))
                else:
                    infos.append(collect_boxes(values[i + j], values[i]))
            
            index, coors, names, ego2occ = loader(args, values[i])
            if index.shape[0] == 0:
                quick_stop(args, values[i], saver)
                continue

            l2e_t = values[i]['lidar2ego_translation']
            l2e_r = values[i]['lidar2ego_rotation']
            lidar2ego = ego2occ.new_tensor(transform_matrix(l2e_t, Quaternion(l2e_r)))
            lidar2occ = ego2occ @ lidar2ego
            box_infos = [transform_boxes(info, lidar2occ) for info in infos]

            curr_info = box_infos[args.n_his]
            box_index = cal_points_in_boxes(coors, names, curr_info)
            index = index[box_index != -1]
            coors = coors[box_index != -1]
            box_index = box_index[box_index != -1]
            if index.shape[0] == 0:
                quick_stop(args, values[i], saver)
                continue

            pt_ids = curr_info['identity'][box_index.cpu().numpy()]
            pt_curr_boxes = curr_info['gt_boxes'][box_index] # (P, 7)
            pt_prev_boxes_list = match_boxes(
                box_infos[args.n_his-1::-1], pt_curr_boxes, pt_ids)[::-1]
            pt_fut_boxes_list = match_boxes(
                box_infos[args.n_his+1:], pt_curr_boxes, pt_ids)
            pt_boxes = pt_prev_boxes_list + [pt_curr_boxes] + pt_fut_boxes_list
            pt_boxes = torch.stack(pt_boxes, dim=1) # (P, T, 7)

            offset = coors - pt_curr_boxes[:, :3]
            Cos, Sin = torch.cos(pt_curr_boxes[:, 6]), torch.sin(pt_curr_boxes[:, 6])
            mat = torch.stack([Cos, Sin, -Sin, Cos], dim=-1).reshape(-1, 2, 2)
            offset[:, :2] = torch.matmul(mat, offset[:, :2, None]).squeeze(-1)

            offset = offset[:, None, :].expand(-1, pt_boxes.shape[1], -1).clone()
            Cos, Sin = torch.cos(pt_boxes[:, :, 6]), torch.sin(pt_boxes[:, :, 6])
            mat = torch.stack([Cos, -Sin, Sin, Cos], dim=-1).reshape(-1, pt_boxes.shape[1], 2, 2)
            offset[..., :2] = torch.matmul(mat, offset[..., :2, None]).squeeze(-1)
            flow = pt_boxes[:, :, :3] + offset - coors[:, None, :]

            saver(args, values[i], index.cpu().numpy(), flow.cpu().numpy())
