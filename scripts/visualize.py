import os
import sys
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, path)

import os
import cv2
import argparse
import importlib
import os.path as osp
import mayavi.mlab as mlab
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import matplotlib as mpl
from datetime import datetime
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet.apis import set_random_seed
from mmdet.datasets import build_dataset
from mmdet.models import build_model
from loaders.builder import build_dataloader


classname_to_color = {  # RGB.
    0: (0, 0, 0),  # Black. noise
    1: (112, 128, 144),  # Slategrey barrier
    2: (220, 20, 60),  # Crimson bicycle
    3: (255, 127, 80),  # Orangered bus
    4: (255, 158, 0),  # Orange car
    5: (233, 150, 70),  # Darksalmon construction
    6: (255, 61, 99),  # Red motorcycle
    7: (0, 0, 230),  # Blue pedestrian
    8: (47, 79, 79),  # Darkslategrey trafficcone
    9: (255, 140, 0),  # Darkorange trailer
    10: (255, 99, 71),  # Tomato truck
    11: (0, 207, 191),  # nuTonomy green driveable_surface
    12: (175, 0, 75),  # flat other
    13: (75, 0, 75),  # sidewalk
    14: (112, 180, 60),  # terrain
    15: (222, 184, 135),  # Burlywood mannade
    16: (0, 175, 0),  # Green vegetation
    17: (140, 140, 140),  # Green vegetation
}

occ3d_palette = np.array([classname_to_color[i] for i in range(len(classname_to_color))])
agnostic_colormap = mpl.colormaps['viridis']
agnostic_palette = (agnostic_colormap(np.linspace(0., 1., 40))[:, :3] * 255).astype(np.uint8)
patch_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]


def decode_points(points, pc_range=[-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]):
    points = points.copy()
    points[..., 0] = points[..., 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
    points[..., 1] = points[..., 1] * (pc_range[4] - pc_range[1]) + pc_range[1]
    points[..., 2] = points[..., 2] * (pc_range[5] - pc_range[2]) + pc_range[2]
    return points


def visualize_occ(x, y, z, labels, palette, voxel_size, classes=None, mode='cube', 
                  color=None, vmin=None, vmax=None, show=False):
    if palette.shape[1] == 3:
        palette = np.concatenate([palette, np.ones((palette.shape[0], 1)) * 255], axis=1)
    fig = mlab.figure(size=(1000, 1000), bgcolor=(1, 1, 1))

    if vmin is None:
        vmin = 1.0
    if vmax is None:
        vmax = len(classes)-1 if isinstance(classes, list) else classes-1
    
    plot = mlab.points3d(x, y, z,
                         labels,
                         color=color,
                         scale_factor=voxel_size,
                         mode=mode,
                         scale_mode = "vector",
                         opacity=1.0,
                         vmin=vmin,
                         vmax=vmax)
    plot.module_manager.scalar_lut_manager.lut.table = palette
    
    f = mlab.gcf()
    f.scene._lift()

    if show:
        mlab.show()
    else:
        save_fig = mlab.screenshot()
        mlab.close()
        return save_fig


def filter_by_range(x, y, z, *others, range):
    mask = (x >= range[0]) & (x <= range[3]) & \
           (y >= range[1]) & (y <= range[4]) & \
           (z >= range[2]) & (z <= range[5])
    results = (x[mask], y[mask], z[mask])
    if len(others) > 0:
        for other in others:
            results += (other[mask], )
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize results')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--weights', required=True, help='Path to checkpoint')
    parser.add_argument('--vis-input', action='store_true', help='Visualize inputs')
    parser.add_argument('--vis-gt', action='store_true', help='Visualize ground-truths')
    parser.add_argument('--vis-raw-pts', action='store_true', help='Visualize raw points')
    parser.add_argument('--occ-file', default='data/nuscenes/gts/', help='Path to Occ3D')
    parser.add_argument('--save-dir', type=str, default='visualizations', help='Visualize results')
    parser.add_argument('--override', nargs='+', action=DictAction, help='Override config')
    args = parser.parse_args()

    # parse configs
    cfgs = Config.fromfile(args.config)
    if args.override is not None:
        cfgs.merge_from_dict(args.override)
    
    run_name = osp.splitext(osp.split(args.config)[-1])[0]
    run_name += '_' + datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    work_dir = os.path.join(args.save_dir, run_name)
    if os.path.exists(work_dir):
        raise FileExistsError('Directory already exists')
    os.makedirs(work_dir, exist_ok=True)

    # register custom module
    importlib.import_module('models')
    importlib.import_module('loaders')

    set_random_seed(0, deterministic=True)
    cudnn.benchmark = True

    for p in cfgs.data.val.pipeline:
        if p['type'] == 'LoadMultiViewImageFromMultiSweeps':
            p['force_offline'] = True
    val_dataset = build_dataset(cfgs.data.val)
    val_loader = build_dataloader(
        val_dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfgs.data.workers_per_gpu,
        num_gpus=1,
        dist=False,
        shuffle=False,
        seed=0,
    )

    model = build_model(cfgs.model)
    model.cuda()
    model = MMDataParallel(model, [0])

    load_checkpoint(model, args.weights, map_location='cuda', strict=False)
    model.eval()

    scene_range = cfgs.point_cloud_range
    voxel_size = cfgs.voxel_size
    W = int((scene_range[3] - scene_range[0]) / voxel_size[0])
    H = int((scene_range[4] - scene_range[1]) / voxel_size[1])
    Z = int((scene_range[5] - scene_range[2]) / voxel_size[2])

    x = (np.arange(0, W) + 0.5) * voxel_size[0] + scene_range[0]
    y = (np.arange(0, H) + 0.5) * voxel_size[1] + scene_range[1]
    z = (np.arange(0, Z) + 0.5) * voxel_size[2] + scene_range[2]
    xx = x[:, None, None].repeat(H, axis=1).repeat(Z, axis=2)
    yy = y[None, :, None].repeat(W, axis=0).repeat(Z, axis=2)
    zz = z[None, None, :].repeat(W, axis=0).repeat(H, axis=1)

    vis_ndarray = []
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            # Visualize input data
            if args.vis_input:
                images = data['img'][0].data[0].cpu().numpy()[0, :6]
                filenames = data['img_metas'][0].data[0][0]['filename'][:6]
                camera_names = [f.split('/')[-2] for f in filenames]

                camera_order = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                                'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT']
                ordered_images = [0 for _ in range(len(images))]
                for camera_name, img in zip(camera_names, images):
                    img = img.transpose(1, 2, 0).copy()
                    cv2.putText(img, camera_name, (0, 20), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 0), 2)
                    index = camera_order.index(camera_name)
                    ordered_images[index] = img
                
                H, W = ordered_images[0].shape[:2]
                ordered_images = np.stack(ordered_images, axis=0).reshape(2, 3, H, W, 3)
                ordered_images = ordered_images.transpose(0, 2, 1, 3, 4)
                ordered_images = ordered_images.reshape(2 * H, 3 * W, 3)
                cv2.imwrite(osp.join(work_dir, f'{i:0>6}_input.jpg'), ordered_images)

            # Visualize ground-truths
            if args.vis_gt:
                scene_name = val_dataset.data_infos[i]['scene_name']
                token = val_dataset.data_infos[i]['token']
                occ_file = osp.join(args.occ_file, scene_name, token, 'labels.npz')
                occ = np.load(occ_file)['semantics']
                x, y, z = xx[occ!=17], yy[occ!=17], zz[occ!=17]
                label = occ[occ!=17].astype(np.int64)
                x, y, z, label = filter_by_range(x, y, z, label, range=patch_range)
                img = visualize_occ(
                    x, y, z,
                    label,
                    occ3d_palette,
                    0.4,
                    list(classname_to_color.keys()),
                    show=False)
                cv2.imwrite(osp.join(work_dir, f'{i:0>6}_gt.jpg'), img[..., ::-1])

            if args.vis_raw_pts:
                img, img_metas = data['img'][0].data[0], data['img_metas'][0].data[0]
                _model = model.module
                img_feats = _model.extract_feat(img=img.cuda(), img_metas=img_metas)
                outs = _model.pts_bbox_head(img_feats, img_metas)

                refine_pts = outs['all_refine_pts'][-1].reshape(-1, 3)
                refine_pts = refine_pts.detach().cpu().numpy()
                refine_pts = decode_points(refine_pts, scene_range)

                cls_scores = outs['all_cls_scores'][-1]
                if cls_scores is None or cls_scores.ndim == 2:
                    label = (refine_pts[:, 2] - patch_range[2]) / (patch_range[5] - patch_range[2])
                    palette = agnostic_palette
                    vmin=0.05
                    vmax=0.95
                elif cls_scores.ndim == 4:
                    cls_scores = cls_scores.reshape(-1, 17)
                    cls_scores = cls_scores.detach().cpu().numpy()
                    label = cls_scores.argmax(axis=-1)
                    palette = occ3d_palette
                    vmin=1.0
                    vmax=17.0
                else:
                    raise NotImplementedError

                x, y, z = refine_pts[:, 0], refine_pts[:, 1], refine_pts[:, 2]
                x, y, z, label = filter_by_range(x, y, z, label, range=patch_range)
                img = visualize_occ(
                    x, y, z,
                    label,
                    palette,
                    0.4,
                    vmin=vmin,
                    vmax=vmax,
                    show=False)
                cv2.imwrite(osp.join(work_dir, f'{i:0>6}_raw_pts.jpg'), img[..., ::-1])

            result = model(return_loss=False, rescale=True, **data)
            label, pos = result[0]['sem_pred'], result[0]['occ_loc']
            x = xx[pos[:, 0], pos[:, 1], pos[:, 2]]
            y = yy[pos[:, 0], pos[:, 1], pos[:, 2]]
            z = zz[pos[:, 0], pos[:, 1], pos[:, 2]]
            x, y, z, label = filter_by_range(x, y, z, label, range=patch_range)
            img = visualize_occ(
                x, y, z,
                label,
                occ3d_palette,
                0.4,
                list(classname_to_color.keys()),
                show=False)
            cv2.imwrite(osp.join(work_dir, f'{i:0>6}_result.jpg'), img[..., ::-1])
