import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32, BaseModule
from mmcv.ops import knn, Voxelization
from mmdet.core import multi_apply, reduce_mean
from mmdet.models import HEADS
from mmdet.models.utils import build_transformer
from mmdet.models.builder import build_loss
from mmdet.models.backbones.resnet import BasicBlock
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
from .bbox.utils import normalize_bbox, encode_points, decode_points
from .utils import VERSION, ASPP


class Integral(nn.Module):
    def __init__(self, reg_max=80):
        super(Integral, self).__init__()
        self.reg_max = reg_max
        self.register_buffer('project',
                             torch.linspace(0, self.reg_max, self.reg_max + 1))

    def forward(self, x):
        x = F.softmax(x, dim=1)
        x = F.linear(x, self.project.type_as(x))
        return x


@HEADS.register_module()
class PointOccHeadDepth(BaseModule):
    def __init__(self,
                 num_classes,
                 in_channels,
                 num_query,
                 transformer=None,
                 pc_range=[],
                 empty_label=17,
                 voxel_size=[],
                 train_cfg=dict(),
                 test_cfg=dict(max_per_img=100),
                 loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0),
                 loss_pts=dict(type='L1Loss'),
                 loss_cls_depth=dict(type='DistributionFocalLoss'),
                 loss_reg_depth=dict(type='L1Loss', loss_weight=0.5),
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg)
        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.empty_label = empty_label
        self.loss_cls = build_loss(loss_cls)
        self.loss_pts = build_loss(loss_pts)
        self.transformer = build_transformer(transformer)
        self.num_refines = self.transformer.num_refines
        self.embed_dims = self.transformer.embed_dims
        self.voxel_generator = Voxelization(
            voxel_size=self.voxel_size,
            point_cloud_range=self.pc_range,
            max_num_points=10, 
            max_voxels=self.num_query * self.num_refines,
        )
        self.loss_cls_depth = build_loss(loss_cls_depth)
        self.loss_reg_depth = build_loss(loss_reg_depth)
        self._init_layers()

    def _init_layers(self):
        self.init_query_points = nn.Embedding(self.num_query, 3)
        nn.init.uniform_(self.init_query_points.weight, 0, 1)

        proj_branch = [
            nn.Conv2d(self.in_channels * 4, self.in_channels, kernel_size=3, padding=1)]
        for i in range(1):
            proj_branch.append(BasicBlock(self.in_channels, self.in_channels))
        self.proj_branch = nn.Sequential(*proj_branch)
        self.depth_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=85,
            kernel_size=3,
            padding=1
        )
        self.label_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.num_classes,
            kernel_size=3,
            padding=1
        )
        self.integral = Integral(84)

    def init_weights(self):
        self.transformer.init_weights()
    
    def forward_proj_branch(self, mlvl_feats):
        B = mlvl_feats[0].shape[0]
        feats = []
        for feat in mlvl_feats:
            C, H, W = feat.shape[-3:]
            feat = feat[:, :6].reshape(B*6, C, H, W)
            feats.append(F.upsample(feat, size=(32, 88)))
        feats = torch.cat(feats, dim=1)

        feats = self.proj_branch(feats)
        depth = self.depth_conv(feats).reshape(B, 6, 85, 32, 88)
        proj_label = self.label_conv(feats).reshape(B, 6, self.num_classes, 32, 88)
        return depth, proj_label

    def forward(self, mlvl_feats, img_metas):
        depth, proj_label = self.forward_proj_branch(mlvl_feats)

        B = mlvl_feats[0].shape[0]
        query_points = self.init_query_points.weight[None, ...].repeat(B, 1, 1)
        # query_feat = self.init_query_feat.weight[None, :, :].repeat(B, 1, 1)
        query_feat = query_points.new_zeros((*query_points.shape[:2], self.embed_dims))
        #[D B, Q, P, num_classes / 3]
        cls_scores, refine_pts = self.transformer(
            query_points,
            query_feat,
            mlvl_feats,
            img_metas=img_metas,
        )

        return dict(depth_pred=depth,
                    label_pred=proj_label,
                    init_points=query_points.unsqueeze(2),
                    all_cls_scores=cls_scores,
                    all_refine_pts=refine_pts)

    @torch.no_grad()
    def _get_target_single(self, refine_pts, gt_points, gt_masks, gt_labels):
        gt_paired_idx = knn(1, refine_pts[None, ...], gt_points[None, ...])
        gt_paired_idx = gt_paired_idx.permute(0, 2, 1).squeeze().long()

        pred_paired_idx = knn(1, gt_points[None, ...], refine_pts[None, ...])
        pred_paired_idx = pred_paired_idx.permute(0, 2, 1).squeeze().long()
        refine_pts_labels = gt_labels[pred_paired_idx]

        gt_pts_preds = refine_pts[gt_paired_idx]
        weights = refine_pts.new_ones(gt_pts_preds.shape[0])
        dist = torch.norm(gt_points - gt_pts_preds, dim=-1)
        empty_dist_thr = self.train_cfg.get('empty_dist_thr', 0.2)
        empty_weights = self.train_cfg.get('empty_weights', 5)
        mask = (dist > empty_dist_thr) & gt_masks
        weights[mask] = empty_weights

        rare_classes = self.train_cfg.get('rare_classes', [0, 2, 5, 8])
        rare_weights = self.train_cfg.get('rare_weights', 10)
        for cls_idx in rare_classes:
            mask = (gt_labels == cls_idx) & gt_masks
            weights[mask] = weights[mask].clamp(min=rare_weights)

        return refine_pts_labels, weights, gt_paired_idx, pred_paired_idx
    
    def get_targets(self):
        # To instantiate the abstract method
        pass

    def loss_single(self,
                    cls_scores,
                    refine_pts,
                    gt_points_list,
                    gt_masks_list,
                    gt_labels_list):
        num_imgs = cls_scores.size(0)
        cls_scores = cls_scores.view(num_imgs, -1, self.num_classes)
        refine_pts = refine_pts.view(num_imgs, -1, 3)
        refine_pts = decode_points(refine_pts, self.pc_range)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        refine_pts_list = [refine_pts[i] for i in range(num_imgs)]

        (labels_list, gt_weights, gt_paired_idx_list, pred_paired_idx_list) = \
            multi_apply(self._get_target_single, refine_pts_list, gt_points_list, 
                        gt_masks_list, gt_labels_list)
        
        gt_paired_pts, pred_paired_pts= [], []
        for i in range(num_imgs):
            gt_paired_pts.append(refine_pts_list[i][gt_paired_idx_list[i]])
            pred_paired_pts.append(gt_points_list[i][pred_paired_idx_list[i]])
        
        cls_scores = torch.cat(cls_scores_list)
        labels = torch.cat(labels_list)
        loss_cls = self.loss_cls(cls_scores, labels, avg_factor=cls_scores.shape[0])
        
        gt_pts = torch.cat(gt_points_list)
        gt_weights = torch.cat(gt_weights)
        gt_paired_pts = torch.cat(gt_paired_pts)
        pred_pts = torch.cat(refine_pts_list)
        pred_paired_pts = torch.cat(pred_paired_pts)

        loss_pts = pred_pts.new_tensor(0)
        loss_pts += self.loss_pts(
            gt_pts, gt_paired_pts, weight=gt_weights[..., None], avg_factor=gt_pts.shape[0])
        loss_pts += self.loss_pts(pred_pts, pred_paired_pts, avg_factor=pred_pts.shape[0])

        return loss_cls, loss_pts
    
    @force_fp32(apply_to=('preds_dicts'))
    def loss(self, depth_map, proj_label, voxel_semantics, mask_camera, preds_dicts):
        # voxelsemantics [B, X200, Y200, Z16] unocuupied=17
        init_points = preds_dicts['init_points']
        all_cls_scores = preds_dicts['all_cls_scores']
        all_refine_pts = preds_dicts['all_refine_pts']

        num_dec_layers = len(all_cls_scores)
        gt_points_list, gt_masks_list, gt_labels_list = \
            self.get_sparse_voxels(voxel_semantics, mask_camera)
        all_gt_points_list = [gt_points_list for _ in range(num_dec_layers)]
        all_gt_masks_list = [gt_masks_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]

        losses_cls, losses_pts = multi_apply(
            self.loss_single, all_cls_scores, all_refine_pts, 
            all_gt_points_list, all_gt_masks_list, all_gt_labels_list
        )

        loss_dict = dict()
        # loss of init_points
        if init_points is not None:
            pseudo_scores = init_points.new_zeros(
                *init_points.shape[:-1], self.num_classes)
            _, init_loss_pts = self.loss_single(
                pseudo_scores, init_points, gt_points_list, 
                gt_masks_list, gt_labels_list)
            loss_dict['init_loss_pts'] = init_loss_pts

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_pts'] = losses_pts[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_pts_i in zip(losses_cls[:-1], losses_pts[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_pts'] = loss_pts_i
            num_dec_layer += 1
        
        # depth and proj label losses
        depth_pred = preds_dicts['depth_pred']
        label_pred = preds_dicts['label_pred']
        valid_mask = proj_label != 17

        depth_target = depth_map[valid_mask] * 2
        depth_pred = depth_pred.permute(0, 1, 3, 4, 2)
        depth_pred = depth_pred[valid_mask]
        depth_reg_pred = self.integral(depth_pred)
        loss_reg_depth = self.loss_reg_depth(
            depth_reg_pred, depth_target, avg_factor=depth_pred.shape[0])
        loss_cls_depth = self.loss_cls_depth(
            depth_pred, depth_target, avg_factor=depth_pred.shape[0])
        loss_dict['loss_reg_depth'] = loss_reg_depth
        loss_dict['loss_cls_depth'] = loss_cls_depth

        label_target = proj_label[valid_mask]
        label_pred = label_pred.permute(0, 1, 3, 4, 2)
        label_pred = label_pred[valid_mask]
        loss_proj_cls = self.loss_cls(
            label_pred, label_target, avg_factor=label_pred.shape[0])
        loss_dict['loss_proj_cls'] = loss_proj_cls
        
        return loss_dict
    
    def get_occ(self, pred_dicts, img_metas, rescale=False):
        all_cls_scores = pred_dicts['all_cls_scores']
        all_refine_pts = pred_dicts['all_refine_pts']
        cls_scores = all_cls_scores[-1].sigmoid()
        refine_pts = all_refine_pts[-1]

        batch_size = refine_pts.shape[0]
        pc_range = refine_pts.new_tensor(self.pc_range)
        voxel_size = refine_pts.new_tensor(self.voxel_size)
        result_list = []
        for i in range(batch_size):
            refine_pts, cls_scores = refine_pts[i], cls_scores[i]
            refine_pts = refine_pts.flatten(0, 1)
            cls_scores = cls_scores.flatten(0, 1)

            refine_pts = decode_points(refine_pts, self.pc_range)
            pts = torch.cat([refine_pts, cls_scores], dim=-1)
            pts_infos, voxels, num_pts = self.voxel_generator(pts)
            voxels = torch.flip(voxels, [1])

            pts, scores = pts_infos[..., :3], pts_infos[..., 3:]
            scores = scores.sum(dim=1) / num_pts[..., None]
            scores, labels = scores.max(dim=-1)

            if self.test_cfg.get('score_thr', 0) != 0:
                score_thr = self.test_cfg.get('score_thr', 0)
                voxels = voxels[scores > score_thr]
                labels = labels[scores > score_thr]
            
            result_list.append(dict(
                sem_pred=labels.detach().cpu().numpy(),
                occ_loc=voxels.detach().cpu().numpy()))

        return result_list
    
    def get_sparse_voxels(self, voxel_semantics, mask_camera):
        B, W, H, Z = voxel_semantics.shape
        device = voxel_semantics.device
        voxel_semantics = voxel_semantics.long()
        pc_range = torch.tensor(self.pc_range, device=device).float()
        scene_size = pc_range[3:] - pc_range[:3]

        x = torch.arange(0, W, dtype=torch.float32, device=device)
        x = (x + 0.5) / W * scene_size[0] + pc_range[0]
        y = torch.arange(0, H, dtype=torch.float32, device=device)
        y = (y + 0.5) / H * scene_size[1] + pc_range[1]
        z = torch.arange(0, Z, dtype=torch.float32, device=device)
        z = (z + 0.5) / Z * scene_size[2] + pc_range[2]

        xx = x[:, None, None].expand(W, H, Z)
        yy = y[None, :, None].expand(W, H, Z)
        zz = z[None, None, :].expand(W, W, Z)
        coors = torch.stack([xx, yy, zz], dim=-1)

        gt_points, gt_masks, gt_labels = [], [], []
        for i in range(B):
            mask = voxel_semantics[i] != self.empty_label
            gt_points.append(coors[mask])
            gt_masks.append(mask_camera[i][mask])
            gt_labels.append(voxel_semantics[i][mask])
        
        return gt_points, gt_masks, gt_labels
