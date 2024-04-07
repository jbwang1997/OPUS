import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32, BaseModule
from mmcv.ops import knn
from mmdet.core import multi_apply, reduce_mean
from mmdet.models import HEADS
from mmdet.models.utils import build_transformer
from mmdet.models.builder import build_loss
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
from .bbox.utils import normalize_bbox, encode_points, decode_points
from .utils import VERSION



@HEADS.register_module()
class PointOccHead(BaseModule):
    def __init__(self,
                 num_classes,
                 in_channels,
                 num_query,
                 transformer=None,
                 pc_range=[],
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
        self.loss_cls = build_loss(loss_cls)
        self.loss_pts = build_loss(loss_pts)
        self.transformer = build_transformer(transformer)
        self.embed_dims = self.transformer.embed_dims
        self.num_points = self.transformer.num_points
        self._init_layers()

    def _init_layers(self):
        self.init_query_points = nn.Embedding(self.num_query, 3)
        nn.init.uniform_(self.init_query_points.weight, 0, 1)

    def init_weights(self):
        self.transformer.init_weights()

    def forward(self, mlvl_feats, img_metas):
        B, P = mlvl_feats[0].shape[0], self.num_points
        init_points = self.init_query_points.weight[None, :, None, :]
        query_points = init_points.repeat(B, 1, P, 1)
        # query_feat = self.init_query_feat.weight[None, :, :].repeat(B, 1, 1)
        query_feat = query_points.new_zeros((*query_points.shape[:2], self.embed_dims))
        cls_scores, refine_pts = self.transformer(
            query_points,
            query_feat,
            mlvl_feats,
            img_metas=img_metas,
        )

        return dict(init_points=init_points,
                    all_cls_scores=cls_scores,
                    all_refine_pts=refine_pts)

    @torch.no_grad()
    def _get_target_single(self, refine_pts, gt_points, gt_labels):
        gt_paired_idx = knn(1, refine_pts[None, ...], gt_points[None, ...])
        gt_paired_idx = gt_paired_idx.permute(0, 2, 1).squeeze().long()

        pred_paired_idx = knn(1, gt_points[None, ...], refine_pts[None, ...])
        pred_paired_idx = pred_paired_idx.permute(0, 2, 1).squeeze().long()

        refine_pts_labels = gt_labels[pred_paired_idx]
        refine_pts_gt_pts = gt_points[pred_paired_idx]
        dist = torch.norm(refine_pts-refine_pts_gt_pts, dim=-1)

        pos_dist = self.train_cfg.get('pos_dist', [0.2, 0.5])
        min_pos_dist, max_pos_dist = min(pos_dist), max(pos_dist)
        dist_thr = (dist.mean() * 0.2).clamp(min_pos_dist, max_pos_dist)
        refine_pts_labels[dist > dist_thr] = self.num_classes

        return refine_pts_labels, gt_paired_idx, pred_paired_idx
    
    def get_targets(self):
        # To instantiate the abstract method
        pass

    def loss_single(self,
                    cls_scores,
                    refine_pts,
                    gt_points_list,
                    gt_labels_list):
        num_imgs = cls_scores.size(0)
        cls_scores = cls_scores.view(num_imgs, -1, self.num_classes)
        refine_pts = refine_pts.view(num_imgs, -1, 3)
        refine_pts = decode_points(refine_pts, self.pc_range)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        refine_pts_list = [refine_pts[i] for i in range(num_imgs)]

        (labels_list, gt_paired_idx_list, pred_paired_idx_list) = multi_apply(
            self._get_target_single, refine_pts_list, gt_points_list, gt_labels_list)
        
        gt_paired_pts, pred_paired_pts= [], []
        for i in range(num_imgs):
            gt_paired_pts.append(refine_pts_list[i][gt_paired_idx_list[i]])
            pred_paired_pts.append(gt_points_list[i][pred_paired_idx_list[i]])
        
        cls_scores = torch.cat(cls_scores_list)
        labels = torch.cat(labels_list)
        loss_cls = self.loss_cls(cls_scores, labels, avg_factor=cls_scores.shape[0])
        
        gt_pts = torch.cat(gt_points_list)
        gt_paired_pts = torch.cat(gt_paired_pts)
        pred_pts = torch.cat(refine_pts_list)
        pred_paired_pts = torch.cat(pred_paired_pts)

        loss_pts = pred_pts.new_tensor(0)
        loss_pts += self.loss_pts(gt_pts, gt_paired_pts, avg_factor=gt_pts.shape[0])
        loss_pts += self.loss_pts(pred_pts, pred_paired_pts, avg_factor=pred_pts.shape[0])

        return loss_cls, loss_pts
    
    @force_fp32(apply_to=('preds_dicts'))
    def loss(self, voxel_semantics, preds_dicts):
        init_points = preds_dicts['init_points']
        all_cls_scores = preds_dicts['all_cls_scores']
        all_refine_pts = preds_dicts['all_refine_pts']

        num_dec_layers = len(all_cls_scores)
        gt_points_list, gt_labels_list = self.get_sparse_voxels(voxel_semantics)
        all_gt_points_list = [gt_points_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]

        losses_cls, losses_pts = multi_apply(
            self.loss_single, all_cls_scores, all_refine_pts,
            all_gt_points_list, all_gt_labels_list
        )

        loss_dict = dict()
        # loss of init_points
        if init_points is not None:
            pseudo_scores = init_points.new_zeros(
                *init_points.shape[:-1], self.num_classes)
            _, init_loss_pts = self.loss_single(
                pseudo_scores, init_points, gt_points_list, gt_labels_list)
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
            scores, labels = cls_scores.max(dim=1)
            if self.test_cfg.get('score_thr', 0) != 0:
                score_thr = self.test_cfg.get('score_thr', 0.1)
                refine_pts = refine_pts[scores > score_thr]
                labels = labels[scores > score_thr]
            
            P, R = refine_pts.shape[:2]
            refine_pts = decode_points(refine_pts, self.pc_range)
            refine_pts = refine_pts.flatten(0, 1)
            occ_index = (refine_pts - pc_range[:3]) / voxel_size
            occ_index = occ_index.long()
            labels = labels[:, None].repeat(1, R)
            labels = labels.flatten(0, 1)

            result_list.append(dict(
                sem_pred=labels.detach().cpu().numpy(),
                occ_loc=occ_index.detach().cpu().numpy()))

        return result_list
    
    def get_sparse_voxels(self, voxel_semantics):
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

        gt_points, gt_labels = [], []
        for i in range(B):
            mask = voxel_semantics[i] != self.num_classes
            gt_points.append(coors[mask])
            gt_labels.append(voxel_semantics[i][mask])
        
        return gt_points, gt_labels
