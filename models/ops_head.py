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
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
from .bbox.utils import normalize_bbox, encode_points, decode_points
from .utils import VERSION
import numpy as np

manual_list=[11, 16, 15, 14, 13,  4, 10, 12,  3,  9,  7,  1,  0,  5,  8,  6,  2]
manual_list.reverse()
manual_weight=np.ones(len(manual_list))
manual_list_array=np.array(manual_list)
manual_weight[manual_list_array[:5]]=10
manual_weight[manual_list_array[5:12]]=5

manual_weight2=manual_weight.copy()
manual_weight2[15]=2


@HEADS.register_module()
class OPSHead(BaseModule):
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
                 init_cfg=None,
                 manual_set=False,
                 manual_mode='1',
                 dis_mode='fb',
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
            max_voxels=self.num_query * self.num_refines[-1],
        )

        if manual_set:
            if manual_mode=='1':
                self.cls_weight=manual_weight
            elif manual_mode=='2':
                self.cls_weight=manual_weight2
            self.cls_weight=torch.from_numpy(self.cls_weight)
        else:
            self.cls_weight=torch.ones(num_classes) # 17

        self.dis_mode=dis_mode

        self._init_layers()

    def _init_layers(self):
        self.init_points = nn.Embedding(self.num_query, 3)
        nn.init.uniform_(self.init_points.weight, 0, 1)
        # nn.init.uniform_(self.init_points_offset.weight, -0.1, 0.1)

    def init_weights(self):
        self.transformer.init_weights()

    def forward(self, mlvl_feats, img_metas):
        B, Q, = mlvl_feats[0].shape[0], self.num_query
        init_points = self.init_points.weight[None, :, None, :].repeat(B, 1, 1, 1)
        query_feat = init_points.new_zeros(B, Q, self.embed_dims)

        cls_scores, refine_pts = self.transformer(
            init_points,
            query_feat,
            mlvl_feats,
            img_metas=img_metas,
        )

        return dict(init_points=init_points,
                    all_cls_scores=cls_scores,
                    all_refine_pts=refine_pts)


    @torch.no_grad()
    def _get_target_single(self, refine_pts, gt_points, gt_masks, gt_labels):
        gt_paired_idx = knn(1, refine_pts[None, ...], gt_points[None, ...])
        gt_paired_idx = gt_paired_idx.permute(0, 2, 1).squeeze().long()

        pred_paired_idx = knn(1, gt_points[None, ...], refine_pts[None, ...])
        pred_paired_idx = pred_paired_idx.permute(0, 2, 1).squeeze().long()
        refine_pts_labels = gt_labels[pred_paired_idx] # assign the neareast gt label to each point 

        gt_pts_preds = refine_pts[gt_paired_idx] # get the neareast pts of every gt
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

    @torch.no_grad()
    def _get_dis_weight(self,pts_list,H=40,W=40,mode='fb'):

        d_max=(H **2 + W **2) ** 0.5
        pred_pts=torch.cat(pts_list)
        d=torch.norm(pred_pts[:,:2],2,-1)

        if mode == 'fb':
            weight= d/d_max + 1
        elif mode=='fb2':
            weight= torch.sqrt(d/d_max) + 1
        elif mode=='fb3':
            weight=torch.square(d/d_max) + 1
        elif mode =='fb4':
            weight= 2 * (d/d_max) + 1
        elif mode =='fb5':
            weight= 0.5 * (d/d_max) + 1
        
        return weight

    def loss_single(self,
                    cls_scores,
                    refine_pts,
                    gt_points_list,
                    gt_masks_list,
                    gt_labels_list):
        num_imgs = cls_scores.size(0) # B
        cls_scores = cls_scores.reshape(num_imgs, -1, self.num_classes)
        refine_pts = refine_pts.reshape(num_imgs, -1, 3)
        refine_pts = decode_points(refine_pts, self.pc_range)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        refine_pts_list = [refine_pts[i] for i in range(num_imgs)]

        (labels_list, gt_weights, gt_paired_idx_list, pred_paired_idx_list) = \
            multi_apply(self._get_target_single, refine_pts_list, gt_points_list, 
                        gt_masks_list, gt_labels_list)
        
        gt_paired_pts, pred_paired_pts= [], []
        for i in range(num_imgs):
            gt_paired_pts.append(refine_pts_list[i][gt_paired_idx_list[i]]) # the neareast pts to gt
            pred_paired_pts.append(gt_points_list[i][pred_paired_idx_list[i]]) # the neareast gt to pts
        
        cls_scores = torch.cat(cls_scores_list)
        labels = torch.cat(labels_list)
        cls_weight=self.cls_weight[None,:].repeat(labels.shape[0],1)

        dis_weight=self._get_dis_weight(pred_paired_pts,mode=self.dis_mode)
        cls_weight=cls_weight.type_as(cls_scores) * dis_weight.unsqueeze(-1).type_as(cls_scores)
        
        cls_weight=cls_weight.type_as(cls_scores)
        loss_cls = self.loss_cls(cls_scores, labels, cls_weight,avg_factor=cls_scores.shape[0])
        
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
    def loss(self, voxel_semantics, mask_camera, preds_dicts):
        # voxelsemantics [B, X200, Y200, Z16] unocuupied=17
        init_points = preds_dicts['init_points']
        all_cls_scores = preds_dicts['all_cls_scores'] # 6 ,B,2k4,32,17
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
        coors = torch.stack([xx, yy, zz], dim=-1) # actual space

        gt_points, gt_masks, gt_labels = [], [], []
        for i in range(B):
            mask = voxel_semantics[i] != self.empty_label
            gt_points.append(coors[mask])
            gt_masks.append(mask_camera[i][mask]) # camera mask and not empty
            gt_labels.append(voxel_semantics[i][mask])
        
        return gt_points, gt_masks, gt_labels
