import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from mmcv.runner import BaseModule, ModuleList
from mmcv.cnn import bias_init_with_prob, Scale
from mmcv.cnn.bricks.transformer import MultiheadAttention, FFN
from mmcv.ops import knn
from mmdet.models.utils.builder import TRANSFORMER
from .bbox.utils import decode_bbox, decode_points, encode_points
from .utils import inverse_sigmoid, DUMP
from .pointocc_sampling import sampling_4d, make_sample_points
from .checkpoint import checkpoint as cp
from .csrc.wrapper import MSMV_CUDA


@TRANSFORMER.register_module()
class PointOccTransformer(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_frames=8,
                 num_points=4,
                 num_layers=6,
                 num_levels=4,
                 num_classes=10,
                 num_refines=16,
                 reset_query_stage=3,
                 pc_range=[],
                 init_cfg=None):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                            'behavior, init_cfg is not allowed to be set'
        super(PointOccTransformer, self).__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        self.pc_range = pc_range
        self.num_refines = num_refines

        self.decoder = PointOccTransformerDecoder(
            embed_dims, num_frames, num_points, num_layers, num_levels,
            num_classes, num_refines, reset_query_stage, pc_range=pc_range)

    @torch.no_grad()
    def init_weights(self):
        self.decoder.init_weights()

    def forward(self, query_points, query_feat, mlvl_feats, img_metas):
        cls_scores, refine_pts = self.decoder(
            query_points, query_feat, mlvl_feats, img_metas)

        cls_scores = torch.nan_to_num(cls_scores)
        refine_pts = torch.nan_to_num(refine_pts)

        return cls_scores, refine_pts


class PointOccTransformerDecoder(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_frames=8,
                 num_points=4,
                 num_layers=6,
                 num_levels=4,
                 num_classes=10,
                 num_refines=16,
                 reset_query_stage=1,
                 pc_range=[],
                 init_cfg=None):
        super(PointOccTransformerDecoder, self).__init__(init_cfg)
        self.num_layers = num_layers
        self.pc_range = pc_range
        self.reset_query_stage = reset_query_stage

        # params are shared across all decoder layers
        self.decoder_layers = ModuleList()
        for i in range(num_layers):
            self.decoder_layers.append(
                PointOccTransformerDecoderLayer(
                    embed_dims, num_frames, num_points, num_levels, num_classes, 
                    num_refines, layer_idx=i, pc_range=pc_range)
            )

    @torch.no_grad()
    def init_weights(self):
        self.decoder_layers.init_weights()

    def forward(self, query_points, query_feat, mlvl_feats, img_metas):
        cls_scores, refine_pts = [], []

        # organize projections matrix and copy to CUDA
        lidar2img = np.asarray([m['lidar2img'] for m in img_metas]).astype(np.float32)
        lidar2img = torch.from_numpy(lidar2img).to(query_feat.device)  # [B, N, 4, 4]
        ego2lidar = np.asarray([m['ego2lidar'] for m in img_metas]).astype(np.float32)
        ego2lidar = torch.from_numpy(ego2lidar).to(query_feat.device)  # [B, N, 4, 4]
        img_metas[0]['ego2img'] = torch.matmul(lidar2img, ego2lidar)

        # group image features in advance for sampling, see `sampling_4d` for more details
        for lvl, feat in enumerate(mlvl_feats):
            B, TN, GC, H, W = feat.shape  # [B, TN, GC, H, W]
            N, T, G, C = 6, TN // 6, 4, GC // 4
            feat = feat.reshape(B, T, N, G, C, H, W)

            if MSMV_CUDA:  # Our CUDA operator requires channel_last
                feat = feat.permute(0, 1, 3, 2, 5, 6, 4)  # [B, T, G, N, H, W, C]
                feat = feat.reshape(B*T*G, N, H, W, C)
            else:  # Torch's grid_sample requires channel_first
                feat = feat.permute(0, 1, 3, 4, 2, 5, 6)  # [B, T, G, C, N, H, W]
                feat = feat.reshape(B*T*G, C, N, H, W)

            mlvl_feats[lvl] = feat.contiguous()

        for i, decoder_layer in enumerate(self.decoder_layers):
            DUMP.stage_count = i

            query_feat, cls_score, refine_pt = decoder_layer(
                query_points, query_feat, mlvl_feats, img_metas)
            query_points = refine_pt.detach().mean(dim=-2)
            if i < self.reset_query_stage:
                # the query_points vary fastly in early stage
                query_feat = torch.zeros_like(query_feat)

            cls_scores.append(cls_score)
            refine_pts.append(refine_pt)

        cls_scores = torch.stack(cls_scores)
        refine_pts = torch.stack(refine_pts)

        return cls_scores, refine_pts


class PointOccTransformerDecoderLayer(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_frames=8,
                 num_points=4,
                 num_levels=4,
                 num_classes=10,
                 num_refines=16,
                 num_cls_fcs=2,
                 num_reg_fcs=2,
                 layer_idx=0,
                 pc_range=[],
                 init_cfg=None):
        super(PointOccTransformerDecoderLayer, self).__init__(init_cfg)

        self.embed_dims = embed_dims
        self.num_classes = num_classes
        self.pc_range = pc_range
        self.num_points = num_points
        self.num_refines = num_refines

        self.position_encoder = nn.Sequential(
            nn.Linear(3, self.embed_dims), 
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )

        self.self_attn = PointOccSelfAttention(
            embed_dims, num_heads=8, dropout=0.1, pc_range=pc_range)
        self.sampling = PointOccSampling(embed_dims, num_frames=num_frames, num_groups=4,
                                         num_points=num_points, num_levels=num_levels,
                                         pc_range=pc_range)
        self.mixing = AdaptiveMixing(in_dim=embed_dims, in_points=num_points * num_frames,
                                     n_groups=4, out_points=32)
        self.ffn = FFN(embed_dims, feedforward_channels=512, ffn_drop=0.1)

        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)

        cls_branch = []
        for _ in range(num_cls_fcs):
            cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(nn.Linear(
            self.embed_dims, self.num_classes * self.num_refines))
        self.cls_branch = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(num_reg_fcs):
            reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU(inplace=True))
        reg_branch.append(nn.Linear(self.embed_dims, 3 * self.num_refines))
        self.reg_branch = nn.Sequential(*reg_branch)

        self.scale = Scale(8)

    @torch.no_grad()
    def init_weights(self):
        self.self_attn.init_weights()
        self.sampling.init_weights()
        self.mixing.init_weights()

        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.cls_branch[-1].bias, bias_init)

    def refine_points(self, points_proposal, points_delta):
        B, Q = points_delta.shape[:2]
        points_delta = points_delta.reshape(B, Q, self.num_refines, 3)

        points_proposal = decode_points(points_proposal, self.pc_range)
        points_proposal = points_proposal.unsqueeze(2)
        new_points = points_proposal + points_delta
        return encode_points(new_points, self.pc_range)

    def forward(self, query_points, query_feat, mlvl_feats, img_metas):
        """
        query_points: [B, Q, 3] [x, y, z]
        """
        query_pos = self.position_encoder(query_points)
        query_feat = query_feat + query_pos

        sampled_feat = self.sampling(
            query_points, query_feat, mlvl_feats, img_metas, self.scale)
        query_feat = self.norm1(self.mixing(sampled_feat, query_feat))
        query_feat = self.norm2(self.self_attn(query_points, query_feat))
        query_feat = self.norm3(self.ffn(query_feat))

        B, Q = query_points.shape[:2]
        cls_score = self.cls_branch(query_feat)  # [B, Q, P * num_classes]
        reg_offset = self.scale(self.reg_branch(query_feat))  # [B, Q, P * 3]
        cls_score = cls_score.reshape(B, Q, self.num_refines, self.num_classes)
        refine_pt = self.refine_points(query_points, reg_offset)

        if DUMP.enabled:
            pass
            # TODO: enable pointocc dump
            # query_bbox_dec = decode_bbox(query_bbox, self.pc_range)
            # bbox_pred_dec = decode_bbox(bbox_pred, self.pc_range)
            # cls_score_sig = torch.sigmoid(cls_score)
            # torch.save(query_bbox_dec.cpu(), '{}/query_bbox_stage{}.pth'.format(DUMP.out_dir, DUMP.stage_count))
            # torch.save(bbox_pred_dec.cpu(), '{}/bbox_pred_stage{}.pth'.format(DUMP.out_dir, DUMP.stage_count))
            # torch.save(cls_score_sig.cpu(), '{}/cls_score_stage{}.pth'.format(DUMP.out_dir, DUMP.stage_count))

        return query_feat, cls_score, refine_pt


# class PointOccSelfAttention(BaseModule):
#     """Scale-adaptive Self Attention"""
#     def __init__(self,
#                  embed_dims=256,
#                  num_heads=8,
#                  split_groups=3,
#                  dropout=0.1,
#                  pc_range=[],
#                  knn=9,
#                  init_cfg=None):
#         super().__init__(init_cfg)
#         self.pc_range = pc_range
#         self.split_groups = split_groups
#         self.num_heads = num_heads
#         self.knn = knn

#         self.attention = MultiheadAttention(embed_dims, num_heads, dropout, batch_first=True)
#         self.gen_tau = nn.Linear(embed_dims, num_heads)

#     @torch.no_grad()
#     def init_weights(self):
#         nn.init.zeros_(self.gen_tau.weight)
#         nn.init.uniform_(self.gen_tau.bias, 0.0, 2.0)

#     def inner_forward(self, query_points, query_feat, layer_idx):
#         """
#         query_points: [B, Q, 3]
#         query_feat: [B, Q, C]
#         """
#         tau = self.gen_tau(query_feat)  # [B, Q, 8]

#         if DUMP.enabled:
#             torch.save(tau.cpu(), '{}/sasa_tau_stage{}.pth'.format(DUMP.out_dir, DUMP.stage_count))

#         # split groups
#         B, NQ, C = query_feat.shape
#         N, Q = self.split_groups, int(NQ / self.split_groups)

#         near_idx = knn(self.knn, query_points).permute(0, 2, 1).flatten(1) # [B, Q * knn]
#         near_idx = near_idx.unsqueeze(-1).repeat(1, 1, C).long()
#         near_feat = torch.gather(query_feat, 1, near_idx).reshape(B * N * Q, self.knn, C)
#         local_query_feat = self.attention(query=near_feat[:, [0], :], key=near_feat)

#         query_feat = query_feat.reshape(B, N, Q, C)
#         key_feat = torch.roll(query_feat, layer_idx % N, dims=1)

#         # calc attn mask
#         dist = self.calc_points_dists(query_points, layer_idx)
#         tau = tau.reshape(B, N, Q, self.num_heads).permute(0, 1, 3, 2) # [B, N, 8, Q]
#         attn_mask = dist[..., None, :, :] * tau[..., None]

#         query_feat = query_feat.flatten(0, 1)
#         key_feat = key_feat.flatten(0, 1)
#         attn_mask = attn_mask.flatten(0, 2)

#         global_query_feat = self.attention(query=query_feat, key=key_feat, attn_mask=attn_mask)
#         query_feat = local_query_feat.reshape(B, N*Q, C) + global_query_feat.reshape(B, N*Q, C)
#         return query_feat

#     def forward(self, query_points, query_feat, layer_idx):
#         if self.training and query_feat.requires_grad:
#             return cp(self.inner_forward, query_points, query_feat,
#                       layer_idx, use_reentrant=False)
#         else:
#             return self.inner_forward(query_points, query_feat, layer_idx)

#     @torch.no_grad()
#     def calc_points_dists(self, points, layer_idx):
#         B, NQ, C = points.shape
#         N, Q = self.split_groups, int(NQ / self.split_groups)

#         points1 = decode_points(points, self.pc_range).reshape(B, N, Q, C)
#         points2 = torch.roll(points1, layer_idx % N, dims=1)
#         dist = torch.norm(points1.unsqueeze(-2) - points2.unsqueeze(-3), dim=-1)
#         return -dist


class PointOccSelfAttention(BaseModule):
    """Scale-adaptive Self Attention"""
    def __init__(self, 
                 embed_dims=256,
                 num_heads=8,
                 dropout=0.1,
                 pc_range=[],
                 init_cfg=None):
        super().__init__(init_cfg)
        self.pc_range = pc_range

        self.attention = MultiheadAttention(embed_dims, num_heads, dropout, batch_first=True)
        self.gen_tau = nn.Linear(embed_dims, num_heads)

    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.gen_tau.weight)
        nn.init.uniform_(self.gen_tau.bias, 0.0, 2.0)

    def inner_forward(self, query_points, query_feat):
        """
        query_points: [B, Q, 6]
        query_feat: [B, Q, C]
        """
        dist = self.calc_points_dists(query_points)
        tau = self.gen_tau(query_feat)  # [B, Q, 8]

        if DUMP.enabled:
            torch.save(tau.cpu(), '{}/sasa_tau_stage{}.pth'.format(DUMP.out_dir, DUMP.stage_count))

        tau = tau.permute(0, 2, 1)  # [B, 8, Q]
        attn_mask = dist[:, None, :, :] * tau[..., None]  # [B, 8, Q, Q]

        attn_mask = attn_mask.flatten(0, 1)  # [Bx8, Q, Q]
        return self.attention(query_feat, attn_mask=attn_mask)

    def forward(self, query_points, query_feat):
        if self.training and query_feat.requires_grad:
            return cp(self.inner_forward, query_points, query_feat,
                      use_reentrant=False)
        else:
            return self.inner_forward(query_points, query_feat)

    @torch.no_grad()
    def calc_points_dists(self, points):
        points = decode_points(points, self.pc_range)
        dist = torch.norm(points.unsqueeze(-2) - points.unsqueeze(-3), dim=-1)
        return -dist


class PointOccSampling(BaseModule):
    """Adaptive Spatio-temporal Sampling"""
    def __init__(self,
                 embed_dims=256,
                 num_frames=4,
                 num_groups=4,
                 num_points=8,
                 num_levels=4,
                 pc_range=[],
                 init_cfg=None):
        super().__init__(init_cfg)

        self.num_frames = num_frames
        self.num_points = num_points
        self.num_groups = num_groups
        self.num_levels = num_levels
        self.pc_range = pc_range

        self.sampling_offset = nn.Linear(embed_dims, num_groups * num_points * 3)
        self.scale_weights = nn.Linear(embed_dims, num_groups * num_points * num_levels)

    def init_weights(self):
        bias = self.sampling_offset.bias.data.view(self.num_groups * self.num_points, 3)
        nn.init.zeros_(self.sampling_offset.weight)
        nn.init.uniform_(bias[:, 0:3], -0.5, 0.5)

    def inner_forward(self, query_points, query_feat, mlvl_feats, img_metas, scale):
        '''
        query_points: [B, Q, 6]
        query_feat: [B, Q, C]
        '''
        B, Q = query_points.shape[:2]
        image_h, image_w, _ = img_metas[0]['img_shape'][0]

        # sampling offset of all frames
        sampling_offset = scale(self.sampling_offset(query_feat))
        sampling_offset = sampling_offset.view(B, Q, self.num_groups, self.num_points, 3)
        sampling_points = make_sample_points(query_points, sampling_offset, self.pc_range)  # [B, Q, G, P, 3]
        sampling_points = sampling_points.reshape(B, Q, 1, self.num_groups, self.num_points, 3)
        sampling_points = sampling_points.expand(B, Q, self.num_frames, self.num_groups, self.num_points, 3)

        # scale weights
        scale_weights = self.scale_weights(query_feat).view(B, Q, self.num_groups, 1, self.num_points, self.num_levels)
        scale_weights = torch.softmax(scale_weights, dim=-1)
        scale_weights = scale_weights.expand(B, Q, self.num_groups, self.num_frames, self.num_points, self.num_levels)

        # sampling
        sampled_feats = sampling_4d(
            sampling_points,
            mlvl_feats,
            scale_weights,
            img_metas[0]['ego2img'],
            image_h, image_w
        )  # [B, Q, G, FP, C]

        return sampled_feats

    def forward(self, query_points, query_feat, mlvl_feats, img_metas, scale):
        if self.training and query_feat.requires_grad:
            return cp(self.inner_forward, query_points, query_feat, mlvl_feats,
                      img_metas, scale, use_reentrant=False)
        else:
            return self.inner_forward(query_points, query_feat, mlvl_feats, 
                                      img_metas, scale)


class AdaptiveMixing(nn.Module):
    """Adaptive Mixing"""
    def __init__(self, in_dim, in_points, n_groups=1, query_dim=None, out_dim=None, out_points=None):
        super(AdaptiveMixing, self).__init__()

        out_dim = out_dim if out_dim is not None else in_dim
        out_points = out_points if out_points is not None else in_points
        query_dim = query_dim if query_dim is not None else in_dim

        self.query_dim = query_dim
        self.in_dim = in_dim
        self.in_points = in_points
        self.n_groups = n_groups
        self.out_dim = out_dim
        self.out_points = out_points

        self.eff_in_dim = in_dim // n_groups
        self.eff_out_dim = out_dim // n_groups

        self.m_parameters = self.eff_in_dim * self.eff_out_dim
        self.s_parameters = self.in_points * self.out_points
        self.total_parameters = self.m_parameters + self.s_parameters

        self.parameter_generator = nn.Linear(self.query_dim, self.n_groups * self.total_parameters)
        self.out_proj = nn.Linear(self.eff_out_dim * self.out_points * self.n_groups, self.query_dim)
        self.act = nn.ReLU(inplace=True)

    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.parameter_generator.weight)

    def inner_forward(self, x, query):
        B, Q, G, P, C = x.shape
        assert G == self.n_groups
        assert P == self.in_points
        assert C == self.eff_in_dim

        '''generate mixing parameters'''
        params = self.parameter_generator(query)
        params = params.reshape(B*Q, G, -1)
        out = x.reshape(B*Q, G, P, C)

        M, S = params.split([self.m_parameters, self.s_parameters], 2)
        M = M.reshape(B*Q, G, self.eff_in_dim, self.eff_out_dim)
        S = S.reshape(B*Q, G, self.out_points, self.in_points)

        '''adaptive channel mixing'''
        out = torch.matmul(out, M)
        out = F.layer_norm(out, [out.size(-2), out.size(-1)])
        out = self.act(out)

        '''adaptive point mixing'''
        out = torch.matmul(S, out)  # implicitly transpose and matmul
        out = F.layer_norm(out, [out.size(-2), out.size(-1)])
        out = self.act(out)

        '''linear transfomation to query dim'''
        out = out.reshape(B, Q, -1)
        out = self.out_proj(out)
        out = query + out

        return out

    def forward(self, x, query):
        if self.training and x.requires_grad:
            return cp(self.inner_forward, x, query, use_reentrant=False)
        else:
            return self.inner_forward(x, query)
