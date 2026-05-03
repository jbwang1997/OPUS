from mmcv.cnn.bricks.registry import CONV_LAYERS
from spconv.pytorch import (SparseConv2d, SparseConv3d, SparseConv4d,
                            SparseConvTranspose2d,
                            SparseConvTranspose3d, SparseInverseConv2d,
                            SparseInverseConv3d, SparseModule,
                            SubMConv2d, SubMConv3d, SubMConv4d)

CONV_LAYERS._register_module(SparseConv2d, 'SparseConv2d', force=True)
CONV_LAYERS._register_module(SparseConv3d, 'SparseConv3d', force=True)
CONV_LAYERS._register_module(SparseConv4d, 'SparseConv4d', force=True)

CONV_LAYERS._register_module(
    SparseConvTranspose2d, 'SparseConvTranspose2d', force=True)
CONV_LAYERS._register_module(
    SparseConvTranspose3d, 'SparseConvTranspose3d', force=True)

CONV_LAYERS._register_module(
    SparseInverseConv2d, 'SparseInverseConv2d', force=True)
CONV_LAYERS._register_module(
    SparseInverseConv3d, 'SparseInverseConv3d', force=True)

CONV_LAYERS._register_module(SubMConv2d, 'SubMConv2d', force=True)
CONV_LAYERS._register_module(SubMConv3d, 'SubMConv3d', force=True)
CONV_LAYERS._register_module(SubMConv4d, 'SubMConv4d', force=True)

from .second_fpn import SECONDFPN
from .second import SECOND
from .sparse_block import SparseBottleneck, SparseBasicBlock, make_sparse_convmodule
from .sparse_encoder import SparseEncoder
from .voxel_encoder import HardSimpleVFE, DynamicSimpleVFE, DynamicVFE
