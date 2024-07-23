# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import ConvModule, build_conv_layer, build_norm_layer
from mmengine.model import BaseModule, constant_init
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm

from mmpose.registry import MODELS
from .base_backbone import BaseBackbone

import torch
from torch.nn import functional as F

class BasicBlock(BaseModule):
    """BasicBlock for ResNet.

    Args:
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        expansion (int): The ratio of ``out_channels/mid_channels`` where
            ``mid_channels`` is the output channels of conv1. This is a
            reserved argument in BasicBlock and should always be 1. Default: 1.
        stride (int): stride of the block. Default: 1
        dilation (int): dilation of convolution. Default: 1
        downsample (nn.Module): downsample operation on identity branch.
            Default: None.
        style (str): `pytorch` or `caffe`. It is unused and reserved for
            unified API with Bottleneck.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=1,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 init_cfg=None):
        # Protect mutable default arguments
        norm_cfg = copy.deepcopy(norm_cfg)
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        assert self.expansion == 1
        assert out_channels % expansion == 0
        self.mid_channels = out_channels // expansion
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, self.mid_channels, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, out_channels, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg,
            in_channels,
            self.mid_channels,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg,
            self.mid_channels,
            out_channels,
            3,
            padding=1,
            bias=False)
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: the normalization layer named "norm2" """
        return getattr(self, self.norm2_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


class Bottleneck(BaseModule):
    """Bottleneck block for ResNet.

    Args:
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        expansion (int): The ratio of ``out_channels/mid_channels`` where
            ``mid_channels`` is the input/output channels of conv2. Default: 4.
        stride (int): stride of the block. Default: 1
        dilation (int): dilation of convolution. Default: 1
        downsample (nn.Module): downsample operation on identity branch.
            Default: None.
        style (str): ``"pytorch"`` or ``"caffe"``. If set to "pytorch", the
            stride-two layer is the 3x3 conv layer, otherwise the stride-two
            layer is the first 1x1 conv layer. Default: "pytorch".
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=4,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 init_cfg=None):
        # Protect mutable default arguments
        norm_cfg = copy.deepcopy(norm_cfg)
        super().__init__(init_cfg=init_cfg)
        assert style in ['pytorch', 'caffe']

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        assert out_channels % expansion == 0
        self.mid_channels = out_channels // expansion
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, self.mid_channels, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, self.mid_channels, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, out_channels, postfix=3)

        self.conv1 = build_conv_layer(
            conv_cfg,
            in_channels,
            self.mid_channels,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg,
            self.mid_channels,
            self.mid_channels,
            kernel_size=3,
            stride=self.conv2_stride,
            padding=dilation,
            dilation=dilation,
            bias=False)

        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            conv_cfg,
            self.mid_channels,
            out_channels,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: the normalization layer named "norm2" """
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        """nn.Module: the normalization layer named "norm3" """
        return getattr(self, self.norm3_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


def get_expansion(block, expansion=None):
    """Get the expansion of a residual block.

    The block expansion will be obtained by the following order:

    1. If ``expansion`` is given, just return it.
    2. If ``block`` has the attribute ``expansion``, then return
       ``block.expansion``.
    3. Return the default value according the the block type:
       1 for ``BasicBlock`` and 4 for ``Bottleneck``.

    Args:
        block (class): The block class.
        expansion (int | None): The given expansion ratio.

    Returns:
        int: The expansion of the block.
    """
    if isinstance(expansion, int):
        assert expansion > 0
    elif expansion is None:
        if hasattr(block, 'expansion'):
            expansion = block.expansion
        elif issubclass(block, BasicBlock):
            expansion = 1
        elif issubclass(block, Bottleneck):
            expansion = 4
        else:
            raise TypeError(f'expansion is not specified for {block.__name__}')
    else:
        raise TypeError('expansion must be an integer or None')

    return expansion


class ResLayer(nn.Sequential):
    """ResLayer to build ResNet style backbone.

    Args:
        block (nn.Module): Residual block used to build ResLayer.
        num_blocks (int): Number of blocks.
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        expansion (int, optional): The expansion for BasicBlock/Bottleneck.
            If not specified, it will firstly be obtained via
            ``block.expansion``. If the block has no attribute "expansion",
            the following default values will be used: 1 for BasicBlock and
            4 for Bottleneck. Default: None.
        stride (int): stride of the first block. Default: 1.
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        downsample_first (bool): Downsample at the first block or last block.
            False for Hourglass, True for ResNet. Default: True
    """

    def __init__(self,
                 block,
                 num_blocks,
                 in_channels,
                 out_channels,
                 expansion=None,
                 stride=1,
                 avg_down=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 downsample_first=True,
                 **kwargs):
        # Protect mutable default arguments
        norm_cfg = copy.deepcopy(norm_cfg)
        self.block = block
        self.expansion = get_expansion(block, expansion)

        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = []
            conv_stride = stride
            if avg_down and stride != 1:
                conv_stride = 1
                downsample.append(
                    nn.AvgPool2d(
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True,
                        count_include_pad=False))
            downsample.extend([
                build_conv_layer(
                    conv_cfg,
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=conv_stride,
                    bias=False),
                build_norm_layer(norm_cfg, out_channels)[1]
            ])
            downsample = nn.Sequential(*downsample)

        layers = []
        if downsample_first:
            layers.append(
                block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    expansion=self.expansion,
                    stride=stride,
                    downsample=downsample,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs))
            in_channels = out_channels
            for _ in range(1, num_blocks):
                layers.append(
                    block(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        expansion=self.expansion,
                        stride=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        **kwargs))
        else:  # downsample_first=False is for HourglassModule
            for i in range(0, num_blocks - 1):
                layers.append(
                    block(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        expansion=self.expansion,
                        stride=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        **kwargs))
            layers.append(
                block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    expansion=self.expansion,
                    stride=stride,
                    downsample=downsample,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs))

        super().__init__(*layers)


@MODELS.register_module()
class ResNetFuseFrequencyDecouplingDESEMixBigff(BaseBackbone):
    """ResNet backbone.

    Please refer to the `paper <https://arxiv.org/abs/1512.03385>`__ for
    details.

    Args:
        depth (int): Network depth, from {18, 34, 50, 101, 152}.
        in_channels (int): Number of input image channels. Default: 3.
        stem_channels (int): Output channels of the stem layer. Default: 64.
        base_channels (int): Middle channels of the first stage. Default: 64.
        num_stages (int): Stages of the network. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
            Default: ``(1, 2, 2, 2)``.
        dilations (Sequence[int]): Dilation of each stage.
            Default: ``(1, 1, 1, 1)``.
        out_indices (Sequence[int]): Output from which stages. If only one
            stage is specified, a single tensor (feature map) is returned,
            otherwise multiple stages are specified, a tuple of tensors will
            be returned. Default: ``(3, )``.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv.
            Default: False.
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Default: -1.
        conv_cfg (dict | None): The config dict for conv layers. Default: None.
        norm_cfg (dict): The config dict for norm layers.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity. Default: True.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default:
            ``[
                dict(type='Kaiming', layer=['Conv2d']),
                dict(
                    type='Constant',
                    val=1,
                    layer=['_BatchNorm', 'GroupNorm'])
            ]``

    Example:
        >>> from mmpose.models import ResNet
        >>> import torch
        >>> self = ResNet(depth=18, out_indices=(0, 1, 2, 3))
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 64, 8, 8)
        (1, 128, 4, 4)
        (1, 256, 2, 2)
        (1, 512, 1, 1)
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 in_channels=3,
                 stem_channels=64,
                 base_channels=64,
                 expansion=None,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(3, ),
                 style='pytorch',
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=False,
                 with_cp=False,
                 zero_init_residual=True,
                 init_cfg=[
                     dict(type='Kaiming', layer=['Conv2d']),
                     dict(
                         type='Constant',
                         val=1,
                         layer=['_BatchNorm', 'GroupNorm'])
                 ]):
        # Protect mutable default arguments
        norm_cfg = copy.deepcopy(norm_cfg)
        super(ResNetFuseFrequencyDecouplingDESEMixBigff, self).__init__(init_cfg)
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        self.depth = depth
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert 1 <= num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.zero_init_residual = zero_init_residual
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.expansion = get_expansion(self.block, expansion)

        self._make_stem_layer(in_channels, stem_channels)
        self.extendlayer = ExtendLayer().cuda()
        self.res_layers = []
        _in_channels = stem_channels
        _out_channels = base_channels * self.expansion
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            res_layer = self.make_res_layer(
                block=self.block,
                num_blocks=num_blocks,
                in_channels=_in_channels,
                out_channels=_out_channels,
                expansion=self.expansion,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg)
            _in_channels = _out_channels
            _out_channels *= 2
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

        self.feat_dim = res_layer[-1].out_channels

    def make_res_layer(self, **kwargs):
        """Make a ResLayer."""
        return ResLayer(**kwargs)

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels, stem_channels):
        """Make stem layer."""
        if self.deep_stem:
            self.stem = nn.Sequential(
                ConvModule(
                    in_channels,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=True),
                ConvModule(
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=True),
                ConvModule(
                    stem_channels // 2,
                    stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=True))
        else:
            self.conv1 = build_conv_layer(
                self.conv_cfg,
                in_channels,
                stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False)
            self.norm1_name, norm1 = build_norm_layer(
                self.norm_cfg, stem_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)
            self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        """Freeze parameters."""
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self):
        """Initialize the weights in backbone."""
        super(ResNetFuseFrequencyDecouplingDESEMixBigff, self).init_weights()

        if (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            # Suppress zero_init_residual if use pretrained model.
            return

        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    constant_init(m.norm3, 0)
                elif isinstance(m, BasicBlock):
                    constant_init(m.norm2, 0)

    def forward(self, x):
        """Forward function."""
        
        x = self.extendlayer(x)
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode."""
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()


@MODELS.register_module()
class ResNetV1dFuseFrequencyDecouplingDESEMixBigff(ResNetFuseFrequencyDecouplingDESEMixBigff):
    r"""ResNetV1d variant described in `Bag of Tricks
    <https://arxiv.org/pdf/1812.01187.pdf>`__.

    Compared with default ResNet(ResNetV1b), ResNetV1d replaces the 7x7 conv in
    the input stem with three 3x3 convs. And in the downsampling block, a 2x2
    avg_pool with stride 2 is added before conv, whose stride is changed to 1.
    """

    def __init__(self, **kwargs):
        super().__init__(deep_stem=True, avg_down=True, **kwargs)

class ExtendLayer(nn.Module):
    def __init__(self):
        #第一层的卷积层  逐层通过maxpool下采样
        super().__init__()
        self.conv1_1_1 = ConvModule(in_channels=3,out_channels=8,kernel_size=3,padding=1)   #(256,192)
        self.conv1_1_2 = ConvModule(in_channels=8,out_channels=8,kernel_size=3,padding=1)
    
        self.maxpool1_1 = nn.MaxPool2d(kernel_size=2,stride=2)                              #(128,96)
        self.conv1_2_1 = ConvModule(in_channels=8,out_channels=16,kernel_size=3,padding=1)
        self.conv1_2_2 = ConvModule(in_channels=16,out_channels=16,kernel_size=3,padding=1)

        self.maxpool1_2 = nn.MaxPool2d(kernel_size=2,stride=2)                              #(64,48)
        self.conv1_3_1 = ConvModule(in_channels=16,out_channels=32,kernel_size=3,padding=1)
        self.conv1_3_2 = ConvModule(in_channels=32,out_channels=32,kernel_size=3,padding=1)
        self.conv1_3_3 = ConvModule(in_channels=32,out_channels=32,kernel_size=3,padding=1)

        self.maxpool1_3 = nn.MaxPool2d(kernel_size=2,stride=2)                              #(32,24)
        self.conv1_4_1 = ConvModule(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        self.conv1_4_2 = ConvModule(in_channels=64,out_channels=64,kernel_size=3,padding=1)
        self.conv1_4_3 = ConvModule(in_channels=64,out_channels=64,kernel_size=3,padding=1)

        # 全局特征的获取
        self.glob_conv1 = ConvModule(in_channels=64,out_channels=16,kernel_size=5,padding=2)
        self.glob_conv2 = ConvModule(in_channels=16,out_channels=16,kernel_size=5,padding=2)
        self.glob_conv3 = ConvModule(in_channels=16,out_channels=16,kernel_size=3,padding=1,act_cfg=None)

        # 第二层的卷积 将特征通道统一到16
        self.conv2_1 = ConvModule(in_channels=8,out_channels=16,kernel_size=3,padding=1)
        self.conv2_2 = ConvModule(in_channels=16,out_channels=16,kernel_size=3,padding=1)
        self.conv2_3 = ConvModule(in_channels=32,out_channels=16,kernel_size=3,padding=1)
        self.conv2_4 = ConvModule(in_channels=64,out_channels=16,kernel_size=3,padding=1)

        #第四层AFFormer 高低通滤波器
        self.LHFrequency1 = FrequencyModule()
        self.LHFrequency2 = FrequencyModule()
        self.LHFrequency3 = FrequencyModule()
        self.LHFrequency4 = FrequencyModule()

        #第五层 上采样合并
        self.deconv4 = nn.ConvTranspose2d(in_channels=16,out_channels=16,kernel_size=4,stride=2,padding=1)
        self.deconv3 = nn.ConvTranspose2d(in_channels=32,out_channels=32,kernel_size=4,stride=2,padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=48,out_channels=48,kernel_size=4,stride=2,padding=1)
        self.loaclconv = ConvModule(in_channels=64,out_channels=64,kernel_size=1,act_cfg=None)
        self.Lendconv = ConvModule(in_channels=64,out_channels=3,kernel_size=1,act_cfg=None)
        self.Gendconv = ConvModule(in_channels=16,out_channels=3,kernel_size=1,act_cfg=None)    #先插值再卷积



    def forward(self,x):
        fea_lay1_1 = self.conv1_1_2(self.conv1_1_1(x))
        fea_lay1_2 = self.conv1_2_2(self.conv1_2_1(self.maxpool1_1(fea_lay1_1)))
        fea_lay1_3 = self.conv1_3_3(self.conv1_3_2(self.conv1_3_1(self.maxpool1_2(fea_lay1_2))))
        fea_lay1_4 = self.conv1_4_3(self.conv1_4_2(self.conv1_4_1(self.maxpool1_3(fea_lay1_3))))

        fea_glob = self.glob_conv3(self.glob_conv2(self.glob_conv1(fea_lay1_4)))

        fea_lay2_1 = self.conv2_1(fea_lay1_1)
        fea_lay2_2 = self.conv2_2(fea_lay1_2)
        fea_lay2_3 = self.conv2_3(fea_lay1_3)
        fea_lay2_4 = self.conv2_4(fea_lay1_4)

        fea_lay4_1 = self.LHFrequency1(fea_lay2_1)
        fea_lay4_2 = self.LHFrequency2(fea_lay2_2)
        fea_lay4_3 = self.LHFrequency3(fea_lay2_3)
        fea_lay4_4 = self.LHFrequency4(fea_lay2_4)

        fea_U4 = self.deconv4(fea_lay4_4)
        fea_U3 = self.deconv3(torch.concat([fea_U4,fea_lay4_3],1))
        fea_U2 = self.deconv2(torch.concat([fea_U3,fea_lay4_2],1))
        fea_L1 = self.Lendconv(self.loaclconv(torch.concat([fea_U2,fea_lay4_1],1)))
        fea_G = self.Gendconv(F.interpolate(input=fea_glob,size=(256,192),mode='bilinear'))

        fea = fea_L1 + fea_G

        return fea

class FrequencyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = nn.AvgPool2d(2,2)
        self.conv1 = ConvModule(in_channels=16,out_channels=16,kernel_size=1,dilation=1)
        self.conv2 = ConvModule(in_channels=16,out_channels=16,kernel_size=3,dilation=2,padding=2)
        self.sigmoid = nn.Sigmoid()
        self.dem = DEModule(16)
        self.SE = ChannelAttention()
        self.feamix = BiGFF(16,16)
    
    def window_partition(self, x, window_size):
        """
        Args:
            x: (B, H, W, C)
            window_size (int): window size

        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        windows = windows.view(-1, window_size * window_size, C)  
        return windows

    def window_reverse(self, windows, window_size, H, W):
        """
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            window_size (int): Window size
            H (int): Height of image
            W (int): Width of image

        Returns:
            x: (B, H, W, C)
        """
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x


    def forward(self,x):
        b, c, h, w = x.size(0), x.size(1), x.size(2), x.size(3)
        fea1 = self.conv1(x)
        fea2 = self.conv2(x)
        ca = self.sigmoid(fea1-fea2)
        feaLF = (1-ca)*x

        y = self.avgpool(x)
        feaLF_nor = F.interpolate(y, size=(h,w), mode='nearest')
        feaHF = x-feaLF_nor

        feaHF = feaHF.flatten(2).transpose(1, 2)    #(B,HW,C)
        feaHF = feaHF.view(b, h, w, c)    #(B,H,W,C)
        feaHF_windows = self.window_partition(feaHF,8)    #(B*HW/64,64,C)
        feaHF_enhanced = self.SE(feaHF_windows)
        feaHF_enhanced = self.window_reverse(feaHF_enhanced,8,h,w)
        feaHF_enhanced = feaHF_enhanced.permute(0,3,1,2)

        feaLF_enhanced = self.dem(feaLF)
        fuseFea = self.feamix(feaLF_enhanced,feaHF_enhanced)
        return fuseFea

class ConvBNReLU(nn.Module):
    '''Module for the Conv-BN-ReLU tuple.'''

    def __init__(self, c_in, c_out, kernel_size, stride, padding, dilation=1, group=1,
                 has_bn=True, has_relu=True, mode='2d'):
        super(ConvBNReLU, self).__init__()
        self.has_bn = has_bn
        self.has_relu = has_relu
        if mode == '2d':
            self.conv = nn.Conv2d(
                c_in, c_out, kernel_size=kernel_size, stride=stride,
                padding=padding, dilation=dilation, bias=False, groups=group)
            norm_layer = nn.BatchNorm2d
        elif mode == '1d':
            self.conv = nn.Conv1d(
                c_in, c_out, kernel_size=kernel_size, stride=stride,
                padding=padding, dilation=dilation, bias=False, groups=group)
            norm_layer = nn.BatchNorm1d
        if self.has_bn:
            self.bn = norm_layer(c_out)
        if self.has_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)
        return x

class QCO_1d(nn.Module):
    def __init__(self, level_num):
        super(QCO_1d, self).__init__()
        self.conv1 = nn.Sequential(ConvBNReLU(32, 32, 3, 1, 1, has_relu=False), nn.LeakyReLU(inplace=True))
        self.conv2 = ConvBNReLU(32, 16, 1, 1, 0, has_bn=False, has_relu=False)
        self.f1 = nn.Sequential(ConvBNReLU(2, 8, 1, 1, 0, has_bn=False, has_relu=False, mode='1d'),
                                nn.LeakyReLU(inplace=True))
        self.f2 = ConvBNReLU(8, 16, 1, 1, 0, has_bn=False, mode='1d')
        self.out = ConvBNReLU(32, 16, 1, 1, 0, has_bn=True, mode='1d')
        self.level_num = level_num

    def forward(self, x):   #(B,64,H,W)
        x = self.conv1(x)   #(B,64,H,W)
        x = self.conv2(x)   #(B,32,H,W)
        N, C, H, W = x.shape
        x_ave = F.adaptive_avg_pool2d(x, (1, 1))    #(B,32,1,1)
        cos_sim = (F.normalize(x_ave, dim=1) * F.normalize(x, dim=1)).sum(1)    #(B,H,W)
        cos_sim = cos_sim.view(N, -1)   #(B,H*W)
        cos_sim_min, _ = cos_sim.min(-1)    #(B)
        cos_sim_min = cos_sim_min.unsqueeze(-1) #(B,1)
        cos_sim_max, _ = cos_sim.max(-1)    #(B)
        cos_sim_max = cos_sim_max.unsqueeze(-1) #(B,1)
        q_levels = torch.arange(self.level_num).float().cuda()  #(32) input-level_num
        q_levels = q_levels.expand(N, self.level_num)   #(B,32)
        q_levels = (2 * q_levels + 1) / (2 * self.level_num) * (cos_sim_max - cos_sim_min) + cos_sim_min    #(B,32)
        q_levels = q_levels.unsqueeze(1)    #(B,1,32)
        q_levels_inter = q_levels[:, :, 1] - q_levels[:, :, 0]  #(B,1)
        q_levels_inter = q_levels_inter.unsqueeze(-1)   #(B,1,1)
        cos_sim = cos_sim.unsqueeze(-1) #(B,H*W,1)
        quant = 1 - torch.abs(q_levels - cos_sim)   #(B,H*W,32)
        quant = quant * (quant > (1 - q_levels_inter))  #(B,H*W,32)
        sta = quant.sum(1)  #(B,32)
        sta = sta / (sta.sum(-1).unsqueeze(-1)) #(B,32)
        sta = sta.unsqueeze(1)  #(B,1,32)
        sta = torch.cat([q_levels, sta], dim=1) #(B,2,32)
        sta = self.f1(sta)  #(B,16,32)
        sta = self.f2(sta)  #(B,32,32)
        x_ave = x_ave.squeeze(-1).squeeze(-1)   #(B,32)
        x_ave = x_ave.expand(self.level_num, N, C).permute(1, 2, 0) #(B,32,32)
        sta = torch.cat([sta, x_ave], dim=1)    #(B,64,32)
        sta = self.out(sta) #(B,32,32)
        return sta, quant

class TEM(nn.Module):
    def __init__(self, level_num):
        super(TEM, self).__init__()
        self.level_num = level_num
        self.qco = QCO_1d(level_num)
        self.k = ConvBNReLU(16, 16, 1, 1, 0, has_bn=False, has_relu=False, mode='1d')
        self.q = ConvBNReLU(16, 16, 1, 1, 0, has_bn=False, has_relu=False, mode='1d')
        self.v = ConvBNReLU(16, 16, 1, 1, 0, has_bn=False, has_relu=False, mode='1d')
        self.out = ConvBNReLU(16, 32, 1, 1, 0, mode='1d')

    def forward(self, x):   #(B,32,H,W)
        N, C, H, W = x.shape
        sta, quant = self.qco(x)    #(B,32,32)  (B,H*W,32)
        k = self.k(sta) #(B,32,32)
        q = self.q(sta) #(B,32,32)
        v = self.v(sta) #(B,32,32)
        k = k.permute(0, 2, 1)  #(B,32,32)
        w = torch.bmm(k, q) #(B,32,32)
        w = F.softmax(w, dim=-1)    #(B,32,32)
        v = v.permute(0, 2, 1)  #(B,32,32)
        f = torch.bmm(w, v) #(B,32,32)
        f = f.permute(0, 2, 1)  #(B,32,32)
        f = self.out(f) #(B,64,32)
        quant = quant.permute(0, 2, 1)  #(B,32,H*W)
        out = torch.bmm(f, quant)   #(B,64,H*W)
        out = out.view(N, 32, H, W)    #(B,64,H,W)
        return out

class DEModule(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.conv_start = ConvBNReLU(in_channel, 32, 1, 1, 0)      #(in_channel, 32, 1, 1, 0)
        self.tem = TEM(16)                                         #(128)
        self.conv_end = ConvBNReLU(64, 16, 1, 1, 0)              #(512, 192, 1, 1, 0)

    def forward(self, x):       #(B,C,H,W)             (B,C,H,W)
        x = self.conv_start(x)  #(B,64,H,W)
        x_tem = self.tem(x)     #(B,64,H,W)
        x = torch.cat([x_tem, x], dim=1)    #(B,128,H,W)
        x = self.conv_end(x)    #(B,16,H,W)            (B,C,H,W)
        return x

class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat=16, squeeze_factor=4,memory_blocks=128):
        super(ChannelAttention, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.subnet = nn.Sequential(
            
            nn.Linear(num_feat, num_feat // squeeze_factor),
            #nn.ReLU(inplace=True)
           )
        self.upnet= nn.Sequential(
            nn.Linear(num_feat // squeeze_factor, num_feat),
            nn.Linear(num_feat, num_feat),
            nn.Sigmoid())
        self.mb =  torch.nn.Parameter(torch.randn(num_feat // squeeze_factor, memory_blocks))
        self.low_dim = num_feat // squeeze_factor

    def forward(self, x):
        b,n,c = x.shape
        t = x.transpose(1,2)
        y = self.pool(t).squeeze(-1)

        low_rank_f = self.subnet(y).unsqueeze(2)

        mbg = self.mb.unsqueeze(0).repeat(b, 1, 1)
        f1 = (low_rank_f.transpose(1,2) ) @mbg  
        f_dic_c = F.softmax(f1 * (int(self.low_dim) ** (-0.5)), dim=-1) # get the similarity information
        y1 = f_dic_c@mbg.transpose(1,2) 
        y2 = self.upnet(y1)
        out = x*y2
        return out
    
class BiGFF(nn.Module):
    '''Bi-directional Gated Feature Fusion.'''
    
    def __init__(self, in_channels, out_channels):
        super(BiGFF, self).__init__()

        self.structure_gate = nn.Sequential(
            nn.Conv2d(in_channels=in_channels + in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.texture_gate = nn.Sequential(
            nn.Conv2d(in_channels=in_channels + in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.structure_gamma = nn.Parameter(torch.zeros(1))
        self.texture_gamma = nn.Parameter(torch.zeros(1))
        self.feature_zip = nn.Sequential(
            nn.Conv2d(32,16,1),
            nn.ReLU()
        )

    def forward(self, texture_feature, structure_feature):

        energy = torch.cat((texture_feature, structure_feature), dim=1)

        gate_structure_to_texture = self.structure_gate(energy)
        gate_texture_to_structure = self.texture_gate(energy)

        texture_feature = texture_feature + self.texture_gamma * (gate_structure_to_texture * structure_feature)
        structure_feature = structure_feature + self.structure_gamma * (gate_texture_to_structure * texture_feature)

        return self.feature_zip(torch.cat((texture_feature, structure_feature), dim=1))