import torch
import torch.nn.functional as F
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec
from detectron2.modeling.backbone.resnet import BottleneckBlock
from detectron2.layers import get_norm, Conv2d


class FeatureLearningLevel(torch.nn.Module):
    def __init__(self, in_channels=1, out_channels=64, stride=2, norm="BN", fl_lateral_channel=64) -> None:
        super(FeatureLearningLevel, self).__init__()
        self.conv1 = Conv2d(in_channels=in_channels, out_channels=out_channels,
                            kernel_size=3, stride=stride, padding=1, norm=get_norm(norm, out_channels))
        self.conv2 = Conv2d(in_channels=in_channels, out_channels=out_channels,
                            kernel_size=3, stride=stride, padding=1, norm=get_norm(norm, out_channels))
        self.conv3 = Conv2d(in_channels=in_channels, out_channels=out_channels,
                            kernel_size=3, stride=stride, padding=1, norm=get_norm(norm, out_channels))
        self.conv4 = Conv2d(in_channels=in_channels, out_channels=out_channels,
                            kernel_size=3, stride=stride, padding=1, norm=get_norm(norm, out_channels))
        curr_kwargs = {}
        curr_kwargs["bottleneck_channels"] = int((out_channels * 4) / 2)
        self.conv_out = BottleneckBlock(
            out_channels * 4, fl_lateral_channel, **curr_kwargs, norm=norm
        )

    def forward(self, x1, x2, x3, x4):
        x1 = F.max_pool2d(F.relu(self.conv1(x1)), kernel_size=3, stride=1, padding=1)
        x2 = F.max_pool2d(F.relu(self.conv1(x2)), kernel_size=3, stride=1, padding=1)
        x3 = F.max_pool2d(F.relu(self.conv1(x3)), kernel_size=3, stride=1, padding=1)
        x4 = F.max_pool2d(F.relu(self.conv1(x4)), kernel_size=3, stride=1, padding=1)
        cat = self.conv_out(torch.cat([x1, x2, x3, x4], dim=1))
        x2, x3, x4 = x1 + x2, x2 + x3, x3 + x4

        return x1, x2, x3, x4, cat


class FeatureLearning(torch.nn.Module):
    def __init__(self, fl_inChannels, fl_outChannels, fl_lateral_channel, norm) -> None:
        super(FeatureLearning, self).__init__()
        self.levels = torch.nn.ModuleList()

        for idx, (inchannel, outchannel) in enumerate(
            zip(fl_inChannels, fl_outChannels)
        ):

            level = FeatureLearningLevel(
                in_channels=inchannel, out_channels=outchannel, stride=2, norm=norm, fl_lateral_channel=fl_lateral_channel
            )

            self.levels.append(level)

    def forward(self, x1, x2, x3, x4):
        outputs = []
        for level in self.levels:

            x1, x2, x3, x4, cat = level(x1, x2, x3, x4)
            outputs.append(cat)

        return outputs


class UACBlock(torch.nn.Module):
    def __init__(self, inchannel, norm) -> None:
        super(UACBlock, self).__init__()
        self.in_channels = inchannel

        self.stride = 1
        self.upsample = torch.nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv1 = Conv2d(
            inchannel,
            inchannel,
            kernel_size=1,
            stride=self.stride,
            norm=get_norm(norm, inchannel),
        )

        curr_kwargs = {}
        curr_kwargs["bottleneck_channels"] = int(inchannel / 2)
        self.bottleneckBlock = BottleneckBlock(
            inchannel * 2, inchannel, **curr_kwargs, norm=norm
        )

    def forward(self, cat1, cat2):

        cat1 = self.upsample(cat1)

        cat1 = self.conv1(cat1)

        cat1 = F.relu_(cat1)

        cat = torch.add(cat1, cat2)

        cat = torch.cat([cat, cat2], dim=1)

        cat = self.bottleneckBlock(cat)

        return cat


class ContextFusion(torch.nn.Module):
    def __init__(self, fl_lateral_channel, nb_levels, norm) -> None:
        super(ContextFusion, self).__init__()
        self.uacBlocks = []
        for idx in range(nb_levels - 1):
            block_name = "uacBloc" + str(idx + 1)
            block = UACBlock(fl_lateral_channel, norm)
            self.add_module(block_name, block)
            self.uacBlocks.append(block)

    def forward(self, features):
        outputs = {}
        nb_features = len(features)

        prev_feature = f"level{nb_features}"
        outputs[prev_feature] = features[-1]

        for id, (feature, uacBlock) in enumerate(
            zip(features[::-1][1:], self.uacBlocks[::-1])
        ):

            outputs[f"level{nb_features - id -1 }"] = uacBlock(
                outputs[prev_feature], feature
            )
            prev_feature = f"level{nb_features - id -1 }"

        return outputs


@BACKBONE_REGISTRY.register()
class BrainSegBackbone(Backbone):
    def __init__(self, cfg, input_shape):
        super(BrainSegBackbone, self).__init__()
        self.fl_inChannels = [1, 32, 64, 128]
        self.fl_outChannels = [32, 64, 128, 256]
        self.fl_lateral_channel = 64
        self.nb_levels = len(self.fl_inChannels)

        self.featureLearning = FeatureLearning(
            self.fl_inChannels,
            self.fl_outChannels,
            self.fl_lateral_channel,
            cfg.MODEL.BRAINSEG.norm,
        )
        self.contextFusion = ContextFusion(
            self.fl_lateral_channel, self.nb_levels, cfg.MODEL.BRAINSEG.norm
        )

    def forward(self, image):

        x1, x2, x3, x4 = torch.split(image, 1, dim=1)
        features = self.featureLearning(x1, x2, x3, x4)
        out = self.contextFusion(features)
        return out

    def output_shape(self):
        return {f"level{i}": ShapeSpec(channels=64, stride=2**i) for i in range(1, 5)}
