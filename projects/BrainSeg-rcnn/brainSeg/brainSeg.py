
import torch
import torch.nn.functional as F
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec
from detectron2.modeling.backbone.resnet import BottleneckBlock


class FeatureLearningLevel(torch.nn.Module):
    def __init__(self, in_channels=1, out_channels=64, stride=2) -> None:
        super().__init__()

        self.convs = []
        for i in range(1, 5):
            conv_name = f'conv{i}'
            conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                   kernel_size=3, stride=stride, padding=1)
            self.add_module(conv_name, conv)
            self.convs.append(conv)

    def forward(self, x1, x2, x3, x4):
        inputs = [x1, x2, x3, x4]
        for idx, (conv, x) in enumerate(zip(self.convs, inputs)):
            x = conv(x)
            x = F.relu_(x)
            x = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
            inputs[idx] = x
        x1, x2, x3, x4 = inputs
        del inputs
        return x1, x2, x3, x4


class FeatureLearning(torch.nn.Module):
    def __init__(self, fl_inChannels, fl_outChannels, fl_lateral_channel) -> None:
        super().__init__()
        self.levels = []
        self.out_convs = []
        for idx, (inchannel, outchannel) in enumerate(zip(fl_inChannels, fl_outChannels)):
            level_name = f'level{idx}'
            level = FeatureLearningLevel(in_channels=inchannel, out_channels=outchannel, stride=2)
            self.add_module(level_name, level)
            self.levels.append(level)

            conv_name = f'out_conv{idx}'
            conv = torch.nn.Conv2d(in_channels=outchannel * 4,
                                   out_channels=fl_lateral_channel, kernel_size=3, stride=1, padding=1)
            self.add_module(conv_name, conv)
            self.out_convs.append(conv)

    def forward(self, x1, x2, x3, x4):
        outputs = []
        for idx, (level, conv) in enumerate(zip(self.levels, self.out_convs)):

            x1, x2, x3, x4 = level(x1, x2, x3, x4)

            outputs.append(conv(torch.cat([x1, x2, x3, x4], dim=1)))

            x2, x3, x4 = x1 + x2, x2 + x3, x3 + x4

        return outputs


class UACBlock(torch.nn.Module):
    def __init__(self, inchannel) -> None:
        super().__init__()
        self.in_channels = inchannel

        self.stride = 1
        self.upsample = torch.nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv1 = torch.nn.Conv2d(inchannel, inchannel, kernel_size=1, stride=self.stride)

        curr_kwargs = {}
        curr_kwargs['bottleneck_channels'] = int(inchannel / 2)
        self.bottleneckBlock = BottleneckBlock(inchannel * 2, inchannel, **curr_kwargs)

    def forward(self, cat1, cat2):

        cat1 = self.upsample(cat1)

        cat1 = self.conv1(cat1)

        cat = torch.add(cat1, cat2)

        cat = torch.cat([cat, cat2], dim=1)

        cat = self.bottleneckBlock(cat)

        return cat


class ContextFusion(torch.nn.Module):
    def __init__(self, fl_lateral_channel, nb_levels) -> None:
        super().__init__()
        self.uacBlocks = []
        for idx in range(nb_levels - 1):
            block_name = 'uacBloc' + str(idx + 1)
            block = UACBlock(fl_lateral_channel)
            self.add_module(block_name, block)
            self.uacBlocks.append(block)

    def forward(self, features):
        outputs = {}
        nb_features = len(features)

        prev_feature = f'level{nb_features}'
        outputs[prev_feature] = features[-1]

        for id, (feature, uacBlock) in enumerate(zip(features[::-1][1:], self.uacBlocks[::-1])):

            outputs[f'level{nb_features - id -1 }'] = uacBlock(outputs[prev_feature], feature)
            prev_feature = f'level{nb_features - id -1 }'

        return outputs


@BACKBONE_REGISTRY.register()
class BrainSegBackbone(Backbone):
    def __init__(self, cfg, input_shape):
        super().__init__()
        self.fl_inChannels = [1, 32, 64, 128]
        self.fl_outChannels = [32, 64, 128, 256]
        self.fl_lateral_channel = 64
        self.nb_levels = len(self.fl_inChannels)

        self.featureLearning = FeatureLearning(
            self.fl_inChannels, self.fl_outChannels, self.fl_lateral_channel)
        self.contextFusion = ContextFusion(self.fl_lateral_channel, self.nb_levels)

    def forward(self, image):

        x1, x2, x3, x4 = torch.split(image, 1, dim=1)
        features = self.featureLearning(x1, x2, x3, x4)
        out = self.contextFusion(features)
        return out

    def output_shape(self):
        return {f'level{i}': ShapeSpec(channels=64, stride=2**i) for i in range(1, 5)}
