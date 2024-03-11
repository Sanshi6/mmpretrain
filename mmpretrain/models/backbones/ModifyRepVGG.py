import torch
import torch.nn as nn
import numpy as np
import torch.utils.checkpoint as checkpoint
from torchsummary import summary

from mmpretrain.registry import MODELS
# from .base_backbone import BaseBackbone
from mmpretrain.models.backbones.base_backbone import BaseBackbone


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                                        bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result


class RepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False,
                 last_relu=True):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.kernel_size = kernel_size

        if self.kernel_size != 1:
            self.crop = int((self.kernel_size - 1) / 2)
        else:
            self.crop = None

        assert kernel_size == 3
        assert padding == 0

        if last_relu:
            self.nonlinearity = nn.ReLU()
        else:
            self.nonlinearity = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True,
                                         padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.BatchNorm2d(
                num_features=in_channels) if out_channels == in_channels and stride == 1 else None

            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, groups=groups)

            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                   padding=0, groups=groups)

            # print('RepVGG Block, identity = ', self.rbr_identity)

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.rbr_reparam(inputs))

        if self.crop is not None:
            t = inputs[:, :, self.crop:-self.crop, self.crop:-self.crop].contiguous()
        else:
            t = inputs

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(t)

        if self.kernel_size != 1:
            rbr_out = self.rbr_1x1(t)
        else:
            rbr_out = 0

        return self.nonlinearity(self.rbr_dense(inputs) + rbr_out + id_out)

    def get_custom_L2(self):
        K3 = self.rbr_dense.conv.weight
        K1 = self.rbr_1x1.conv.weight
        t3 = (self.rbr_dense.bn.weight / ((self.rbr_dense.bn.running_var + self.rbr_dense.bn.eps).sqrt())).reshape(-1,
                                                                                                                   1, 1,
                                                                                                                   1).detach()
        t1 = (self.rbr_1x1.bn.weight / ((self.rbr_1x1.bn.running_var + self.rbr_1x1.bn.eps).sqrt())).reshape(-1, 1, 1,
                                                                                                             1).detach()

        l2_loss_circle = (K3 ** 2).sum() - (K3[:, :, 1:2,
                                            1:2] ** 2).sum()  # The L2 loss of the "circle" of weights in 3x3 kernel. Use regular L2 on them.
        eq_kernel = K3[:, :, 1:2, 1:2] * t3 + K1 * t1  # The equivalent resultant central point of 3x3 kernel.
        l2_loss_eq_kernel = (eq_kernel ** 2 / (
                t3 ** 2 + t1 ** 2)).sum()  # Normalize for an L2 coefficient comparable to regular L2.
        return l2_loss_eq_kernel + l2_loss_circle

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels,
                                     out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation,
                                     groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True


class RepVGG(BaseBackbone):

    def __init__(self, num_blocks, num_classes=-1, width_multiplier=None, override_groups_map=None, deploy=False,
                 use_se=False, use_checkpoint=False):
        super(RepVGG, self).__init__()
        self.num_classes = num_classes
        assert len(width_multiplier) == 4
        self.deploy = deploy
        self.override_groups_map = override_groups_map or dict()
        assert 0 not in self.override_groups_map
        self.use_se = use_se
        self.use_checkpoint = use_checkpoint

        self.in_planes = min(64, int(64 * width_multiplier[0]))
        self.stage0 = RepVGGBlock(in_channels=3, out_channels=self.in_planes, kernel_size=3, stride=2, padding=0,
                                  deploy=self.deploy, use_se=self.use_se)
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(int(64 * width_multiplier[0]), num_blocks[0], stride=2)
        self.stage2 = self._make_stage(int(128 * width_multiplier[1]), num_blocks[1], stride=1)
        self.stage3 = self._make_stage(int(256 * width_multiplier[2]), num_blocks[2], stride=2, last_relu=False)
        # self.stage4 = self._make_stage(int(512 * width_multiplier[3]), num_blocks[3], stride=1, last_relu=False)
        self.feature_size = 512
        if self.num_classes > 0:
            self.gap = nn.AdaptiveAvgPool2d(output_size=1)
            self.linear = nn.Linear(int(256 * width_multiplier[2]), num_classes)

    def _make_stage(self, planes, num_blocks, stride, last_relu=True):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        relu = True
        for idx, stride in enumerate(strides):
            if idx == len(strides) - 1 and last_relu is False:
                relu = False
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(RepVGGBlock(in_channels=self.in_planes, out_channels=planes, kernel_size=3,
                                      stride=stride, padding=0, groups=cur_groups, deploy=self.deploy,
                                      use_se=self.use_se, last_relu=relu))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.ModuleList(blocks)

    def forward(self, x):
        out = self.stage0(x)

        for stage in (self.stage1, self.stage2, self.stage3):
            for block in stage:
                if self.use_checkpoint:
                    out = checkpoint.checkpoint(block, out)
                else:
                    out = block(out)
        if self.num_classes > 0:
            out = self.gap(out)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
        return out


@MODELS.register_module()
def create_RepVGG_B1(deploy=False, use_checkpoint=False, num_classes=-1):
    return RepVGG(num_classes=num_classes, num_blocks=[4, 3, 2, 1], width_multiplier=[2, 2, 2, 4],
                  override_groups_map=None, deploy=deploy, use_checkpoint=use_checkpoint)


@MODELS.register_module()
def create_RepVGG_B2(deploy=False, use_checkpoint=False, num_classes=-1):
    return RepVGG(num_classes=num_classes, num_blocks=[4, 3, 2, 1], width_multiplier=[1, 1, 1, 4],
                  override_groups_map=None, deploy=deploy, use_checkpoint=use_checkpoint)


if __name__ == "__main__":
    x = torch.randn(1, 3, 127, 127)
    model = create_RepVGG_B1(num_classes=1000)
    model.eval()

    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            nn.init.uniform_(module.running_mean, 0, 0.1)
            nn.init.uniform_(module.running_var, 0, 0.1)
            nn.init.uniform_(module.weight, 0, 0.1)
            nn.init.uniform_(module.bias, 0, 0.1)

    train_y = model(x)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()

    print(model)
    deploy_y = model(x)
    print('========================== The diff is', ((train_y - deploy_y) ** 2).sum())
    summary(model, (3, 127, 127), device='cpu')
