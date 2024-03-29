import torch
import torch.nn as nn
import numpy as np
import torch.utils.checkpoint as checkpoint
from torchsummary import summary
from mmpretrain.registry import MODELS
from .base_backbone import BaseBackbone


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                                        bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result


class RepVGGBlock(BaseBackbone):
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


@MODELS.register_module()
class RepVGG10(nn.Module):

    def __init__(self, deploy=False):
        super(RepVGG10, self).__init__()

        self.deploy = deploy
        self.feature_size = 512  # todo

        self.stage0 = nn.Sequential()
        self.stage1 = nn.Sequential()
        self.stage2 = nn.Sequential()
        self.stage3 = nn.Sequential()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.stage0.append(
            RepVGGBlock(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=0, deploy=self.deploy))
        self.stage0.append(
            RepVGGBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, deploy=self.deploy))

        self.stage1.append(
            RepVGGBlock(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0, deploy=self.deploy))
        self.stage1.append(
            RepVGGBlock(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=0, deploy=self.deploy))

        self.stage2.append(
            RepVGGBlock(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=0, deploy=self.deploy))
        self.stage2.append(
            RepVGGBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0, deploy=self.deploy))

        self.stage3.append(
            RepVGGBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=0, deploy=self.deploy))
        self.stage3.append(
            RepVGGBlock(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=0, deploy=self.deploy))
        self.stage3.append(
            RepVGGBlock(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=0, deploy=self.deploy))
        self.stage3.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        x = self.stage0(x)
        x = self.pool(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return (x, )


if __name__ == "__main__":
    x = torch.randn(1, 3, 127, 127)
    model = RepVGG10(deploy=False)
    model.eval()

    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            nn.init.uniform_(module.running_mean, 0, 0.1)
            nn.init.uniform_(module.running_var, 0, 0.1)
            nn.init.uniform_(module.weight, 0, 0.1)
            nn.init.uniform_(module.bias, 0, 0.1)
    summary(model, (3, 127, 127), device='cpu')

    train_y = model(x)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()

    print(model)
    deploy_y = model(x)
    print('========================== The diff is', ((train_y - deploy_y) ** 2).sum())
    summary(model, (3, 127, 127), device='cpu')
