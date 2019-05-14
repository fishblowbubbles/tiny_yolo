from layers import YOLOLayer, MaxPoolLayer, RouteLayer
import numpy as np
import torch
import torch.nn as nn


class Component(nn.Module):
    def __init__(self, op, batch_norm=False):
        super(Component, self).__init__()
        self.op = op
        self.batch_norm = batch_norm
        self.modules = []

    def forward(self, x):
        seq = nn.Sequential(*self.modules)
        return seq(x)

    def add_module(self, module):
        self.modules.append(module)

    def load_weights(self, weights, idx):
        if self.op != "convolutional":
            return idx
        if self.batch_norm:
            cv, bn, _ = self.modules
            idx = self.with_bn(cv, bn, weights, idx)
        else:
            cv, _ = self.modules
            idx = self.without_bn(cv, weights, idx)
        return idx

    def with_bn(self, cv, bn, weights, idx):
        n_bias = bn.bias.numel()
        """ BIAS """
        b = torch.from_numpy(weights[idx:idx + n_bias]).view_as(bn.bias)
        bn.bias.data.copy_(b)
        idx += n_bias
        """ WEIGHT (BN) """
        W = torch.from_numpy(weights[idx:idx + n_bias]).view_as(bn.bias)
        bn.weight.data.copy_(W)
        idx += n_bias
        """ RUNNING MEAN """
        running_mean = torch.from_numpy(weights[idx:idx + n_bias]).view_as(
            bn.running_mean)
        bn.running_mean.data.copy_(running_mean)
        idx += n_bias
        """ RUNNING VARIANCE """
        running_var = torch.from_numpy(weights[idx:idx + n_bias]).view_as(
            bn.running_var)
        bn.running_var.data.copy_(running_var)
        idx += n_bias

        n_weight = cv.weight.numel()
        """ WEIGHT (CV) """
        W = torch.from_numpy(weights[idx:idx + n_weight]).view_as(cv.weight)
        cv.weight.data.copy_(W)
        idx += n_weight

        return idx

    def without_bn(self, cv, weights, idx):
        n_bias = cv.bias.numel()
        """ BIAS """
        b = torch.from_numpy(weights[idx:idx + n_bias]).view_as(cv.bias)
        cv.bias.data.copy_(b)
        idx += n_bias

        n_weight = cv.weight.numel()
        """ WEIGHT """
        W = torch.from_numpy(weights[idx:idx + n_weight]).view_as(cv.weight)
        cv.weight.data.copy_(W)
        idx += n_weight

        return idx


class Tiny(nn.Module):
    def __init__(self, n_classes, mask, anchors, hyperparams):
        super(Tiny, self).__init__()

        self.mask = mask
        self.anchors = [anchors[i] for i in mask]

        self.n_classes = n_classes
        self.n_out = (n_classes + 5) * len(self.anchors)

        self.hyperparams = hyperparams
        self.components = []
        """
        WARNING! MONTROSITY AHEAD!
        Edit values at your own risk.
        """
        cv_0 = Component(op="convolutional", batch_norm=True)
        cv_0.add_module(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False,
            ))
        cv_0.add_module(
            nn.BatchNorm2d(
                num_features=16,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            ))
        cv_0.add_module(nn.LeakyReLU(negative_slope=0.1))
        self.components.append(cv_0)

        mp_1 = Component("maxpool")
        mp_1.add_module(MaxPoolLayer(kernel_size=2, stride=2))
        self.components.append(mp_1)

        cv_2 = Component("convolutional", batch_norm=True)
        cv_2.add_module(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False,
            ))
        cv_2.add_module(
            nn.BatchNorm2d(
                num_features=32,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            ))
        cv_2.add_module(nn.LeakyReLU(negative_slope=0.1))
        self.components.append(cv_2)

        mp_3 = Component("maxpool")
        mp_3.add_module(MaxPoolLayer(kernel_size=2, stride=2))
        self.components.append(mp_3)

        cv_4 = Component("convolutional", batch_norm=True)
        cv_4.add_module(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False,
            ))
        cv_4.add_module(
            nn.BatchNorm2d(
                num_features=64,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            ))
        cv_4.add_module(nn.LeakyReLU(negative_slope=0.1))
        self.components.append(cv_4)

        mp_5 = Component("maxpool")
        mp_5.add_module(MaxPoolLayer(kernel_size=2, stride=2))
        self.components.append(mp_5)

        cv_6 = Component("convolutional", batch_norm=True)
        cv_6.add_module(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False,
            ))
        cv_6.add_module(
            nn.BatchNorm2d(
                num_features=128,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            ))
        cv_6.add_module(nn.LeakyReLU(negative_slope=0.1))
        self.components.append(cv_6)

        mp_7 = Component("maxpool")
        mp_7.add_module(MaxPoolLayer(kernel_size=2, stride=2))
        self.components.append(mp_7)

        cv_8 = Component("convolutional", batch_norm=True)
        cv_8.add_module(
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False,
            ))
        cv_8.add_module(
            nn.BatchNorm2d(
                num_features=256,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            ))
        cv_8.add_module(nn.LeakyReLU(negative_slope=0.1))
        self.components.append(cv_8)

        mp_9 = Component("maxpool")
        mp_9.add_module(MaxPoolLayer(kernel_size=2, stride=2))
        self.components.append(mp_9)

        cv_10 = Component("convolutional", batch_norm=True)
        cv_10.add_module(
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False,
            ))
        cv_10.add_module(
            nn.BatchNorm2d(
                num_features=512,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            ))
        cv_10.add_module(nn.LeakyReLU(negative_slope=0.1))
        self.components.append(cv_10)

        mp_11 = Component("maxpool")
        mp_11.add_module(MaxPoolLayer(kernel_size=2, stride=1, padding=1))
        self.components.append(mp_11)

        cv_12 = Component("convolutional", batch_norm=True)
        cv_12.add_module(
            nn.Conv2d(
                in_channels=512,
                out_channels=1024,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False,
            ))
        cv_12.add_module(
            nn.BatchNorm2d(
                num_features=1024,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            ))
        cv_12.add_module(nn.LeakyReLU(negative_slope=0.1))
        self.components.append(cv_12)

        cv_13 = Component("convolutional", batch_norm=True)
        cv_13.add_module(
            nn.Conv2d(
                in_channels=1024,
                out_channels=256,
                kernel_size=(1, 1),
                stride=(1, 1),
                bias=False,
            ))
        cv_13.add_module(
            nn.BatchNorm2d(
                num_features=256,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            ))
        cv_13.add_module(nn.LeakyReLU(negative_slope=0.1))
        self.components.append(cv_13)

        cv_14 = Component("convolutional", batch_norm=True)
        cv_14.add_module(
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False,
            ))
        cv_14.add_module(
            nn.BatchNorm2d(
                num_features=512,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            ))
        cv_14.add_module(nn.LeakyReLU(negative_slope=0.1))
        self.components.append(cv_14)

        cv_15 = Component("convolutional")
        cv_15.add_module(
            nn.Conv2d(
                in_channels=512,
                out_channels=self.n_out,
                kernel_size=(1, 1),
                stride=(1, 1),
            ))
        cv_15.add_module(nn.LeakyReLU(negative_slope=0.1))
        self.components.append(cv_15)

        yl_16 = Component("yolo")
        yl_16.add_module(YOLOLayer(self.anchors, self.n_classes))
        self.components.append(yl_16)

        rt_17 = Component("route")
        rt_17.add_module(RouteLayer([-4]))
        self.components.append(rt_17)

        cv_18 = Component("convolutional", batch_norm=True)
        cv_18.add_module(
            nn.Conv2d(
                in_channels=256,
                out_channels=128,
                kernel_size=(1, 1),
                stride=(1, 1),
                bias=False,
            ))
        cv_18.add_module(
            nn.BatchNorm2d(
                num_features=128,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            ))
        cv_18.add_module(nn.LeakyReLU(negative_slope=0.1))
        self.components.append(cv_18)

        up_19 = Component("upsample")
        up_19.add_module(nn.Upsample(scale_factor=2.0, mode="nearest"))
        self.components.append(up_19)

        rt_20 = Component("route")
        rt_20.add_module(RouteLayer([-1, 8]))
        self.components.append(rt_20)

        cv_21 = Component("convolutional", batch_norm=True)
        cv_21.add_module(
            nn.Conv2d(
                in_channels=384,
                out_channels=256,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False,
            ))
        cv_21.add_module(
            nn.BatchNorm2d(
                num_features=256,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            ))
        cv_21.add_module(nn.LeakyReLU(negative_slope=0.1))
        self.components.append(cv_21)

        cv_22 = Component("convolutional")
        cv_22.add_module(
            nn.Conv2d(
                in_channels=256,
                out_channels=self.n_out,
                kernel_size=(1, 1),
                stride=(1, 1),
            ))
        cv_22.add_module(nn.LeakyReLU(negative_slope=0.1))
        self.components.append(cv_22)

        yl_23 = Component("yolo")
        yl_23.add_module(YOLOLayer(self.anchors, self.n_classes))
        self.components.append(yl_23)

    def forward(self, x):
        outputs = []
        for component in self.components:
            if component.op != "route":
                x = component(x)
            else:
                x = component(outputs)
            outputs.append(x)
        return outputs

    def load_weights(self, path, partial=False, stop_at=None):
        with open(path, "rb") as file:
            headers = np.fromfile(file, dtype=np.int32, count=5)
            weights = np.fromfile(file, dtype=np.float32)
            file.close()

        if not partial:
            stop_at = len(self.components)

        idx = 0
        for i in range(stop_at):
            idx = self.components[i].load_weights(weights, idx)

        try:
            assert idx == len(weights)
        except AssertionError:
            print("NO. OF WEIGHTS DO NOT MATCH!")
            print("No. of Pretrained Weights = {}".format(len(weights)))
            print("No. of Loaded Weights = {}".format(idx))