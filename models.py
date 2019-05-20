from layers import YOLO, Route, Upsample
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
        x = nn.Sequential(*self.modules)(x)
        return x

    def add_module(self, module):
        self.modules.append(module)

    def load_weights(self, weights, idx):
        if self.op != "convolutional":
            return idx
        if self.batch_norm:
            cv, bn, _ = self.modules
            idx = self._with_bn(cv, bn, weights, idx)
        else:
            cv = self.modules[0]
            idx = self._without_bn(cv, weights, idx)
        return idx

    def _with_bn(self, cv, bn, weights, idx):
        n_bias, n_weight = bn.bias.numel(), cv.weight.numel()
        # batchnorm
        idx = self._cp_data(target=bn.bias, weights=weights, n_elem=n_bias, idx=idx)
        idx = self._cp_data(target=bn.weight, weights=weights, n_elem=n_bias, idx=idx)
        idx = self._cp_data(target=bn.running_mean, weights=weights, n_elem=n_bias, idx=idx)
        idx = self._cp_data(target=bn.running_var, weights=weights, n_elem=n_bias, idx=idx)
        # convolutional
        idx = self._cp_data(target=cv.weight, weights=weights, n_elem=n_weight, idx=idx)
        return idx

    def _without_bn(self, cv, weights, idx):
        n_bias, n_weight = cv.bias.numel(), cv.weight.numel()
        idx = self._cp_data(target=cv.bias, weights=weights, n_elem=n_bias, idx=idx)
        idx = self._cp_data(target=cv.weight, weights=weights, n_elem=n_weight, idx=idx)
        return idx
    
    def _cp_data(self, target, weights, n_elem, idx):
        data = torch.from_numpy(weights[idx: idx + n_elem]).view_as(target)
        target.data.copy_(data)
        idx += n_elem
        return idx


class Tiny(nn.Module):
    def __init__(self, n_classes, hyperparams, threshold=0.5):
        super(Tiny, self).__init__()
        self.threshold = threshold
        self.n_classes = n_classes
        self.n_out = (n_classes + 5) * 3
        self.hyperparams = hyperparams
        self.components = []
        """
        WARNING! MONTROSITY AHEAD!
        Edit values at your own risk.
        """
        # [ convolutional ]
        cv_0 = Component(op="convolutional", batch_norm=True)
        cv_0.add_module(nn.Conv2d(in_channels=3,
                                  out_channels=16,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(1, 1),
                                  bias=False))
        cv_0.add_module(nn.BatchNorm2d(num_features=16, 
                                       eps=1e-05, 
                                       momentum=0.9))
        cv_0.add_module(nn.LeakyReLU(negative_slope=0.1))
        self.components.append(cv_0)
        # [ maxpool ]
        mp_1 = Component("maxpool")
        mp_1.add_module(nn.MaxPool2d(kernel_size=2, 
                                     stride=2))
        self.components.append(mp_1)
        # [ convolutional ]
        cv_2 = Component("convolutional", batch_norm=True)
        cv_2.add_module(nn.Conv2d(in_channels=16,
                                  out_channels=32,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(1, 1),
                                  bias=False))
        cv_2.add_module(nn.BatchNorm2d(num_features=32, 
                                       eps=1e-05, 
                                       momentum=0.9))
        cv_2.add_module(nn.LeakyReLU(negative_slope=0.1))
        self.components.append(cv_2)
        # [ maxpool ]
        mp_3 = Component("maxpool")
        mp_3.add_module(nn.MaxPool2d(kernel_size=2, 
                                     stride=2))
        self.components.append(mp_3)
        # [ convolutional ]
        cv_4 = Component("convolutional", batch_norm=True)
        cv_4.add_module(nn.Conv2d(in_channels=32,
                                  out_channels=64,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(1, 1),
                                  bias=False))
        cv_4.add_module(nn.BatchNorm2d(num_features=64, 
                                       eps=1e-05, 
                                       momentum=0.9))
        cv_4.add_module(nn.LeakyReLU(negative_slope=0.1))
        self.components.append(cv_4)
        # [ maxpool ]
        mp_5 = Component("maxpool")
        mp_5.add_module(nn.MaxPool2d(kernel_size=2, 
                                     stride=2))
        self.components.append(mp_5)
        # [ convolutional ]
        cv_6 = Component("convolutional", batch_norm=True)
        cv_6.add_module(nn.Conv2d(in_channels=64,
                                  out_channels=128,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(1, 1),
                                  bias=False))
        cv_6.add_module(nn.BatchNorm2d(num_features=128, 
                                       eps=1e-05, 
                                       momentum=0.9))
        cv_6.add_module(nn.LeakyReLU(negative_slope=0.1))
        self.components.append(cv_6)
        # [ maxpool ]
        mp_7 = Component("maxpool")
        mp_7.add_module(nn.MaxPool2d(kernel_size=2, 
                                     stride=2))
        self.components.append(mp_7)
        # [ convolutional ]
        cv_8 = Component("convolutional", batch_norm=True)
        cv_8.add_module(nn.Conv2d(in_channels=128,
                                  out_channels=256,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(1, 1),
                                  bias=False))
        cv_8.add_module(nn.BatchNorm2d(num_features=256, 
                                       eps=1e-05, 
                                       momentum=0.9))
        cv_8.add_module(nn.LeakyReLU(negative_slope=0.1))
        self.components.append(cv_8)
        # [ maxpool ]
        mp_9 = Component("maxpool")
        mp_9.add_module(nn.MaxPool2d(kernel_size=2, 
                                     stride=2))
        self.components.append(mp_9)
        # [ convolutional ]
        cv_10 = Component("convolutional", batch_norm=True)
        cv_10.add_module(nn.Conv2d(in_channels=256,
                                   out_channels=512,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1),
                                   bias=False))
        cv_10.add_module(nn.BatchNorm2d(num_features=512, 
                                        eps=1e-05, 
                                        momentum=0.9))
        cv_10.add_module(nn.LeakyReLU(negative_slope=0.1))
        self.components.append(cv_10)
        # [ maxpool ]
        mp_11 = Component("maxpool")
        mp_11.add_module(nn.ZeroPad2d(padding=(0, 1, 0, 1)))
        mp_11.add_module(nn.MaxPool2d(kernel_size=2, 
                                      stride=1))
        self.components.append(mp_11)
        # [ convolutional ]
        cv_12 = Component("convolutional", batch_norm=True)
        cv_12.add_module(nn.Conv2d(in_channels=512,
                                   out_channels=1024,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1),
                                   bias=False))
        cv_12.add_module(nn.BatchNorm2d(num_features=1024, 
                                        eps=1e-05, 
                                        momentum=0.9))
        cv_12.add_module(nn.LeakyReLU(negative_slope=0.1))
        self.components.append(cv_12)
        # [ convolutional ]
        cv_13 = Component("convolutional", batch_norm=True)
        cv_13.add_module(nn.Conv2d(in_channels=1024,
                                   out_channels=256,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   bias=False))
        cv_13.add_module(nn.BatchNorm2d(num_features=256, 
                                        eps=1e-05, 
                                        momentum=0.9))
        cv_13.add_module(nn.LeakyReLU(negative_slope=0.1))
        self.components.append(cv_13)
        # [ convolutional ]
        cv_14 = Component("convolutional", batch_norm=True)
        cv_14.add_module(nn.Conv2d(in_channels=256,
                                   out_channels=512,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1),
                                   bias=False))
        cv_14.add_module(nn.BatchNorm2d(num_features=512, 
                                        eps=1e-05, 
                                        momentum=0.9))
        cv_14.add_module(nn.LeakyReLU(negative_slope=0.1))
        self.components.append(cv_14)
        # [ convolutional ]
        cv_15 = Component("convolutional")
        cv_15.add_module(nn.Conv2d(in_channels=512,
                                   out_channels=self.n_out,
                                   kernel_size=(1, 1),
                                   stride=(1, 1)))
        self.components.append(cv_15)
        # [ yolo ]
        yl_16 = Component("yolo")
        yl_16.add_module(YOLO(anchors=[(81, 82), (135, 169), (344, 319)], 
                              n_classes=self.n_classes))
        self.components.append(yl_16)
        # [ route ]
        rt_17 = Component("route")
        rt_17.add_module(Route(layers=[-4]))
        self.components.append(rt_17)
        # [ convolutional ]
        cv_18 = Component("convolutional", batch_norm=True)
        cv_18.add_module(nn.Conv2d(in_channels=256,
                                   out_channels=128,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   bias=False))
        cv_18.add_module(nn.BatchNorm2d(num_features=128, 
                                        eps=1e-05, 
                                        momentum=0.9))
        cv_18.add_module(nn.LeakyReLU(negative_slope=0.1))
        self.components.append(cv_18)
        # [ upsample ]
        up_19 = Component("upsample")
        up_19.add_module(Upsample(scale_factor=2.0, 
                                  mode="nearest"))
        self.components.append(up_19)
        # [ route ]
        rt_20 = Component("route")
        rt_20.add_module(Route(layers=[-1, 8]))
        self.components.append(rt_20)
        # [ convolutional ]
        cv_21 = Component("convolutional", batch_norm=True)
        cv_21.add_module(nn.Conv2d(in_channels=384,
                                   out_channels=256,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1),
                                   bias=False))
        cv_21.add_module(nn.BatchNorm2d(num_features=256, 
                                        eps=1e-05, 
                                        momentum=0.9))
        cv_21.add_module(nn.LeakyReLU(negative_slope=0.1))
        self.components.append(cv_21)
        # [ convolutional ]
        cv_22 = Component("convolutional")
        cv_22.add_module(nn.Conv2d(in_channels=256,
                                   out_channels=self.n_out,
                                   kernel_size=(1, 1),
                                   stride=(1, 1)))
        self.components.append(cv_22)
        # [ yolo ]
        yl_23 = Component("yolo")
        yl_23.add_module(YOLO(anchors=[(10, 14), (23, 27), (37, 58)], 
                              n_classes=self.n_classes))
        self.components.append(yl_23)

    def forward(self, x):
        outputs = []
        for component in self.components:
            if component.op != "route":
                x = component(x)
            else:
                x = component(outputs)
            outputs.append(x)
        x = torch.cat((outputs[16], outputs[23]), dim=1)
        return x

    def load_weights(self, path, partial=False, stop_at=None):
        with open(path, "rb") as file:
            headers = np.fromfile(file, dtype=np.int32, count=5)
            weights = np.fromfile(file, dtype=np.float32)
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
