import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class YOLOLayer(nn.Module):
    def __init__(self, anchors, n_classes, img_size=416):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.n_classes = n_classes
        self.img_size = img_size

    def forward(self, x):
        """
        Performs bounding box regression.
        """
        batch_size, _, grid_h, grid_w = x.shape
        proposal = (x.view(batch_size, len(self.anchors), self.n_classes + 5,
                           grid_h, grid_w).permute(0, 1, 3, 4, 2).contiguous())

        offset_x = torch.FloatTensor(np.arange(grid_w)).view(-1, 1)
        offset_y = torch.FloatTensor(np.arange(grid_h)).view(-1, 1)

        scaled_anchors = torch.FloatTensor([(a[0] / (self.img_size / grid_h),
                                             a[1] / (self.img_size / grid_w))
                                            for a in self.anchors])
        anchor_w = scaled_anchors[:, 0:1].view((1, len(self.anchors), 1, 1))
        anchor_h = scaled_anchors[:, 1:2].view((1, len(self.anchors), 1, 1))

        output = torch.FloatTensor(proposal.shape)
        """ P_w * d_x(P) + P_x """
        output[..., 0] = torch.sigmoid(proposal[..., 0]).data + offset_x
        """ P_h * d_y(P) + P_y """
        output[..., 1] = torch.sigmoid(proposal[..., 1]).data + offset_y
        """ P_w * exp(d_w(P)) """
        output[..., 2] = torch.exp(proposal[..., 2]).data * anchor_w
        """ P_h * exp(d_h(P)) """
        output[..., 3] = torch.exp(proposal[..., 3]).data * anchor_h
        """ CONFIDENCE """
        output[..., 4] = torch.sigmoid(proposal[..., 4])
        """ CLASS PREDICTIONS """
        output[..., 5:] = torch.sigmoid(proposal[..., 5:])

        return output


class MaxPoolLayer(nn.Module):
    """
    A wrapper for nn.MaxPool2d to accomodate asymmetrical padding.
    """
    def __init__(self, kernel_size, stride, padding=0):
        super(MaxPoolLayer, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        x = F.pad(x, (0, self.padding, 0, self.padding), mode="replicate")
        x = nn.MaxPool2d(kernel_size=self.kernel_size, stride=self.stride)(x)
        return x


class RouteLayer(nn.Module):
    def __init__(self, layers):
        super(RouteLayer, self).__init__()
        self.layers = layers

    def forward(self, x):
        x = torch.cat([x[i] for i in self.layers], dim=1)
        return x