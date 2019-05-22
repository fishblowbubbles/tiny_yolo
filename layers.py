import torch
import torch.nn as nn
import torch.nn.functional as F


class YOLO(nn.Module):
    def __init__(self, n_classes, img_size, anchors):
        super(YOLO, self).__init__()
        self.n_classes = n_classes
        self.img_size = img_size
        self.anchors = anchors
        
    def forward(self, x):
        """
        Perform bounding box regression.
        """
        batch_size, _, grid_w, grid_h = x.shape
        proposal = (x.view(batch_size, len(self.anchors), self.n_classes + 5, grid_w, grid_h)
                    .permute(0, 1, 3, 4, 2)
                    .contiguous())
        offset_x, offset_y = (torch.arange(grid_w).repeat(grid_h, 1).view([1, 1, grid_h, grid_w]).type(torch.FloatTensor),
                              torch.arange(grid_h).repeat(grid_w, 1).t().view([1, 1, grid_h, grid_w]).type(torch.FloatTensor))
        stride_w, stride_h = self.img_size[0] / grid_w, self.img_size[1] / grid_h
        scaled_anchors = torch.FloatTensor([(a[0] / stride_w, a[1] / stride_h)
                                            for a in self.anchors])
        anchor_w, anchor_h = (scaled_anchors[:, 0:1].view((1, len(self.anchors), 1, 1)), 
                              scaled_anchors[:, 1:2].view((1, len(self.anchors), 1, 1)))
        output = torch.FloatTensor(proposal.shape)
        # x = P_w * d_x(P) + P_x
        output[..., 0] = (torch.sigmoid(proposal[..., 0]).data + offset_x) * stride_w
        # y = P_h * d_y(P) + P_y
        output[..., 1] = (torch.sigmoid(proposal[..., 1]).data + offset_y) * stride_h
        # w = P_w * exp(d_w(P))
        output[..., 2] = (torch.exp(proposal[..., 2]).data * anchor_w) * stride_w
        # h = P_h * exp(d_h(P))
        output[..., 3] = (torch.exp(proposal[..., 3]).data * anchor_h) * stride_h
        # confidence
        output[..., 4] = torch.sigmoid(proposal[..., 4])
        # class predictions
        output[..., 5:] = torch.sigmoid(proposal[..., 5:])
        output = output.view(batch_size, 
                             len(self.anchors) * grid_h * grid_w, 
                             self.n_classes + 5)
        return output


class Route(nn.Module):
    def __init__(self, layers):
        super(Route, self).__init__()
        self.layers = layers

    def forward(self, x):
        x = torch.cat([x[i] for i in self.layers], dim=1)
        return x


class Upsample(nn.Module):
    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x
