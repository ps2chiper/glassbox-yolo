from typing import List

import torch
import torch.nn as nn
import math

from .conv import Conv, DWConv


class DFL(nn.Module):

    def __init__(self, 
                 positions_dist: int = 16) -> None:
        super().__init__()
        self.positions_dist = positions_dist
        self.conv = nn.Conv2d(positions_dist, 1, 1, bias=False).requires_grad_(False)
        weight = torch.arange(positions_dist, dtype=torch.float32).view(1, positions_dist, 1, 1)
        self.conv.weight.data.copy_(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, a = x.shape
        x = x.view(b, 4, self.positions_dist, a).softmax(2).transpose(2, 1)
        return self.conv(x).view(b, 4, a)


class Detect(nn.Module):
    """Detection head."""

    dynamic = False
    export = False
    shape = None

    def __init__(self, 
                 number_classes: int, 
                 channels: List[int], 
                 img_size: int = 640) -> None:
        super().__init__()
        if not channels:
            raise ValueError("Detect head requires at least one input channel entry")

        self.number_classes = number_classes
        self.len_of_channels = len(channels)
        self.reg_max = 16
        self.no = self.number_classes + self.reg_max * 4
        self.stride = torch.tensor([8.0, 16.0, 32.0][: self.len_of_channels]) # torch.zeros(self.len_of_channels)
        self.img_size = img_size


        bbox_conv_channels = max((16, channels[0] // 4, self.reg_max * 4))
        class_conv_channels = max(channels[0], min(self.number_classes, 100))

        self.bbox_conv = nn.ModuleList(
            nn.Sequential(Conv(c, bbox_conv_channels, 3), Conv(bbox_conv_channels, bbox_conv_channels, 3), nn.Conv2d(bbox_conv_channels, 4 * self.reg_max, 1))
            for c in channels
        )
        self.class_conv = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(c, c, 3), Conv(c, class_conv_channels, 1)),
                nn.Sequential(DWConv(class_conv_channels, class_conv_channels, 3), Conv(class_conv_channels, class_conv_channels, 1)),
                nn.Conv2d(class_conv_channels, self.number_classes, 1),
            )
            for c in channels
        )

        self.dfl = DFL(self.reg_max)


    def forward(self, 
                xs: List[torch.Tensor]) -> List[torch.Tensor]:
        for i in range(self.len_of_channels):
            xs[i] = torch.cat((self.bbox_conv[i](xs[i]), self.class_conv[i](xs[i])), 1)
        if self.training:
            return xs
        outputs = self.process_for_inference(xs)

        return outputs if self.export else (outputs, xs)
    
    def process_for_inference(self, xs: List[torch.Tensor]) -> torch.Tensor:
        """Decode each pyramid level in its own grid before joining the predictions."""
        levels = []
        for level, prediction in enumerate(xs):
            height, width = prediction.shape[-2:]
            logits, classes = prediction.flatten(2).split(
                (4 * self.reg_max, self.number_classes), dim=1
            )
            left, top, right, bottom = self.dfl(logits).unbind(dim=1)
            grid_x = torch.arange(width, device=prediction.device, dtype=prediction.dtype)
            grid_y = torch.arange(height, device=prediction.device, dtype=prediction.dtype)
            rows, cols = torch.meshgrid(grid_y + 0.5, grid_x + 0.5, indexing="ij")
            x_min, x_max = cols.flatten() - left, cols.flatten() + right
            y_min, y_max = rows.flatten() - top, rows.flatten() + bottom
            boxes = torch.stack(((x_min + x_max) * 0.5, (y_min + y_max) * 0.5,
                                 x_max - x_min, y_max - y_min), dim=1)
            pixels = boxes * self.stride[level].to(prediction)
            levels.append(torch.cat((pixels, classes.sigmoid()), dim=1))
        return torch.cat(levels, dim=-1)

    @torch.no_grad()
    def bias_init(self):
        for level, stride in enumerate(self.stride):
            cells = (self.img_size / float(stride)) ** 2
            log_prior = math.log(5.0 / (cells * self.number_classes))
            self.class_conv[level][-1].bias.fill_(log_prior)
            self.bbox_conv[level][-1].bias.fill_(1.0)
