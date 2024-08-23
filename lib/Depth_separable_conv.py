import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()

        # 深度卷积层
        self.depthwise = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size,
                                                 stride, padding, groups=in_channels),
                                       nn.BatchNorm2d(in_channels),
                                       # activation_layer
                                       nn.LeakyReLU(0.1, inplace=True)
                                       )
        # 逐点卷积层
        self.pointwise = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1),
                                       nn.BatchNorm2d(out_channels),
                                       # activation_layer
                                       nn.LeakyReLU(0.1, inplace=True)
                                       )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

if __name__ == "__main__":
    block = DepthwiseSeparableConv(64,64)  # 创建一个输入通道数为64的CoTAttention实例
    input = torch.rand(1, 64, 64, 64)  # 创建一个随机输入
    output = block(input)  # 通过CoTAttention模块处理输入
    print(output.shape)  # 打印输入和输出的尺寸