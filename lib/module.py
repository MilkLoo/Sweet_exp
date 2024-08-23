"""
    构建自己的模块，串联、并联、划分通道系数并联、交互。
"""
# from lib.AKconv import AKConv
# from lib.LKA import LKA
# from lib.MDTA import MDAttention
# from lib.MobileViTv2Attention import MobileViTv2Attention
import torch
import torch.nn as nn
from lib.GAM import GAM_Attention
from lib.CoordAttention import CoordAtt
from lib.SE import SEAttention

# 串联

# class AK_LKA_MDTA(nn.Module):
#     def __init__(self, dim, branch_ratio=0.25):
#         super().__init__()
#         sp = int(dim * branch_ratio)
#         self.LKA = LKA(sp)
#         self.AKConv = AKConv(inc=sp, outc=sp, num_param=3)
#         self.MDTA = MDAttention(dim=(dim - sp * 2))
#         self.conv = nn.Conv2d(dim, dim, 1)
#         self.split_indexes = (sp, sp, dim - 2 * sp)
#
#     def forward(self, x):
#         res = x
#         x_LKA, x_AKConv, x_MDTA = torch.split(x, self.split_indexes, dim=1)
#         x1 = self.LKA(x_LKA)
#         x2 = self.AKConv(x_AKConv)
#         x3 = self.MDTA(x_MDTA)
#         x = torch.cat((x1, x2, x3), dim=1)
#         x = self.conv(x)
#         return x + res


class MidAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1x1 = nn.Conv2d(4096, 2048, kernel_size=1, stride=1)
        self.coord = CoordAtt(2048, 2048)
        self.gam = GAM_Attention(2048)
        self.se = SEAttention(2048)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
        x4_g = self.gam(x)
        x4_l = self.coord(x)
        x5 = torch.cat((x4_g, x4_l), dim=1)
        x5 = self.conv1x1(x5)
        x5 = self.se(x5)
        x5 = self.gap(x5).view(x5.size(0), -1)
        x5 = self.softmax(x5).view(x5.size(0), x5.size(1), 1, 1)
        x6 = torch.mul(x4_g, x5)
        x7 = torch.add(x4_l, x6)
        return x7

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):  # 对于全连接层
                nn.init.normal_(m.weight, std=0.001)  # 权重使用正态分布初始化
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  # 偏置初始化为0


if __name__ == "__main__":
    model = MidAttention()
    input_1 = torch.rand(1, 2048, 8, 8)
    output = model(input_1)
    print(input_1.shape, "\n", output.shape)
