import torch
import torch.nn as nn
from torch.nn import init


class SEAttention(nn.Module):
    # 初始化SE模块，channel为通道数，reduction为降维比率
    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化层，将特征图的空间维度压缩为1x1
        self.fc = nn.Sequential(  # 定义两个全连接层作为激励操作，通过降维和升维调整通道重要性
            nn.Linear(channel, channel // reduction, bias=False),  # 降维，减少参数数量和计算量
            nn.ReLU(inplace=True),  # ReLU激活函数，引入非线性
            nn.Linear(channel // reduction, channel, bias=False),  # 升维，恢复到原始通道数
            nn.Sigmoid()  # Sigmoid激活函数，输出每个通道的重要性系数
        )

    # 权重初始化方法
    def init_weights(self):
        for m in self.modules():  # 遍历模块中的所有子模块
            if isinstance(m, nn.Conv2d):  # 对于卷积层
                init.kaiming_normal_(m.weight, mode='fan_out')  # 使用Kaiming初始化方法初始化权重
                if m.bias is not None:
                    init.constant_(m.bias, 0)  # 如果有偏置项，则初始化为0
            elif isinstance(m, nn.BatchNorm2d):  # 对于批归一化层
                init.constant_(m.weight, 1)  # 权重初始化为1
                init.constant_(m.bias, 0)  # 偏置初始化为0
            elif isinstance(m, nn.Linear):  # 对于全连接层
                init.normal_(m.weight, std=0.001)  # 权重使用正态分布初始化
                if m.bias is not None:
                    init.constant_(m.bias, 0)  # 偏置初始化为0

    # 前向传播方法
    def forward(self, x):
        b, c, _, _ = x.size()  # 获取输入x的批量大小b和通道数c
        y = self.avg_pool(x).view(b, c)  # 通过自适应平均池化层后，调整形状以匹配全连接层的输入
        y = self.fc(y).view(b, c, 1, 1)  # 通过全连接层计算通道重要性，调整形状以匹配原始特征图的形状
        return x * y.expand_as(x)  # 将通道重要性系数应用到原始特征图上，进行特征重新校准


# 示例使用
if __name__ == '__main__':
    input = torch.randn(50, 512, 7, 7)  # 随机生成一个输入特征图
    bb = nn.AdaptiveAvgPool2d(1)
    res = bb(input).view(input.size(0), -1)
    s = nn.Softmax(dim=1)
    res = s(res).view(input.size(0), input.size(1), 1, 1)
    res = torch.mul(input, res)
    # bb = nn.Conv2d(512, 1024, kernel_size=1)
    # # se = SEAttention(channel=512, reduction=8)  # 实例化SE模块，设置降维比率为8
    # # output = se(input)  # 将输入特征图通过SE模块进行处理
    # res = bb(input)
    print(res.shape)
    # # print(output.shape)  # 打印处理后的特征图形状，验证SE模块的作用
    # import torch
    # import torch.nn.functional as F
    #
    # # 假设 features 是一个大小为 (batch_size, num_channels, height, width) 的特征张量
    # features = torch.randn(1, 2048, 8, 8)
    #
    # # 使用全局平均池化对特征进行降维，得到每个通道的平均值
    # global_avg_pool = F.adaptive_avg_pool2d(features, (1, 1))
    #
    # # 将结果展平为大小为 (batch_size, num_channels)
    # global_avg_pool = global_avg_pool.view(global_avg_pool.size(0), -1)
    #
    # # 对平均值进行 softmax 操作，得到每个通道的概率分布
    # softmaxed_features = F.softmax(global_avg_pool, dim=1).view(1, 2048, 1, 1)
    #
    # res = torch.mul(softmaxed_features, features)
    # print(res.shape)
    #
    # print(softmaxed_features.shape)
