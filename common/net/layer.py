import torch
import torch.nn as nn
from torch.nn import functional as F
from timm.models.layers import DropPath

# from config import cfg

"""
    以后改网络可以在这里添加，编写code 添加定制化层
"""


def make_linear_layers(feat_dims, relu_final=True, use_bn=False):
    """
    函数用于构建一个包含线性层（全连接层）的神经网络
    功能：
        使用nn.Linear构建线性层，输入和输出维度分别为feat_dims[i]和feat_dims[i + 1]
        如果不是最后一层（或者是最后一层但relu_final为True），则在每个线性层后添加ReLU激活函数
        如果使用批归一化（use_bn为True），则在每个线性层后添加批归一化层
        最终，函数返回一个包含线性层和激活函数的神经网络（nn.Sequential）。
        这种构建线性层和激活函数的方法是一种通用的模式，常用于创建神经网络的前向传播结构。
    :param feat_dims: 一个包含每个线性层输入和输出维度的列表
    :param relu_final: 一个布尔值，表示是否在最后一层使用ReLU激活函数
    :param use_bn: 一个布尔值，表示是否在每个线性层后使用批归一化（Batch Normalization）
    :return: 一个包含线性层和激活函数（ReLU）的神经网络
    """
    layers = []
    for i in range(len(feat_dims) - 1):
        layers.append(nn.Linear(feat_dims[i], feat_dims[i + 1]))

        # Do not use ReLU for final estimation
        if i < len(feat_dims) - 2 or (i == len(feat_dims) - 2 and relu_final):
            if use_bn:
                layers.append(nn.BatchNorm1d(feat_dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


def make_conv_layers(feat_dims, kernel=3, stride=1, padding=1, bnrelu_final=True):
    """
    根据指定的配置创建一系列卷积层
    功能：
        遍历feat_dims列表，创建一系列卷积层。
        在每个卷积层后面可选地添加批量归一化和ReLU激活，除了最后一层。
        如果bnrelu_final为False，则在最后一层中跳过批量归一化和ReLU激活。
        返回包含所有层的PyTorch Sequential模型。
    :param feat_dims: 包含每个卷积层的输入和输出通道维度的列表
    :param kernel: 卷积核的大小
    :param stride: 卷积中的步幅
    :param padding: 应用于输入的填充
    :param bnrelu_final: 一个布尔值，指示是否在最后一层使用批量归一化和ReLU激活
    :return: 定制化的配置构建卷积神经网络（CNN）架构
    """
    layers = []
    for i in range(len(feat_dims) - 1):
        layers.append(
            nn.Conv2d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i + 1],
                kernel_size=kernel,
                stride=stride,
                padding=padding
            ))
        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims) - 2 or (i == len(feat_dims) - 2 and bnrelu_final):
            layers.append(nn.BatchNorm2d(feat_dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


def make_conv1d_layers(feat_dims, kernel=3, stride=1, padding=1, bnrelu_final=True):
    """
    根据指定的配置创建一系列一维卷积层
    功能：
        遍历feat_dims列表，创建一系列一维卷积层。
        在每个卷积层后面可选地添加批量归一化和ReLU激活，除了最后一层。
        如果bnrelu_final为False，则在最后一层中跳过批量归一化和ReLU激活。
        返回包含所有层的PyTorch Sequential模型
    :param feat_dims: 包含每个卷积层的输入和输出通道维度的列表
    :param kernel: 卷积核的大小
    :param stride: 卷积中的步幅
    :param padding: 应用于输入的填充
    :param bnrelu_final: 一个布尔值，指示是否在最后一层使用批量归一化和ReLU激活
    :return: 定制化的配置构建一维卷积神经网络（CNN）架构
    """
    layers = []
    for i in range(len(feat_dims) - 1):
        layers.append(
            nn.Conv1d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i + 1],
                kernel_size=kernel,
                stride=stride,
                padding=padding
            ))
        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims) - 2 or (i == len(feat_dims) - 2 and bnrelu_final):
            layers.append(nn.BatchNorm1d(feat_dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


def make_deconv_layers(feat_dims, bnrelu_final=True):
    """
    根据指定的配置创建一系列反卷积（转置卷积）层
    功能:
        遍历feat_dims列表，创建一系列反卷积层。
        在每个反卷积层后面可选地添加批量归一化和ReLU激活，除了最后一层。
        如果bnrelu_final为False，则在最后一层中跳过批量归一化和ReLU激活。
        返回包含所有层的PyTorch Sequential模型
    :param feat_dims: 包含每个反卷积层的输入和输出通道维度的列表
    :param bnrelu_final: 一个布尔值，指示是否在最后一层使用批量归一化和ReLU激活
    :return: 定制化的配置构建反卷积神经网络（ Deconvolutional Neural Network）架构
    """
    layers = []
    for i in range(len(feat_dims) - 1):
        layers.append(
            nn.ConvTranspose2d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i + 1],
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
                bias=False))

        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims) - 2 or (i == len(feat_dims) - 2 and bnrelu_final):
            layers.append(nn.BatchNorm2d(feat_dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


class GraphConvBlock(nn.Module):
    """
    实现了图卷积块的功能
    初始化参数：
        adj：图的邻接矩阵，描述图中节点之间的连接关系。
        dim_in：输入特征的维度。
        dim_out：输出特征的维度。
    成员变量：
        adj：存储图的邻接矩阵。
        vertex_num：图中节点的数量。
        fcbn_list：包含多个模块的模块列表，每个模块由线性层和批量归一化组成。
    前向传播方法（forward）：
        接收输入特征张量（feat）。
        对于每个节点，通过对应的线性层和批量归一化层处理输入特征。
        将节点特征在维度1上堆叠，以形成批处理的节点特征。
        将邻接矩阵乘以节点特征，以考虑节点之间的连接关系。
        应用ReLU激活函数。
        返回处理后的节点特征。
    """

    def __init__(self, adj, dim_in, dim_out):
        super(GraphConvBlock, self).__init__()
        self.adj = adj
        self.vertex_num = adj.shape[0]  # 15
        self.fcbn_list = nn.ModuleList(
            [nn.Sequential(*[nn.Linear(dim_in, dim_out), nn.BatchNorm1d(dim_out)]) for _ in range(self.vertex_num)])

    def forward(self, feat):
        batch_size = feat.shape[0]

        # apply kernel for each vertex
        feat = torch.stack([fcbn(feat[:, i, :]) for i, fcbn in enumerate(self.fcbn_list)], 1)

        # apply adj
        adj = self.adj.cuda()[None, :, :].repeat(batch_size, 1, 1)
        feat = torch.bmm(adj, feat)

        # apply activation function
        out = F.relu(feat)
        return out


class GraphResBlock(nn.Module):
    """
    模块通过使用两个图卷积块来处理输入特征，并在其之间添加残差连接，有助于网络训练过程中更好地传播梯度
    初始化参数：
        adj：图的邻接矩阵，描述图中节点之间的连接关系。
        dim：输入和输出特征的维度。
    成员变量：
        adj：存储图的邻接矩阵。
        graph_block1和graph_block2：两个GraphConvBlock模块，每个模块都包含两个图卷积层。
    前向传播方法（forward）：
        调用graph_block1处理输入特征。
        将处理后的特征传递给graph_block2进行进一步处理。
        将两个处理后的特征相加，实现残差连接。
        返回最终输出特征。
    """

    def __init__(self, adj, dim):
        super(GraphResBlock, self).__init__()
        self.adj = adj
        self.graph_block1 = GraphConvBlock(adj, dim, dim)
        self.graph_block2 = GraphConvBlock(adj, dim, dim)

    def forward(self, feat):
        feat_out = self.graph_block1(feat)
        feat_out = self.graph_block2(feat_out)
        out = feat_out + feat
        return out


# 研究内容2的模型所需要的层
class EarlyConv(nn.Module):
    def __init__(self):
        super(EarlyConv, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x


class CDilated(nn.Module):
    """
    This class defines the dilated convolution.
    """

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1, bias=False):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        """
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=bias,
                              dilation=d, groups=groups)

    def forward(self, input):
        """
        :param input: input feature map
        :return: transformed feature map
        """

        output = self.conv(input)
        return output


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class DilatedConv(nn.Module):
    """
    A single Dilated Convolution layer in the Consecutive Dilated Convolutions (CDC) module.
    """

    def __init__(self, dim, k, dilation=1, stride=1, drop_path=0.):
        """
        :param dim: input dimension
        :param k: kernel size
        :param dilation: dilation rate
        :param drop_path: drop_path rate
        :param layer_scale_init_value:
        :param expan_ratio: inverted bottelneck residual
        """

        super().__init__()

        self.ddwconv = CDilated(dim, dim, kSize=k, stride=stride, groups=dim, d=dilation)
        self.bn1 = nn.BatchNorm2d(dim)

        self.norm = LayerNorm(dim, eps=1e-6)
        self.act = nn.GELU()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x

        x = self.ddwconv(x)
        x = self.bn1(x)
        x = self.act(x)
        x = input + self.drop_path(x)

        return x


class BNGELU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-5)
        self.act = nn.GELU()

    def forward(self, x):
        output = self.bn(x)
        output = self.act(output)

        return output


class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding=0, dilation=(1, 1), groups=1, bn_act=False, bias=False):
        super().__init__()

        self.bn_act = bn_act

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_act:
            self.bn_gelu = BNGELU(nOut)

    def forward(self, x):
        output = self.conv(x)

        if self.bn_act:
            output = self.bn_gelu(output)

        return output
