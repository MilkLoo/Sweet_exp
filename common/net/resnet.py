import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models.resnet import model_urls


# from lib.module import AK_LKA_MDTA
# from lib.GAM import GAM_Attention
# from lib.CoordAttention import CoordAtt
# from lib.SE import SEAttention


# from layer import make_conv_layers


class ResNetBackbone(nn.Module):

    def __init__(self, resnet_type):

        resnet_spec = {18: (BasicBlock, [2, 2, 2, 2], [64, 64, 128, 256, 512], 'resnet18'),
                       34: (BasicBlock, [3, 4, 6, 3], [64, 64, 128, 256, 512], 'resnet34'),
                       50: (Bottleneck, [3, 4, 6, 3], [64, 256, 512, 1024, 2048], 'resnet50'),
                       101: (Bottleneck, [3, 4, 23, 3], [64, 256, 512, 1024, 2048], 'resnet101'),
                       152: (Bottleneck, [3, 8, 36, 3], [64, 256, 512, 1024, 2048], 'resnet152')}
        block, layers, channels, name = resnet_spec[resnet_type]

        self.name = name
        self.inplanes = 64
        super(ResNetBackbone, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.conv1x1 = nn.Conv2d(4096, 2048, kernel_size=1, stride=1)
        # self.coord = CoordAtt(2048, 2048)
        # self.gam = GAM_Attention(2048)
        # self.se = SEAttention(2048)
        # self.gap = nn.AdaptiveAvgPool2d(1)
        # self.softmax = nn.Softmax(dim=1)
        # self.se_1 = SEAttention(channel=256)
        # self.se_2 = SEAttention(channel=512)
        # self.se_3 = SEAttention(channel=1024)
        # self.se_4 = SEAttention(channel=2048)
        # self.AK_LKA_MDTA_1 = AK_LKA_MDTA(dim=256)
        # self.AK_LKA_MDTA_2 = AK_LKA_MDTA(dim=512)
        # self.AK_LKA_MDTA_3 = AK_LKA_MDTA(dim=1024)
        # self.AK_LKA_MDTA_4 = AK_LKA_MDTA(dim=2048)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, mean=0, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, skip_early=False):
        if not skip_early:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            return x

        x1 = self.layer1(x)  # 这里的x是   bs x 256 x 64 x 64
        # x1 = self.AK_LKA_MDTA_1(x1)
        x2 = self.layer2(x1)  # 512
        # x2 = self.AK_LKA_MDTA_2(x2)
        x3 = self.layer3(x2)  # 1024
        # x3 = self.AK_LKA_MDTA_3(x3)
        x4 = self.layer4(x3)  # 2048
        # x4 = self.AK_LKA_MDTA_4(x4)
        # x4_g = self.gam(x4)
        # x4_l = self.coord(x4)
        # x5 = torch.cat((x4_g, x4_l), dim=1)
        # x5 = self.conv1x1(x5)
        # x5 = self.se(x5)
        # x5 = self.gap(x5).view(x5.size(0), -1)
        # x5 = self.softmax(x5).view(x5.size(0), x5.size(1), 1, 1)
        # x6 = torch.mul(x4_g, x5)
        # x7 = torch.add(x4_l, x6)
        return x4

    def init_weights(self):
        model_url = model_urls[self.name]
        org_resnet = torch.hub.load_state_dict_from_url(model_url)
        # org_resnet = torch.hub.load("pytorch/vision", "resnet50", weights="IMAGENET1K_V2")

        # org_resnet = torch.utils.model_zoo.load_url(model_urls[self.name])
        # drop original resnet fc layer, add 'None' in case of no fc layer, that will raise error
        org_resnet.pop('fc.weight', None)
        org_resnet.pop('fc.bias', None)

        self.load_state_dict(org_resnet)
        print("Initialize resnet from model zoo")

    # def init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.Linear):  # 对于全连接层
    #             nn.init.normal_(m.weight, std=0.001)  # 权重使用正态分布初始化
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)  # 偏置初始化为0
