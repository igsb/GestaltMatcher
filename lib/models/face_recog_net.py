## face_recog_net.py
# Basic model for learning face representation
# from https://arxiv.org/abs/1411.7923

import torch.nn as nn


class FaceRecogNet(nn.Module):
    def __init__(self, in_channels, num_classes, norm_type=nn.BatchNorm2d, act_type=nn.ReLU):
        super(FaceRecogNet, self).__init__()
        self.num_classes = num_classes

        self.act = act_type()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True) #we round up the output size for max pools
        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        self.dropout = nn.Dropout(p=0.5)

        # Basically we do this for the separate optimizer settings
        self.base = nn.ModuleList()
        self.classifier = nn.ModuleList()

        # 100x100xin_channels
        self.base.append(Conv_Norm_Act(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1,
                                       norm_type=norm_type, act_type=act_type))
        self.base.append(Conv_Norm_Act(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1,
                                       norm_type=norm_type, act_type=act_type))

        # 100x100x64
        self.base.append(self.max_pool)

        # 50x50x64
        self.base.append(Conv_Norm_Act(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1,
                                       norm_type=norm_type, act_type=act_type))
        self.base.append(Conv_Norm_Act(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1,
                                       norm_type=norm_type, act_type=act_type))

        # 50x50x128
        self.base.append(self.max_pool)

        # 25x25x128
        self.base.append(Conv_Norm_Act(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=1,
                                       norm_type=norm_type, act_type=act_type))
        self.base.append(Conv_Norm_Act(in_channels=96, out_channels=192, kernel_size=3, stride=1, padding=1,
                                       norm_type=norm_type, act_type=act_type))

        # 25x25x192
        self.base.append(self.max_pool)

        # 13x13x192
        self.base.append(Conv_Norm_Act(in_channels=192, out_channels=128, kernel_size=3, stride=1, padding=1,
                                       norm_type=norm_type, act_type=act_type))
        self.base.append(Conv_Norm_Act(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1,
                                       norm_type=norm_type, act_type=act_type))

        # 13x13x256
        self.base.append(self.max_pool)

        # 7x7x256
        self.base.append(Conv_Norm_Act(in_channels=256, out_channels=160, kernel_size=3, stride=1, padding=1,
                                       norm_type=norm_type, act_type=act_type))
        self.base.append(nn.Conv2d(in_channels=160, out_channels=320, kernel_size=3, stride=1, padding=1))
        print(f"Conv2d in: 160, out: 320")

        # 7x7x320
        self.base.append(self.avg_pool)

        # 1x1x320
        # dropout (p=0.5)

        # 320
        self.classifier.append(nn.Linear(in_features=320, out_features=num_classes))  # for CASIA WebFace: 10575 classes
        print(f"Linear in: 320, out: {num_classes}")

    def forward(self, x):

        for mod in self.base:
            x = mod(x)

        # flatten for fully connected layers
        representation = x.view(x.size(0), -1)

        x = self.dropout(representation)
        for mod in self.classifier:
            x = mod(x)

        # expected output size of x: (batch_size, 10575), representations: (320)
        return x, representation


class Conv_Norm_Act(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, norm_type=nn.BatchNorm2d, act_type=nn.ReLU):
        super(Conv_Norm_Act, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = norm_type(out_channels)
        self.act = act_type()
        print(f"Conv_Norm_Act in: {in_channels}, out: {out_channels}, act: {act_type}, norm: {norm_type}")

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))
