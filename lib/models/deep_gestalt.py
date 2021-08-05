import torch
import torch.nn as nn

from lib.models.face_recog_net import FaceRecogNet


class DeepGestalt(nn.Module):
    def __init__(self, in_channels, num_classes, device, norm_type=nn.BatchNorm2d, act_type=nn.ReLU, freeze=False,
                 pretrained=True, pretrained_path="saved_models/s1_casia_adam_FaceRecogNet_e50_ReLU_BN_bs100.pt"):
        super(DeepGestalt, self).__init__()

        temp = FaceRecogNet(in_channels, 10575, norm_type, act_type)
        self.dropout = nn.Dropout(p=0.5)

        # load pretrained weights
        if pretrained:
            print("Loading pretrained weights")
            temp.load_state_dict(torch.load(pretrained_path, map_location=device))

        # freeze model
        if freeze:
            print("Freezing model weights")
            for param in temp.parameters():
                param.requires_grad = False
        else:
            # Unfreezing all conv layer
            for idx, mod in enumerate(temp.base.children()):
                if idx > -1:
                    if isinstance(mod, Conv_Norm_Act):
                        mod.conv.requires_grad = True
                if isinstance(mod, nn.Conv2d):
                    print(f"Unfroze last conv layer of CNN ({mod})")
                    for param in mod.parameters():
                        param.requires_grad = True

        self.num_classes = num_classes

        self.act = act_type()
        self.base = temp.base

        # replace classifier
        self.classifier = nn.ModuleList()
        self.classifier.append(nn.Linear(in_features=320, out_features=num_classes))

        print(f"Replaced final layer with (Linear in: 320, out: {num_classes})")

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

    def init_layer_weights(self):
        def init_weights(m):
            if type(m) == nn.Linear:
                #torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.fill_(0.0)
        self.classifier.apply(init_weights)

class Conv_Norm_Act(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, norm_type=nn.BatchNorm2d, act_type=nn.ReLU):
        super(Conv_Norm_Act, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = norm_type(out_channels)
        self.act = act_type()
        print(f"Conv_Norm_Act in: {in_channels}, out: {out_channels}, act: {act_type}, norm: {norm_type}")

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))
