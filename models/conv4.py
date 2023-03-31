import torch.nn as nn


# def conv_block(in_channels, out_channels):
#     return nn.Sequential(
#         nn.Conv2d(in_channels, out_channels, 3, padding=1),
#         nn.BatchNorm2d(out_channels),
#         nn.ReLU(),
#         nn.MaxPool2d(2)
#     )

# class ConvNet4(nn.Module):

#     def __init__(self,args, x_dim=3, hid_dim=64, z_dim=640):
#         super().__init__()
#         self.args = args
#         self.encoder = nn.Sequential(
#             conv_block(x_dim, hid_dim),
#             conv_block(hid_dim, hid_dim),
#             conv_block(hid_dim, hid_dim),
#             conv_block(hid_dim, z_dim),
#         )

#     def forward(self, x):
#         x = self.encoder(x)
#         return x
import torch.nn.functional as F
import torch
import torch.nn as nn




# model = ConvNet4(num_classes=len(train_data.classes)).to(device)

# ([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(15),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5071, 0.4867, 0.4408),
#                          (0.2675, 0.2565, 0.2761)),
# ])
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
def featureL2Norm(feature):
    epsilon = 1e-6
    norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
    return torch.div(feature, norm)

class mySelfCorrelationComputation(nn.Module):
    def __init__(self, channel,kernel_size=(1, 1), padding=0):
        super(mySelfCorrelationComputation, self).__init__()
        planes =[640, 64, 64, 640]
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding)
        self.conv1 = conv3x3(channel, channel)
        self.relu = nn.LeakyReLU(0.1)
        self.conv1x1_in = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=1, bias=False, padding=0),
                                        nn.BatchNorm2d(planes[1]),
                                        nn.ReLU(inplace=True))
        # self.relu = nn.ReLU(inplace=False)
        self.bn2 = nn.BatchNorm2d(channel)
        self.conv1x1_in = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=1, bias=False, padding=0),
                                        nn.BatchNorm2d(channel),
                                        nn.ReLU(inplace=True))
        self.embeddingFea = nn.Sequential(nn.Conv2d(channel*2, channel,
                                                     kernel_size=1, bias=False, padding=0),
                                           nn.BatchNorm2d(channel),
                                           nn.ReLU(inplace=True))
        self.conv1x1_out = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1, bias=False, padding=0),
            nn.BatchNorm2d(channel))
        self.maxpool = nn.MaxPool2d(1)

    def forward(self, x):

        # x = self.conv1x1_in(x)
        # x = self.conv1(x)
        # x = self.bn2(x)
        # x = self.relu(x)
        b, c, h, w = x.shape
        #
        x0 = self.relu(x)
        x = x0
        x = F.normalize(x, dim=1, p=2)
        identity = x

        x = self.unfold(x)  # 提取出滑动的局部区域块，这里滑动窗口大小为5*5，步长为1
        # b, cuv, h, w  （80,640*5*5,5,5)
        x = x.view(b, c, self.kernel_size[0], self.kernel_size[1], h, w)  # b, c, u, v, h, w
        x = x * identity.unsqueeze(2).unsqueeze(2)  # 通过unsqueeze增维使identity和x变为同维度  公式（1）
        # b, c, u, v, h, w * b, c, 1, 1, h, w
        x = x.view(b, -1, h, w)
        # x = x.permute(0, 1, 4, 5, 2, 3).contiguous()  # b, c, h, w, u, v
        # torch.contiguous()方法首先拷贝了一份张量在内存中的地址，然后将地址按照形状改变后的张量的语义进行排列
        # x = x.mean(dim=[-1, -2])
        feature_gs = featureL2Norm(x)

        # concatenate
        feature_cat = torch.cat([identity, feature_gs], 1)

        # embed
        feature_embd = self.embeddingFea(feature_cat)
        feature_embd = self.conv1x1_out(feature_embd)
        feature_embd = self.relu(feature_embd)
        feature_embd = self.maxpool(feature_embd)
        return feature_embd


def gaussian_normalize( x, dim, eps=1e-05):
    x_mean = torch.mean(x, dim=dim, keepdim=True)
    x_var = torch.var(x, dim=dim, keepdim=True)  # 求dim上的方差
    x = torch.div(x - x_mean, torch.sqrt(x_var + eps))  # （x原始-x平均）/根号下x_var
    return x

def normalize_feature(x):
    return x - x.mean(1).unsqueeze(1)  # x-x.mean(1)行求平均值并在channal维上增加一个维度

class SqueezeExcitation(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SqueezeExcitation, self).__init__()
        hdim = 64
        self.conv1x1_in = nn.Sequential(nn.Conv2d(channel, hdim, kernel_size=1, bias=False, padding=0),
                                        nn.BatchNorm2d(hdim),
                                        nn.ReLU(inplace=False))
        self.conv1x1_out = nn.Sequential(nn.Conv2d(hdim, channel, kernel_size=1, bias=False, padding=0),
                                        nn.BatchNorm2d(channel),
                                        nn.ReLU(inplace=False))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(hdim, hdim // reduction, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(hdim // reduction, hdim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1x1_in(x)
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        x = x * y.expand_as(x)
        x = self.conv1x1_out(x)
        return x







class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 利用1x1卷积代替全连接
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class cbam_block(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=3):
        super(cbam_block, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class ConvNet4(nn.Module):

    def __init__(self,args, x_dim=3, hid_dim=64, z_dim=640):
        super().__init__()
        self.args = args
        self.conv_block1 = conv_block(x_dim, hid_dim)
        self.conv_block2 = conv_block(hid_dim, 160)
        self.conv_block3 = conv_block(160, 320)
        self.conv_block4 = conv_block(320, z_dim)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(640, num_classes)
        self.scr_module0 = mySelfCorrelationComputation(channel=64,kernel_size=(1, 1), padding=0)
        self.scr_module1 = mySelfCorrelationComputation(channel=160, kernel_size=(1, 1), padding=0)
        self.scr_module2 = mySelfCorrelationComputation(channel=320, kernel_size=(1, 1), padding=0)
        self.scr_module = mySelfCorrelationComputation(channel=640, kernel_size=(1, 1), padding=0)
        # self.bn = nn.BatchNorm1d(int(640))
        # self.linear = nn.Linear(int(640), num_classes)
        self.relu = nn.LeakyReLU(0.1)
        self.maxpool = nn.MaxPool2d(1)
        # self.scr_module = cbam_block(channel=640)
        # self.scr_module = SqueezeExcitation(channel=640)
        self.conv1x1_out = nn.Sequential(
            nn.Conv2d(1120, 640, kernel_size=1, bias=False, padding=0),
            nn.BatchNorm2d(640))

    def forward(self, x):
        out1 = self.conv_block1(x)
        out1_s = self.scr_module0(out1)
        out1 = out1+out1_s


        out2 = self.conv_block2(out1)
        out2_s = self.scr_module1(out2)
        out2 = out2 + out2_s


        out3 = self.conv_block3(out2)
        out3_s = self.scr_module2(out3)
        out3 = out3 + out3_s


        out4 = self.conv_block4(out3)
        out4_s = self.scr_module(out4)
        out4 = out4 + out4_s


#___________________________________________________________
#         out2 = F.avg_pool2d(out2, out2.size()[2:])
#         out3 = F.avg_pool2d(out3, out3.size()[2:])
#         out4 = F.avg_pool2d(out4, out4.size()[2:])

#         out2 = F.layer_norm(out2, out2.size()[1:])
#         out3 = F.layer_norm(out3, out3.size()[1:])
#         out4 = F.layer_norm(out4, out4.size()[1:])

        out = torch.cat([out4,out3,out2], 1)
        out = self.conv1x1_out(out)
        out = self.relu(out)
        out = self.maxpool(out)

#__________________________________________
#         x = self.avgpool(out)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)

        return x
