import torch
import torch.nn as nn
import torch.nn.functional as F

from ssd.modeling import registry


def sp_attention_block1(in_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, 1, kernel_size=1, bias=False),
        # nn.Sigmoid()
    ).cuda()

def sp_attention_block2(in_channels, ratio=8):
    output_channels = in_channels//ratio
    return nn.Sequential(
        #1x1 Conv+BN+ReLU
        nn.Conv2d(in_channels, output_channels, kernel_size=1, bias=False),
        nn.BatchNorm2d(output_channels),
        nn.ReLU(),
        #3x3 dilation conv+BN+ReLU
        nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=4, dilation=4),
        nn.BatchNorm2d(output_channels),
        nn.ReLU(),
        nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=4, dilation=4),
        nn.BatchNorm2d(output_channels),
        nn.ReLU(),
        #1x1 Conv
        nn.Conv2d(output_channels, 1, kernel_size=1, bias=False),
        # nn.Sigmoid()
    ).cuda()

def deconv_block(in_channel, out_channel, kernel_size, stride, padding):
    return nn.Sequential(
        # deconv + BN + ReLU
        nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channel),
        # nn.ReLU()
    )

def ch_attention_block(in_channels, reduction=16):
    return nn.Sequential(
        # channel attention
        nn.Linear(in_channels, in_channels//reduction, bias=False), 
        nn.ReLU(inplace=True), 
        nn.Linear(in_channels//reduction, in_channels, bias=False), 
        # nn.Sigmoid()
    ).cuda()

def conv1x1_ReLU(in_channels, output_channels): # 用来升维度或降维度
    return nn.Sequential(
        nn.Conv2d(in_channels, output_channels, kernel_size=1, stride=1), # 避免使用原本用来做分类和回归的SSD参数来变成注意力，所以额外产生再产生一个
        nn.ReLU()
    ).cuda()

def add_sp_attention(cfg):
    layers = []
    flg = True
    for k, v in enumerate(cfg):
        if v == 'X':
            flg = False
            continue
        if flg:
            layers += [sp_attention_block2(v)]
        else:
            layers += [sp_attention_block1(v)]
    return layers

def add_ch_attention(cfg, reduction=16):
    layers = []
    for k, v in enumerate(cfg):
        layers += [ch_attention_block(v, reduction)]
    return layers

def add_deconv(cfg):
    layers = []
    for k, (in_channel, out_channel, kernel_size, stride, padding) in enumerate(cfg):
        layers += [deconv_block(in_channel, out_channel, kernel_size, stride, padding)]
    return layers

attention_base = {
    '300': [512, 1024, 'X', 512, 256, 256, 256]
}
deconv_base = {
    '300': [(512, 1024, 3, 2, 1), (1024, 512, 4, 2, 1)]
}


class NECKTHREEMED(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        size = cfg.INPUT.IMAGE_SIZE
        attention_config = attention_base[str(size)]
        deconv_config = deconv_base[str(size)]
        self.my_sp_attention = nn.ModuleList(add_sp_attention(attention_config[:2]))
        self.conv1x1_ch = conv1x1_ReLU(attention_config[-1], attention_config[-1])
        self.my_ch_attention = nn.ModuleList(add_ch_attention([attention_config[-1]]))
        self.my_deconv = nn.ModuleList(add_deconv(deconv_config))
        self.relu = nn.ReLU()
        self.reset_parameters() # 初始化模型extra layer module里conv卷积层的参数

    def reset_parameters(self):
        for m in self.my_sp_attention:
            for n in m.modules():
                if isinstance(n, nn.Conv2d):
                    nn.init.xavier_uniform_(n.weight)
                if isinstance(n, nn.BatchNorm2d):
                    nn.init.uniform_(n.weight)
        for m in self.my_ch_attention:
            for n in m.modules():
                if isinstance(n, nn.Linear):
                    nn.init.kaiming_normal_(n.weight)
        for m in self.my_deconv:
            for n in m.modules():
                if isinstance(n, nn.ConvTranspose2d):
                    nn.init.xavier_uniform_(n.weight)
                if isinstance(n, nn.BatchNorm2d):
                    nn.init.uniform_(n.weight)

    def forward(self, features):      
        b, c, h, w = features[-1].size()
        tmp_conv11_2 = self.conv1x1_ch[0](features[-1])
        pre_conv11_2_ch_attention_mask = (self.my_ch_attention[0](tmp_conv11_2.view(b, c))).view(b, c, 1, 1)
        deconv1 = self.my_deconv[0](features[2])
        deconv2 = self.my_deconv[1](deconv1)
        deconv_list = [deconv2, deconv1]

        new_features = []
        for i, (feature, deconv) in enumerate(zip(features[:2], deconv_list)):
            pre_sp_attention_mask = self.my_sp_attention[i](feature) # 获得当前层feature的空间注意力掩膜
            # conv11_2_ch_attention_mask size is (b, 256, 1, 1)
            sp_ch_attention_mask = torch.sigmoid(torch.add(pre_sp_attention_mask, pre_conv11_2_ch_attention_mask)) # 当前层feature空间注意力掩膜 + 通道注意力掩膜 
            # 和feature相乘时channels数不对，sp_ch_mask的channels数是256（conv11_2_ch_mask的channels数）
            _groups = int(feature.size()[1] / sp_ch_attention_mask.size()[1])
            # feature = torch.add(feature, deconv)
            if _groups != 1:
                sp_ch_mask_repeated = sp_ch_attention_mask.repeat((1, _groups, 1, 1))
                sp_ch_branch = torch.mul(sp_ch_mask_repeated + 1, feature)
            else:
                sp_ch_branch = torch.mul(sp_ch_attention_mask + 1, feature)
            sp_ch_branch = torch.add(sp_ch_branch, deconv)
            new_features.append(self.relu(sp_ch_branch))
        new_features.extend(features[2:])

        return tuple(new_features)


@registry.NECKS.register('neckthreemed')
def neckthreemed(cfg):
    model = NECKTHREEMED(cfg)
    return model
