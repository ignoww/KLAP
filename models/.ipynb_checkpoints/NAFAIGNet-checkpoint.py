import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_ch_network import Dynamic_conv2d, Dynamic_conv2d_v2



class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None
    
    
class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)
    
        
    
class SimpleGate(nn.Module):

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
    
class AdaptiveAvgPool2d_f(nn.Module):
    def __init__(self):
        super(AdaptiveAvgPool2d_f, self).__init__()

        self.avg = nn.AdaptiveAvgPool2d(1)
    def forward(self, x_guide_feature):
        x,guide_feature = x_guide_feature
        x = self.avg(x)
        return [x,guide_feature]
class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
#        Dynamic_conv2d(in_planes, out_planes, kernel_size= 3, stride=1, padding=0, dilation=1, groups=1, bias=True, K=10, init_weight=True)
        self.conv1 = Dynamic_conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = Dynamic_conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = Dynamic_conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            AdaptiveAvgPool2d_f(),
            Dynamic_conv2d_v2(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = Dynamic_conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = Dynamic_conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp_guide_feature):
        inp,guide_feature = inp_guide_feature
        x = inp

        x = self.norm1(x)

        x = self.conv1(x,guide_feature)
        x = self.conv2(x,guide_feature)
        x = self.sg(x)
        x = x * self.sca([x,guide_feature])
        x = self.conv3(x,guide_feature)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y),guide_feature)
        x = self.sg(x)
        x = self.conv5(x,guide_feature)

        x = self.dropout2(x)

        return [y + x * self.gamma,guide_feature]


class NAFNet(nn.Module):
#        self.conv2d = Dynamic_conv2d(in_channels, out_channels, kernel_size= kernel_size,stride=stride,mask_blur = mask_blur[name],mask_rain = mask_rain[name],mask_noise = mask_noise[name])

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()
        width = 32
        enc_blk_nums = [1, 1, 1, 28]
        dec_blk_nums = [1, 1, 1, 1]
        middle_blk_num = 1
        self.intro = Dynamic_conv2d(in_channels=1, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = Dynamic_conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                Dynamic_conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    Dynamic_conv2d_v2(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp_feature):
        inp,guide_feature = inp_feature
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp,guide_feature)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x,_ = encoder([x,guide_feature])
            encs.append(x)
            x = down(x,guide_feature)

        x,_ = self.middle_blks([x,guide_feature])

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up([x,guide_feature])
            x = x + enc_skip
            x,_ = decoder([x,guide_feature])

        x = self.ending(x,guide_feature)
        x = x 

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

