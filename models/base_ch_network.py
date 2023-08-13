import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu', norm=None):
        super(DeconvBlock, self).__init__()
        self.deconv = Dynamic_conv2d_transpose(output_size, input_size, kernel_size= kernel_size, stride=stride, padding=padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x,feature):
        
        if self.norm is not None:
            out = self.bn(self.deconv(x,feature))
        else:
            out = self.deconv(x,feature)

        if self.activation is not None:
            return self.act(out)
        else:
            return out

class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None):
        super(ConvBlock, self).__init__()
        self.conv = Dynamic_conv2d(input_size, output_size, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

        self.norm = norm
        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x,feature):
        if self.norm is not None:
            out = self.bn(self.conv(x,feature))
        else:
            out = self.conv(x,feature)

        if self.activation != 'no':
            return self.act(out)
        else:
            return out

class Decoder_MDCBlock1(torch.nn.Module):
    def __init__(self, num_filter, num_ft, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu', norm=None, mode='iter1'):
        super(Decoder_MDCBlock1, self).__init__()
        self.mode = mode
        self.num_ft = num_ft - 1
        self.down_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for i in range(self.num_ft):
            self.down_convs.append(
                ConvBlock(num_filter*(2**i), num_filter*(2**(i+1)), kernel_size, stride, padding, bias, activation, norm=None)
            )
            self.up_convs.append(
                DeconvBlock(num_filter*(2**(i+1)), num_filter*(2**i), kernel_size, stride, padding, bias, activation, norm=None)
            )

    def forward(self, ft_h, ft_l_list,feature):
        if self.mode == 'iter1' or self.mode == 'conv':
            ft_h_list = []
            for i in range(len(ft_l_list)):
                ft_h_list.append(ft_h)
                ft_h = self.down_convs[self.num_ft- len(ft_l_list) + i](ft_h)

            ft_fusion = ft_h
            for i in range(len(ft_l_list)):
                ft_fusion = self.up_convs[self.num_ft-i-1](ft_fusion - ft_l_list[i]) + ft_h_list[len(ft_l_list)-i-1]

        if self.mode == 'iter2':
            ft_fusion = ft_h
            for i in range(len(ft_l_list)):
                ft = ft_fusion
                for j in range(self.num_ft - i):
                    ft = self.down_convs[j](ft,feature)
                ft = ft - ft_l_list[i]
                for j in range(self.num_ft - i):
                    ft = self.up_convs[self.num_ft - i - j - 1](ft,feature)
                ft_fusion = ft_fusion + ft

        if self.mode == 'iter3':
            ft_fusion = ft_h
            for i in range(len(ft_l_list)):
                ft = ft_fusion
                for j in range(i+1):
                    ft = self.down_convs[j](ft)
                ft = ft - ft_l_list[len(ft_l_list) - i - 1]
                for j in range(i+1):
                    ft = self.up_convs[i + 1 - j - 1](ft)
                ft_fusion = ft_fusion + ft

        if self.mode == 'iter4':
            ft_fusion = ft_h
            for i in range(len(ft_l_list)):
                ft = ft_h
                for j in range(self.num_ft - i):
                    ft = self.down_convs[j](ft)
                ft = ft - ft_l_list[i]
                for j in range(self.num_ft - i):
                    ft = self.up_convs[self.num_ft - i - j - 1](ft)
                ft_fusion = ft_fusion + ft

        return ft_fusion
    
class Encoder_MDCBlock1(torch.nn.Module):
    def __init__(self, num_filter, num_ft, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu', norm=None, mode='iter1'):
        super(Encoder_MDCBlock1, self).__init__()
        self.mode = mode
        self.num_ft = num_ft - 1
        self.up_convs = nn.ModuleList()
        self.down_convs = nn.ModuleList()
        for i in range(self.num_ft):
            self.up_convs.append(
                DeconvBlock(num_filter//(2**i), num_filter//(2**(i+1)), kernel_size, stride, padding, bias, activation, norm=None)
            )
            self.down_convs.append(
                ConvBlock(num_filter//(2**(i+1)), num_filter//(2**i), kernel_size, stride, padding, bias, activation, norm=None)
            )

    def forward(self, ft_l, ft_h_list,feature):
        if self.mode == 'iter1' or self.mode == 'conv':
            ft_l_list = []
            for i in range(len(ft_h_list)):
                ft_l_list.append(ft_l)
                ft_l = self.up_convs[self.num_ft- len(ft_h_list) + i](ft_l,feature)

            ft_fusion = ft_l
            for i in range(len(ft_h_list)):
                ft_fusion = self.down_convs[self.num_ft-i-1](ft_fusion - ft_h_list[i],feature) + ft_l_list[len(ft_h_list)-i-1]

        if self.mode == 'iter2':
            ft_fusion = ft_l
            for i in range(len(ft_h_list)):
                ft = ft_fusion
                for j in range(self.num_ft - i):
                    ft = self.up_convs[j](ft,feature)
                ft = ft - ft_h_list[i]
                for j in range(self.num_ft - i):
                    ft = self.down_convs[self.num_ft - i - j - 1](ft,feature)
                ft_fusion = ft_fusion + ft

        if self.mode == 'iter3':
            ft_fusion = ft_l
            for i in range(len(ft_h_list)):
                ft = ft_fusion
                for j in range(i+1):
                    ft = self.up_convs[j](ft)
                ft = ft - ft_h_list[len(ft_h_list) - i - 1]
                for j in range(i+1):
                    ft = self.down_convs[i + 1 - j - 1](ft)
                ft_fusion = ft_fusion + ft

        if self.mode == 'iter4':
            ft_fusion = ft_l
            for i in range(len(ft_h_list)):
                ft = ft_l
                for j in range(self.num_ft - i):
                    ft = self.up_convs[j](ft)
                ft = ft - ft_h_list[i]
                for j in range(self.num_ft - i):
                    ft = self.down_convs[self.num_ft - i - j - 1](ft)
                ft_fusion = ft_fusion + ft

        return ft_fusion
    
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.001)
        torch.nn.init.xavier_uniform(m.weight)
#net.apply(init_weights)
#        self.conv2d = Dynamic_conv2d(in_channels, out_channels, kernel_size= kernel_size,stride=stride,mask_blur = mask_blur[name],mask_rain = mask_rain[name],mask_noise = mask_noise[name])
class Dynamic_conv2d_v2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size= 3, stride=1, padding=0, dilation=1, groups=1, bias=True, K=10, init_weight=True):
        super(Dynamic_conv2d_v2, self).__init__()
        assert in_channels % groups == 0
        self.in_planes = in_channels
        self.out_planes = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.if_bias = bias
        self.K = K

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels//groups, kernel_size, kernel_size), requires_grad=True)
        self.transformer_bayer = nn.Parameter(torch.zeros(1,out_channels, in_channels//groups, kernel_size, kernel_size), requires_grad=True)
        self.transformer_quad = nn.Parameter(torch.zeros(1,out_channels, in_channels//groups, kernel_size, kernel_size), requires_grad=True)
        self.transformer_nano = nn.Parameter(torch.zeros(1,out_channels, in_channels//groups, kernel_size, kernel_size), requires_grad=True)
        self.transformer_qxq = nn.Parameter(torch.zeros(1,out_channels, in_channels//groups, kernel_size, kernel_size), requires_grad=True)
#        self.transformer = nn.Parameter(torch.randn(K, out_channels, in_channels//groups, kernel_size, kernel_size), requires_grad=True)

        self.mask_bayer = nn.Parameter(torch.zeros((1,self.out_planes, self.in_planes//groups, 1,1)).cuda(), requires_grad=False)
        self.mask_quad = nn.Parameter(torch.zeros((1,self.out_planes, self.in_planes//groups, 1,1)).cuda(), requires_grad=False)
        self.mask_nano = nn.Parameter(torch.zeros((1,self.out_planes, self.in_planes//groups, 1,1)).cuda(), requires_grad=False)
        self.mask_qxq = nn.Parameter(torch.zeros((1,self.out_planes, self.in_planes//groups, 1,1)).cuda(), requires_grad=False)
        if self.if_bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels), requires_grad=True)
        else:
            self.bias = None



    def forward(self, x_softmax_attention):
        x, softmax_attention = x_softmax_attention
        batch_size, in_planes, height, width = x.size()
        batch_size_soft, in_planes_soft = softmax_attention.size()
        softmax_attention = softmax_attention.view(batch_size_soft,in_planes_soft,1,1,1)
        
        x = x.contiguous().view(1, -1, height, width)
        weight_bayer = self.transformer_bayer*self.mask_bayer.clone() * softmax_attention[:,1:2,:,:,:]
        weight_quad = self.transformer_quad*self.mask_quad.clone() * softmax_attention[:,2:3,:,:,:]
        weight_nano = self.transformer_nano*self.mask_nano.clone() * softmax_attention[:,0:1,:,:,:]
        weight_qxq = self.transformer_qxq*self.mask_qxq.clone() * softmax_attention[:,0:1,:,:,:]
        weight = weight_bayer + weight_quad + weight_nano + weight_qxq
        weight_bias = self.weight.view(1,self.out_planes, self.in_planes//self.groups, self.kernel_size, self.kernel_size).clone()
        
        aggregate_weight = (weight_bias + weight).view(-1, self.in_planes//self.groups, self.kernel_size, self.kernel_size)

        
        output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups*batch_size)


        if self.bias is not None:
            output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1)) + self.bias.view(1,-1,1,1).clone()
        else:
            output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))

        return output
    
class Dynamic_conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size= 3, stride=1, padding=0, dilation=1, groups=1, bias=True, K=10, init_weight=True):
        super(Dynamic_conv2d, self).__init__()
        assert in_channels % groups == 0
        self.in_planes = in_channels
        self.out_planes = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.if_bias = bias
        self.K = K

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels//groups, kernel_size, kernel_size), requires_grad=True)
        self.transformer_bayer = nn.Parameter(torch.zeros(1,out_channels, in_channels//groups, kernel_size, kernel_size), requires_grad=True)
        self.transformer_quad = nn.Parameter(torch.zeros(1,out_channels, in_channels//groups, kernel_size, kernel_size), requires_grad=True)
        self.transformer_nano = nn.Parameter(torch.zeros(1,out_channels, in_channels//groups, kernel_size, kernel_size), requires_grad=True)
        self.transformer_qxq = nn.Parameter(torch.zeros(1,out_channels, in_channels//groups, kernel_size, kernel_size), requires_grad=True)
#        self.transformer = nn.Parameter(torch.randn(K, out_channels, in_channels//groups, kernel_size, kernel_size), requires_grad=True)

        self.mask_bayer = nn.Parameter(torch.zeros((1,self.out_planes, self.in_planes//groups, 1,1)).cuda(), requires_grad=False)
        self.mask_quad = nn.Parameter(torch.zeros((1,self.out_planes, self.in_planes//groups, 1,1)).cuda(), requires_grad=False)
        self.mask_nano = nn.Parameter(torch.zeros((1,self.out_planes, self.in_planes//groups, 1,1)).cuda(), requires_grad=False)
        self.mask_qxq = nn.Parameter(torch.zeros((1,self.out_planes, self.in_planes//groups, 1,1)).cuda(), requires_grad=False)
        if self.if_bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels), requires_grad=True)
        else:
            self.bias = None



    def forward(self, x, softmax_attention):
        batch_size, in_planes, height, width = x.size()
        batch_size_soft, in_planes_soft = softmax_attention.size()
        softmax_attention = softmax_attention.view(batch_size_soft,in_planes_soft,1,1,1)
        x = x.contiguous().view(1, -1, height, width)
        weight_bayer = self.transformer_bayer*self.mask_bayer.clone() * softmax_attention[:,0:1,:,:,:]
        weight_quad = self.transformer_quad*self.mask_quad.clone() * softmax_attention[:,1:2,:,:,:]
        weight_nano = self.transformer_nano*self.mask_nano.clone() * softmax_attention[:,2:3,:,:,:]        
        weight_qxq = self.transformer_qxq*self.mask_qxq.clone() * softmax_attention[:,3:4,:,:,:]
        
        weight = weight_bayer + weight_quad + weight_nano + weight_qxq 
        weight_bias = self.weight.view(1,self.out_planes, self.in_planes//self.groups, self.kernel_size, self.kernel_size).clone()
        
        aggregate_weight = (weight_bias + weight).view(-1, self.in_planes//self.groups, self.kernel_size, self.kernel_size)

        
        output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups*batch_size)


        if self.bias is not None:
            output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1)) + self.bias.view(1,-1,1,1).clone()
        else:
            output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))

        return output

class Dynamic_conv2d_transpose(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size= 3, stride=1, padding=0, dilation=1, groups=1, bias=True, K=10, init_weight=True):
        super(Dynamic_conv2d_transpose, self).__init__()
        assert in_planes % groups == 0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.if_bias = bias
        self.K = K

        self.weight = nn.Parameter(torch.randn(out_planes, in_planes//groups, kernel_size, kernel_size), requires_grad=True)

        self.transformer_bayer = nn.Parameter(torch.zeros(1,out_planes, in_planes//groups, kernel_size, kernel_size), requires_grad=True)
        self.transformer_quad = nn.Parameter(torch.zeros(1,out_planes, in_planes//groups, kernel_size, kernel_size), requires_grad=True)
        self.transformer_nano = nn.Parameter(torch.zeros(1,out_planes, in_planes//groups, kernel_size, kernel_size), requires_grad=True)
        self.transformer_qxq = nn.Parameter(torch.zeros(1,out_planes, in_planes//groups, kernel_size, kernel_size), requires_grad=True)

        self.mask_bayer = nn.Parameter(torch.zeros((1,self.out_planes, self.in_planes//groups, 1,1)).cuda(), requires_grad=False)
        self.mask_quad = nn.Parameter(torch.zeros((1,self.out_planes, self.in_planes//groups, 1,1)).cuda(), requires_grad=False)
        self.mask_nano = nn.Parameter(torch.zeros((1,self.out_planes, self.in_planes//groups, 1,1)).cuda(), requires_grad=False)
        self.mask_qxq = nn.Parameter(torch.zeros((1,self.out_planes, self.in_planes//groups, 1,1)).cuda(), requires_grad=False)
        
        if self.if_bias:
            self.bias = nn.Parameter(torch.Tensor(in_planes), requires_grad=True)
        else:
            self.bias = None
    def forward(self, x, softmax_attention):
        batch_size, in_planes, height, width = x.size()
        batch_size_soft, in_planes_soft = softmax_attention.size()
        softmax_attention = softmax_attention.view(batch_size_soft,in_planes_soft,1,1,1)
        x = x.contiguous().view(1, -1, height, width)
        
        weight_bayer = self.transformer_bayer*self.mask_bayer.clone() * softmax_attention[:,0:1,:,:,:]
        weight_quad = self.transformer_quad*self.mask_quad.clone() * softmax_attention[:,1:2,:,:,:]
        weight_nano = self.transformer_nano*self.mask_nano.clone() * softmax_attention[:,2:3,:,:,:]
        
        weight_qxq = self.transformer_qxq*self.mask_qxq.clone() * softmax_attention[:,3:4,:,:,:]
        
        weight = weight_bayer + weight_quad + weight_nano + weight_qxq
        
        weight_bias = self.weight.view(1,self.out_planes, self.in_planes, self.kernel_size, self.kernel_size)

        aggregate_weight = (weight_bias + weight).view(-1, self.in_planes, self.kernel_size, self.kernel_size)

        
        output = F.conv_transpose2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups*batch_size)


        if self.bias is not None:
            output = output.view(batch_size, self.in_planes, output.size(-2), output.size(-1))+ self.bias.view(1,-1,1,1)
        else:
            output = output.view(batch_size, self.in_planes, output.size(-2), output.size(-1))

        return output
