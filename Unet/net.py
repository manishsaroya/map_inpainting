import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_fun


class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [torch.cat((image,image,image),dim=1)]

        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))

        return results[1:]


class PConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv2d = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding).to(device)
        self.mask2d = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding).to(device)
        self.conv2d.apply(weights_init('kaiming'))
        self.mask2d.weight.data.fill_(1.0)
        self.mask2d.bias.data.fill_(0.0)

        # mask is not updated
        for param in self.mask2d.parameters():
            param.requires_grad = True

    def forward(self, input, input_mask):
        # http://masc.cs.gmu.edu/wiki/partialconv
        # C(X) = W^T * X + b, C(0) = b, D(M) = 1 * M + 0 = sum(M)
        # W^T* (M .* X) / sum(M) + b = [C(M .* X) – C(0)] / D(M) + C(0)

        input_0 = input.new_zeros(input.size())



        output = F.conv2d(
            input * input_mask, self.conv2d.weight, self.conv2d.bias,
            self.conv2d.stride, self.conv2d.padding, self.conv2d.dilation,
            self.conv2d.groups)

        output_0 = F.conv2d(input_0, self.conv2d.weight, self.conv2d.bias,
                            self.conv2d.stride, self.conv2d.padding,
                            self.conv2d.dilation, self.conv2d.groups)

        with torch.no_grad():
            output_mask = F.conv2d(
                input_mask, self.mask2d.weight, self.mask2d.bias,
                self.mask2d.stride, self.mask2d.padding, self.mask2d.dilation,
                self.mask2d.groups)

        n_z_ind = (output_mask != 0.0)
        z_ind = (output_mask == 0.0)  # skip all the computation

        output[n_z_ind] = \
            (output[n_z_ind] - output_0[n_z_ind]) / output_mask[n_z_ind] + \
            output_0[n_z_ind]
        output[z_ind] = 0.0

        output_mask[n_z_ind] = 1.0
        output_mask[z_ind] = 0.0

        return output, output_mask


class PCBActiv(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, sample='none-3', activ='relu'):
        super().__init__()
        if sample == 'down-5':
            self.conv = PConv2d(in_ch, out_ch, 5, 2, 2)
        elif sample == 'down-7':
            self.conv = PConv2d(in_ch, out_ch, 7, 2, 3)
        elif sample == 'down-3':
            self.conv = PConv2d(in_ch, out_ch, 3, 2, 1)
        else:
            self.conv = PConv2d(in_ch, out_ch, 3, 1, 1)

        if bn:
            self.bn = nn.BatchNorm2d(out_ch)
        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input, input_mask):
        h, h_mask = self.conv(input, input_mask)
        if hasattr(self, 'bn'):
            h = self.bn(h)
        if hasattr(self, 'activation'):
            h = self.activation(h)
        return h, h_mask


class PConvUNet(nn.Module):
    def __init__(self, layer_size=7):
        super().__init__()
        self.freeze_enc_bn = False
        self.layer_size = layer_size
        self.enc_1 = PCBActiv(3, 64, bn=False, sample='down-7')
        self.enc_2 = PCBActiv(64, 128, sample='down-5')
        self.enc_3 = PCBActiv(128, 256, sample='down-5')
        self.enc_4 = PCBActiv(256, 512, sample='down-3')
        for i in range(4, self.layer_size):
            name = 'enc_{:d}'.format(i + 1)
            setattr(self, name, PCBActiv(512, 512, sample='down-3'))

        for i in range(4, self.layer_size):
            name = 'dec_{:d}'.format(i + 1)
            setattr(self, name, PCBActiv(512 + 512, 512, activ='leaky'))
        self.dec_4 = PCBActiv(512 + 256, 256, activ='leaky')
        self.dec_3 = PCBActiv(256 + 128, 128, activ='leaky')
        self.dec_2 = PCBActiv(128 + 64, 64, activ='leaky')
        self.dec_1 = PCBActiv(64 + 3, 3, bn=False, activ=None)

    def forward(self, input, input_mask):
        h_dict = {}  # for the output of enc_N
        h_mask_dict = {}  # for the output of enc_N

        h_dict['h_0'], h_mask_dict['h_0'] = input, input_mask

        h_key_prev = 'h_0'
        for i in range(1, self.layer_size + 1):
            l_key = 'enc_{:d}'.format(i)
            h_key = 'h_{:d}'.format(i)
            h_dict[h_key], h_mask_dict[h_key] = getattr(self, l_key)(
                h_dict[h_key_prev], h_mask_dict[h_key_prev])
            h_key_prev = h_key

        h_key = 'h_{:d}'.format(self.layer_size)
        h, h_mask = h_dict[h_key], h_mask_dict[h_key]

        # concat upsampled output of h_enc_N-1 and dec_N+1, then do dec_N
        # (exception)
        #                            input         dec_2            dec_1
        #                            h_enc_7       h_enc_8          dec_8

        for i in range(self.layer_size, 0, -1):
            enc_h_key = 'h_{:d}'.format(i - 1)
            dec_l_key = 'dec_{:d}'.format(i)

            h = F.upsample(h, scale_factor=2)
            h_mask = F.upsample(h_mask, scale_factor=2)

            h = torch.cat([h, h_dict[enc_h_key]], dim=1)
            h_mask = torch.cat([h_mask, h_mask_dict[enc_h_key]], dim=1)
            h, h_mask = getattr(self, dec_l_key)(h, h_mask)

        return h, h_mask

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        if self.freeze_enc_bn:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d) and 'enc' in name:
                    module.eval()


if __name__ == '__main__':
    size = (1, 3, 5, 5)
    input = torch.ones(size)
    input_mask = torch.ones(size)
    input_mask[:, :, 2:, :][:, :, :, 2:] = 0

    conv = PConv2d(3, 3, 3, 1, 1)
    l1 = nn.L1Loss()
    input.requires_grad = True

    output, output_mask = conv(input, input_mask)
    loss = l1(output, torch.randn(1, 3, 5, 5))
    loss.backward()

    assert (torch.sum(input.grad != input.grad).item() == 0)
    assert (torch.sum(torch.isnan(conv.conv2d.weight.grad)).item() == 0)
    assert (torch.sum(torch.isnan(conv.conv2d.bias.grad)).item() == 0)

    from IPython import embed

    embed()
    exit()

    # model = PConvUNet()
    # output, output_mask = model(input, input_mask)
