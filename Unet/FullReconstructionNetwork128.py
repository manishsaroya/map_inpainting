import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from net import PConv2d
import PIL
from matplotlib import pyplot
from loadMasks import CustomMasks


class Net(nn.Module):

    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = PConv2d(1, 64, 7, 2, 3)
        self.conv2 = PConv2d(64, 128, 5, 2, 2)
        self.conv3 = PConv2d(128, 256, 5, 2, 2)
        self.conv4 = PConv2d(256, 512, 3, 2, 1)
        self.conv5 = PConv2d(512, 512, 3, 2, 1)
        self.conv6 = PConv2d(512, 512, 3, 2, 1)
        #self.conv7 = PConv2d(512, 512, 3, 2, 1)
        #self.conv8 = PConv2d(512, 512, 3, 2, 1)

        self.conv8_1 = PConv2d(1024, 512, 3, 1, 1)
        self.conv8_2 = PConv2d(512, 512, 3, 1, 1)
        self.conv9 = PConv2d(768, 256, 3, 1, 1)
        self.conv10 = PConv2d(384, 128, 3, 1, 1)
        self.conv11 = PConv2d(192, 64, 3, 1, 1)
        self.conv12 = PConv2d(65, 1, 3, 1, 1)

        self.batchNorm1 = nn.BatchNorm2d(64).cuda()
        self.batchNorm2 = nn.BatchNorm2d(128).cuda()
        self.batchNorm3 = nn.BatchNorm2d(256).cuda()
        self.batchNorm4 = nn.BatchNorm2d(512).cuda()
        self.batchNorm8_1 = nn.BatchNorm2d(512).cuda()
        self.batchNorm9 = nn.BatchNorm2d(256).cuda()
        self.batchNorm10 = nn.BatchNorm2d(128).cuda()
        self.batchNorm11 = nn.BatchNorm2d(64).cuda()

        self.Relu = nn.ReLU()
        self.leakyRelu = nn.LeakyReLU(negative_slope=0.2)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)

        self.output_mask = []

    def forward(self, input, input_mask):


        conv1_output, conv1_output_mask = self.conv1(input, input_mask)
        batchNorm1_output = self.batchNorm1(conv1_output)
        # print(batchNorm1_output.shape)
        relu1_output =  self.Relu(batchNorm1_output)
        # 256x256
        conv2_output, conv2_output_mask =  self.conv2(relu1_output, conv1_output_mask)
        batchNorm2_output =  self.batchNorm2(conv2_output)
        # print(conv2_output.shape)
        relu2_output =  self.Relu(batchNorm2_output)
        # 128x128
        conv3_output, conv3_output_mask =  self.conv3(relu2_output, conv2_output_mask)
        batchNorm3_output =  self.batchNorm3(conv3_output)
        # print(conv3_output.shape)
        relu3_output =  self.Relu(batchNorm3_output)
        # 64x64
        conv4_output, conv4_output_mask =  self.conv4(relu3_output, conv3_output_mask)
        batchNorm4_output =  self.batchNorm4(conv4_output)
        # print(conv4_output.shape)
        relu4_output =  self.Relu(batchNorm4_output)
        # 32x32
        conv5_output, conv5_output_mask =  self.conv5(relu4_output, conv4_output_mask)
        batchNorm5_output =  self.batchNorm4(conv5_output)
        # print(conv5_output.shape)
        relu5_output =  self.Relu(batchNorm5_output)
        # 16x16
        conv6_output, conv6_output_mask =  self.conv6(relu5_output, conv5_output_mask)
        batchNorm6_output =  self.batchNorm4(conv6_output)
        # print(conv6_output.shape)
        relu6_output =  self.Relu(batchNorm6_output)
        # 8x8
        #conv7_output, conv7_output_mask =  self.conv7(relu6_output, conv6_output_mask)
        #batchNorm7_output =  self.batchNorm4(conv7_output)
        # print(conv7_output.shape)
        #relu7_output =  self.Relu(batchNorm7_output)
        # 4x4

        #upsample2 =  self.upsample(relu6_output)
        #upsample2_mask =  self.upsample(relu6_output)
        #concat2 = torch.cat((upsample2, relu6_output), 1)
        #concat2_mask = torch.cat((upsample2_mask, conv6_output_mask), 1)
        #conv10_output, conv10_output_mask =  self.conv8_1(concat2, concat2_mask)
        #batchNorm10_output =  self.batchNorm8_1(conv10_output)
        #leakyRelu2 =  self.leakyRelu(batchNorm10_output)
        # print(conv10_output.shape)

        upsample3 =  self.upsample(relu6_output)
        upsample3_mask =  self.upsample(conv6_output_mask)
        concat3 = torch.cat((upsample3, relu5_output), 1)
        concat3_mask = torch.cat((upsample3_mask, conv5_output_mask), 1)
        conv11_output, conv11_output_mask =  self.conv8_1(concat3, concat3_mask)
        batchNorm11_output =  self.batchNorm8_1(conv11_output)
        leakyRelu3 =  self.leakyRelu(batchNorm11_output)
        # print(conv11_output.shape)

        upsample4 =  self.upsample(leakyRelu3)
        upsample4_mask =  self.upsample(conv11_output_mask)
        concat4 = torch.cat((upsample4, relu4_output), 1)
        concat4_mask = torch.cat((upsample4_mask, conv4_output_mask), 1)
        conv12_output, conv12_output_mask =  self.conv8_1(concat4, concat4_mask)
        batchNorm12_output =  self.batchNorm8_1(conv12_output)
        leakyRelu4 =  self.leakyRelu(batchNorm12_output)
        # print(conv12_output.shape)

        upsample5 =  self.upsample(leakyRelu4)
        upsample5_mask =  self.upsample(conv12_output_mask)
        concat5 = torch.cat((upsample5, relu3_output), 1)
        concat5_mask = torch.cat((upsample5_mask, conv3_output_mask), 1)
        conv13_output, conv13_output_mask =  self.conv9(concat5, concat5_mask)
        batchNorm13_output =  self.batchNorm9(conv13_output)
        leakyRelu5 =  self.leakyRelu(batchNorm13_output)
        # print(conv13_output.shape)

        upsample6 =  self.upsample(leakyRelu5)
        upsample6_mask =  self.upsample(conv13_output_mask)
        concat6 = torch.cat((upsample6, relu2_output), 1)
        concat6_mask = torch.cat((upsample6_mask, conv2_output_mask), 1)
        conv14_output, conv14_output_mask =  self.conv10(concat6, concat6_mask)
        batchNorm14_output =  self.batchNorm10(conv14_output)
        leakyRelu6 =  self.leakyRelu(batchNorm14_output)
        # print(conv14_output.shape)

        upsample7 =  self.upsample(leakyRelu6)
        upsample7_mask =  self.upsample(conv14_output_mask)
        concat7 = torch.cat((upsample7, relu1_output), 1)
        concat7_mask = torch.cat((upsample7_mask, conv1_output_mask), 1)
        conv15_output, conv15_output_mask =  self.conv11(concat7, concat7_mask)
        batchNorm15_output =  self.batchNorm11(conv15_output)
        leakyRelu7 =  self.leakyRelu(batchNorm15_output)
        # print(conv15_output.shape)

        upsample8 =  self.upsample(leakyRelu7)
        upsample8_mask =  self.upsample(conv15_output_mask)
        concat8 = torch.cat((upsample8, input), 1)
        concat8_mask = torch.cat((upsample8_mask, input_mask), 1)
        conv16_output, conv16_output_mask =  self.conv12(concat8, concat8_mask)
        leakyRelu8 =  self.leakyRelu(conv16_output)

        self.output_mask = conv16_output_mask

        return leakyRelu8

