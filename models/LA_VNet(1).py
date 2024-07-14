import torch
import torch.nn as nn
import torch.nn.functional as F

from .sync_batchnorm import SynchronizedBatchNorm3d





class LACBH(nn.Module):
    def __init__(self, in_channels, out_channel):
        super(LACBH, self).__init__()

        self.in_chns = in_channels
        self.out_chns = out_channel

        self.conv1 = nn.Sequential(
            nn.Conv3d(self.in_chns, self.out_chns, kernel_size=(1, 1, 3),
                      padding=(0, 0, 1)),
            nn.BatchNorm3d(self.out_chns),
            nn.ReLU6(inplace=True)
        )

        self.conv2_1 = nn.Sequential(
            nn.Conv3d(self.out_chns, self.out_chns, kernel_size=(3, 3, 1),
                      padding=(1, 1, 0), groups=1),
            nn.BatchNorm3d(self.out_chns),
            nn.ReLU6(inplace=True),
            nn.Dropout(p=0.2)
        )

        self.conv2_2 = nn.Sequential(
            nn.AvgPool3d(kernel_size=4, stride=2, padding=1),
            nn.Conv3d(self.out_chns, self.out_chns, kernel_size=1,
                      padding=0),
            nn.BatchNorm3d(self.out_chns),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        )

    def forward(self, x):

        x = self.conv1(x)

        x1 = self.conv2_1(x)
        x2 = self.conv2_2(x)
        x2 = torch.sigmoid(x2)
        x = x1 + x2 * x
        return x


class LACBL(nn.Module):
    def __init__(self, in_channels, out_channel):
        super(LACBL, self).__init__()

        self.in_chns = in_channels
        self.out_chns = out_channel

        self.conv1 = nn.Sequential(
            nn.Conv3d(self.in_chns, self.out_chns, kernel_size=(1, 1, 3),
                      padding=(0, 0, 1)),
            nn.BatchNorm3d(self.out_chns),
            nn.ReLU6(inplace=True)
        )

        self.conv2_1 = nn.Sequential(
            nn.Conv3d(self.out_chns, self.out_chns, kernel_size=(3, 3, 1),
                      padding=(1, 1, 0), groups=self.out_chns),
            nn.BatchNorm3d(self.out_chns),
            nn.ReLU6(inplace=True),
            nn.Dropout(p=0.2)
        )

        self.conv2_2 = nn.Sequential(
            # nn.AvgPool3d(kernel_size=4, stride=2),
            nn.Conv3d(self.out_chns, self.out_chns, kernel_size=1,
                      padding=0),
            nn.BatchNorm3d(self.out_chns),
            nn.ReLU6(inplace=True),
            nn.Dropout(p=0.2)
        )

    def forward(self, x):
        x = self.conv1(x)

        x1 = self.conv2_1(x)
        x2 = self.conv2_2(x)
        x2 = torch.sigmoid(x2)
        x = x1 + x2 * x
        return x





class InitialConv(nn.Module):
    def __init__(self, out_channels=8):
        super().__init__()
        self.out_channels = out_channels

        self.conv = nn.Conv3d(in_channels=1, out_channels=out_channels, kernel_size=5, stride=1, padding=2)
        self.bn   = nn.BatchNorm3d(out_channels)
        self.conv_down = nn.Conv3d(in_channels=out_channels, out_channels=out_channels*2, kernel_size=2, stride=2, padding=0)
        self.bn_down = nn.BatchNorm3d(out_channels*2)

    def forward(self, x):
        layer = F.relu(self.bn(self.conv(x)))
        layer = torch.add(layer, torch.cat([x[:,0:1]]*self.out_channels, 1))

        conv = F.relu(self.bn_down(self.conv_down(layer)))
        return layer, conv


class DownConvBlock2b(nn.Module):
    def __init__(self, out_channels=16):
        super().__init__()
        self.out_channels = out_channels

        self.conv_a = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2)
        self.bn_a = nn.BatchNorm3d(out_channels)
        self.conv_b = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2)
        self.bn_b = nn.BatchNorm3d(out_channels)
        self.conv_down = nn.Conv3d(in_channels=out_channels, out_channels=out_channels * 2, kernel_size=2, stride=2, padding=0)
        self.bn_down = nn.BatchNorm3d(out_channels * 2)

    def forward(self, x):
        layer = F.relu(self.bn_a(self.conv_a(x)))
        layer = F.relu(self.bn_b(self.conv_b(layer)))
        layer = torch.add(layer, x)

        conv = F.relu(self.bn_down(self.conv_down(layer)))
        return layer, conv


class UpConvBlock2b(nn.Module):
    def __init__(self, out_channels=32, undersampling_factor=4):
        super().__init__()
        self.out_channels = out_channels
        self.undersampling_factor = undersampling_factor

        self.conv_a = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2)
        self.bn_a = nn.BatchNorm3d(out_channels)
        self.conv_b = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2)
        self.bn_b = nn.BatchNorm3d(out_channels)
        self.conv_up = nn.ConvTranspose3d(in_channels=out_channels, out_channels=out_channels // undersampling_factor, kernel_size=2, stride=2, padding=0)
        self.bn_up = nn.BatchNorm3d(out_channels // undersampling_factor)

    def forward(self, x):
        layer = F.relu(self.bn_a(self.conv_a(x)))
        layer = F.relu(self.bn_b(self.conv_b(layer)))
        layer = torch.add(layer, x)

        conv = F.relu(self.bn_up(self.conv_up(layer)))
        return layer, conv


class DownConvBlock3b(nn.Module):
    def __init__(self, out_channels=32):
        super().__init__()
        self.out_channels = out_channels

        self.conv_a = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2)
        self.bn_a = nn.BatchNorm3d(out_channels)
        self.conv_b = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2)
        self.bn_b = nn.BatchNorm3d(out_channels)
        self.conv_c = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2)
        self.bn_c = nn.BatchNorm3d(out_channels)
        self.conv_down = nn.Conv3d(in_channels=out_channels, out_channels=out_channels * 2, kernel_size=2, stride=2, padding=0)
        self.bn_down = nn.BatchNorm3d(out_channels * 2)

    def forward(self, x):
        layer = F.relu(self.bn_a(self.conv_a(x)))
        layer = F.relu(self.bn_b(self.conv_b(layer)))
        layer = F.relu(self.bn_c(self.conv_c(layer)))
        layer = torch.add(layer, x)

        conv = F.relu(self.bn_down(self.conv_down(layer)))
        return layer, conv


class UpConvBlock3b(nn.Module):
    def __init__(self, out_channels=128, undersampling_factor=2):
        super().__init__()
        self.out_channels = out_channels

        self.conv_a = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2)
        self.bn_a = nn.BatchNorm3d(out_channels)
        self.conv_b = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2)
        self.bn_b = nn.BatchNorm3d(out_channels)
        self.conv_c = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2)
        self.bn_c = nn.BatchNorm3d(out_channels)
        self.conv_up = nn.ConvTranspose3d(in_channels=out_channels, out_channels=out_channels // undersampling_factor, kernel_size=2, stride=2, padding=0)
        self.bn_up = nn.BatchNorm3d(out_channels // undersampling_factor)

    def forward(self, x):
        layer = F.relu(self.bn_a(self.conv_a(x)))
        layer = F.relu(self.bn_b(self.conv_b(layer)))
        layer = F.relu(self.bn_c(self.conv_c(layer)))
        layer = torch.add(layer, x)

        conv = F.relu(self.bn_up(self.conv_up(layer)))
        return layer, conv


class FinalConv(nn.Module):
    def __init__(self, num_outs=2, out_channels=32):
        super().__init__()
        self.conv = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2)
        self.bn = nn.BatchNorm3d(out_channels)
        self.conv_1x1 = nn.Conv3d(in_channels=out_channels, out_channels=num_outs, kernel_size=1, stride=1, padding=0)
        self.bn_1x1 = nn.BatchNorm3d(num_outs)
        self.final = F.softmax
    def forward(self, x):
        layer = F.relu(self.bn(self.conv(x)))
        layer = torch.add(layer, x)
        layer = self.bn_1x1(self.conv_1x1(layer))
        layer = self.final(layer, dim=1)
        return layer


class CatBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        cat = torch.cat((x1,x2), 1)
        return cat




class hd(nn.Module):
    def __init__(self, num_in, num_out):
        super(hd, self).__init__()
        self.conv = nn.Conv3d(num_in, num_out, kernel_size=3, stride=1, padding=1, bias=True)
        self.norm = SynchronizedBatchNorm3d(num_out)
        self.conv1x1 = nn.Conv3d(num_in, num_out, kernel_size=1, stride=1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x, y):
        out = F.relu(self.norm(self.conv(y)))
        y = F.relu(self.conv1x1(y))
        y = out + y
        y = self.sigmoid(y)
        s = x * y
        l = s + x
        return l




















class VNet3D(nn.Module):

    def __init__(self, num_outs=2, channels=8):

        super(VNet3D, self).__init__()

        self.init_conv  = InitialConv(out_channels=channels)

        self.LACBH1 = LACBH(channels*2, channels*2)
        self.down_block_1 = DownConvBlock2b(out_channels=channels * 2)
        
        self.LACBH2 = LACBH(channels*4, channels*4)
        self.down_block_2 = DownConvBlock3b(out_channels=channels * 4)
        
        self.LACBH3 = LACBH(channels*8, channels*8)
        self.down_block_3 = DownConvBlock3b(out_channels=channels * 8)



        self.LACBL = LACBL(channels*16, channels*16)
        self.up_block_4   = UpConvBlock3b(out_channels=channels * 16, undersampling_factor=2)
        self.cat_block_4  = CatBlock()
        
        self.LACBH4 = LACBH(channels*16, channels*16)
        self.up_block_3   = UpConvBlock3b(out_channels=channels * 16, undersampling_factor=4)
        self.cat_block_3  = CatBlock()
        
        self.LACBH5 = LACBH(channels*8, channels*8)
        self.up_block_2   = UpConvBlock3b(out_channels=channels * 8, undersampling_factor=4)
        self.cat_block_2  = CatBlock()
        
        self.LACBH6 = LACBH(channels*4, channels*4)
        self.up_block_1   = UpConvBlock2b(out_channels=channels * 4, undersampling_factor=4)
        self.cat_block_1  = CatBlock()

        #self.out_conv     = FinalConv(num_outs=num_outs, out_channels=channels * 2)
        self.out_conv     = FinalConv(num_outs=num_outs, out_channels=channels )


        #attention
        self.hd = hd(8, 8)

    def forward(self, x):

        layer_down_1, conv_down_1 = self.init_conv(x) #torch.Size([4, 16, 112, 128, 88])
        lacbh_conv_down_1 = self.LACBH1(conv_down_1)
        layer_down_2, conv_down_2 = self.down_block_1(lacbh_conv_down_1) #torch.Size([4, 32, 56, 64, 44]) 
        lacbh_conv_down_2 = self.LACBH2(conv_down_2)
        layer_down_3, conv_down_3 = self.down_block_2(lacbh_conv_down_2) #torch.Size([4, 64, 28, 32, 22])
        lacbh_conv_down_3 = self.LACBH3(conv_down_3)
        layer_down_4, conv_down_4 = self.down_block_3(lacbh_conv_down_3) #torch.Size([4, 128, 14, 16, 11])
        lacbl_conv_down_4 = self.LACBL(conv_down_4)

        layer_up_4, conv_up_4 = self.up_block_4(lacbl_conv_down_4) #torch.Size([4, 64, 28, 32, 22])
        cat4 = self.cat_block_4(conv_up_4, layer_down_4) #torch.Size([4, 128, 28, 32, 22])

        lacbh_cat4 = self.LACBH4(cat4)
        layer_up_3, conv_up_3 = self.up_block_3(lacbh_cat4) #torch.Size([4, 32, 56, 64, 44])
        cat3 = self.cat_block_3(conv_up_3, layer_down_3)

        lacbh_cat3 = self.LACBH5(cat3)
        layer_up_2, conv_up_2 = self.up_block_2(lacbh_cat3) #torch.Size([4, 16, 112, 128, 88])
        cat2 = self.cat_block_2(conv_up_2, layer_down_2)

        lacbh_cat2 = self.LACBH6(cat2)
        layer_up_1, conv_up_1 = self.up_block_1(lacbh_cat2) #torch.Size([4, 8, 224, 256, 176])
        #cat1 = self.cat_block_1(conv_up_1, layer_down_1)#torch.Size([4, 8, 224, 256, 176]) torch.Size([4, 8, 224, 256, 176])#torch.Size([4, 16, 224, 256, 176])
        
        cat1 = self.hd(conv_up_1, layer_down_1)
        layer_out = self.out_conv(cat1)#torch.Size([1, 3, 224, 256, 176])

        return layer_out

