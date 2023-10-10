import torch
import torch.nn as nn

class down_block(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,stride=1,padding=1):
        super(down_block,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU(inplace=True)
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class attention_net(nn.Module):
    def __init__(self,channel):
        super(attention_net,self).__init__()
        self.conv_enc = nn.Conv2d(channel, channel, kernel_size=1)
        self.conv_dec = nn.Conv2d(channel, channel, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channel, 1, kernel_size=1)
        self.conv3 = nn.Conv2d(1, 1, kernel_size=1)
        self.softmax = nn.Softmax2d()
        self.conv4 = nn.Conv2d(1, channel, kernel_size=1)
    def forward(self,encoded,decoded):
        encoded = self.conv_enc(encoded)
        decoded = self.conv_dec(decoded)
        x = torch.add(encoded,decoded)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.softmax(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = torch.mul(x,encoded)
        x = self.relu(x)
        return x


class up_block(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=2,stride=2,padding=0):
        super(up_block,self).__init__()
        self.trans = nn.ConvTranspose2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding,output_padding=0)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.4)
    def forward(self,x):
        x = self.trans(x)
        x = self.bn(x)
        x = self.dropout(x)
        return x 


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels,act='relu',padding = 1):
        super().__init__()
        assert act.lower() in ['relu', 'none'], f'expected activation to be either relu or none got {act} instead.'
        self.act = act
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2,padding=padding)
        self.relu = nn.LeakyReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        if self.act =='relu':
            return self.relu(x)
        else:
            return x


class Unet(nn.Module):
    #the Unet model, works with any input shape 2^n x 2^n
    def __init__(self,channels=[3,32,64,128,256,512]):
        super(Unet,self).__init__()
        self.conv1= down_block(channels[0],channels[1])
        # self.conv1_2= down_block(channels[1],channels[1])

        self.maxpool= nn.MaxPool2d(kernel_size=(2,2),stride=2)
        self.conv2= down_block(channels[1],channels[2])
        # self.conv2_2= down_block(channels[2],channels[2])

        self.conv3= down_block(channels[2],channels[3])
        # self.conv3_2= down_block(channels[3],channels[3])

        self.conv4= down_block(channels[3],channels[4])
        # self.conv4_2= down_block(channels[4],channels[4])

        self.conv5 = down_block(channels[4],channels[5])
        # self.conv5_2 = down_block(channels[5],channels[5])

        self.up1 = up_block(channels[5],channels[4])
        self.up2 = up_block(channels[4],channels[3])
        self.up3 = up_block(channels[3],channels[2])
        self.up4 = up_block(channels[2],channels[1])
        self.up5 = up_block(channels[1],channels[0])
        self.upconv1 = down_block(channels[5],channels[4])
        self.upconv2 = down_block(channels[4],channels[3])
        self.upconv3 = down_block(channels[3],channels[2])
        self.upconv4 = down_block(channels[2],channels[1])
        self.upconv5 = nn.Conv2d(channels[1],5,kernel_size=1)
        self.att1 = attention_net(channels[4])
        self.att2 = attention_net(channels[3])
        self.att3 = attention_net(channels[2])
        self.att4 = attention_net(channels[1])
        self.relu=nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        out1 = self.conv1(x)
        state1 = self.maxpool(out1)

        out2 = self.conv2(state1)
        state2 = self.maxpool(out2)

        out3 = self.conv3(state2)
        state3 = self.maxpool(out3)

        out4 = self.conv4(state3)
        state4 = self.maxpool(out4
                              )
        out5 = self.conv5(state4)
        state5 = self.up1(out5)

        # out6 = self.upconv1(torch.cat([self.att1(state5, out4), out4], dim=1))
        out6 = self.upconv1(torch.cat([state5, out4], dim=1))

        state6 = self.up2(out6)
        # out7 = self.upconv2(torch.cat([self.att2(state6, out3), out3], dim=1))
        out7 = self.upconv2(torch.cat([state6, out3], dim=1))

        state7 = self.up3(out7)
        # out8 = self.upconv3(torch.cat([self.att3(state7,out2), out2], dim=1))
        out8 = self.upconv3(torch.cat([state7, out2], dim=1))

        state8 = self.up4(out8)
        # out9 = self.upconv4(torch.cat([self.att4(state8,out1), out1], dim=1))
        out9 = self.upconv4(torch.cat([state8, out1], dim=1))

        outputs = self.upconv5(out9)
      
        return outputs