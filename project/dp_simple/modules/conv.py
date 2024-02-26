import torch
from torch import nn
from einops import rearrange, repeat
# implement 3d convolution 2+1
class Conv21(nn.Module):
    def __init__(self,input_channels,output_channels,dropout,
                 kernel_size,depth,padding,
                 stride,activation='relu'):
        super(Conv21, self).__init__()
        # input should be a 5D tensor (batch,depth,channel,height,width)
        kernel = (1,kernel_size,kernel_size)
        self.conv2d = nn.Conv3d(input_channels,output_channels,kernel_size=kernel,
                                padding=(kernel[0]-1,padding,padding),stride=(1,stride,stride))
        self.conv1d = nn.Conv3d(output_channels,output_channels,kernel_size=(depth,1,1),
                                padding="same",stride=1)
        self.dropout = nn.Dropout3d(dropout)
        if activation == 'relu':
            self.relu = nn.ReLU()
        elif activation == 'linear':
            self.relu = nn.Identity()
        self.bn = nn.InstanceNorm3d(output_channels)
    def forward(self,x):
        x = self.conv2d(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.conv1d(x)
        x = self.dropout(x)

        x = self.bn(x)
        
        return x

class resConv21(nn.Module):
    def __init__(self,input_channels,output_channels,dropout,
                 kernel_size,depth,padding,
                 stride,activation='relu'):
        super(resConv21, self).__init__()
        self.conv21 = Conv21(input_channels,output_channels,dropout,
                             kernel_size,depth,padding,
                             stride)
        if activation == 'relu':
            self.relu = nn.ReLU()
        elif activation == 'linear':
            self.relu = nn.Identity()
    def forward(self,x):
       
        res = x
        x = self.conv21(x) 
       
        x= res+x
        x = self.relu(x)
        return x
    
class C3D(nn.Module):
    def __init__(self,input_channels,output_channels,dropout,
                 kernel_size,depth,padding,
                 stride,activation='relu'):
        super(C3D, self).__init__()
        self.conv = nn.Conv3d(input_channels,output_channels,kernel_size=kernel_size,
                              padding=padding,stride=stride)
        if activation == 'relu':
            self.relu = nn.ReLU()
        elif activation == 'linear':
            self.relu = nn.Identity()
        
        self.bn = nn.InstanceNorm3d(output_channels)
        self.dropout = nn.Dropout3d(dropout)
    def forward(self,x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        x = self.dropout(x)
        return x
class R3D(nn.Module):
    def __init__(self,input_channels,output_channels,dropout,
                 kernel_size,depth,padding,
                 stride,activation='relu'):
        super(R3D, self).__init__()
        self.conv = nn.Conv3d(input_channels,output_channels,kernel_size=kernel_size,
                              padding=padding,stride=stride)
        if activation == 'relu':
            self.relu = nn.ReLU()
        elif activation == 'linear':
            self.relu = nn.Identity()
        self.bn = nn.InstanceNorm3d(output_channels)
        self.dropout = nn.Dropout3d(dropout)
    def forward(self,x):
        x = self.conv(x) + x
        x = self.relu(x)
        x = self.bn(x)
        x = self.dropout(x)
        return x

class upsample(nn.Module):
    def __init__(self,scale_factor,mode):
        super(upsample, self).__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor,mode=mode)
    def forward(self,x):
        b,c,d,h,w = x.shape
        x = rearrange(x,'b c d h w -> b (c d) h w')

        x = self.upsample(x)
        x = rearrange(x,'b (c d) h w -> b c d h w',c=c,d=d)
        
        return x
class downsample(nn.Module):
    def __init__(self,scale_factor,mode):
        super(downsample, self).__init__()
        self.downsample = nn.AvgPool3d(kernel_size=scale_factor)
    def forward(self,x):
        return self.downsample(x)

# selective kernel fusion
class skff(nn.Module):
    def __init__(self,channels):
        super(skff, self).__init__()
        self.conv = nn.Conv3d(channels,channels,kernel_size=3,padding=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
    def forward(self,x):
        x = self.conv(x)
        x = self.sigmoid(x)
        x = x * x
        x = self.relu(x)
        return x




        
        



