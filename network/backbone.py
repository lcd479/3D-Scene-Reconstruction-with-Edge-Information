from .SwinTransformer import *
from .Swin_utils import *

import torch
import torch.nn.functional as F
import torch.nn as nn

from siren_pytorch import *

class backbone(nn.Module):
    
    def __init__(self, img_size = (640,480)):
        super(backbone, self).__init__()
        
        self.img_size = img_size
        self.Backbone_Swin = Backbone_Swin(img_size= (640,480), embed_dim=128,depths=[2,2,18,2], 
                                 num_heads=[4,8,16,32], drop_path_rate=0.2, window_size=12,use_checkpoint=True)
        self.Backbone_Swin.load_from('./swin_base_patch4_window12_384_22k.pth')        
        
        in_f = [1024, 512, 256,128]
        out_f = [80, 40,24]
        
        self.SNet0  = SirenNet(dim_in = in_f[3], dim_hidden=256, dim_out= out_f[2], num_layers = 5,w0= 1.0, w0_initial=30.0, final_activation= None)
        self.SNet1  = SirenNet(dim_in = in_f[2], dim_hidden=256, dim_out= out_f[1], num_layers = 5,w0= 1.0, w0_initial=30.0, final_activation= None)
        self.SNet2  = SirenNet(dim_in = in_f[1], dim_hidden=256, dim_out= out_f[0], num_layers = 5,w0= 1.0, w0_initial=30.0, final_activation= None)

        self.inner0 = nn.Conv2d(out_f[1], out_f[0], 1, bias = True)
        self.inner1 = nn.Conv2d(out_f[2], out_f[0], 1, bias = True)

        self.conv0 = nn.Conv2d(in_f[3], out_f[2], 1, bias = False)
        self.conv1 = nn.Conv2d(in_f[2], out_f[1], 1, bias = False)
        self.conv2 = nn.Conv2d(in_f[1], out_f[0], 1, bias = False)
        self.conv3 = nn.Conv2d(in_f[0], out_f[0], 1, bias = False)
        
        self.out0 = nn.Conv2d(out_f[0], out_f[0], 1, bias = False)
        self.out1 = nn.Conv2d(out_f[0], out_f[1], 1, bias = False)
        self.out2 = nn.Conv2d(out_f[0], out_f[2], 1, bias = False)
        
    def forward(self, x): 
        
        B = x.shape[0]
        out_list = self.Backbone_Swin(x)
        
        S_out0 = self.SNet0(out_list[0].view(B, 128,-1).permute(0,2,1)).permute(0,2,1)
        S_out1 = self.SNet1(out_list[1].view(B, 256,-1).permute(0,2,1)).permute(0,2,1)
        S_out2 = self.SNet2(out_list[2].view(B, 512,-1).permute(0,2,1)).permute(0,2,1)
    

        T_out0 = S_out0.view(B, 24, 120,160)
        T_out1 = S_out1.view(B, 40, 60, 80)
        T_out2 = S_out2.view(B, 80, 30, 40)

        outputs = []
        intra_feat = T_out2 
        out = self.out0(intra_feat)
        outputs.append(out)
     
        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode='nearest') + self.inner0(T_out1)
        out = self.out1(intra_feat)
        outputs.append(out)
        
        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode='nearest') + self.inner1(T_out0)
        out = self.out2(intra_feat)
        outputs.append(out)


        return outputs[::-1]
