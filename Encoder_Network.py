import torch.nn as nn
import torch.nn.functional as F
import Resnet

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels = 128, hidden_layers=32,residual_layers=2):
        super(Encoder, self).__init__()

        self.layer_1 = nn.Sequential(
                            nn.Conv2d(in_channels,out_channels//2,4,2,1),
                            nn.ReLU()
                        
        )
        self.layer_2 = nn.Sequential(
                            nn.Conv2d(out_channels//2,out_channels,4,2,1),
                            nn.ReLU()
        )
        self.layer_3 = nn.Sequential(
                            nn.Conv2d(out_channels,out_channels,3,1,1),
                            nn.ReLU()
        )
        self.residual_block = Resnet.ResidualBlocks(out_channels,out_channels,residual_layers,hidden_layers)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.residual_block(x)
        return x