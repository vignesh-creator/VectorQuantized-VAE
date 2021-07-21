import torch.nn as nn
import torch.nn.functional as F
import Resnet

class Decoder(nn.Module):
    def __init__(self, out_channels=128, residual_layers=2, hidden_layers=32):
        super(Decoder, self).__init__()
        
        self.inverse_residual_block = Resnet.ResidualBlocks(out_channels,out_channels,residual_layers,hidden_layers)
        
        self.trans_layer1 = nn.Sequential(
                                    nn.ConvTranspose2d(out_channels,out_channels//2,4,2,1),
                                    nn.ReLU()
        )
        self.trans_layer2 = nn.Sequential(
                                    nn.ConvTranspose2d(out_channels//2,3,4,2,1),
        )

    def forward(self, x):
        x = self.inverse_residual_block(x)
        x = self.trans_layer1(x)
        x = self.trans_layer2(x)
        return x