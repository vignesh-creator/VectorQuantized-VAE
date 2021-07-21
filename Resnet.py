import loader
import torch.nn as nn
import torch.nn.functional as F
device = loader.device

# This Resnet structure is taken from Author's Implementation
class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super(Residual, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels,hidden_channels,kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden_channels,out_channels,kernel_size=1, stride=1, bias=False),
            nn.ReLU(),
        )
    
    def forward(self, x):
        x = x + self.block(x)
        return x


class ResidualBlocks(nn.Module):
    def __init__(self, in_channels,out_channels, residual_layers, hidden_channels):
        super(ResidualBlocks, self).__init__()
        self.residual_layers = residual_layers
        self.layers = nn.ModuleList(
            [
                Residual(in_channels,out_channels,hidden_channels)
                             for _ in range(self.residual_layers)
                ]
            )

    def forward(self, x):
        for i in range(self.residual_layers):
            x = self.layers[i](x)
            x = F.relu(x)
        return x
