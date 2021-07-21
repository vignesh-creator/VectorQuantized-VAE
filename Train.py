import loader
import Encoder_Network
import Decoder_Network
import torch
import torch.optim
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import Quantize
import matplotlib.pyplot as plt

'''
You can simply run Train to get Reconstruction loss and codebook loss and their plots.But if you want
to verify testloss and see test reconstructions, run the test code(Test code trains the Network and performs test Operation)
'''

# Building Encoder network
device = loader.device
batch_size = loader.batch_size

class Network(nn.Module):
    def __init__(self,num_embeddings, embedding_dim,in_channels=3):
        super(Network, self).__init__()
        
        self.encoder = Encoder_Network.Encoder(in_channels)

        self.hidden1 = nn.Conv2d(128, embedding_dim,1,1)
        
        self.vq = Quantize.VectorQuantization(num_embeddings, embedding_dim)

        self.hidden2 = nn.Conv2d(embedding_dim,128,3,1,1)

        self.decoder = Decoder_Network.Decoder()
        
    def forward(self, x):
        x = self.encoder(x) #outputs an image of size (batch x 128 x original_width x original_height) 

        x = self.hidden1(x) #outputs an image of size (batch x 64 x width x height)

        quantized_vector,vq_loss= self.vq(x) #No change in size of image

        x = self.hidden2(quantized_vector) #outputs an image of size (batch x 128 x width x height)

        x = self.decoder(x) #outputs image with dimensions of original image

        return x,vq_loss


''' The size of codebook dimension that we initialised in this Network is (512,64). We are following the Author's 
    Hyperparameters throughout the Network. You can change the dimensions as required
'''
AutoEncoder = Network(512,64)
optimizer = torch.optim.Adam(AutoEncoder.parameters(), lr=1e-3)


def train(X):
    recon_list = []
    vq_loss_list = []
    for i in range(0,loader.train_iterations):
        (data,label) = next(iter(X))
        data = data.to(device)
        optimizer.zero_grad()
        output, vq_loss = AutoEncoder(data)
        recon_loss = F.mse_loss(output,data)
        loss = recon_loss + vq_loss
        loss.backward()
        optimizer.step()
        recon_list.append(recon_loss.item())
        vq_loss_list.append(vq_loss.item())
        if (i+1) % 2 == 0:
            print('for iteration ',i+1,',Reconstruction loss is:', np.mean(recon_list[0:]))
    return recon_list,vq_loss_list


training_loader = loader.training_loader
Recon_loss,vq_loss = train(training_loader)


#visualisation 
Recon_loss =  Recon_loss
vq_loss = vq_loss

#plotting Recon Loss
plt.plot(Recon_loss)
plt.title("Reconstruction Loss")
plt.ylabel('cost')
plt.xlabel('epoch')
plt.show()

#plotting vq-loss
plt.plot(vq_loss)
plt.title("VQ Loss")
plt.ylabel('cost')
plt.xlabel('epoch')
plt.show()

