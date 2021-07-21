import torch
import loader
import torch.nn as nn
import torch.nn.functional as F
device = loader.device

class VectorQuantization(nn.Module):
    def __init__(self, number_embeddings, embedding_dimension):
        super(VectorQuantization, self).__init__()
        
        self.num_embeddings = number_embeddings
        self.embedding_dim = embedding_dimension
        
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim).to(device)
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, x):
        #Convert (Batch,channel,Height,Width) to (Batch,Height,Width,Channel) so it's easy to flatten it later
        x = x.view(x.shape[0],x.shape[2],x.shape[3],x.shape[1])

        # Flatten input
        latent = x.view(-1, self.embedding_dim)

        # Calculate the L2 Norm between latent and Embedded weights
        distances = (torch.sum(latent**2, dim=1, keepdim=True) + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(latent, self.embedding.weight.t()))
        
        # Encoding starts from here
        encoded_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        
        # One Hot Encoding starts from here
        encoded_vector = F.one_hot(encoded_indices, self.num_embeddings).type(x.dtype)
        encoded_vector = encoded_vector.view(encoded_vector.shape[0],encoded_vector.shape[2])
        
        # Quantization of the Encodings
        quantized_vector = torch.matmul(encoded_vector, self.embedding.weight)

        #Bringing back to normal Shape
        quantized_vector = quantized_vector.view(x.shape)
        
        
        # Loss
        beta = 0.25 #Its a Hyperparameter
        commitment_loss = F.mse_loss(quantized_vector.detach(), x)
        codebook_loss = F.mse_loss(quantized_vector, x.detach())
        vq_loss = codebook_loss + beta * commitment_loss
        
        ''' The codebook cannot undergo backpropagation.So, In backprop, we need to go to encoder after completing the
            backprop in decoder without disturbing the blackbox(the VQ method). This can be done with the below code line 
        '''

        quantized_vector = x + (quantized_vector - x).detach()

        q_shape = quantized_vector.shape

        quantized_vector = quantized_vector.view(q_shape[0],q_shape[3],q_shape[1],q_shape[2])

        return quantized_vector,vq_loss