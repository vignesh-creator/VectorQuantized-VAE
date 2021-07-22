# Vector Quantized Variational Autoencoder

This Repo contains pyTorch Implementation of the paper [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) .
As of now, VectorQuantization is done. VectorQuantizedEMA method is not yet Implemented but will be done soon.
And also,will try to pair the Network with an Autoregressive Model soon.

If you want to Train the model,plot Loss and view the Test reconstructions,then use command
> python test.py

If you just want to train the model and view train Loss plots, then use
> python train.py

The hyperparameters of the Model and Network Architecture is taken from author's Implementation.

### The Reconstruction Loss

![](https://i.imgur.com/TtpF5I1.png)


### The Codebook Loss
![](https://i.imgur.com/jqbrMHG.png)

### Original and Reconstructed Images
![](https://i.imgur.com/vBeGsMr.jpg)


### After 3000 Training iterations
![](https://i.imgur.com/ZRHqOHm.jpg)





