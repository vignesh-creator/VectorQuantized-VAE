from torch._C import device
import Train
import loader
import numpy as np
import matplotlib.pyplot as plt
import torchvision
device = loader.device

'''
Run Test.py to train the Net and view test set reconstructions
'''

def display(X):
    X = X.numpy()
    fig = plt.imshow(np.transpose(X, (1,2,0)), interpolation='nearest')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.show()

#Checking reconstructions on Test set
(test_img,Y) = next(iter(loader.validation_loader))
test_img = test_img.to(device)
output = Train.AutoEncoder(test_img)

display(torchvision.utils.make_grid(test_img.cpu()+0.5), )
display(torchvision.utils.make_grid(output[0].cpu().data)+0.5,)