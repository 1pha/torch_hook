import numpy as np
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def one_sided_padding(x):
    rand1 = random.randrange(0,15,3)
    rand2 = random.randrange(0,15,3)

    zero = np.zeros(shape=[28,28,1])
    zero[rand1:rand1+12,rand2:rand2+12,:]=np.asarray(x).reshape(12,12,1)
    return zero

def get_dataloader(batch_size=128):

    mnist_train = dset.MNIST("./", train=True, 
                            transform=transforms.Compose([
                                transforms.RandomCrop(22),
                                transforms.Resize(12),
                                transforms.Lambda(one_sided_padding),
                                transforms.ToTensor(),
                            ]), 
                            target_transform=None, 
                            download=False)

    mnist_test = dset.MNIST("./", train=False,
                            transform=transforms.Compose([
                                transforms.RandomCrop(22),
                                transforms.Resize(12),
                                transforms.Lambda(one_sided_padding),
                                transforms.ToTensor(),
                            ]),
                            target_transform=None, 
                            download=False)

    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_loader, test_loader