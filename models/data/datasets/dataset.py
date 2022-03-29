import os
import torch
import torchvision
from torchvision import transforms
import torchvision.datasets as datasets
from torch.utils.data.distributed import DistributedSampler


class Dataset:
    def __init__(self, img_size, batch_size, workers, isDistributed=False, data_dir=None, MNIST=True):
        self.img_size = img_size
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.workers = workers
        self.isDistributed = isDistributed
        self.MNIST = MNIST

    def dataloader(self):
        if self.MNIST:
            normalize = transforms.Normalize((0.1307,), (0.3081,))

            train_dataset = torchvision.datasets.MNIST(root=self.data_dir,
                                                        train=True,
                                                        transform=transforms.Compose([transforms.RandomResizedCrop(self.img_size),
                                                                                        transforms.RandomHorizontalFlip(),
                                                                                        transforms.ToTensor(),
                                                                                        normalize,]),
                                                        download=True)

            val_dataset = torchvision.datasets.MNIST(root=self.data_dir, train=False,
                                                                transform=transforms.Compose([
                                                                    transforms.Resize(256),
                                                                    transforms.CenterCrop(self.img_size),
                                                                    transforms.ToTensor(),
                                                                    normalize,]),
                                                                download=True)
        else:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            train_dataset = datasets.ImageFolder(
                                    os.path.join(self.data_dir, 'train'),
                                    transforms.Compose([
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize,
                                    ]))

            val_dataset = datasets.ImageFolder(
                                    os.path.join(self.data_dir, 'val'),
                                    transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        normalize,
                                    ]))

        train_sampler = None
        if self.isDistributed:
            train_sampler =DistributedSampler(train_dataset)
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                            batch_size=self.batch_size,
                            sampler=train_sampler,
                num_workers=self.workers, pin_memory=True)

            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.batch_size, sampler=DistributedSampler(val_dataset),
                num_workers=self.workers, pin_memory=True)
        else:
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle= True,
                num_workers=self.workers, pin_memory=True)

            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.batch_size, shuffle=False,
                num_workers=self.workers, pin_memory=True)

        return train_loader, val_loader, train_sampler