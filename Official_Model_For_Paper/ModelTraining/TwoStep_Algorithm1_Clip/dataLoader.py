import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
from VOCPart import VOCPart

class DataLoader():
    def __init__(self,dataset,batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def load_data(self, img_size=32):
        data_dir = '/home/zengyuyuan/data/CIFAR10'
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            ])
        }
        if self.dataset == 'cifar-10':
            data_train = datasets.CIFAR10(root=data_dir,
                                          transform=data_transforms['train'],
                                          train=True,
                                          download=True)

            data_test = datasets.CIFAR10(root=data_dir,
                                         transform=data_transforms['val'],
                                         train=False,
                                         download=True)
        if self.dataset == 'cifar-100':
            data_train = datasets.CIFAR100(root=data_dir,
                                          transform=data_transforms['train'],
                                          train=True,
                                          download=True)

            data_test = datasets.CIFAR100(root=data_dir,
                                         transform=data_transforms['val'],
                                         train=False,
                                         download=True)
        if self.dataset == 'mnist':
            data_dir = '/home/zengyuyuan/data/MNIST'
            mnist_transforms = {
                'train': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.1307,], [0.3081])
                ]),
                'val': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.1307, ], [0.3081])
                ])
            }
            data_train = datasets.MNIST(root=data_dir,
                                        transform = mnist_transforms['train'],
                                        train=True,
                                        download=True)
            data_test = datasets.MNIST(root=data_dir,
                                       transform=mnist_transforms['val'],
                                       train=False,
                                       download=True)
        if self.dataset == 'VOCpart':
            data_train = VOCPart('/home/haoyu/data/VOCPart', train=True ,requires=['img'], size=img_size)
            data_test = VOCPart('/home/haoyu/data/VOCPart', train=False, requires=['img'], size=img_size)
            # requires=['img','obj_mask', 'part_mask']

        image_datasets = {'train': data_train, 'val': data_test}
        # change list to Tensor as the input of the models
        dataloaders = {}
        dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'],
                                                           batch_size=self.batch_size, pin_memory=True,
                                                           shuffle=True, num_workers=16)
        dataloaders['val'] = torch.utils.data.DataLoader(image_datasets['val'],
                                                         batch_size=self.batch_size, pin_memory=True,
                                                         shuffle=False, num_workers=16)

        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

        return dataloaders,dataset_sizes




