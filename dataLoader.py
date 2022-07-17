import numpy as np
import torchvision
import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as tf
import torchvision.transforms.functional as F
from PIL import Image

IMAGENET_STATS = dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
CIFAR_10_STATS = dict(mean=(0.4914, 0.4822, 0.4465), std= (0.2023, 0.1994, 0.2010))
BASE_TRANSFORMS = tf.Compose([tf.ToTensor(),
                              tf.Resize((64, 64)),
                              tf.Normalize(**IMAGENET_STATS)])
TRIGGER_TRANSFORMS = tf.Compose([tf.ToTensor(),
                                tf.Resize((64, 64)),
                                tf.Normalize(**CIFAR_10_STATS)])


class cifar10_client(torch.utils.data.Dataset):
    def __init__(self, num_client, transform=BASE_TRANSFORMS, train=True, dataset=None):
        self.train = train
        self.transform = transform
        self.dataset = torchvision.datasets.CIFAR10(root='/mnt/HDD/torch_data', train=train, download=True) if dataset==None else dataset
        self.data = self.dataset.data[np.array(self.dataset.targets) == num_client]
        self.label = num_client
        # self.label = [self.dataset.targets[i] for i, flag in enumerate(np.array(self.dataset.targets) == num_client) if flag]

    def __getitem__(self, item):
        img = self.transform(self.data[item])
        return img, self.label

    def __len__(self):
        return self.data.shape[0]


class cifar10_client_test(torch.utils.data.Dataset):
    def __init__(self, num_client, transform=BASE_TRANSFORMS):
        np.random.seed(2021)
        self.dataset = torchvision.datasets.CIFAR10(root='/mnt/HDD/torch_data', train=False, download=False)
        self.transform = transform
        self.pos_index = np.where(np.array(self.dataset.targets) == num_client)[0]
        self.neg_index = np.random.choice(np.where(np.array(self.dataset.targets) != num_client)[0], len(self.pos_index))
        self.data = self.dataset.data[np.r_[self.pos_index, self.neg_index]]
        self.label = np.array(self.dataset.targets)[np.r_[self.pos_index, self.neg_index]]

    def __getitem__(self, item):
        img = self.transform(self.data[item])
        return img, self.label[item]

    def __len__(self):
        return len(self.pos_index)*2

class cifar10_client_iid(torch.utils.data.Dataset):
    def __init__(self, num_client, transform=BASE_TRANSFORMS, train=True):
        self.train = train
        self.transform = transform
        self.lenth = 5000 if train else 1000
        self.indice = np.arange(num_client*self.lenth, (num_client+1)*self.lenth)
        self.dataset = torchvision.datasets.CIFAR10(root='/mnt/HDD/torch_data', train=train, download=True)
        self.data = Subset(self.dataset.data, self.indice)
        self.targets = Subset(self.dataset.targets, self.indice)
    def __getitem__(self, item):
        img = self.transform(self.data[item])
        return img, self.targets[item]

    def __len__(self):
        return len(self.data)


def random_initial(dataname):
    img_shape = None
    IMAGE_STATS = None
    if dataname in ['cifar', 'chmnist']:
        img_shape = (3, 64, 64)
        # IMAGE_STATS = dict(mean=(0.4914, 0.4822, 0.4465), std= (0.2023, 0.1994, 0.2010))
        IMAGE_STATS = dict(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    elif dataname == 'mnist':
        img_shape = (1, 28, 28)
        IMAGE_STATS = dict(mean=[0.5], std=[0.5])
    elif dataname == 'celeba':
        img_shape = (3, 128, 128)
        IMAGE_STATS = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    random_initial = torch.rand(img_shape)
    F.normalize(random_initial, IMAGE_STATS['mean'], IMAGE_STATS['std'], inplace=True)
    return random_initial


def avg_trigger(triggers):
    with torch.no_grad():
        trigger = sum(triggers) / len(triggers)
    return trigger


def gen_distributions(num_classes, num_clients):
    distribution = np.zeros((10, 10))
    for index in range(num_clients):
        distribution[index][np.array(range(index, index + num_classes)) % 10] = 1 / num_classes
    return distribution

def load_trigger(num_client, img_name=None):
    file_name = f"Triggers/{num_client+1}.png" if img_name==None else f"Triggers/{img_name}"
    trigger = Image.open(file_name).convert('RGB')
    return torch.as_tensor(TRIGGER_TRANSFORMS(trigger), dtype=torch.float)