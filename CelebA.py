from functools import partial

import numpy as np
import pandas as pd
import os
import PIL
import torch
from torch.utils.data import Dataset
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import torchvision.transforms as tf
CELEBA_TRANSFORMS = tf.Compose([
    tf.ToTensor(),
    tf.CenterCrop((178, 178)),
    tf.Resize((128, 128)),
    tf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class CelebADataset(Dataset):
    def __init__(self, root='/mnt/HDD/torch_data', train=True, transform=None, target_transform=None, num_clients=50):

        self.root = root + '/celeba'
        self.transform = transform
        self.target_transform = target_transform
        self.num_clients = num_clients
        self.train = train
        self.identity = self._set_identity()
        self.filename = self.identity.index.values
        self.target = self.identity[1].values

    def _set_identity(self):
        fn = partial(os.path.join, self.root)
        identity = pd.read_csv(fn("identity_CelebA.csv"), delim_whitespace=True, header=None, index_col=0)
        identity_index = identity[1].value_counts()[:self.num_clients].index.to_list()

        identity = identity[identity[1].isin(identity_index)]
        le = preprocessing.LabelEncoder()
        le.fit(identity.values.reshape(-1))
        identity[1] = le.transform(identity[1].values)
        train_DF, test_DF = train_test_split(identity, train_size=0.8, random_state=2021,
                                             stratify=identity[1])
        return train_DF if self.train else test_DF

    def __getitem__(self, index: int):
        X = PIL.Image.open(os.path.join(self.root,
                                        "img_align_celeba",
                                        "img_align_celeba",
                                        self.filename[index]))

        target = self.target[index]
        if self.transform is not None:
            X = self.transform(X)
        return X, target

    def __len__(self) -> int:
        return len(self.identity)


class CelebA_client(torch.utils.data.Dataset):
    def __init__(self, num_client, transform=CELEBA_TRANSFORMS, train=True):
        self.train = train
        self.transform = transform
        self.dataset = CelebADataset(root='/mnt/HDD/torch_data', train=train)
        self.data = self.dataset.filename[self.dataset.target == num_client]
        self.label = num_client
        # self.label = [self.dataset.targets[i] for i, flag in enumerate(np.array(self.dataset.targets) == num_client) if flag]

    def __getitem__(self, item):
        img = self.transform(PIL.Image.open(os.path.join(self.dataset.root,
                                        "img_align_celeba",
                                        "img_align_celeba",
                                        self.data[item])))
        return img, self.label

    def __len__(self):
        return self.data.shape[0]


class CelebA_client_test(torch.utils.data.Dataset):
    def __init__(self, num_client, transform=CELEBA_TRANSFORMS):
        np.random.seed(2021)
        self.dataset = CelebADataset(root='/mnt/HDD/torch_data', train=False)
        self.transform = transform
        self.pos_index = np.where(np.array(self.dataset.target) == num_client)[0]
        self.neg_index = np.random.choice(np.where(np.array(self.dataset.target) != num_client)[0], len(self.pos_index))
        self.data = self.dataset.filename[np.r_[self.pos_index, self.neg_index]]
        self.label = np.array(self.dataset.target)[np.r_[self.pos_index, self.neg_index]]

    def __getitem__(self, item):
        img = self.transform(PIL.Image.open(os.path.join(self.dataset.root,
                                        "img_align_celeba",
                                        "img_align_celeba",
                                        self.data[item])))
        return img, self.label[item]

    def __len__(self):
        return len(self.pos_index)*2
