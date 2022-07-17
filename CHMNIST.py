import torchvision.transforms as tf
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
import os
from torch.utils.data import DataLoader
import numpy as np

CHMNIST_TRANSFORMS = tf.Compose([
    tf.ToTensor(),
    tf.Resize((64, 64)),
    tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

Birds_TRANSFORMS = tf.Compose([
    tf.ToTensor(),
    tf.Resize((128, 128)),
    tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


class CHMNIST_client_test(torch.utils.data.Dataset):
    def __init__(self, num_client, transform=CHMNIST_TRANSFORMS):
        np.random.seed(2021)
        self.dataset = CHMNIST_client_allclass(train=False)
        self.transform = transform
        self.pos_index = np.where(np.array(self.dataset.targets) == num_client)[0]
        self.neg_index = np.random.choice(np.where(np.array(self.dataset.targets) != num_client)[0], len(self.pos_index))
        # print(np.r_[self.pos_index, self.neg_index])
        self.data = np.array(self.dataset.images)[np.r_[self.pos_index, self.neg_index]]
        # print(self.data)
        self.label = np.array(self.dataset.targets)[np.r_[self.pos_index, self.neg_index]]

    def __getitem__(self, item):
        img_fn = self.data[item]
        label = self.label[item]
        img = Image.open(img_fn)
        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.pos_index)*2

class CHMNIST_client(torch.utils.data.Dataset):
    def __init__(self, num_client,root ='/home/yuchen/Projects/CHMNIST',train=True, download=True, transform = CHMNIST_TRANSFORMS):
        # self.root = '/home/yuchen/Projects/BioID'
        self.images = []
        self.root = root
        self.targets = []
        self.label = num_client
        self.train = train
        self.download = download
        self.transform = transform

        x_train, x_test, y_train, y_test = self._train_test_split()

        if self.train:
            self._setup_dataset(x_train, y_train)
        else:
            self._setup_dataset(x_test, y_test)
            # self.transforms = Birds_TRANSFORMS_TEST

    def _train_test_split(self):
        img_names = []
        img_label = []
        for i, folder_name in enumerate(os.listdir(self.root)):
            for j, img_name in enumerate(os.listdir(self.root + '/' +folder_name)):
                if int(int(folder_name[0:2])-1) == self.label:
                    img_names.append(os.path.join(self.root+'/', folder_name, img_name))
                    img_label.append(self.label)
        x_train,x_test, y_train, y_test = train_test_split(img_names, img_label, train_size=0.8,
                                                            random_state=1)

        return x_train, x_test, y_train, y_test

    def _setup_dataset(self, x, y):
            self.images = x
            self.targets = y

    def __getitem__(self, item):
        img_fn = self.images[item]
        label = self.targets[item]
        img = Image.open(img_fn)
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.images)

class CHMNIST_client_allclass(torch.utils.data.Dataset):
    def __init__(self, root ='/home/yuchen/Projects/CHMNIST',train=True, download=True, transform = CHMNIST_TRANSFORMS):
        # self.root = '/home/yuchen/Projects/BioID'
        self.images = []
        self.root = root
        self.targets = []
        self.train = train
        self.download = download
        self.transform = transform

        x_train, x_test, y_train, y_test = self._train_test_split()

        if self.train:
            self._setup_dataset(x_train, y_train)
        else:
            self._setup_dataset(x_test, y_test)
            # self.transforms = Birds_TRANSFORMS_TEST

    def _train_test_split(self):
        img_names = []
        img_label = []
        for i, folder_name in enumerate(os.listdir(self.root)):
            for j, img_name in enumerate(os.listdir(self.root + '/' +folder_name)):
                img_names.append(os.path.join(self.root+'/', folder_name, img_name))
                img_label.append(int(folder_name[0:2])-1)
        x_train,x_test, y_train, y_test = train_test_split(img_names, img_label, train_size=0.8,
                                                            random_state=1)

        return x_train, x_test, y_train, y_test

    def _setup_dataset(self, x, y):
            self.images = x
            self.targets = y

    def __getitem__(self, item):
        img_fn = self.images[item]
        label = self.targets[item]
        img = Image.open(img_fn)
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.images)