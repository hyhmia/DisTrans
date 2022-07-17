import torch
from torchvision.models import alexnet, resnet18
from torch.nn.functional import relu, softmax
from torch.nn.utils import spectral_norm
from torch import nn, tanh
import copy


def weights_init_uniform(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)


def agg_weights(weights):
    with torch.no_grad():
        weights_avg = copy.deepcopy(weights[0])
        for k in weights_avg.keys():
            for i in range(1, len(weights)):
                weights_avg[k] += weights[i][k]
            weights_avg[k] = torch.div(weights_avg[k], len(weights))
    return weights_avg


def blend_image(images, trigger, a):
    return images*(1-a)+trigger*a, images*(1+a)-trigger*a


def evaluate_global(users, test_dataloders):
    testing_corrects = 0
    testing_sum = 0
    for index in range(len(users)):
        corrects, sum = users[index].evaluate(test_dataloders[index])
        testing_corrects += corrects
        testing_sum += sum
    print(f"Acc: {testing_corrects / testing_sum}")


class SingleModel_pretrained(torch.nn.Module):
    def __init__(self, backbone='resnet', n_classes=10):
        super().__init__()
        if backbone == 'alexnet':
            self.model = alexnet(pretrained=True)
            n_ftrs = self.model.classifier[-1].out_features
            self.fc = torch.nn.Linear(n_ftrs, n_classes)
            self.fc.apply(weights_init_uniform)
        elif backbone == 'resnet':
            self.model = resnet18(pretrained=True)
            n_ftrs = self.model.fc.in_features
            self.model.fc = torch.nn.Sequential()
            self.fc = torch.nn.Linear(n_ftrs, n_classes)
        elif backbone == 'cnn_cifar':
            self.model = nn.Sequential(torch.nn.Conv2d(3, 6, (5,)),
                                       torch.nn.MaxPool2d(2, 2),
                                       torch.nn.Conv2d(6, 16, (5,)),
                                       torch.nn.Flatten(),
                                       torch.nn.Linear(16 * 5 * 5, 100)
                                       )
            self.fc = torch.nn.Linear(100, n_classes)
        elif backbone == 'cnn_bioid':
            self.model = nn.Sequential(torch.nn.Conv2d(1, 6, 5),
                                       torch.nn.MaxPool2d(2, 2),
                                       torch.nn.Conv2d(6, 16, 5),
                                       torch.nn.Flatten(),
                                       torch.nn.Linear(16*122*122, 100),
                                       )
            self.fc = torch.nn.Linear(100, n_classes)
            self.model.apply(weights_init_uniform)
            self.fc.apply(weights_init_uniform)
        elif backbone == 'cnn_mnist':
            self.model = nn.Sequential(torch.nn.Conv2d(1, 6, 5),
                                       torch.nn.MaxPool2d(2, 2),
                                       torch.nn.Conv2d(6, 16, 5),
                                       torch.nn.Flatten(),
                                       torch.nn.Linear(8*16*8, 100),
                                       # torch.nn.ReLU()
                                       )
            self.fc = torch.nn.Linear(100, n_classes)
            self.model.apply(weights_init_uniform)
            self.fc.apply(weights_init_uniform)

    def forward(self, x):
        embedding = self.model(x)
        logits = self.fc(embedding)
        return logits, softmax(logits, dim=1)


class DualModel_pretrained(torch.nn.Module):
    def __init__(self, backbone='resnet', n_classes=10):
        super().__init__()
        if backbone == 'alexnet':
            self.model = alexnet(pretrained=True)
            self.model.classifier[-1] = nn.Linear(4096, 256)
            self.fc = torch.nn.Linear(2 * 256, n_classes)
            # n_ftrs = self.model.classifier[-1].out_features
            # self.fc = torch.nn.Linear(2*n_ftrs, n_classes)
            self.fc.apply(weights_init_uniform)
        elif backbone == 'resnet':
            self.model = resnet18(pretrained=True)
            n_ftrs = self.model.fc.in_features
            self.model.fc = torch.nn.Sequential()
            self.fc = torch.nn.Linear(2*n_ftrs, n_classes)
            # self.model.apply(weights_init_uniform)
            self.fc.apply(weights_init_uniform)

    def forward(self, x_1, x_2):
        embedding_1 = self.model(x_1)
        # print(embedding_1.shape)
        embedding_2 = self.model(x_2)

        logits = self.fc(torch.cat((embedding_1, embedding_2), 1))
        return logits, softmax(logits, dim=1)


class DualModel_CNN(torch.nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 100)
        self.fc2 = torch.nn.Linear(200, n_classes)

    def forward(self, x_1, x_2):
        embedding_1 = self.get_embedding(x_1)
        embedding_2 = self.get_embedding(x_2)

        logits = self.fc2(torch.cat((embedding_1, embedding_2), 1))
        return logits, softmax(logits)

    def get_embedding(self, x):
        x = self.pool(relu(self.conv1(x)))
        x = self.pool(relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        return relu(self.fc1(x))


class TriggerHyper(nn.Module):
    def __init__(self, n_nodes, embedding_dim, out_dim=64, ngf=16, hidden_dim=100, n_hidden=3):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=n_nodes, embedding_dim=embedding_dim)
        self.ngf = ngf
        layers = [nn.Linear(embedding_dim, hidden_dim)]
        for _ in range(n_hidden):
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.mlp = nn.Sequential(*layers)
        self.netG = nn.Linear(hidden_dim, out_dim * out_dim * 3)
        self.mlp.apply(weights_init_uniform)
        self.netG.apply(weights_init_uniform)

    def forward(self, idx):
        emd = self.embedding(idx)
        features = self.mlp(emd)
        trigger = tanh(self.netG(features).view(3, 64, 64))
        return trigger


class Generator(nn.Module):
    def __init__(self, nz):
        super(Generator, self).__init__()
        ngf=64
        self.fc = nn.Linear(nz, 100)
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(100, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )
        self.main.apply(weights_init_uniform)
        self.fc.apply(weights_init_uniform)

    def forward(self, input):
        features = self.fc(input)
        return self.main(features.view(-1, 100, 1, 1))


class TriggerHyperDis(nn.Module):
    def __init__(self, embedding_dim, data_shape, hidden_dim=100, n_hidden=1):
        super().__init__()
        layers = [spectral_norm(nn.Linear(embedding_dim, hidden_dim))]
        for _ in range(n_hidden):
            layers.append(nn.ReLU(inplace=True))
            layers.append(spectral_norm(nn.Linear(hidden_dim, hidden_dim)))
        self.mlp = nn.Sequential(*layers)
        self.mlp.apply(weights_init_uniform)
        # self.netG = spectral_norm(nn.Linear(hidden_dim, data_shape[0] * data_shape[1] * data_shape[2]))
        self.netG = Generator(nz = hidden_dim)
        self.data_shape = data_shape
        self.netG.apply(weights_init_uniform)

    def forward(self, embedding):
        features = self.mlp(embedding)
        trigger = self.netG(features).view(*self.data_shape)
        return trigger
