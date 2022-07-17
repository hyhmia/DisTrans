from modelUtil import *
from dataLoader import *
import torchmetrics

class User:
    def __init__(self, index, alpha, device, model, n_classes, train_dataloader, gen_lr=1e-3, disc_lr=5e-3, dataname='cifar'):
        self.index = index
        self.alpha = alpha
        self.model = DualModel_pretrained(n_classes=n_classes, backbone=model)
        self.train_dataloader = train_dataloader
        self.trigger = random_initial(dataname).to(device)
        self.trigger.requires_grad = True
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.gen_lr = gen_lr
        self.disc_lr = disc_lr
        self.acc_metric = torchmetrics.Accuracy().to(device)
        self.device = device

    def train(self):
        self.model = self.model.to(self.device)
        gen_optimizer = torch.optim.SGD([self.trigger], self.gen_lr)
        dis_optimizer = torch.optim.SGD(self.model.parameters(), self.disc_lr)
        self.model.train()
        for images, labels in self.train_dataloader:
            images, labels = images.to(self.device), labels.to(self.device)
            gen_optimizer.zero_grad()
            dis_optimizer.zero_grad()
            inputs_left, inputs_right = blend_image(images, self.trigger, self.alpha)
            logits, preds = self.model(inputs_left, inputs_right)
            loss = self.loss_fn(logits, labels)
            loss.backward()
            dis_optimizer.step()
            gen_optimizer.step()
            self.acc_metric(preds, labels)
        print(f"Client: {self.index} ACC: {self.acc_metric.compute()}")
        self.acc_metric.reset()
        self.model.to('cpu')

    def evaluate(self, dataloader):
        self.model.to(self.device)
        self.model.eval()
        testing_corrects = 0
        testing_sum = 0
        for images, labels in dataloader:
            images, labels = images.to(self.device), labels.to(self.device)
            inputs_left, inputs_right = blend_image(images, self.trigger, self.alpha)
            _, preds = self.model(inputs_left, inputs_right)
            testing_corrects += torch.sum(torch.argmax(preds, dim=1) == labels)
            testing_sum += len(labels)
        self.model.to('cpu')
        return testing_corrects, testing_sum

    def get_model_state_dict(self):
        return self.model.state_dict()

    def set_model_state_dict(self, weights):
        self.model.load_state_dict(weights)

    def set_trigger(self, agg_trigger):
        self.trigger.data = copy.deepcopy(agg_trigger)


class AvgUser:
    def __init__(self, index, device, model, n_classes, train_dataloader, gen_lr=1e-3, disc_lr=5e-3):
        self.index = index
        self.model = SingleModel_pretrained(n_classes=n_classes, backbone=model)
        self.train_dataloader = train_dataloader
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.gen_lr = gen_lr
        self.disc_lr = disc_lr
        self.acc_metric = torchmetrics.Accuracy().to(device)
        self.device = device

    def train(self):
        self.model = self.model.to(self.device)
        dis_optimizer = torch.optim.SGD(self.model.parameters(), self.disc_lr)
        self.model.train()
        for images, labels in self.train_dataloader:
            images, labels = images.to(self.device), labels.to(self.device)
            dis_optimizer.zero_grad()
            logits, preds = self.model(images)
            loss = self.loss_fn(logits, labels)
            loss.backward()
            dis_optimizer.step()
            self.acc_metric(preds, labels)
        print(f"Client: {self.index} ACC: {self.acc_metric.compute()}")
        self.acc_metric.reset()
        self.model.to('cpu')

    def evaluate(self, dataloader):
        self.model.to(self.device)
        self.model.eval()
        testing_corrects = 0
        testing_sum = 0
        for images, labels in dataloader:
            images, labels = images.to(self.device), labels.to(self.device)
            _, preds = self.model(images)
            testing_corrects += torch.sum(torch.argmax(preds, dim=1) == labels)
            testing_sum += len(labels)
        self.model.to('cpu')
        return testing_corrects, testing_sum

    def get_model_state_dict(self):
        return self.model.state_dict()

    def set_model_state_dict(self, weights):
        self.model.load_state_dict(weights)

