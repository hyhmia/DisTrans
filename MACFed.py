from dataLoader import *
from modelUtil import *
from datasets import *
from FedUser import User
import torch
from datetime import date
import os

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
today = date.today().isoformat()
DATA_NAME = "cifar"
NUM_CLIENTS = 10
NUM_CLASSES = 10
NUM_CLASES_PER_CLIENT=2
MODEL = "alexnet"
a = 0.2
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE_GEN = 1e-3
LEARNING_RATE_DIS = 5e-3
OFFSET_AGG = False
parent_dir = "/mnt/HDD/weights/MACFed_torch/"
os.makedirs(os.path.join(parent_dir, f"Base/AGGSingle/{today}_Clients_{NUM_CLIENTS}"), exist_ok=True)
# trainsets = [cifar10_client(num_client=i, train=True) for i in range(NUM_CLIENTS)]
# testsets = [cifar10_client(num_client=i, train=False) for i in range(NUM_CLIENTS)]
# train_dataloaders = [DataLoader(trainsets[i], batch_size=BATCH_SIZE, shuffle=False) for i in range(NUM_CLIENTS)]
# test_dataloaders = [DataLoader(testsets[i], batch_size=BATCH_SIZE, shuffle=False) for i in range(NUM_CLIENTS)]
train_dataloaders, test_dataloaders = gen_random_loaders(DATA_NAME, '/mnt/HDD/torch_data', NUM_CLIENTS,
                                                         BATCH_SIZE,NUM_CLASES_PER_CLIENT, NUM_CLASSES)
users = [User(i, a, device, MODEL, NUM_CLASSES,
              train_dataloaders[i], gen_lr=LEARNING_RATE_GEN,
              disc_lr=LEARNING_RATE_DIS, dataname=DATA_NAME) for i in range(NUM_CLIENTS)]

for epoch in range(EPOCHS):
    for index in range(NUM_CLIENTS): users[index].train()
    weights_agg = agg_weights([user.get_model_state_dict() for user in users])
    if OFFSET_AGG:
        trigger_agg = avg_trigger([user.trigger for user in users])
    for i in range((len(users))):
        users[i].set_model_state_dict(weights_agg)
        if OFFSET_AGG:
            users[i].set_trigger(trigger_agg)
    print(f"Epoch: {epoch}")
    evaluate_global(users, train_dataloaders)
    evaluate_global(users, test_dataloaders)