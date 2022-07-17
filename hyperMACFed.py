from dataLoader import *
from modelUtil import *
from FedUser import User
import torch
from datetime import date
import os
from datasets import *
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
today = date.today().isoformat()
DATA_NAME = "cifar"
datashape = (3, 64, 64)
NUM_CLIENTS = 10
NUM_CLASSES = 10
NUM_CLASES_PER_CLIENT = 10
MODEL = "alexnet"
a = 0.3
EPOCHS = 300
BATCH_SIZE = 64
LEARNING_RATE_GEN = 1e-3
LEARNING_RATE_DIS = 5e-3
MODE="hypernet"
# trainsets = [cifar10_client(num_client=i, train=True) for i in range(NUM_CLIENTS)]
# testsets = [cifar10_client(num_client=i, train=False) for i in range(NUM_CLIENTS)]
# train_dataloaders = [DataLoader(trainsets[i], batch_size=BATCH_SIZE, shuffle=False) for i in range(NUM_CLIENTS)]
# test_dataloaders = [DataLoader(testsets[i], batch_size=BATCH_SIZE, shuffle=False) for i in range(NUM_CLIENTS)]
train_dataloaders, test_dataloaders, distributions = gen_random_loaders(DATA_NAME, '/mnt/HDD/torch_data', NUM_CLIENTS,
                                                         BATCH_SIZE,NUM_CLASES_PER_CLIENT, NUM_CLASSES, return_distributions=True)

users = [User(i, a, device, MODEL, NUM_CLASSES,
              train_dataloaders[i], gen_lr=LEARNING_RATE_GEN,
              disc_lr=LEARNING_RATE_DIS, dataname=DATA_NAME) for i in range(NUM_CLIENTS)]
hnet = TriggerHyperDis(embedding_dim=10, data_shape=datashape).to(device)
embeddings = torch.tensor(distributions, dtype=torch.float32)
for i in range((len(users))):
    users[i].set_trigger(hnet(embeddings[i].to(device)).data)
# print(embeddings)
# # embeddings = [torch.nn.functional.one_hot(torch.tensor([i]), num_classes=NUM_CLIENTS).type(torch.float32) for i in range(NUM_CLIENTS)]
# embeddings = [torch.tensor([0.1]*NUM_CLIENTS).type(torch.float32) for i in range(NUM_CLIENTS)]
hnet_optimizer = torch.optim.SGD(hnet.parameters(), lr=1e-2)
loss_hn = torch.nn.MSELoss()
for epoch in range(EPOCHS):
    for index in range(NUM_CLIENTS): users[index].train()
    weights_agg = agg_weights([user.get_model_state_dict() for user in users])

    # Update hyperNet:
    for index in range(NUM_CLIENTS):
        hnet_optimizer.zero_grad()
        trigger = hnet(embeddings[index].to(device))
        loss = loss_hn(trigger, users[index].trigger)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(hnet.parameters(), 50)
        hnet_optimizer.step()

    for i in range((len(users))):
        users[i].set_model_state_dict(weights_agg)
        users[i].set_trigger(hnet(embeddings[i].to(device)).data)
        # users[i].set_trigger(trigger_agg)
    print(f"Epoch: {epoch}")
    evaluate_global(users, train_dataloaders)
    evaluate_global(users, test_dataloaders)
    if epoch == EPOCHS-1:
        torch.save(weights_agg, f'weights/{DATA_NAME}_{MODEL}_{MODE}.pth')
        for i in range(NUM_CLIENTS):
            np.save(f'weights/{DATA_NAME}_{MODEL}_{MODE}_{i}.npy', users[i].trigger.detach().cpu().numpy())