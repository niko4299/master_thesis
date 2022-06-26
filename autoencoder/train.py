from model import CAEC
import torchvision
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import autoencoder.config as config

model = CAEC(in_channels=3, encoder_arhitecture_layers_size=config.encoder_arhitecture_layers_size,decoder_arhitecture_layers_size=config.decoder_arhitecture_layers_size)
transform = torchvision.transforms.Compose([
    # you can add other transformations in this list
    torchvision.transforms.ToTensor()
])

dataset = torchvision.datasets.ImageFolder("/home/niko/pravi_diplomski/mongo_db/data_sim",transform = transform)

def get_subset(indices, start, end):
    return indices[start : start + end]


TRAIN_PCT, VALIDATION_PCT = 0.3, 0.2  # rest will go for test
train_count = int(len(dataset) * TRAIN_PCT)
validation_count = int(len(dataset) * VALIDATION_PCT)

indices = torch.randperm(len(dataset))
train_indices = get_subset(indices, 0, train_count)
validation_indices = get_subset(indices, train_count, validation_count)
test_indices = get_subset(indices, train_count + validation_count, len(dataset))
train_dataloader = torch.utils.data.DataLoader(dataset, sampler=SubsetRandomSampler(train_indices),batch_size = 16,)
validation_dataloader = torch.utils.data.DataLoader(dataset, sampler=SubsetRandomSampler(validation_indices))
test_dataloader = torch.utils.data.DataLoader(dataset, sampler=SubsetRandomSampler(test_indices))
num_epochs = 10
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())
losses = []

for epoch in range(1,num_epochs):

    for x,_ in tqdm(train_dataloader):
        optimizer.zero_grad()
        x_return = model(x)
        loss = criterion(x_return,x)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    print(np.mean(losses))
    losses = []
    if epoch % 2 == 0:
        x,_ = next(iter(train_dataloader))
        fig = plt.figure()
        with torch.no_grad():
           x_return = model(x)
        plt.imshow(  x_return.permute(1, 2, 0)  )
        fig.savefig('./images/test_{epoch}.png', dpi=fig.dpi)
        fig = plt.figure()
        plt.imshow(  x.permute(1, 2, 0)  )
        fig.savefig('./images/train_{epoch}.png', dpi=fig.dpi)

    