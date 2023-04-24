import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split


from .simsiam import SimSiam


class SupervisedModel(nn.Module):
    """Standard supervised model with resnet18 backbone"""
    
    def __init__(self, pretrained=True, linearevaluation=False):
        super(SupervisedModel, self).__init__()
        self.model = torchvision.models.resnet18(weights=pretrained)

        if linearevaluation:
            # freeze parameters         
            for param in self.model.parameters():
                param.requires_grad = False

        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 10) 

    def forward(self, x):
        return self.model(x)
    

class SimSiamDownstream(nn.Module):
    def __init__(self, trained_model_path, device, linearevaluation=True):
        super(SimSiamDownstream, self).__init__()
        model = SimSiam()
        model.load_state_dict(torch.load(trained_model_path, map_location=device))
        self.simsiam = model

        if linearevaluation:
            # freeze parameters         
            for param in self.simsiam.parameters():
                param.requires_grad = False

        self.classifier = nn.Linear(2048, 10)


    def forward(self, x):
        z, _ = self.simsiam(x)
        x = self.classifier(z)
        return x
    

def get_downstream_data(percentage_of_data=0.1, batch_size=4):
    """Returns train and testloader for downstream task (cifar10 image classification).
    
    Returns *percentage_of_data* data for downstream task training.
    """
    # preprocessing
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]) # ((mean, mean, mean), (std, std, st))

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    
    # Train only on *percentage_of_data* of training data
    subset = percentage_of_data  * len(trainset)
    train_subset = torch.utils.data.Subset(trainset, [i for i in range(int(subset))])

    # validation set
    train_size = int(0.8 * len(train_subset))
    val_size = len(train_subset) - train_size
    train_subset, val_subset = random_split(train_subset, [train_size, val_size])


    trainloader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)
    
    valloader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    return trainloader, valloader, testloader


def train_downstream(num_epochs, model, trainloader, criterion, optimizer, device):
    model.train()
    
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            #inputs, labels = data
            inputs, labels = data[0].to(device), data[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            #with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 500 == 499:    # print every 2000 mini-batches
                print(f'[{epoch + 1 }, {i + 1:5d}] loss: {running_loss / 500:.3f}')
                running_loss = 0.0

    print('Finished Training')
    return model


def train_simsiam_downstream(num_epochs, model, trainloader, criterion, optimizer, device):
    model.simsiam.eval()
    model.classifier.train()
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            #inputs, labels = data
            inputs, labels = data[0].to(device), data[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 500 == 499:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 500:.3f}')
                running_loss = 0.0

    print('Finished Training')
    return model


def evaluate_downstream(model, testloader, device):
    model.eval()
    
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = model(inputs)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct // total
    print(f'Accuracy of the network on test images: {accuracy} %')
    return accuracy


def evaluate_simsiam_downstream(model, testloader, device):
    model.eval()
    
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = model(inputs)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct // total
    print(f'Accuracy of the network on test images: {accuracy} %')
    return accuracy