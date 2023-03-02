from datapreparation import *
from simsiam import *
from utils import *
from evaluation import *
import torch


def train_simsiam(num_epochs, trainloader, optimizer):
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        epoch_loss = 0.0
        running_loss = 0.0
        for i, data in enumerate(trainloader):            
            # get the inputs; data is a list of [inputs, labels]
            # inputs, labels = data
            images, _ = data[0], data[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # get the two views (with random augmentations):
            x1 = images[0].to(device)
            x2 = images[1].to(device)
            
            # forward + backward + optimize
            z1, p1 = model(x1)
            z2, p2 = model(x2)
            #loss = criterion(outputs, labels)
            loss = D(p1, z2)/2 + D(p2, z1)/2
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_loss += loss.item()
            if i % 100 == 99:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0
        print("epoch loss = ", epoch_loss/len(trainloader))
    print('Finished Training')

if __name__=="__main__":
    # load cifar-10 data
    trainloader, testloader = prepare_data()
    # load model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = SimSiam()
    model = model.to(device)
    # init train settings
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9, weight_decay=0.0005)
    train_simsiam(5, trainloader, optimizer)

    # save model
    PATH = "models/simsiam_new.pth"
    torch.save(model.state_dict(), PATH)