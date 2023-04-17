import argparse
import torch

from SimSiam.simsiam.datapreparation import *
from SimSiam.simsiam.simsiam import *
from SimSiam.simsiam.utils import *
# from SimSiam.simsiam.evaluation import *

from SimSiam.federated_simsiam.client import *
from SimSiam.federated_simsiam.server import *
from SimSiam.federated_simsiam.datapreparation import *
from tqdm import tqdm


class LinearEvaluationSimSiam(nn.Module):
    def __init__(self, trained_model_path, device, linearevaluation=True):
        super(LinearEvaluationSimSiam, self).__init__()
        model = SimSiam()
        model.load_state_dict(torch.load(trained_model_path, map_location=device))
        self.encoder = model.encoder.to(device)

        if linearevaluation:
            # freeze parameters         
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.classifier = nn.Linear(2048, 10).to(device)

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x
    

# Supervised model
class SupervisedModel(nn.Module):
    """Standard supervised model with resnet18 backbone"""
    
    def __init__(self, device, pretrained=True, linearevaluation=False):
        super(SupervisedModel, self).__init__()
        self.model = torchvision.models.resnet18(weights=pretrained).to(device)

        if linearevaluation:
            # freeze parameters         
            for param in self.model.parameters():
                param.requires_grad = False

        #num_ftrs = self.model.fc.in_features
        #self.model.fc = nn.Linear(num_ftrs, 10) 
        out_features = self.model.fc.out_features
        self.classifier = nn.Linear(out_features, 10)
        

    def forward(self, x):
        x = self.model(x)
        x = self.classifier(x)
        return x
    

# def linear_evaluation_supervised(trainloader, testloader, criterion, lr, momentum):
#     """Trains and evaluates a supervised model on CIFAR-10"""

#     model = SupervisedModel(pretrained=True, linearevaluation=True)
#     model = model.to(device)
#     optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
#     model = train_downstream(opt.num_epochs, model, trainloader, criterion, optimizer, device)
#     accuracy = evaluate_downstream(model, testloader, device)
#     return accuracy

# def linear_evaluation_simsiam(trained_model_path, trainloader, testloader, criterion, lr, momentum):
#     """Trains and evaluates SimSIam on CIFAR-10"""

#     simsiam = SimSiamDownstream(trained_model_path=trained_model_path, device=device, linearevaluation=True)
#     simsiam = simsiam.to(device)
#     optimizer = optim.SGD(simsiam.parameters(), lr=lr, momentum=momentum)
#     simsiam = train_simsiam_downstream(opt.num_epochs, simsiam, trainloader, criterion, optimizer, device)
#     accuracy = evaluate_simsiam_downstream(simsiam, testloader, device)
#     return accuracy

# # cifar-10 data for classification
# trainloader, testloader = get_downstream_data(percentage_of_data=opt.data_percentage, batch_size=opt.batch_size)

# # linear evaluation supervised model
# supervised_accuracy = linear_evaluation_supervised(trainloader, testloader, criterion, lr, momentum)
# print("accuracy for supervised model: ", supervised_accuracy)

# # linear evaluation SimSiam
# simsiam_accuracy = linear_evaluation_simsiam(opt.simsiam_path, trainloader, testloader, criterion, lr, momentum)
# print("accuracy for standard simsiam: ", simsiam_accuracy)

# # linear evaluation fedavg SimSiam
# fedavg_simsiam_accuracy = linear_evaluation_simsiam(opt.fedavg_simsiam_path, trainloader, testloader, criterion, lr, momentum)
# print("accuracy for fedAVG simsiam: ", fedavg_simsiam_accuracy)
#


if __name__=="__main__":
    """ 
    python3 evaluation_comparison.py --data_percentage 0.1 --epochs 50 --lr 0.03 --batch_size 32 --simsiam_path 'simsiam_800.pth'

    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_percentage', type=float, default=0.1, help='percentage of data used for training')
    parser.add_argument('--epochs', type=int, default=5, help="number of epochs used for downstream training")
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for downstream training')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--simsiam_path', type=str, default="simsiam.pth", help='path to trained simsiam model')
    parser.add_argument('--fedavg_simsiam_path', type=str, default="fedavg_simsiam.pth", help='path to trained fedaveraged simsiam model')

    opt = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    criterion = nn.CrossEntropyLoss()
    lr = opt.lr
    momentum = 0.9

    trainloader, testloader = get_downstream_data(opt.data_percentage, batch_size=opt.batch_size)

    simsiam = LinearEvaluationSimSiam(opt.simsiam_path, device)

    # training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(simsiam.classifier.parameters(), lr=lr, momentum=momentum)

    # simsiam.encoder.eval()   # this is linear evaluation
    # simsiam.classifier.train()
   
    # simsiam.train() # this is finetuning
    
    global_progress = tqdm(range(0, opt.epochs), desc=f'Evaluating SimSiam')
    for epoch in global_progress:  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            #inputs, labels = data
            inputs, labels = data[0].to(device), data[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = simsiam(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 500 == 499:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 500:.3f}')
                running_loss = 0.0
    print('Finished Training')


    # evaluation
    simsiam.eval()
    
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = simsiam(inputs)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct // total
    print(f'Accuracy of SimSiam on test images: {accuracy} %')



    # Supervised model
    # model = SupervisedModel(device, pretrained=True) # Change pretrained to true/false
    # model = model.to(device)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)


    # model.model.eval()    # this is linear evaluation
    # model.classifier.train()
    # model.train() # this is finetuning

    model = resnet18(pretrained=True).to(device)

# Replace the last layer with a linear layer for CIFAR-10
    model.fc = nn.Linear(model.fc.in_features, 10).to(device)

    # Train the linear layer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.fc.parameters(), lr=lr, momentum=momentum)

    
    global_progress = tqdm(range(0, opt.epochs), desc=f'Evaluating Supervised Model')
    for epoch in global_progress:  # loop over the dataset multiple times
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

    # eval
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
    print(f'Accuracy of the supervised network on test images: {accuracy} %')

