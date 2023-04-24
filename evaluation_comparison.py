import argparse
import torch

from SimSiam.simsiam.datapreparation import *
from SimSiam.simsiam.simsiam import *
from SimSiam.simsiam.utils import *
# from SimSiam.simsiam.evaluation import *

from SimSiam.federated_simsiam.client import *
from SimSiam.federated_simsiam.server import *
from SimSiam.federated_simsiam.datapreparation import *
import matplotlib.pyplot as plt
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


if __name__=="__main__":
    """ 
    python3 evaluation_comparison.py --data_percentage 0.1 --epochs 30 --lr 0.1 --batch_size 32 --simsiam_path 'fedavg_iid_5_7_100.pth'

    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_percentage', type=float, default=0.1, help='percentage of data used for training')
    parser.add_argument('--epochs', type=int, default=5, help="number of epochs used for downstream training")
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for downstream training')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--simsiam_path', type=str, default="simsiam.pth", help='path to trained simsiam model')
    # parser.add_argument('--fedavg_simsiam_path', type=str, default="fedavg_simsiam.pth", help='path to trained fedaveraged simsiam model')

    opt = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    criterion = nn.CrossEntropyLoss()
    lr = opt.lr
    momentum = 0.9

    trainloader, valloader, testloader = get_downstream_data(opt.data_percentage, batch_size=opt.batch_size)

    # split train

    # SimSiam
    simsiam = LinearEvaluationSimSiam(opt.simsiam_path, device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(simsiam.classifier.parameters(), lr=lr, momentum=momentum)

    simsiam.encoder.eval()   # this is linear evaluation
    simsiam.classifier.train()
    # simsiam.train() # this is finetuning
    
    accuracies = []
    global_progress = tqdm(range(0, opt.epochs), desc=f'Training SimSiam')
    for epoch in global_progress:  # loop over the dataset multiple times
        correct = 0
        total = 0
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()

            outputs = simsiam(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
   
        # print accuracy after every epoch
        # acc = 100 * correct / total
        # accuracies.append(acc)
        # print(f'Accuracy after epoch {epoch + 1}: {acc:.2f}%')
        correct = 0
        total = 0
        with torch.no_grad():
            for data in valloader:
                inputs, labels = data[0].to(device), data[1].to(device)
                # calculate outputs by running images through the network
                outputs = simsiam(inputs)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct // total
        accuracies.append(accuracy)
        print(f'Validation accuracy after epoch {epoch + 1}: {accuracy:.2f}%')

    plt.plot(accuracies)
    plt.ylim(0, 100)
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.savefig(f"{opt.simsiam_path}_evaluation_{opt.data_percentage}.png")
    # evaluation
    simsiam.eval()
    
    correct = 0
    total = 0
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
    model = resnet18(pretrained=True)
    # Replace the last layer with a linear layer for CIFAR-10
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)

    # Train the linear layer (only input model.fc parameters into optimizer)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.fc.parameters(), lr=lr, momentum=momentum)

    
    global_progress = tqdm(range(0, opt.epochs), desc=f'Training Supervised Model')
    accuracies = []
    for epoch in global_progress:  # loop over the dataset multiple times
        correct = 0
        total = 0
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        #     # print statistics
        #     # running_loss += loss.item()
        #     _, predicted = torch.max(outputs.data, 1)
        #     total += labels.size(0)
        #     correct += (predicted == labels).sum().item()
   
        # # print accuracy after every epoch
        # acc = 100 * correct / total
        # accuracies.append(acc)
        # print(f'Accuracy after epoch {epoch + 1}: {acc:.2f}%')

        # eval
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct // total
        accuracies.append(accuracy)
        print(f'Validation accuracy of the supervised network on test images: {accuracy} %')


    plt.plot(accuracies)
    plt.ylim(0, 100)
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.savefig(f"supervised_evaluation_{opt.data_percentage}.png")


    # eval
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct // total
    print(f'Accuracy of the supervised network on test images: {accuracy} %')

