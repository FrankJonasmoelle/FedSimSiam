import argparse

from SimSiam.simsiam.datapreparation import *
from SimSiam.simsiam.simsiam import *
from SimSiam.simsiam.utils import *
from SimSiam.simsiam.evaluation import *

from SimSiam.federated_simsiam.client import *
from SimSiam.federated_simsiam.server import *
from SimSiam.federated_simsiam.datapreparation import *


def linear_evaluation_supervised(trainloader, testloader, criterion, lr, momentum):
    """Trains and evaluates a supervised model on CIFAR-10"""

    model = SupervisedModel(pretrained=True, linearevaluation=True)
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    model = train_downstream(opt.num_epochs, model, trainloader, criterion, optimizer, device)
    accuracy = evaluate_downstream(model, testloader, device)
    return accuracy

def linear_evaluation_simsiam(trained_model_path, trainloader, testloader, criterion, lr, momentum):
    """Trains and evaluates SimSIam on CIFAR-10"""

    simsiam = SimSiamDownstream(trained_model_path=trained_model_path, device=device, linearevaluation=True)
    simsiam = simsiam.to(device)
    optimizer = optim.SGD(simsiam.parameters(), lr=lr, momentum=momentum)
    simsiam = train_simsiam_downstream(opt.num_epochs, simsiam, trainloader, criterion, optimizer, device)
    accuracy = evaluate_simsiam_downstream(simsiam, testloader, device)
    return accuracy


if __name__=="__main__":
    """ 
    python3 evaluation_comparison.py --data_percentage 0.1 --num_epochs 5 --batch_size 32 --simsiam_path 'models/simsiam.pth' --fedavg_simsiam_path 'models/simsiam_fedavg.pth'

    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    criterion = nn.CrossEntropyLoss()
    lr = 0.001
    momentum = 0.9

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_percentage', type=float, default=0.01, help='percentage of data used for training')
    parser.add_argument('--num_epochs', type=int, default=5, help="number of epochs used for downstream training")
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--simsiam_path', type=str, default="simsiam.pth", help='path to trained simsiam model')
    parser.add_argument('--fedavg_simsiam_path', type=str, default="fedavg_simsiam.pth", help='path to trained fedaveraged simsiam model')

    opt = parser.parse_args()

    # cifar-10 data for classification
    trainloader, testloader = get_downstream_data(percentage_of_data=opt.data_percentage, batch_size=opt.batch_size)

    # linear evaluation supervised model
    supervised_accuracy = linear_evaluation_supervised(trainloader, testloader, criterion, lr, momentum)
    print("accuracy for supervised model: ", supervised_accuracy)

    # linear evaluation SimSiam
    simsiam_accuracy = linear_evaluation_simsiam(opt.simsiam_path, trainloader, testloader, criterion, lr, momentum)
    print("accuracy for standard simsiam: ", simsiam_accuracy)
    
    # linear evaluation fedavg SimSiam
    fedavg_simsiam_accuracy = linear_evaluation_simsiam(opt.fedavg_simsiam_path, trainloader, testloader, criterion, lr, momentum)
    print("accuracy for fedAVG simsiam: ", fedavg_simsiam_accuracy)
