from SimSiam.simsiam.datapreparation import *
from SimSiam.simsiam.simsiam import *
from SimSiam.simsiam.utils import *
from SimSiam.simsiam.evaluation import *

from SimSiam.federated_simsiam.client import *
from SimSiam.federated_simsiam.server import *
from SimSiam.federated_simsiam.datapreparation import *
from SimSiam.federated_simsiam.evaluation import *
import argparse
import logging

logging.basicConfig(filename='logging.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

if __name__=="__main__":
    """ 
    python3 evaluation_comparison.py --data_percentage 0.01 --num_epochs 5 --batch_size 32 --simsiam_path 'simsiam_25epochs_128bs.pth'
    
    """
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_percentage', type=int, default=0.01, help='percentage of data used for training')
    parser.add_argument('--num_epochs', type=int, default=5, help="number of epochs used for downstream training")
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--simsiam_path', type=str, default="simsiam.pth", help='path to trained simsiam model')
    parser.add_argument('--fedavg_simsiam_path', type=str, default="fedavg_simsiam.pth", help='path to trained fedaveraged simsiam model')

    opt = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    criterion = nn.CrossEntropyLoss()
    lr = 0.001
    momentum = 0.9

    # cifar-10 data for classification
    trainloader, testloader = get_downstream_data(percentage_of_data=opt.data_percentage, batch_size=opt.batch_size)

    ####################################
    # linear evaluation supervised model
    ####################################
    supervised_model = SupervisedModel(pretrained=True, linearevaluation=False)
    supervised_model = supervised_model.to(device)
    optimizer = optim.SGD(supervised_model.parameters(), lr=lr, momentum=momentum)

    supervised_model = train_downstream(opt.num_epochs, supervised_model, trainloader, criterion, optimizer, device)
    supervised_accuracy = evaluate_downstream(supervised_model, testloader, device)
    
    print("accuracy for supervised model: ", supervised_accuracy)
    logging.info("accuracy for supervised model: ", supervised_accuracy)
    ####################################
    # linear evaluation standard SimSiam
    ####################################
    simsiam_model = SimSiamDownstream(trained_model_path=opt.simsiam_path, device=device, linearevaluation=False)
    simsiam_model = simsiam_model.to(device)
    
    optimizer = optim.SGD(simsiam_model.parameters(), lr=lr, momentum=momentum)
    simsiam_model = train_simsiam_downstream(opt.num_epochs, simsiam_model, trainloader, criterion, optimizer, device)
    simsiam_accuracy = evaluate_simsiam_downstream(simsiam_model, testloader, device)

    print("accuracy for standard simsiam: ", simsiam_accuracy)
    logging.info("accuracy for standard simsiam: ", simsiam_accuracy)

    ##################################
    # linear evaluation fedavg SimSiam
    ##################################
    # fedavg_simsiam = torch.load(opt.fedavg_simsiam_path)
    # fedavg_simsiam = DownstreamEvaluation(fedavg_simsiam)
    # fedavg_simsiam = fedavg_simsiam.to(device)
    # optimizer = optim.SGD(fedavg_simsiam.parameters(), lr=lr, momentum=momentum)  
    # fedavg_simsiam = train_downstream(opt.num_epochs, fedavg_simsiam, trainloader, criterion, optimizer, device)
    # fedavg_simsiam_accuracy = evaluate_downstream(fedavg_simsiam, testloader, device)

    # print("accuracy for fedAVG simsiam: ", fedavg_simsiam_accuracy)
    # logging.info("accuracy for fedAVG simsiam: ", fedavg_simsiam_accuracy)
