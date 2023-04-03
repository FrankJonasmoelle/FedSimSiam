from SimSiam.simsiam.datapreparation import *
from SimSiam.simsiam.simsiam import *
from SimSiam.simsiam.utils import *
from SimSiam.simsiam.evaluation import *

from SimSiam.federated_simsiam.client import *
from SimSiam.federated_simsiam.server import *
from SimSiam.federated_simsiam.datapreparation import *
from SimSiam.federated_simsiam.evaluation import *
import argparse


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--num_clients', type=int, default=2, help='number of clients')
    parser.add_argument('--iid', type=bool, default=False, help='split dataset iid or not')
    parser.add_argument('--num_rounds', type=int, default=1, help='number of training rounds')
    parser.add_argument('--local_epochs', type=int, default=5, help='number of client epochs for training')
    parser.add_argument('--output_path', type=str, default='fedavg_simsiam.pth')

    opt = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    server = Server(num_clients=opt.num_clients, iid=opt.iid, num_rounds=opt.num_rounds, local_epochs=opt.local_epochs)
    # trains the federated model
    server.learn_federated_simsiam()
    # save model
    PATH = opt.output_path
    torch.save(server.model.state_dict(), PATH)

    #####################
    # FedAVG evaluation
    #####################
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    averaged_model = server.model
    averaged_model = DownstreamEvaluation(averaged_model)
    averaged_model = averaged_model.to(device)


    # optimizer = optim.SGD(averaged_model.parameters(), lr=0.001, momentum=0.9)    
    # criterion = nn.CrossEntropyLoss()

    # trainloader, testloader = get_downstream_data(percentage_of_data=0.1, batch_size=32)
    # train_downstream(5, averaged_model, trainloader, criterion, optimizer, device)
    # acc_avg = evaluate_downstream(averaged_model, testloader, device)
    # print("accuracy fedavg: ", acc_avg)

    # #################################
    # # comparison to supervised model
    # #################################
    # model = SupervisedModel(pretrained=True, linearevaluation=True)
    # model = model.to(device)

    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # train_downstream(5, model, trainloader, criterion, optimizer, device)
    # acc_supervised = evaluate_downstream(model, testloader, device)
    # print("accuracy supervised model: ", acc_supervised)