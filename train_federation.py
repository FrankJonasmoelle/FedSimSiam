from SimSiam.simsiam.datapreparation import *
from SimSiam.simsiam.simsiam import *
from SimSiam.simsiam.utils import *
from SimSiam.simsiam.evaluation import *

from SimSiam.federated_simsiam.client import *
from SimSiam.federated_simsiam.server import *
from SimSiam.federated_simsiam.datapreparation import *
import argparse


if __name__=="__main__":
    """
    python3 train_federation.py --num_clients 5 --alpha 0.5 --num_rounds 30 --local_epochs 25 --batch_size 64 --output_path 'fedca_iid_5_30_25.pth'
    """
    parser = argparse.ArgumentParser()  
    
    parser.add_argument('--num_clients', type=int, default=2, help='number of clients')
    parser.add_argument('--alpha', type=float, default=0.5, help='alpha determines level of non-iid-nes')
    parser.add_argument('--num_rounds', type=int, default=1, help='number of training rounds')
    parser.add_argument('--local_epochs', type=int, default=5, help='number of client epochs for training')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size for client training')
    parser.add_argument('--output_path', type=str, default='fedavg_simsiam.pth')

    opt = parser.parse_args()

    IID = True

    server = Server(num_clients=opt.num_clients, iid=IID, output_path=opt.output_path, num_rounds=opt.num_rounds, 
                    local_epochs=opt.local_epochs, batch_size=opt.batch_size)
    # trains the federated model
    server.learn_federated_simsiam()
    # save model
    PATH = opt.output_path
    torch.save(server.model.state_dict(), PATH)