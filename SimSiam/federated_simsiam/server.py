import torch
import copy
from collections import OrderedDict

from ..simsiam.simsiam import *
from ..simsiam.evaluation import *
from .datapreparation import *
from .client import *


class Server:
    def __init__(self, num_clients, iid, num_rounds, local_epochs, batch_size, alpha=0.5):
        self.num_clients = num_clients
        self.iid = iid
        self.num_rounds = num_rounds # number of rounds that models should be trained on clients
        self.local_epochs = local_epochs # number of epochs each client is trained per round
        self.batch_size = batch_size
        self.alpha = alpha

    def setup(self):
        self.model = SimSiam()
        local_trainloaders, test_loader = create_datasets(self.num_clients, self.iid, self.batch_size, self.alpha)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.clients = self.create_clients(local_trainloaders)
        self.testloader = test_loader
        self.send_model()
        
    def create_clients(self, local_trainloaders):
        clients = []
        for i, dataloader in enumerate(local_trainloaders):
            client = Client(client_id=i, model=self.model, dataloader=dataloader, local_epochs=self.local_epochs)
            clients.append(client)
        return clients

    def send_model(self):
        """Send the updated global model to selected/all clients."""
        for client in self.clients:
            client.model = copy.deepcopy(self.model)

    def average_model(self, coefficients):
        """Average the updated and transmitted parameters from each selected client."""
        averaged_weights = OrderedDict()

        for i, client in enumerate(self.clients):
            local_weights = client.model.state_dict()

            for key in self.model.state_dict().keys():
                if i == 0:
                    averaged_weights[key] = coefficients[i] * local_weights[key]
                else:
                    averaged_weights[key] += coefficients[i] * local_weights[key]
        self.model.load_state_dict(averaged_weights)


    def train_federated_model(self):
        # send current model
        self.send_model()
        
        # TODO: Sample only subset of clients

        # update clients (train client models)
        for client in self.clients:
            client.client_update()
        
        # average models
        total_size = sum([len(client.dataloader.dataset[1]) for client in self.clients])
        mixing_coefficients = [len(client.dataloader.dataset[1]) / total_size for client in self.clients]
        self.average_model(mixing_coefficients)
    
    
    def learn_federated_simsiam(self):
        self.setup()
        for i in range(self.num_rounds):
            self.train_federated_model()
            # downstream_accuracy = self.evaluate_global_model(num_epochs=5) 
            # print(downstream_accuracy)
        # save final averaged model
        PATH = "simsiam_fedavg.pth"
        torch.save(self.model.state_dict(), PATH)
        self.send_model()