import torch
import copy
from collections import OrderedDict
import matplotlib.pyplot as plt

from ..simsiam.simsiam import *
from ..simsiam.evaluation import *
from ..simsiam.monitoring import *
from ..simsiam.datapreparation import prepare_data
from .datapreparation import *
from .client import *


class Server:
    def __init__(self, num_clients, iid, output_path, num_rounds, local_epochs, batch_size, alpha=0.5):
        self.num_clients = num_clients
        self.iid = iid
        self.output_path = output_path 
        self.num_rounds = num_rounds # number of rounds that models should be trained on clients
        self.local_epochs = local_epochs # number of epochs each client is trained per round
        self.batch_size = batch_size
        self.alpha = alpha

    def setup(self):
        self.model = SimSiam()
        self.alignmentmodel = SimSiam()
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

    def send_alignment(self, alignmentset, alignmentmodel):
        for client in self.clients:
            client.alignmentset = alignmentset
            client.alignmentmodel = alignmentmodel

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

        # calculate K_mean
        if self.clients[0].K is not None: # skip if K hasn't been calculated yet (first iteration)
            list_of_Ks = []
            for client in self.clients:
                list_of_Ks.append(client.K)

            K_mean = np.mean(list_of_Ks, axis=0)
            print(K_mean)

            # send K_mean (averaged feature matrix across clients)
            for client in self.clients:
                client.K_mean = K_mean
            
        # update clients (train client models)
        for client in self.clients:
            client.client_update()
        
        # average models
        total_size = sum([len(client.dataloader.dataset[1]) for client in self.clients])
        mixing_coefficients = [len(client.dataloader.dataset[1]) / total_size for client in self.clients]
        self.average_model(mixing_coefficients)
    
    
    def learn_federated_simsiam(self):
        knn_accuracies = [0]
        _, memoryloader, testloader = prepare_data(batch_size=self.batch_size)

        self.setup()

        #self.train_alignment_model(sample_size=3000, subset_size=50, epochs=20)
        self.alignmentset, self.sub_alignmentset = alignment_dataset(self.batch_size, sample_size=50, subset_size=50)
        self.send_alignment(alignmentset=self.sub_alignmentset, alignmentmodel=self.alignmentmodel)
        
        for i in range(self.num_rounds):
            print(f"Round {i+1}")
            self.train_federated_model()
            # knn monitoring after each round
            accuracy = knn_monitor(self.model.encoder, memoryloader, testloader, self.device, k=min(25, len(memoryloader.dataset))) 
            knn_accuracies.append(accuracy)

            # save final averaged model
            torch.save(self.model.state_dict(), self.output_path)
            
            # plot knn
            plt.plot(knn_accuracies)
            plt.ylim(0, 100)
            plt.xlabel("round")
            plt.ylabel("accuracy")
            plt.savefig(f"knn_accuracy_fedavg_{self.iid}_{self.num_clients}_{self.num_rounds}_{self.local_epochs}.png")
