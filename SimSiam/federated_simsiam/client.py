import torch
import torch.optim as optim
from tqdm import tqdm

from ..simsiam.simsiam import D

class Client:
    def __init__(self, client_id, model, dataloader, local_epochs):
        self.client_id = client_id
        self.dataloader = dataloader
        self.model = model
        self.local_epochs = local_epochs
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def client_update(self):
        self.model.train()
        self.model.to(self.device)
        optimizer = optim.SGD(self.model.parameters(), lr=0.03, momentum=0.9, weight_decay=0.0005)

        global_progress = tqdm(range(0, self.local_epochs), desc=f'Training')
        for epoch in global_progress:
            self.model.train()
            
            local_progress=tqdm(self.dataloader, desc=f'Epoch {epoch}/{opt.epochs}')
            for idx, data in enumerate(local_progress):
                images = data[0]
                optimizer.zero_grad()
                data_dict = self.model.forward(images[0].to(self.device, non_blocking=True), images[1].to(self.device, non_blocking=True))
                loss = data_dict['loss'].mean()

                loss.backward()
                optimizer.step()
                # lr_scheduler.step()
                
                local_progress.set_postfix(data_dict)

            #if args.train.knn_monitor and epoch % args.train.knn_interval == 0: 
            #accuracy = knn_monitor(model.encoder, memory_loader, test_loader, device, k=min(25, len(memory_loader.dataset))) 
            
            epoch_dict = {"epoch":epoch}
            global_progress.set_postfix(epoch_dict)

    def client_evaluate(self):
        """evaluates model on local dataset TODO: Should this be done in self-supervised learning and if so, how?"""
        # insert evaluate() method of SimSiam
        pass