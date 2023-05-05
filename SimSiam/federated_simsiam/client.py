import torch
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F 
import numpy as np

from ..simsiam.simsiam import D
from ..optimizers import *
from .cka import linear_CKA

class Client:
    def __init__(self, client_id, model, dataloader, local_epochs):
        self.client_id = client_id
        self.dataloader = dataloader
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = model.to(self.device)
        self.local_epochs = local_epochs
        self.alignmentset = None
        self.K = None
        self.K_mean = None

    def client_update(self):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(device)
        self.alignmentmodel.to(device)
        self.model.train()
        
        # optimizer = optim.SGD(self.model.parameters(), lr=0.03, momentum=0.9, weight_decay=0.0005)

        # fixed parameters
        warmup_epochs = 10
        warmup_lr = 0
        base_lr = 0.03
        final_lr = 0
        momentum = 0.9
        weight_decay = 0.0005
        batch_size = 64

        optimizer = get_optimizer(
            'sgd', self.model, 
            lr=base_lr*batch_size/256, 
            momentum=momentum,
            weight_decay=weight_decay)

        lr_scheduler = LR_Scheduler(
            optimizer, warmup_epochs, warmup_lr*batch_size/256, 
            self.local_epochs, base_lr*batch_size/256, final_lr*batch_size/256, 
            len(self.dataloader),
            constant_predictor_lr=True
        )

        global_progress = tqdm(range(0, self.local_epochs), desc=f'Training client {self.client_id + 1}')
        for epoch in global_progress:
            self.model.train()

            local_progress=tqdm(self.dataloader, desc=f'Epoch {epoch + 1}/{self.local_epochs}')
            for idx, data in enumerate(local_progress):
                images = data[0]
                optimizer.zero_grad()
                data_dict = self.model.forward(images[0].to(self.device, non_blocking=True), images[1].to(self.device, non_blocking=True))
                loss = data_dict['loss'].mean()

                features_list = []
                with torch.no_grad():
                    for idx, data in enumerate(self.alignmentset):
                        images = data[0]
                        embedding = self.model.encoder(images[0].cuda(non_blocking=True))
                        features_list.append(embedding.cpu().numpy())
                       
                self.K = np.vstack(features_list)

                if self.K_mean is None: # skip for first iteration as K_mean cannot be calculated
                    total_loss = loss
                else:
                    mu = 0.5
                    linear_CKA_score = linear_CKA(self.K, self.K_mean) # linear_CKA outputs score between 0 (similar) and 1 (dissimilar)
                    total_loss = loss + mu*linear_CKA_score  

                # print('total loss: ', total_loss)
                total_loss.backward()
                optimizer.step()
                lr_scheduler.step()
                
                data_dict['loss'] = total_loss
                local_progress.set_postfix(data_dict)
            
            epoch_dict = {"epoch":epoch}
            global_progress.set_postfix(epoch_dict)

    def client_evaluate(self):
        """evaluates model on local dataset TODO: Should this be done in self-supervised learning and if so, how?"""
        # insert evaluate() method of SimSiam
        pass