import torch
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F 

from ..simsiam.simsiam import D
from ..optimizers import *

class Client:
    def __init__(self, client_id, model, dataloader, local_epochs):
        self.client_id = client_id
        self.dataloader = dataloader
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = model.to(self.device)
        self.local_epochs = local_epochs
        self.alignmentmodel = None
        self.alignmentset = None

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

                #loss_align = 0
                cos_similarities = []
                for idx, data in enumerate(self.alignmentset):
                    images = data[0]
                    embedding_localmodel = self.model.encoder(images[0].cuda(non_blocking=True))
                    embedding_alignmentmodel = self.alignmentmodel.encoder(images[0].cuda(non_blocking=True))
                    # normalize embedding
                    embedding_localmodel = F.normalize(embedding_localmodel, dim=1)
                    embedding_alignmentmodel = F.normalize(embedding_alignmentmodel, dim=1)

                    # l2_norm = torch.norm(embedding_alignmentmodel - embedding_localmodel, p=2)
                    #loss_align += l2_norm
                    
                    # Try cosine similarity
                    cos_similarity = - F.cosine_similarity(embedding_localmodel, embedding_alignmentmodel, dim=-1).mean()
                    cos_similarities.append(cos_similarity)
                cos_similarity = sum(cos_similarities) / len(cos_similarities)
                
                # print('normal loss: ', loss)
                # print('loss_align :', loss_align)

                beta = 0.5
                #total_loss = loss + beta*loss_align
                total_loss = 1/2 * (loss + beta*cos_similarity)
                # print('total loss: ', total_loss)
                total_loss.backward()
                optimizer.step()
                lr_scheduler.step()
                
                data_dict['loss'] = total_loss
                local_progress.set_postfix(data_dict)

            #if args.train.knn_monitor and epoch % args.train.knn_interval == 0: 
            #accuracy = knn_monitor(model.encoder, memory_loader, test_loader, device, k=min(25, len(memory_loader.dataset))) 
            
            epoch_dict = {"epoch":epoch}
            global_progress.set_postfix(epoch_dict)

    def client_evaluate(self):
        """evaluates model on local dataset TODO: Should this be done in self-supervised learning and if so, how?"""
        # insert evaluate() method of SimSiam
        pass