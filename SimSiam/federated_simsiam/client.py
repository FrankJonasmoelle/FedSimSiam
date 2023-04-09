import torch
import torch.optim as optim

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

        for epoch in range(self.local_epochs):  # loop over the dataset multiple times
            epoch_loss = 0.0
            running_loss = 0.0
            for i, data in enumerate(self.dataloader):            
                # get the inputs; data is a list of [inputs, labels]
                # inputs, labels = data
                images, labels = data[0], data[1].to(self.device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # get the two views (with random augmentations):
                x1 = images[0].to(self.device)
                x2 = images[1].to(self.device)
                
                # forward + backward + optimize
                z1, p1 = self.model(x1)
                z2, p2 = self.model(x2)
                #loss = criterion(outputs, labels)
                loss = D(p1, z2)/2 + D(p2, z1)/2
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                epoch_loss += loss.item()
                if i % 100 == 99:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0
            print("epoch loss = ", epoch_loss/len(self.dataloader))
        print('Finished Training')

    def client_evaluate(self):
        """evaluates model on local dataset TODO: Should this be done in self-supervised learning and if so, how?"""
        # insert evaluate() method of SimSiam
        pass