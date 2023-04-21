from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torchvision
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
import numpy as np

class MyDataset(Dataset):
    def __init__(self, x, y, is_train=True):#, transform_x=None):
        self.x = x
        self.y = y
      #  self.transform_x = transform_x
        self.is_train = is_train

    def __getitem__(self, idx):
        x = self.x[idx]
        x = Image.fromarray(x.astype(np.uint8))

        y = self.y[idx]

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        augmentation = [
            transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
            ]

        if self.is_train:
            transform = transforms.Compose(augmentation)

            x1 = transform(x)
            x2 = transform(x)
            return [x1, x2], y

        else:
            transform=transforms.Compose([transforms.ToTensor(), normalize])

            x = transform(x)
            return x, y
    
    def __len__(self):
        return len(self.x)
    
def load_data_iid(trainset, num_clients, batch_size):
    shuffled_indices = torch.randperm(len(trainset))
    training_x = trainset.data[shuffled_indices]
    training_y = torch.Tensor(trainset.targets)[shuffled_indices]

    split_size = len(trainset) // num_clients
    split_datasets = list(
                        zip(
                            torch.split(torch.Tensor(training_x), split_size),
                            torch.split(torch.Tensor(training_y), split_size)
                        )
                    )
    new_split_datasets = [(dataset[0].numpy(), dataset[1].tolist()) for dataset in split_datasets]
    new_split_datasets = [(dataset[0], list(map(int, dataset[1]))) for dataset in new_split_datasets]

    local_trainset = [MyDataset(local_dataset[0], local_dataset[1], is_train=True) for local_dataset in new_split_datasets]

    local_dataloaders = [DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True) for dataset in local_trainset]
    return local_dataloaders


def sample_images(class_to_indices, proportions, n):
    indices = []
    for c, prop in enumerate(proportions):
        num_samples = int(n * prop)
        num_samples = min(num_samples, len(class_to_indices[c]))
        samples = np.random.choice(class_to_indices[c], size=num_samples, replace=False)
        indices.extend(samples)
    return indices


def load_data_non_iid(trainset, num_clients, batch_size, alpha=0.5):
    num_classes = 10
    class_to_indices = {i: [] for i in range(num_classes)}
    for idx, (_, label) in enumerate(trainset):
        class_to_indices[label].append(idx)

    # Distribute the dataset in a non-iid way onto clients
    client_indices = {i: [] for i in range(num_clients)}
    n_per_client = len(trainset) // num_clients

    for i in range(num_clients):
        # Sample a proportion vector from the Dirichlet distribution
        proportions = np.random.dirichlet([alpha] * num_clients)
        # Sample images from each class based on the proportion vector
        sampled_indices = sample_images(class_to_indices, proportions, n_per_client)
        print(len(sampled_indices))
        # Create a subset of the dataset for the client
        client_indices[i] = sampled_indices

    xs = []
    ys = []
    for i in range(num_clients):
        training_x = trainset.data[client_indices[i]]
        training_y = torch.Tensor(trainset.targets)[client_indices[i]]
        xs.append(training_x)
        ys.append(training_y)

    local_trainset = [MyDataset(xs[i], ys[i], is_train=True) for i in range(num_clients)]
    local_dataloaders = [DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True) for dataset in local_trainset]
    return local_dataloaders


def create_datasets(num_clients, iid, batch_size, alpha):
    """Split the whole dataset in IID or non-IID manner for distributing to clients."""

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, 
                                           transform=transforms.Compose([transforms.ToTensor(), normalize]))

    if iid:
        local_dataloaders = load_data_iid(trainset, num_clients, batch_size)
    else: 
        local_dataloaders = load_data_non_iid(trainset, num_clients, batch_size, alpha)


    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2, pin_memory=True)
    return local_dataloaders, testloader