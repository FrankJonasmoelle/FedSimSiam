import torch
import torchvision
import torchvision.transforms as transforms


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


def prepare_data(batch_size=64):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

    # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
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
    #train_set, val_set = torch.utils.data.random_split(trainset, [45000, 5000])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=TwoCropsTransform(transforms.Compose(augmentation)))
    # memoryset used for knn monitoring
    memoryset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transforms.Compose([transforms.ToTensor(), normalize]))
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transforms.Compose([transforms.ToTensor(), normalize]))

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    memoryloader = torch.utils.data.DataLoader(memoryset, batch_size=batch_size, shuffle=False, num_workers=4)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

    return trainloader, memoryloader, testloader