from sklearn.neighbors import KNeighborsClassifier
import torchvision
import torchvision.transforms as transforms

from .simsiam import *
from .evaluation import *
from .datapreparation import *
from .utils import *



def get_knn_monitoring_dataset():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

    trainset = torchvision.datasets.CIFAR10(root='./SimSiam/data', train=True,
                                            download=True, transform=transforms.Compose([transforms.ToTensor(), normalize]))

    train_set, val_set = torch.utils.data.random_split(trainset, [45000, 5000])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=len(train_set), 
                                                shuffle=True)

    val_loader = torch.utils.data.DataLoader(val_set, batch_size=len(val_set), 
                                                shuffle=True)

    # Step 3: Initialize an empty tensor
    train_images_tensor = torch.Tensor()
    train_labels_tensor = torch.Tensor()

    val_images_tensor = torch.Tensor()
    val_labels_tensor = torch.Tensor()

    # Step 4: Concatenate images into a single tensor
    for images, labels in train_loader:
        train_images_tensor = torch.cat((train_images_tensor, images), 0)
        train_labels_tensor = torch.cat((train_labels_tensor, labels), 0)

    for images, labels in val_loader:
        val_images_tensor = torch.cat((val_images_tensor, images), 0)
        val_labels_tensor = torch.cat((val_labels_tensor, labels), 0)
        
    return train_images_tensor, train_labels_tensor, val_images_tensor, val_labels_tensor


def knn_monitor(model, train_images_tensor, train_labels_tensor, val_images_tensor, val_labels_tensor, k=5):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # randomly sample subset of train and validation data
    num_samples = 500
    train_indices = torch.randperm(len(train_images_tensor))[:num_samples]
    val_indices = torch.randperm(len(val_images_tensor))[:num_samples]

    train_images_tensor = train_images_tensor[train_indices]
    train_labels_tensor = train_labels_tensor[train_indices]

    val_images_tensor = val_images_tensor[val_indices]
    val_labels_tensor = val_labels_tensor[val_indices]

    train_images_tensor = train_images_tensor.to(device)
    val_images_tensor = val_images_tensor.to(device)
    
    # Extract features from the SimSiam model
    train_embeddings, _ = model(train_images_tensor)
    val_embeddings, _ = model(val_images_tensor)

    # Initialize a KNN classifier
    knn = KNeighborsClassifier(n_neighbors=k)

    # Train the KNN classifier on the training set embeddings
    knn.fit(train_embeddings.detach().cpu().numpy(), train_labels_tensor.detach().cpu().numpy())

    # Compute the KNN accuracy on the validation set
    knn_acc = knn.score(val_embeddings.detach().cpu().numpy(), val_labels_tensor.detach().cpu().numpy())
    return knn_acc