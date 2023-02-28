import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def calculate_nn(image_id, model, testset):
  """takes in image_id of testset and calculates distances to all images"""
  image, label = testset[image_id]
  image = image.view(1,3,32,32)
  z, _ = model(image) # f(z)
  nn_dict = {}
  tracker = 0
  with torch.no_grad():
      for i in range(len(testset)):  
        test_image, label = testset[i]
        test_image = test_image.view(1,3,32,32)
        z_test, _ = model(test_image) # f(z)
        dist = torch.norm(z_test - z, dim=1, p=None)
        nn_dict[f"{tracker}"] = dist.float()
        tracker += 1
  sorted_nn_dict = sorted(nn_dict.items(), key=lambda kv: kv[1], reverse=False)
  return sorted_nn_dict

def visualize_knn(image_id, testset, k=3):
  """visualizes original image and its k nearest neighbors in the embedding space"""
  sorted_nn_dict = calculate_nn(image_id)
  ls_of_img_tensors = []
  for i in range(k+1):
    id, dist = sorted_nn_dict[i]
    img, label = testset[int(id)]
    ls_of_img_tensors.append(img)
  img_tensor = torch.stack(ls_of_img_tensors)
  imshow(torchvision.utils.make_grid(img_tensor))



def calculate_knn_accuracy(model, testset, k=3):
    true_predicted = 0
    total_predicted = 0
    distance_memory = np.zeros((len(testset), len(testset))) # use dynamic programming

    with torch.no_grad():
        for i in range(len(testset)):
            curr_image, curr_label = testset[i]
            curr_image = curr_image.view(1,3,32,32)
            z_curr, _ = model(curr_image) # f(z)
            for j in range(len(testset)):
                if distance_memory[i][j] == 0: # fill in memory
                    comp_image, comp_label = testset[j]
                    comp_image = comp_image.view(1,3,32,32)
                    z_comp, _ = model(comp_image) # f(z)
                    dist = torch.norm(z_curr - z_comp, dim=1, p=None)
                    distance_memory[i][j] = dist.float()
                else:
                  dist = distance_memory[i][j]

    return distance_memory