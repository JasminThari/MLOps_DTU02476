import torch
from torch.utils.data import DataLoader, TensorDataset
import os

def mnist():
    """Return train and test dataloaders for MNIST."""
    #print(os.getcwd())
    folder_path = "/Users/jasmink.j.thari/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/MSc HCAI/1.Sem/MLOps/dtu_mlops/data/corruptmnist"
    os.chdir(folder_path)
    folder_path = os.getcwd()
    #print(os.getcwd())
    # Initialize lists to store training images, training targets, and testing data
    data_train_images = []
    data_train_targets = []
    data_test_images = None
    data_test_targets = None

    # Load training data
    for i in range(6):
        file_name_images = f'train_images_{i}.pt'
        file_name_targets = f'train_target_{i}.pt'
        file_path_images = f'{folder_path}/{file_name_images}'
        file_path_targets = f'{folder_path}/{file_name_targets}'

        data_images = torch.load(file_path_images)
        data_targets = torch.load(file_path_targets)

        data_train_images.append(data_images)
        data_train_targets.append(data_targets)

    # Load testing data
    file_name_images_test = 'test_images.pt'
    file_name_targets_test = 'test_target.pt'
    file_path_images_test = f'{folder_path}/{file_name_images_test}'
    file_path_targets_test = f'{folder_path}/{file_name_targets_test}'

    data_test_images = torch.load(file_path_images_test)
    data_test_targets = torch.load(file_path_targets_test)

    # Concatenate and convert the loaded data into tensors
    train_images = torch.cat(data_train_images, dim=0)
    train_targets = torch.cat(data_train_targets, dim=0)

    # Create DataLoader instances for training and testing datasets
    batch_size = 64  # Adjust as needed
    train_dataset = TensorDataset(train_images, train_targets)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(data_test_images, data_test_targets)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
    return train_loader, test_loader