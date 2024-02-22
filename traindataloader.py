import torch 
import numpy as np 
import os 
import random 
from torch.utils.data import Dataset, DataLoader 
from setup import reset_seed

reset_seed()

class TrainThreeWayDataset(Dataset):
    """
    Custom dataset for training a three-way image classifier (normal, amd, glaucoma).
    """
    def __init__(self, normal_folder, amd_folder, glaucoma_folder, shuffle=True):
        self.normal_files = [os.path.join(normal_folder, f) for f in os.listdir(normal_folder) if f.endswith('.npy')]
        self.amd_files = [os.path.join(amd_folder, f) for f in os.listdir(amd_folder) if f.endswith('.npy')]
        self.glaucoma_files = [os.path.join(glaucoma_folder, f) for f in os.listdir(glaucoma_folder) if f.endswith('.npy')]
        
        if shuffle:
            random.shuffle(self.normal_files)
            random.shuffle(self.amd_files)
            random.shuffle(self.glaucoma_files)
        
        self.all_files = self.normal_files + self.amd_files + self.glaucoma_files 
        self.labels = [0] * len(self.normal_files) + [1] * len(self.amd_files) + [2] * len(self.glaucoma_files) 
        
    def __len__(self):
        """ Returns the total number of images in the dataset. """
        return len(self.all_files)

    def __getitem__(self, idx):
        """
        Loads and returns a single image and its label.
        Returns:
            tuple: (image_tensor, label)
        """
        image_path = self.all_files[idx]
        image_array = np.load(image_path)
        image_array = np.transpose(image_array, (0, 2, 1))
        image_tensor = torch.tensor(image_array, dtype=torch.float32)
        label = self.labels[idx]
        return image_tensor, label

class TrainTaskSampler:
    """
   Samples tasks for few-shot meta-learning. Each task contains 'n_way' classes, 
   with 'k_shot' support examples and 'q_query' query examples per class.
   """
    def __init__(self, dataset, n_way, k_shot, q_query, shuffle=True):
        """
        Args:
            dataset (TrainThreeWayDataset): The dataset to sample from.
            n_way (int): Number of classes per task.
            k_shot (int): Number of support examples per class.
            q_query (int): Number of query examples per class.
            shuffle (bool, optional):  If True, shuffles data before sampling. Defaults to True.
        """
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.shuffle = shuffle

    def sample_task(self):
        task_data = []
        task_labels = []
        class_labels = np.unique(self.dataset.labels)
        selected_classes = np.random.choice(class_labels, self.n_way, replace=False)

        for i, c in enumerate(selected_classes):
            indices = np.where(np.array(self.dataset.labels) == c)[0]
            selected_indices = np.random.choice(indices, self.k_shot + self.q_query, replace=False)
            for j, ind in enumerate(selected_indices):
                img, _ = self.dataset[ind]
                task_data.append(img)
                task_labels.append(i)

        return torch.stack(task_data), torch.tensor(task_labels)


class TrainMetaLoader(DataLoader):
    def __init__(self, task_sampler, tasks_per_batch):
        self.task_sampler = task_sampler
        self.tasks_per_batch = tasks_per_batch

    def __iter__(self):
        for _ in range(self.tasks_per_batch):
            task_data, task_labels = self.task_sampler.sample_task()
            yield task_data, task_labels

