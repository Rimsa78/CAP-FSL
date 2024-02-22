import torch 
import numpy as np 
import os 
import random 
from torch.utils.data import Dataset, DataLoader 
from setup import reset_seed

reset_seed()

class TestThreeWayDataset(Dataset):
    def __init__(self, normal_folder, amd_folder, glaucoma_folder):
        """
        Custom dataset for testing a three-way image classifier (normal, amd, glaucoma).
        """
        self.normal_files = [os.path.join(normal_folder, f) for f in os.listdir(normal_folder) if f.endswith('.npy')]
        self.amd_files = [os.path.join(amd_folder, f) for f in os.listdir(amd_folder) if f.endswith('.npy')]
        self.glaucoma_files = [os.path.join(glaucoma_folder, f) for f in os.listdir(glaucoma_folder) if f.endswith('.npy')]
        
        self.all_files = self.normal_files + self.amd_files + self.glaucoma_files 
        self.labels = [0] * len(self.normal_files) + [1] * len(self.amd_files) + [2] * len(self.glaucoma_files) 
        
    def __len__(self):
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

class TestTaskSampler:
    def __init__(self, dataset, n_way, k_shot, q_query):
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query

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


class TestMetaLoader(DataLoader):
    def __init__(self, task_sampler, tasks_per_batch):
        self.task_sampler = task_sampler
        self.tasks_per_batch = tasks_per_batch

    def __iter__(self):
        for _ in range(self.tasks_per_batch):
            task_data, task_labels = self.task_sampler.sample_task()
            yield task_data, task_labels
