# -*- coding: utf-8 -*-
import torch
import torch.optim as optim
import time
import gc
from traindataloader import TrainThreeWayDataset, TrainTaskSampler, TrainMetaLoader
from resnet import initialize_model
from setup import reset_seed, matching_network_loss

def initialize_dataset(normal_folder, amd_folder, glaucoma_folder, n_way, s_support, q_query, tasks_per_batch):
    # Instantiate dataset and loaders
    three_way_dataset = TrainThreeWayDataset(normal_folder, amd_folder, glaucoma_folder)
    task_sampler = TrainTaskSampler(three_way_dataset, n_way, s_support, q_query)
    meta_train_loader = TrainMetaLoader(task_sampler, tasks_per_batch)
    return three_way_dataset, task_sampler, meta_train_loader

def initialize_training(device, learning_rate=0.0004935640937348862):
    # Initialize model and optimizer
    feature_extractor = initialize_model().to(device)
    optimizer = optim.Adam(feature_extractor.parameters(), lr=learning_rate)
    return feature_extractor, optimizer

def train_epoch(device, meta_train_loader, feature_extractor, optimizer, n_way, s_support):
    epoch_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    task_count = 0

    for task_data, task_labels in meta_train_loader:
        optimizer.zero_grad()
        
        support_set_data = []
        support_set_labels = []
        query_set_data = []
        query_set_labels = []

        for i in range(n_way):
            class_indices = (task_labels == i).nonzero(as_tuple=True)[0]
            support_indices = class_indices[:s_support]
            query_indices = class_indices[s_support:]
            support_set_data.append(task_data[support_indices])
            support_set_labels.append(task_labels[support_indices])
            query_set_data.append(task_data[query_indices])
            query_set_labels.append(task_labels[query_indices])

        support_set_data = torch.cat(support_set_data, 0).unsqueeze(1)  # Adding channel dimension
        support_set_labels = torch.cat(support_set_labels, 0)
        query_set_data = torch.cat(query_set_data, 0).unsqueeze(1)  # Adding channel dimension
        query_set_labels = torch.cat(query_set_labels, 0)
        
        support_set_data, support_set_labels = support_set_data.to(device), support_set_labels.to(device)
        query_set_data, query_set_labels = query_set_data.to(device), query_set_labels.to(device)
        
        # Get embeddings from the feature extractor
        support_set_embeddings = feature_extractor(support_set_data)
        query_set_embeddings = feature_extractor(query_set_data)
        
        # Calculate the loss and perform a backward pass using matching_network_loss function
        loss, acc, predicted_labels, query_set_labels = matching_network_loss(support_set_embeddings, support_set_labels, query_set_embeddings, query_set_labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        correct_predictions += acc * query_set_labels.size(0)
        total_predictions += query_set_labels.size(0)
        task_count += 1
    
    epoch_accuracy = correct_predictions / total_predictions
    epoch_loss = epoch_loss / task_count
    # Calculate class-specific accuracies
   
    return epoch_loss, epoch_accuracy

def main_train_loop(normal_folder, amd_folder, glaucoma_folder, n_way=3, s_support=5, q_query=5, tasks_per_batch=120, epochs=25):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    reset_seed()
    torch.cuda.empty_cache()
    gc.collect()

    # Initialize dataset, model, and optimizer
    _, _, meta_train_loader = initialize_dataset(normal_folder, amd_folder, glaucoma_folder, n_way, s_support, q_query, tasks_per_batch)
    feature_extractor, optimizer = initialize_training(device)

    start_time = time.time()
    best_val_loss = float('inf')
    for epoch in range(epochs):
        # Perform a single epoch of training
        epoch_loss, epoch_accuracy = train_epoch(device, meta_train_loader, feature_extractor, optimizer, n_way, s_support)
        print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')
        # Additional logic for model saving and printing class accuracies
        if epoch_loss < best_val_loss:
            best_val_loss = epoch_loss
            torch.save(feature_extractor.state_dict(), 'Models/5-MN.pth')
            print("Best Model Saved")
    end_time = time.time()
    # Print total training time 
    end_time = time.time()
    total_training_time = end_time - start_time
# Convert total training time to hours, minutes, and seconds
    hours =  total_training_time // 3600
    minutes = (total_training_time % 3600) // 60
    seconds = total_training_time % 60

    print(f"Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")

if __name__ == "__main__":
    normal_folder = 'Data/Train/Normal' 
    amd_folder = 'Data/Train/AMD' 
    glaucoma_folder = 'Data/Train/POAG'
   
    main_train_loop(normal_folder, amd_folder, glaucoma_folder)

