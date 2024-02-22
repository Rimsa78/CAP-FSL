# -*- coding: utf-8 -*-
import torch
import torch.optim as optim
import time
import gc
from traindataloader import TrainThreeWayDataset, TrainTaskSampler, TrainMetaLoader
from resnet import initialize_model
from setup import form_subspaces, reset_seed, distance_to_subspace, subspace_classifier, subspace_loss, calculate_accuracy

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
    # Initialize tensors for class-specific accuracies
    class_correct_predictions = torch.zeros(n_way).to(device)
    class_total_predictions = torch.zeros(n_way).to(device)

    for task_data, task_labels in meta_train_loader:
        optimizer.zero_grad()
        assert task_labels.max().item() < n_way, "Labels exceed the number of classes."
        assert task_labels.min().item() >= 0, "Labels should be positive integers."

        support_set_data = []
        support_set_labels = []
        query_set_data = []
        query_set_labels = []
        
        for i in range(n_way):
            class_indices = (task_labels == i).nonzero(as_tuple=True)[0]
            support_indices = class_indices[:s_support]
            support_set_data.append(task_data[support_indices])
            support_set_labels.append(task_labels[support_indices])
            query_indices = class_indices[s_support:]
            query_set_data.append(task_data[query_indices])
            query_set_labels.append(task_labels[query_indices])
        
        support_set_data = torch.cat(support_set_data, 0).unsqueeze(1)
        support_set_labels = torch.cat(support_set_labels, 0)
        query_set_data = torch.cat(query_set_data, 0).unsqueeze(1)
        query_set_labels = torch.cat(query_set_labels, 0)
        
        support_set_data, support_set_labels = support_set_data.to(device), support_set_labels.to(device)
        query_set_data, query_set_labels = query_set_data.to(device), query_set_labels.to(device)

        
        feature_vectors_support = feature_extractor(support_set_data)
        feature_vectors_query = feature_extractor(query_set_data)
        subspaces = form_subspaces(feature_vectors_support, support_set_labels, subspace_dim=5)

        loss = subspace_loss(feature_vectors_query, query_set_labels, subspaces)
        loss.backward()
        optimizer.step()
        
        # Accuracy calculation
        predicted_labels = torch.tensor([subspace_classifier(vec, subspaces)[0] for vec in feature_vectors_query]).to(device)
        acc = calculate_accuracy(predicted_labels, query_set_labels)
        
        # Class-specific accuracy update
        for i in range(n_way):
            class_total_predictions[i] += (query_set_labels == i).sum().item()
            class_correct_predictions[i] += (predicted_labels[query_set_labels == i] == i).sum().item()

        epoch_loss += loss.item()
        correct_predictions += acc * len(query_set_labels)
        total_predictions += len(query_set_labels)
        task_count += 1
    
    epoch_accuracy = correct_predictions / total_predictions
    epoch_loss = epoch_loss / task_count
    # Calculate class-specific accuracies
    class_accuracies = class_correct_predictions / class_total_predictions + 1e-10
    print("Class Accuracies:", class_accuracies)
    return epoch_loss, epoch_accuracy, class_correct_predictions, class_total_predictions

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
        epoch_loss, epoch_accuracy, _, _ = train_epoch(device, meta_train_loader, feature_extractor, optimizer, n_way, s_support)
        print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')
        # Additional logic for model saving and printing class accuracies 
        if epoch_loss < best_val_loss:
            best_val_loss = epoch_loss
            torch.save(feature_extractor.state_dict(), 'Models/5-DSN.pth')
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

