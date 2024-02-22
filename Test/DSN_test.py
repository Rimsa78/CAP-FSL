# -*- coding: utf-8 -*-
import gc
from testdataloader import TestMetaLoader, TestThreeWayDataset, TestTaskSampler
import torch
import torch.optim as optim
from resnet import initialize_model
from setup import reset_seed, subspace_classifier, subspace_loss, calculate_accuracy, form_subspaces
import time
from sklearn.metrics import confusion_matrix

torch.cuda.empty_cache()  # Clears cached memory
gc.collect()  

 # Parameters
n_way = 3  
s_support = 5  
q_query = 5  

def data_initialize():
    normal = 'Data/Test/Normal'
    amd = 'Data/Test/AMD'
    glaucoma = 'Data/Test/POAG' 

    # Instantiate the dataset and sampler without the mode argument 
    three_way_test_dataset = TestThreeWayDataset(normal, amd, glaucoma) 
    three_way_test_task_sampler = TestTaskSampler(three_way_test_dataset, n_way=n_way, k_shot=s_support, q_query=q_query)

    return three_way_test_dataset, three_way_test_task_sampler

def test_initialize():
    reset_seed()  # Ensures reproducibility
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_extractor = initialize_model().to(device)

    # Load pre-trained model
    feature_extractor.load_state_dict(torch.load('Models/DS-5.pth'))
    feature_extractor.eval()  

    optimizer = optim.Adam(feature_extractor.parameters(), lr=0.0004935640937348862)

    return device, feature_extractor, optimizer

def calculate_and_print_metrics(all_true_labels, all_predicted_labels, class_specific_TP, class_specific_FP, class_specific_TN, class_specific_FN, start_time):
    end_time = time.time()
    total_training_time = end_time - start_time
    hours = total_training_time // 3600
    minutes = (total_training_time % 3600) // 60
    seconds = total_training_time % 60

    print(f"Total testing time: {int(hours)}h {int(minutes)}m {int(seconds)}s")

    # Calculate and print global metrics
    for class_label in class_specific_TP:
        TP = class_specific_TP[class_label]
        FP = class_specific_FP[class_label]
        TN = class_specific_TN[class_label]
        FN = class_specific_FN[class_label]

        sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall = sensitivity  # Recall is the same as sensitivity
        f_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0

        print(f"Class {class_label}:")
        print(f"Accuracy: {accuracy:.4f}, Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F-Score: {f_score:.4f}\n")

    # Generate the confusion matrix
    conf_matrix = confusion_matrix(all_true_labels, all_predicted_labels)
    print("Confusion Matrix:\n", conf_matrix)
    
def test_loop(meta_test_loader, device, feature_extractor, optimizer):
    total_test_loss = 0.0
    total_test_accuracy = 0.0
    total_test_tasks = 0
    task_accuracies = []  

    total_TP, total_FP, total_TN, total_FN = 0, 0, 0, 0
    all_predicted_labels = []
    all_true_labels = []
    class_specific_TP = {}
    class_specific_FP = {}
    class_specific_TN = {}
    class_specific_FN = {}

    start_time = time.time()

    for test_task_data, test_task_labels in meta_test_loader:
        optimizer.zero_grad()  # Clear gradients

        # Sort the support and query sets by class
        support_set_data, support_set_labels, query_set_data, query_set_labels = [], [], [], [] 
        for i in range(3):
            class_indices = (test_task_labels == i).nonzero(as_tuple=True)[0]
            support_indices = class_indices[:s_support]
            support_set_data.append(test_task_data[support_indices])
            support_set_labels.append(test_task_labels[support_indices])
            query_indices = class_indices[s_support:]
            query_set_data.append(test_task_data[query_indices])
            query_set_labels.append(test_task_labels[query_indices])

        # Convert to tensors and move to device 
        support_set_data = torch.cat(support_set_data).unsqueeze(1).to(device)
        support_set_labels = torch.cat(support_set_labels).to(device)
        query_set_data = torch.cat(query_set_data).unsqueeze(1).to(device)
        query_set_labels = torch.cat(query_set_labels).to(device)

        # Generate feature vectors
        feature_vectors_support = feature_extractor(support_set_data)
        feature_vectors_query = feature_extractor(query_set_data)

        subspaces = form_subspaces(feature_vectors_support, support_set_labels, subspace_dim=5)

        loss = subspace_loss(feature_vectors_query, query_set_labels, subspaces)

        predicted_labels = torch.tensor([subspace_classifier(vec, subspaces)[0] for vec in feature_vectors_query]).to(device)
        acc = calculate_accuracy(predicted_labels, query_set_labels)
         
        total_test_loss += loss.item()
        total_test_accuracy += acc
        total_test_tasks += 1
        task_accuracies.append(acc) 

        all_true_labels.extend(query_set_labels.cpu().numpy())
        all_predicted_labels.extend(predicted_labels.cpu().numpy())
        
        for class_label in torch.unique(query_set_labels):
            TP = ((predicted_labels == class_label) & (query_set_labels == class_label)).sum().item()
            FP = ((predicted_labels == class_label) & (query_set_labels != class_label)).sum().item()
            TN = ((predicted_labels != class_label) & (query_set_labels != class_label)).sum().item()
            FN = ((predicted_labels != class_label) & (query_set_labels == class_label)).sum().item()
            
            # Accumulate metrics for each class
            if class_label.item() not in class_specific_TP:
                class_specific_TP[class_label.item()] = TP
                class_specific_FP[class_label.item()] = FP
                class_specific_TN[class_label.item()] = TN
                class_specific_FN[class_label.item()] = FN
            else:
                class_specific_TP[class_label.item()] += TP
                class_specific_FP[class_label.item()] += FP
                class_specific_TN[class_label.item()] += TN
                class_specific_FN[class_label.item()] += FN


    # Calculate and print global metrics 
    calculate_and_print_metrics(
        all_true_labels, 
        all_predicted_labels, 
        class_specific_TP, 
        class_specific_FP, 
        class_specific_TN, 
        class_specific_FN,
        start_time
    ) 
 

def __main__():
    # Initialization
    three_way_test_dataset, three_way_test_task_sampler = data_initialize()
    meta_test_loader = TestMetaLoader(three_way_test_task_sampler, tasks_per_batch=12)
    device, feature_extractor, optimizer = test_initialize()

    # Testing Loop
    test_loop(meta_test_loader, device, feature_extractor, optimizer)

if __name__ == "__main__":
    __main__()