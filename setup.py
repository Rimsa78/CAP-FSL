import torch
import torch.nn.functional as F
import numpy as np
import random
from torch.linalg import svd

"""
    Constructs low-dimensional subspaces for each unique class. 
    Returns: A dictionary where keys are class labels and values are subspace bases.
    """
def form_subspaces(feature_vectors, labels, subspace_dim):
    subspaces = {}
    for label in labels.unique():
        class_features = feature_vectors[labels == label]
        u, _, _ = svd(class_features.T)
        subspaces[label.item()] = u[:, :subspace_dim]
    return subspaces

"""
    Calculates the distance between a query vector and a subspace.
    Returns: Mean squared error distance between the query vector and its projection.
    """
def distance_to_subspace(query_vector, subspace):
    projection = subspace @ subspace.T @ query_vector
    return F.mse_loss(query_vector, projection, reduction='sum')

""" Predicts the class of a query feature vector using subspace distances."""
def subspace_classifier(query_features, subspaces):
    distances = {label: distance_to_subspace(query_features, subspace) for label, subspace in subspaces.items()}
    predicted_label = min(distances, key=distances.get)
    return predicted_label, distances[predicted_label]

"""Calculates a loss encouraging classification based on subspace distances."""
def subspace_loss(query_set_data, query_set_labels, subspaces):
    loss = 0.0
    for i, query_vector in enumerate(query_set_data):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        label = query_set_labels[i]
        distances = torch.stack([distance_to_subspace(query_vector, subspaces[l.item()]) for l in query_set_labels.unique()])
        distances = distances.to(device)
        probabilities = F.softmax(-distances, dim=0)
        loss += -torch.log(probabilities[label])
    return loss / query_set_data.size(0)

"""Calculates the percentage of correct predictions."""
def calculate_accuracy(predicted_labels, true_labels):
    correct = (predicted_labels == true_labels).sum().item()
    total = true_labels.size(0)
    return correct / total

"""Calculates loss and accuracy for a matching network."""
def matching_network_loss(support_set_embeddings, support_set_labels, query_set_embeddings, query_set_labels):
    n_way = torch.unique(support_set_labels).size(0)
    # Calculate cosine similarities between support and query embeddings
    similarities = torch.mm(query_set_embeddings, support_set_embeddings.t())
    # Apply softmax to compute attention scores over the support set
    attention_scores = torch.nn.functional.softmax(similarities, dim=1)
    # Compute the predicted probabilities for query set labels
    preds = torch.mm(attention_scores, torch.nn.functional.one_hot(support_set_labels, num_classes=n_way).float())
    # Compute the loss using the predicted probabilities
    loss_val = torch.nn.functional.cross_entropy(preds, query_set_labels)
    _, predicted_labels = preds.max(1)
    acc_val = (predicted_labels == query_set_labels).float().mean().item()
    
    return loss_val, acc_val,predicted_labels, query_set_labels

"""Compute cross-attention weights where the query set influences attention on the support set."""
def cross_attention(support_set, query_set):
    query = query_set
    key = value = support_set
    scores = torch.matmul(query, key.transpose(-2, -1)) / (key.shape[-1] ** 0.5)
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, value)


"""Calculates prototypical loss incorporating cross-attention"""
def can_prototypical_loss(support_set_data, support_set_labels, query_set_data, query_set_labels):
    classes = torch.unique(support_set_labels)
    prototypes = torch.zeros([len(classes), support_set_data.shape[1]], dtype=torch.float32)

    # Apply cross-attention to the support set influenced by the query set
    attended_support_set = cross_attention(support_set_data, query_set_data)
    # Compute the prototype for each class
    for i, label in enumerate(classes):
        # Find indices of all examples in the support set that belong to the current class
        idx = torch.eq(support_set_labels, label).nonzero(as_tuple=True)
        # Calculate the mean of the attended features for the current class as its prototype
        prototype = torch.mean(attended_support_set[idx], dim=0)
        prototypes[i] = prototype

    prototypes = prototypes.to(query_set_data.device)
    # Compute pairwise distances between query examples and prototypes using Euclidean distance
    dists = torch.cdist(query_set_data, prototypes, p=2)
    # Normalize distances to have zero mean and unit variance across each query example
    normalized_dists = (dists - dists.mean(dim=1, keepdim=True)) / (dists.std(dim=1, keepdim=True) + 1e-5)
    # Compute log softmax of the negative normalized distances for stability and efficiency
    log_p_y = torch.log_softmax(-normalized_dists, dim=1)
    # Calculate the loss value as the negative log probability of the true class labels
    loss_val = -log_p_y.gather(1, query_set_labels.unsqueeze(1)).squeeze().view(-1).mean()
    # Predict labels by choosing the class with the highest log probability
    predicted_labels = torch.argmax(log_p_y, 1)
    acc_val = torch.eq(predicted_labels, query_set_labels).sum().item() / query_set_labels.size(0)

    return loss_val, acc_val, log_p_y, predicted_labels, query_set_labels

"""Random Initialization for reproducibility."""
def reset_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

"""Regular Prototypical Loss without attention mechanism"""
def pn_prototypical_loss(support_set_data, support_set_labels, query_set_data, query_set_labels):
    classes = torch.unique(support_set_labels)
    prototypes = torch.zeros([len(classes), support_set_data.shape[1]], dtype=torch.float32)
    for i, label in enumerate(classes):
        idx = torch.eq(support_set_labels, label).nonzero(as_tuple=True)
        prototype = torch.mean(support_set_data[idx], dim=0)
        prototypes[i] = prototype
    
    prototypes = prototypes.to(query_set_data.device)
    dists = torch.cdist(query_set_data, prototypes, p=2)
    normalized_dists = (dists - dists.mean(dim=1, keepdim=True)) / (dists.std(dim=1, keepdim=True) + 1e-5)
    log_p_y = torch.log_softmax(-normalized_dists, dim=1)
    loss_val = -log_p_y.gather(1, query_set_labels.unsqueeze(1)).squeeze().view(-1).mean()
    acc_val = torch.eq(torch.argmax(log_p_y, 1), query_set_labels).sum().item() / query_set_labels.size(0)
    predicted_labels = torch.argmax(log_p_y, 1)
    
    return loss_val, acc_val, log_p_y, predicted_labels, query_set_labels

