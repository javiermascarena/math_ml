from collections import defaultdict
import numpy as np
from torch.utils.data import Subset


def get_subset_with_all_labels(dataset, subset_length, n_samples, reproducible=True):
    """
    Generates a subset of the given dataset with a specified total length, ensuring that each label
    is represented by at least n_samples observations. 

    Parameters:
    -----------
    dataset : torch.utils.data.Dataset
        The original dataset where each element is assumed to be a tuple (data, label).
    subset_length : int
        The desired total number of samples in the final subset.
    n_samples : int
        The minimum number of samples required per label in the subset. This helps ensure that after 
        splitting, every label is present in both the training and testing sets.
    reproducible : bool
        If True, sets the random seed for numpy to ensure that shuffling of indices is reproducible.

    Returns:
    --------
    torch.utils.data.Subset:
        A subset of the original dataset of length 'subset_length' that contains at least 'n_samples'
        observations for each label.
    
    Raises:
    -------
    ValueError:
        If any label in the dataset has fewer than n_samples available observations.
    """
    
    # Get the dataset labels into numpy arrays
    labels = np.array(dataset.targets)

    selected_indices = []
    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)

    # Loop over each label
    for n, label in enumerate(unique_labels): 
        # Get the indices for the label
        indices = np.where(labels==label)[0]
        if len(indices) < n_samples: 
            raise ValueError(f"Label {label} does not have at least {n_samples}\
                               samples; cannot guarantee both splits will include it.")
        
        if reproducible: 
            np.random.seed(42)
        # Randomly sample n_samples indices from this label without replacement.
        sampled_indices = np.random.choice(indices, size=n_samples, replace=False)
        selected_indices.extend(sampled_indices.tolist())

        print(f"Completed label: {n}/{n_labels}")
            
    # Remove duplicate indices (should not be the case)
    selected_indices = list(set(selected_indices))
    
    # Add more random samples until reaching subset_length
    all_indices = set(range(len(dataset)))
    remaining_indices = list(all_indices - set(selected_indices))
    if reproducible:
        np.random.seed(42) 
    np.random.shuffle(remaining_indices, )

    needed = subset_length - len(selected_indices)
    if needed > 0:
        selected_indices.extend(remaining_indices[:needed])
    else: 
        np.random.shuffle(selected_indices)
        selected_indices = selected_indices[:subset_length]
    
    return Subset(dataset, selected_indices)

