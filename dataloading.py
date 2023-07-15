import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset
from PIL import Image
import os
import torchvision.models as models
import time
import copy
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import random
from collections import defaultdict

def get_indexes(arr, value):
    indexes = []
    for i in range(len(arr)):
        if arr[i] == value:
            indexes.append(i)
    return indexes

def get_length_per_class(dataloader, classes):
    class_counts = defaultdict(int)
    total = 0
    for batch in dataloader:
        _, labels = batch 
        labels = labels.numpy().tolist()
        for label in labels:
            class_counts[label] += 1
            total +=1

    class_counts = dict(sorted(class_counts.items()))
    for class_label, count in class_counts.items():
        print(f"Class {classes[class_label]}: {count} samples out of {total}")
def load_data(data_dir,
                           batch_size,
                           data_type,
                           noise_type,
                           noise_percentage,                           
                           transform,                           
                           data_percentage=1,
                           show_classes = False):
    
    if noise_type == "None":
        noise_type = ""
        noise_percentage = ""
    else:
        noise_type = "/" + str(noise_type)
        noise_percentage = "/" + str(noise_percentage)
    path = data_dir + noise_type + "/" + data_type + noise_percentage
    print("path: ", path)
    dataset = ImageFolder(root=path, transform=transform)
    original_classes = dataset.classes 
    num_samples = len(dataset)
    indices = list(range(num_samples))

    labels = dataset.targets
    class_to_idx = dataset.class_to_idx
    needed_length = int(num_samples*data_percentage/100)
    expected_length_per_class = int(needed_length/len(original_classes))
    print(f"needed_length: {needed_length}, expected_length_per_class: {expected_length_per_class}")
    if data_percentage != 100:
        new_indices = []
        for key, value in class_to_idx.items():
            all_indixes_of_class = get_indexes(labels, value)
            new_indices.extend(all_indixes_of_class[:expected_length_per_class])
    else:
        new_indices = indices
    length_dataset = len(new_indices)
    print("length of final dataset:", length_dataset)
    sampler = SubsetRandomSampler(new_indices)

    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

    if show_classes:
        get_length_per_class(dataloader, original_classes)

    return dataloader, length_dataset, original_classes
    

