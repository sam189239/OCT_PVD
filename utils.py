## Imports ##

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import numpy as np
from torchvision.transforms import functional as F
import cv2
from tqdm import tqdm
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from collections import Counter
from collections import defaultdict
import random
from torch.utils.data import Subset
from sklearn.utils.class_weight import compute_class_weight



## Pre-processing ##

class BilateralFilter:
    def __call__(self, img):
        img = np.array(img)
        denoised_img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
        return transforms.functional.to_pil_image(denoised_img)
    
class CLAHETransform:
    def __call__(self, img):
        img_np = np.array(img)  # Convert to NumPy array
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img_gray)
        img_clahe_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)
        img_tensor = (img_clahe_rgb)
        return img_tensor
    
class CannyEdgeTransform:
    def __call__(self, img):
        img_np = np.array(img)
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(img_gray, threshold1=100, threshold2=200)
        img_edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        img_tensor = (img_edges_rgb)
        return img_tensor

class GaussianBlur:
    def __call__(self, img):
        img = np.array(img)
        denoised_img = cv2.GaussianBlur(img, (5, 5), 0)
        return transforms.functional.to_pil_image(denoised_img)


## Visualize images ##

# Denormalize function to convert back to original image range
def denormalize(img, mean, std):
    img = img * std[:, None, None] + mean[:, None, None]  # Unnormalize
    img = np.clip(img, 0, 1)  # Clip values between 0 and 1
    return img

# Visualize image samples from each class
def visualize_samples(dataset, classes, num_images_per_class=3):
    # Initialize the plot
    fig, axes = plt.subplots(len(classes), num_images_per_class, figsize=(num_images_per_class * 3, len(classes) * 3))

    class_counts = {cls: 0 for cls in classes}
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Iterate through dataset to find and plot images from each class
    for img, label in dataset:
        class_name = classes[label]
        if class_counts[class_name] < num_images_per_class:
            # Denormalize the image
            img = img.numpy()
            img = denormalize(img, mean, std)
            img = np.transpose(img, (1, 2, 0))  # Change from (C, H, W) to (H, W, C) for plotting
            
            # Plot the image
            ax = axes[label, class_counts[class_name]]
            ax.imshow(img)
            ax.set_title(f'Class: {class_name}')
            ax.axis('off')
            
            # Increment the count for that class
            class_counts[class_name] += 1
            
        # Stop if we have enough images for all classes
        if all(count >= num_images_per_class for count in class_counts.values()):
            break
    
    plt.tight_layout()
    plt.show()


## Model training ##

def train_fn(model, train_loader, test_loader, model_name, num_epochs=5,
             device='cpu', criterion=nn.CrossEntropyLoss(), 
             optimizer=None, plot=True, lr=0.001):
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    best_val_acc = 0
    total_start_time = time.time()
    for epoch in (range(num_epochs)):
        epoch_start_time = time.time()
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for (inputs, labels) in tqdm((train_loader)):
            inputs, labels = inputs.to(device), labels.to(device)
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0) ## ?
            # Calculate training accuracy
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        epoch_acc = correct_train / total_train
        train_accuracies.append(epoch_acc)
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for (inputs, labels) in tqdm((test_loader)):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                # Calculate validation accuracy
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        val_epoch_loss = val_loss / len(test_loader)
        val_losses.append(val_epoch_loss)
        val_epoch_acc = correct_val / total_val
        val_accuracies.append(val_epoch_acc)
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, '
              f'Val Loss: {val_epoch_loss:.4f}, Val Accuracy: {val_epoch_acc:.2f}, '
              f'Epoch Time: {epoch_duration:.2f}s')
        
        if len(val_accuracies)==1 or val_accuracies[-1] >= best_val_acc:
            best_val_acc = val_accuracies[-1]
            torch.save(model.state_dict(), "models/"+model_name+".pth")
            print("Saving weights...")
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    if plot:
        epochs = range(1, num_epochs + 1)
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, label='Train Loss')
        plt.plot(epochs, val_losses, label='Val Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accuracies, label='Train Accuracy')
        plt.plot(epochs, val_accuracies, label='Val Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Validation Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.show()
    print(f'Total Training Time: {total_duration:.2f}s')
    return model, train_losses, val_losses

def eval_fn(model, val_loader, device='cpu', binary = False):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    print(f'Validation Accuracy: {accuracy:.4f}')
    if binary:
        print(f'Validation F1: {f1_score(all_labels, all_preds):.4f}')
    else:
        print(f'Validation F1: {f1_score(all_labels, all_preds, average="macro"):.4f}')
    return accuracy


## Models ##

def resnet50_modelinit(device='cuda'):
    resnet50 = models.resnet50(pretrained=True)

    for param in resnet50.parameters():
        param.requires_grad = False ## Try again with true !!

    num_ftrs = resnet50.fc.in_features
    resnet50.fc = nn.Sequential(
        nn.Linear(num_ftrs, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 4) 
    )

    resnet50 = resnet50.to(device)
    optimizer = optim.Adam(resnet50.fc.parameters(), lr=0.001)

    return resnet50, optimizer