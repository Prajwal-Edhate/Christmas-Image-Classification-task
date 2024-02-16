import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from PIL import Image
import pandas as pd
import torch
import torchvision
from torchvision.datasets import ImageFolder
import torch.optim as optim
from torch.utils.data import DataLoader
from data import ChristmasImages
from torchvision import transforms
from torchvision.utils import save_image
from model import Network
import torch.nn as nn
import random
from tqdm import tqdm
import numpy as np
#from ConvNeXt.models.convnext import convnext_tiny

from concurrent.futures import ThreadPoolExecutor

def train(model, loss_function, optimizer, train_loader, val_loader, device, num_epoch):

    # Send model to device
    model.to(device)
    torch.cuda.empty_cache()

    for epoch in range(num_epoch):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            del inputs, labels
            torch.cuda.empty_cache()
            
        print(f"Epoch {epoch}/{num_epoch}, Loss: {running_loss/len(train_loader)}")

        #Validation
        if epoch % 1 == 0:
            model.save_model()
            model.eval()
            val_loss = 0.0
            correct_predictions = 0
            total_samples = 0

            with torch.no_grad():
                for val_inputs, val_labels in val_loader:
                    val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)

                    val_outputs = model(val_inputs)
                    val_loss += loss_function(val_outputs, val_labels).item()

                    _, predicted = torch.max(val_outputs, 1)
                    correct_predictions += (predicted == val_labels).sum().item()
                    total_samples += val_labels.size(0)

                    del val_inputs, val_labels
                    torch.cuda.empty_cache()
            
            average_val_loss = val_loss / len(val_loader)
            accuracy = correct_predictions / total_samples

            print(f"Epoch {epoch}/{num_epoch}, Validation Loss: {average_val_loss}, Accuracy: {accuracy * 100}%")

    model.save_model()    
    print("Training Complete")

def add_gausian_noise(img, mean = 0, std = 25):
    img_array = np.array(img)
    noise = np.random.normal(mean, std, img_array.shape)
    noisy_img_array = img_array + noise
    noisy_img = np.clip(noisy_img_array, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)

def Augmix2(dataset, a = 0.12, b = 0.08, c = 0.8, m = 0.9):
    a = torch.tensor(a)
    b = torch.tensor(b)
    c = torch.tensor(c)
    m = torch.tensor(m)

    #original = [(img,label) for img, label in tqdm(dataset, desc = "Convert", unit = "Image")]

    transform1 = transforms.Compose([transforms.ToPILImage(),
                                     transforms.Resize((224,224)),
                                     transforms.RandomAffine(degrees = 0, translate = (0.1, 0.1)),
                                     transforms.ToTensor()])
    
    transform2 = transforms.Compose([transforms.ToPILImage(),
                                     transforms.Resize((224,224)),
                                     transforms.RandomAffine(degrees = 25, translate = (0, 0)),
                                     transforms.ToTensor()])
    
    transform3 = transforms.Compose([transforms.ToPILImage(),
                                     transforms.Resize((224,224)),
                                     transforms.ColorJitter(brightness = 0.6, contrast = 0.04, saturation = 0.4, hue = 0.2),
                                     transforms.ToTensor()])

    for index in tqdm(range(len(dataset)),desc = "Augmentation", unit = "Image"):
        o_img, label= dataset[index]
        img1, _ = dataset[index]
        img2, _ = dataset[index]
        img3, _ = dataset[index]
        img1 = transform1(img1)
        img1 = img1 * a
        img2 = transform2(img2)
        img2 = img2 * b
        img3 = transform3(img3)
        img3 = img3 * c
        new_img = img1 + img2 + img3
        aug_img = (1-m) * o_img + m * new_img
        dataset[index] = (aug_img,label)
    
    return dataset


# Define Transformations
transform_default = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

transform5 = transforms.Compose([(transforms.ToPILImage()),
                                  transforms.Resize((224,224)),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.RandomAffine(degrees = 15, translate = (0.05,0.09)),
                                  transforms.RandomResizedCrop(size=(224, 224),scale=(0.8,0.8)),
                                  transforms.ColorJitter(brightness = 0.4, contrast = 0.06, saturation = 0.4, hue = 0.2),
                                  transforms.Lambda(lambda x: add_gausian_noise(x, mean = 0, std = 25)),
                                  transforms.ToTensor()])

transform6 = transforms.Compose([(transforms.ToPILImage()),
                                  transforms.Resize((224,224)),
                                  transforms.RandomAffine(degrees = 90, translate = (0.01,0.02)),
                                  transforms.Lambda(lambda x: add_gausian_noise(x, mean = 0, std = 25)),
                                  transforms.ToTensor()])

gray = transforms.Compose([transforms.ToPILImage(),
                           transforms.Grayscale(),
                           transforms.Lambda(lambda x: x.convert("RGB")),
                           transforms.Lambda(lambda x: add_gausian_noise(x, mean = 0, std = 25)),
                           transforms.ToTensor()])

to_tensor = transforms.ToTensor()


# Define data path
train_data_path = '/home/g063898/Kaggle/data/train'
train_val_path = '/home/g063898/Kaggle/data/val'

# Load Training Images
data_set = ChristmasImages(train_data_path, training = True)
test_data = ChristmasImages(train_val_path, training = False)

# define Batch size, number of epochs, split for validation
num_epoch = 46
batch_size = 64
val_split = 0.92

# Split for Validation and Training
total_samples = len(data_set)
train_samples = int(val_split * total_samples)
val_samples = total_samples - train_samples
learning_rate = 0.001

# Load model
model = Network()

model.load_state_dict(torch.load('model_2.pkl'))


# Split the dataset
train_data, val_data = torch.utils.data.random_split(data_set, [train_samples, val_samples])

original = [(img,label) for img, label in tqdm(train_data, desc = "Convert", unit = "Image")]

clone3 = [(transform6(img), label) for img, label in tqdm(original, desc = "Augmentation_6", unit = "Image")]

# Augmeneted with Augmix
augmented_data_1 = Augmix2(dataset=original)

augmented_data_2 = Augmix2(dataset=clone3, a = 0.1, b = 0.15, c = 0.75)

new_data = augmented_data_1 + augmented_data_2 


# Use data loader
train_loader = DataLoader(new_data, batch_size = batch_size, shuffle = True, num_workers = 4)
val_loader = DataLoader(val_data, batch_size = 128, shuffle = True, num_workers = 4)


# Loss Function and Optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adamax(model.parameters(), lr = learning_rate, weight_decay = 1e-5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train(model=model, loss_function=loss_function, optimizer=optimizer, train_loader=train_loader, val_loader=val_loader,device=device, num_epoch=num_epoch)
