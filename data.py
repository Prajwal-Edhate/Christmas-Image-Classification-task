from torch.utils.data import Dataset
import torch
from PIL import Image
from torchvision import transforms
import os
import natsort

class ChristmasImages(Dataset):
    
    def __init__(self, path, training=True):
        super().__init__()
        self.training = training
        self.path = path
        self.transform = transforms.Compose([(transforms.Resize((224,224))),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        # If training == True, path contains subfolders
        # containing images of the corresponding classes
        # If training == False, path directly contains
        # the test images for testing the classifier
        if training:
            # Training data
            self.class_names = sorted(os.listdir(self.path))
            self.class_to_label = {class_name: index for index, class_name in enumerate(self.class_names)}

            self.file_paths = []
            self.labels = []

            for label, class_name in enumerate(self.class_names):
                class_path = os.path.join(self.path, class_name)
                for file_name in os.listdir(class_path):
                    file_path = os.path.join(class_path, file_name)
                    self.file_paths.append(file_path)
                    self.labels.append(label)

        else:
            # For testing data
            self.file_paths = []
            self.file_names = []
            for file_name in natsort.natsorted(os.listdir(self.path)):
                file_path = os.path.join(self.path, file_name)
                self.file_paths.append(file_path)
                self.file_names.append(file_name)

        
    def __getitem__(self, index):
        # If self.training == False, output (image, )
        # where image will be used as input for your model
        if self.training:
            image_path = self.file_paths[index]
            label = self.labels[index]

            # Load Image
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
            return image, label
        else:
            image_path = self.file_paths[index]
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
            return (image, )
        
        raise NotImplementedError
    
    def __len__(self):
        return len(self.file_paths)
    
        
