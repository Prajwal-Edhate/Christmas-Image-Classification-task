import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import pandas as pd
import torch
from torch.utils.data import DataLoader
from data import ChristmasImages

from model import Network


# Define data path
train_data_path = '/home/g063898/Kaggle/data/train'
train_val_path = '/home/g063898/Kaggle/data/val'

# Load Training Images
test_data = ChristmasImages(train_val_path, training = False)
test_loader = DataLoader(test_data, batch_size = 40, shuffle=False)

# define Batch size, number of epochs, split for validation
num_epoch = 46
batch_size = 64
val_split = 0.92

# Load model
model = Network()

model.load_state_dict(torch.load('model.pkl'))

model.eval()

predictions = []
ids = []

for image in test_loader:
    #print(type(image))
    with torch.no_grad():
        outputs = model(image[0])
        _,predict = torch.max(outputs,1)
        predictions.extend(predict.tolist())


ids = [i for i in range(160)]

df = pd.DataFrame({'ID':ids,'Category': predictions})
df.to_csv('submission_2.csv', index=False)

