import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

class StomataDataset(Dataset):
    """
    Dataset that returns (image, is_opening, aperture).
    is_opening: tensor 1.0 if there is a measurable aperture, 0.0 if NC.
    aperture: aperture value, or 0.0 if NC.
    """
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file, sep=";")
        self.image_dir = image_dir
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        filename = row[0]
        label_str = str(row[1])
        img_path = os.path.join(self.image_dir, filename)
        image = Image.open(img_path).convert('RGB')
        if label_str.upper() == 'NC':
            is_opening = 0.0
            apertura = 0.0
        else:
            is_opening = 1.0
            apertura = float(label_str)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(is_opening), torch.tensor(apertura)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data_csv = 'Training_Values.csv'
image_dir = 'Training_images'
full_ds = StomataDataset(data_csv, image_dir, transform)
train_size = int(0.8 * len(full_ds))
val_size = len(full_ds) - train_size
torch.manual_seed(42)
train_ds, val_ds = random_split(full_ds, [train_size, val_size])
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
