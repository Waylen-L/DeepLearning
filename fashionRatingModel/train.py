import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt


# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 1. Load dataset
data_dir = 'dataset'  # Replace with your actual dataset path


# Create random data
data = {
    'filename': [f"{i}.jpg" for i in range(1, 36)],
    'rating': [round(random.uniform(7.5, 9.9), 1) for _ in range(35)]
}

df = pd.DataFrame(data)
df.to_csv(os.path.join(data_dir, 'labels.csv'), index=False)

labels_path = os.path.join(data_dir, 'labels.csv')
df = pd.read_csv(labels_path, encoding='utf-8')
df['filename'] = df['filename'].apply(lambda x: os.path.join(data_dir, x))


# 2. Custom dataset
class FashionDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx]['filename']
        image = Image.open(img_name).convert('RGB')
        rating = self.dataframe.iloc[idx]['rating']

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(rating, dtype=torch.float32)


# Data preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Split dataset
train_df = df.sample(frac=0.8, random_state=42)
val_df = df.drop(train_df.index)

train_dataset = FashionDataset(train_df, transform=transform)
val_dataset = FashionDataset(val_df, transform=transform)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)


# 3. Define model
class FashionRatingModel(nn.Module):
    def __init__(self):
        super(FashionRatingModel, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet = self.resnet.to(device)  # Move the entire ResNet model to GPU
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        ).to(device)  # Ensure new layers are also on GPU

    def forward(self, x):
        return self.resnet(x)

model = FashionRatingModel().to(device)
print(f"Model is on GPU: {next(model.parameters()).is_cuda}")

# 4. Define loss function and optimizer
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.resnet.fc.parameters(), lr=0.0001)


# 5. Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=30):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, ratings in train_loader:
            images, ratings = images.to(device), ratings.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.squeeze(), ratings)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, ratings in val_loader:
                images, ratings = images.to(device), ratings.to(device)
                outputs = model(images)
                loss = criterion(outputs.squeeze(), ratings)
                val_loss += loss.item() * images.size(0)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

    return train_losses, val_losses


# 6. Train the model
train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer)

# 7. Save the final model
torch.save(model.state_dict(), 'fashion_rating_final_model.pth')

# 8. Visualize training and validation losses
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_plot.png')
plt.show()