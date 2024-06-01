import os
import torch
import math
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from dataset import GazeDataset
from model import GazeModel
from utils import HuberLoss
from config import config


def gaze_to_vector(yaw, pitch):
    x = -torch.cos(pitch) * torch.sin(yaw)
    y = -torch.sin(pitch)
    z = -torch.cos(pitch) * torch.cos(yaw)
    return torch.stack([x, y, z], dim=1)

def angular_error(pred_gaze, true_gaze):
    pred_gaze = pred_gaze / torch.norm(pred_gaze, dim=1, keepdim=True)
    true_gaze = true_gaze / torch.norm(true_gaze, dim=1, keepdim=True)
    dot_product = torch.sum(pred_gaze * true_gaze, dim=1)
    return torch.acos(torch.clamp(dot_product, -1.0, 1.0)) * (180.0 / torch.tensor(math.pi))

def train():
    # Load the dataset
    dataset = GazeDataset(config.hdf5_path)

    # Split dataset into training and validation subsets
    # indices = list(range(len(dataset)))
    # train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)
    indices = list(range(1000))
    train_indices, val_indices = train_test_split(indices, test_size=0.1, random_state=42)

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # Initialize the model
    gaze_model = GazeModel().to(config.device)
    
    criterion = HuberLoss(delta=config.delta)
    optimizer = optim.Adam(gaze_model.parameters(), lr=config.learning_rate)

    gaze_model.train()

    best_val_loss = float('inf')

    for epoch in range(config.num_epochs):
        running_loss = 0.0
        for batch_idx, (data, labels) in enumerate(train_loader):
            data = {k: v.to(config.device) for k, v in data.items()}
            labels = {k: v.to(config.device) for k, v in labels.items()}

            optimizer.zero_grad()

            # Forward pass
            gaze, query = gaze_model(
                data['image'], data['left_eye'], data['right_eye'], data['pose'], data['head_rot'], data['subject_id'].long()
            )
            
            # Calculate loss
            loss = criterion(gaze, labels['gaze'])
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{config.num_epochs}], Loss: {running_loss/len(train_loader)}")

        # Validation step
        gaze_model.eval()
        val_loss = 0.0
        total_angular_error = 0.0
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(val_loader):
                data = {k: v.to(config.device) for k, v in data.items()}
                labels = {k: v.to(config.device) for k, v in labels.items()}

                gaze, query = gaze_model(
                    data['image'], data['left_eye'], data['right_eye'], data['pose'], data['head_rot'], data['subject_id'].long()
                )
                
                loss = criterion(gaze, labels['gaze'])
                val_loss += loss.item()


                # Calculate angular error
                pred_gaze_vector = gaze_to_vector(gaze[:, 1], gaze[:, 0])
                true_gaze_vector = gaze_to_vector(labels['gaze'][:, 1], labels['gaze'][:, 0])
                angular_err = angular_error(pred_gaze_vector, true_gaze_vector)
                total_angular_error += torch.sum(angular_err).item()


        val_loss /= len(val_loader)
        avg_angular_error = total_angular_error / len(val_dataset)
        print(f"Epoch [{epoch+1}/{config.num_epochs}], Validation Loss: {val_loss}, Angular Error: {avg_angular_error} degrees")

        # Save the model weights after every epoch
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(gaze_model.state_dict(), os.path.join(config.save_dir, 'best_gaze_model.pth'))
            print(f'Best model weights saved at epoch {epoch+1} with validation loss {val_loss}')



if __name__ == "__main__":
    train()
