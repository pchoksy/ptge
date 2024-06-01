import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from dataset import GazeDataset
from model import GazeModel, CalibrationModel
from config import config

def train_calibration_model():
    # Load the dataset
    dataset = GazeDataset(config.hdf5_path)

    # Split dataset into training and validation subsets
    indices = list(range(len(dataset)))
    train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # Initialize the GazeModel and load the trained weights
    gaze_model = GazeModel().to(config.device)
    gaze_model.load_state_dict(torch.load(os.path.join(config.save_dir, 'best_gaze_model.pth')))
    gaze_model.eval()

    # Initialize the CalibrationModel
    calibration_model = CalibrationModel(
        gaze_model=gaze_model,
        nhead=8,
        num_encoder_layers=4,
        dim_feedforward=512,
        output_dim=32  # The output dimension is now 16 for the embedding vector
    ).to(config.device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(calibration_model.parameters(), lr=config.learning_rate)

    best_val_loss = float('inf')

    for epoch in range(config.num_epochs):
        # Training phase
        calibration_model.train()
        running_loss = 0.0

        for batch_idx, (data, labels) in enumerate(train_loader):
            data = {k: v.to(config.device) for k, v in data.items()}
            labels = {k: v.to(config.device) for k, v in labels.items()}

            optimizer.zero_grad()

            # Extract features using the GazeModel
            with torch.no_grad():
                gaze, query = gaze_model(
                    data['image'], data['left_eye'], data['right_eye'], data['pose'], data['head_rot'], data['subject_id'].long()
                )

                # Get the true embedding from the GazeModel
                true_embedding = gaze_model.subject_embeddings(data['subject_id'].long())

            # Create the calibration query
            q = torch.cat([query, gaze], dim=1)

            # Forward pass through the CalibrationModel
            calib_embedding = calibration_model(q)

            # Calculate loss (MSE between predicted embedding and true embedding)
            loss = criterion(calib_embedding, true_embedding)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{config.num_epochs}], Loss: {running_loss/len(train_loader)}")

        # Validation phase
        calibration_model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(val_loader):
                data = {k: v.to(config.device) for k, v in data.items()}
                labels = {k: v.to(config.device) for k, v in labels.items()}

                # Extract features using the GazeModel
                gaze, query = gaze_model(
                    data['image'], data['left_eye'], data['right_eye'], data['pose'], data['head_rot'], data['subject_id'].long()
                )

                # Get the true embedding from the GazeModel
                true_embedding = gaze_model.subject_embeddings(data['subject_id'].long())

                # Create the calibration query
                q = torch.cat([query, gaze], dim=1)

                # Forward pass through the CalibrationModel
                calib_embedding = calibration_model(q)

                # Calculate loss (MSE between predicted embedding and true embedding)
                loss = criterion(calib_embedding, true_embedding)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"Epoch [{epoch+1}/{config.num_epochs}], Validation Loss: {val_loss}")

        # Save the model weights after every epoch
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(calibration_model.state_dict(), os.path.join(config.save_dir, 'best_calibration_model.pth'))
            print(f'Best calibration model weights saved at epoch {epoch+1} with validation loss {val_loss}')

if __name__ == "__main__":
    train_calibration_model()
