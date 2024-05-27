import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import GazeDataset
from gaze_model import GazeModel
#from models.calibration_model import CalibrationModel
from utils import HuberLoss
from config import config

def train():

    dataset = GazeDataset(config.hdf5_path)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    #gaze_model = GazeModel().to(config.device)
    gaze_model = GazeModel()
    # gaze_model = nn.DataParallel(gaze_model)  # Wrap the model with DataParallel
    # gaze_model = gaze_model.cuda()  # Move the model to GPU
    #print(gaze_model)
    #calibration_model = CalibrationModel(input_dim=1024).to(config.device)

    criterion = HuberLoss(delta=config.delta)
    optimizer = optim.Adam(list(gaze_model.parameters()), lr=config.learning_rate)

    gaze_model.train()

    #calibration_model.train()

    for epoch in range(config.num_epochs):
        running_loss = 0.0
        for batch_idx, (data, labels) in enumerate(dataloader):
            # data = {k: v.to(config.device) for k, v in data.items()}
            # labels = {k: v.to(config.device) for k, v in labels.items()}
            data = {k: v for k, v in data.items()}
            labels = {k: v for k, v in labels.items()}

            optimizer.zero_grad()
            #print(f"Batch {batch_idx} - Image shape: {data['image'].shape}, Left eye shape: {data['left_eye'].shape}, Right eye shape: {data['right_eye'].shape}, Pose shape: {data['pose'].shape}")
            #print(f"Batch {batch_idx} - Labels shape: {labels['gaze'].shape}")

            gaze, query = gaze_model(
                data['image'], data['left_eye'], data['right_eye'], data['pose'], data['rotation']
                            )
            
            #print(f"Batch {batch_idx} - Gaze prediction shape: {gaze.shape}")

            loss = criterion(gaze, labels['gaze'])
            #print(f"loss {batch_idx} tensor({loss}, device='{config.device}', grad_fn=<AddBackward0>)")
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{config.num_epochs}], Loss: {running_loss/len(dataloader)}")

if __name__ == "__main__":
    train()

