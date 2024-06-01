import os
import torch
import random
import torch.nn as nn
import math
from torch.utils.data import DataLoader, Subset
from dataset import GazeDataset
from model import GazeModel, CalibrationModel
from config import config

def load_models():
    # Load the GazeModel and CalibrationModel
    gaze_model = GazeModel().to(config.device)
    calibration_model = CalibrationModel(
        gaze_model=gaze_model,
        nhead=8,
        num_encoder_layers=4,
        dim_feedforward=512,
        output_dim=32  # The output dimension is now 32 for the embedding vector
    ).to(config.device)

    # Load the trained weights
    gaze_model.load_state_dict(torch.load(os.path.join(config.save_dir, 'best_gaze_model.pth')))
    calibration_model.load_state_dict(torch.load(os.path.join(config.save_dir, 'best_calibration_model.pth')))

    return gaze_model, calibration_model

def sample_last_images(dataset, num_samples=100):
    total_images = len(dataset)
    start_index = max(total_images - num_samples, 0)
    indices = list(range(start_index, total_images))
    return Subset(dataset, indices)

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

def test_gaze_model(gaze_model, calibration_model, dataset):
    gaze_model.eval()
    calibration_model.eval()
    
    with torch.no_grad():
        for i in range(len(dataset)):
            # Get a single sample from the dataset
            data, labels = dataset[i]

            # Move data to the correct device
            data = {k: v.unsqueeze(0).to(config.device) for k, v in data.items()}
            labels = {k: v.unsqueeze(0).to(config.device) for k, v in labels.items()}

            # Ensure subject_id indices are within range
            max_index = gaze_model.subject_embeddings.num_embeddings - 1
            invalid_indices = (data['subject_id'] < 0) | (data['subject_id'] > max_index)
            if invalid_indices.any():
                raise ValueError(f"Found invalid subject_id indices: {data['subject_id'][invalid_indices]}")
            
            
            # Extract features using the GazeModel
            gaze, query = gaze_model(
                data['image'], data['left_eye'], data['right_eye'], data['pose'], data['head_rot'], data['subject_id'].long()
            )
            print('size of embeddings: ', data['subject_id'].shape)
            # Create the calibration query
            q = torch.cat([query, gaze], dim=1)
            
            # Get the calibrated embeddings
            calibrated_embeddings = calibration_model(q)

            
            # Predict the gaze using the modified GazeModel
            right_eye_features = gaze_model.eye_cnn(data['right_eye'].permute(0, 3, 1, 2))
            right_eye_features = right_eye_features.reshape(right_eye_features.size(0), -1)
            left_eye_features = gaze_model.eye_cnn(data['left_eye'].permute(0, 3, 1, 2))
            left_eye_features = left_eye_features.reshape(left_eye_features.size(0), -1)
        
        
            face_features = gaze_model.face_cnn(data['image'].permute(0, 3, 1, 2))
            face_features = face_features.reshape(face_features.size(0), -1)
            rotation = data['head_rot'].reshape(data['head_rot'].size(0), -1)

        
            combined_features = torch.cat([right_eye_features, left_eye_features, face_features, rotation, data['pose'].squeeze(dim=2), calibrated_embeddings.squeeze(dim=1)], dim=1)
            predicted_gaze = gaze_model.fc(combined_features)

            # predicted_gaze,pred_query = gaze_model(
            #     data['image'], data['left_eye'], data['right_eye'], data['pose'], data['head_rot'], calibrated_embeddings.long()
            # )
            # print('testing over')
             # Calculate angular error
            pred_gaze_vector = gaze_to_vector(predicted_gaze[:, 1], predicted_gaze[:, 0])
            true_gaze_vector = gaze_to_vector(labels['gaze'][:, 1], labels['gaze'][:, 0])
            angular_errors = angular_error(pred_gaze_vector, true_gaze_vector)
            
            # Print the predicted gaze, ground truth gaze, and angular error for comparison
            for i in range(len(predicted_gaze)):
                print(f"Subject ID: {data['subject_id'][i].item()}, Predicted gaze: {predicted_gaze[i].cpu().numpy()}, Ground truth gaze: {labels['gaze'][i].cpu().numpy()}, Angular error: {angular_errors[i].item()} degrees")

def main():
    # Load the dataset
    # Load the dataset
    dataset = GazeDataset(config.hdf5_path) #give path to test dataset

    # Load the trained models
    gaze_model, calibration_model = load_models()

    # Test the GazeModel with the calibrated embeddings
    test_gaze_model(gaze_model, calibration_model, dataset)


if __name__ == "__main__":
    main()
