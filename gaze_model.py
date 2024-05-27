import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class GazeModel(nn.Module):
    def __init__(self):
        super(GazeModel, self).__init__()
        self.eye_cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
            # models.vgg16(pretrained=True).features[:9],  # first four convolutional layers of VGG16 pretrained on ImageNet
            # nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), padding='same'),
            # nn.ReLU(inplace=True),
            # nn.BatchNorm2d(64),
            # nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(2, 2)),
            # nn.ReLU(inplace=True),
            # nn.BatchNorm2d(64),
            # nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(3, 3)),
            # nn.ReLU(inplace=True),
            # nn.BatchNorm2d(64),
            # nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(4, 5)),
            # nn.ReLU(inplace=True),
            # nn.BatchNorm2d(128),
            # nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(5, 11)),
            # nn.ReLU(inplace=True),
            # nn.BatchNorm2d(128),
        )

        self.face_cnn = nn.Sequential(
            # models.vgg16(pretrained=True).features[:9],  # first four convolutional layers of VGG16 pretrained on ImageNet
            # nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), padding='same'),
            # nn.ReLU(inplace=True),
            # nn.BatchNorm2d(64),
            # nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(2, 2)),
            # nn.ReLU(inplace=True),
            # nn.BatchNorm2d(64),
            # nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(3, 3)),
            # nn.ReLU(inplace=True),
            # nn.BatchNorm2d(64),
            # nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(5, 5)),
            # nn.ReLU(inplace=True),
            # nn.BatchNorm2d(128),
            # nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(11, 11)),
            # nn.ReLU(inplace=True),
            # nn.BatchNorm2d(128),
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self._initialize_weights()


        self.fc = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def _initialize_weights(self):
        # Create a dummy input to calculate the size of the flattened features
        with torch.no_grad():
            dummy_left = torch.zeros(1, 3, 224, 224)
            dummy_right = torch.zeros(1, 3, 224, 224)
            dummy_face = torch.zeros(1, 3, 224, 224)
            lefteye_features = self.eye_cnn(dummy_left)
            righteye_features = self.eye_cnn(dummy_right)
            face_features = self.face_cnn(dummy_face)
            self.flattened_size = lefteye_features.numel() + righteye_features.numel() + face_features.numel() + 3  # Adding 3 for the pose
            

    def forward(self, image, left, right, pose ):

        right_eye_features = self.eye_cnn(right.permute(0, 3, 1, 2))
        right_eye_features = right_eye_features.reshape(right_eye_features.size(0), -1)
        left_eye_features = self.eye_cnn(left.permute(0, 3, 1, 2))
        left_eye_features = left_eye_features.reshape(left_eye_features.size(0), -1)
        
        
        face_features = self.face_cnn(image.permute(0, 3, 1, 2))
        face_features = face_features.reshape(face_features.size(0), -1)

        combined_features = torch.cat([right_eye_features, left_eye_features, face_features, pose.squeeze(dim=2)], dim=1)
        #print('comb_size', combined_features.numel())
        gaze = self.fc(combined_features)

       
        return gaze, combined_features



