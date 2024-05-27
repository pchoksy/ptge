# import os
# import cv2
# import h5py
# import numpy as np
# import torch
# from torch.utils.data import Dataset


# class GazeDataset(Dataset):
#     def __init__(self, hdf5_path):
#         self.hdf5_path = hdf5_path
#         self.data = []
#         self.labels = []

#         with h5py.File(hdf5_path, 'r') as hdf:
#             for subject in hdf.keys():
#                 subject_group = hdf[subject]
#                 for img_id in subject_group.keys():
#                     img_group = subject_group[img_id]
                    
#                     img_face = img_group['image'][()]  # Original face image
#                     img_left = img_group['left'][()]    # Normalized left eye image
#                     img_right = img_group['right'][()]  # Normalized right eye image
#                     pose = img_group['pose'][()]        # pose
#                     gaze = img_group['gaze'][()]        # Gaze direction

#                     self.data.append({
#                         'image': img_face,
#                         'left_eye': img_left,
#                         'right_eye': img_right,
#                         'pose': pose
#                     })
                    
#                     self.labels.append({
#                         'gaze': gaze
#                     })

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         data = self.data[idx]
#         labels = self.labels[idx]

#         # Convert data and labels to tensors
#         for key in data:
#             data[key] = torch.tensor(data[key], dtype=torch.float32)
#         for key in labels:
#             labels[key] = torch.tensor(labels[key], dtype=torch.float32)

#         return data, labels

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class GazeDataset(Dataset):
    def __init__(self, hdf5_path):
        self.hdf5_path = hdf5_path
        self.data = []
        self.labels = []

        with h5py.File(hdf5_path, 'r') as hdf:
            for subject in hdf.keys():  # Iterate through each subject (e.g., p00, p01, ..., p14)
                subject_group = hdf[subject]

                # Assuming all datasets are directly under the subject group
                try:
                    image_group = subject_group['image']
                    left_group = subject_group['left']
                    right_group = subject_group['right']
                    pose_group = subject_group['pose']
                    gaze_group = subject_group['gaze']

                    for img_id in image_group.keys():  # Iterate through each image ID
                        try:
                            img_face = image_group[img_id][()]   # Original face image
                            img_left = left_group[img_id][()]    # Normalized left eye image
                            img_right = right_group[img_id][()]  # Normalized right eye image
                            pose = pose_group[img_id][()]        # Pose information
                            gaze = gaze_group[img_id][()]        # Gaze direction

                            self.data.append({
                                'image': img_face,
                                'left_eye': img_left,
                                'right_eye': img_right,
                                'pose': pose
                            })

                            self.labels.append({
                                'gaze': gaze
                            })
                        except KeyError as e:
                            print(f"KeyError: {e}. Skipping entry {img_id} in subject {subject}")
                except KeyError as e:
                    print(f"KeyError: {e}. Skipping subject {subject}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        labels = self.labels[idx]

        # Convert data and labels to tensors
        for key in data:
            data[key] = torch.tensor(data[key], dtype=torch.float32)
        for key in labels:
            labels[key] = torch.tensor(labels[key], dtype=torch.float32)

        return data, labels




