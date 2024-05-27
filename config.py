import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    def __init__(self):
        self.learning_rate = 3e-4
        self.batch_size = 8
        self.num_epochs = 50
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.hdf5_path = '/media/dl/DL/pc/pgte/MPIIFaceGaze_multiregion.h5'
        self.save_dir = '/media/dl/DL/pc/pgte/weights'
        self.delta = 1.5

config = Config()

