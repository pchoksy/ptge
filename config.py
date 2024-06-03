import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    def __init__(self):
        self.learning_rate = 3e-4
        self.batch_size = 8
        self.num_epochs = 50
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.hdf5_path = './' + '/MPIIFaceGaze_multiregion.h5' #train dataset hdf file
        self.test_hdf5_path = './' + '/MPIIFaceGaze_multiregion.h5' #test dataset hdf file
        self.save_dir = './' + '/weights' # create folder named 'weights', to store weights
        self.delta = 1.5

config = Config()

