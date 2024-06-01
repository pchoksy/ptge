# Preprocess the data and create HDF5 file

To normalize the MPIIFaceGaze dataset, pleas put the file normalize_data.py in the MPIIFaceGaze folder. You also need the files from MPIIGaze dataset. They are the files in the folder of 'Evaluation Subset/sample list for eye image' and the '.mat' file.

In normalize_data.py, change path of this file on line number: 172

     please run: 'python normalize_data.py'

# Train the gaze model and calibration model

To train model, create folder name 'weights'and change necessary path into config file.
      
     please run: 'python gaze_train.py'
     please run: 'python calibration_train.py'

# Test the model
Test model prints ground truth gaze and predicted gaze along with their angular error. Change necessary path into config file

     please run: 'python test.py'
