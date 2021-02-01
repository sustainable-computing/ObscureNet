# ObscureNet
Following is the code and implementation of ObscureNet and all the baselines used in the paper published and accepted in IoTDI'21. The paper is entitled "ObscureNet: Learning Attribute-invariant Latent Representationfor Anonymizing Sensor Data".

Different directories are named according to the techniques and baselines discussed in the paper.

## Datasets
Two Human Activity Recognition (HAR) datasets used in this implementation are the MotionSense and MobiAct datasets. Please download the datasets accordingly and used the dataset builder codes in this repository and dataset directories to create the processed data used by the programs. To get the results use the dataset_subject_information CSV file provided by this repo and ignore the one in the original downloaded dataset MobiAct V 2.0.

MobiAct V 2.0 Dataset: https://drive.google.com/file/d/0B5VcW5yHhWhielo5NTk1Q3ZiWDQ/edit

MotionSense Dataset: https://github.com/mmalekzadeh/motion-sense/tree/master/data

## Development Setup
Python3 === 3.6.9
Tensorflow === 1.14.0
Pytorch === 1.4.0
Keras === 2.3.1
