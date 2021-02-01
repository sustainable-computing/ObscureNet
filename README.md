# ObscureNet
This repository contains the implementation of ObscureNet and the baselines proposed in our IoTDI'21 paper entitled "ObscureNet: Learning Attribute-invariant Latent Representationfor Anonymizing Sensor Data".

Each directory is named after a privacy-preserving method described in the paper.

## Datasets
The 2 Human Activity Recognition (HAR) datasets used to evaluate different methods are MotionSense and MobiAct. 
You can download them from the following websites and use the provided converter (dataset_builder.py) to preprocess the data and turn it into the format that our code expects:
* [MobiAct V2.0 Dataset:](https://bmi.hmu.gr/the-mobifall-and-mobiact-datasets-2)
* [MotionSense Dataset:](https://github.com/mmalekzadeh/motion-sense/tree/master/data)

To reproduce the results of our paper, use the CSV file dataset_subjects, which is provided in this repo, instead of the original one that comes with the MobiAct dataset.

## Dependencies

| Package       | Version       |
| ------------- |:-------------:| 
| Python3       | 3.6.9         |
| Tensorflow    | 1.14.0        |
| PyTorch       | 1.4.0         |
| Keras         | 2.3.1         |

## How to cite ObscureNet
Omid Hajihassani, Omid Ardakanian and Hamzeh Khazaei. 2021. ObscureNet: Learning Attribute-invariant Latent Representation for Anonymizing Sensor Data, In _Proceedings of the 6th ACM/IEEE Conference on Internet of Things Design and Implementation (IoTDI)_.
