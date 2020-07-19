# MMO-detection
## Introduction

Code for maximum Markovian order detection.

## Dataset

#### Data Souce

In our paper, we used datasets of pigeon flock, midge group and dog group. Unfortunately, we do not have the copyright of these data. Therefore, we cannot upload these data with our code.

Now, in our code, we use a dataset generated by Vicsek model as a demo to show our algorithm. The detected optimal order is 1, as Vicsek model is a first-order model.

If you want to get access to those data, you might have to ask the owner of the corresponding dataset for permission.

#### Data format

You can apply the algorithm to your custimized dataset.

Your dataset should be made by the following format and requirements:

1. The dataset should be a binary file, which can be read by Pickle.
2. The dimension of the dataset should be n * t * d, where n denotes the number of the individuals in the group, t denotes the number of time stamps and d denotes the dimension of the motion data. To be specific, if there is a 2-D motion dataset of 10 pigeons which consists of 1000 time stamps of observation, then the dimension of the dataset should be 10 * 1000 * 2.
3. The input dataset should be the time-series dataset of velocity fluctuation (see eq.3 in our paper) or acceleration due to the requirement of oCEP algorithm.

If your dataset is in MatLab file format (.mat file), you can easily convert it to binary file by using Scipy and Pickle. For more details, see [here](https://stackoverflow.com/questions/874461/read-mat-files-in-python).

## Usage

First, put the motion dataset in folder ``MMO-detection/data/`` and make sure the dataset file is named as ``input_data``.

Then, enter folder ``MMO-detection/`` and run ``main.py`` . The optimal order will be detected. If you are using command line, you can use

```bash
cd MMO-detection/
python3 main.py
```

## Requirements

The program is written by Python and Python version should be 3.6 or latter. The specific packages required are as follows:

| Package | Version             |
| ------- | ------------------- |
| pathpy  | 2.1.0               |
| Numpy   | The lattest version |
| Pickle  | The lattest version |
| Scipy   | The lattest version |

If you are using Anaconda, you can set up the environment by following command line:

To install pathpy, we recommend to use the command line

```bash
pip install pathpy==2.1.0
```

Or you can download the package from [here](https://pypi.org/simple/pathpy/) and install it locally (not recommended).

To install rest of the packages, you can use the command lines listed below.

```bash
conda install numpy
conda install pickle
conda install scipy
```

