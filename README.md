# Brain Tumor Prediction using Deep Learning


## Introduction
The brain tumor is deemed to be the 10th leading cause of death. A brain tumor is formed when certain abnormal cells in the brain multiply and start affecting surrounding nerves in the brain. The diagnosis and treatment of brain tumours can be much more accurate and faster with the help of automatic segmentation. Various image classification techniques can be applied on the MRI images and using Machine Learning and Neural Networks, these images can be trained to predict the classification of brain tumors with high accuracy. In this paper, we have applied various image classification techniques namely Convolutional Neural Network (CNN), Multilayer Perceptron (MLP), and Transfer Learning using ResNet to classify and predict brain tumours with high accuracy.

## Dataset
The dataset used in this project is [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) from Kaggle, which is a collection of 7000 images classified into 4 classes - 
* Glioma
* Meningioma
* Pituitary
* No Tumor
The image data can be found in the following subdirectory of this repo - main/data/

## Preprocessing the data
#### Data Normalization
To normalize the image data, the pixel values in image array are divided by 255 to have values between 0 and 1.
#### Data Augmentation
To perform data augmentation, following augmentation techniques are used:
* Rotation - randomly rotate images by 90 degrees
* Horizontal Flip - randomly flip images horizontally
* Vertical Flip - randomly flip images vertically
* Brightness Range - randomly brighten the images within a specified range

## Training the model
To train the model, I have used the Sequential() class of tensorflow.keras library. In this sequential model, we add our layers as required.
The code for the training of the model can be found in - main/src/

## Evaluation the model
Evaluation of the model is done using the following matrices:
* Accuracy
* Precision
* Recall
* F1 Score
The code for the evaluation can be found in - main/src/

## Instructions to run the project
* Run the folloiwng command to clone the github repo
```bash
git clone https://github.com/laveenbhatia/Brain-Tumor-Prediction-Using-Deep-Learning.git
```

* Run the following command to install the required dependencies
```bash
pip install -r requirements.txt
```

* If you wish to use your own custom data to train then add your custom images in the repective class folder in data/Brain_Tumor_Training/[Class]
* To train the model, go the the folder main/src. Here you will see 3 different folders - CNN, MLP, RESNET. Each folder caontains the source code to train the respective model. For example, the folder CNN contains a file Model_Train_CNN.py. Run this to train the CNN model. If required, the hyper parameters can be tuned in this file itself.
* Similarly to evaluate the model, go to main/src. Then go to the folder for which model you wish to evaluate and run Model_Eval.py.