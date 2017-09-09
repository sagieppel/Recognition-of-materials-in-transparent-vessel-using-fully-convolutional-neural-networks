# Using fully Convolutional Neural nets for recognition and segmentation of materials in transparent vessels, with emphasis on chemistry lab settings. 
A fully convolutional neural net (FCN) that was modified to be used with the [materials in vessels data set](https://github.com/sagieppel/Materials-in-Vessels-data-set).
The code is based on the more general purposes code for FCN that can be download from [here](https://github.com/sagieppel/Fully-convolutional-neural-network-FCN-for-semantic-segmentation-Tensorflow-implementation)

The FCN was modified labeling of each pixel in the image according to several levels of classes given below (Figure 1):
1) Vessel regions and background regions. 
2) Filled vessel regions and empty vessel regions.
3) Liquid phase and solid phase regions.
4) Exact phase pixel-wise classification (Solid, Liquid, Powder, Suspension, Foam…)

[Dataset of annotated images of materials in glass vessels](https://github.com/sagieppel/Materials-in-Vessels-data-set) and their pixel-wise semantic segmentation  is supplied to support this task and can be download from: [here](https://drive.google.com/file/d/0B6njwynsu2hXRFpmY1pOV1A4SFE/view?usp=sharing)
The code is discussed in the paper [Setting an attention region for convolutional neural networks using region selective features, for recognition of materials within glass vessels](https://arxiv.org/abs/1708.08711)
![](/Figure1.png)


## Details input/output
convolutional neural network (FCN) in chemistry lab setting
 
The fully convolutional neural network receives an image with materials in a glass vessel and performs semantic segmentation of the image, such that each pixel the in the image is assigned several labels. The network performed the pixel-wise labeling in several levels of class granularity and return per pixels label for each categories set. All the predictions are generated simultaneously in one iteration of the net.
The output segmentation maps/images are as following (Figure 1): 

a. Vessel/Background: For each pixel assign the value of 1 if it in the vessel and 0 otherwise.

b. Filled/Empty: similar to above but also distinguish between the filled and empty region of the vessel. For each pixel assign, one of the three values: 0) Background, 1) Empty vessel. 2) Filled vessel 

c. Phase type: Similar to above but distinguish between liquid and solid regions of the filled vessel.   For each pixel assign, one of the four values: 0) Background, 1) Empty vessel. 2) Liquid. 3) Solid.

d. Exact phase type: Similar to above but distinguish between naterials phase. For each pixel assign one of 15 values: 1) background. 2) Vessel. 3) Liquid. 4) Liquid Phase two. 5) Suspension. 6) Emulsion. 7) Foam. 8) Solid. 9) Gel. 10) Powder. 11) Granular. 12) Bulk. 13) Bulk Liquid. 14) Solid Phase two. 15) Vapor.

![](/Figure2.jpg) 
 
## Requirements
This network was run and trained with [Python 3.6 Anaconda](https://www.continuum.io/downloads) package and [Tensorflow 1.1](https://www.tensorflow.org/install/).
The training was done using Nvidia GTX 1080, on Linux Ubuntu 16.04.
 
## Setup

1) Download the code from the repository.
2) Download pre-trained vgg16 net and put in the /Model_Zoo subfolder in the main code folder. A pre-trained vgg16 net can be download from [here](ftp://mi.eng.cam.ac.uk/pub/mttt2/models/vgg16.npy) or from (https://drive.google.com/file/d/0B6njwynsu2hXZWcwX0FKTGJKRWs/view?usp=sharing)[https://drive.google.com/file/d/0B6njwynsu2hXZWcwX0FKTGJKRWs/view?usp=sharing]
3) Download [dataset of images materials in transparent vessels](https://github.com/sagieppel/Materials-in-Vessels-data-set) and extract in /Data_Zoo folder in the main code dir. The dataset can be download from [https://drive.google.com/file/d/0B6njwynsu2hXRFpmY1pOV1A4SFE/view?usp=sharing](https://drive.google.com/file/d/0B6njwynsu2hXRFpmY1pOV1A4SFE/view?usp=sharing) 
5) If not interested in training the net then download a pre-trained model and extract to /log folder in the main code dir. The pre-trained model could be download from: [here](https://drive.google.com/file/d/0B6njwynsu2hXWi1YZ3JKRmdLOWc/view?usp=sharing)

## Tutorial
### Predicting: pixel-wise classification and segmentation of images 
Run: Inference.py    
Notes: Make sure that the Image_dir variable is pointing to a valid image folder, and that the /log folder contains a trained net.

### Training network:
 Run:  Train.py 
 
 Note: Make sure the /Data_Zoo folder contains the downloaded data set

### Evaluating net performance using intersection over union (IOU):
 
Run: Evaluate_Net_IOU.py
 
Notes:  Make sure you downloaded the dataset to /Data_Zoo and that the /log folder contains a trained net.
 
## Background 
The code is based on fully convolutional neural net (FCN) described in the paper [Fully Convolutional Networks for Semantic Segmentation](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)
The main modification is that the last prediction layer is split to give a prediction at several levels of classes for each pixel. Similarly, training of the network was done with several loss functions simultaneously one for each set of classes. See BuildNetVgg16.py for the network structure. The code is based on https://github.com/shekkizh/FCN.tensorflow by Sarath Shekkizhar with different ecoder mode.
The net is based on the pre-trained VGG16 model by [Marvin Teichmann](https://github.com/MarvinTeichmann)




#### Thanks
The images in the [data set](https://github.com/sagieppel/Materials-in-Vessels-data-set) were labeled by Mor bismuth and Alexandra Emanual. Images were taken with permission from Youtube channels NileRed, NurdeRage, and ChemPlayer. 


## Links
If the vessel region is already known and used as input for the net see:
https://github.com/sagieppel/Convolutional_Neural_Nets_With_ROI_input_and_region_selective_features_For_Materials_in_vessels

