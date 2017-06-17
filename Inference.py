# Run prediction and genertae pixelwise annotation for every pixels in the image .
# Output saved as label images, and label image overlay on the original image
# By defualt this should work, as is if you follow the intsructions provide in the readme file
# 1) Make sure you you have trained model in logs_dir (See Train.py for creating trained model)
# 2) Set the Image_Dir to the folder where the input image for prediction are located
# 3) Set Pred_Dir the folder where you want the output annotated images to be save
#--------------------------------------------------------------------------------------------------------------------
import tensorflow as tf
import numpy as np
import scipy.misc as misc
import sys
import BuildNetVgg16
import TensorflowUtils
import os
logs_dir= "logs/"# "path to logs directory where trained model and information will be stored"
Image_Dir="Data_Zoo/Materials_In_Vessels/Test_Images/"# Images for testing network
Pred_Dir="Output_Prediction/" # Library where the output prediction will be written
model_path="Model_Zoo/vgg16.npy"# "Path to pretrained vgg16 model for encoder"
NameEnd="" # Add this string to the ending of the file name optional

#-----------------------------------------Check if models and data are availale----------------------------------------------------
if not os.path.isfile(model_path):
    print("Warning: Cant find pretrained vgg16 model for network initiation. Please download  mode from:")
    print("ftp://mi.eng.cam.ac.uk/pub/mttt2/models/vgg16.npy")
    print("Or from")
    print("https://drive.google.com/file/d/0B6njwynsu2hXZWcwX0FKTGJKRWs/view?usp=sharing")
    print("and place in: Model_Zoo/")

if not os.path.isdir(Image_Dir):
    print("Warning: Cant find images for interference. You can downolad test images from:")
    print("https://drive.google.com/file/d/0B6njwynsu2hXelJJOFdqRjhGWWM/view?usp=sharing")
    print("and extract in: Data_Zoo/")
TensorflowUtils.maybe_download_and_extract(model_path.split('/')[0], "ftp://mi.eng.cam.ac.uk/pub/mttt2/models/vgg16.npy") #If not exist try to download pretrained vgg16 net for network initiation
#-------------------------------------------------------------------------------------------------------------------------
NUM_CLASSES = 15+2+3+4
###################Overlay Label on image Mark label on  the image in transperent form###################################################################################
def OverLayLabel(ImgIn,Label,W):
    #ImageIn is the image
    #Label is the label per pixel
    # W is the relative weight in which the labels will be marked on the image
    # Return image with labels marked over it
    Img=ImgIn.copy()
    TR=[0,1,0,0,0,1,1,0.5,0  ,0  ,0  ,0.5,0,1   ,0.5]
    TB=[0,0,1,0,1,0,1,0  ,0.5,0  ,0.5,0  ,0,0.5 ,0.5]
    TG=[0,0,0,1,1,1,0,0  ,0  ,0.5,0.5,0.5,0,0.25,0.5]
    R=Img[:,:,0].copy()
    G=Img[:,:,1].copy()
    B=Img[:,:,2].copy()
    for i in range(1,15):
        R[Label == i] = TR[i]*255
        G[Label == i] = TG[i]*255
        B[Label == i] = TB[i]*255
    Img[:, :, 0] = Img[:, :, 0] * (1 - W) + R * W
    Img[:, :, 1] = Img[:, :, 1] * (1 - W) + G * W
    Img[:, :, 2] = Img[:, :, 2] * (1 - W) + B * W
    return Img
#####################################################################################################################################
def LoadImage(ImageName):
    Img=misc.imread(ImageName)
    #Img = misc.imresize(Img, [Im_Hight,Im_Width], interp='bilinear')
    Img=Img[:,:,0:3]
    Img= np.expand_dims(Img, axis=0)
    return Img

################################################################################################################################################################################
def main(argv=None):
    keep_prob = tf.placeholder(tf.float32, name="keep_probabilty")  # Dropout probability
    image = tf.placeholder(tf.float32, shape=[None, None, None, 3],name="input_image")  # Input image batch first dimension image number second dimension width third dimension height 4 dimension RGB
    #-------------------------Build Net----------------------------------------------------------------------------------------------
    Net = BuildNetVgg16.BUILD_NET_VGG16(vgg16_npy_path=model_path)  # Create class for the network
    Net.build(image, NUM_CLASSES, keep_prob) # Build net and load intial weights (weights before training)
    #----------------------Create list of images for annotation prediction (all images in Image_Dir)-----------------------------------------------
    ImageFiles=[]   #Create list of images in Image_Dir  for label prediction
    ImageFiles += [each for each in os.listdir(Image_Dir) if each.endswith('.PNG') or each.endswith('.JPG') or each.endswith('.TIF') or each.endswith('.GIF') or each.endswith('.png') or each.endswith('.jpg') or each.endswith('.tif') or each.endswith('.gif') ] # Get list of training images



    print('Number of  images='+str(len(ImageFiles)))




#-------------------------Load Trained model if you dont have trained model see: Train.py-----------------------------------------------------------------------------------------------------------------------------

    sess = tf.Session() #Start Tensorflow session

    print("Setting up Saver...")
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(logs_dir)
    if ckpt and ckpt.model_checkpoint_path: # if train model exist restore it
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")
    else:
        print("ERROR NO TRAINED MODEL IN: "+ckpt.model_checkpoint_path+" See Train.py for creating train network ")
        print("or download from: https://drive.google.com/file/d/0B6njwynsu2hXMjdLcjNfb2tNSUE/view?usp=sharing and extract in log_dir")
        sys.exit()


#--------------------Create output directories for predicted label, one folder for each granulairy of label prediciton---------------------------------------------------------------------------------------------------------------------------------------------

    if not os.path.exists(Pred_Dir): os.makedirs(Pred_Dir)
    if not os.path.exists(Pred_Dir+"/OverLay"): os.makedirs(Pred_Dir+"/OverLay")
    if not os.path.exists(Pred_Dir + "/OverLay/Vessel/"): os.makedirs(Pred_Dir + "/OverLay/Vessel/")
    if not os.path.exists(Pred_Dir + "/OverLay/OnePhase/"): os.makedirs(Pred_Dir + "/OverLay/OnePhase/")
    if not os.path.exists(Pred_Dir + "/OverLay/LiquiSolid/"): os.makedirs(Pred_Dir + "/OverLay/LiquiSolid/")
    if not os.path.exists(Pred_Dir + "/OverLay/AllPhases/"): os.makedirs(Pred_Dir + "/OverLay/AllPhases/")
    if not os.path.exists(Pred_Dir + "/Label"): os.makedirs(Pred_Dir + "/Label")
    if not os.path.exists(Pred_Dir + "/Label/Vessel/"): os.makedirs(Pred_Dir + "/Label/Vessel/")
    if not os.path.exists(Pred_Dir + "/Label/OnePhase/"): os.makedirs(Pred_Dir + "/Label/OnePhase/")
    if not os.path.exists(Pred_Dir + "/Label/LiquiSolid/"): os.makedirs(Pred_Dir + "/Label/LiquiSolid/")
    if not os.path.exists(Pred_Dir + "/Label/AllPhases/"): os.makedirs(Pred_Dir + "/Label/AllPhases/")
    if not os.path.exists(Pred_Dir + "/AllPredicitionsDisplayed/"): os.makedirs(Pred_Dir + "/AllPredicitionsDisplayed/")
    
    print("Running Predictions:")
    print("Saving output to:" + Pred_Dir)
 #----------------------Go over all images and predict semantic segmentation in various of classes-------------------------------------------------------------
    for fim in range(len(ImageFiles)):
        print(str(fim)+") "+Image_Dir + ImageFiles[fim])

        Images= LoadImage(Image_Dir + ImageFiles[fim]) # Load nex image

        #Run net for label prediction
        AllPhases, LiquidSolid, OnePhase, Vessel= sess.run([Net.AllPhasesPred, Net.LiquidSolidPred, Net.PhasePred, Net.VesselPred ], feed_dict={image: Images, keep_prob: 1.0})


#------------------------Save predicted labels and the labels in /label/ folder, and the label overlay on the image in /overlay/ folder
        misc.imsave(Pred_Dir + "/Label/Vessel/" + ImageFiles[fim]+NameEnd  , Vessel[0])
        misc.imsave(Pred_Dir + "/OverLay/Vessel/"+ ImageFiles[fim]+NameEnd  , OverLayLabel(Images[0],Vessel[0], 0.6))
        misc.imsave(Pred_Dir + "/Label/OnePhase/" + ImageFiles[fim] + NameEnd, OnePhase[0])
        misc.imsave(Pred_Dir + "/OverLay/OnePhase/" + ImageFiles[fim] + NameEnd,OverLayLabel(Images[0], OnePhase[0], 0.6))
        misc.imsave(Pred_Dir + "/Label/LiquiSolid/" + ImageFiles[fim] + NameEnd, LiquidSolid[0])
        misc.imsave(Pred_Dir + "/OverLay/LiquiSolid/" + ImageFiles[fim] + NameEnd,OverLayLabel(Images[0], LiquidSolid[0], 0.6))
        misc.imsave(Pred_Dir + "/Label/AllPhases/" + ImageFiles[fim] + NameEnd,  AllPhases[0])
        misc.imsave(Pred_Dir + "/OverLay/AllPhases/" + ImageFiles[fim] + NameEnd,OverLayLabel(Images[0], AllPhases[0], 0.6))
        misc.imsave(Pred_Dir + "/AllPredicitionsDisplayed/" + ImageFiles[fim],np.concatenate((Images[0], OverLayLabel(Images[0],Vessel[0],0.7),OverLayLabel(Images[0], OnePhase[0], 0.7),OverLayLabel(Images[0], LiquidSolid[0], 0.7), OverLayLabel(Images[0], AllPhases[0], 0.7)), axis=1))
print("Finished Running")
if __name__ == "__main__":
    tf.app.run()
