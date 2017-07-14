# Take Fine grain labels classe and generate course grain labels
# Note the data set already contain coarse grained labels so this is not needed
import numpy as np
import os
import scipy.misc as misc

#Input fine  grain labels dir
LabelDir="/home/sagi/TensorFlowProjects/DATA_SET/CHEMISTY/LabelsAll/" #Fine grain label input dir for all
# output Coarse grain annotation dir
VesselDir="/home/sagi/TensorFlowProjects/DATA_SET/CHEMISTY/VesselLabels/"
OnePhaseDir="/home/sagi/TensorFlowProjects/DATA_SET/CHEMISTY/OnePhaseLabels/"
LiquidSolidDir="/home/sagi/TensorFlowProjects/DATA_SET/CHEMISTY/LiquidSolidLabels/"
AllPhasesDir="/home/sagi/TensorFlowProjects/DATA_SET/CHEMISTY/AllPhasesLabels/"
if not os.path.exists(VesselDir): os.mkdir(VesselDir)
if not os.path.exists(OnePhaseDir): os.mkdir(OnePhaseDir)
if not os.path.exists(LiquidSolidDir): os.mkdir(LiquidSolidDir)
if not os.path.exists(AllPhasesDir): os.mkdir(AllPhasesDir)
LabelFiles=[]
LabelFiles += [each for each in os.listdir(LabelDir) if each.endswith('.png')]
for itr in range(len(LabelFiles)):
    print(itr)
    Label = misc.imread(LabelDir+LabelFiles[itr])
    if Label==None:
        print("Fail to read: "+LabelFiles[itr])

    OutLabel=np.zeros(Label.shape)
    OutLabel[Label>0]=1
    misc.imsave(VesselDir + LabelFiles[itr], OutLabel.astype(np.uint8))
    OutLabel[Label > 1] = 2
    misc.imsave(OnePhaseDir+LabelFiles[itr], OutLabel.astype(np.uint8))
    OutLabel[Label > 1]=2 #Liquid
    OutLabel[Label >7]=3#Liquid
    OutLabel[Label == 14]=1#solid
    misc.imsave(LiquidSolidDir + LabelFiles[itr], OutLabel.astype(np.uint8))
    misc.imsave(AllPhasesDir + LabelFiles[itr], Label.astype(np.uint8))
#Fine Grain Classes
#All Phases Classes Labels=["Empty","Vessel","Liquid","Liquid Phase two","Suspension", "Emulsion","Foam","Solid","Gel","Powder","Granular","Bulk","Bulk Liquid","Solid Phase","Vapor"]
#Coarse Grain Classes
#Vesse Classes Labels=["Background","Vessel"]
#Phase Classes Labels =["BackGround","Empty Vessel","Filled"]
#Liquid Solid Classes = ["BackGround", "Empty Vessel","Liquid","Solid"]
