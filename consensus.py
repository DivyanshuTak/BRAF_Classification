import os
import numpy as np 
import nibabel as nib 
import csv 
import pandas as pd 
import pickle 
import shutil 
import random 
import matplotlib.pyplot as plt 
import SimpleITK as sitk 
from scipy.ndimage.measurements import center_of_mass
from skimage.measure import label
import csv
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.metrics import roc_curve



##========================================================================================================
##                          FIRST CHECK FOR BRAF vs WILDTYPE
##========================================================================================================
# first get the non braf indexes 
v600e = pd.read_csv(os.environ.get("PWD")+"/BRAF/2d_data/v600e_classifier.csv")
fusion = pd.read_csv(os.environ.get("PWD")+"/BRAF/2d_data/fusion_classifier.csv")
wildtype = pd.read_csv(os.environ.get("PWD")+"/BRAF/2d_data/wildtype_classifier.csv")
all_class_gt = pd.read_csv(os.environ.get("PWD")+"/BRAF/2d_data/tr_img_df.csv")

gt = list(v600e["label"])
v600e_pred = list(v600e["y_pred_class"])
fusion_pred = list(fusion["y_pred_class"])
wildtype_pred = list(wildtype["y_pred_class"])
allclass_gt = list(all_class_gt["label"]) 

# calculate predictions 
y_pred_class = [int(((x==0 or y==0) and z==1)) for x,y,z in zip(v600e_pred, fusion_pred, wildtype_pred)]
y_pred_score = []

# calculate scores 
v600e_score = list(v600e["y_pred"])
fusion_score = list(fusion["y_pred"])
wildtype_score = list(wildtype["y_pred"])

for index, _ in enumerate(v600e_pred):
    P_A0 = 1 - v600e_score[index]
    P_B0 = 1 - fusion_score[index]
    P_A0_B0 = P_A0 * P_B0
    P_A0_or_B0 = P_A0 + P_B0 - P_A0_B0
    P_C1 = wildtype_score[index]
    P_result = P_A0_or_B0 * P_C1
    y_pred_score.append(P_result)
    
    
#print(classification_report(gt, y_pred_class, target_names=["BRAF", "NON BRAF"]))
#auc = roc_auc_score(gt, y_pred_score)
#auc = np.around(auc, 3)
#print('auc:', auc)


y_pred_class_mask = [1-a for a in y_pred_class]
#print(len(y_pred_class_mask))
#print(len(y_pred_score))


## SAVE THE SCORES FOR BRAF MASKS 

external = pd.read_csv(os.environ.get("PWD")+"/BRAF/2d_data/wildtype_classifier.csv")
external["y_pred"] = y_pred_score
external["y_pred_class"] = y_pred_class

df_test_pred = external[['ID', 'label', 'y_pred', 'y_pred_class']]
df_test_pred.to_csv(os.path.join(os.environ.get("PWD")+"/BRAF/2d_data", "BRAF_mask_external.csv")) 


##========================================================================================================
##                          SECOND CHECK FOR FUSION 
##========================================================================================================

fusion = pd.read_csv(os.environ.get("PWD")+"/BRAF/2d_data/fusion_classifier.csv")
fusion_gt = list(fusion["label"])
fusion_pred = list(fusion["y_pred_class"])
fusion_score = list(fusion["y_pred"])
# get prediction
new_fusion_pred = [a * b for a, b in zip(y_pred_class_mask, fusion_pred)]
#print(classification_report(fusion_gt, new_fusion_pred, target_names=["ALL", "Fusion"]))


# get score 
#masked_scores_braf = [a * b for a, b in zip(y_pred_class_mask, fusion_score)]#y_pred_class_mask*y_pred_score
#masked_scores_fusion = [a * b for a, b in zip(y_pred_class, y_pred_score)]#y_pred_class*fusion_score
first_stage_fusion_score = []
for index, each in enumerate(y_pred_class_mask):
    if each==1:
        first_stage_fusion_score.append(fusion_score[index])
    else:
        first_stage_fusion_score.append(fusion_score[index]*y_pred_score[index])

#auc = roc_auc_score(fusion_gt, first_stage_fusion_score)
#auc = np.around(auc, 3)
#print('auc:', auc)



## SAVE THE SCORES FOR FIRST STAGE FUSION

external = pd.read_csv(os.environ.get("PWD")+"/BRAF/2d_data/fusion_classifier.csv")
external["y_pred"] = first_stage_fusion_score
external["y_pred_class"] = new_fusion_pred

df_test_pred = external[['ID', 'label', 'y_pred', 'y_pred_class']]
df_test_pred.to_csv(os.path.join(os.environ.get("PWD")+"/BRAF/2d_data", "FIRST_STAGE_fusion_external.csv")) 


##========================================================================================================
##                          THIRD CHECK FOR V600E
##========================================================================================================
## use fusion as the mask for v600e prediction 

y_pred_class_mask_fusion = [1-a for a in new_fusion_pred]
#print(len(y_pred_class_mask_fusion))

v600e = pd.read_csv(os.environ.get("PWD")+"/BRAF/2d_data/v600e_classifier.csv")
v600e_gt = list(v600e["label"])
v600e_pred = list(v600e["y_pred_class"])
v600e_score = list(v600e["y_pred"])

v600e_pred_single_stage = v600e_pred*1
#v600e_pred_double_stage = new_v600e_pred*1


#new_v600e_pred_single_stage = [a * b for a, b in zip(y_pred_class_mask_fusion, v600e_pred_single_stage)]
new_v600e_pred_double_stage = [a * b for a, b in zip(y_pred_class_mask_fusion, v600e_pred_single_stage)]
#print(classification_report(v600e_gt, new_v600e_pred_double_stage, target_names=["ALL", "V600E"]))


second_stage_v600e_score = []
for index, each in enumerate(y_pred_class_mask_fusion):
    if each==1:
        second_stage_v600e_score.append(v600e_score[index])
    else:
        second_stage_v600e_score.append(v600e_score[index]*first_stage_fusion_score[index])

#auc = roc_auc_score(v600e_gt, second_stage_v600e_score)
#auc = np.around(auc, 3)
#print('auc:', auc)

#print(classification_report(v600e_gt, new_v600e_pred_single_stage, target_names=["ALL", "V600E"]))



## SAVE THE SCORES FOR SECOND STAGE V600E

external = pd.read_csv(os.environ.get("PWD")+"/BRAF/2d_data/v600e_classifier.csv")
external["y_pred"] = second_stage_v600e_score
external["y_pred_class"] = new_v600e_pred_double_stage

df_test_pred = external[['ID', 'label', 'y_pred', 'y_pred_class']]
df_test_pred.to_csv(os.path.join(os.environ.get("PWD")+"/BRAF/2d_data", "SECOND_STAGE_v600e_external.csv")) 