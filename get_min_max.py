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



# util functions 
# function to get the top and the bottom slice index of tumor 

def get_top_bottom_slice_index(mask_image_path):
    mask = nib.load(mask_image_path).get_fdata()
    index = np.where(mask != 0)
    return np.min(index[2]),np.max(index[2])
    
#get_top_bottom_slice_index("/home/divyanshu/BRAF/monai/data/masks/2280828.nii.gz")

image_path = os.environ.get("PWD") + "/aidan_segmentation/nnUNet_pLGG/output_preprocess/nnunet/imagesTs/input_0000.nii.gz"
mask_path = os.environ.get("PWD") + "/aidan_segmentation/nnUNet_pLGG/output_mask/input_t2w_mri.nii.gz"

metadata_dict = {
    "bch mrn" : [],
    "label" : [],
    "top z index" : [],
    "bottom z index" : [],
    "image path" : [],
    "mask path" : [],
}


bottom_z, top_z = get_top_bottom_slice_index(mask_path)

metadata_dict["bch mrn"].append(0000000)
metadata_dict["label"].append(0)
metadata_dict["top z index"].append(top_z)
metadata_dict["bottom z index"].append(bottom_z)
metadata_dict["image path"].append(image_path)
metadata_dict["mask path"].append(mask_path)

df = pd.DataFrame(metadata_dict)

# Save the DataFrame to a CSV file
csv_file = 'zmin_zmax.csv'
df.to_csv(csv_file, index=False)

