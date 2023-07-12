#!/bin/bash

# activate conda environment
source ~/ls/etc/profile.d/conda.sh
conda activate bwh_plgg

## set environment variables
export CUDA_VISIBLE_DEVICES=1
export PROJECT_DIRECTORY="./"

## switch to the segmentation directory 
cd aidan_segmentation/nnUNet_pLGG
## run python script for preprocessing the images 
python pipeline_mri_preprocess_3d.py


## set the required flags 
export RESULTS_FOLDER="$PWD/nnUNet/nnUNet_trained_models"
echo $RESULTS_FOLDER

## run the segmentation command 
nnUNet_predict -i $PWD/output_preprocess/nnunet/imagesTs  -o  $PWD/output_mask  -t 871 -m 3d_fullres --save_npz

## change the environment for BRAF prediction  
conda deactivate 
conda activate 2d_approach
cd ../.. 

## get the min and max tumor index from segmentation  
python get_min_max.py
## get the sliced data 
python pLGG/get_BRAF_data_v2.py

## Infer the models 
# infer wildtype classifier
python pLGG/main2.py --saved_model tumor__wildtype_radimagenet_fusion_crosstrain_fullimage_internaltestasvalidationResNet50_imagenet_23_0.73.h5 --subtype wildtype
# infer fusion classifier 
python pLGG/main2.py --saved_model tumor_fusion_radimagenet_fullimage_internaltestasvalidationResNet50_imagenet_21_0.75.h5 --subtype fusion
# infer v600e classifier
python pLGG/main2.py --saved_model tumor_v600e_radimagenet_wildtypecrosstrain_filteredv600e_fullimage_internaltestasvalidationResNet50__35_0.73.h5 --subtype v600e
## run the consensus decision block script 
python consensus.py
## output the classification decision 
python pLGG/decision.py
