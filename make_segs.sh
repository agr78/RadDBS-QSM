#! /bin/bash
# Make QSM atlas and transform segmentation to patient space
# QSMs can be reconstructed from complex gradient echo data with `main.m`
# Prefixes, identifiers, and list paths can be updated
# Alexandra G. Roberts
# 10/07/2024
# Cornell MRI Lab
# If this code is used, please cite:
#
# A. Roberts, et al. χ-DBS: An Open-Source Susceptibility Atlas Tool for Deep Brain 
# Stimulation Target Visualization and Segmentation. Mov Disord. 2025; 40
# 
# Tustison NJ et al.,
# N4ITK: improved N3 bias correction. 
# IEEE Trans Med Imaging. 2010 Jun;29(6):1310-20. doi: 10.1109/TMI.2010.2046908.
# 
# B. B. Avants et al., 
# "The optimal template effect in hippocampus studies of diseased populations," 
# NeuroImage, vol. 49, no. 3, pp. 2457-2466, 2010, doi: 10.1016/j.neuroimage.2009.09.062.

# Read each subject ID
while read -r line; do
echo $line
prefix=${line#*./000000}
id=${prefix%/*}
echo "Beginning registration for $id"
# Check if bias correction was completed
if ! test -f ./n4bc/n4bc_$id.nii.gz; then
	    N4BiasFieldCorrection -d 3 -i ./orig/$id.nii.gz -o ./n4bc/mag_n4bc_$id.nii.gz
fi
# Check if brain has been extracted
if ! test -f ./xtracted/mag_$id"BrainExtractionBrain.nii.gz"; then
     antsBrainExtraction.sh -d 3 -a ./n4bc/mag_n4bc_$id.nii.gz \
    -e ~/atlas/MICCAI2012-Multi-Atlas-Challenge-Data/T_template0.nii.gz \
    -m ~/atlas/MICCAI2012-Multi-Atlas-Challenge-Data/T_template0_BrainCerebellumProbabilityMask.nii.gz \
    -o ./xtracted/mag_$id -f ~/atlas/MICCAI2012-Multi-Atlas-Challenge-Data/T_template0_BrainCerebellumRegistrationMask.nii.gz 
fi
done < /your/mGRE/magnitude/list
# Then in the ./xtracted directory, run
nohup antsMultivariateTemplateConstruction2.sh -d 3 -o mag -j 64 ./
# Apply atlas transforms to magnitude co-registered QSM
while read -r line; do
    line_dir=${line#*qsm_}
    id_date=${line_dir%.nii.gz}
    id=${id_date%'_'*}
    date=${id_date#*_}
    echo $line_dir $id $date
    mag2ps=$(find -name "antsBTPtemplate0mag_"${id}_${date}"*WarpedToTemplate.nii.gz" -type "f")
    warp=$(find -name "antsBTPmag_"${id}_${date}"*Warp.nii.gz" -type "f" | tail -n 1)
    affine=$(find -name "antsBTPmag_"${id}_${date}"*GenericAffine.mat" -type "f")
    echo "Registering qsm_${id}_${date}.nii.gz to atlas using reference $mag2ps, deformation $warp, and affine transform $affine"
    antsApplyTransforms \
    -i ./qsm_${id}_${date}.nii.gz \
    -r ./$mag2ps \
    -o "./"qsm_${id}_${date}"_2_atlas.nii.gz" \
    -t ./$warp \
    -t [$affine]
    done < /your/QSM/list
# Draw initial labels on QSM atlas, save as `seg.nii.gz`, and transform to patient space
while read -r line; do
    sub_dir=${line%*BrainExtraction*}
    id=${sub_dir#./magmag_}
    echo "Applying transform to $id"
    inv_trans=$(find ./ -name "magmag_$id""BrainExtractionBrain*InverseWarp.nii.gz" -type f)
    antsApplyTransforms \
    -i "seg.nii.gz" \
    -r "mag_"$id"BrainExtractionBrain.nii.gz" \
    -o "./labels/labels2qsm_$id.nii.gz" \
    -t [$line,1] \
    -t $inv_trans \
    -n NearestNeighbor
done < /your/QSM/label/list