RAW_DIR="../data/raw"
CLEANED_DIR="../data/cleaned"
AUGMENTED_DIR="../data/augmented"
PREPROCESSED_DIR="../data/preprocessed"

# clean training dataset
python cleaning/launch.py --imgs_dir ${RAW_DIR}/train/images --bbs_dir ${RAW_DIR}/train/labels --out_dir ${CLEANED_DIR}/train
echo "Finished cleaning training dataset"

# clean validation dataset
python cleaning/launch.py --imgs_dir ${RAW_DIR}/val/images --bbs_dir ${RAW_DIR}/val/labels --out_dir ${CLEANED_DIR}/val
echo "Finished cleaning validation dataset"

# augment training dataset
python augmentation/launch.py --config augmentation/configs/default.json --imgs_dir ${CLEANED_DIR}/train/images --bbs_dir ${CLEANED_DIR}/train/labels --out_dir ${AUGMENTED_DIR}/train
echo "Finished augmenting training dataset"

# combine cleaned and augmented as final datasets
mkdir -p ${PREPROCESSED_DIR}/train
mv ${CLEANED_DIR}/train/images ${PREPROCESSED_DIR}/train
mv ${CLEANED_DIR}/train/labels ${PREPROCESSED_DIR}/train
mv ${AUGMENTED_DIR}/train/images/* ${PREPROCESSED_DIR}/train/images
mv ${AUGMENTED_DIR}/train/labels/* ${PREPROCESSED_DIR}/train/labels
echo "Finished consolidating training dataset"

mkdir -p ${PREPROCESSED_DIR}/val
mv ${CLEANED_DIR}/val/images ${PREPROCESSED_DIR}/val
mv ${CLEANED_DIR}/val/labels ${PREPROCESSED_DIR}/val
echo "Finished consolidating validation dataset"
