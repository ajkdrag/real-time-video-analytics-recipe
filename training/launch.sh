DATA="../data/data.yaml"
HYP="hyps/finetune_1.yaml"
OUT_DIR="../artifacts"
OUT_FOLDER="latest"
NOW=$(date +%s)

IMG_SZ=640
BATCH_SZ=2
EPOCHS=50
NET="yolov5s.pt"

cd yolov5_torch
python train.py --epochs ${EPOCHS} \
    --data ${DATA} \
    --img-size ${IMG_SZ} \
    --batch-size ${BATCH_SZ} \
    --weights ${NET} \
    --hyp ${HYP} \
    --project ${OUT_DIR}/${OUT_FOLDER} \
    --name ${NOW} \
    --exist-ok \
    --workers 1

python export.py --weights ${OUT_DIR}/${OUT_FOLDER}/${NOW}/weights/best.pt --img ${IMG_SZ} --batch 1 --include torchscript
