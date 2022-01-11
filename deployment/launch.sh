#!/bin/bash

WARMUP_REQUESTS=20
RESOURCES_VOL="/home/resources"
MODEL_PATH=${RESOURCES_VOL}/best.torchscript
HANDLER_PATH=${RESOURCES_VOL}/ts_handler.py
CATEGORY_MAP=${RESOURCES_VOL}/index_to_name.json

torch-model-archiver --model-name yolov5_exp_1 \
    --version 0.1 \
    --serialized-file ${MODEL_PATH} \
    --handler ${HANDLER_PATH} \
    --extra-files ${CATEGORY_MAP},${HANDLER_PATH} \
    --export-path model-store

echo "Exported the model. Starting server ..."

torchserve --start --model-store model-store --models yolov5_exp_1.mar  --ts-config config.properties
sleep 2

for ((i=1; i<=$WARMUP_REQUESTS; i++))
do
    curl http://localhost:8080/predictions/yolov5_exp_1 -F "body=@/home/resources/warmup.jpg"
    echo "warmup num: ${i}"
done

echo "Warmup runs completed."

tail -f /dev/null