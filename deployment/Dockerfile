# Build the image torchserve locally before running this, cf github torchserve:
# https://github.com/pytorch/serve/tree/master/docker
FROM torchserve-dev:1.0
USER root
RUN apt-get update
RUN apt-get install -y libgl1-mesa-glx
RUN apt-get install -y libglib2.0-0
RUN apt-get install -y python3-distutils

RUN pip3 install --upgrade pip
RUN pip install torch-model-archiver
RUN pip install opencv-python
RUN python3 -c "import cv2"

COPY ./launch.sh .
EXPOSE 8080 8081

RUN chmod +x launch.sh
ENTRYPOINT ["sh", "-c", "./launch.sh"]

