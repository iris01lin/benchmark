FROM tensorflow/tensorflow:latest-gpu

ADD . /var/resnet
WORKDIR /var/resnet

RUN pip install --upgrade pip
RUN pip install tensorflow-datasets

ENTRYPOINT ["python", "resnet101.py"]
