FROM tensorflow/tensorflow:latest-gpu

ADD . /var/resnet

# Install necessary Python packages
RUN pip install --upgrade pip
RUN pip install tensorflow-datasets

ENTRYPOINT ["python", "/var/resnet/resnet101.py"]
