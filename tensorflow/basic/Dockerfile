FROM tensorflow/tensorflow:latest-gpu

ADD . /var/tf_mnist
WORKDIR /var/tf_mnist

RUN pip install --upgrade pip
RUN pip install tensorflow-datasets

ENTRYPOINT ["python", "mnist.py"]