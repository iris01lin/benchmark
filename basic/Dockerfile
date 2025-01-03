FROM tensorflow/tensorflow:latest-gpu

ADD . /var/tf_mnist
# Install necessary Python packages
RUN pip install --upgrade pip
RUN pip install tensorflow-datasets

ENTRYPOINT ["python", "/var/tf_mnist/mnist.py"]