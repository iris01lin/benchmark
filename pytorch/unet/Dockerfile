FROM vault.habana.ai/gaudi-docker/1.18.0/ubuntu22.04/habanalabs/pytorch-installer-2.4.0:latest

ENV HABANA_VISIBLE_DEVICES=all
ADD . /unet
WORKDIR /unet

RUN pip install --disable-pip-version-check -r requirements.txt
RUN mkdir -p /data /results
RUN python3 download.py --task 01
RUN python3 preprocess.py --task 01 --dim 2

ENTRYPOINT ["python3", "main.py"]