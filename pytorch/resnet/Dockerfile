FROM vault.habana.ai/gaudi-docker/1.18.0/ubuntu22.04/habanalabs/pytorch-installer-2.4.0:latest
ENV HABANA_VISIBLE_DEVICES=all

ADD . /var/resnet
WORKDIR /var/resnet
RUN python3 -m pip install optimum[habana]

CMD ["python3", "gaudi_spawn.py", "--world_size", "1", "--use_mpi", "resnet101.py", "--num_hpus", "1"]

