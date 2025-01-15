Train Pytorch Resnet101 model

## Setup
1. Please follow the instructions provided in the [Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html) to set up the environment.

2. Build a docker image  with the following command. (For gaudi 1.18.0)
```
docker build -f Dockerfile -t pytorch-resnet-hpu .
```


## Training Example
**Run Training on 1 HPU**
```
docker run -it --runtime=habana --ipc=host --rm --name resnet-hpu pytorch-resnet-hpu
```
The default dataset is MNIST, if want to use food101 dataset
```
docker run -it --runtime=habana --ipc=host --rm --name resnet-hpu pytorch-resnet-hpu python3 resnet101.py --num_hpus 1 --dataset food101
```

**Run Training on 8 HPU**
```
# both world_size and num_hpus need to update
docker run -it --runtime=habana --ipc=host --rm --name pytorch-resnet-hpu pytorch-resnet-hpu python3 gaudi_spawn.py --world_size 8 --use_mpi resnet101.py --num_hpus 8 --dataset food101
```
or
```
docker run -it --runtime=habana --ipc=host --rm --name pytorch-resnet-hpu pytorch-resnet-hpu mpirun -n 8 --bind-to core --map-by slot:PE=7 --rank-by core --report-bindings --allow-run-as-root python3 resnet101.py --num_hpus 8 --dataset food101
```

**Run Training on CPU**
```
docker run -it --ipc=host --rm --name pytorch-resnet-cpu pytorch-resnet-hpu python3 resnet101.py --device cpu 
```

## Run a Pytorch job by Yunikorn
## Setup
1. You need to set up YuniKorn scheduler on K8s cluster, please refer to [this doc](https://yunikorn.apache.org/docs/#install).
2. Install [training-operator](https://github.com/kubeflow/training-operator) that makes it easy to run distributed 
   or non-distributed ML jobs on K8s. You can install it with the following command.
```   
kubectl apply -k "github.com/kubeflow/training-operator/manifests/overlays/standalone?ref=v1.3.0"
helm repo add yunikorn https://apache.github.io/yunikorn-release
helm install yunikorn yunikorn/yunikorn --namespace yunikorn --create-namespace --version 1.3.0
```
You need to create a PyTorchJob and configure it to use YuniKorn scheduler.

Limit the job duration, epochs, batch size and runtime by editing yaml file
```yaml
spec:
    runtimeClassName: habana
    hostIPC: true
    schedulerName: yunikorn
    activeDeadlineSeconds: 300
    containers:
    - args:
        ...
        - python3 
        - gaudi_spawn.py 
        - --world_size=8 
        - --use_mpi
        - basic.py 
        - --num_hpus=8
        - --device=hpu
        - --batch_size=64
        - --epochs=5
        ...

```

```yaml
#Using CPU
spec:
    schedulerName: yunikorn
    activeDeadlineSeconds: 300
    containers:
    - args:
        ...
        - python3
        - basic.py
        - --device=cpu
        - --batch_size=64
        - --epochs=5
        ...

```

Create the PyTorchJob
```
kubectl create -f pytorch-job-resnet.yaml
```