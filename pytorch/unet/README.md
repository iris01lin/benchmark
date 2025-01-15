# Train Pytorch Unet

## Setup
1. Please follow the instructions provided in the [Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html) to set up the environment.

2. Build a docker image  with the following command. (For gaudi 1.18.0)
```
docker build -f Dockerfile -t pytorch-unet-hpu .
```


## Training Example
**Run Training on 1 HPU**
```
docker run -it --runtime=habana --ipc=host --rm --name unet-hpu pytorch-unet-hpu --hpus 1
```

**Run Training on 8 HPU**
```
docker run -it --runtime=habana --ipc=host --rm --name unet-hpu pytorch-unet-hpu --hpus 8
```

**Run Training on CPU**
```
docker run -it --ipc=host --rm --name unet-cpu pytorch-unet-hpu --cpus 1 --device cpu --framework pytorch-lightning
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

Limit the job duration, min/max epochs, train/val batch size, the HPU numbers and runtime by editing yaml file
```yaml
spec:
    runtimeClassName: habana
    hostIPC: true
    schedulerName: yunikorn
    activeDeadlineSeconds: 300
    containers:
    - args:
        ...
        - --hpus=1
        - --batch_size=64
        - --val_batch_size=64
        - --min_epochs=3
        - --max_epochs=10
        ...

```
Create the PyTorchJob
```
kubectl create -f pytorch-job-unet.yaml
```