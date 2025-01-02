# TensorFlow training with YuniKorn

## Setup
1. You need to set up YuniKorn scheduler on K8s cluster, please refer to [this doc](https://yunikorn.apache.org/docs/#install).
2. Install [training-operator](https://github.com/kubeflow/training-operator) that makes it easy to run distributed 
   or non-distributed ML jobs on K8s. You can install it with the following command.
```   
kubectl apply -k "github.com/kubeflow/training-operator/manifests/overlays/standalone?ref=v1.3.0"
```
3. Build a docker image  with the following command.
```
docker build -f Dockerfile -t kubeflow/tf-resnet-test:1.0 .
```

## Run a TensorFlow job
You need to create a TFjob and configure it to use YuniKorn scheduler.

Limit the job duration and the GPU utilization by editing yaml file
```yaml
spec:
    schedulerName: yunikorn
    activeDeadlineSeconds: 150
    containers:
    - args:
        ...
        - "--num_gpus"
        - "0"
        - "--gpu_memory_limit"
        - "4096"
        - "--epochs"
        - "5"
        ...

```

```
kubectl create -f tf-job-resnet.yaml
```