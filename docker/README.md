# Kaldi Docker images

Kaldi offers two set of images: CPU-based images and GPU-based images. Daily builds of the latest version of the master branch (both CPU and GPU images) are pushed daily to [DockerHub](https://hub.docker.com/r/kaldiasr/kaldi). 

## Using pre-built images 
Sample usage of the CPU based images:
```bash
docker run -it kaldiasr/kaldi:latest bash
``` 

Sample usage of the GPU based images:

Note: use [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) to run the GPU images.

```bash
docker run -it --runtime=nvidia kaldiasr/kaldi:gpu-latest bash
```

## Building images locally
For building the CPU-based image:
```bash
cd docker/debian9.8-cpu
docker build --tag kaldiasr/kaldi:latest .
```

and for GPU-based image:
```bash
cd docker/ubuntu16.04-gpu
docker build --tag kaldiasr/kaldi:gpu-latest .
```
