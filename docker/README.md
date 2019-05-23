# Kaldi Docker Images

Sample usage of the CPU based images:
```
docker run -it mdoulaty/kaldi:latest bash
``` 

Sample usage of the GPU based images:

Note: use [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) to run the GPU images.

```
docker run -it --runtime=nvidia mdoulaty/kaldi:gpu-latest bash
```
