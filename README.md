# ColorMEF

This is an implementation of ColorMEF model from paper "ColorMEF: A Novel Transformer Based Multi-Exposure Fusion Model"

## Details

This specific MEF model fuses all of image information inside the model (luminance AND colors)
Model expects images to be of YCbCr colorspace. If you want to train on RGB, you have to check
what modifications to model are necessary

## Prerequisites

Current implementation has been trained and used using PyTorch >= 2.0

We provide a Dockerfile configuration to build docker container where you can run experiments. Use of VSCode DevContainers is strongly recommended.
If you don't want to use docker, you'll have to provide an environment with installed PyTorch matching the version mentioned above

For additional package requirements, check `requirements.txt` file.

## Training

To train the model, you must configure `train.sh` script with necessary arguments. All arguments can be viewed in `main.py`

If you're training without docker, simply launch `train.sh` in order to launch training

To train inside docker container, you need to build the container first:

```sh
docker compose -f ./docker-compose.yml build
```

And then launch train service

```sh
docker compose up train_service
```

## Inference

For inference, you should launch `inference.py` script. The script assumes, you have a working model checkpoint and a `run.json` from training time, in order to load some additional model parameters. Currently inference can be launched only inside dev-container or locally. If you want to create a docker service, you can edit the `docker-compose.yml` file and add additional service for inference
