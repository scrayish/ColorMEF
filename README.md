# ColorMEF

This is an implementation of ColorMEF model from paper *Insert paper here*

## Details

This specific MEF model fuses all of image information inside the model (luminance AND colors)
Model expects images to be of YCbCr colorspace. If you want to train on RGB, you have to check
what modifications to model are necessary

---
## Training

For training, you should launch only `train.sh` script and it ought to initialize
as planned.

To change things for training, simply edit the bash script and launch training

---
## Inference

For inference, you should launch `inference.py` script. The script assumes, you have a working
model checkpoint and a `run.json` from training time, in order to load some additional model parameters

---

## MISC

Training was conducted on closed-source dataset. Data is private only and cannot be redistributed. You'll have
to gather your own dataset for training purposes.

Check requirements.txt for package dependencies. Usage of VSCode DevContainers is strongly recommended.
Otherwise, you'll need to provide an alternative environment sporting PyTorch
