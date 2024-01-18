"""

Multi-example inference, which is not scuffed. (Because that's allegedly bad... to be scuffed. Piss off...)
Assuming data structure then! You'll have to adjust, if you want it to work. Otherwise, fuck off :)

"""


from __future__ import print_function
import argparse
import torch
import os
import shutil
import json
from datetime import datetime
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from torchvision import transforms, utils
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader
from circular_pad_panorama import CircularPadPanoramaNormal

from image_dataset import ImageDataset
from batch_transformers import (
    BatchToTensor,
    BatchRGBToYCbCr,
    BatchTestResolution,
    YCbCrToRGB,
)


# Better save than utils save, as it uses maximum quality possible for JPEG saving:
def save_image_better(tensor, fp, format=None):
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        utils._log_api_usage_once(save_image_better)
    grid = utils.make_grid(tensor)
    ndarr = (
        grid.mul(255)
        .add_(0.5)
        .clamp_(0, 255)
        .permute(1, 2, 0)
        .to("cpu", torch.uint8)
        .numpy()
    )
    im = Image.fromarray(ndarr)
    im.save(fp, format=format, quality=100)


def get_padding_function(mode: str):
    if mode == "circular":
        return CircularPadPanoramaNormal
    else:
        return torch.nn.ReflectionPad2d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--model-folder", type=str, required=True)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Specify output, else dropping next to data",
    )

    args, _ = parser.parse_known_args()

    # args, _ = parser.parse_known_args(
    #     [
    #         "--data-path",
    #         "/mnt/machine_learning/datasets/exposure_fusion/gcam_test_set",
    #         "--model-folder",
    #         "/home/worker/Documents/Matiss/exposure-fusion/checkpoints/run-23-02-01--11-43-47-PAN_G_GT",
    #         "--output",
    #         "/mnt/machine_learning/datasets/exposure_fusion/inference_results/",
    #     ]
    # )

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        print("No CUDA available. Using CPU")

    # Read run.json and get details:
    with open(str(Path(args.model_folder) / "run.json"), "r") as f:
        data = json.load(f)
        f.close()

    # Apply data to args namespace for simplicity sake:
    args.__dict__ = {**args.__dict__, **data}

    hr_transform = transforms.Compose(
        [
            BatchTestResolution(
                10000, interpolation=InterpolationMode.BILINEAR),
            BatchToTensor(),
            BatchRGBToYCbCr(),
        ]
    )
    lr_transform = transforms.Compose(
        [
            BatchTestResolution(
                data["low_size"], interpolation=InterpolationMode.BILINEAR
            ),
            BatchToTensor(),
            BatchRGBToYCbCr(),
        ]
    )

    fuse_expos = data["fuse_expos"]

    # Data setup:
    dataset = ImageDataset(
        data_path=args.data_path,
        hr_transform=hr_transform,
        lr_transform=lr_transform,
        fuse_expos=fuse_expos,
        inference=True,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    image_size = tuple(dataset[0][1]["I_lr"].size()[2:])

    # Initialize model:
    Model = getattr(__import__(
        "models." + data["model"], fromlist=["Model"]), "Model")
    if "transformer" in args.model:
        pad_fn = get_padding_function(args.pad)
        model = Model(
            image_size,
            args.dimensions,
            args.heads,
            int(args.dimensions * 4),
            1,
            pad_fn,
            args.enhance_length,
            args.readout,
        ).to(device)
    else:
        model = Model().to(device)

    state = torch.load(
        str(Path(args.model_folder) / "state_epoch_087.tar"), map_location=device
    )
    model.load_state_dict(state["model_state"])

    del state

    # Setup output path for this process:
    if args.output:
        print("Output passed, using legacy save format")
        path_output_general = str(
            Path(args.output)
            / "{}-results".format(datetime.now().strftime(f"%y-%m-%d--%H-%M-%S"))
        )
        path_result_images = str(Path(path_output_general) / "result-images")
        for w in fuse_expos:
            os.makedirs(str(Path(path_output_general) /
                        f"weights-{w}"), exist_ok=True)
        # Copy run json to folder:
        shutil.copy(
            str(Path(args.model_folder) / "run.json"),
            str(Path(path_output_general) / "run.json"),
        )
    else:
        print("No path given, will save images in 'fusion' folder next to data")
        path_result_images = str(Path(args.data_path) / "fusion")
        # Copy run json to folder:
        shutil.copy(
            str(Path(args.model_folder) / "run.json"),
            str(Path(args.data_path) / "run.json"),
        )

    os.makedirs(path_result_images, exist_ok=True)

    # Eco mode:
    model.eval()
    torch.set_grad_enabled(False)

    # Print, where outputs will be located:
    print(f"Outputs for this run will be located in: {path_result_images}")

    # Launch model and pray for forgiveness:
    for name, sample in tqdm(loader, desc="Running inference"):
        i_hr, i_lr = sample["I_hr"], sample["I_lr"]
        i_hr = torch.squeeze(i_hr, dim=0)
        i_lr = torch.squeeze(i_lr, dim=0)

        Y_hr, Cb_hr, Cr_hr = torch.tensor_split(i_hr.to(device), 3, dim=1)
        Y_lr, Cb_lr, Cr_lr = torch.tensor_split(i_lr.to(device), 3, dim=1)

        # Delete extra variables that aren't needed anymore:
        del i_hr, i_lr

        o_hr, o_hr_cb, o_hr_cr, w_hr, w_hr_cb, w_hr_cr = model.forward(
            Y_lr, Y_hr, Cb_lr, Cb_hr, Cr_lr, Cr_hr
        )

        # Save image to paths:
        O_hr_RGB = YCbCrToRGB()(
            torch.cat([o_hr.to("cpu"), o_hr_cb.to(
                "cpu"), o_hr_cr.to("cpu")], dim=1)
        )
        O_hr_RGB = torch.clamp(O_hr_RGB, min=0.0, max=1.0)
        # Fix name:
        name_true = name[0]
        save_image_better(O_hr_RGB, str(Path(path_result_images) / name_true))
        # Save weights as well, if output path was given:
        if args.output:
            for weight, mode in zip(w_hr, fuse_expos):
                save_image_better(
                    weight,
                    str(
                        Path(path_result_images).parent /
                        f"weights-{mode}" / name_true
                    ),
                )


if __name__ == "__main__":
    main()
    print("Done")
