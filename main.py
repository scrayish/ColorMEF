"""

Main file
Maybe could try, transfer this stuff to 3D convolutions, so can work with larger batch size than 1

"""


from __future__ import print_function
import argparse
import torch
import numpy as np
import time as prof_time
import os
import json
import logging
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from torchvision.transforms import InterpolationMode
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from tensorboard_utils import CustomSummaryWriter
from circular_pad_panorama import CircularPadPanoramaNormal
from torchmetrics import MeanMetric

from PIL import Image

from image_dataset import ImageDataset
from mefssim import MEF_MSSSIM
from batch_transformers import (
    BatchToTensor,
    BatchRGBToYCbCr,
    BatchTestResolution,
    BatchRandomResolution,
    YCbCrToRGB,
)


def write_run_json(filepath, info):
    with open(filepath, "w") as f:
        json.dump(info, f, indent=4)
        f.close()


def get_padding_function(mode: str):
    if mode == "circular":
        return CircularPadPanoramaNormal
    else:
        return torch.nn.ReflectionPad2d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", type=str, required=True)
    parser.add_argument(
        "-use_cuda", type=lambda x: (str(x).lower() == "true"), default=True
    )
    parser.add_argument("-seed", type=int, default=2022)
    parser.add_argument(
        "-data_type",
        type=str,
        required=True,
        help="What data (still or pan) is used for training",
    )
    parser.add_argument("-train_data", type=str, required=True)
    parser.add_argument("-test_data", type=str, required=True)
    parser.add_argument("-checkpoint_path", type=str, default="./checkpoints/")
    parser.add_argument(
        "-checkpoint", type=str, default=None, help="Name of checkpoint to load"
    )
    parser.add_argument("-output_path", type=str, required=True)
    parser.add_argument("-fuse_expos", nargs="*",
                        default=["dark", "middle", "bright"])
    parser.add_argument(
        "-remove_dark", type=lambda x: (str(x).lower() == "true"), default="false"
    )

    parser.add_argument("-low_size", type=int, default=128)
    parser.add_argument(
        "-high_size",
        type=int,
        default=512,
        help="Apparently, if pass None, will get random res",
    )
    parser.add_argument("-epochs", type=int, default=4)
    parser.add_argument("-learning_rate", type=float, default=1e-4)
    parser.add_argument("-decay_interval", type=int, default=1000)
    parser.add_argument("-decay_ratio", type=float, default=0.1)
    parser.add_argument(
        "-log_scaling", type=lambda x: (str(x).lower() == "true"), default=False
    )
    parser.add_argument("-static_scale_coeff", type=float, default=1.0)
    parser.add_argument("-chroma_to_luma_coeff", type=float, default=1.0)
    parser.add_argument("-resume", type=bool, default=False)
    parser.add_argument(
        "-epochs_per_eval",
        type=int,
        default=10,
        help="How many epochs need to pass to eval",
    )
    parser.add_argument(
        "-epochs_per_save",
        type=int,
        default=20,
        help="How many epochs need to pass to save",
    )
    # Transformer specific parameters (will not interfere with other models)
    parser.add_argument("-dimensions", type=int, default=512)
    parser.add_argument("-heads", type=int, default=8)
    parser.add_argument("-enhance_length", type=int, default=5)
    parser.add_argument("-pad", type=str, default="reflect")
    parser.add_argument("-readout", type=str, default="identity")

    args, _ = parser.parse_known_args()

    # Example model call arguments
    # args, _ = parser.parse_known_args(
    #     args=[
    #         "-model",
    #         "model",
    #         "-data_type",
    #         "run-specific-info-should-go-here",
    #         "-train_data",
    #         "/path/to/train/dataset",
    #         "-test_data",
    #         "/path/to/validation/or/test/dataset",
    #         "-output_path",
    #         "/path/to/result/folder",
    #         "-low_size",
    #         "960",
    #         "-high_size",
    #         "1250",
    #         "-epochs_per_save",
    #         "0",
    #         "-epochs_per_eval",
    #         "1",
    #         "-epochs",
    #         "100",
    #         "-learning_rate",
    #         "1e-4",
    #     ]
    # )

    # Check device
    if torch.cuda.is_available() and args.use_cuda:
        device = "cuda"
        # Set device to 1:
        device = torch.device("cuda:0")
    else:
        print("Using CPU as per config") if not args.use_cuda else print(
            "No CUDA available, using CPU"
        )
        device = "cpu"

    # Create folder and path to save checkpoints to
    run_path = (
        "run-" + datetime.now().strftime(f"%y-%m-%d--%H-%M-%S") +
        f"-{args.data_type}"
    )
    weight_path = f"{args.checkpoint_path}/{run_path}"
    output_path = f"{args.output_path}/{run_path}-results"
    os.makedirs(weight_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    # Define transforms for images
    train_hr_transform = transforms.Compose(
        [
            BatchRandomResolution(
                args.high_size, interpolation=InterpolationMode.BILINEAR
            ),  # Bilinear = 2 (int)
            BatchToTensor(),
            BatchRGBToYCbCr(),
        ]
    )

    train_lr_transform = transforms.Compose(
        [
            BatchRandomResolution(
                args.low_size, interpolation=InterpolationMode.BILINEAR
            ),
            BatchToTensor(),
            BatchRGBToYCbCr(),
        ]
    )

    test_hr_transform = transforms.Compose(
        [
            BatchTestResolution(
                args.high_size, interpolation=InterpolationMode.BILINEAR
            ),
            BatchToTensor(),
            BatchRGBToYCbCr(),
        ]
    )

    test_lr_transform = train_lr_transform

    batch_size = 1
    # Train dataset configuration
    train_data = ImageDataset(
        data_path=args.train_data,
        hr_transform=train_hr_transform,
        lr_transform=train_lr_transform,
        fuse_expos=args.fuse_expos,
    )
    train_loader = DataLoader(
        train_data,
        batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
    )

    # Test dataset configuration
    test_data = ImageDataset(
        data_path=args.test_data,
        # target=str(Path(args.test_data) / "ground_truth"),
        hr_transform=test_hr_transform,
        lr_transform=test_lr_transform,
        fuse_expos=args.fuse_expos,
    )
    test_loader = DataLoader(
        test_data,
        batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
    )

    image_size = tuple(test_data[0]["I_lr"].size()[2:])

    # Model initialization
    Model = getattr(
        __import__("models." + args.model, fromlist=["Model"]), "Model"
    )  # Gets model based on argument given. Pretty cool, can't remember this line ever

    # I guess, we can do some cool memes here with model loading
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
    print(args.model)

    # Metrics for logging results from training
    metric_dict = {
        "train_mean_y": MeanMetric(),
        "train_mean_cb": MeanMetric(),
        "train_mean_cr": MeanMetric(),
        "train_mean_y_raw": MeanMetric(),
        "train_mean_cb_raw": MeanMetric(),
        "train_mean_cr_raw": MeanMetric(),
        "test_mean_y": MeanMetric(),
        "test_mean_cb": MeanMetric(),
        "test_mean_cr": MeanMetric(),
        "test_mean_y_raw": MeanMetric(),
        "test_mean_cb_raw": MeanMetric(),
        "test_mean_cr_raw": MeanMetric(),
    }

    # Loss function initialization
    log_scaling = args.log_scaling
    loss_function = MEF_MSSSIM(
        is_lum=False,
        scale=log_scaling,
        full_structure=False,
        pad_fn=get_padding_function(args.pad),
    ).to(device)
    loss_function_color = MEF_MSSSIM(
        is_lum=False,
        scale=log_scaling,
        full_structure=False,
        pad_fn=get_padding_function(args.pad),
    ).to(device)
    learning_rate = args.learning_rate
    optimizer = optim.Adam(model.parameters(), learning_rate)

    # Other parameters for training and what not
    epochs = args.epochs
    start_epoch = 0
    beta = 0.9
    loss_best = np.Inf
    local_counter = 1
    train_loss = []
    test_result = []
    epochs_per_eval = args.epochs_per_eval
    epochs_per_save = args.epochs_per_save
    coeff_luma = args.static_scale_coeff
    coeff_chroma = coeff_luma * args.chroma_to_luma_coeff

    # Scheduler setup:
    scheduler = StepLR(
        optimizer,
        last_epoch=start_epoch - 1,
        step_size=args.decay_interval,
        gamma=args.decay_ratio,
    )

    # Load model if needed:
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optim_state"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"]

        del state

    # set model after potential weight load to device
    # model = model.to(device)
    # Edit up args and add some stuff needed to run this
    args.loss_function = type(
        loss_function
    ).__name__  # Very useful function actually, I enjoy
    args.optimizer = type(optimizer).__name__
    write_run_json(filepath=weight_path + "/run.json", info=args.__dict__)
    write_run_json(filepath=output_path + "/run.json", info=args.__dict__)

    # Setup summary writer:
    writer = CustomSummaryWriter(logdir=weight_path)

    # Anomaly detection
    torch.autograd.set_detect_anomaly(True)

    # Train/test loop
    for epoch in tqdm(range(start_epoch, epochs), desc="Training network", ncols=100):
        # Reset metrics
        running_loss = 0 if not train_loss else train_loss[-1]
        for metric in metric_dict:
            metric_dict[metric].reset()

        for loader in train_loader, test_loader:
            # Since test phase is so different from train phase, separating them
            if loader is train_loader:
                index = 0
                # Enable model gradient and set training mode:
                model.train()
                torch.set_grad_enabled(True)

                for sample in tqdm(
                    loader, desc="Going through train loader", ncols=100
                ):
                    i_hr, i_lr = (
                        sample["I_hr"],
                        sample["I_lr"],
                    )

                    i_hr = torch.squeeze(i_hr, dim=0).to(device)
                    i_lr = torch.squeeze(i_lr, dim=0).to(device)

                    Y_hr, Cb_hr, Cr_hr = torch.tensor_split(i_hr, 3, dim=1)
                    Y_lr, Cb_lr, Cr_lr = torch.tensor_split(i_lr, 3, dim=1)

                    # Loads more shit to get where I need to. And piss off. Cunt:
                    O_hr, O_hr_cb, O_hr_cr, w_hr, w_hr_cb, w_hr_cr = model(
                        Y_lr, Y_hr, Cb_lr, Cb_hr, Cr_lr, Cr_hr
                    )

                    # Test if weight maps sum to 1:
                    summed_weight_map = torch.sum(
                        w_hr.to("cpu").detach(), dim=0)
                    if (
                        torch.abs(
                            torch.max(summed_weight_map) -
                            torch.min(summed_weight_map)
                        )
                        > 1e-4
                    ):
                        print(
                            "Weight maps don't quite sum to singular value, what the fuck is this..."
                        )

                    # Individual loss for each component:
                    loss_Y = (1 - loss_function(O_hr, Y_hr)) * coeff_luma
                    loss_Cb = (
                        1 - loss_function_color(O_hr_cb, Cb_hr)) * coeff_chroma
                    loss_Cr = (
                        1 - loss_function_color(O_hr_cr, Cr_hr)) * coeff_chroma

                    # Colors fuse first
                    loss_color = loss_Cb + loss_Cr
                    # Then together with luminance
                    loss_total = loss_Y + loss_color
                    q = (
                        loss_Y.to("cpu").item()
                        # + loss_halo.to("cpu").item()
                        + loss_Cb.to("cpu").item()
                        + loss_Cr.to("cpu").item()
                    )

                    # Do metrics here:
                    metric_dict["train_mean_y"].update(loss_Y.to("cpu").item())
                    metric_dict["train_mean_cb"].update(
                        loss_Cb.to("cpu").item())
                    metric_dict["train_mean_cr"].update(
                        loss_Cr.to("cpu").item())
                    metric_dict["train_mean_y_raw"].update(
                        loss_Y.to("cpu").item() / coeff_luma
                    )
                    metric_dict["train_mean_cb_raw"].update(
                        loss_Cb.to("cpu").item() / coeff_chroma
                    )
                    metric_dict["train_mean_cr_raw"].update(
                        loss_Cr.to("cpu").item() / coeff_chroma
                    )

                    running_loss = beta * running_loss + (1 - beta) * q
                    loss_corrected = running_loss / (1 - beta**local_counter)

                    loss_total.backward()
                    optimizer.step()
                    model.zero_grad()

                    local_counter += 1
                    index += 1

                train_loss.append(loss_corrected)

            # Since test phase is much different from train phase, it needs to be done separately
            else:
                # Disable gradient and set model to evaluation
                model.eval()
                torch.set_grad_enabled(False)

                scores = []

                for sample_idx, sample in tqdm(
                    enumerate(loader), desc="Going through test loader"
                ):
                    i_hr, i_lr = (
                        sample["I_hr"],
                        sample["I_lr"],
                        # sample["I_target"],
                    )
                    i_hr = torch.squeeze(i_hr, dim=0).to(device)
                    i_lr = torch.squeeze(i_lr, dim=0).to(device)

                    Y_hr, Cb_hr, Cr_hr = torch.tensor_split(i_hr, 3, dim=1)
                    Y_lr, Cb_lr, Cr_lr = torch.tensor_split(i_lr, 3, dim=1)

                    # Color fuse input
                    O_hr, O_Cb_hr, O_Cr_hr, W_hr, W_Cb_hr, W_Cr_hr = model(
                        Y_lr, Y_hr, Cb_lr, Cb_hr, Cr_lr, Cr_hr
                    )

                    loss = (
                        1 - loss_function.forward(O_hr, Y_hr).to("cpu").item()
                    ) * coeff_luma
                    loss_Cb = (
                        1 -
                        loss_function_color.forward(
                            O_Cb_hr, Cb_hr).to("cpu").item()
                    ) * coeff_chroma
                    loss_Cr = (
                        1 -
                        loss_function_color.forward(
                            O_Cr_hr, Cr_hr).to("cpu").item()
                    ) * coeff_chroma
                    q = loss + loss_Cb + loss_Cr

                    metric_dict["test_mean_y"].update(loss)
                    metric_dict["test_mean_cb"].update(loss_Cb)
                    metric_dict["test_mean_cr"].update(loss_Cr)
                    metric_dict["test_mean_y_raw"].update(loss / coeff_luma)
                    metric_dict["test_mean_cb_raw"].update(
                        loss_Cb / coeff_chroma)
                    metric_dict["test_mean_cr_raw"].update(
                        loss_Cr / coeff_chroma)

                    scores.append(q)
                    # I am updating this part - metric calculation is happening every epoch. Saving images is what depends on epochs
                    # If you want to save on every epoch, epochs_per_eval has to be 1 (for clarity sake)
                    # If epochs_per_eval = 1, then we save the last image instead
                    if (
                        epoch % epochs_per_eval == 0 or epoch == epochs - 1
                    ) and epochs_per_eval != 0:
                        O_hr_RGB = YCbCrToRGB()(
                            torch.cat(
                                [O_hr.to("cpu"), O_Cb_hr.to(
                                    "cpu"), O_Cr_hr.to("cpu")],
                                dim=1,
                            )
                        )
                        O_hr_RGB = torch.clamp(O_hr_RGB, min=0.0, max=1.0)

                        if epochs_per_eval != 1:
                            num = epoch
                        else:
                            num = 999

                        # Enhanced saving procedure:
                        # Saving each example weights and images in separate maps for better storing
                        # and navigation later on:
                        output_test_images = output_path + \
                            f"/{sample_idx:05}/images"
                        output_weight_maps = output_path + \
                            f"/{sample_idx:05}/weights"
                        output_weight_cb_maps = (
                            output_path + f"/{sample_idx:05}/weights_Cb"
                        )
                        output_weight_cr_maps = (
                            output_path + f"/{sample_idx:05}/weights_Cr"
                        )
                        if (
                            not Path(output_test_images).is_dir()
                            and not Path(output_weight_maps).is_dir()
                        ):
                            os.makedirs(
                                output_path + f"/{sample_idx:05}/images", exist_ok=True
                            )
                            os.makedirs(
                                output_path + f"/{sample_idx:05}/weights", exist_ok=True
                            )
                            os.makedirs(
                                str(
                                    Path(output_path)
                                    / f"{sample_idx:05}"
                                    / "weights_Cb"
                                ),
                                exist_ok=True,
                            )
                            os.makedirs(
                                str(
                                    Path(output_path)
                                    / f"{sample_idx:05}"
                                    / "weights_Cr"
                                ),
                                exist_ok=True,
                            )

                        utils.save_image(
                            O_hr_RGB, output_test_images +
                            f"/image_e_{num:03}.jpg"
                        )
                        utils.save_image(
                            W_hr, output_weight_maps +
                            f"/weight_e_{num:03}.jpg"
                        )
                        # Save grayscale images for chromatic color maps, so I can see how extreme they are:
                        utils.save_image(
                            W_Cb_hr, output_weight_cb_maps +
                            f"/weight_e_{num:03}.jpg"
                        )
                        utils.save_image(
                            W_Cr_hr, output_weight_cr_maps +
                            f"/weight_e_{num:03}.jpg"
                        )

                # Append test result to test result
                test_result.append(np.mean(np.asarray(scores)))

            pass

        # Model state saving (also save last model weights)
        if epochs_per_save != 0:
            if epoch % epochs_per_save == 0 or epoch == epochs - 1:
                state = {
                    "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                }
                torch.save(state, f"{weight_path}/state_epoch_{epoch:03}.tar")
        else:
            state = {
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
            }
            torch.save(state, f"{weight_path}/state_epoch_last.tar")

        # Save the best model weights based on luma loss (raw luma, not scaled)
        if loss_best > loss / coeff_luma:
            state = {
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
            }
            torch.save(state, f"{weight_path}/state_epoch_best.tar")
            loss_best = loss / coeff_luma

        # Prepare dictionary to quickly put into metric dictionary
        metric_dict_to_tb = {}
        for metric in metric_dict:
            metric_dict_to_tb[metric] = metric_dict[metric].compute()

        # Update tensorboard writer with new information
        writer.add_hparams(
            hparam_dict=args.__dict__,
            metric_dict={
                "train_loss": train_loss[-1],
                "test_loss": test_result[-1] if test_result else 0,
                **metric_dict_to_tb,
            },
            name=run_path,
            global_step=epoch + 1,
        )

        # Step scheduler
        scheduler.step()

        # Update run log json
        json_info = {
            **args.__dict__,
            "train_loss": train_loss[-1],
            "test_loss": test_result[-1] if test_result else 0,
            "epoch": epoch + 1,
        }
        write_run_json(filepath=weight_path + "/run.json", info=json_info)
        write_run_json(filepath=output_path + "/run.json", info=json_info)

        # flush writer
        writer.flush()

    print("Done")


if __name__ == "__main__":
    main()
