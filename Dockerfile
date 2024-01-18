FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime

SHELL ["/bin/bash", "-c"]

ARG USERNAME=exposure-fusion
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install git libgl1 libglib2.0-0

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Create directories where to mount stuff
ARG work_dir=/workspaces/exposure-fusion/
ENV WORK_DIR=$work_dir

RUN mkdir -p $work_dir

# Activate user?
USER USERNAME

WORKDIR /workspaces/exposure-fusion/
