FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel

# Trick to avoid conflict GPG error when installing software via apt
# https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64
RUN rm /etc/apt/sources.list.d/cuda.list

# Install tools required for our workspace
RUN apt update
RUN apt install -y task-spooler

# Install Python requirements
COPY requirements.txt ./requirements.txt
RUN python3 -m pip install -U --force-reinstall pip
RUN python3 -m pip install -r requirements.txt --default-timeout=9999

# Set working path
WORKDIR "/mnt"
