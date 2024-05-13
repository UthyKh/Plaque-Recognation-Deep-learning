FROM python:3.8-slim

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
    python3-opencv ca-certificates python3-dev git wget sudo build-essential
RUN ln -sv /usr/bin/python3 /usr/bin/python

# create a non-root user
ARG USER_ID=1000
RUN useradd -m --no-log-init --system --uid ${USER_ID} appuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER appuser
WORKDIR /home/appuser


ENV PATH="/home/appuser/.local/bin:${PATH}"

# Upgrade pip
RUN pip install --upgrade pip

# install dependencies
# See https://pytorch.org/ for other options if you use a different version of CUDA
RUN pip install --user tensorboard cmake cached-property
RUN pip install --user torch==1.10 torchvision==0.11.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html

RUN pip install --user 'git+https://github.com/facebookresearch/fvcore'

# install detectron2
RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo

RUN pip install --user -e detectron2_repo

# Set a fixed model cache directory.
USER root
RUN mkdir /fastapi
USER appuser

COPY requirements.txt /fastapi

WORKDIR /fastapi

RUN pip install --user -r requirements.txt

COPY . /home/appuser/fastapi

# Set the correct working directory
WORKDIR /home/appuser/fastapi
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]