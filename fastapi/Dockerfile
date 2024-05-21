FROM python:3.8-slim  # Use a lightweight Python base image

ENV DEBIAN_FRONTEND=noninteractive  

RUN apt-get update && apt-get install -y \
    python3-dev build-essential wget git  # Essential development tools

ARG USER_ID=1000
RUN useradd -m -s /bin/bash --no-log-init --system --uid $USER_ID appuser
RUN echo "appuser:your_strong_password" | chpasswd <<< /dev/null  
ENV PATH="/home/appuser/.local/bin:$PATH"  
USER appuser
WORKDIR /home/appuser

RUN python -m ensurepip --upgrade

RUN pip install torch==1.10 torchvision==0.11.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html \
                tensorboard cmake cached-property 'git+https://github.com/facebookresearch/fvcore'

RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo || true  # Suppress non-fatal errors

RUN pip install --user -e detectron2_repo

RUN mkdir /fastapi  # Consider a user-specific directory

COPY requirements.txt /fastapi

WORKDIR /fastapi  # Use the intended working directory where your app resides
RUN pip install --user -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
