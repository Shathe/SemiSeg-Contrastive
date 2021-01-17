FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel
RUN apt-get update
RUN apt-get  -y install python3-pip
RUN pip3 install --upgrade pip
RUN apt-get install -y libsm6 libxext6 libxrender-dev libgl1-mesa-glx ffmpeg
COPY requirements.txt /tmp/
RUN pip3 install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install -r /tmp/requirements.txt

