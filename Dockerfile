FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel
RUN apt-get update
RUN apt-get  -y install python3-pip
RUN pip3 install --upgrade pip
RUN apt-get install -y libsm6 libxext6 libxrender-dev
COPY requirements.txt /tmp/
RUN pip3 install -r /tmp/requirements.txt
