FROM nvcr.io/nvidia/pytorch:20.10-py3

COPY requirements.txt /tmp/
RUN pip3 install -r /tmp/requirements.txt

