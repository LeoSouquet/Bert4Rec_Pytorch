FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=en_US.UTF-8


RUN apt-get update -y \
    && apt-get -y upgrade \
    && apt-get -y install git


RUN git clone https://github.com/LeoCyclope/Bert4Rec_Pytorch.git \
    && cd Bert4Rec_Pytorch \
    && python -m pip install -r requirements.txt