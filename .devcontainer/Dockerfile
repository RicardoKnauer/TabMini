FROM mcr.microsoft.com/devcontainers/anaconda:0-3

RUN apt-get update
RUN apt-get install -y pkg-config build-essential libopenblas-dev julia

RUN conda init &&\
       conda update --name base conda &&\
       conda create -n tabmini python=3.10.13 -y &&\
       eval "$(conda shell.bash hook)" &&\
       conda activate tabmini

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN conda run --name tabmini