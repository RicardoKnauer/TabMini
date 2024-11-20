FROM continuumio/miniconda3:main

ARG METHOD
ARG OUTPUT_PATH
ARG TIME_LIMIT

RUN apt update && apt install -y pkg-config build-essential libopenblas-dev julia

RUN conda init &&\
    conda update --name base conda &&\
    conda create -n tabmini python=3.10.13 -y &&\
    eval "$(conda shell.bash hook)" &&\
    conda activate tabmini

COPY requirements/requirements_${METHOD}.txt requirements.txt
RUN pip install -r requirements.txt

ENV APP_HOME=/app
WORKDIR $APP_HOME
COPY . $APP_HOME

ENTRYPOINT ["python", "example.py"]