FROM continuumio/miniconda3:main

ARG METHOD
ARG OUTPUT_PATH
ARG TIME_LIMIT

RUN apt update && apt install -y pkg-config build-essential libopenblas-dev

RUN conda init &&\
    conda update --name base conda &&\
    conda create -n tabmini python=3.10.13 -y &&\
    eval "$(conda shell.bash hook)" &&\
    conda activate tabmini

COPY requirements_${METHOD}.txt requirements.txt
RUN pip install -r requirements.txt

ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . $APP_HOME

ENTRYPOINT ["bash", "run_in_environment.sh", "${METHOD}", "${OUTPUT_PATH}", "${TIME_LIMIT}"]