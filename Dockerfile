FROM continuumio/miniconda3:23.10.0-1

RUN apt update && apt install -y pkg-config build-essential libopenblas-dev julia

RUN conda init &&\
    conda update --name base conda &&\
    conda create -n tabmini python=3.10.13 -y &&\
    eval "$(conda shell.bash hook)" &&\
    conda activate tabmini

COPY requirements.txt .
RUN pip install -r requirements.txt

ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . $APP_HOME

RUN conda run --name tabmini
ENTRYPOINT ["python", "example.py"]