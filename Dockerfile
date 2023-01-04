FROM nvidia/cuda:10.2-devel-ubuntu18.04

LABEL maintainer = "Dias Bakhtiyarov dbakhtiyarov@nu.edu.kz"

RUN apt-get update -yqq && apt-get install -y \
    git wget curl build-essential

RUN git clone --depth=1 https://github.com/espnet/espnet
RUN cd /espnet/tools &&\
    ./setup_anaconda.sh venv espnet 3.9

RUN cd /espnet/tools &&\
    make TH_VERSION=1.10.1

ENV PATH="/espnet/tools/venv/bin:$PATH"
RUN echo "source activate espnet" > ~/.bashrc
# RUN rm /bin/sh && ln -s /bin/bash /bin/sh
SHELL ["/bin/bash", "-c"]
RUN source activate espnet &&\
    pip install parallel_wavegan==0.5.5 flask gunicorn jiwer

RUN apt-get update -yqq && apt-get install -y \
    p7zip-full

ARG BASE_URL
ARG MODEL_FILENAME
ARG VOCODER_NAME

RUN cd /espnet &&\
    wget ${BASE_URL}/${MODEL_FILENAME} -O tts_model.zip &&\
    7z x tts_model.zip &&\
    wget ${BASE_URL}/${VOCODER_NAME}.zip -O tts_vocoder.zip &&\
    7z x tts_vocoder.zip

ENV MODEL_PATH="/espnet/exp/tts_train_raw_char" \
    VOCODER_PATH="/espnet/${VOCODER_NAME}" \
    BASE_URL=$BASE_URL \
    MODEL_FILENAME=$MODEL_FILENAME \
    VOCODER_NAME=$VOCODER_NAME

COPY ./app /app
WORKDIR /app

COPY entrypoint.sh ./
ENTRYPOINT ["./entrypoint.sh"]
