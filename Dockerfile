FROM nvidia/cuda:11.5.2-devel-ubuntu18.04

LABEL maintainer = "Dias Bakhtiyarov dbakhtiyarov@nu.edu.kz"

RUN apt-get update -yqq && apt-get upgrade -y

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/espnet/tools/anaconda/bin:$PATH


RUN apt-get update -yqq && apt-get install -y \
    software-properties-common\
    curl wget fastjar tree git cmake \
    g++ &&\
    git clone https://github.com/espnet/espnet &&\
    cd espnet/tools &&\
    ./setup_anaconda.sh anaconda espnet 3.9

SHELL ["/bin/bash", "--login", "-c"]
WORKDIR /espnet
COPY entrypoint.sh ./

ENTRYPOINT /espnet/entrypoint.sh $0 $@
RUN echo ". /espnet/tools/anaconda/etc/profile.d/conda.sh" >> ~/.profile &&\
    conda init bash &&\
    echo "conda activate espnet" >> ~/.profile &&\
    echo "conda activate espnet" >> ~/.bashrc &&\
    chmod +x /espnet/entrypoint.sh

RUN conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch &&\
    cd /espnet/tools && make
RUN pip install parallel_wavegan flask
CMD ["python3", "app.py"]

ARG BASE_URL
ARG MODEL_FILENAME
ARG VOCODER_NAME

RUN cd /espnet &&\
    wget ${BASE_URL}/${MODEL_FILENAME} -O tts_model.zip &&\
    jar xvf tts_model.zip &&\
    wget ${BASE_URL}/${VOCODER_NAME}.zip -O tts_vocoder.zip &&\
    jar xvf tts_vocoder.zip

ENV MODEL_PATH="/espnet/exp/tts_train_raw_char" \
    VOCODER_PATH="/espnet/${VOCODER_NAME}" \
    BASE_URL=$BASE_URL \
    MODEL_FILENAME=$MODEL_FILENAME \
    VOCODER_NAME=$VOCODER_NAME

COPY app.py .