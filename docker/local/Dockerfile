FROM ubuntu:16.04

MAINTAINER k188583@ie.u-ryukyu.ac.jp

USER root

ARG U_NAME=k188583

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget \
        git \
        vim \
        openssl \
        ca-certificates \
        sudo \
        bzip2 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        && \
    apt-get clean

RUN useradd $U_NAME -m && \
    echo "%$U_NAME ALL=(ALL) NOPASSWD: ALL" >>/etc/sudoers.d/$U_NAME

COPY ["jupyter_notebook_config.py", "jl", "tb", "/"]
RUN mkdir /home/$U_NAME/.jupyter && \
    mv /jupyter_notebook_config.py /home/$U_NAME/.jupyter/ && \
    chown -R $U_NAME:$U_NAME /home/$U_NAME/.jupyter && \
    mv /jl /usr/local/bin/ && \
    chmod 755 /usr/local/bin/jl && \
    mv /tb /usr/local/bin/ && \
    chmod 755 /usr/local/bin/tb

USER $U_NAME
ENV HOME /home/$U_NAME
WORKDIR $HOME

RUN mkdir .pyenv && git clone git://github.com/yyuu/pyenv.git .pyenv
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/versions/anaconda3-5.0.0/bin:$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

RUN  pyenv install anaconda3-5.0.0 && \
     pyenv global anaconda3-5.0.0 && \
     pyenv rehash && \
     pip install -U pip && \
     conda update -y conda && \
     pip install \
        jupyterlab \
        tensorflow \
        Keras \
        matplotlib \
        numpy \
        scikit-image \
        scikit-learn \
        opencv-python \
        imageio \
        pygame \
    && \
    git clone https://github.com/openai/gym.git && \
    cd gym && \
    pip install -e . && \
    cd .. && \
    git clone https://github.com/ntasfi/PyGame-Learning-Environment.git ./ple && \
    cd ple && \
    pip install -e . && \
    cd .. && \
    git clone https://github.com/lusob/gym-ple.git && \
    cd gym-ple && \
    pip install -e . && \
    cd ..

EXPOSE 8888
EXPOSE 6006
