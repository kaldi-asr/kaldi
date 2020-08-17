FROM python:3.8-buster

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        g++ \
        make \
        automake \
        autoconf \
        bzip2 \
        unzip \
        wget \
        sox \
        libtool \
        git \
        subversion \
        python2.7 \
        python3 \
        zlib1g-dev \
        gfortran \
        ca-certificates \
        patch \
        ffmpeg \
        vim \
        bc && \
    rm -rf /var/lib/apt/lists/*


RUN git clone --depth 1 https://github.com/parrot-com/kaldi.git /kaldi && \
    cd /kaldi && \
    cd /kaldi/tools && \
    ./extras/install_mkl.sh && \
    make -j $(nproc) && \
    cd /kaldi/src && \
    ./configure --shared --use-cuda=no && \
    make depend -j $(nproc) && \
    make -j $(nproc) && \
    ls /kaldi/src && \
    find /kaldi/src/* -depth -type d \
       -not -name gmm \
       -not -name transform \
       -not -name feat \
       -not -name featbin \
       -not -name nnet2 \
       -not -name ivector \
       -not -name base \
       -not -name nnet3 \
       -not -name fstext \
       -not -name lat \
       -not -name latbin \
       -not -name online2 \
       -not -name online2bin \
       -not -name rnnlm \
       -not -name nnet3bin \
       -not -name rnnlmbin \
       -not -name lib \
       -not -name chain \
       -not -name cudamatrix \
       -not -name decoder \
       -not -name lm \
       -not -name hmm \
       -not -name tree \
       -not -name util \
       -not -name matrix \
       -exec rm -rf {} \;

RUN rm -f /kaldi/utils && ln -s /kaldi/egs/wsj/s5/utils /kaldi && \
    rm -f /kaldi/steps && ln -s /kaldi/egs/wsj/s5/steps/ /kaldi
