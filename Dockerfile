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

COPY . /kaldi

RUN cd /kaldi && \
    cd /kaldi/tools && \
    ./extras/install_mkl.sh && \
    make -j $(nproc) && \
    cd /kaldi/src && \
    ./configure --shared --use-cuda=no && \
    make depend -j $(nproc) && \
    make -j $(nproc) && \
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
       -exec rm -rf {} \; && \
    find /kaldi -type f -name "*.cc" -o -name "*.o" -delete && \
    find /kaldi -type f -name "*train*" -delete && \
    rm -f /kaldi/src/online2bin/online2-tcp-nnet3-decode-faster && \
    rm -f /kaldi/src/online2bin/online2-wav-nnet3-wake-word-decoder-faster && \
    rm -f /kaldi/src/online2bin/online2-wav-nnet3-latgen-grammar && \
    rm -f /kaldi/src/online2bin/online2-wav-nnet3-latgen-incremental && \
    rm -f /kaldi/src/online2bin/online2-wav-nnet3-latgen-grammar && \
    rm -f /kaldi/src/online2bin/online2-wav-nnet2-latgen-threaded && \
    rm -f /kaldi/src/online2bin/online2-wav-gmm-latgen-faster && \
    rm -rf /kaldi/.git && \
    rm -rf kaldi/tools/srilm

RUN rm -f /kaldi/utils && ln -s /kaldi/egs/wsj/s5/utils /kaldi/utils && \
    rm -f /kaldi/steps && ln -s /kaldi/egs/wsj/s5/steps/ /kaldi/steps

FROM python:3.8-buster

COPY --from=0 /kaldi/ /kaldi/
COPY --from=0 /opt/intel/compilers_and_libraries_2020.0.166/linux/mkl/lib/intel64_lin/ \
              /opt/intel/compilers_and_libraries_2020.0.166/linux/mkl/lib/intel64_lin/
