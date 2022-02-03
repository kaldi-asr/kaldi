# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.209.6/containers/javascript-node/.devcontainer/base.Dockerfile

# [Choice] Node.js version (use -bullseye variants on local arm64/Apple Silicon): 16, 14, 12, 16-bullseye, 14-bullseye, 12-bullseye, 16-buster, 14-buster, 12-buster
ARG VARIANT="16-bullseye"
FROM mcr.microsoft.com/vscode/devcontainers/javascript-node:0-${VARIANT}

# [Optional] Uncomment this section to install additional OS packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>

# [Optional] Uncomment if you want to install an additional version of node using nvm
# ARG EXTRA_NODE_VERSION=10
# RUN su node -c "source /usr/local/share/nvm/nvm.sh && nvm install ${EXTRA_NODE_VERSION}"

# [Optional] Uncomment if you want to install more global node modules
# RUN su node -c "npm install -g <your-package-list-here>"

RUN apt-get -qq update && \
    apt-get install -y -q --no-install-recommends \
    build-essential \
    curl \
    git \
    pkg-config \
    # Libs below are used to install tools
    libssl-dev \
    zlib1g-dev \
    automake \
    autoconf \
    unzip \
    wget \
    # used during mkl installation
    sox \
    gfortran \
    python2.7 \
    # Needed to kill the server on debug mode.
    lsof \
    libzmq3-dev \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# RUN ln -s /usr/bin/python2.7 /usr/bin/python
