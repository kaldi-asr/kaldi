# -----------------------------------------------------------------------------
#       Dockerfile to build Kaldi gstreamer server with tue-env
# -----------------------------------------------------------------------------

# Set the base image to Ubuntu with tue-env
FROM tueroboticsamigo/tue-env:master

# File Author / Maintainer
MAINTAINER Arpit Aggarwal

# Update the image and install basic packages
RUN sudo apt-get update -qq && \
    tue-get install kaldi


