# ----------------------------------------------------------
#       Dockerfile to build base working Ubuntu image
# ----------------------------------------------------------

# Set the base image to Ubuntu 16.04
FROM ubuntu:16.04

# File Author / Maintainer
MAINTAINER Arpit Aggarwal

# Make image to be noninteractive to use with CI
# Comment this line when not using this image with CI
ENV DEBIAN_FRONTEND=noninteractive

# Set environment variable USER
ENV USER=hero

# Update the image and install basic packages
RUN apt-get update -qq > /dev/null && apt-get install --assume-yes --no-install-recommends apt-utils ca-certificates curl dbus dialog git sudo wget -qq > /dev/null

# Add user USER with SUDO previleges and disable SUDO password
RUN addgroup -gid 2000 $USER && \
    adduser --disabled-password --gecos "" -uid 2000 -gid 2000 $USER && \
    addgroup $USER adm && \
    addgroup $USER sudo && \
    echo "%sudo ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/"$USER"

USER "$USER"

# Set WORKDIR to home of arpit
WORKDIR /home/"$USER"

# Copy contents of repository into directory _git in the home of the user
COPY / ./_git/

# Change owner of directory to USER from root
RUN sudo chown -R $USER:$USER _git

# Run setup and install all dependencies
RUN sudo ./_git/install.bash --complete

# Start container of the Docker image at bash prompt
CMD ["/bin/bash"]

