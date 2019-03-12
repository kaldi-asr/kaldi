# -----------------------------------------------------------------------------
#       Dockerfile to build Kaldi gstreamer server with tue-env
# -----------------------------------------------------------------------------

# Set the base image to Ubuntu with tue-env
FROM tuerobotics/tue-env:master

# Update the image and install basic packages
RUN sudo apt-get update -qq && \
    # Make tue-env available to the intermediate image
    # This step needs to be executed at every RUN step
    source ~/.bashrc && \
    # Install kaldi
    tue-get install kaldi

