#!/bin/bash
# Downlaods Api.ai chain model into exp/api.ai-model/ (will replace one if exists)

DOWNLOAD_URL="https://api.ai/downloads/api.ai-kaldi-asr-model.zip"

echo "Downloading model"
wget -N $DOWNLOAD_URL || ( echo "Unable to download model: $DOWNLOAD_URL" && exit 1 );

echo "Unpacking model"
unzip api.ai-kaldi-asr-model.zip || ( echo "Unable to extract api.ai-kaldi-asr-model.zip" && exit 1 );

echo "Moving model to exp/api.ai-model/"
if [ ! -d exp ]; then
	mkdir exp;
fi;

if [ -d exp/api.ai-model ]; then
	echo "Found existing model, removing";
	rm -rf exp/api.ai-model/
fi

mv api.ai-kaldi-asr-model exp/api.ai-model || ( echo "Unable to move model to exp/" && exit 1 )

echo "Model is ready to use use recognize-wav.sh to do voice recognition"
