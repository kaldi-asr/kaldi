#! /usr/bin/env bash
#
# Environment variables to use by speech_recognition repo

export KALDI_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export GST_PLUGIN_PATH=$KALDI_ROOT/src/gst-plugin${GST_PLUGIN_PATH:+:${GST_PLUGIN_PATH}}
