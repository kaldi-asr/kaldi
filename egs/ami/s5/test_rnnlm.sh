rnnlm-init --binary=false rnnlm.config 0.mdl
rnnlm-train --use-gpu=no 0.mdl ark:egs.1 1.mdl
