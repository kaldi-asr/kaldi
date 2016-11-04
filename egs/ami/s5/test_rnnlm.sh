rnnlm-init --binary=false rnnlm.config rnnlm.mdl
rnnlm-train --use-gpu=no rnnlm.mdl ark:egs.1 2.mdl
