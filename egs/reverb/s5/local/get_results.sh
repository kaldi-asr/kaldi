#!/bin/bash

# Reproduce selected results in Table 1 from Weninger et al. (2014)
# "Our baselines"

# LDA-STC  fMLLR  MCT    DT     LM     MBR
# No       No     No     No     BG     No
local/calc_wer.sh
# No       No     Yes    No     BG     No
local/calc_wer.sh --am tri2a_mc
# No       Yes    Yes    No     BG     No
local/calc_wer.sh --am tri2a_mc --decode basis_fmllr
# Yes      Yes    Yes    No     TG     No
local/calc_wer.sh --am tri2b_mc --lm tg_5k --decode basis_fmllr
# Yes      Yes    Yes    Yes    TG     No
local/calc_wer.sh --am tri2b_mc_mmi_b0.1 --lm tg_5k --decode basis_fmllr
# Yes      Yes    Yes    Yes    TG     Yes
local/calc_wer.sh --am tri2b_mc_mmi_b0.1 --lm tg_5k --decode mbr_basis_fmllr
