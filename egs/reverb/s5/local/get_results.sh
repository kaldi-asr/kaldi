#!/bin/bash

# Reproduce results in Table 1 from Weninger et al. (2014)
# "Our baselines"

# LDA-STC  fMLLR  MCT    DT     LM     MBR
# No       No     No     No     BG     No
local/summarize_results.pl --lmw=15 tri2a
# No       No     Yes    No     BG     No
local/summarize_results.pl --lmw=15 tri2a_mc
# No       Yes    Yes    No     BG     No
local/summarize_results.pl --lmw=15 tri2a_mc basis_fmllr
# Yes      No     No     No     BG     No
local/summarize_results.pl --lmw=15 tri2b
# Yes      No     Yes    No     BG     No
local/summarize_results.pl --lmw=15 tri2b_mc
# Yes      Yes    Yes    No     BG     No
local/summarize_results.pl --lmw=15 tri2b_mc basis_fmllr
# Yes      No     Yes    Yes    BG     No
local/summarize_results.pl --lmw=15 tri2b_mc_mmi_b0.1
# Yes      Yes    Yes    Yes    BG     No
local/summarize_results.pl --lmw=15 tri2b_mc_mmi_b0.1 basis_fmllr
# Yes      Yes    Yes    Yes    TG     No
local/summarize_results.pl --lm=tg_5k --lmw=15 tri2b_mc_mmi_b0.1 basis_fmllr
# Yes      Yes    Yes    Yes    TG     Yes
local/summarize_results.pl --lm=tg_5k --lmw=15 tri2b_mc_mmi_b0.1 mbr_basis_fmllr
