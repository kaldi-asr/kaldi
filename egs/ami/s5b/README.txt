
This s5b recipe is a streamlined and simplified version of the s5 recipe, with
many components removed.

 Before running run.sh, please run run_prepare_shared.sh.

 Afterwards, you can run:
    run.sh --mic ihm    # builds system for independent headset microphone
    run.sh --mic sdm1   # single distant micropophone
    run.sh --mic mdm8   # multiple distant microphones + beamforming.

 Note: the sdm1 and mdm8 systems depend on the ihm system, because for
 best results we use the IHM alignments to train the neural nets.
 Please see RESULTS_* for results.

- For information about the database see : http://groups.inf.ed.ac.uk/ami/corpus/overview.shtml

