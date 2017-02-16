How to setup the BABEL database multilingual training environment
=================================================================
a) Preparation: you need to make sure the BABEL data and the F4DE scoring software
   is set up as it is in JHU, or change this setup accordingly.  This will probably
   be hard and will involve some trial and error.  Some relevant pathnames can be
   found in conf/lang/* and ./path.sh
   This step is as same as (a) in normal babel (egs/babel/s5d).


b) Prepare the data and alignments for languages in multilingual setup.
    ba) create empty directory exp/language-name and data/language-name
        e.g. mkdir exp/101-cantonese;  mkdir data/101-cantonese
        language-name should be the name used in config file in conf/lang.
    bb) prepare the data and alignment tri5 (Read egs/babel/s5d/README.txt
        for more details.)
    bc) make soft-link  in data/lang-name and exp/lang-name to corresponding
        data and exp dir for all languages.
        e.g.
        (
        cd data/101-cantonese
        ln -s /path-to-101-cantonese-data-dir/train .
        ln -s /path-to-101-cantonese-data-dir/lang .

        cd exp/101-cantonese
        ln -s /path-to-101-cantonese-exp-dir/tri5 .
        )
    bd) you can create local.conf and define training config for multilingual training
        e.g. s5/local.conf

        echo -e "
        use_pitch=true
        \nuse_ivector=true
        \n#lda-mllt transform for used to train global-ivector
        \nlda_mllt_lang=101-cantonese
        \n#lang_list=(space-separated-list-of-multilingual-langs)
        \nlang_list=(101-cantonese 102-assamese 103-bengali)
        \ndecode_lang_list=(101-cantonese)
        \nuse_flp=true # fullLP train-data and alignment used in training.
        " > local.conf

Running the multilingual training script
=========================================
a) You can run the following script to train multilingual TDNN model using
    xent objective.
    local/nnet3/run_tdnn_multilingual.sh

    This script does the following steps.
    aa) Generates 3 speed-perturbed version of training data and
        its high resolution 40-dim MFCC (+pitch) features and tri5_ali{_sp}

    ab) Creates pooled training data using all training languages and generates
        global i-vector extractor over pooled data.

    ac) Generates separate egs-dir in exp/lang-name/nnet3/egs for all languages
        in lagn_list

    ad) Creates multilingual-egs-dir and train the multilingual model.

    ac) Generates decoding results for languages in decode_lang_list.

b) You can run the following scripts to train multilingual model with
    bottleneck layer with dim 'bnf_dim' and generate bottleneck features for
    'lang-name' in data/lang-name/train{_sp}_bnf and train SAT model on top
    of MFCC+BNF features (exp/lang-name/tri6).
    local/nnet3/run_multilingual_bnf.sh --bnf-dim bnf_dim lang-name

    You can also use trained multilingual model (the default component name
    used to extract bnf is tdnn_bn.renorm) as
    local/nnet3/run_multilingual_bnf.sh \
      --multilingual-dir exp/nnet3/tdnn_multi_bnf lang-name
