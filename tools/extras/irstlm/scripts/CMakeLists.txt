INSTALL(PROGRAMS
    add-start-end.sh build-lm-qsub.sh build-lm.sh build-sublm.pl goograms2ngrams.pl lm-stat.pl mdtsel.sh merge-sublm.pl ngram-split.pl rm-start-end.sh sort-lm.pl split-dict.pl split-ngt.sh wrapper 
    DESTINATION bin
    PERMISSIONS OWNER_EXECUTE OWNER_READ OWNER_WRITE
    )
