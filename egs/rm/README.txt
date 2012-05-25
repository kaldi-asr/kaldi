About the Resource Management corpus:
    Clean speech in a medium-vocabulary task consisting
    of commands to a (presumably imaginary) computer system.  About 3 hours
    of training data. 
    Available from the LDC as catalog number LDC93S3A (it may be possible to
    get the same data using combinations of other catalog numbers, but this
    is the one we used).

Each subdirectory of this directory contains the
scripts for a sequence of experiments.  Note: s3 is the "default" set of
scripts at the moment.

  s1: This setup is experiments with GMM-based systems with various 
      Maximum Likelihood 
      techniques including global and speaker-specific transforms.
      See a parallel setup in ../wsj/s1
      This setup is now slightly deprecated: probably you should look
      at the s3 recipes.
      
  s2: This setup is experiments with pure hybrid system.

  s3: This is "new-style" recipes.  We recommend to look here first, for
      RM recipes.
      However, the WSJ or Switchboard s3/ recipes are probably a better
      place to look if you're trying to set up something on your own 
      data, because they're more configurable.

  s4: A recipe based on freely available subset of RM data. The recipe uses
      Sphinx feature files distributed by CMU and metadata available on LDC
      website. Script and data layouts are based on egs/rm/s3.

  s5: This is the "new-new-style" recipe.  It is now finished.
      All further work will be on top of this style of recipe.  Note: 
      unlike previous recipes, this now uses the same underlying
      scripts as the WSJ recipe.

