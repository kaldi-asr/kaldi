#!/usr/bin/env python3

# I ran from the shell:
# . ./path.sh
# steps/nnet3/report/convert_model.py exp/chain/tdnn1g_sp/24.mdl{,.pkl}
# steps/nnet3/report/convert_model.py exp/chain/tdnn1g_sp/25.mdl{,.pkl}
# .. and then this script:
# local/chain/diagnostic/report_example.py

# Note: I make no claim that the information in the generated report is
# understandable in general; it's just something I was plotting for
# my own information.  The point of this script is to demonstrate
# how to use steps/nnet3/report/convert_model.py.

import sys
sys.path.append("steps/nnet3/report")
import convert_model
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pickle
import numpy as np

# instead of the pickle.load commands, you could do in python, as follows:
# (but dumping them to disk first is faster in case you'll be running this
# script more than once).
# model1 = convert_model.read_model("exp/chain/tdnn1g_sp/24.mdl")
# model2 = convert_model.read_model("exp/chain/tdnn1g_sp/25.mdl")
model1 = pickle.load(open("exp/chain/tdnn1g_sp/24.mdl.pkl", "rb"))
model2 = pickle.load(open("exp/chain/tdnn1g_sp/25.mdl.pkl", "rb"))

convert_model.compute_derived_quantities(model1)
convert_model.compute_derived_quantities(model2)
convert_model.compute_progress(model1, model2)


f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3)
plt.tight_layout()
fs=5
ss=4
ax1.scatter(model1['tdnn4.affine']['col-norms-3'],
            model1['tdnn3.affine']['row-change'], s=ss)
ax1.set_title('row-change3 versus  column-norms4',
              fontsize=fs)

ax2.scatter(model1['tdnn4.affine']['col-norms-3'],
            model1['tdnn3.affine']['rel-row-change'], s=ss)
ax2.set_title('rel-row-change3 versus  column-norms4',
              fontsize=fs)

ax3.scatter(model1['tdnn4.affine']['col-norms-3'],
              model1['tdnn3.affine']['row-norms'], s=ss)
ax3.set_title('row-norms3 versus  column-norms4',
              fontsize=fs)

ax4.scatter(model1['tdnn4.affine']['col-norms'],
            model1['tdnn4.affine']['rel-col-change'], s=ss)
ax4.set_title('rel-col-change4 versus col-norms4',
              fontsize=fs)

ax5.scatter(model1['tdnn3.batchnorm']['stats-stddev'],
            model1['tdnn4.affine']['col-norms-3'], s=ss)
ax5.set_title('col-norms4 versus batch-norm-stddev3',
              fontsize=fs)


#ax6.scatter(np.reciprocal(model1['tdnn3.relu']['deriv-avg']) * model1['tdnn4.affine']['col-norms-3'],
#            model1['tdnn3.affine']['row-norms'], s=ss)
#ax6.set_title('row-norms3 vs predicted-row-norms3',
#              fontsize=fs)

ax6.scatter(model2['tdnn3.relu']['deriv-avg'] * model2['tdnn3.relu']['oderiv-rms'],
            # model1['tdnn3.relu']['oderiv-rms'],
            model2['tdnn3.affine']['row-norms'], s=ss)
ax6.set_xlim(left=0.00, right=0.009)
ax6.set_title('row-norms3 vs ideriv-rms3',
              fontsize=fs)


plt.savefig('progress.pdf')
