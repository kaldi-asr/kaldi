There is a copy of this document on Google Docs, which renders the equations better:
[link](https://docs.google.com/document/d/1pie-PU6u2NZZC_FzocBGGm6mpfBJMiCft9UoG0uA1kA/edit?usp=sharing)

* * *

# GOP on Kaldi

The Goodness of Pronunciation (GOP) is a variation of the posterior probability, for phone level pronunciation scoring.
GOP is widely used in pronunciation evaluation and mispronunciation detection tasks.

This implementation is mainly based on the following paper:

Hu, W., Qian, Y., Soong, F. K., & Wang, Y. (2015). Improved mispronunciation detection with deep neural network trained acoustic models and transfer learning based logistic regression classifiers. Speech Communication, 67(January), 154-166.

## GOP-GMM

In the conventional GMM-HMM based system, GOP was first proposed in (Witt et al., 2000). It was defined as the duration normalised log of the posterior:

$$
GOP(p)=\frac{1}{t_e-t_s+1} \log p(p|\mathbf o)
$$

where $\mathbf o$ is the input observations, $p$ is the canonical phone, $t_s, t_e$ are the start and end frame indexes.

Assuming $p(q_i)\approx p(q_j)$ for any $q_i, q_j$, we have:

$$
\log p(p|\mathbf o)=\frac{p(\mathbf o|p)p(p)}{\sum_{q\in Q} p(\mathbf o|q)p(q)}
                   \approx\frac{p(\mathbf o|p)}{\sum_{q\in Q} p(\mathbf o|q)}
$$

where $Q$ is the whole phone set.

The numerator of the equation is calculated from forced alignment result and the denominator is calculated from an Viterbi decoding with a unconstrained phone loop.

We do not implement GOP-GMM for Kaldi, as GOP-NN performs much better than GOP-GMM.

## GOP-NN

The definition of GOP-NN is a bit different from the GOP-GMM. GOP-NN was defined as the log phone posterior ratio between the canonical phone and the one with the highest score (Hu et al., 2015).

Firstly we define Log Phone Posterior (LPP):

$$
LPP(p)=\log p(p|\mathbf o; t_s,t_e)
$$

Then we define the GOP-NN using LPP:

$$
GOP(p)=\log \frac{LPP(p)}{\max_{q\in Q} LPP(q)}
$$

LPP could be calculated as:

$$
LPP(p) \approx \frac{1}{t_e-t_s+1} \sum_{t=t_s}^{t_e}\log p(p|o_t)
$$

$$
p(p|o_t) = \sum_{s \in p} p(s|o_t)
$$

where $s$ is the senone label, $\{s|s \in p\}$ is the states belonging to those triphones whose current phone is $p$.

## Phone-level Feature

Normally the classifier-based approach archives better performance than GOP-based approach.

Different from GOP based method, an extra supervised training process is needed. The input features for supervised training are phone-level, segmental features. The phone-level feature is defined as:

$$
{[LPP(p_1),\cdots,LPP(p_M), LPR(p_1|p_i), \cdots, LPR(p_j|p_i),\cdots]}^T
$$

where the Log Posterior Ratio (LPR) between phone $p_j$ and $p_i$ is defined as:

$$
LPR(p_j|p_i) = \log p(p_j|\mathbf o; t_s, t_e) - \log p(p_i|\mathbf o; t_s, t_e)
$$

## Implementation

This implementation consists of a executable binary `bin/compute-gop` and some scripts.

`compute-gop` computes GOP and extracts phone-level features using nnet output probabilities.
The output probabilities are assumed to be from a log-softmax layer.

The script `run.sh` shows a typical pipeline based on librispeech's model and data.

In Hu's paper, GOP was computed using a feed-forward DNN.
We have tried to use the output-xent of a chain model to compute GOP, but the result was not good.
We guess the HMM topo of chain model may not fit for GOP.

The nnet3's TDNN (no chain) model performs well in GOP computing, so this recipe uses it.

## Acknowledgement
The author of this recipe would like to thank Xingyu Na for his works of model tuning and his helpful suggestions.
