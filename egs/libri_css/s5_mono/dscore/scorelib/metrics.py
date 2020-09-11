"""Functions for scoring frame-level diarization output."""
# TODO: Module is too long. Refactor.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import re
import shutil
import subprocess
import tempfile

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix, issparse

from .rttm import write_rttm
from .uem import gen_uem, write_uem
from .utils import clip, xor

__all__ = ['bcubed', 'conditional_entropy', 'contingency_matrix', 'der',
           'goodman_kruskal_tau', 'jer', 'mutual_information']


EPS = np.finfo(float).eps


def contingency_matrix(ref_labels, sys_labels):
    """Return contingency matrix between ``ref_labels`` and ``sys_labels``.

    Parameters
    ----------
    ref_labels : ndarray, (n_samples,) or (n_samples, n_ref_classes)
        Reference labels encoded using one-hot scheme.

    sys_labels : ndarray, (n_samples,) or ((n_samples, n_sys_classes)
        System labels encoded using one-hot scheme.

    Returns
    -------
    cm : ndarray, (n_ref_classes, n_sys_classes)
        Contigency matrix whose ``i, j``-th entry is the number of times the
        ``i``-th reference label and ``j``-th system label co-occur.
    """
    if ref_labels.ndim != sys_labels.ndim:
        raise ValueError(
            'ref_labels and sys_labels should either both be 1D arrays of '
            'labels or both be 2D arrays of one-hot encoded labels: shapes '
            'are %r, %r' % (ref_labels.shape, sys_labels.shape))
    if ref_labels.shape[0] != sys_labels.shape[0]:
        raise ValueError(
            'ref_labels and sys_labels must have same size: received %d '
            'and %d' % (ref_labels.shape[0], sys_labels.shape[0]))
    if ref_labels.ndim == 1:
        ref_classes, ref_class_inds = np.unique(
            ref_labels, return_inverse=True)
        sys_classes, sys_class_inds = np.unique(
            sys_labels, return_inverse=True)
        n_frames = ref_labels.size
        cm = coo_matrix(
            (np.ones(n_frames), (ref_class_inds, sys_class_inds)),
            shape=(ref_classes.size, sys_classes.size),
            dtype=np.int)
        cm = cm.toarray()
    else:
        ref_labels = ref_labels.astype('int64', copy=False)
        sys_labels = sys_labels.astype('int64', copy=False)
        cm = ref_labels.T.dot(sys_labels)
        if issparse(cm):
            cm = cm.toarray()
    return cm


def bcubed(ref_labels, sys_labels, cm=None):
    """Return B-cubed precision, recall, and F1.

    The B-cubed precision of an item is the proportion of items with its
    system label that share its reference label (Bagga and Baldwin, 1998).
    Similarly, the B-cubed recall of an item is the proportion of items
    with its reference label that share its system label. The overall B-cubed
    precision and recall, then, are the means of the precision and recall for
    each item.

    Parameters
    ----------
    ref_labels : ndarray, (n_frames,)
        Reference labels.

    sys_labels : ndarray, (n_frames,)
        System labels.

    cm : ndarray, (n_ref_classes, n_sys_classes)
        Contingency matrix between reference and system labelings. If None,
        will be computed automatically from ``ref_labels`` and ``sys_labels``.
        Otherwise, the given value will be used and ``ref_labels`` and
        ``sys_labels`` ignored.
        (Default: None)

    Returns
    -------
    precision : float
        B-cubed precision.

    recall : float
        B-cubed recall.

    f1 : float
        B-cubed F1.

    References
    ----------
    Bagga, A. and Baldwin, B. (1998). "Algorithms for scoring coreference
    chains." Proceedings of LREC 1998.
    """
    if cm is None:
        cm = contingency_matrix(ref_labels, sys_labels)
    cm = cm.astype('float64')
    cm_norm = cm / cm.sum()
    precision = np.sum(cm_norm * (cm / cm.sum(axis=0)))
    recall = np.sum(cm_norm * (cm / np.expand_dims(cm.sum(axis=1), 1)))
    f1 = 2*(precision*recall)/(precision + recall)
    return precision, recall, f1


def goodman_kruskal_tau(ref_labels, sys_labels, cm=None):
    """Return Goodman-Kruskal tau between ``ref_labels`` and ``sys_labels``.

    Parameters
    ----------
    ref_labels : ndarray, (n_frames,)
        Reference labels.

    sys_labels : ndarray, (n_frames,)
        System labels.

    cm : ndarray, (n_ref_classes, n_sys_classes)
        Contingency matrix between reference and system labelings. If None,
        will be computed automatically from ``ref_labels`` and ``sys_labels``.
        Otherwise, the given value will be used and ``ref_labels`` and
        ``sys_labels`` ignored.
        (Default: None)

    Returns
    -------
    tau_ref_sys : float
        Value between 0 and 1 that is high when ``ref_labels`` is predictive
        of ``sys_labels`` and low when ``ref_labels`` provides essentially no
        information about ``sys_labels``.

    tau_sys_ref : float
        Value between 0 and 1 that is high when ``sys_labels`` is predictive
        of ``ref_labels`` and low when ``sys_labels`` provides essentially no
        information about ``ref_labels``.

    References
    ----------
    - Goodman, L.A. and Kruskal, W.H. (1954). "Measures of association for
      cross classifications." Journal of the American Statistical Association.
    - Pearson, R. (2016). GoodmanKruskal: Association Analysis for Categorical
      Variables. https://CRAN.R-project.org/package=GoodmanKruskal.
    """
    if cm is None:
        cm = contingency_matrix(ref_labels, sys_labels)
    cm = cm.astype('float64')
    cm = cm / cm.sum()
    ref_marginals = cm.sum(axis=1)
    sys_marginals = cm.sum(axis=0)
    n_ref_classes, n_sys_classes = cm.shape

    # Tau(ref, sys).
    if n_sys_classes == 1:
        # Special case: only single class in system labeling, so any
        #               reference labeling is perfectly predictive.
        tau_ref_sys = 1.
    else:
        vy = 1 - np.sum(sys_marginals**2)
        xy_term = np.sum(cm**2, axis=1)
        vy_bar_x = 1 - np.sum(xy_term / ref_marginals)
        tau_ref_sys = (vy - vy_bar_x) / vy

    # Tau(sys, ref).
    if n_ref_classes == 1:
        # Special case: only single class in reference labeling, so any
        #               system labeling is perfectly predictive.
        tau_sys_ref = 1.
    else:
        vx = 1 - np.sum(ref_marginals**2)
        yx_term = np.sum(cm**2, axis=0)
        vx_bar_y = 1 - np.sum(yx_term / sys_marginals)
        tau_sys_ref = (vx - vx_bar_y) / vx

    return tau_ref_sys, tau_sys_ref


def conditional_entropy(ref_labels, sys_labels, cm=None, nats=False):
    """Return conditional entropy of ``ref_labels`` given ``sys_labels``.

    The conditional entropy ``H(ref | sys)`` quantifies how much information
    is needed to describe the reference labeling given that the system labeling
    is known. It is 0 when the labelings are identical and increases as the
    system labeling becomes less descriptive of the reference labeling.

    Parameters
    ----------
    ref_labels : ndarray, (n_frames,)
        Reference labels.

    sys_labels : ndarray, (n_frames,)
        System labels.

    cm : ndarray, (n_ref_classes, n_sys_classes)
        Contingency matrix between reference and system labelings. If None,
        will be computed automatically from ``ref_labels`` and ``sys_labels``.
        Otherwise, the given value will be used and ``ref_labels`` and
        ``sys_labels`` ignored.
        (Default: None)

    nats : bool, optional
        If True, return conditional entropy in nats. Otherwise, return in bits.
        (Default: False)

    References
    ----------
    - https://en.wikipedia.org/wiki/Conditional_entropy
    - Cover, T.M. and Thomas, J.A. (1991). Elements of Information Theory.
    - Rosenberg, A. and Hirschberg, J. (2007). "V-Measure: A conditional
      entropy-based external cluster evaluation measure." Proceedings of EMNLP
      2007.
    """
    log = np.log if nats else np.log2
    if cm is None:
        cm = contingency_matrix(ref_labels, sys_labels)
    sys_marginals = cm.sum(axis=0)
    N = cm.sum()
    ref_inds, sys_inds = np.nonzero(cm)
    vals = cm[ref_inds, sys_inds] # Non-zero values of contingency matrix.
    sys_marginals = sys_marginals[sys_inds] # Corresponding marginals.
    sigma = vals/N * (log(sys_marginals) - log(vals))
    return sigma.sum()


VALID_NORM_METHODS = set(['min', 'sum', 'sqrt', 'max'])

def mutual_information(ref_labels, sys_labels, cm=None, nats=False,
                       norm_method='sqrt'):
    """Return mutual information between ``ref_labels`` and ``sys_labels``.

    The mutual information ``I(ref, sys)`` quantifies how much information is
    shared by the reference and system labelings; that is, how much knowing
    one labeling reduces uncertainty about the other. It is 0 in the case that
    the labelings are independent and increases as they become more predictive
    of each other with a least upper bound of ``min(H(ref), H(sys))``.

    Normalized mutual information converts mutual information into a similarity
    metric ranging on [0, 1]. Multiple normalization schemes are available,
    set by the ``norm_method`` argument, which takes the following values:

    - ``min``  --  normalize by ``min(H(ref), H(sys))``
    - ``sum``  --  normalize by ``0.5*(H(ref) + H(sys))``
    - ``sqrt``  --  normalize by ``sqrt(H(ref)*H(sys))``
    - ``max``  --  normalize by ``max(H(ref), H(sys))``

    Parameters
    ----------
    ref_labels : ndarray, (n_frames,)
        Reference labels.

    sys_labels : ndarray, (n_frames,)
        System labels.

    cm : ndarray, (n_ref_classes, n_sys_classes)
        Contingency matrix between reference and system labelings. If None,
        will be computed automatically from ``ref_labels`` and ``sys_labels``.
        Otherwise, the given value will be used and ``ref_labels`` and
        ``sys_labels`` ignored.
        (Default: None)

    nats : bool, optional
        If True, return nats. Otherwise, return bits.
        (Default: False)

    norm_method : str, optional
        Normalization method for NMI computation.
        (Default: 'sqrt')

    Returns
    -------
    mi : float
        Mutual information.

    nmi : float
        Normalized mutual information.

    References
    ----------
    - https://en.wikipedia.org/wiki/Mutual_information
    - Cover, T.M. and Thomas, J.A. (1991). Elements of Information Theory.
    - Strehl, A. and Ghosh, J. (2002). "Cluster ensembles  -- A knowledge
      reuse framework for combining multiple partitions." Journal of Machine
      Learning Research.
    - Nguyen, X.V., Epps, J., and Bailey, J. (2010). "Information theoretic
      measures for clustering comparison: Variants, properties, normalization
      and correction for chance." Journal of Machine Learning Research.
    """
    if norm_method not in VALID_NORM_METHODS:
        raise ValueError('"%s" is not a valid NMI normalization method.')
    log = np.log if nats else np.log2
    if cm is None:
        cm = contingency_matrix(ref_labels, sys_labels)

    # Special cases in which one or more of H(ref) and H(sys) is
    # 0.
    n_ref_classes, n_sys_classes = cm.shape
    if xor(n_ref_classes == 1, n_sys_classes == 1):
        # Case 1: MI is by definition 0 as should be NMI, regardless of
        #         normalization.
        return 0.0, 0.0
    if n_ref_classes == n_sys_classes == 1:
        # Case 2: MI is 0, but as the data is not split, each clustering
        #         is perfectly predictive of the other, so set NMI to 1.
        return 0.0, 1.0

    # Mutual information.
    N = cm.sum()
    ref_marginals = cm.sum(axis=1)
    sys_marginals = cm.sum(axis=0)
    ref_inds, sys_inds = np.nonzero(cm)
    vals = cm[ref_inds, sys_inds] # Non-zero values of contingency matrix.
    outer = ref_marginals[ref_inds]*sys_marginals[sys_inds]
    sigma = (vals/N) * (
        log(vals) - log(outer) + log(N))
    mi = sigma.sum()
    mi = max(mi, 0.)

    # Normalized mutual information.
    def h(p):
        p = p[p > 0]
        return max(-np.sum(p*log(p)), 0)
    h_ref = h(ref_marginals / N)
    h_sys = h(sys_marginals / N)
    if norm_method == 'max':
        denom = max(h_ref, h_sys)
    elif norm_method == 'sum':
        denom = 0.5*(h_ref + h_sys)
    elif norm_method == 'sqrt':
        denom = np.sqrt(h_ref*h_sys)
    elif norm_method == 'min':
        denom = min(h_ref, h_sys)
    nmi = mi / denom
    nmi = clip(nmi, 0., 1.)

    return mi, nmi


SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
MDEVAL_BIN = os.path.join(SCRIPT_DIR, 'md-eval-22.pl')
FILE_REO = re.compile(r'(?<=Speaker Diarization for).+(?=\*\*\*)')
SCORED_SPEAKER_REO = re.compile(r'(?<=SCORED SPEAKER TIME =)[\d.]+')
MISS_SPEAKER_REO = re.compile(r'(?<=MISSED SPEAKER TIME =)[\d.]+')
FA_SPEAKER_REO = re.compile(r'(?<=FALARM SPEAKER TIME =)[\d.]+')
ERROR_SPEAKER_REO = re.compile(r'(?<=SPEAKER ERROR TIME =)[\d.]+')

# TODO: Working with md-eval is a PITA, even with modifications to the
#       reporting. Suggest looking into moving over to pyannote's
#       implementation.
def der(ref_turns, sys_turns, collar=0.0, ignore_overlaps=False, overlap_only=False, uem=None):
    """Return overall diarization error rate.

    Diarization error rate (DER), introduced for the NIST Rich Transcription
    evaluations, is computed as the sum of the following:

    - speaker error  --  percentage of scored time for which the wrong speaker
      id is assigned within a speech region
    - false alarm speech  --   percentage of scored time for which a nonspeech
      region is incorrectly marked as containing speech
    - missed speech  --  percentage of scored time for which a speech region is
      incorrectly marked as not containing speech

    As with word error rate, a score of zero indicates perfect performance and
    higher scores (which may exceed 100) indicate poorer performance.

    DER is computed as defined in the NIST RT-09 evaluation plan using version
    22 of the ``md-eval.pl`` scoring script. When ``ignore_overlaps=False``,
    this is equivalent to running the following command:

        md-eval.pl -r ref.rttm -s sys.rttm -c collar -u uemf

    where ``ref.rttm`` and ``sys.rttm`` are RTTM files produced from
    ``ref_turns`` and ``sys_turns`` respectively and ``uemf`` is an
    Un-partitioned Evaluation Map (UEM) file delimiting the scoring regions.
    If a ``UEM`` instance is supplied via the``uem`` argument, this file will
    be created from the supplied UEM. Otherwise, it will be generated
    automatically from ``ref_turns`` and ``sys_turns`` using the
    ``uem.gen_uem`` function. Similarly, when ``ignore_overlaps=True``:

        md-eval.pl -r ref.rttm -s sys.rttm -c collar -u uemf -1

    Or, when ``overlap_only=True``:

        md-eval.pl -r ref.rttm -s sys.rttm -c collar -u uemf -2

    Parameters
    ----------
    ref_turns : list of Turn
        Reference speaker turns.

    sys_turns : list of Turn
        System speaker turns.

    collar : float, optional
        Size of forgiveness collar in seconds. Diarization output will not be
        evaluated within +/- ``collar`` seconds of reference speaker
        boundaries.
        (Default: 0.0)

    ignore_overlaps : bool, optional
        If True, ignore regions in the reference diarization in which more
        than one speaker is speaking.
        (Default: False)

    overlap_only : bool, optional
        If True, only score regions in the reference RTTM where multiple
        speakers are speaking.
        (Default: False)

    uem : UEM, optional
        Evaluation map. If not supplied, will be generated automatically from
        ``ref_turns`` and ``sys_turns``.
        (Default: None)

    Returns
    -------
    file_to_der : dict
        Mapping from files to diarization error rates (in percent) for those
        files.

    global_der : float
        Overall diarization error rate (in percent).

    References
    ----------
    NIST. (2009). The 2009 (RT-09) Rich Transcription Meeting Recognition
    Evaluation Plan. https://web.archive.org/web/20100606041157if_/http://www.itl.nist.gov/iad/mig/tests/rt/2009/docs/rt09-meeting-eval-plan-v2.pdf
    """
    tmp_dir = tempfile.mkdtemp()

    # Write RTTMs.
    ref_rttm_fn = os.path.join(tmp_dir, 'ref.rttm')
    write_rttm(ref_rttm_fn, ref_turns)
    sys_rttm_fn = os.path.join(tmp_dir, 'sys.rttm')
    write_rttm(sys_rttm_fn, sys_turns)

    # Write UEM.
    if uem is None:
        uem = gen_uem(ref_turns, sys_turns)
    uemf = os.path.join(tmp_dir, 'all.uem')
    write_uem(uemf, uem)

    # Actually score.
    try:
        cmd = [MDEVAL_BIN,
               '-af',
               '-r', ref_rttm_fn,
               '-s', sys_rttm_fn,
               '-c', str(collar),
               '-u', uemf,
              ]
        if ignore_overlaps:
            cmd.append('-1')
        if overlap_only:
            cmd.append('-2')
        stdout = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        stdout = e.output
    finally:
        shutil.rmtree(tmp_dir)

    # Parse md-eval output to extract by-file and total scores.
    stdout = stdout.decode('utf-8')
    file_ids = [m.strip() for m in FILE_REO.findall(stdout)]
    file_ids = [file_id[2:] if file_id.startswith('f=') else file_id
                for file_id in file_ids]
    
    scored_speaker_times = np.array(
        [float(m) for m in SCORED_SPEAKER_REO.findall(stdout)])
    miss_speaker_times = np.array(
        [float(m) for m in MISS_SPEAKER_REO.findall(stdout)])
    fa_speaker_times = np.array(
        [float(m) for m in FA_SPEAKER_REO.findall(stdout)])
    error_speaker_times = np.array(
        [float(m) for m in ERROR_SPEAKER_REO.findall(stdout)])
    
    with np.errstate(invalid='ignore', divide='ignore'):
        error_times = miss_speaker_times + fa_speaker_times + error_speaker_times
        ders = error_times / scored_speaker_times
    ders[np.isnan(ders)] = 0 # Numerator and denominator both 0.
    ders[np.isinf(ders)] = 1 # Numerator > 0, but denominator = 0.
    ders *= 100. # Convert to percent.

    # Reconcile with UEM, keeping in mind that in the edge case where no
    # reference turns are observed for a file, md-eval doesn't report results
    # for said file.
    file_to_der_base = dict(zip(file_ids, ders))
    file_to_der = {}
    for file_id in uem:
        try:
            der = file_to_der_base[file_id]
        except KeyError:
            # Check for any system turns for that file, which should be FAs,
            # assuming that the turns have been cropped to the UEM scoring
            # regions.
            n_sys_turns = len(
                [turn for turn in sys_turns if turn.file_id == file_id])
            der = 100. if n_sys_turns else 0.0
        file_to_der[file_id] = der
    global_der = file_to_der_base['ALL']

    return file_to_der, global_der


def jer(file_to_ref_durs, file_to_sys_durs, file_to_cm, min_ref_dur=0):
    """Return Jacard error rate.

    Jaccard error rate (JER) rate is based on the Jaccard index, a similarity
    measure used to evaluate the output of image segmentation systems. An
    optimal mapping between reference and system speakers is determined and
    for each pair the Jaccard index is computed. The Jaccard error rate is then
    defined as 1 minus the average of these scores.

    More concretely, assume we have ``N`` reference speakers and ``M`` system
    speakers. An optimal mapping between speakers is determined using the
    Hungarian algorithm so that each reference speaker is paired with at most
    one system speaker and each system speaker with at most one reference
    speaker. Then, for each reference speaker ``ref`` the speaker-specific
    Jaccard error rate is ``(FA + MISS)/TOTAL``, where:
    - ``TOTAL`` is the duration of the union of reference and system speaker
      segments; if the reference speaker was not paired with a system speaker,
      it is the duration of all reference speaker segments
    - ``FA`` is the total system speaker time not attributed to the reference
      speaker; if the reference speaker was not paired with a system speaker,
      it is 0
    - ``MISS`` is the total reference speaker time not attributed to the
      system speaker; if the reference speaker was not paired with a system
      speaker, it is equal to ``TOTAL``
    The Jaccard error rate then is the average of the speaker specific Jaccard
    error rates.

    JER and DER are highly correlated with JER typically being higher, especially
    in recordings where one or more speakers is particularly dominant. Where it
    tends to track DER is in outliers where the diarization is especially bad,
    resulting on one or more unmapped system speakers whose speech is not then
    penalized. In these cases, where DER can easily exceed 500%, JER will never
    exceed 100% and may be far lower if the reference speakers are handled
    correctly. For this reason, it may be useful to pair JER with another metric
    evaluating speech detection and/or speaker overlap detection.

    Parameters
    ----------
    file_to_ref_durs : dict
        Mapping from files to durations of reference speakers in those files.

    file_to_sys_durs : dict
        Mapping from files to durations of system speakers in those files.

    file_to_cm : dict
        Mapping from files to contingency matrices for speakers in those files.

    min_ref_dur : float, optional
        Minimum reference speaker duration. Reference speakers with durations
        less than ``min_ref_dur`` will be excluded for scoring purposes. Setting
        this to a small non-zero number may stabilize JER when the reference
        segmentation contains multiple extraneous speakers.
        (Default: 0.0)

    Returns
    -------
    file_to_jer : dict
        Mapping from files to Jaccard error rates (in percent) for those files.

    global_jer : float
        Overall Jaccard error rate (in percent).

    References
    ----------
    https://en.wikipedia.org/wiki/Jaccard_index
    """
    # TODO: Explore treating non-speech as additional speaker for computation to
    #       more gracefully deal with exceptionally poor system performance.
    ref_dur_fids = set(file_to_ref_durs.keys())
    sys_dur_fids = set(file_to_sys_durs.keys())
    cm_fids = set(file_to_cm.keys())
    if not ref_dur_fids == sys_dur_fids == cm_fids:
        raise ValueError(
            'All passed dicts must have same keys.')
    file_ids = ref_dur_fids
    file_to_jer = {}
    all_speaker_jers = []
    n_ref_speakers_global = 0
    n_sys_speakers_global = 0
    for file_id in file_ids:
        # Filter.
        ref_durs = file_to_ref_durs[file_id]
        sys_durs = file_to_sys_durs[file_id]
        cm = file_to_cm[file_id]
        ref_keep = ref_durs >= min_ref_dur
        ref_durs = ref_durs[ref_keep]
        cm = cm[ref_keep, ]
        n_ref_speakers = ref_durs.size
        n_sys_speakers = sys_durs.size
        n_ref_speakers_global += n_ref_speakers
        n_sys_speakers_global += n_sys_speakers

        # Handle edge cases where either reference or system segmentation
        # posited no speech.
        if n_ref_speakers == 0 and n_sys_speakers > 0:
            # Case 1: no reference speech.
            file_to_jer[file_id] = 100.0
            continue
        elif n_ref_speakers > 0 and n_sys_speakers == 0:
            # Case 2: no system speech.
            file_to_jer[file_id] = 100.0
            all_speaker_jers.extend([100.]*n_ref_speakers)
            continue
        elif n_ref_speakers == 0 and n_sys_speakers == 0:
            # Case 3: no reference or system speech
            file_to_jer[file_id] = 0.0
            continue

        # Determine all speaker-level JER.
        ref_durs = np.tile(ref_durs, [n_sys_speakers, 1]).T
        sys_durs = np.tile(sys_durs, [n_ref_speakers, 1])
        intersect = cm
        union = ref_durs + sys_durs - intersect
        jer_speaker = 1 - intersect / union

        # Find dominant mapping by Hungarian algorithm (scipy >= 0.17) and compute
        # JER.
        ref_speaker_inds, sys_speaker_inds = linear_sum_assignment(jer_speaker)
        jers = np.ones(n_ref_speakers, dtype='float64')
        for ref_speaker_ind, sys_speaker_ind in zip(
                ref_speaker_inds, sys_speaker_inds):
            jers[ref_speaker_ind] = jer_speaker[ref_speaker_ind,
                                                sys_speaker_ind]
        jers *= 100.
        file_to_jer[file_id] = jers.mean()
        all_speaker_jers.extend(jers)

    # Determine global JER.
    if n_ref_speakers_global == 0 and n_sys_speakers_global > 0:
        # Case 1: no reference speech on ANY file.
        global_jer = 100.
    elif n_ref_speakers_global > 0 and n_sys_speakers_global == 0:
        # Case 2: no system speech on ANY file.
        global_jer = 100.
    elif n_ref_speakers_global == n_sys_speakers_global == 0:
        # Case 3: no reference OR system speech on ANY file.
        global_jer = 0.0
    else:
        # General case: at least 1 reference and 1 system speaker present.
        global_jer = np.mean(all_speaker_jers)

    return file_to_jer, global_jer
