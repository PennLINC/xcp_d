# -*- coding: utf-8 -*-
"""Functions related to the documentation.

docdict contains the standard documentation entries used across xcp_d.

source: Eric Larson and MNE-python team.
https://github.com/mne-tools/mne-python/blob/main/mne/utils/docs.py
"""
import sys

###################################
# Standard documentation entries
#
docdict = dict()

docdict["omp_nthreads"] = """
omp_nthreads : :obj:`int`
    Maximum number of threads an individual process may use.
"""

docdict["mem_gb"] = """
mem_gb : :obj:`float`
    Memory limit, in gigabytes.
"""

docdict["fmri_dir"] = """
fmri_dir : :obj:`str`
    Path to the preprocessed derivatives dataset.
    For example, "/path/to/dset/derivatives/fmriprep/".
"""

docdict["output_dir"] = """
output_dir : :obj:`str`
    Path to the output directory for ``xcp_d`` derivatives.
    This should not include the ``xcp_d`` folder.
    For example, "/path/to/dset/derivatives/".
"""

docdict["work_dir"] = """
work_dir : :obj:`str`
    Directory in which to store workflow execution state and temporary files.
"""

docdict["analysis_level"] = """
analysis_level : {"participant"}
    The analysis level for ``xcp_d``. Must be specified as "participant".
"""

docdict["t1w_to_mni"] = """
t1w_to_mni : :obj:`str`
    Path to the T1w-to-MNI transform file.
    May be "identity", for testing purposes.
"""

docdict["mni_to_t1w"] = """
mni_to_t1w : :obj:`str`
    Path to the MNI-to-T1w transform file.
    May be "identity", for testing purposes.
"""

docdict["params"] = """
params : {"36P", "24P", "27P", "acompcor", "acompcor_gsr", \
          "aroma", "aroma_gsr", "custom"}, optional
    Shorthand for the parameter set to extract from the confounds TSV.
    Default is "36P", most expansive option.
"""

docdict["input_type"] = """
input_type : {"fmriprep", "dcan", "hcp"}
    The format of the BIDS derivatives.
"""

docdict["smoothing"] = """
smoothing : float
    The full width at half maximum (FWHM), in millimeters,
    of the smoothing kernel that will be applied to the post-processed and denoised data.
"""

docdict["head_radius"] = """
head_radius : float
    Radius of the head, in millimeters, for framewise displacement calculation.

    ``xcp_d``'s default head radius is 50. The recommended value for infants is 35.
"""

docdict["fd_thresh"] = """
fd_thresh : float
    Framewise displacement threshold for censoring, in millimeters.
    Any framewise displacement values higher than the threshold are flagged as "high motion".
    Default is 0.2 mm.
"""

docdict["bandpass_filter"] = """
bandpass_filter : :obj:`bool`
    If True, a Butterworth bandpass filter will be applied to the fMRI data after interpolation,
    but before regression.
    If False, bandpass filtering will not be performed.
"""

docdict["lower_bpf"] = """
lower_bpf : :obj:`float`
    Lower cut-off frequency for the Butterworth bandpass filter, in Hertz.
    The bandpass filter is applied to the fMRI data after post-processing and denoising.
    Bandpass filtering will only be performed if ``bandpass_filter`` is True.
    This parameter is used in conjunction with ``upper_bpf`` and ``bpf_order``.
"""

docdict["upper_bpf"] = """
upper_bpf : :obj:`float`
    Upper cut-off frequency for the Butterworth bandpass filter, in Hertz.
    The bandpass filter is applied to the fMRI data after post-processing and denoising.
    Bandpass filtering will only be performed if ``bandpass_filter`` is True.
    This parameter is used in conjunction with ``lower_bpf`` and ``bpf_order``.
"""

docdict["bpf_order"] = """
bpf_order : :obj:`int`
    Number of filter coefficients for Butterworth bandpass filter.
    Bandpass filtering will only be performed if ``bandpass_filter`` is True.
    This parameter is used in conjunction with ``lower_bpf`` and ``upper_bpf``.
"""

docdict["motion_filter_type"] = """
motion_filter_type : {None, "lp", "notch"}
    Type of band-stop filter to use for removing respiratory artifact from motion regressors.
    If None, no filter will be applied.
"""

docdict["motion_filter_order"] = """
motion_filter_order : :obj:`int`
    Number of filter coefficients for the band-stop filter, for filtering motion regressors.
    Motion filtering is only performed if ``motion_filter_type`` is not None.
    This parameter is used in conjunction with ``band_stop_max`` and ``band_stop_min``.
"""

docdict["band_stop_min"] = """
band_stop_min : :obj:`float`
    Lower frequency for the band-stop motion filter, in breaths-per-minute (bpm).
    Motion filtering is only performed if ``motion_filter_type`` is not None.
    This parameter is used in conjunction with ``motion_filter_order`` and ``band_stop_max``.
"""

docdict["band_stop_max"] = """
band_stop_max : :obj:`float`
    Upper frequency for the band-stop motion filter, in breaths-per-minute (bpm).
    Motion filtering is only performed if ``motion_filter_type`` is not None.
    This parameter is used in conjunction with ``motion_filter_order`` and ``band_stop_min``.
"""

docdict["name"] = """
name : :obj:`str`, optional
    Name of the workflow. This is used for working directories and workflow graphs.
"""

docdict_indented = {}


def _indentcount_lines(lines):
    """Minimum indent for all lines in line list.

    >>> lines = [' one', '  two', '   three']
    >>> _indentcount_lines(lines)
    1
    >>> lines = []
    >>> _indentcount_lines(lines)
    0
    >>> lines = [' one']
    >>> _indentcount_lines(lines)
    1
    >>> _indentcount_lines(['    '])
    0

    """
    indentno = sys.maxsize
    for line in lines:
        stripped = line.lstrip()
        if stripped:
            indentno = min(indentno, len(line) - len(stripped))
    if indentno == sys.maxsize:
        return 0
    return indentno


def fill_doc(f):
    """Fill a docstring with docdict entries.

    Parameters
    ----------
    f : callable
        The function to fill the docstring of. Will be modified in place.

    Returns
    -------
    f : callable
        The function, potentially with an updated ``__doc__``.

    """
    docstring = f.__doc__
    if not docstring:
        return f
    lines = docstring.splitlines()
    # Find the minimum indent of the main docstring, after first line
    if len(lines) < 2:
        icount = 0
    else:
        icount = _indentcount_lines(lines[1:])
    # Insert this indent to dictionary docstrings
    try:
        indented = docdict_indented[icount]
    except KeyError:
        indent = ' ' * icount
        docdict_indented[icount] = indented = {}
        for name, dstr in docdict.items():
            lines = dstr.splitlines()
            try:
                newlines = [lines[0]]
                for line in lines[1:]:
                    newlines.append(indent + line)
                indented[name] = '\n'.join(newlines)
            except IndexError:
                indented[name] = dstr
    try:
        f.__doc__ = docstring % indented
    except (TypeError, ValueError, KeyError) as exp:
        funcname = f.__name__
        funcname = docstring.split('\n')[0] if funcname is None else funcname
        raise RuntimeError('Error documenting %s:\n%s'
                           % (funcname, str(exp)))
    return f
