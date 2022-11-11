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

docdict[
    "omp_nthreads"
] = """
omp_nthreads : :obj:`int`
    Maximum number of threads an individual process may use.
"""

docdict[
    "mem_gb"
] = """
mem_gb : :obj:`float`
    Memory limit, in gigabytes.
"""

docdict[
    "fmri_dir"
] = """
fmri_dir : :obj:`str`
    Path to the preprocessed derivatives dataset.
    For example, "/path/to/dset/derivatives/fmriprep/".
"""

docdict[
    "output_dir"
] = """
output_dir : :obj:`str`
    Path to the output directory for ``xcp_d`` derivatives.
    This should not include the ``xcp_d`` folder.
    For example, "/path/to/dset/derivatives/".
"""

docdict[
    "work_dir"
] = """
work_dir : :obj:`str`
    Directory in which to store workflow execution state and temporary files.
"""

docdict[
    "analysis_level"
] = """
analysis_level : {"participant"}
    The analysis level for ``xcp_d``. Must be specified as "participant".
"""

docdict[
    "t1w_to_mni"
] = """
t1w_to_mni : :obj:`str`
    Path to the T1w-to-MNI transform file.
    May be "identity", for testing purposes.
"""

docdict[
    "mni_to_t1w"
] = """
mni_to_t1w : :obj:`str`
    Path to the MNI-to-T1w transform file.
    May be "identity", for testing purposes.
"""

docdict[
    "params"
] = """
params : {"36P", "24P", "27P", "acompcor", "acompcor_gsr", \
          "aroma", "aroma_gsr", "custom"}, optional
    Shorthand for the parameter set to extract from the confounds TSV.
    Default is "36P", most expansive option.
"""

docdict[
    "input_type"
] = """
input_type : {"fmriprep", "dcan", "hcp", "nibabies"}
    The format of the incoming preprocessed BIDS derivatives.
    DCAN- and HCP-format derivatives will automatically be converted to a more
    BIDS-compliant format.
    fMRIPrep and Nibabies derivatives are assumed to be roughly equivalent in terms of
    file organization and naming.
"""

docdict[
    "smoothing"
] = """
smoothing : float
    The full width at half maximum (FWHM), in millimeters,
    of the Gaussian smoothing kernel that will be applied to the post-processed and denoised data.
    ALFF and ReHo outputs will also be smoothing with this kernel.
"""

docdict[
    "head_radius"
] = """
head_radius : float
    Radius of the head, in millimeters, for framewise displacement calculation.

    ``xcp_d``'s default head radius is 50. The recommended value for infants is 35.
"""

docdict[
    "fd_thresh"
] = """
fd_thresh : float
    Framewise displacement threshold for censoring, in millimeters.
    Any framewise displacement values higher than the threshold are flagged as "high motion".
    Default is 0.2 mm.
"""

docdict[
    "bandpass_filter"
] = """
bandpass_filter : :obj:`bool`
    If True, a Butterworth bandpass filter will be applied to the fMRI data after interpolation,
    but before regression.
    If False, bandpass filtering will not be performed.
"""

docdict[
    "lower_bpf"
] = """
lower_bpf : :obj:`float`
    Lower cut-off frequency for the Butterworth bandpass filter, in Hertz.
    The bandpass filter is applied to the fMRI data after post-processing and denoising.
    Bandpass filtering will only be performed if ``bandpass_filter`` is True.
    This parameter is used in conjunction with ``upper_bpf`` and ``bpf_order``.
"""

docdict[
    "upper_bpf"
] = """
upper_bpf : :obj:`float`
    Upper cut-off frequency for the Butterworth bandpass filter, in Hertz.
    The bandpass filter is applied to the fMRI data after post-processing and denoising.
    Bandpass filtering will only be performed if ``bandpass_filter`` is True.
    This parameter is used in conjunction with ``lower_bpf`` and ``bpf_order``.
"""

docdict[
    "bpf_order"
] = """
bpf_order : :obj:`int`
    Number of filter coefficients for Butterworth bandpass filter.
    Bandpass filtering will only be performed if ``bandpass_filter`` is True.
    This parameter is used in conjunction with ``lower_bpf`` and ``upper_bpf``.
"""

docdict[
    "motion_filter_type"
] = """
motion_filter_type : {None, "lp", "notch"}
    Type of band-stop filter to use for removing respiratory artifact from motion regressors.
    If None, no filter will be applied.

    If the filter type is set to "notch", then both ``band_stop_min`` and ``band_stop_max``
    must be defined.
    If the filter type is set to "lp", then only ``band_stop_max`` must be defined.
"""

docdict[
    "motion_filter_order"
] = """
motion_filter_order : :obj:`int`
    Number of filter coefficients for the band-stop filter, for filtering motion regressors.
    Motion filtering is only performed if ``motion_filter_type`` is not None.
    This parameter is used in conjunction with ``band_stop_max`` and ``band_stop_min``.
"""

docdict[
    "band_stop_min"
] = """
band_stop_min : :obj:`float` or None
    Lower frequency for the band-stop motion filter, in breaths-per-minute (bpm).
    Motion filtering is only performed if ``motion_filter_type`` is not None.
    If used with the "lp" ``motion_filter_type``, this parameter essentially corresponds to a
    low-pass filter (the maximum allowed frequency in the filtered data).
    This parameter is used in conjunction with ``motion_filter_order`` and ``band_stop_max``.

    Here is a list of recommended values, based on participant age:

    ================= =================
    Age Range (years) Recommended Value
    ================= =================
    < 1               30
    1 - 2             25
    2 - 6             20
    6 - 12            15
    12 - 18           12
    19 - 65           12
    65 - 80           12
    > 80              10
    ================= =================

    When ``motion_filter_type`` is set to "lp" (low-pass filter), another commonly-used value for
    this parameter is 6 BPM (equivalent to 0.1 Hertz), based on :footcite:t:`gratton2020removal`.
"""

docdict[
    "band_stop_max"
] = """
band_stop_max : :obj:`float` or None
    Upper frequency for the band-stop motion filter, in breaths-per-minute (bpm).
    Motion filtering is only performed if ``motion_filter_type`` is not None.
    This parameter is only used if ``motion-filter-type`` is set to "notch".
    This parameter is used in conjunction with ``motion_filter_order`` and ``band_stop_min``.

    Here is a list of recommended values, based on participant age:

    ================= =================
    Age Range (years) Recommended Value
    ================= =================
    < 1               60
    1 - 2             50
    2 - 6             35
    6 - 12            25
    12 - 18           20
    19 - 65           18
    65 - 80           28
    > 80              30
    ================= =================
"""

docdict[
    "name"
] = """
name : :obj:`str`, optional
    Name of the workflow. This is used for working directories and workflow graphs.
"""

docdict[
    "cifti"
] = """
cifti : :obj:`bool`
    Post-process surface data (CIFTIs) instead of volumetric data (NIFTIs).
    This parameter is overridden when DCAN- or HCP-format data are provided.
    Default is False.
"""

docdict[
    "atlas_names"
] = """
atlas_names : :obj:`list` of :obj:`str`
    A list of atlases used for parcellating the BOLD data.
    The list of atlas names is generated by :func:`xcp_d.utils.atlas.get_atlas_names`.
    The atlases include: "Schaefer117", "Schaefer217", "Schaefer317", "Schaefer417",
    "Schaefer517", "Schaefer617", "Schaefer717", "Schaefer817", "Schaefer917",
    "Schaefer1017", "Glasser", "Gordon", and "subcortical" (Tian).
"""

docdict[
    "timeseries"
] = """
timeseries : :obj:`list` of :obj:`str`
    List of paths to atlas-specific time series files.
"""

docdict[
    "correlations"
] = """
correlations : :obj:`list` of :obj:`str`
    List of paths to atlas-specific ROI-to-ROI correlation files.
"""

docdict[
    "process_surfaces"
] = """
process_surfaces : :obj:`bool`, optional
    If True, a workflow will be run to warp native-space (fsnative) reconstructed cortical
    surfaces (surf.gii files) produced by Freesurfer into standard (fsLR) space.
    These surface files are primarily used for visual quality control.
    By default, this workflow is disabled.

    .. list-table:: The surface files that are generated by the workflow
        :align: left
        :header-rows: 1
        :stub-columns: 1

        * - Filename
          - Description
        * - ``<source_entities>_space-fsLR_den-32k_hemi-<L|R>_pial.surf.gii``
          - The gray matter / pial matter border.
        * - ``<source_entities>_space-fsLR_den-32k_hemi-<L|R>_smoothwm.surf.gii``
          - The smoothed gray matter / white matter border for the cortex.
        * - ``<source_entities>_space-fsLR_den-32k_hemi-<L|R>_midthickness.surf.gii``
          - The midpoints between wm and pial surfaces.
            This is derived from the FreeSurfer graymid
            (``mris_expand`` with distance=0.5 applied to the WM surfs).
        * - ``<source_entities>_space-fsLR_den-32k_hemi-<L|R>_inflated.surf.gii``
          - An inflation of the midthickness surface (useful for visualization).
            This file is only created if the input type is "hcp" or "dcan".
        * - ``<source_entities>_space-fsLR_den-32k_hemi-<L|R>_desc-hcp_midthickness.surf.gii``
          - The midpoints between wm and pial surfaces.
            This is created by averaging the coordinates from the wm and pial surfaces.
        * - ``<source_entities>_space-fsLR_den-32k_hemi-<L|R>_desc-hcp_inflated.surf.gii``
          - An inflation of the midthickness surface (useful for visualization).
            This is derived from the HCP midthickness file.
            This file is only created if the input type is "fmriprep" or "nibabies".
        * - ``<source_entities>_space-fsLR_den-32k_hemi-<L|R>_desc-hcp_vinflated.surf.gii``
          - A very-inflated midthicknesss surface (also for visualization).
            This is derived from the HCP midthickness file.
            This file is only created if the input type is "fmriprep" or "nibabies".
"""

docdict[
    "subject_id"
] = """
subject_id : :obj:`str`
    The participant ID. This SHOULD NOT include the ``sub-`` prefix.
"""

docdict[
    "layout"
] = """
layout : :obj:`bids.layout.BIDSLayout`
    BIDSLayout indexing the ingested (e.g., fMRIPrep-format) derivatives.
"""

docdict[
    "dummytime"
] = """
dummytime : :obj:`float`
    Number of seconds to remove from the beginning of each run.
    This parameter is deprecated. Please use ``dummy_scans`` instead.
    This parameter will only take effect if ``dummy_scans`` is 0.
"""

docdict[
    "dummy_scans"
] = """
dummy_scans : :obj:`int` or "auto"
    Number of volumes to remove from the beginning of each run.
    If set to 'auto', xcp_d will extract non-steady-state volume indices from the
    preprocessing derivatives' confounds file.
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
        indent = " " * icount
        docdict_indented[icount] = indented = {}
        for name, dstr in docdict.items():
            lines = dstr.splitlines()
            try:
                newlines = [lines[0]]
                for line in lines[1:]:
                    newlines.append(indent + line)
                indented[name] = "\n".join(newlines)
            except IndexError:
                indented[name] = dstr
    try:
        f.__doc__ = docstring % indented
    except (TypeError, ValueError, KeyError) as exp:
        funcname = f.__name__
        funcname = docstring.split("\n")[0] if funcname is None else funcname
        raise RuntimeError(f"Error documenting {funcname}:\n{str(exp)}")
    return f
