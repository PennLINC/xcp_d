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
