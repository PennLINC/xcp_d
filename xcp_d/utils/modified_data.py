# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Functions for interpolating over high-motion volumes."""
import numpy as np

from xcp_d.utils.doc import fill_doc


@fill_doc
def compute_fd(confound, head_radius=50):
    """Compute framewise displacement.

    NOTE: TS- Which kind of FD? Power?
    NOTE: TS- What units must rotation parameters be in?

    Parameters
    ----------
    confound : pandas.DataFrame
        DataFrame with six motion parameters.
    %(head_radius)s

    Returns
    -------
    fdres : numpy.ndarray
        The framewise displacement time series.
    """
    confound = confound.replace(np.nan, 0)
    mpars = confound[["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]].to_numpy()
    diff = mpars[:-1, :6] - mpars[1:, :6]
    diff[:, 3:6] *= head_radius
    fd_res = np.abs(diff).sum(axis=1)
    fdres = np.hstack([0, fd_res])

    return fdres


@fill_doc
def generate_mask(fd_res, fd_thresh):
    """Create binary temporal mask flagging high-motion volumes.

    Parameters
    ----------
    fd_res : numpy.ndarray of shape (T)
        Framewise displacement time series.
        T = time.
    %(fd_thresh)s

    Returns
    -------
    tmask : numpy.ndarray of shape (T)
        The temporal mask. Zeros are low-motion volumes. Ones are high-motion volumes.
    """
    tmask = np.zeros(len(fd_res), dtype=int)
    tmask[fd_res > fd_thresh] = 1

    return tmask
