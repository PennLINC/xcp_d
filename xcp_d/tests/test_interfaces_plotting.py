"""Tests for the xcp_d.interfaces.plotting module."""

import os

import numpy as np
import pandas as pd

from xcp_d.interfaces.plotting import CensoringPlot


def test_censoring_plot_marks_censor_between(tmp_path_factory):
    """The censoring plot draws a separate legend entry for censor-between volumes."""
    import matplotlib  # noqa: ICN001

    # Emit legend text as <text> elements rather than glyph references, so it is greppable.
    matplotlib.rcParams['svg.fonttype'] = 'none'

    tmpdir = tmp_path_factory.mktemp('test_censoring_plot_censor_between')
    n_volumes = 50

    rng = np.random.default_rng(0)
    motion_df = pd.DataFrame({'framewise_displacement': rng.random(n_volumes) * 0.5})
    motion_file = os.path.join(tmpdir, 'motion.tsv')
    motion_df.to_csv(motion_file, sep='\t', index=False)

    fd_arr = np.zeros(n_volumes, dtype=int)
    fd_arr[[10, 13]] = 1
    between_arr = np.zeros(n_volumes, dtype=int)
    between_arr[11:13] = 1
    censoring_df = pd.DataFrame(
        {
            'framewise_displacement': fd_arr,
            'censor_between': between_arr,
            'denoising': ((fd_arr + between_arr) > 0).astype(int),
        }
    )
    temporal_mask = os.path.join(tmpdir, 'tmask.tsv')
    censoring_df.to_csv(temporal_mask, sep='\t', index=False)

    interface = CensoringPlot(
        motion_file=motion_file,
        temporal_mask=temporal_mask,
        dummy_scans=0,
        TR=2.0,
        head_radius=50,
        motion_filter_type=None,
        fd_thresh=0.3,
    )
    results = interface.run(cwd=tmpdir)
    assert os.path.isfile(results.outputs.out_file)

    with open(results.outputs.out_file) as fo:
        svg_text = fo.read()

    assert 'Censor-Between Volumes' in svg_text
    assert 'Motion-Censored Volumes' in svg_text
