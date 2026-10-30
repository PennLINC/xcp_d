"""Tests for the xcp_d.utils.modified_data module."""

import numpy as np
import pytest

from xcp_d.utils.modified_data import censor_between_outliers


def test_censor_between_outliers_zero_is_noop():
    """A censor_between of 0 flags nothing."""
    outlier_mask = np.array([1, 0, 0, 1, 0, 1])
    result = censor_between_outliers(outlier_mask, 0)
    assert np.array_equal(result, np.zeros(6, dtype=int))


def test_censor_between_outliers_interior_run():
    """Interior runs are flagged when short enough and kept when not."""
    # One 2-volume run and one 3-volume run, both bounded by outliers.
    outlier_mask = np.array([1, 0, 0, 1, 0, 0, 0, 1])

    # A 2-volume run is flagged; the 3-volume run is not.
    assert np.array_equal(
        censor_between_outliers(outlier_mask, 2),
        np.array([0, 1, 1, 0, 0, 0, 0, 0]),
    )
    # At 3, both runs are flagged.
    assert np.array_equal(
        censor_between_outliers(outlier_mask, 3),
        np.array([0, 1, 1, 0, 1, 1, 1, 0]),
    )
    # At 1, neither run is flagged.
    assert np.array_equal(
        censor_between_outliers(outlier_mask, 1),
        np.zeros(8, dtype=int),
    )


def test_censor_between_outliers_boundary_length():
    """A run of exactly censor_between is flagged; censor_between + 1 is kept."""
    outlier_mask = np.array([1, 0, 0, 0, 1])
    assert censor_between_outliers(outlier_mask, 3).sum() == 3
    assert censor_between_outliers(outlier_mask, 2).sum() == 0


def test_censor_between_outliers_edges():
    """Run boundaries count as outliers, so leading and trailing runs are flagged."""
    outlier_mask = np.array([0, 0, 1, 0, 0, 0, 0, 1, 0, 0])
    result = censor_between_outliers(outlier_mask, 2)
    # Leading 2-volume run and trailing 2-volume run flagged; interior 4-volume run kept.
    assert np.array_equal(result, np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 1]))


def test_censor_between_outliers_all_outliers():
    """With no non-outlier volumes there are no runs to flag."""
    outlier_mask = np.ones(5, dtype=int)
    assert np.array_equal(censor_between_outliers(outlier_mask, 3), np.zeros(5, dtype=int))


def test_censor_between_outliers_no_outliers():
    """With no outliers the whole series is one run, flagged only if it is short enough."""
    outlier_mask = np.zeros(5, dtype=int)
    assert np.array_equal(censor_between_outliers(outlier_mask, 4), np.zeros(5, dtype=int))
    assert np.array_equal(censor_between_outliers(outlier_mask, 5), np.ones(5, dtype=int))
    assert np.array_equal(censor_between_outliers(outlier_mask, 6), np.ones(5, dtype=int))


def test_censor_between_outliers_empty():
    """An empty mask produces an empty result."""
    result = censor_between_outliers(np.array([], dtype=int), 3)
    assert result.size == 0


def test_censor_between_outliers_disjoint_and_shaped():
    """The result is int, the same length as the input, and disjoint from it."""
    rng = np.random.default_rng(0)
    outlier_mask = (rng.random(200) > 0.7).astype(int)
    result = censor_between_outliers(outlier_mask, 3)
    assert result.shape == outlier_mask.shape
    assert np.issubdtype(result.dtype, np.integer)
    assert np.all((result + outlier_mask) <= 1)


def test_censor_between_outliers_accepts_bool():
    """A boolean input mask is handled the same as an integer one."""
    bool_mask = np.array([True, False, False, True])
    int_mask = bool_mask.astype(int)
    assert np.array_equal(
        censor_between_outliers(bool_mask, 2),
        censor_between_outliers(int_mask, 2),
    )


@pytest.mark.parametrize('censor_between', [1, 2, 5])
def test_censor_between_outliers_retained_runs_are_long_enough(censor_between):
    """After expansion, every retained run is longer than censor_between."""
    rng = np.random.default_rng(1)
    outlier_mask = (rng.random(300) > 0.8).astype(int)
    denoising = (
        (outlier_mask + censor_between_outliers(outlier_mask, censor_between)) > 0
    ).astype(int)
    padded = np.concatenate(([1], denoising, [1]))
    diffs = np.diff(padded)
    run_lengths = np.flatnonzero(diffs == 1) - np.flatnonzero(diffs == -1)
    assert np.all(run_lengths > censor_between)


def test_flag_bad_run_accounts_for_censor_between(tmp_path_factory):
    """Post-scrubbing duration shrinks as censor_between grows."""
    import os

    import pandas as pd

    from xcp_d.utils.modified_data import flag_bad_run

    tmpdir = tmp_path_factory.mktemp('test_flag_bad_run_censor_between')
    n_volumes = 100
    t_r = 2.0

    # Build a motion file whose FD alternates between low and high in a way that leaves
    # several short runs of retained volumes.
    rng = np.random.default_rng(0)
    motion_df = pd.DataFrame(
        {
            'trans_x': rng.random(n_volumes) * 0.01,
            'trans_y': rng.random(n_volumes) * 0.01,
            'trans_z': rng.random(n_volumes) * 0.01,
            'rot_x': rng.random(n_volumes) * 0.001,
            'rot_y': rng.random(n_volumes) * 0.001,
            'rot_z': rng.random(n_volumes) * 0.001,
            'rmsd': rng.random(n_volumes) * 0.01,
        }
    )
    # Inject large jumps every third volume so that outliers are scattered.
    motion_df.loc[::3, 'trans_x'] += 2.0
    motion_file = os.path.join(tmpdir, 'motion.tsv')
    motion_df.to_csv(motion_file, sep='\t', index=False)

    kwargs = {
        'motion_file': motion_file,
        'dummy_scans': 0,
        'TR': t_r,
        'motion_filter_type': None,
        'motion_filter_order': None,
        'band_stop_min': None,
        'band_stop_max': None,
        'head_radius': 50,
        'fd_thresh': 0.3,
    }

    baseline = flag_bad_run(**kwargs, censor_between=0)
    expanded = flag_bad_run(**kwargs, censor_between=3)

    assert expanded < baseline
    assert expanded >= 0

    # Censoring disabled short-circuits regardless of censor_between.
    assert flag_bad_run(**{**kwargs, 'fd_thresh': 0}, censor_between=3) == np.inf
