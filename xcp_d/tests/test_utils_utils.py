"""Test functions in xcp_d.utils.utils."""

import numpy as np
import pandas as pd
import pytest
from scipy import signal, stats

from xcp_d.utils import utils


def test_estimate_brain_radius(ds001419_data):
    """Ensure that the brain radius estimation function returns the right value."""
    bold_mask = ds001419_data["brain_mask_file"]

    radius = utils.estimate_brain_radius(bold_mask, head_radius="auto")
    assert radius == 77.39268749395897

    radius = utils.estimate_brain_radius(bold_mask, head_radius=50)
    assert radius == 50


def test_denoise_with_nilearn():
    """Test xcp_d.utils.utils.denoise_with_nilearn."""
    high_pass, low_pass, filter_order, TR = 0.01, 0.08, 2, 2

    n_voxels, n_volumes, n_signals, n_confounds = 100, 10000, 2, 5

    data_arr = np.zeros((n_volumes, n_voxels))

    # Create signals and add them to the data
    rng = np.random.default_rng(0)
    signal_timeseries = rng.standard_normal(size=(n_volumes, n_signals))
    signal_timeseries = signal.detrend(signal_timeseries, axis=0)
    signal_timeseries = stats.zscore(signal_timeseries, axis=0)
    signal_weights = rng.standard_normal(size=(n_signals, n_voxels))

    for i_signal in range(n_signals):
        # The first n_signals voxels are only affected by the corresponding signal
        signal_weights[:, i_signal] = 0
        signal_weights[i_signal, i_signal] = 1

    signal_arr = np.dot(signal_timeseries, signal_weights)
    data_arr += signal_arr

    # Check that signals are present in the "raw" data
    sample_mask = np.ones(n_volumes, dtype=bool)
    _check_signal(data_arr, signal_timeseries, sample_mask, atol=0.01)

    # Create confounds and add them to the data
    rng = np.random.default_rng(1)
    confound_timeseries = rng.standard_normal(size=(n_volumes, n_confounds))
    confound_timeseries = signal.detrend(confound_timeseries, axis=0)

    # Orthogonalize the confound_timeseries w.r.t. the signal_timeseries
    signal_betas = np.linalg.lstsq(signal_timeseries, confound_timeseries, rcond=None)[0]
    pred_confound_timeseries = np.dot(signal_timeseries, signal_betas)
    confound_timeseries = confound_timeseries - pred_confound_timeseries

    confound_timeseries = stats.zscore(confound_timeseries, axis=0)
    confound_weights = rng.standard_normal(size=(n_confounds, n_voxels))
    for i_confound in range(n_confounds):
        # The first n_confounds + n_signals voxels are only affected by the corresponding confound
        confound_weights[:, i_confound + n_signals] = 0
        confound_weights[i_confound, i_confound + n_signals] = 1

    confound_arr = np.dot(confound_timeseries, confound_weights)
    data_arr += confound_arr
    confounds_df = pd.DataFrame(
        confound_timeseries,
        columns=[f"confound_{i}" for i in range(n_confounds)],
    )

    # Check that signals are present in the "raw" data at this point
    sample_mask = np.ones(n_volumes, dtype=bool)
    _check_signal(data_arr, signal_timeseries, sample_mask, atol=0.01)

    # Now add trends
    rng = np.random.default_rng(2)
    trend = np.arange(n_volumes).astype(np.float32)
    trend -= trend.mean()
    trend_weights = rng.standard_normal(size=(1, n_voxels))
    data_arr += np.dot(trend[:, np.newaxis], trend_weights)
    orig_data_arr = data_arr.copy()
    orig_signal_timeseries = signal_timeseries.copy()

    # Check that signals are still present
    sample_mask = np.ones(n_volumes, dtype=bool)
    _check_signal(data_arr, signal_timeseries, sample_mask, atol=0.01)

    # Check that censoring doesn't cause any obvious problems
    sample_mask = np.ones(n_volumes, dtype=bool)
    sample_mask[10:20] = False
    sample_mask[150:160] = False
    _check_signal(data_arr, signal_timeseries, sample_mask, atol=0.01)

    # First, try out filtering without censoring or denoising
    params = {
        "confounds": None,
        "voxelwise_confounds": None,
        "sample_mask": np.ones(n_volumes, dtype=bool),
        "low_pass": low_pass,
        "high_pass": high_pass,
        "filter_order": filter_order,
        "TR": TR,
    }
    out_arr = utils.denoise_with_nilearn(preprocessed_bold=data_arr, **params)
    assert out_arr.shape == (n_volumes, n_voxels)
    assert not np.allclose(out_arr, data_arr)  # data aren't modified

    # Now, no filtering (censoring + denoising + interpolation)
    # With 10000 volumes, censoring shouldn't have much impact on the betas
    sample_mask = np.ones(n_volumes, dtype=bool)
    sample_mask[10:20] = False
    sample_mask[150:160] = False
    params = {
        "confounds": confounds_df,
        "voxelwise_confounds": None,
        "sample_mask": sample_mask,
        "low_pass": None,
        "high_pass": None,
        "filter_order": 0,
        "TR": TR,
    }
    out_arr = utils.denoise_with_nilearn(preprocessed_bold=data_arr, **params)
    assert out_arr.shape == (n_volumes, n_voxels)

    assert np.array_equal(data_arr, orig_data_arr)
    assert np.array_equal(signal_timeseries, orig_signal_timeseries)
    _check_trend(out_arr, trend, sample_mask)
    _check_confounds(out_arr, confound_timeseries, sample_mask)
    _check_signal(data_arr, signal_timeseries, sample_mask)
    _check_signal(out_arr, signal_timeseries, sample_mask)

    # Denoise without censoring or filtering
    params = {
        "confounds": confounds_df,
        "voxelwise_confounds": None,
        "sample_mask": np.ones(n_volumes, dtype=bool),
        "low_pass": None,
        "high_pass": None,
        "filter_order": 0,
        "TR": TR,
    }
    out_arr = utils.denoise_with_nilearn(preprocessed_bold=data_arr, **params)
    assert out_arr.shape == (n_volumes, n_voxels)
    assert not np.allclose(out_arr, data_arr)  # data aren't modified
    _check_trend(out_arr, trend, sample_mask)
    _check_confounds(out_arr, confound_timeseries, sample_mask)
    _check_signal(data_arr, signal_timeseries, sample_mask)
    _check_signal(out_arr, signal_timeseries, sample_mask)

    # Run without denoising (censoring + interpolation + filtering)
    sample_mask = np.ones(n_volumes, dtype=bool)
    sample_mask[10:20] = False
    sample_mask[150:160] = False
    params = {
        "confounds": None,
        "voxelwise_confounds": None,
        "sample_mask": sample_mask,
        "low_pass": low_pass,
        "high_pass": high_pass,
        "filter_order": filter_order,
        "TR": TR,
    }
    out_arr = utils.denoise_with_nilearn(preprocessed_bold=data_arr, **params)
    assert out_arr.shape == (n_volumes, n_voxels)
    assert not np.allclose(out_arr, data_arr)  # data aren't modified

    # signals were retained
    _check_signal(data_arr, signal_timeseries, sample_mask)
    # XXX: I don't like how high the atol is here, but it's not terrible
    filtered_signals = utils.denoise_with_nilearn(preprocessed_bold=signal_timeseries, **params)
    _check_signal(out_arr, filtered_signals, sample_mask, atol=0.06)

    # Ensure that interpolation + filtering doesn't cause problems at beginning/end of scan
    # Create an updated censoring file with outliers at first and last two volumes
    sample_mask = np.ones(n_volumes, dtype=bool)
    sample_mask[:2] = False
    sample_mask[-2:] = False
    n_censored_volumes = np.sum(~sample_mask)
    assert n_censored_volumes == 4

    # Run without denoising or filtering (censoring + interpolation only)
    params = {
        "confounds": None,
        "voxelwise_confounds": None,
        "sample_mask": sample_mask,
        "low_pass": None,
        "high_pass": None,
        "filter_order": 0,
        "TR": TR,
    }
    out_arr = utils.denoise_with_nilearn(preprocessed_bold=data_arr, **params)
    assert out_arr.shape == (n_volumes, n_voxels)
    assert not np.allclose(out_arr, data_arr)  # data aren't modified
    # The first two volumes should be the same as the third (first non-outlier) volume
    assert np.allclose(out_arr[0, :], out_arr[2, :])
    assert np.allclose(out_arr[1, :], out_arr[2, :])
    assert not np.allclose(out_arr[2, :], out_arr[3, :])
    # The last volume should be the same as the third-to-last (last non-outlier) volume
    assert np.allclose(out_arr[-1, :], out_arr[-3, :])
    assert np.allclose(out_arr[-2, :], out_arr[-3, :])
    assert not np.allclose(out_arr[-3, :], out_arr[-4, :])

    # signals were retained
    filtered_signals = utils.denoise_with_nilearn(preprocessed_bold=signal_timeseries, **params)

    _check_signal(data_arr, signal_timeseries, sample_mask)
    _check_signal(out_arr, filtered_signals, sample_mask)


def test_denoise_with_nilearn_voxelwise():
    """Test xcp_d.utils.utils.denoise_with_nilearn with voxel-wise regressors.

    Just a smoke test.
    """
    high_pass, low_pass, filter_order, TR = 0.01, 0.08, 2, 2
    n_voxels, n_volumes, n_confounds, n_voxelwise_confounds = 1000, 300, 5, 3
    data_arr = np.random.random((n_volumes, n_voxels))
    confounds = np.random.random((n_volumes, n_confounds))
    confounds_df = pd.DataFrame(
        confounds,
        columns=[f"confound_{i}" for i in range(n_confounds)],
    )
    voxelwise_confounds = [
        np.random.random((n_volumes, n_voxels)) for _ in range(n_voxelwise_confounds)
    ]
    sample_mask = np.ones(n_volumes, dtype=bool)
    sample_mask[40:60] = False

    # Denoising with bandpass filtering and censoring
    params = {
        "confounds": confounds_df,
        "voxelwise_confounds": voxelwise_confounds,
        "sample_mask": sample_mask,
        "low_pass": low_pass,
        "high_pass": high_pass,
        "filter_order": filter_order,
        "TR": TR,
    }
    out_arr = utils.denoise_with_nilearn(preprocessed_bold=data_arr, **params)
    assert out_arr.shape == (n_volumes, n_voxels)

    # Denoising without bandpass filtering
    params = {
        "confounds": confounds_df,
        "voxelwise_confounds": voxelwise_confounds,
        "sample_mask": sample_mask,
        "low_pass": None,
        "high_pass": None,
        "filter_order": None,
        "TR": TR,
    }
    out_arr = utils.denoise_with_nilearn(preprocessed_bold=data_arr, **params)
    assert out_arr.shape == (n_volumes, n_voxels)

    # Denoising with bandpass filtering but no general confounds
    params = {
        "confounds": None,
        "voxelwise_confounds": voxelwise_confounds,
        "sample_mask": sample_mask,
        "low_pass": low_pass,
        "high_pass": high_pass,
        "filter_order": filter_order,
        "TR": TR,
    }
    out_arr = utils.denoise_with_nilearn(preprocessed_bold=data_arr, **params)
    assert out_arr.shape == (n_volumes, n_voxels)

    # Denoising without bandpass filtering or general confounds
    params = {
        "confounds": None,
        "voxelwise_confounds": voxelwise_confounds,
        "sample_mask": sample_mask,
        "low_pass": None,
        "high_pass": None,
        "filter_order": None,
        "TR": TR,
    }
    out_arr = utils.denoise_with_nilearn(preprocessed_bold=data_arr, **params)
    assert out_arr.shape == (n_volumes, n_voxels)


def _check_trend(data, trend, sample_mask, atol=0.01):
    """Ensure that the trend was removed by the denoising process."""
    trend_corr = np.corrcoef(trend[sample_mask], data[sample_mask, :].T)[0, 1:]
    assert all(np.abs(trend_corr) < atol)


def _check_confounds(data, confounds, sample_mask, atol=0.01):
    """Ensure that confounds were removed by the denoising process."""
    for i_confound in range(confounds.shape[1]):
        confound_corr = np.corrcoef(
            confounds[sample_mask, i_confound],
            data[sample_mask, :].T,
        )[0, 1:]
        assert all(np.abs(confound_corr) < atol)


def _check_signal(data, signals, sample_mask, atol=0.001):
    """Ensure that signals were retained by the denoising process."""
    signal_betas = []

    # Add constant and linear trend
    dm = np.hstack(
        (
            signals,
            np.ones((signals.shape[0], 1)),
            np.arange(signals.shape[0])[:, np.newaxis],
        ),
    )
    dm = dm[sample_mask, :]
    for i_signal in range(signals.shape[1]):
        res = np.linalg.lstsq(
            dm,
            data[sample_mask, i_signal][:, np.newaxis],
            rcond=None,
        )
        signal_beta = res[0][i_signal]
        signal_betas.append(signal_beta)

    assert np.allclose(signal_betas, 1.0, atol=atol)


def test_list_to_str():
    """Test the list_to_str function."""
    string = utils.list_to_str(["a"])
    assert string == "a"

    string = utils.list_to_str(["a", "b"])
    assert string == "a and b"

    string = utils.list_to_str(["a", "b", "c"])
    assert string == "a, b, and c"

    with pytest.raises(ValueError, match="Zero-length list provided."):
        utils.list_to_str([])


def test_get_bold2std_and_t1w_xfms(ds001419_data):
    """Test get_bold2std_and_t1w_xfms."""
    bold_file_nlin6asym = ds001419_data["nifti_file"]
    nlin6asym_to_anat_xfm = ds001419_data["template_to_anat_xfm"]

    # MNI152NLin6Asym --> MNI152NLin2009cAsym/T1w
    (
        xforms_to_mni,
        xforms_to_mni_invert,
        xforms_to_t1w,
        xforms_to_t1w_invert,
    ) = utils.get_bold2std_and_t1w_xfms(
        bold_file_nlin6asym,
        nlin6asym_to_anat_xfm,
    )
    assert len(xforms_to_mni) == 1
    assert len(xforms_to_mni_invert) == 1
    assert len(xforms_to_t1w) == 1
    assert len(xforms_to_t1w_invert) == 1

    # MNI152NLin2009cAsym --> MNI152NLin2009cAsym/T1w
    bold_file_nlin2009c = bold_file_nlin6asym.replace(
        "space-MNI152NLin6Asym_",
        "space-MNI152NLin2009cAsym_",
    )
    nlin2009c_to_anat_xfm = nlin6asym_to_anat_xfm.replace(
        "from-MNI152NLin6Asym_",
        "from-MNI152NLin2009cAsym_",
    )
    (
        xforms_to_mni,
        xforms_to_mni_invert,
        xforms_to_t1w,
        xforms_to_t1w_invert,
    ) = utils.get_bold2std_and_t1w_xfms(
        bold_file_nlin2009c,
        nlin2009c_to_anat_xfm,
    )
    assert len(xforms_to_mni) == 1
    assert len(xforms_to_mni_invert) == 1
    assert len(xforms_to_t1w) == 1
    assert len(xforms_to_t1w_invert) == 1

    # MNIInfant --> MNI152NLin2009cAsym/T1w
    bold_file_infant = bold_file_nlin6asym.replace(
        "space-MNI152NLin6Asym_",
        "space-MNIInfant_cohort-1_",
    )
    infant_to_anat_xfm = nlin6asym_to_anat_xfm.replace(
        "from-MNI152NLin6Asym_",
        "from-MNIInfant+1_",
    )
    (
        xforms_to_mni,
        xforms_to_mni_invert,
        xforms_to_t1w,
        xforms_to_t1w_invert,
    ) = utils.get_bold2std_and_t1w_xfms(
        bold_file_infant,
        infant_to_anat_xfm,
    )
    assert len(xforms_to_mni) == 1
    assert len(xforms_to_mni_invert) == 1
    assert len(xforms_to_t1w) == 1
    assert len(xforms_to_t1w_invert) == 1

    # T1w --> MNI152NLin2009cAsym/T1w
    bold_file_t1w = bold_file_nlin6asym.replace("space-MNI152NLin6Asym_", "space-T1w_")
    with pytest.raises(ValueError, match="BOLD space 'T1w' not supported."):
        utils.get_bold2std_and_t1w_xfms(
            bold_file_t1w,
            nlin6asym_to_anat_xfm,
        )

    # T1w --> MNI152NLin6Asym --> MNI152NLin2009cAsym/T1w
    bold_file_t1w = bold_file_nlin6asym.replace("space-MNI152NLin6Asym_", "space-T1w_")
    with pytest.raises(ValueError, match="BOLD space 'T1w' not supported."):
        utils.get_bold2std_and_t1w_xfms(
            bold_file_t1w,
            nlin6asym_to_anat_xfm,
        )

    # native --> MNI152NLin2009cAsym/T1w
    bold_file_native = bold_file_nlin6asym.replace("space-MNI152NLin6Asym_", "")
    with pytest.raises(ValueError, match="BOLD space 'native' not supported."):
        utils.get_bold2std_and_t1w_xfms(
            bold_file_native,
            nlin6asym_to_anat_xfm,
        )

    # native --> MNI152NLin6Asym --> MNI152NLin2009cAsym/T1w
    bold_file_native = bold_file_nlin6asym.replace("space-MNI152NLin6Asym_", "")
    with pytest.raises(ValueError, match="BOLD space 'native' not supported."):
        utils.get_bold2std_and_t1w_xfms(
            bold_file_native,
            nlin6asym_to_anat_xfm,
        )

    # tofail --> MNI152NLin2009cAsym/T1w
    bold_file_tofail = bold_file_nlin6asym.replace("space-MNI152NLin6Asym_", "space-tofail_")
    with pytest.raises(ValueError, match="Transform does not match BOLD space"):
        utils.get_bold2std_and_t1w_xfms(
            bold_file_tofail,
            nlin6asym_to_anat_xfm,
        )

    tofail_to_anat_xfm = nlin6asym_to_anat_xfm.replace("from-MNI152NLin6Asym_", "from-tofail_")
    with pytest.raises(ValueError, match="Space 'tofail'"):
        utils.get_bold2std_and_t1w_xfms(
            bold_file_tofail,
            tofail_to_anat_xfm,
        )


def test_get_std2bold_xfms(ds001419_data):
    """Test get_std2bold_xfms.

    get_std2bold_xfms finds transforms to go from a source file's space to the BOLD file's space.
    """
    bold_file_nlin6asym = ds001419_data["nifti_file"]

    # MNI152NLin6Asym --> MNI152NLin6Asym with source file containing tpl entity
    xforms_to_mni = utils.get_std2bold_xfms(
        bold_file_nlin6asym,
        source_file="tpl-MNI152NLin6Asym_T1w.nii.gz",
        source_space=None,
    )
    assert len(xforms_to_mni) == 1

    # MNI152NLin6Asym --> MNI152NLin6Asym with source file containing space entity
    xforms_to_mni = utils.get_std2bold_xfms(
        bold_file_nlin6asym,
        source_file="space-MNI152NLin6Asym_T1w.nii.gz",
        source_space=None,
    )
    assert len(xforms_to_mni) == 1

    SPACES = [
        ("MNI152NLin6Asym", "MNI152NLin6Asym", 1),
        ("MNI152NLin6Asym", "MNI152NLin2009cAsym", 1),
        ("MNI152NLin6Asym", "MNIInfant", 2),
        ("MNI152NLin2009cAsym", "MNI152NLin2009cAsym", 1),
        ("MNI152NLin2009cAsym", "MNI152NLin6Asym", 1),
        ("MNI152NLin2009cAsym", "MNIInfant", 1),
        ("MNIInfant", "MNIInfant", 1),
        ("MNIInfant", "MNI152NLin2009cAsym", 1),
        ("MNIInfant", "MNI152NLin6Asym", 2),
    ]
    for space_check in SPACES:
        target_space, source_space, n_xforms = space_check
        bold_file_target_space = bold_file_nlin6asym.replace(
            "space-MNI152NLin6Asym_",
            f"space-{target_space}_",
        )
        xforms_to_mni = utils.get_std2bold_xfms(
            bold_file_target_space,
            source_file=None,
            source_space=source_space,
        )
        assert len(xforms_to_mni) == n_xforms

    # Outside of the supported spaces, we expect an error
    # No space or tpl entity in source file
    with pytest.raises(ValueError, match="Unknown space"):
        utils.get_std2bold_xfms(bold_file_nlin6asym, source_file="T1w.nii.gz", source_space=None)

    # MNI152NLin6Asym --> tofail
    bold_file_tofail = bold_file_nlin6asym.replace("space-MNI152NLin6Asym_", "space-tofail_")
    with pytest.raises(ValueError, match="BOLD space 'tofail' not supported"):
        utils.get_std2bold_xfms(bold_file_tofail, source_file=None, source_space="MNI152NLin6Asym")

    # tofail --> MNI152NLin6Asym
    with pytest.raises(ValueError, match="Source space 'tofail' not supported"):
        utils.get_std2bold_xfms(bold_file_nlin6asym, source_file=None, source_space="tofail")


def test_fwhm2sigma():
    """Test fwhm2sigma."""
    fwhm = 8
    sigma = utils.fwhm2sigma(fwhm)
    assert np.allclose(sigma, 3.39728)


def test_select_first():
    """Test _select_first."""
    lst = ["a", "b", "c"]
    assert utils._select_first(lst) == "a"

    lst = "abc"
    assert utils._select_first(lst) == "a"


def test_listify():
    """Test _listify."""
    inputs = [
        1,
        (1,),
        "a",
        ["a"],
        ["a", ["b", "c"]],
        ("a", "b"),
    ]
    outputs = [
        [1],
        (1,),
        ["a"],
        ["a"],
        ["a", ["b", "c"]],
        ("a", "b"),
    ]
    for i, input_ in enumerate(inputs):
        expected_output = outputs[i]
        output = utils._listify(input_)
        assert output == expected_output


def test_make_dictionary():
    """Test _make_dictionary."""
    metadata = {"Sources": ["a"]}
    out_metadata = utils._make_dictionary(metadata, Sources=["b"])
    # Ensure the original dictionary isn't modified.
    assert metadata["Sources"] == ["a"]
    assert out_metadata["Sources"] == ["a", "b"]

    metadata = {"Test": "a"}
    out_metadata = utils._make_dictionary(metadata, Sources=["b"])
    assert out_metadata["Sources"] == ["b"]

    metadata = {"Test": ["a"]}
    out_metadata = utils._make_dictionary(metadata, Sources="b")
    assert out_metadata["Sources"] == "b"

    metadata = {"Sources": "a"}
    out_metadata = utils._make_dictionary(metadata, Sources=["b"])
    # Ensure the original dictionary isn't modified.
    assert metadata["Sources"] == "a"
    assert out_metadata["Sources"] == ["a", "b"]

    metadata = {"Sources": ["a"]}
    out_metadata = utils._make_dictionary(metadata, Sources="b")
    # Ensure the original dictionary isn't modified.
    assert metadata["Sources"] == ["a"]
    assert out_metadata["Sources"] == ["a", "b"]

    out_metadata = utils._make_dictionary(metadata=None, Sources=["b"])
    assert out_metadata["Sources"] == ["b"]


def test_transpose_lol():
    """Test _transpose_lol."""
    inputs = [
        [
            ["a", "b", "c"],
            [1, 2, 3],
        ],
        [
            ["a", "b", "c", "d"],
            [1, 2, 3],
        ],
    ]
    outputs = [
        [
            ["a", 1],
            ["b", 2],
            ["c", 3],
        ],
        [
            ["a", 1],
            ["b", 2],
            ["c", 3],
        ],
    ]
    for i, input_ in enumerate(inputs):
        expected_output = outputs[i]
        assert utils._transpose_lol(input_) == expected_output
