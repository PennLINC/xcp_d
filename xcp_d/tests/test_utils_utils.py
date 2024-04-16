"""Test functions in xcp_d.utils.utils."""

import numpy as np
import pandas as pd
import pytest
from scipy import signal

from xcp_d.utils import utils


def test_estimate_brain_radius(ds001419_data):
    """Ensure that the brain radius estimation function returns the right value."""
    bold_mask = ds001419_data["brain_mask_file"]

    radius = utils.estimate_brain_radius(bold_mask, head_radius="auto")
    assert radius == 78.12350298308195

    radius = utils.estimate_brain_radius(bold_mask, head_radius=50)
    assert radius == 50


def test_butter_bandpass():
    """Test butter_bandpass."""
    n_volumes, n_voxels = 100, 1000
    data = np.random.random((n_volumes, n_voxels))
    sampling_rate = 0.5  # TR of 2 seconds
    filters = [
        {"low_pass": 0.1, "high_pass": 0.01},
        {"low_pass": 0.1, "high_pass": 0},
        {"low_pass": 0, "high_pass": 0.01},
    ]
    for filter in filters:
        filtered_data = utils.butter_bandpass(
            data=data,
            sampling_rate=sampling_rate,
            **filter,
        )
        assert filtered_data.shape == (n_volumes, n_voxels)

    with pytest.raises(ValueError, match="Filter parameters are not valid."):
        utils.butter_bandpass(
            data=data,
            sampling_rate=sampling_rate,
            low_pass=0,
            high_pass=0,
        )


def test_denoise_with_nilearn(ds001419_data, tmp_path_factory):
    """Test xcp_d.utils.utils.denoise_with_nilearn."""
    high_pass, low_pass, filter_order, TR = 0.01, 0.08, 2, 2

    n_voxels, n_volumes, n_signals, n_confounds = 100, 200, 2, 5

    data_arr = np.empty((n_volumes, n_voxels))

    # Create confounds and add them to the data
    rng = np.random.default_rng(0)
    confound_timeseries = rng.standard_normal(size=(n_volumes, n_confounds))
    confound_timeseries = signal.detrend(confound_timeseries, axis=0)
    confound_weights = rng.standard_normal(size=(n_confounds, n_voxels))
    confound_arr = np.dot(confound_timeseries, confound_weights)
    data_arr += confound_arr

    # Create signals and add them to the data
    rng = np.random.default_rng(1)
    signal_timeseries = rng.standard_normal(size=(n_volumes, n_signals))
    signal_timeseries = signal.detrend(signal_timeseries, axis=0)
    signal_weights = rng.standard_normal(size=(n_signals, n_voxels))
    signal_arr = np.dot(signal_timeseries, signal_weights)
    data_arr += signal_arr

    # Now add trends
    rng = np.random.default_rng(2)
    trend = np.arange(n_volumes)
    trend -= trend.mean()
    trend_weights = rng.standard_normal(size=(1, n_voxels))
    data_arr += np.dot(trend[:, np.newaxis], trend_weights)

    confounds_df = pd.DataFrame(
        confound_timeseries,
        columns=[f"confound_{i}" for i in range(n_confounds)],
    )

    # First, try out filtering without censoring or denoising
    out_arr = utils.denoise_with_nilearn(
        preprocessed_bold=data_arr,
        confounds=None,
        sample_mask=np.ones(n_volumes, dtype=bool),
        low_pass=low_pass,
        high_pass=high_pass,
        filter_order=filter_order,
        TR=TR,
    )
    assert out_arr.shape == (n_volumes, n_voxels)

    # Now, no filtering (censoring + denoising + interpolation)
    sample_mask = np.ones(n_volumes, dtype=bool)
    sample_mask[10:20] = False
    sample_mask[150:160] = False
    out_arr = utils.denoise_with_nilearn(
        preprocessed_bold=data_arr,
        confounds=confounds_df,
        sample_mask=sample_mask,
        low_pass=None,
        high_pass=None,
        filter_order=None,
        TR=TR,
    )
    assert out_arr.shape == (n_volumes, n_voxels)

    # Denoise without censoring or filtering
    out_arr = utils.denoise_with_nilearn(
        preprocessed_bold=data_arr,
        confounds=confounds_df,
        sample_mask=np.ones(n_volumes, dtype=bool),
        low_pass=None,
        high_pass=None,
        filter_order=None,
        TR=TR,
    )
    assert out_arr.shape == (n_volumes, n_voxels)
    assert np.allclose(out_arr, signal_arr)

    # Finally, run without denoising (censoring + interpolation + filtering)
    out_arr = utils.denoise_with_nilearn(
        preprocessed_bold=data_arr,
        confounds=None,
        sample_mask=sample_mask,
        low_pass=low_pass,
        high_pass=high_pass,
        filter_order=filter_order,
        TR=TR,
    )
    assert out_arr.shape == (n_volumes, n_voxels)

    # Ensure that interpolation + filtering doesn't cause problems at beginning/end of scan
    # Create an updated censoring file with outliers at first and last two volumes
    sample_mask[:1] = False
    sample_mask[-2:] = False
    n_censored_volumes = np.sum(sample_mask)
    assert n_censored_volumes == 4

    # Run without denoising or filtering (censoring + interpolation only)
    out_arr = utils.denoise_with_nilearn(
        preprocessed_bold=data_arr,
        confounds_file=None,
        sample_mask=sample_mask,
        low_pass=None,
        high_pass=None,
        filter_order=0,
        TR=TR,
    )
    assert out_arr.shape == (n_volumes, n_voxels)
    # The first two volumes should be the same as the third (first non-outlier) volume
    assert np.allclose(out_arr[0, :], out_arr[2, :])
    assert np.allclose(out_arr[1, :], out_arr[2, :])
    assert not np.allclose(out_arr[2, :], out_arr[3, :])
    # The last volume should be the same as the third-to-last (last non-outlier) volume
    assert np.allclose(out_arr[-1, :], out_arr[-3, :])
    assert np.allclose(out_arr[-2, :], out_arr[-3, :])
    assert not np.allclose(out_arr[-3, :], out_arr[-4, :])


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
    bold_file_nlin2009c = ds001419_data["nifti_file"]
    nlin2009c_to_anat_xfm = ds001419_data["template_to_anat_xfm"]

    # MNI152NLin2009cAsym --> MNI152NLin2009cAsym/T1w
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

    # MNI152NLin6Asym --> MNI152NLin2009cAsym/T1w
    bold_file_nlin6asym = bold_file_nlin2009c.replace(
        "space-MNI152NLin2009cAsym_",
        "space-MNI152NLin6Asym_",
    )
    nlin6asym_to_anat_xfm = nlin2009c_to_anat_xfm.replace(
        "from-MNI152NLin2009cAsym_",
        "from-MNI152NLin6Asym_",
    )
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

    # MNIInfant --> MNI152NLin2009cAsym/T1w
    bold_file_infant = bold_file_nlin2009c.replace(
        "space-MNI152NLin2009cAsym_",
        "space-MNIInfant_cohort-1_",
    )
    infant_to_anat_xfm = nlin2009c_to_anat_xfm.replace(
        "from-MNI152NLin2009cAsym_",
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
    bold_file_t1w = bold_file_nlin2009c.replace("space-MNI152NLin2009cAsym_", "space-T1w_")
    with pytest.raises(ValueError, match="BOLD space 'T1w' not supported."):
        utils.get_bold2std_and_t1w_xfms(
            bold_file_t1w,
            nlin2009c_to_anat_xfm,
        )

    # T1w --> MNI152NLin6Asym --> MNI152NLin2009cAsym/T1w
    bold_file_t1w = bold_file_nlin2009c.replace("space-MNI152NLin2009cAsym_", "space-T1w_")
    with pytest.raises(ValueError, match="BOLD space 'T1w' not supported."):
        utils.get_bold2std_and_t1w_xfms(
            bold_file_t1w,
            nlin6asym_to_anat_xfm,
        )

    # native --> MNI152NLin2009cAsym/T1w
    bold_file_native = bold_file_nlin2009c.replace("space-MNI152NLin2009cAsym_", "")
    with pytest.raises(ValueError, match="BOLD space 'native' not supported."):
        utils.get_bold2std_and_t1w_xfms(
            bold_file_native,
            nlin2009c_to_anat_xfm,
        )

    # native --> MNI152NLin6Asym --> MNI152NLin2009cAsym/T1w
    bold_file_native = bold_file_nlin2009c.replace("space-MNI152NLin2009cAsym_", "")
    with pytest.raises(ValueError, match="BOLD space 'native' not supported."):
        utils.get_bold2std_and_t1w_xfms(
            bold_file_native,
            nlin6asym_to_anat_xfm,
        )

    # tofail --> MNI152NLin2009cAsym/T1w
    bold_file_tofail = bold_file_nlin2009c.replace("space-MNI152NLin2009cAsym_", "space-tofail_")
    with pytest.raises(ValueError, match="Transform does not match BOLD space"):
        utils.get_bold2std_and_t1w_xfms(
            bold_file_tofail,
            nlin2009c_to_anat_xfm,
        )

    tofail_to_anat_xfm = nlin2009c_to_anat_xfm.replace("from-MNI152NLin2009cAsym_", "from-tofail_")
    with pytest.raises(ValueError, match="Space 'tofail'"):
        utils.get_bold2std_and_t1w_xfms(
            bold_file_tofail,
            tofail_to_anat_xfm,
        )


def test_get_std2bold_xfms(ds001419_data):
    """Test get_std2bold_xfms.

    get_std2bold_xfms finds transforms to go from the input file's space to MNI152NLin6Asym.
    """
    bold_file_nlin2009c = ds001419_data["nifti_file"]

    # MNI152NLin6Asym --> MNI152NLin2009cAsym
    xforms_to_mni = utils.get_std2bold_xfms(bold_file_nlin2009c)
    assert len(xforms_to_mni) == 1

    # MNI152NLin6Asym --> MNI152NLin6Asym
    bold_file_nlin6asym = bold_file_nlin2009c.replace(
        "space-MNI152NLin2009cAsym_",
        "space-MNI152NLin6Asym_",
    )
    xforms_to_mni = utils.get_std2bold_xfms(bold_file_nlin6asym)
    assert len(xforms_to_mni) == 1

    # MNI152NLin6Asym --> MNIInfant
    bold_file_infant = bold_file_nlin2009c.replace(
        "space-MNI152NLin2009cAsym_",
        "space-MNIInfant_cohort-1_",
    )
    xforms_to_mni = utils.get_std2bold_xfms(bold_file_infant)
    assert len(xforms_to_mni) == 2

    # MNI152NLin6Asym --> tofail
    bold_file_tofail = bold_file_nlin2009c.replace("space-MNI152NLin2009cAsym_", "space-tofail_")
    with pytest.raises(ValueError, match="Space 'tofail'"):
        utils.get_std2bold_xfms(bold_file_tofail)


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
