"""Tests for framewise displacement calculation."""

import json
import os

import nibabel as nb
import numpy as np
import pandas as pd
import pytest
import yaml

from xcp_d.data import load as load_data
from xcp_d.interfaces import censoring


def test_generate_confounds(ds001419_data, tmp_path_factory):
    """Check results."""
    tmpdir = tmp_path_factory.mktemp('test_generate_confounds')
    in_file = ds001419_data['nifti_file']
    confounds_file = ds001419_data['confounds_file']
    confounds_json = ds001419_data['confounds_json']

    df = pd.read_table(confounds_file)
    with open(confounds_json) as fo:
        metadata = json.load(fo)

    # Replace confounds tsv values with values that should be omitted
    df.loc[1:3, 'trans_x'] = [6, 8, 9]
    df.loc[4:6, 'trans_y'] = [7, 8, 9]
    df.loc[7:9, 'trans_z'] = [12, 8, 9]

    # Modify JSON file
    metadata['trans_x'] = {'test': 'hello'}

    # Rename with same convention as initial confounds tsv
    confounds_tsv = os.path.join(tmpdir, 'edited_confounds.tsv')
    df.to_csv(confounds_tsv, sep='\t', index=False, header=True)
    confounds_tsv2 = os.path.join(tmpdir, 'edited_confounds_with_signal.tsv')
    df['signal__fingerpress_condition'] = np.random.random(df.shape[0])
    df.to_csv(confounds_tsv2, sep='\t', index=False, header=True)

    confounds_files = {'preproc_confounds': {'file': confounds_tsv, 'metadata': metadata}}

    # Test with no motion filtering
    config = load_data.readable('nuisance/24P.yml')
    config = yaml.safe_load(config.read_text())
    interface = censoring.GenerateConfounds(
        in_file=in_file,
        confounds_config=config,
        TR=0.8,
        confounds_files=confounds_files,
        motion_filter_type=None,
        motion_filter_order=0,
        band_stop_min=0,
        band_stop_max=0,
        dataset_links={},
        out_dir=str(tmpdir),
    )
    results = interface.run(cwd=tmpdir)

    assert os.path.isfile(results.outputs.confounds_tsv)
    out_confounds_file = results.outputs.confounds_tsv
    out_df = pd.read_table(out_confounds_file)
    assert out_df.shape[1] == 24  # 24(P)
    assert 'trans_x' in out_df.columns

    # Test with notch motion filtering
    config = load_data.readable('nuisance/24P.yml')
    config = yaml.safe_load(config.read_text())
    interface = censoring.GenerateConfounds(
        in_file=in_file,
        confounds_config=config,
        TR=0.8,
        confounds_files=confounds_files,
        motion_filter_type='notch',
        motion_filter_order=4,
        band_stop_min=12,
        band_stop_max=20,
        dataset_links={},
        out_dir=str(tmpdir),
    )
    results = interface.run(cwd=tmpdir)

    assert os.path.isfile(results.outputs.confounds_tsv)
    out_confounds_file = results.outputs.confounds_tsv
    out_df = pd.read_table(out_confounds_file)
    assert out_df.shape[1] == 24  # 24(P)
    assert 'trans_x_filtered' in out_df.columns

    # Test with low-pass motion filtering
    config = load_data.readable('nuisance/24P.yml')
    config = yaml.safe_load(config.read_text())
    interface = censoring.GenerateConfounds(
        in_file=in_file,
        confounds_config=config,
        TR=0.8,
        confounds_files=confounds_files,
        motion_filter_type='lp',
        motion_filter_order=4,
        band_stop_min=6,
        band_stop_max=0,
        dataset_links={},
        out_dir=str(tmpdir),
    )
    results = interface.run(cwd=tmpdir)

    assert os.path.isfile(results.outputs.confounds_tsv)
    out_confounds_file = results.outputs.confounds_tsv
    out_df = pd.read_table(out_confounds_file)
    assert out_df.shape[1] == 24  # 24(P)
    assert 'trans_x_filtered' in out_df.columns

    # Test with regular expressions in confounds_config
    config = load_data.readable('nuisance/acompcor_gsr.yml')
    config = yaml.safe_load(config.read_text())
    interface = censoring.GenerateConfounds(
        in_file=in_file,
        confounds_config=config,
        TR=0.8,
        confounds_files=confounds_files,
        motion_filter_type=None,
        motion_filter_order=0,
        band_stop_min=0,
        band_stop_max=0,
        dataset_links={},
        out_dir=str(tmpdir),
    )
    results = interface.run(cwd=tmpdir)

    assert os.path.isfile(results.outputs.confounds_tsv)
    out_confounds_file = results.outputs.confounds_tsv
    out_df = pd.read_table(out_confounds_file)
    assert out_df.shape[1] == 31  # 31 parameters

    # Test with signal regressors
    config = load_data.readable('nuisance/24P.yml')
    config = yaml.safe_load(config.read_text())
    config['confounds']['preproc_confounds']['columns'].append('signal__fingerpress_condition')
    confounds_files = {'preproc_confounds': {'file': confounds_tsv2, 'metadata': metadata}}
    interface = censoring.GenerateConfounds(
        in_file=in_file,
        confounds_config=config,
        TR=0.8,
        confounds_files=confounds_files,
        motion_filter_type=None,
        motion_filter_order=0,
        band_stop_min=0,
        band_stop_max=0,
        dataset_links={},
        out_dir=str(tmpdir),
    )
    results = interface.run(cwd=tmpdir)

    assert os.path.isfile(results.outputs.confounds_tsv)
    out_confounds_file = results.outputs.confounds_tsv
    out_df = pd.read_table(out_confounds_file)
    assert out_df.shape[1] == 24  # 24 parameters (doesn't include 25th signal column)
    assert 'signal__fingerpress_condition' not in out_df.columns
    assert all(col.endswith('_orth') for col in out_df.columns)
    assert 'signal__fingerpress_condition' in results.outputs.confounds_metadata.keys()

    # Test with image-based confounds
    config = load_data.readable('nuisance/rapidtide+24P.yml')
    config = yaml.safe_load(config.read_text())
    confounds_files = {
        'preproc_confounds': {'file': confounds_tsv, 'metadata': metadata},
        'rapidtide_slfo': {'file': in_file, 'metadata': {}},
    }
    interface = censoring.GenerateConfounds(
        in_file=in_file,
        confounds_config=config,
        TR=0.8,
        confounds_files=confounds_files,
        motion_filter_type=None,
        motion_filter_order=0,
        band_stop_min=0,
        band_stop_max=0,
        dataset_links={},
        out_dir=str(tmpdir),
    )
    results = interface.run(cwd=tmpdir)

    assert os.path.isfile(results.outputs.confounds_tsv)
    out_confounds_file = results.outputs.confounds_tsv
    out_df = pd.read_table(out_confounds_file)
    assert out_df.shape[1] == 25  # 24P + rapidtide (stand-in for the voxel-wise regressor)
    assert os.path.isfile(results.outputs.confounds_images[0])
    assert out_df['rapidtide_slfo'].isna().all()

    # Test with image-based confounds and a signal column (will fail)
    config = load_data.readable('nuisance/rapidtide+24P.yml')
    config = yaml.safe_load(config.read_text())
    config['confounds']['preproc_confounds']['columns'].append('signal__fingerpress_condition')

    confounds_files = {
        'preproc_confounds': {'file': confounds_tsv2, 'metadata': metadata},
        'rapidtide_slfo': {'file': in_file, 'metadata': {}},
    }
    interface = censoring.GenerateConfounds(
        in_file=in_file,
        confounds_config=config,
        TR=0.8,
        confounds_files=confounds_files,
        motion_filter_type=None,
        motion_filter_order=0,
        band_stop_min=0,
        band_stop_max=0,
        dataset_links={},
        out_dir=str(tmpdir),
    )
    with pytest.raises(NotImplementedError):
        results = interface.run(cwd=tmpdir)


def test_process_motion(ds001419_data, tmp_path_factory):
    """Test censoring.ProcessMotion."""
    tmpdir = tmp_path_factory.mktemp('test_process_motion')

    motion_file = ds001419_data['confounds_file']
    motion_json = ds001419_data['confounds_json']

    # Basic test without filtering
    interface = censoring.ProcessMotion(
        motion_file=motion_file,
        motion_json=motion_json,
        TR=2.0,
        fd_thresh=0,
        head_radius=50,
        motion_filter_type=None,
        motion_filter_order=None,
        band_stop_min=None,
        band_stop_max=None,
    )
    results = interface.run(cwd=tmpdir)
    assert os.path.isfile(results.outputs.motion_file)
    assert os.path.isfile(results.outputs.temporal_mask)

    # Basic test with censoring, but no filtering
    interface = censoring.ProcessMotion(
        motion_file=motion_file,
        motion_json=motion_json,
        TR=2.0,
        fd_thresh=0.2,
        head_radius=50,
        motion_filter_type=None,
        motion_filter_order=None,
        band_stop_min=None,
        band_stop_max=None,
    )
    results = interface.run(cwd=tmpdir)
    assert os.path.isfile(results.outputs.motion_file)
    assert os.path.isfile(results.outputs.temporal_mask)

    # Basic test with filtering, but not censoring
    interface = censoring.ProcessMotion(
        motion_file=motion_file,
        motion_json=motion_json,
        TR=2.0,
        fd_thresh=0,
        head_radius=50,
        motion_filter_type='notch',
        motion_filter_order=4,
        band_stop_min=12,
        band_stop_max=20,
    )
    results = interface.run(cwd=tmpdir)
    assert os.path.isfile(results.outputs.motion_file)
    assert os.path.isfile(results.outputs.temporal_mask)

    # Basic test with filtering and censoring
    interface = censoring.ProcessMotion(
        motion_file=motion_file,
        motion_json=motion_json,
        TR=2.0,
        fd_thresh=0.2,
        head_radius=50,
        motion_filter_type='notch',
        motion_filter_order=4,
        band_stop_min=12,
        band_stop_max=20,
    )
    results = interface.run(cwd=tmpdir)
    assert os.path.isfile(results.outputs.motion_file)
    assert os.path.isfile(results.outputs.temporal_mask)


def test_removedummyvolumes_nifti(ds001419_data, tmp_path_factory):
    """Test RemoveDummyVolumes() for NIFTI input data."""
    # Define inputs
    tmpdir = tmp_path_factory.mktemp('test_RemoveDummyVolumes_nifti')

    boldfile = ds001419_data['nifti_file']
    confounds_file = ds001419_data['confounds_file']

    # Find the original number of volumes acc. to nifti & confounds timeseries
    original_confounds = pd.read_table(confounds_file)
    original_nvols_nifti = nb.load(boldfile).shape[3]

    # Test a nifti file with 0 volumes to remove
    remove_nothing = censoring.RemoveDummyVolumes(
        bold_file=boldfile,
        confounds_tsv=confounds_file,
        confounds_images=[boldfile],
        motion_file=confounds_file,
        temporal_mask=confounds_file,
        dummy_scans=0,
        dummy_scan_source=confounds_file,
    )
    results = remove_nothing.run(cwd=tmpdir)
    undropped_confounds = pd.read_table(results.outputs.confounds_tsv_dropped_TR)
    # Were the files created?
    assert os.path.exists(results.outputs.bold_file_dropped_TR)
    assert os.path.exists(results.outputs.confounds_tsv_dropped_TR)
    assert os.path.exists(results.outputs.confounds_images_dropped_TR[0])
    # Have the confounds stayed the same shape?
    assert undropped_confounds.shape == original_confounds.shape
    # Has the nifti stayed the same shape?
    assert (
        nb.load(results.outputs.bold_file_dropped_TR).get_fdata().shape[3] == original_nvols_nifti
    )

    # Test a nifti file with 1-10 volumes to remove
    for n in range(10):
        remove_n_vols = censoring.RemoveDummyVolumes(
            bold_file=boldfile,
            confounds_tsv=confounds_file,
            confounds_images=[boldfile],
            motion_file=confounds_file,
            temporal_mask=confounds_file,
            dummy_scans=n,
            dummy_scan_source=confounds_file,
        )
        results = remove_n_vols.run(cwd=tmpdir)
        dropped_confounds = pd.read_table(results.outputs.confounds_tsv_dropped_TR)
        # Were the files created?
        assert os.path.exists(results.outputs.bold_file_dropped_TR)
        assert os.path.exists(results.outputs.confounds_tsv_dropped_TR)
        assert os.path.exists(results.outputs.confounds_images_dropped_TR[0])
        # Have the confounds changed correctly?
        assert dropped_confounds.shape[0] == original_confounds.shape[0] - n
        # Has the nifti changed correctly?
        n_vols = nb.load(results.outputs.bold_file_dropped_TR).get_fdata().shape[3]
        if n_vols != (original_nvols_nifti - n):
            print(f'Tests failing at N = {n}.')
            raise ValueError(f'Number of volumes in dropped nifti is {n_vols}.')


def test_removedummyvolumes_cifti(ds001419_data, tmp_path_factory):
    """Test RemoveDummyVolumes() for CIFTI input data."""
    # Define inputs
    tmpdir = tmp_path_factory.mktemp('test_RemoveDummyVolumes_cifti')

    boldfile = ds001419_data['cifti_file']
    confounds_file = ds001419_data['confounds_file']

    # Find the original number of volumes acc. to cifti & confounds timeseries
    original_confounds = pd.read_table(confounds_file)
    original_nvols_cifti = nb.load(boldfile).shape[0]

    # Test a cifti file with 0 volumes to remove
    remove_nothing = censoring.RemoveDummyVolumes(
        bold_file=boldfile,
        confounds_tsv=confounds_file,
        confounds_images=[boldfile],
        motion_file=confounds_file,
        temporal_mask=confounds_file,
        dummy_scans=0,
        dummy_scan_source=confounds_file,
    )
    results = remove_nothing.run(cwd=tmpdir)
    undropped_confounds = pd.read_table(results.outputs.confounds_tsv_dropped_TR)
    # Were the files created?
    assert os.path.exists(results.outputs.bold_file_dropped_TR)
    assert os.path.exists(results.outputs.confounds_tsv_dropped_TR)
    # Have the confounds stayed the same shape?
    assert undropped_confounds.shape == original_confounds.shape
    # Has the cifti stayed the same shape?
    assert (
        nb.load(results.outputs.bold_file_dropped_TR).get_fdata().shape[0] == original_nvols_cifti
    )

    # Test a cifti file with 1-10 volumes to remove
    for n in range(10):
        remove_n_vols = censoring.RemoveDummyVolumes(
            bold_file=boldfile,
            confounds_tsv=confounds_file,
            confounds_images=[boldfile],
            motion_file=confounds_file,
            temporal_mask=confounds_file,
            dummy_scans=n,
            dummy_scan_source=confounds_file,
        )

        results = remove_n_vols.run(cwd=tmpdir)
        dropped_confounds = pd.read_table(results.outputs.confounds_tsv_dropped_TR)
        # Were the files created?
        assert os.path.exists(results.outputs.bold_file_dropped_TR)
        assert os.path.exists(results.outputs.confounds_tsv_dropped_TR)
        # Have the confounds changed correctly?
        assert dropped_confounds.shape[0] == original_confounds.shape[0] - n
        # Has the cifti changed correctly?
        n_vols = nb.load(results.outputs.bold_file_dropped_TR).get_fdata().shape[0]
        if n_vols != (original_nvols_cifti - n):
            print(f'Tests failing at N = {n}.')
            raise ValueError(f'Number of volumes in dropped cifti is {n_vols}.')


def test_random_censor(tmp_path_factory):
    """Test RandomCensor."""
    tmpdir = tmp_path_factory.mktemp('test_random_censor')
    n_volumes, n_outliers = 500, 100
    exact_scans = [100, 200, 300, 400]

    outliers_arr = np.zeros(n_volumes, dtype=int)
    rng = np.random.default_rng(0)
    outlier_idx = rng.choice(np.arange(n_volumes, dtype=int), size=n_outliers, replace=False)
    outliers_arr[outlier_idx] = 1
    temporal_mask_df = pd.DataFrame(data=outliers_arr, columns=['framewise_displacement'])
    original_temporal_mask = os.path.join(tmpdir, 'orig_tmask.tsv')
    temporal_mask_df.to_csv(original_temporal_mask, index=False, sep='\t')

    # Run the RandomCensor interface without any exact_scans.
    interface = censoring.RandomCensor(
        temporal_mask_metadata={},
        temporal_mask=original_temporal_mask,
        exact_scans=[],
        random_seed=0,
    )
    results = interface.run(cwd=tmpdir)
    assert results.outputs.temporal_mask == original_temporal_mask  # same file as input
    assert isinstance(results.outputs.temporal_mask_metadata, dict)

    # Run the interface with exact_scans
    interface = censoring.RandomCensor(
        temporal_mask_metadata={},
        temporal_mask=original_temporal_mask,
        exact_scans=exact_scans,
        random_seed=0,
    )
    results = interface.run(cwd=tmpdir)
    assert os.path.isfile(results.outputs.temporal_mask)
    assert isinstance(results.outputs.temporal_mask_metadata, dict)
    new_temporal_mask_df = pd.read_table(results.outputs.temporal_mask)
    new_temporal_mask_df_no_outliers = new_temporal_mask_df.loc[
        new_temporal_mask_df['framewise_displacement'] == 0
    ]
    for exact_scan in exact_scans:
        exact_scan_col = f'exact_{exact_scan}'
        assert exact_scan_col in new_temporal_mask_df_no_outliers.columns
        # The column's values should sum to the number of volumes minus the number of retained.
        # Outliers don't show up here.
        assert new_temporal_mask_df_no_outliers[exact_scan_col].sum() == n_volumes - (
            exact_scan + n_outliers
        )
        # The outlier volumes and exact-scan censored volumes shouldn't overlap.
        assert all(
            new_temporal_mask_df_no_outliers[[exact_scan_col, 'framewise_displacement']].sum(
                axis=1
            )
            <= 1
        )


def test_censor(ds001419_data, tmp_path_factory):
    """Test Censor interface."""
    tmpdir = tmp_path_factory.mktemp('test_censor')
    nifti_file = ds001419_data['nifti_file']
    cifti_file = ds001419_data['cifti_file']
    in_img = nb.load(nifti_file)
    n_volumes = in_img.shape[3]
    censoring_df = pd.DataFrame(columns=['framewise_displacement'], data=np.zeros(n_volumes))
    temporal_mask = os.path.join(tmpdir, 'temporal_mask.tsv')
    censoring_df.to_csv(temporal_mask, sep='\t', index=False)

    # Test with a NIfTI file, with no censored volumes
    interface = censoring.Censor(
        in_file=nifti_file,
        temporal_mask=temporal_mask,
        column='framewise_displacement',
    )
    results = interface.run(cwd=tmpdir)
    out_file = results.outputs.out_file
    assert os.path.isfile(out_file)
    out_img = nb.load(out_file)
    assert out_img.shape[3] == n_volumes

    # Test with a CIFTI file, with no censored volumes
    interface = censoring.Censor(
        in_file=cifti_file,
        temporal_mask=temporal_mask,
        column='framewise_displacement',
    )
    results = interface.run(cwd=tmpdir)
    out_file = results.outputs.out_file
    assert os.path.isfile(out_file)
    out_img = nb.load(out_file)
    assert out_img.shape[0] == n_volumes

    # Create a temporal mask with some censored volumes
    n_censored_volumes = 10
    n_retained_volumes = n_volumes - n_censored_volumes
    censoring_df.loc[range(10), 'framewise_displacement'] = 1
    # Add random censor column
    censoring_df['random_censor'] = 0
    censoring_df.loc[20:29, 'random_censor'] = 1
    censoring_df.to_csv(temporal_mask, sep='\t', index=False)

    # Test with a NIfTI file, with some censored volumes
    interface = censoring.Censor(
        in_file=nifti_file,
        temporal_mask=temporal_mask,
        column='framewise_displacement',
    )
    results = interface.run(cwd=tmpdir)
    out_file = results.outputs.out_file
    assert os.path.isfile(out_file)
    out_img = nb.load(out_file)
    assert out_img.shape[3] == n_retained_volumes

    # Test with additional censoring column
    interface2 = censoring.Censor(
        in_file=results.outputs.out_file,
        temporal_mask=temporal_mask,
        column='random_censor',
    )
    results2 = interface2.run(cwd=tmpdir)
    out_file2 = results2.outputs.out_file
    assert os.path.isfile(out_file2)
    out_img2 = nb.load(out_file2)
    assert out_img2.shape[3] == (n_retained_volumes - 10)

    # Test with a CIFTI file, with some censored volumes
    interface = censoring.Censor(
        in_file=cifti_file,
        temporal_mask=temporal_mask,
        column='framewise_displacement',
    )
    results = interface.run(cwd=tmpdir)
    out_file = results.outputs.out_file
    assert os.path.isfile(out_file)
    out_img = nb.load(out_file)
    assert out_img.shape[0] == n_retained_volumes

    # Test with additional censoring column
    interface2 = censoring.Censor(
        in_file=results.outputs.out_file,
        temporal_mask=temporal_mask,
        column='random_censor',
    )
    results2 = interface2.run(cwd=tmpdir)
    out_file2 = results2.outputs.out_file
    assert os.path.isfile(out_file2)
    out_img2 = nb.load(out_file2)
    assert out_img2.shape[0] == (n_retained_volumes - 10)
