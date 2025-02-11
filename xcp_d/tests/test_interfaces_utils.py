"""Tests for xcp_d.interfaces.utils module."""

import json
import os

import nibabel as nb
import numpy as np
import pandas as pd

from xcp_d.interfaces.utils import LINCQC, ConvertTo32


def test_conversion_to_32bit_nifti(ds001419_data, tmp_path_factory):
    """Convert nifti files to 32-bit."""
    tmpdir = tmp_path_factory.mktemp('test_conversion_to_32bit')

    float_file = ds001419_data['nifti_file']
    int_file = ds001419_data['brain_mask_file']

    float64_file = os.path.join(tmpdir, 'float64.nii.gz')
    int64_file = os.path.join(tmpdir, 'int64.nii.gz')

    # Create a float64 image to downcast
    float_img = nb.load(float_file)
    float64_img = nb.Nifti1Image(
        np.random.random(float_img.shape[:3]).astype(np.float64),
        affine=float_img.affine,
        header=float_img.header,
    )
    float64_img.header.set_data_dtype(np.float64)  # need to set data dtype too
    assert float64_img.dataobj.dtype == np.float64
    float64_img.to_filename(float64_file)

    # Create an int64 image to downcast
    int_img = nb.load(int_file)
    int64_img = nb.Nifti1Image(
        np.random.randint(0, 2, size=int_img.shape[:3], dtype=np.int64),
        affine=int_img.affine,
        header=int_img.header,
    )
    int64_img.header.set_data_dtype(np.int64)  # need to set data dtype too
    assert int64_img.dataobj.dtype == np.int64
    int64_img.to_filename(int64_file)
    int64_img_2 = nb.load(int64_file)
    assert int64_img_2.dataobj.dtype == np.int64

    # Run converter
    converter_interface = ConvertTo32()
    converter_interface.inputs.boldref = float64_file
    converter_interface.inputs.bold_mask = int64_file
    results = converter_interface.run(cwd=tmpdir)
    float32_file = results.outputs.boldref
    int32_file = results.outputs.bold_mask

    # Check that new files were created
    assert float64_file != float32_file
    assert int64_file != int32_file

    float32_img = nb.load(float32_file)
    assert float32_img.dataobj.dtype == np.float32
    int32_img = nb.load(int32_file)
    assert int32_img.dataobj.dtype == np.int32


def test_conversion_to_32bit_cifti(ds001419_data, tmp_path_factory):
    """Convert nifti files to 32-bit."""
    tmpdir = tmp_path_factory.mktemp('test_conversion_to_32bit')

    float_file = ds001419_data['cifti_file']

    float64_file = os.path.join(tmpdir, 'float64.dtseries.nii')
    int64_file = os.path.join(tmpdir, 'int64.dtseries.nii')

    # Create a float64 image to downcast
    float_img = nb.load(float_file)
    float64_img = nb.Cifti2Image(
        np.random.random(float_img.shape).astype(np.float64),
        header=float_img.header,
        nifti_header=float_img.nifti_header,
    )
    float64_img.nifti_header.set_data_dtype(np.float64)  # need to set data dtype too
    assert float64_img.dataobj.dtype == np.float64
    float64_img.to_filename(float64_file)

    # Create an int64 image to downcast
    int64_img = nb.Cifti2Image(
        np.random.randint(0, 2, size=float_img.shape, dtype=np.int64),
        header=float_img.header,
        nifti_header=float_img.nifti_header,
    )
    int64_img.nifti_header.set_data_dtype(np.int64)  # need to set data dtype too
    assert int64_img.dataobj.dtype == np.int64
    int64_img.to_filename(int64_file)
    int64_img_2 = nb.load(int64_file)
    assert int64_img_2.dataobj.dtype == np.int64

    # Run converter
    converter_interface = ConvertTo32()
    converter_interface.inputs.boldref = float64_file
    converter_interface.inputs.bold_mask = int64_file
    results = converter_interface.run(cwd=tmpdir)
    float32_file = results.outputs.boldref
    int32_file = results.outputs.bold_mask

    # Check that new files were created
    assert float64_file != float32_file
    assert int64_file != int32_file

    float32_img = nb.load(float32_file)
    assert float32_img.dataobj.dtype == np.float32
    int32_img = nb.load(int32_file)
    assert int32_img.dataobj.dtype == np.int32


def test_lincqc(ds001419_data, tmp_path):
    """Test the LINCQC interface."""
    # Create test data
    img = nb.load(ds001419_data['nifti_file'])
    n_timepoints = img.shape[3]
    dummy_scans = 5

    # Create motion parameters
    motion_data = {
        'framewise_displacement': np.random.uniform(0, 1, n_timepoints),
        'rmsd': np.random.uniform(0, 0.5, n_timepoints),
    }
    motion_file = tmp_path / 'motion.tsv'
    pd.DataFrame(motion_data).to_csv(motion_file, sep='\t', index=False)

    # Create temporal mask
    temporal_mask_data = {'framewise_displacement': np.zeros(n_timepoints, dtype=int)}
    # Mark some volumes as censored (value = 1)
    temporal_mask_data['framewise_displacement'][10:15] = 1
    temporal_mask_file = tmp_path / 'temporal_mask.tsv'
    pd.DataFrame(temporal_mask_data).to_csv(temporal_mask_file, sep='\t', index=False)

    # Initialize and run interface
    lincqc = LINCQC(
        name_source='sub-01_task-rest_bold.nii.gz',
        bold_file=ds001419_data['nifti_file'],
        dummy_scans=dummy_scans,
        motion_file=ds001419_data['confounds_file'],
        cleaned_file=ds001419_data['nifti_file'],
        TR=2.0,
        head_radius=50,
        temporal_mask=temporal_mask_file,
        bold_mask_inputspace=ds001419_data['brain_mask_file'],
        anat_mask_anatspace=ds001419_data['brain_mask_file'],
        template_mask=ds001419_data['brain_mask_file'],
        bold_mask_anatspace=ds001419_data['brain_mask_file'],
        bold_mask_stdspace=ds001419_data['brain_mask_file'],
    )

    res = lincqc.run(cwd=tmp_path)

    # Test outputs exist
    assert os.path.exists(res.outputs.qc_file)
    assert os.path.exists(res.outputs.qc_metadata)

    # Load and check QC results
    qc_df = pd.read_table(res.outputs.qc_file)
    with open(res.outputs.qc_metadata) as f:
        qc_metadata = json.load(f)

    # Test basic expectations
    assert 'mean_fd' in qc_df.columns
    assert 'mean_dvars_initial' in qc_df.columns
    assert 'num_dummy_volumes' in qc_df.columns
    assert qc_df['num_dummy_volumes'].iloc[0] == dummy_scans
    assert qc_df['num_censored_volumes'].iloc[0] == 5  # We censored 5 volumes

    # Test metadata
    assert 'mean_fd' in qc_metadata
    assert 'Description' in qc_metadata['mean_fd']
    assert 'Units' in qc_metadata['mean_fd']
