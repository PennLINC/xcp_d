"""Tests for xcp_d.interfaces.utils module."""
import os

import nibabel as nb
import numpy as np

from xcp_d.interfaces.utils import ConvertTo32


def test_conversion_to_32bit_nifti(fmriprep_with_freesurfer_data, tmp_path_factory):
    """Convert nifti files to 32-bit."""
    tmpdir = tmp_path_factory.mktemp("test_conversion_to_32bit")

    float_file = fmriprep_with_freesurfer_data["nifti_file"]
    int_file = fmriprep_with_freesurfer_data["brain_mask_file"]

    float64_file = os.path.join(tmpdir, "float64.nii.gz")
    int64_file = os.path.join(tmpdir, "int64.nii.gz")

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


def test_conversion_to_32bit_cifti(fmriprep_with_freesurfer_data, tmp_path_factory):
    """Convert nifti files to 32-bit."""
    tmpdir = tmp_path_factory.mktemp("test_conversion_to_32bit")

    float_file = fmriprep_with_freesurfer_data["cifti_file"]

    float64_file = os.path.join(tmpdir, "float64.dtseries.nii")
    int64_file = os.path.join(tmpdir, "int64.dtseries.nii")

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
