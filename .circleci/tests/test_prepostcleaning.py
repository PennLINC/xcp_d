"""Tests for prepostcleaning interfaces."""
import os

import nibabel as nb
import numpy as np
import pytest
from pkg_resources import resource_filename as pkgrf

from xcp_d.interfaces.prepostcleaning import CiftiZerosToNaNs, ConvertTo32
from xcp_d.interfaces.workbench import CiftiParcellate


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
    converter_interface.inputs.ref_file = float64_file
    converter_interface.inputs.bold_mask = int64_file
    results = converter_interface.run(cwd=tmpdir)
    float32_file = results.outputs.ref_file
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
    converter_interface.inputs.ref_file = float64_file
    converter_interface.inputs.bold_mask = int64_file
    results = converter_interface.run(cwd=tmpdir)
    float32_file = results.outputs.ref_file
    int32_file = results.outputs.bold_mask

    # Check that new files were created
    assert float64_file != float32_file
    assert int64_file != int32_file

    float32_img = nb.load(float32_file)
    assert float32_img.dataobj.dtype == np.float32
    int32_img = nb.load(int32_file)
    assert int32_img.dataobj.dtype == np.int32


@pytest.mark.skip(reason="The cifti atlas and data files' get_fdata vertex orders do not match.")
def test_cifti_parcellation(fmriprep_with_freesurfer_data, tmp_path_factory):
    """Check that the CIFTI parcellation approach works."""
    tmpdir = tmp_path_factory.mktemp("test_cifti_parcellation")

    atlas_file = pkgrf(
        "xcp_d",
        "data/ciftiatlas/Tian_Subcortex_S3_3T_32k.dlabel.nii",
    )
    cifti_file = fmriprep_with_freesurfer_data["cifti_file"]

    cifti_img = nb.load(cifti_file)
    atlas_img = nb.load(atlas_file)

    cifti_data = cifti_img.get_fdata()
    atlas_data = atlas_img.get_fdata()[0, :]

    # Select half of the vertices in node 5
    node_5 = np.where(atlas_data == 5)[0]
    node_5_good = node_5[::2]
    node_5_bad = node_5[1::2]
    # Select all of the vertices in node 10
    node_10 = np.where(atlas_data == 10)[0]

    cifti_data = np.random.normal(loc=100, scale=5, size=cifti_data.shape)
    cifti_data_zeros = cifti_data.copy()
    cifti_data_nans = cifti_data.copy()
    cifti_data_500s = cifti_data.copy()

    cifti_data_zeros[:, node_5_bad] = 0
    cifti_data_zeros[:, node_10] = 0
    cifti_data_nans[:, node_5_bad] = np.nan
    cifti_data_nans[:, node_10] = np.nan
    cifti_data_500s[:, node_5_bad] = 500
    cifti_data_500s[:, node_10] = 500

    node_5_good_mean = np.mean(cifti_data[:, node_5_good], axis=1)
    node_5_500s_mean = np.mean(cifti_data_500s[:, node_5], axis=1)
    node_10_zeros_mean = np.zeros(cifti_data.shape[0])
    node_10_500s_mean = np.full(cifti_data.shape[0], 500)

    # Parcellate the original file
    cifti_file_sim = os.path.join(tmpdir, "cifti_simulated.dtseries.nii")
    cifti_img_sim = nb.Cifti2Image(
        dataobj=cifti_data,
        header=cifti_img.header,
        file_map=cifti_img.file_map,
        nifti_header=cifti_img.nifti_header,
    )
    cifti_img_sim.to_filename(cifti_file_sim)

    orig_cifti_parc = _run_parcellation(cifti_file_sim, atlas_file, tmpdir)
    orig_cifti_parc_data = nb.load(orig_cifti_parc).get_fdata()
    orig_cifti_node_5 = orig_cifti_parc_data[:, 4]
    orig_cifti_node_10 = orig_cifti_parc_data[:, 9]
    assert not np.allclose(node_5_good_mean, orig_cifti_node_5)
    assert not np.allclose(node_5_500s_mean, orig_cifti_node_5)
    assert not np.allclose(node_10_zeros_mean, orig_cifti_node_10)
    assert not np.allclose(node_10_500s_mean, orig_cifti_node_10)

    # Parcellate and check the zeroed-out file
    # The zeros should be ignored.
    cifti_file_zeros = os.path.join(tmpdir, "cifti_zeros.dtseries.nii")
    cifti_img_zeros = nb.Cifti2Image(
        dataobj=cifti_data_zeros,
        header=cifti_img.header,
        file_map=cifti_img.file_map,
        nifti_header=cifti_img.nifti_header,
    )
    cifti_img_zeros.to_filename(cifti_file_zeros)

    orig_cifti_parc = _run_parcellation(cifti_file_zeros, atlas_file, tmpdir)
    orig_cifti_parc_data = nb.load(orig_cifti_parc).get_fdata()
    orig_cifti_node_5 = orig_cifti_parc_data[:, 4]
    orig_cifti_node_10 = orig_cifti_parc_data[:, 9]
    assert np.allclose(node_5_good_mean, orig_cifti_node_5)
    assert not np.allclose(node_5_500s_mean, orig_cifti_node_5)
    assert np.allclose(node_10_zeros_mean, orig_cifti_node_10)
    assert not np.allclose(node_10_500s_mean, orig_cifti_node_10)

    # Parcellate and check the NaNed-out file
    cifti_file_nans = os.path.join(tmpdir, "cifti_nans.dtseries.nii")
    cifti_img_nans = nb.Cifti2Image(
        dataobj=cifti_data_nans,
        header=cifti_img.header,
        file_map=cifti_img.file_map,
        nifti_header=cifti_img.nifti_header,
    )
    cifti_img_nans.to_filename(cifti_file_nans)

    orig_cifti_parc = _run_parcellation(cifti_file_nans, atlas_file, tmpdir)
    orig_cifti_parc_data = nb.load(orig_cifti_parc).get_fdata()
    orig_cifti_node_5 = orig_cifti_parc_data[:, 4]
    orig_cifti_node_10 = orig_cifti_parc_data[:, 9]
    assert np.allclose(node_5_good_mean, orig_cifti_node_5)
    assert not np.allclose(node_5_500s_mean, orig_cifti_node_5)
    assert np.allclose(node_10_zeros_mean, orig_cifti_node_10)
    assert not np.allclose(node_10_500s_mean, orig_cifti_node_10)

    # Parcellate and check the 500-filled file
    cifti_file_500s = os.path.join(tmpdir, "cifti_500s.dtseries.nii")
    cifti_img_500s = nb.Cifti2Image(
        dataobj=cifti_data_500s,
        header=cifti_img.header,
        file_map=cifti_img.file_map,
        nifti_header=cifti_img.nifti_header,
    )
    cifti_img_500s.to_filename(cifti_file_500s)

    orig_cifti_parc = _run_parcellation(cifti_file_500s, atlas_file, tmpdir)
    orig_cifti_parc_data = nb.load(orig_cifti_parc).get_fdata()
    orig_cifti_node_5 = orig_cifti_parc_data[:, 4]
    orig_cifti_node_10 = orig_cifti_parc_data[:, 9]
    assert not np.allclose(node_5_good_mean, orig_cifti_node_5)
    assert np.allclose(node_5_500s_mean, orig_cifti_node_5)
    assert not np.allclose(node_10_zeros_mean, orig_cifti_node_10)
    assert np.allclose(node_10_500s_mean, orig_cifti_node_10)


def test_cifti_parcellation_basic(fmriprep_with_freesurfer_data, tmp_path_factory):
    """Check that the CIFTI parcellation approach works.

    This is a basic version of ``test_cifti_parcellation``.
    Instead of testing across whole nodes, this test only changes a single vertex.
    """
    tmpdir = tmp_path_factory.mktemp("test_cifti_parcellation_basic")

    atlas_file = pkgrf("xcp_d", "data/ciftiatlas/Tian_Subcortex_S3_3T_32k.dlabel.nii")
    cifti_file = fmriprep_with_freesurfer_data["cifti_file"]

    cifti_img = nb.load(cifti_file)

    cifti_data = cifti_img.get_fdata()
    cifti_data = np.ones(cifti_data.shape)

    # Parcellate the simulated file to find the node associated with the modified vertex.
    VERTEX_IDX = 60000
    cifti_data_locator = cifti_data.copy()
    cifti_data_locator[:, VERTEX_IDX] = 1000000
    cifti_file_loc = os.path.join(tmpdir, "cifti_simulated.dtseries.nii")
    cifti_img_loc = nb.Cifti2Image(
        dataobj=cifti_data_locator,
        header=cifti_img.header,
        file_map=cifti_img.file_map,
        nifti_header=cifti_img.nifti_header,
    )
    cifti_img_loc.to_filename(cifti_file_loc)

    cifti_file_loc_parc = _run_parcellation(cifti_file_loc, atlas_file, tmpdir)
    cifti_data_loc_parc = nb.load(cifti_file_loc_parc).get_fdata()
    loc_node = np.where(cifti_data_loc_parc[0, :] > 1)[0][0]

    # Parcellate and check the zeroed-out file
    # The zeros should be ignored, so the affected node should have all ones
    cifti_data_zeros = cifti_data.copy()
    cifti_data_zeros[:, VERTEX_IDX] = 0
    cifti_file_zeros = os.path.join(tmpdir, "cifti_zeros.dtseries.nii")
    cifti_img_zeros = nb.Cifti2Image(
        dataobj=cifti_data_zeros,
        header=cifti_img.header,
        file_map=cifti_img.file_map,
        nifti_header=cifti_img.nifti_header,
    )
    cifti_img_zeros.to_filename(cifti_file_zeros)

    cifti_file_zeros_parc = _run_parcellation(cifti_file_zeros, atlas_file, tmpdir)
    cifti_data_zeros_parc = nb.load(cifti_file_zeros_parc).get_fdata()
    assert np.all(cifti_data_zeros_parc[:, loc_node] == 1)

    # Parcellate and check the NaNed-out file
    # The NaNs should be ignored, so the affected node should have all ones
    cifti_data_nans = cifti_data.copy()
    cifti_data_nans[:, VERTEX_IDX] = np.nan
    cifti_file_nans = os.path.join(tmpdir, "cifti_nans.dtseries.nii")
    cifti_img_nans = nb.Cifti2Image(
        dataobj=cifti_data_nans,
        header=cifti_img.header,
        file_map=cifti_img.file_map,
        nifti_header=cifti_img.nifti_header,
    )
    cifti_img_nans.to_filename(cifti_file_nans)

    cifti_file_nans_parc = _run_parcellation(cifti_file_nans, atlas_file, tmpdir)
    cifti_data_nans_parc = nb.load(cifti_file_nans_parc).get_fdata()
    assert np.all(cifti_data_nans_parc[:, loc_node] == 1)


def _run_parcellation(in_file, atlas_file, tmpdir):
    replace_empty_vertices = CiftiZerosToNaNs()
    replace_empty_vertices.inputs.in_file = in_file
    replace_results = replace_empty_vertices.run(cwd=tmpdir)
    parcellate_data = CiftiParcellate(direction="COLUMN", only_numeric=True)
    parcellate_data.inputs.in_file = replace_results.outputs.out_file
    parcellate_data.inputs.atlas_label = atlas_file
    parc_results = parcellate_data.run(cwd=tmpdir)
    return parc_results.outputs.out_file
