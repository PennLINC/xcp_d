"""Tests for prepostcleaning interfaces."""
import os

import nibabel as nb
import numpy as np
from pkg_resources import resource_filename as pkgrf

from xcp_d.interfaces.prepostcleaning import CiftiPrepareForParcellation, ConvertTo32
from xcp_d.interfaces.workbench import CiftiCreateDenseFromTemplate, CiftiParcellate
from xcp_d.utils.write_save import read_ndata, write_ndata


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


def test_cifti_parcellation_basic(fmriprep_with_freesurfer_data, tmp_path_factory):
    """Check that the CIFTI parcellation approach works.

    This is a basic version of ``test_cifti_parcellation``.
    Instead of testing across whole parcels, this test only changes a single vertex.
    """
    tmpdir = tmp_path_factory.mktemp("test_cifti_parcellation_basic")

    atlas_file = pkgrf("xcp_d", "data/ciftiatlas/Tian_Subcortex_S3_3T_32k.dlabel.nii")
    cifti_file = fmriprep_with_freesurfer_data["cifti_file"]

    cifti_img = nb.load(cifti_file)

    cifti_data = cifti_img.get_fdata()
    cifti_data = np.ones(cifti_data.shape)

    # Parcellate the simulated file to find the parcel associated with the modified vertex.
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

    # Create file with one vertex that is all zeros.
    # The zeros should be ignored, so the affected parcel should have all ones.
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

    # Create file with one vertex that has a zero in first timepoint.
    # The single zero should be treated as real data,
    # so the affected parcel should have a value < 1 in the affected timepoint.
    cifti_data_zeros = cifti_data.copy()
    cifti_data_zeros[0, VERTEX_IDX] = 0
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
    assert cifti_data_zeros_parc[0, loc_node] < 1

    # Create file with one vertex that is all NaNs.
    # The NaNs should be ignored, so the affected parcel should have all zeros.
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

    # Create file with one vertex that is has a NaN in the first timepoint.
    # The NaN should be ignored, so the affected parcel should have all zeros.
    cifti_data_nans = cifti_data.copy()
    cifti_data_nans[0, VERTEX_IDX] = np.nan
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
    # Resample the atlas to the same order as the data file.
    ccdft = CiftiCreateDenseFromTemplate()
    ccdft.inputs.template_cifti = in_file
    ccdft.inputs.label = atlas_file
    ccdft.inputs.cifti_out = "resampled_atlas.dlabel.nii"
    ccdft_results = ccdft.run(cwd=tmpdir)

    # Zero out any parcels that have <50% coverage.
    cpfp = CiftiPrepareForParcellation()
    cpfp.inputs.data_file = in_file
    cpfp.inputs.atlas_file = ccdft_results.outputs.cifti_out
    cpfp.inputs.TR = 1
    replace_results = cpfp.run(cwd=tmpdir)

    # Apply the parcellation.
    parcellate_data = CiftiParcellate(direction="COLUMN", only_numeric=True)
    parcellate_data.inputs.in_file = replace_results.outputs.out_file
    parcellate_data.inputs.atlas_label = atlas_file
    parc_results = parcellate_data.run(cwd=tmpdir)

    return parc_results.outputs.out_file


def test_cifti_parcellation_resampling(fmriprep_with_freesurfer_data, tmp_path_factory):
    """Test CIFTI parcellation with resampling."""
    tmpdir = tmp_path_factory.mktemp("test_cifti_conn")

    boldfile = fmriprep_with_freesurfer_data["cifti_file"]
    atlas_file = pkgrf(
        "xcp_d",
        "data/ciftiatlas/Schaefer2018_100Parcels_17Networks_order.dlabel.nii",
    )
    N_PARCELS = 100

    ccdft = CiftiCreateDenseFromTemplate()
    ccdft.inputs.template_cifti = boldfile
    ccdft.inputs.label = atlas_file
    ccdft.inputs.cifti_out = "resampled_atlas.dlabel.nii"
    ccdft.run(cwd=tmpdir)

    resampled_atlas_file = os.path.join(tmpdir, "resampled_atlas.dlabel.nii")
    assert os.path.isfile(resampled_atlas_file)

    original_atlas_data = read_ndata(atlas_file)
    original_atlas_data = original_atlas_data[:, 0]  # squeeze to 1D

    resampled_atlas_data = read_ndata(resampled_atlas_file)
    resampled_atlas_data = resampled_atlas_data[:, 0]  # squeeze to 1D

    bold_data = read_ndata(boldfile)
    assert bold_data.shape[0] != original_atlas_data.shape[0]
    assert bold_data.shape[0] == resampled_atlas_data.shape[0]

    parcellated_bold_data = np.zeros_like(bold_data)
    for i_parcel in range(1, N_PARCELS + 1):
        parcel_idx = resampled_atlas_data == i_parcel
        parcellated_bold_data[parcel_idx, :] = i_parcel

    modified_data_file = os.path.join(tmpdir, "modified_data.dtseries.nii")
    write_ndata(parcellated_bold_data, boldfile, modified_data_file)
    assert os.path.isfile(modified_data_file)

    # The modified data file should match the resampled atlas file
    parcellated_modified_data_file = _run_parcellation(
        modified_data_file,
        resampled_atlas_file,
        tmpdir,
    )
    parcellated_data = read_ndata(parcellated_modified_data_file)
    assert parcellated_data.shape == (N_PARCELS, bold_data.shape[1])

    for i_parcel in range(parcellated_data.shape[0]):
        parcel_val = i_parcel + 1
        assert np.all(parcellated_data[i_parcel, :] == parcel_val)

    # The modified data file should also match the original atlas file
    parcellated_modified_data_file = _run_parcellation(
        modified_data_file,
        atlas_file,
        tmpdir,
    )
    parcellated_data = read_ndata(parcellated_modified_data_file)
    assert parcellated_data.shape == (N_PARCELS, bold_data.shape[1])

    for i_parcel in range(parcellated_data.shape[0]):
        parcel_val = i_parcel + 1
        assert np.all(parcellated_data[i_parcel, :] == parcel_val)
