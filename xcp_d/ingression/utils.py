# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Functions to support ingression of non-BIDS preprocessing derivatives."""

import filecmp
import json
import os
import shutil

import numpy as np
import pandas as pd
from nilearn import image, maskers
from nipype import logging
from niworkflows.interfaces.confounds import NormalizeMotionParams
from niworkflows.viz.utils import compose_view, cuts_from_bbox, plot_registration

from xcp_d import config
from xcp_d.interfaces.workbench import CiftiCreateDenseScalar

LOGGER = logging.getLogger('nipype.utils')

# Standard space and BIDS conventions used across ingression converters.
TEMPLATE_SPACE = 'MNI152NLin6Asym'
BIDS_VERSION = '1.9.0'
CONFOUNDS_TSV_SUFFIX = '_desc-confounds_timeseries.tsv'
CONFOUNDS_JSON_SUFFIX = '_desc-confounds_timeseries.json'


def _expand_motion_columns(motion_df):
    """Add derivative1 and power2 columns for each existing column in the DataFrame."""
    columns = motion_df.columns.tolist()
    for col in columns:
        motion_df[f'{col}_derivative1'] = motion_df[col].diff()
    columns = motion_df.columns.tolist()
    for col in columns:
        motion_df[f'{col}_power2'] = motion_df[col] ** 2


def _write_confounds_timeseries(confounds_df, out_dir, prefix):
    """Write confounds DataFrame to TSV and JSON sidecar in BIDS derivatives format."""
    tsv_path = os.path.join(out_dir, f'{prefix}{CONFOUNDS_TSV_SUFFIX}')
    json_path = os.path.join(out_dir, f'{prefix}{CONFOUNDS_JSON_SUFFIX}')
    confounds_df.to_csv(tsv_path, sep='\t', na_rep='n/a', index=False)
    write_json({col: {'Description': ''} for col in confounds_df.columns}, json_path)


def get_identity_transform_destinations(anat_dir_bids, subses_ents, volspace=TEMPLATE_SPACE):
    """Return destination paths for T1w<->template identity transform files.

    Parameters
    ----------
    anat_dir_bids : str
        Path to BIDS anatomical derivatives directory.
    subses_ents : str
        BIDS subject/session entity string (e.g., "sub-01_ses-1").
    volspace : str, optional
        Template space name. Default is TEMPLATE_SPACE (MNI152NLin6Asym).

    Returns
    -------
    list of str
        Two paths: from-T1w_to-template and from-template_to-T1w transform files.
    """
    return [
        os.path.join(anat_dir_bids, f'{subses_ents}_from-T1w_to-{volspace}_mode-image_xfm.txt'),
        os.path.join(anat_dir_bids, f'{subses_ents}_from-{volspace}_to-T1w_mode-image_xfm.txt'),
    ]


def write_scans_tsv(copy_dictionary, subject_dir_bids, subses_ents):
    """Write a scans TSV mapping BIDS derivative filenames to their source paths.

    Parameters
    ----------
    copy_dictionary : dict
        Mapping from source file path to list of destination file paths.
        Values may be lists (for copy mapping) or single paths (e.g. morph outputs).
    subject_dir_bids : str
        Path to subject (or subject/session) BIDS directory.
    subses_ents : str
        BIDS entity string used in the output filename (e.g., "sub-01_ses-1").
    """
    scans_dict = {}
    for src, dests in copy_dictionary.items():
        for dest in dests if isinstance(dests, list) else [dests]:
            scans_dict[dest] = src
    scans_df = pd.DataFrame(list(scans_dict.items()), columns=['filename', 'source_file'])
    scans_tsv = os.path.join(subject_dir_bids, f'{subses_ents}_scans.tsv')
    scans_df.to_csv(scans_tsv, sep='\t', index=False)


def collect_anatomical_files(anat_dir_orig, anat_dir_bids, base_anatomical_ents):
    """Collect anatomical files from ABCD or HCP-YA derivatives.

    Parameters
    ----------
    anat_dir_orig : str
        Path to original anatomical derivatives directory containing source files
        (e.g., T1w.nii.gz, ribbon.nii.gz, brainmask_fs.nii.gz).
    anat_dir_bids : str
        Path to output BIDS-compliant anatomical directory.
    base_anatomical_ents : str
        BIDS entity string to use as filename prefix (e.g., "sub-01_space-MNI152NLin6Asym_res-2").

    Returns
    -------
    dict
        Dictionary mapping source file paths to lists of destination file paths.
        Files that do not exist in the source directory are omitted with a warning.
    """
    ANAT_DICT = {
        # XXX: Why have T1w here and T1w_restore for HCP?
        'T1w.nii.gz': 'desc-preproc_T1w.nii.gz',
        'ribbon.nii.gz': 'desc-ribbon_T1w.nii.gz',
        # Use either brainmask_fs or brainmask_fs.2.0, depending on which is available.
        'brainmask_fs.nii.gz': 'desc-brain_mask.nii.gz',
        'brainmask_fs.2.0.nii.gz': 'desc-brain_mask.nii.gz',
    }
    copy_dictionary = {}

    for in_str, out_str in ANAT_DICT.items():
        anat_orig = os.path.join(anat_dir_orig, in_str)
        anat_fmriprep = os.path.join(anat_dir_bids, f'{base_anatomical_ents}_{out_str}')
        if os.path.isfile(anat_orig):
            copy_dictionary[anat_orig] = [anat_fmriprep]
        else:
            LOGGER.warning(f'File DNE: {anat_orig}')

    return copy_dictionary


def collect_meshes(anat_dir_orig, anat_dir_bids, sub_id, subses_ents):
    """Collect mesh files from ABCD or HCP-YA derivatives.

    Collects pial and white matter surface meshes for both hemispheres from
    the fsaverage_LR32k subdirectory.

    Parameters
    ----------
    anat_dir_orig : str
        Path to original anatomical derivatives directory. Must contain an
        fsaverage_LR32k subdirectory with surface GIFTI files.
    anat_dir_bids : str
        Path to output BIDS-compliant anatomical directory.
    sub_id : str
        Subject identifier without "sub-" prefix.
    subses_ents : str
        BIDS subject/session entity string (e.g., "sub-01" or "sub-01_ses-1").

    Returns
    -------
    dict
        Dictionary mapping source mesh file paths to lists of destination file paths.
        Files that do not exist in the source directory are omitted with a warning.
    """
    SURFACE_DICT = {
        '{hemi}.pial.32k_fs_LR.surf.gii': 'hemi-{hemi}_pial.surf.gii',
        '{hemi}.white.32k_fs_LR.surf.gii': 'hemi-{hemi}_smoothwm.surf.gii',
    }

    fsaverage_dir_orig = os.path.join(anat_dir_orig, 'fsaverage_LR32k')
    copy_dictionary = {}
    for in_str, out_str in SURFACE_DICT.items():
        for hemi in ['L', 'R']:
            hemi_in_str = in_str.format(hemi=hemi)
            hemi_out_str = out_str.format(hemi=hemi)
            surf_orig = os.path.join(fsaverage_dir_orig, f'{sub_id}.{hemi_in_str}')
            surf_fmriprep = os.path.join(
                anat_dir_bids,
                f'{subses_ents}_space-fsLR_den-32k_{hemi_out_str}',
            )
            if os.path.isfile(surf_orig):
                copy_dictionary[surf_orig] = [surf_fmriprep]
            else:
                LOGGER.warning(f'File DNE: {surf_orig}')

    return copy_dictionary


def collect_morphs(anat_dir_orig, anat_dir_bids, sub_id, subses_ents):
    """Collect and convert morphometry files to CIFTIs.

    Converts hemisphere-specific GIFTI morphometry files (thickness, curvature,
    sulcal depth, myelin maps) to combined CIFTI dscalar files.

    Parameters
    ----------
    anat_dir_orig : str
        Path to original anatomical derivatives directory. Must contain an
        fsaverage_LR32k subdirectory with morphometry GIFTI files.
    anat_dir_bids : str
        Path to output BIDS-compliant anatomical directory where CIFTI files
        will be written.
    sub_id : str
        Subject identifier without "sub-" prefix.
    subses_ents : str
        BIDS subject/session entity string (e.g., "sub-01" or "sub-01_ses-1").

    Returns
    -------
    dict
        Dictionary mapping source GIFTI file paths to output CIFTI file paths.
        Unlike other collect functions, this returns the actual output paths
        (not wrapped in lists) because the files are converted rather than copied.
    """
    SURFACE_DICT = {
        'thickness.32k_fs_LR.shape.gii': 'thickness',
        'corrThickness.32k_fs_LR.shape.gii': 'desc-corrected_thickness',
        'curvature.32k_fs_LR.shape.gii': 'curv',
        'sulc.32k_fs_LR.shape.gii': 'sulc',
        'MyelinMap.32k_fs_LR.func.gii': 'myelinw',
        'SmoothedMyelinMap.32k_fs_LR.func.gii': 'desc-smoothed_myelinw',
    }

    fsaverage_dir_orig = os.path.join(anat_dir_orig, 'fsaverage_LR32k')
    mapping_dictionary = {}
    for in_str, out_str in SURFACE_DICT.items():
        lh_file = os.path.join(fsaverage_dir_orig, f'{sub_id}.L.{in_str}')
        rh_file = os.path.join(fsaverage_dir_orig, f'{sub_id}.R.{in_str}')
        out_file = os.path.join(
            anat_dir_bids,
            f'{subses_ents}_space-fsLR_den-91k_{out_str}.dscalar.nii',
        )

        if not os.path.isfile(lh_file) or not os.path.isfile(rh_file):
            LOGGER.warning(f'File(s) DNE:\n\t{lh_file}\n\t{rh_file}')
            continue

        # Use nprocs because this is run outside of nipype
        interface = CiftiCreateDenseScalar(
            left_metric=lh_file,
            right_metric=rh_file,
            out_file=out_file,
            num_threads=config.nipype.nprocs,
        )
        interface.run()
        mapping_dictionary[lh_file] = out_file
        mapping_dictionary[rh_file] = out_file

    return mapping_dictionary


def collect_hcp_confounds(
    task_dir_orig,
    out_dir,
    prefix,
    work_dir,
    bold_file,
    brainmask_file,
    csf_mask_file,
    wm_mask_file,
):
    """Create confound regressors from ABCD-BIDS or HCP-YA derivatives.

    Parameters
    ----------
    task_dir_orig : str
        Path to folder containing original preprocessing derivatives.
    out_dir : str
        Path to BIDS derivatives 'func' folder, to which the confounds file will be written.
    prefix : str
        The filename prefix to use for the confounds file. E.g., "sub-X_ses-Y_task-rest".
    work_dir : str
        Path to working directory, where temporary files created by nilearn during the masking
        procedure will be stored.
    bold_file : str
        Path to preprocessed BOLD file.
    brainmask_file : str
        Path to brain mask file in same space/resolution as BOLD file.
    csf_mask_file : str
        Path to CSF mask file in same space/resolution as BOLD file.
    wm_mask_file : str
        Path to WM mask file in same space/resolution as BOLD file.
    """
    mvreg_file = os.path.join(task_dir_orig, 'Movement_Regressors.txt')
    assert os.path.isfile(mvreg_file)
    rmsd_file = os.path.join(task_dir_orig, 'Movement_AbsoluteRMS.txt')
    assert os.path.isfile(rmsd_file)

    mvreg = pd.read_csv(mvreg_file, header=None, delimiter=r'\s+')
    mvreg = mvreg.iloc[:, 0:6]
    mvreg.columns = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']

    # Rotations in HCP Movement_Regressors are in degrees; convert to radians.
    for col in mvreg.columns:
        if col.startswith('rot'):
            mvreg[col] = mvreg[col] * np.pi / 180

    _expand_motion_columns(mvreg)
    mvreg['framewise_displacement'] = 0  # Placeholder; XCP-D recalculates FD.

    # Extract mean signals from brain, CSF, and WM masks.
    mean_gs = extract_mean_signal(
        mask=brainmask_file,
        nifti=bold_file,
        work_dir=work_dir,
    )
    mean_csf = extract_mean_signal(
        mask=csf_mask_file,
        nifti=bold_file,
        work_dir=work_dir,
    )
    mean_wm = extract_mean_signal(
        mask=wm_mask_file,
        nifti=bold_file,
        work_dir=work_dir,
    )
    rmsd = np.loadtxt(rmsd_file)

    brainreg = pd.DataFrame(
        {'global_signal': mean_gs, 'white_matter': mean_wm, 'csf': mean_csf, 'rmsd': rmsd}
    )
    for col in ['global_signal', 'white_matter', 'csf']:
        brainreg[f'{col}_derivative1'] = brainreg[col].diff()
        brainreg[f'{col}_derivative1_power2'] = brainreg[f'{col}_derivative1'] ** 2
        brainreg[f'{col}_power2'] = brainreg[col] ** 2

    confounds_df = pd.concat([mvreg, brainreg], axis=1)
    _write_confounds_timeseries(confounds_df, out_dir, prefix)


def collect_ukbiobank_confounds(
    task_dir_orig,
    out_dir,
    prefix,
    work_dir,
    bold_file,
    brainmask_file,
):
    """Create confound regressors from UK Biobank derivatives.

    Parameters
    ----------
    task_dir_orig : str
        Path to folder containing original preprocessing derivatives.
    out_dir : str
        Path to BIDS derivatives 'func' folder, to which the confounds file will be written.
    prefix : str
        The filename prefix to use for the confounds file. E.g., "sub-X_ses-Y_task-rest".
    work_dir : str
        Path to working directory, where temporary files created by nilearn during the masking
        procedure will be stored.
    bold_file : str
        Path to preprocessed BOLD file.
    brainmask_file : str
        Path to brain mask file in same space/resolution as BOLD file.
    """
    # Find necessary files
    par_file = os.path.join(task_dir_orig, 'mc', 'prefiltered_func_data_mcf.par')
    assert os.path.isfile(par_file), os.listdir(os.path.join(task_dir_orig, 'mc'))
    rmsd_file = os.path.join(task_dir_orig, 'mc', 'prefiltered_func_data_mcf_abs.rms')
    assert os.path.isfile(rmsd_file)

    tmpdir = os.path.join(work_dir, prefix)
    os.makedirs(tmpdir, exist_ok=True)

    # Collect motion confounds and their expansions
    normalize_motion = NormalizeMotionParams(format='FSL', in_file=par_file)
    normalize_motion_results = normalize_motion.run(cwd=tmpdir)
    motion_data = np.loadtxt(normalize_motion_results.outputs.out_file)
    confounds_df = pd.DataFrame(
        data=motion_data,
        columns=['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z'],
    )
    _expand_motion_columns(confounds_df)
    confounds_df['framewise_displacement'] = 0  # Placeholder; XCP-D recalculates FD.
    confounds_df['rmsd'] = np.loadtxt(rmsd_file)

    # Global signal is the primary regressor for UKB denoising.
    confounds_df['global_signal'] = extract_mean_signal(
        mask=brainmask_file,
        nifti=bold_file,
        work_dir=work_dir,
    )
    for col in ['global_signal']:
        confounds_df[f'{col}_derivative1'] = confounds_df[col].diff()
        confounds_df[f'{col}_derivative1_power2'] = confounds_df[f'{col}_derivative1'] ** 2
        confounds_df[f'{col}_power2'] = confounds_df[col] ** 2

    _write_confounds_timeseries(confounds_df, out_dir, prefix)


def extract_mean_signal(mask, nifti, work_dir):
    """Extract mean signal within mask from a NIFTI image.

    Uses nilearn's NiftiMasker to extract voxel time series within the mask,
    then computes the mean across voxels at each time point.

    Parameters
    ----------
    mask : str
        Path to binary mask NIFTI file defining the region of interest.
    nifti : str
        Path to 4D NIFTI file from which to extract the signal.
    work_dir : str
        Path to working directory used for nilearn's caching mechanism.

    Returns
    -------
    numpy.ndarray
        1D array of mean signal values, one per time point (volume).
    """
    assert os.path.isfile(mask), f'File DNE: {mask}'
    assert os.path.isfile(nifti), f'File DNE: {nifti}'
    masker = maskers.NiftiMasker(mask_img=mask, memory=work_dir, memory_level=5)
    signals = masker.fit_transform(nifti)
    return np.nanmean(signals, axis=1)


def plot_bbreg(fixed_image, moving_image, contour, out_file='report.svg'):
    """Generate a boundary-based registration quality assurance figure.

    Creates an SVG visualization comparing a fixed anatomical image with a
    moving functional image (typically a bold reference), with optional
    contour overlay to assess registration quality.

    Parameters
    ----------
    fixed_image : str
        Path to the fixed/reference anatomical image (typically T1w).
    moving_image : str
        Path to the moving image (typically BOLD reference or SBRef).
    contour : str or None
        Path to an image to use for contour overlay (typically ribbon.nii.gz),
        or None to skip contour visualization.
    out_file : str, optional
        Path for the output SVG file. Default is 'report.svg'.

    Returns
    -------
    str
        Path to the generated SVG file.
    """
    fixed_image_nii = image.load_img(fixed_image)
    moving_image_nii = image.load_img(moving_image)
    moving_image_nii = image.resample_img(
        moving_image_nii, target_affine=np.eye(3), interpolation='nearest'
    )
    contour_nii = image.load_img(contour) if contour is not None else None

    mask_nii = image.threshold_img(fixed_image_nii, 1e-3)

    n_cuts = 7
    if contour_nii:
        cuts = cuts_from_bbox(contour_nii, cuts=n_cuts)
    else:
        cuts = cuts_from_bbox(mask_nii, cuts=n_cuts)

    compose_view(
        plot_registration(
            fixed_image_nii,
            'fixed-image',
            estimate_brightness=True,
            cuts=cuts,
            label='fixed',
            contour=contour_nii,
            compress='auto',
        ),
        plot_registration(
            moving_image_nii,
            'moving-image',
            estimate_brightness=True,
            cuts=cuts,
            label='moving',
            contour=contour_nii,
            compress='auto',
        ),
        out_file=out_file,
    )
    return out_file


def copy_files_in_dict(copy_dictionary):
    """Copy files according to a source-to-destination mapping dictionary.

    Parameters
    ----------
    copy_dictionary : dict
        Dictionary mapping source file paths (str) to lists of destination
        file paths (list of str). Each source file will be copied to all
        specified destinations. A warning is logged if a source file is
        mapped to multiple destinations.

    Raises
    ------
    ValueError
        If any dictionary value is not a list.
    """
    for file_orig, files_fmriprep in copy_dictionary.items():
        if not isinstance(files_fmriprep, list):
            raise ValueError(
                f'Entry for {file_orig} should be a list, but is a {type(files_fmriprep)}'
            )

        if len(files_fmriprep) > 1:
            LOGGER.warning(f'File used for more than one output: {file_orig}')

        for file_fmriprep in files_fmriprep:
            copy_file(file_orig, file_fmriprep)


def copy_file(src, dst):
    """Copy a file from source path to destination path.

    Only performs the copy if the destination file does not exist or differs
    from the source file (based on shallow file comparison).

    Parameters
    ----------
    src : str
        Path to the source file.
    dst : str
        Path to the destination file.
    """
    if not os.path.exists(dst) or not filecmp.cmp(src, dst):
        shutil.copyfile(src, dst)


def write_json(data, outfile):
    """Write a dictionary to a JSON file.

    Parameters
    ----------
    data : dict
        Dictionary to serialize as JSON.
    outfile : str
        Path to the output JSON file.

    Returns
    -------
    str
        Path to the written JSON file.
    """
    with open(outfile, 'w') as f:
        json.dump(data, f, sort_keys=True, indent=4)

    return outfile
