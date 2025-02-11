# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Functions to support ingression of non-BIDS preprocessing derivatives."""

import json
import os

import numpy as np
from nilearn import image, maskers
from nipype import logging
from niworkflows.interfaces.confounds import NormalizeMotionParams

from xcp_d import config
from xcp_d.interfaces.workbench import CiftiCreateDenseScalar

LOGGER = logging.getLogger('nipype.utils')


def collect_anatomical_files(anat_dir_orig, anat_dir_bids, base_anatomical_ents):
    """Collect anatomical files from ABCD or HCP-YA derivatives."""
    ANAT_DICT = {
        # XXX: Why have T1w here and T1w_restore for HCP?
        'T1w.nii.gz': 'desc-preproc_T1w.nii.gz',
        'brainmask_fs.nii.gz': 'desc-brain_mask.nii.gz',
        'ribbon.nii.gz': 'desc-ribbon_T1w.nii.gz',
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
    """Collect mesh files from ABCD or HCP-YA derivatives."""
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
    """Collect and convert morphometry files to CIFTIs."""
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
    import pandas as pd

    mvreg_file = os.path.join(task_dir_orig, 'Movement_Regressors.txt')
    if not os.path.isfile(mvreg_file):
        raise ValueError(f'File does not exist: {mvreg_file}')
    rmsd_file = os.path.join(task_dir_orig, 'Movement_AbsoluteRMS.txt')
    if not os.path.isfile(rmsd_file):
        raise ValueError(f'File does not exist: {rmsd_file}')

    mvreg = pd.read_csv(mvreg_file, header=None, delimiter=r'\s+')

    # Only use the first six columns
    mvreg = mvreg.iloc[:, 0:6]
    mvreg.columns = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']

    # convert rotations from degrees to radians
    rot_columns = [c for c in mvreg.columns if c.startswith('rot')]
    for col in rot_columns:
        mvreg[col] = mvreg[col] * np.pi / 180

    # get derivatives of motion columns
    columns = mvreg.columns.tolist()
    for col in columns:
        mvreg[f'{col}_derivative1'] = mvreg[col].diff()

    # get powers
    columns = mvreg.columns.tolist()
    for col in columns:
        mvreg[f'{col}_power2'] = mvreg[col] ** 2

    # Use dummy column for framewise displacement, which will be recalculated by XCP-D.
    mvreg['framewise_displacement'] = 0

    # use masks: brain, csf, and wm mask to extract timeseries
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

    # get derivatives and powers
    brainreg['global_signal_derivative1'] = brainreg['global_signal'].diff()
    brainreg['white_matter_derivative1'] = brainreg['white_matter'].diff()
    brainreg['csf_derivative1'] = brainreg['csf'].diff()

    brainreg['global_signal_derivative1_power2'] = brainreg['global_signal_derivative1'] ** 2
    brainreg['global_signal_power2'] = brainreg['global_signal'] ** 2

    brainreg['white_matter_derivative1_power2'] = brainreg['white_matter_derivative1'] ** 2
    brainreg['white_matter_power2'] = brainreg['white_matter'] ** 2

    brainreg['csf_derivative1_power2'] = brainreg['csf_derivative1'] ** 2
    brainreg['csf_power2'] = brainreg['csf'] ** 2

    # Merge the two DataFrames
    confounds_df = pd.concat([mvreg, brainreg], axis=1)

    # write out the confounds
    regressors_tsv_fmriprep = os.path.join(
        out_dir,
        f'{prefix}_desc-confounds_timeseries.tsv',
    )
    confounds_df.to_csv(regressors_tsv_fmriprep, sep='\t', na_rep='n/a', index=False)

    regressors_json_fmriprep = os.path.join(
        out_dir,
        f'{prefix}_desc-confounds_timeseries.json',
    )
    confounds_dict = {col: {'Description': ''} for col in confounds_df.columns}
    write_json(confounds_dict, regressors_json_fmriprep)


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
    import os

    import pandas as pd

    # Find necessary files
    par_file = os.path.join(task_dir_orig, 'mc', 'prefiltered_func_data_mcf.par')
    if not os.path.isfile(par_file):
        raise ValueError(f'File does not exist: {par_file}')
    rmsd_file = os.path.join(task_dir_orig, 'mc', 'prefiltered_func_data_mcf_abs.rms')
    if not os.path.isfile(rmsd_file):
        raise ValueError(f'File does not exist: {rmsd_file}')

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

    columns = confounds_df.columns.tolist()
    for col in columns:
        new_col = f'{col}_derivative1'
        confounds_df[new_col] = confounds_df[col].diff()

    columns = confounds_df.columns.tolist()
    for col in columns:
        new_col = f'{col}_power2'
        confounds_df[new_col] = confounds_df[col] ** 2

    # Use dummy column for framewise displacement, which will be recalculated by XCP-D.
    confounds_df['framewise_displacement'] = 0

    # Add RMS
    rmsd = np.loadtxt(rmsd_file)
    confounds_df['rmsd'] = rmsd

    # Collect global signal (the primary regressor used for denoising UKB data,
    # since the data are already denoised).
    confounds_df['global_signal'] = extract_mean_signal(
        mask=brainmask_file,
        nifti=bold_file,
        work_dir=work_dir,
    )
    # get derivatives and powers
    confounds_df['global_signal_derivative1'] = confounds_df['global_signal'].diff()
    confounds_df['global_signal_derivative1_power2'] = (
        confounds_df['global_signal_derivative1'] ** 2
    )
    confounds_df['global_signal_power2'] = confounds_df['global_signal'] ** 2

    # write out the confounds
    regressors_tsv_fmriprep = os.path.join(
        out_dir,
        f'{prefix}_desc-confounds_timeseries.tsv',
    )
    confounds_df.to_csv(regressors_tsv_fmriprep, sep='\t', na_rep='n/a', index=False)

    regressors_json_fmriprep = os.path.join(
        out_dir,
        f'{prefix}_desc-confounds_timeseries.json',
    )
    confounds_dict = {col: {'Description': ''} for col in confounds_df.columns}
    write_json(confounds_dict, regressors_json_fmriprep)


def extract_mean_signal(mask, nifti, work_dir):
    """Extract mean signal within mask from NIFTI."""
    if not os.path.isfile(mask):
        raise ValueError(f'File does not exist: {mask}')
    if not os.path.isfile(nifti):
        raise ValueError(f'File does not exist: {nifti}')
    masker = maskers.NiftiMasker(mask_img=mask, memory=work_dir, memory_level=5)
    signals = masker.fit_transform(nifti)
    return np.mean(signals, axis=1)


def plot_bbreg(fixed_image, moving_image, contour, out_file='report.svg'):
    """Plot bbref_fig_fmriprep results."""
    import numpy as np
    from niworkflows.viz.utils import compose_view, cuts_from_bbox, plot_registration

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
    """Copy files in dictionary."""
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
    """Copy a file from source to dest.

    source and dest must be file-like objects,
    i.e. any object with a read or write method, like for example StringIO.
    """
    import filecmp
    import shutil

    if not os.path.exists(dst) or not filecmp.cmp(src, dst):
        shutil.copyfile(src, dst)


def write_json(data, outfile):
    """Write dictionary to JSON file."""
    with open(outfile, 'w') as f:
        json.dump(data, f, sort_keys=True, indent=4)

    return outfile
