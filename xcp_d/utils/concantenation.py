# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Functions for concatenating scans across runs."""
import fnmatch
import glob
import os
import re
import shutil
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from natsort import natsorted
from nilearn.image import concat_imgs
from nipype.interfaces.ants import ApplyTransforms
from templateflow.api import get as get_template

from xcp_d.utils.plot import _get_tr, plot_svgx
from xcp_d.utils.qcmetrics import compute_dvars
from xcp_d.utils.utils import get_transformfile
from xcp_d.utils.write_save import read_ndata


def concatenatebold(subjlist, fmridir, outputdir, work_dir):
    """Concatenate BOLD files along time dimension.

    This function does not return anything, but it writes out the concatenated file.

    Parameters
    ----------
    subjlist : list of str
        List of subject identifiers.
    fmridir : str
        Path to the input directory (e.g., fMRIPrep derivatives dataset).
    outputdir : str
        Path to the output directory (i.e., xcp_d derivatives dataset).
    work_dir : str
        The working directory.
    """
    # Ensure each subject ID starts with sub-
    subjlist = [_prefix(subject) for subject in subjlist]
    search_pattern = os.path.join(
        outputdir,
        subjlist[0],
        "**/func/*_desc-denoised*bold*nii*",
    )
    fmr = glob.glob(search_pattern, recursive=True)
    if len(fmr) == 0:
        raise ValueError(f"No files detected in {search_pattern}.")

    cifti = False if fmr[0].endswith('nii.gz') else True
    concat_func = concatenate_cifti if cifti else concatenate_nifti
    fname_pattern = ".dtseries.nii" if cifti else ".nii.gz"

    for subject in subjlist:
        # get session if there
        session_folders = glob.glob(
            os.path.join(
                outputdir,
                subject,
                f'ses-*/func/*_desc-denoised*bold*{fname_pattern}',
            )
        )
        if len(session_folders):
            session_ids = sorted(list(set([_getsesid(sf) for sf in session_folders])))
            for session in session_ids:
                concat_func(
                    subid=subject,
                    fmridir=fmridir,
                    outputdir=outputdir,
                    ses=session,
                    work_dir=work_dir,
                )
        else:
            concat_func(
                subid=subject,
                fmridir=fmridir,
                outputdir=outputdir,
                work_dir=work_dir,
            )


def make_dcan_df(fds_files, name):
    """Create an HDF5-format file containing a DCAN-format dataset.

    Parameters
    ----------
    fds_files : list of str
        List of files from which to extract information.
    name : str
        Name of the HDF5-format file to be created.

    Notes
    -----
    FD_threshold: a number >= 0 that represents the FD threshold used to calculate
    the metrics in this list.
    frame_removal: a binary vector/array the same length as the number of frames
    in the concatenated time series, indicates whether a frame is removed (1) or
    not (0)
    format_string (legacy): a string that denotes how the frames were excluded
    -- uses a notation devised by Avi Snyder
    total_frame_count: a whole number that represents the total number of frames
    in the concatenated series
    remaining_frame_count: a whole number that represents the number of remaining
    frames in the concatenated series
    remaining_seconds: a whole number that represents the amount of time remaining
    after thresholding
    remaining_frame_mean_FD: a number >= 0 that represents the mean FD of the
    remaining frames
    """
    print('making dcan')
    # Temporary workaround for differently-named motion files until we have a BIDS-ish
    # filename construction function
    if "desc-filtered" in fds_files[0]:
        split_str = "_desc-filtered"
    else:
        split_str = "_motion"

    cifti_file = (
        fds_files[0].split(split_str)[0] + '_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii'
    )
    nifti_file = (
        fds_files[0].split(split_str)[0] + '_space-MNI152NLin2009cAsym_desc-denoised_bold.nii.gz'
    )

    if os.path.isfile(cifti_file):
        TR = _get_tr(cifti_file)
    elif os.path.isfile(nifti_file):
        TR = _get_tr(nifti_file)
    else:
        raise Exception(
            f"One or more files not found:\n\t{fds_files[0]}\n\t{cifti_file}\n\t{nifti_file}"
        )

    # Load filtered framewise_displacement values from files and concatenate
    filtered_motion_dfs = [pd.read_table(fds_file) for fds_file in fds_files]
    filtered_motion_df = pd.concat(filtered_motion_dfs, axis=0)
    fd = filtered_motion_df["framewise_displacement"].values

    # NOTE: TS- Maybe close the file object or nest in a with statement?
    dcan = h5py.File(name, "w")
    for thresh in np.linspace(0, 1, 101):
        thresh = np.around(thresh, 2)
        dcan.create_dataset(f"/dcan_motion/fd_{thresh}/skip",
                            data=0,
                            dtype='float')
        dcan.create_dataset(f"/dcan_motion/fd_{thresh}/binary_mask",
                            data=(fd > thresh).astype(int),
                            dtype='float')
        dcan.create_dataset(f"/dcan_motion/fd_{thresh}/threshold",
                            data=thresh,
                            dtype='float')
        dcan.create_dataset(
            f"/dcan_motion/fd_{thresh}/total_frame_count",
            data=len(fd),
            dtype='float')
        dcan.create_dataset(
            f"/dcan_motion/fd_{thresh}/remaining_total_frame_count",
            data=len(fd[fd <= thresh]),
            dtype='float')
        dcan.create_dataset(
            f"/dcan_motion/fd_{thresh}/remaining_seconds",
            data=len(fd[fd <= thresh]) * TR,
            dtype='float')
        dcan.create_dataset(
            f"/dcan_motion/fd_{thresh}/remaining_frame_mean_FD",
            data=(fd[fd <= thresh]).mean(),
            dtype='float')


def concatenate_nifti(subid, fmridir, outputdir, ses=None, work_dir=None):
    """Concatenate NIFTI files along the time dimension.

    This function doesn't return anything, but it writes out the concatenated file.

    TODO: Make file search more general and leverage pybids

    Parameters
    ----------
    subid : str
        Subject identifier.
    fmridir : str
        Path to the input directory (e.g., fMRIPrep derivatives dataset).
    outputdir : str
        Path to the output directory (i.e., xcp_d derivatives dataset).
    ses : str or None, optional
        Session identifier, if applicable. Default is None.
    work_dir : str or None, optional
        Working directory, if available. Default is None.
    """
    # files to be concatenated
    datafile = [
        '_desc-filtered_motion.tsv',  # Must come first to set motion_suffix
        '_desc-denoised_bold.nii.gz',
        '_desc-denoisedSmoothed_bold.nii.gz',
        '_atlas-Glasser_timeseries.tsv',
        '_atlas-Gordon_timeseries.tsv',
        '_atlas-Schaefer117_timeseries.tsv',
        '_atlas-Schaefer617_timeseries.tsv',
        '_atlas-Schaefer217_timeseries.tsv',
        '_atlas-Schaefer717_timeseries.tsv',
        '_atlas-Schaefer317_timeseries.tsv',
        '_atlas-Schaefer817_timeseries.tsv',
        '_atlas-Schaefer417_timeseries.tsv',
        '_atlas-Schaefer917_timeseries.tsv',
        '_atlas-Schaefer517_timeseries.tsv',
        '_atlas-Schaefer1017_timeseries.tsv',
        '_atlas-subcortical_timeseries.tsv',
    ]

    if ses is None:
        all_func_files = glob.glob(os.path.join(outputdir, subid, 'func', '*'))
        func_dir = os.path.join(fmridir, subid, 'func')
        figures_dir = os.path.join(outputdir, subid, 'figures')
    else:
        all_func_files = glob.glob(os.path.join(outputdir, subid, f'ses-{ses}', 'func', '*'))
        func_dir = os.path.join(fmridir, subid, f'ses-{ses}', 'func')
        figures_dir = os.path.join(outputdir, subid , 'figures')

    # extract the task list
    denoised_bold_files = fnmatch.filter(all_func_files, '*_desc-denoised_bold.nii.gz')
    tasklist = [
        denoised_bold_file.split('task-')[-1].split('_')[0]
        for denoised_bold_file in denoised_bold_files
    ]
    tasklist = sorted(list(set(tasklist)))

    # do for each task
    for task in tasklist:
        denoised_task_files = natsorted(
            fnmatch.filter(
                all_func_files,
                f'*{task}*run*_desc-denoised*bold*.nii.gz',
            )
        )

        # denoised_task_files may be in different space like native or MNI or T1w
        if len(denoised_task_files) == 0:
            # If no files found for this task, move on to the next task.
            continue

        regressed_dvars = []

        res = denoised_task_files[0]
        resid = res.split('run-')[1].partition('_')[-1]

        preproc_base_search_pattern = (
            os.path.basename(res.split('run-')[0]) + '*' + resid.partition('_desc')[0]
        )
        file_search_base = res.split('run-')[0] + '*run*' + resid.partition('_desc')[0]
        concatenated_file_base = res.split('run-')[0] + resid.partition('_desc')[0]
        concatenated_filename_base = os.path.basename(concatenated_file_base)

        for file_pattern in datafile:
            found_files = natsorted(glob.glob(file_search_base + file_pattern))

            # This is a hack to work around the fact that the motion file may have a desc
            # entity or not.
            motion_suffix = "_desc-filtered_motion.tsv"
            if file_pattern.endswith("_motion.tsv"):
                # Remove space entity from filenames, because motion files don't have it.
                mot_file_search_base = re.sub("space-[a-zA-Z0-9]+", "", file_search_base)
                mot_concatenated_file_base = re.sub(
                    "_space-[a-zA-Z0-9]+",
                    "",
                    concatenated_file_base,
                )

                found_files = natsorted(glob.glob(mot_file_search_base + file_pattern))
                if not len(found_files):
                    motion_suffix = "_motion.tsv"
                    found_files = natsorted(
                        glob.glob(f"{mot_file_search_base}{motion_suffix}"),
                    )

                assert len(found_files), f"{mot_file_search_base}{motion_suffix}"
                outfile = mot_concatenated_file_base + motion_suffix
            else:
                outfile = concatenated_file_base + file_pattern

            if file_pattern.endswith('tsv'):
                concatenate_tsv_files(found_files, outfile)

            if file_pattern.endswith('_motion.tsv'):
                name = f"{concatenated_file_base}{file_pattern.split('.')[0]}-DCAN.hdf5"
                make_dcan_df(found_files, name)

            elif file_pattern.endswith('nii.gz'):
                mask = natsorted(
                    glob.glob(
                        os.path.join(
                            func_dir,
                            f'{preproc_base_search_pattern}*_desc-brain_mask.nii.gz',
                        ),
                    ),
                )[0]

                combine_img = concat_imgs(found_files)
                combine_img.to_filename(outfile)

                for found_file in found_files:
                    dvar = compute_dvars(read_ndata(found_file, mask))
                    dvar[0] = np.mean(dvar)
                    regressed_dvars.append(dvar)

        preproc_files = natsorted(
            glob.glob(
                os.path.join(
                    func_dir,
                    f'{preproc_base_search_pattern}*_desc-preproc_bold.nii.gz',
                ),
            ),
        )

        mask = natsorted(
            glob.glob(
                os.path.join(
                    func_dir,
                    f'{preproc_base_search_pattern}*_desc-brain_mask.nii.gz',
                ),
            ),
        )[0]

        segfile = get_segfile(preproc_files[0])
        TR = _get_tr(preproc_files[0])

        rawdata = os.path.join(tempfile.mkdtemp(), 'rawdata.nii.gz')

        combine_img = concat_imgs(preproc_files)
        combine_img.to_filename(rawdata)

        precarpet = os.path.join(
            figures_dir,
            f'{concatenated_filename_base}_desc-precarpetplot_bold.svg',
        )
        postcarpet = os.path.join(
            figures_dir,
            f'{concatenated_filename_base}_desc-postcarpetplot_bold.svg',
        )
        raw_dvars = []
        for f in preproc_files:
            dvar = compute_dvars(read_ndata(f, mask))
            dvar[0] = np.mean(dvar)
            raw_dvars.append(dvar)

        plot_svgx(
            rawdata=rawdata,
            regressed_data=f'{concatenated_file_base}_desc-denoised_bold.nii.gz',
            residual_data=f'{concatenated_file_base}_desc-denoised_bold.nii.gz',
            filtered_motion=f'{mot_concatenated_file_base}{motion_suffix}',
            raw_dvars=raw_dvars,
            regressed_dvars=regressed_dvars,
            filtered_dvars=regressed_dvars,
            processed_filename=postcarpet,
            unprocessed_filename=precarpet,
            mask=mask,
            seg_data=segfile,
            TR=TR,
            work_dir=work_dir,
        )

        # link or copy bb svgs
        gboldbbreg = os.path.join(
            figures_dir,
            f'{concatenated_filename_base}_desc-bbregister_bold.svg',
        )
        bboldref = os.path.join(
            figures_dir,
            f'{concatenated_filename_base}__desc-boldref_bold.svg',
        )

        preproc_file_base = os.path.basename(preproc_files[0]).split(
            '_desc-preproc_bold.nii.gz'
        )[0]
        bb1reg = os.path.join(
            figures_dir,
            f'{preproc_file_base}_desc-bbregister_bold.svg',
        )
        bb1ref = os.path.join(
            figures_dir,
            f'{preproc_file_base}_desc-boldref_bold.svg',
        )

        shutil.copy(bb1reg, gboldbbreg)
        shutil.copy(bb1ref, bboldref)


def concatenate_cifti(subid, fmridir, outputdir, ses=None, work_dir=None):
    """Concatenate CIFTI files along the time dimension.

    This function doesn't return anything, but it writes out the concatenated file.

    TODO: Make file search more general and leverage pybids

    Parameters
    ----------
    subid : str
        Subject identifier.
    fmridir : str
        Path to the input directory (e.g., fMRIPrep derivatives dataset).
    outputdir : str
        Path to the output directory (i.e., xcp_d derivatives dataset).
    ses : str or None, optional
        Session identifier, if applicable. Default is None.
    work_dir : str or None, optional
        Working directory, if available. Default is None.
    """
    datafile = [
        '_desc-filtered_motion.tsv',  # Must come first to set motion_suffix
        '_desc-denoised_bold.dtseries.nii',
        '_desc-denoisedSmoothed_bold.dtseries.nii',
        '_atlas-Glasser_den-91k_timeseries.ptseries.nii',
        '_atlas-Gordon_den-91k_timeseries.ptseries.nii',
        '_atlas-Schaefer117_den-91k_timeseries.ptseries.nii',
        '_atlas-Schaefer217_den-91k_timeseries.ptseries.nii',
        '_atlas-Schaefer317_den-91k_timeseries.ptseries.nii',
        '_atlas-Schaefer417_den-91k_timeseries.ptseries.nii',
        '_atlas-Schaefer517_den-91k_timeseries.ptseries.nii',
        '_atlas-Schaefer617_den-91k_timeseries.ptseries.nii',
        '_atlas-Schaefer717_den-91k_timeseries.ptseries.nii',
        '_atlas-Schaefer817_den-91k_timeseries.ptseries.nii',
        '_atlas-Schaefer917_den-91k_timeseries.ptseries.nii',
        '_atlas-Schaefer1017_den-91k_timeseries.ptseries.nii',
        '_atlas-subcortical_den-91k_timeseries.ptseries.nii',
    ]

    if ses is None:
        all_func_files = glob.glob(os.path.join(outputdir, subid, 'func', '*'))
        func_dir = os.path.join(fmridir, subid, 'func')
        figures_dir = os.path.join(outputdir, subid, 'figures')
    else:
        all_func_files = glob.glob(os.path.join(outputdir, subid, f'ses-{ses}', 'func', '*'))
        func_dir = os.path.join(fmridir, subid, f'ses-{ses}', 'func')
        figures_dir = os.path.join(outputdir, subid , 'figures')

    # extract the task list
    denoised_bold_files = fnmatch.filter(
        all_func_files,
        '*den-91k_desc-denoised*bold.dtseries.nii',
    )
    tasklist = [
        denoised_bold_file.split('task-')[-1].split('_')[0]
        for denoised_bold_file in denoised_bold_files
    ]
    tasklist = sorted(list(set(tasklist)))

    # do for each task
    for task in tasklist:
        denoised_task_files = natsorted(
            fnmatch.filter(
                all_func_files,
                f'*{task}*run*den-91k_desc-denoised*bold.dtseries.nii',
            ),
        )

        if len(denoised_task_files) == 0:
            # If no files found for this task, move on to the next task.
            continue

        regressed_dvars = []
        res = denoised_task_files[0]
        resid = res.split('run-')[1].partition('_')[-1]

        preproc_base_search_pattern = (
            os.path.basename(res.split('run-')[0]) + '*run*_den-91k_bold.dtseries.nii'
        )
        file_search_base = res.split('run-')[0] + '*run*' + resid.partition('_desc')[0]
        concatenated_file_base = res.split('run-')[0] + resid.partition('_desc')[0]
        concatenated_filename_base = os.path.basename(concatenated_file_base)

        for file_pattern in datafile:
            found_files = natsorted(glob.glob(file_search_base + file_pattern))

            # This is a hack to work around the fact that the motion file may have a desc
            # entity or not.
            motion_suffix = "_desc-filtered_motion.tsv"
            if file_pattern.endswith("_motion.tsv"):
                # Remove space and den entities from filenames, because motion files don't have it.
                mot_file_search_base = re.sub("space-[a-zA-Z0-9]+", "", file_search_base)
                mot_file_search_base = re.sub("_den-[a-zA-Z0-9]+", "", mot_file_search_base)
                mot_concatenated_file_base = re.sub(
                    "_space-[a-zA-Z0-9]+",
                    "",
                    concatenated_file_base,
                )
                mot_concatenated_file_base = re.sub(
                    "_den-[a-zA-Z0-9]+",
                    "",
                    mot_concatenated_file_base,
                )

                found_files = natsorted(glob.glob(mot_file_search_base + file_pattern))
                if not len(found_files):
                    motion_suffix = "_motion.tsv"
                    found_files = natsorted(
                        glob.glob(f"{mot_file_search_base}{motion_suffix}")
                    )
                    if not len(found_files):
                        raise FileNotFoundError(
                            f"Files not found: {mot_file_search_base}{motion_suffix}"
                        )

                outfile = mot_concatenated_file_base + motion_suffix
            else:
                outfile = concatenated_file_base + file_pattern

            if file_pattern.endswith('ptseries.nii'):
                temp_concatenated_file_base = concatenated_file_base.split('_den-91k')[0]
                outfile = temp_concatenated_file_base + file_pattern
                found_files = natsorted(
                    glob.glob(
                        res.split('run-')[0] + '*run*' + file_pattern
                    )
                )
                combinefile = " -cifti ".join(found_files)
                os.system('wb_command -cifti-merge ' + outfile + ' -cifti ' + combinefile)

            if file_pattern.endswith('_motion.tsv'):
                concatenate_tsv_files(found_files, outfile)

                name = f"{concatenated_file_base}{file_pattern.split('.')[0]}-DCAN.hdf5"
                make_dcan_df(found_files, name)

            if file_pattern.endswith('dtseries.nii'):
                found_files = natsorted(glob.glob(file_search_base + file_pattern))
                combinefile = " -cifti ".join(found_files)
                os.system('wb_command -cifti-merge ' + outfile + ' -cifti ' + combinefile)

                if file_pattern.endswith('_desc-denoised_bold.dtseries.nii'):
                    for denoised_bold_file in found_files:
                        dvar = compute_dvars(read_ndata(denoised_bold_file))
                        dvar[0] = np.mean(dvar)
                        regressed_dvars.append(dvar)

        raw_dvars = []
        preproc_files = natsorted(glob.glob(os.path.join(func_dir, preproc_base_search_pattern)))
        for f in preproc_files:
            dvar = compute_dvars(read_ndata(f))
            dvar[0] = np.mean(dvar)
            raw_dvars.append(dvar)

        TR = _get_tr(preproc_files[0])
        rawdata = os.path.join(tempfile.mkdtemp(), 'den-91k_bold.dtseries.nii')
        combinefile = " -cifti ".join(preproc_files)
        os.system('wb_command -cifti-merge ' + rawdata + ' -cifti ' + combinefile)

        precarpet = os.path.join(
            figures_dir,
            f'{concatenated_filename_base}_desc-precarpetplot_bold.svg',
        )
        postcarpet = os.path.join(
            figures_dir,
            f'{concatenated_filename_base}_desc-postcarpetplot_bold.svg',
        )

        raw_dvars = np.array(raw_dvars).flatten()
        regressed_dvars = np.array(regressed_dvars).flatten()
        plot_svgx(
            rawdata=rawdata,
            regressed_data=f'{concatenated_file_base}_desc-denoised_bold.dtseries.nii',
            residual_data=f'{concatenated_file_base}_desc-denoised_bold.dtseries.nii',
            filtered_motion=f'{mot_concatenated_file_base}{motion_suffix}',
            raw_dvars=raw_dvars,
            regressed_dvars=regressed_dvars,
            filtered_dvars=regressed_dvars,
            processed_filename=postcarpet,
            unprocessed_filename=precarpet,
            TR=TR,
            work_dir=work_dir)

        # link or copy bb svgs
        preproc_file_base = os.path.basename(preproc_files[0]).split(
            '_den-91k_bold.dtseries.nii'
        )[0]
        gboldbbreg = os.path.join(
            figures_dir,
            f'{concatenated_filename_base}_desc-bbregister_bold.svg',
        )
        bboldref = os.path.join(
            figures_dir,
            f'{concatenated_filename_base}_desc-boldref_bold.svg',
        )
        bb1reg = os.path.join(
            figures_dir,
            f'{preproc_file_base}_desc-bbregister_bold.svg',
        )
        bb1ref = os.path.join(
            figures_dir,
            f'{preproc_file_base}_desc-boldref_bold.svg',
        )

        shutil.copy(bb1reg, gboldbbreg)
        shutil.copy(bb1ref, bboldref)


def get_segfile(bold_file):
    """Select the segmentation file associated with a given BOLD file.

    This function identifies the appropriate MNI-space discrete segmentation file for carpet
    plots, then applies the necessary transforms to warp the file into BOLD reference space.
    The warped segmentation file will be written to a temporary file and its path returned.

    Parameters
    ----------
    bold_file : str
        Path to the BOLD file.

    Returns
    -------
    segfile : str
        The associated segmentation file.
    """
    # get transform files
    dd = Path(os.path.dirname(bold_file))
    anatdir = str(dd.parent) + '/anat'

    if Path(anatdir).is_dir():
        mni_to_t1 = glob.glob(
            anatdir + '/*MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5')[0]
    else:
        anatdir = str(dd.parent.parent) + '/anat'
        mni_to_t1 = glob.glob(
            anatdir + '/*MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5')[0]

    transformfilex = get_transformfile(bold_file=bold_file,
                                       mni_to_t1w=mni_to_t1,
                                       t1w_to_native=_t12native(bold_file))

    boldref = bold_file.split('desc-preproc_bold.nii.gz')[0] + 'boldref.nii.gz'

    segfile = tempfile.mkdtemp() + 'segfile.nii.gz'
    carpet = str(
        get_template('MNI152NLin2009cAsym',
                     resolution=1,
                     desc='carpet',
                     suffix='dseg',
                     extension=['.nii', '.nii.gz']))

    # seg_data file to bold space
    at = ApplyTransforms()
    at.inputs.dimension = 3
    at.inputs.input_image = carpet
    at.inputs.reference_image = boldref
    at.inputs.output_image = segfile
    at.inputs.interpolation = 'MultiLabel'
    at.inputs.transforms = transformfilex
    os.system(at.cmdline)

    return segfile


def _t12native(fname):
    """Select T1w-to-scanner transform associated with a given BOLD file.

    TODO: Update names and refactor

    Parameters
    ----------
    fname : str
        The BOLD file from which to identify the transform.

    Returns
    -------
    t12ref : str
        Path to the T1w-to-scanner transform.
    """
    directx = os.path.dirname(fname)
    filename = os.path.basename(fname)
    fileup = filename.split('desc-preproc_bold.nii.gz')[0].split('space-')[0]
    t12ref = directx + '/' + fileup + 'from-T1w_to-scanner_mode-image_xfm.txt'
    return t12ref


def concatenate_tsv_files(tsv_files, fileout):
    """Concatenate framewise displacement time series across files.

    This function doesn't return anything, but it writes out the ``fileout`` file.

    Parameters
    ----------
    tsv_files : list of str
        Paths to TSV files to concatenate.
    fileout : str
        Path to the file that will be written out.
    """
    # TODO: Support headers in timeseries files
    if tsv_files[0].endswith("timeseries.tsv"):
        # timeseries files have no header
        data = [np.loadtxt(tsv_file, delimiter="\t") for tsv_file in tsv_files]
        data = np.vstack(data)
        np.savetxt(fileout, data, fmt='%.5f', delimiter='\t')
    else:
        # other tsv files have a header
        data = [pd.read_table(tsv_file) for tsv_file in tsv_files]
        data = pd.concat(data, axis=0)
        data.to_csv(fileout, sep="\t", index=False)


def _getsesid(filename):
    """Get session id from filename if available.

    Parameters
    ----------
    filename : str
        The BIDS filename from which to extract the session ID.

    Returns
    -------
    ses_id : str or None
        The session ID in the filename.
        If the file does not have a session entity, ``None`` will be returned.
    """
    ses_id = None
    base_filename = os.path.basename(filename)

    file_id = base_filename.split('_')
    for k in file_id:
        if 'ses' in k:
            ses_id = k.split('-')[1]
            break

    return ses_id


def _prefix(subid):
    """Extract or compile subject entity from subject ID.

    Parameters
    ----------
    subid : str
        A subject ID (e.g., 'sub-XX' or just 'XX').

    Returns
    -------
    str
        Subject entity (e.g., 'sub-XX').
    """
    if subid.startswith('sub-'):
        return subid
    return '-'.join(('sub', subid))
