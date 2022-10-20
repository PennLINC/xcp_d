# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Functions for concatenating scans across runs."""
import fnmatch
import glob
import os
import re
import shutil
import tempfile
from json import loads
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from bids.layout import BIDSLayout, Config
from natsort import natsorted
from nilearn.image import concat_imgs
from nipype.interfaces.ants import ApplyTransforms
from pkg_resources import resource_filename as _pkgres
from templateflow.api import get as get_template

from xcp_d.utils.plot import _get_tr, plot_svgx
from xcp_d.utils.qcmetrics import compute_dvars
from xcp_d.utils.utils import get_transformfile
from xcp_d.utils.write_save import read_ndata

_pybids_spec = loads(Path(_pkgres("xcp_d", "data/nipreps.json")).read_text())
path_patterns = _pybids_spec["default_path_patterns"]


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
    cfg = Config(
        "xcpd",
        entities=_pybids_spec["entities"],
        default_path_patterns=path_patterns,
    )
    layout = BIDSLayout(outputdir, validate=False, derivatives=True, config=cfg)

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
                    layout=layout,
                )
        else:
            concat_func(
                subid=subject,
                fmridir=fmridir,
                outputdir=outputdir,
                work_dir=work_dir,
                layout=layout,
            )


def _get_concat_name(layout, in_file):
    in_file_entities = in_file.get_entities()
    in_file_entities["run"] = None
    concat_file = layout.build_path(
        in_file_entities,
        path_patterns=path_patterns,
        strict=False,
        validate=False,
    )
    return concat_file


def concatenate_niimgs(files, out_file):
    if files[0].extension == ".nii.gz":
        concat_preproc_img = concat_imgs([f.path for f in files])
        concat_preproc_img.to_filename(out_file)
    else:
        combinefile = " -cifti ".join([f.path for f in files])
        os.system('wb_command -cifti-merge ' + out_file + ' -cifti ' + combinefile)


def concatenate_bold(fmridir, outputdir, work_dir, subjects):
    """I still need preproc_files, brain mask, segfile, TR"""
    # NOTE: The config has no effect when derivatives is True :(
    layout = BIDSLayout(outputdir, validate=False, derivatives=True)
    layout_fmriprep = BIDSLayout(fmridir, validate=False, derivatives=True)

    for subject in subjects:
        if subject.startswith("sub-"):
            subject = subject[4:]

        sessions = layout.get_sessions(subject=subject)
        if not sessions:
            sessions = [None]

        for session in sessions:
            base_entities = {
                "subject": subject,
                "session": session,
                "datatype": "func",
            }
            tasks = layout.get_tasks(
                desc="denoised",
                suffix="bold",
                extension=[".nii.gz", ".dtseries.nii"],
                **base_entities,
            )
            for task in tasks:
                task_entities = base_entities[:]
                task_entities["task"] = task

                motion_files = layout.get(
                    desc=["filtered", None],
                    suffix="motion",
                    extension=".tsv",
                    **task_entities,
                )
                if len(motion_files) == 0:
                    continue
                elif len(motion_files) == 1:
                    # Make DCAN HDF5 file from single motion file
                    dcan_df_file = f"{'.'.join(motion_files[0].split('.')[:-1])}-DCAN.hdf5"
                    make_dcan_df([motion_files[0].path], dcan_df_file)
                    continue

                # Make DCAN HDF5 file
                concat_motion_file = _get_concat_name(layout, motion_files[0])
                dcan_df_file = f"{'.'.join(concat_motion_file.split('.')[:-1])}-DCAN.hdf5"
                make_dcan_df([motion_file.path for motion_file in motion_files], dcan_df_file)

                # Concatenate motion files
                concatenate_tsv_files(motion_files, concat_motion_file)

                # Concatenate outlier files
                outlier_files = layout.get(
                    desc=None,
                    suffix="outliers",
                    extension=".tsv",
                    **task_entities,
                )
                concat_outlier_file = _get_concat_name(layout, outlier_files[0])
                concatenate_tsv_files(outlier_files, concat_outlier_file)

                # otherwise, concatenate stuff
                output_spaces = layout.get_spaces(
                    desc="denoised",
                    suffix="bold",
                    extension=[".nii.gz", ".dtseries.nii"],
                    **task_entities,
                )

                for space in output_spaces:
                    space_entities = task_entities[:]
                    space_entities["space"] = space

                    # Mask file
                    mask_files = layout_fmriprep.get(
                        desc=["brain"],
                        suffix="mask",
                        extension=[".nii.gz"],
                        **space_entities,
                    )
                    if len(mask_files) != 1:
                        raise ValueError(f"Too many files found: {mask_files}")

                    mask = mask_files[0].path

                    # Preprocessed BOLD files
                    preproc_files = layout_fmriprep.get(
                        desc=["preproc", None],
                        suffix="bold",
                        extension=[".nii.gz", ".dtseries.nii"],
                        **space_entities,
                    )
                    concat_preproc_file = os.path.join(tempfile.mkdtemp(), "rawdata.nii.gz")
                    concatenate_niimgs(preproc_files, concat_preproc_file)

                    # Calculate DVARS from preprocessed BOLD
                    raw_dvars = []
                    for preproc_file in preproc_files:
                        dvar = compute_dvars(read_ndata(preproc_file, mask))
                        dvar[0] = np.mean(dvar)
                        raw_dvars.append(dvar)
                    raw_dvars = np.concatenate(raw_dvars)

                    # TODO: Use layout_fmriprep for this
                    segfile = get_segfile(preproc_files[0].path)
                    TR = _get_tr(preproc_files[0].path)

                    # Denoised BOLD files
                    bold_files = layout.get(
                        desc="denoised",
                        suffix="bold",
                        extension=[".nii.gz", ".dtseries.nii"],
                        **space_entities,
                    )
                    concat_bold_file = _get_concat_name(layout, bold_files[0])
                    concatenate_niimgs(bold_files, concat_bold_file)

                    # Calculate DVARS from denoised BOLD
                    regressed_dvars = []
                    for bold_file in bold_files:
                        dvar = compute_dvars(read_ndata(bold_file, mask))
                        dvar[0] = np.mean(dvar)
                        regressed_dvars.append(dvar)
                    regressed_dvars = np.concatenate(regressed_dvars)

                    # Concatenate smoothed BOLD files if they exist
                    smooth_bold_files = layout.get(
                        desc="denoisedSmoothed",
                        suffix="bold",
                        extension=[".nii.gz", ".dtseries.nii"],
                        **space_entities,
                    )
                    if len(smooth_bold_files):
                        concat_file = _get_concat_name(layout, smooth_bold_files[0])
                        concatenate_niimgs(smooth_bold_files, concat_file)

                    # Carpet plots
                    carpet_entities = bold_files[0].entities
                    carpet_entities["run"] = None
                    carpet_entities["datatype"] = "figures"
                    carpet_entities["description"] = "precarpetplot"
                    carpet_entities["extension"] = ".svg"
                    precarpet = layout.build_path(
                        carpet_entities,
                        path_patterns=path_patterns,
                        strict=False,
                        validate=False,
                    )
                    carpet_entities["description"] = "postcarpetplot"
                    postcarpet = layout.build_path(
                        carpet_entities,
                        path_patterns=path_patterns,
                        strict=False,
                        validate=False,
                    )

                    # Build figures
                    plot_svgx(
                        rawdata=concat_preproc_file,
                        regressed_data=concat_bold_file,
                        residual_data=concat_bold_file,
                        filtered_motion=concat_motion_file,
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
                    in_fig_entities = preproc_files[0].entities
                    in_fig_entities["datatype"] = "figures"
                    in_fig_entities["extension"] = ".svg"
                    bbreg_fig_in = layout_fmriprep.get(
                        desc="bbregister",
                        **in_fig_entities,
                    )
                    boldref_fig_in = layout_fmriprep.get(
                        desc="bbregister",
                        **in_fig_entities,
                    )

                    out_fig_entities = bold_files[0].entities
                    out_fig_entities["run"] = None
                    out_fig_entities["description"] = "bbregister"
                    out_fig_entities["datatype"] = "figures"
                    out_fig_entities["extension"] = ".svg"
                    bbreg_fig_out = layout.build_path(
                        out_fig_entities,
                        path_patterns=path_patterns,
                        strict=False,
                        validate=False,
                    )
                    out_fig_entities["description"] = "boldref"
                    boldref_fig_out = layout.build_path(
                        out_fig_entities,
                        path_patterns=path_patterns,
                        strict=False,
                        validate=False,
                    )

                    shutil.copy(bbreg_fig_in, bbreg_fig_out)
                    shutil.copy(boldref_fig_in, boldref_fig_out)

                    # Now timeseries files
                    atlases = layout.get_atlases(
                        suffix="timeseries",
                        extension=[".tsv", ".ptseries.nii"],
                        **space_entities,
                    )
                    for atlas in atlases:
                        atlas_timeseries_files = layout.get(
                            atlas=atlas,
                            suffix="timeseries",
                            extension=[".tsv", ".ptseries.nii"],
                            **space_entities,
                        )
                        concat_file = _get_concat_name(layout, atlas_timeseries_files[0])
                        if atlas_timeseries_files[0].extension == ".tsv":
                            concatenate_tsv_files(atlas_timeseries_files, concat_file)
                        elif atlas_timeseries_files[0].extension == ".ptseries.nii":
                            concatenate_niimgs(atlas_timeseries_files, concat_file)
                        else:
                            raise ValueError(f"Unknown extension for {atlas_timeseries_files[0].path}")


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


def _get_motion_file(bold_file):
    path, bold_filename = os.path.split(bold_file)
    # Remove space entity from filenames, because motion files don't have it.
    ENTITIES_TO_REMOVE = ["space", "den"]
    motion_file_base = bold_filename
    for etr in ENTITIES_TO_REMOVE:
        # NOTE: This wouldn't work on sub bc there's no leading underscore.
        motion_file_base = re.sub(f"_{etr}-[a-zA-Z0-9]+", "", motion_file_base)

    # Remove the last entity (desc), suffix, and extension
    motion_file_base = motion_file_base.split("_desc-")[0]

    # Add the path back in
    motion_file_base = os.path.join(path, motion_file_base)

    # This is a hack to work around the fact that the motion file may have a desc
    # entity or not.
    motion_file = motion_file_base + "_desc-filtered_motion.tsv"
    if not os.path.isfile(motion_file):
        motion_file = motion_file_base + "_motion.tsv"

    if not os.path.isfile(motion_file):
        raise ValueError(f"File not found: {motion_file}")

    return motion_file


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
        '_outliers.tsv',
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
        # Select unsmoothed denoised BOLD files
        denoised_task_files = natsorted(
            fnmatch.filter(all_func_files, f'*_task-{task}_*desc-denoised_bold.nii.gz')
        )

        # TODO: Make this robust to different output spaces.
        # Currently, different spaces will be treated like different runs.
        if len(denoised_task_files) == 0:
            # If no files found for this task, move on to the next task.
            continue
        elif len(denoised_task_files) == 1:
            # If only one file is found, there's only one run.
            # In this case we just need to make the DCAN HDF5 file from the filtered motion file.
            motion_file = _get_motion_file(denoised_task_files[0])

            dcan_df_name = f"{'.'.join(motion_file.split('.')[:-1])}-DCAN.hdf5"
            make_dcan_df(motion_file, dcan_df_name)
            continue
        else:
            # If multiple runs of motion files are found, concatenate them and make the HDF5.
            motion_files = [_get_motion_file(f) for f in denoised_task_files]
            concat_motion_file = re.sub("_run-[0-9]+", "", motion_files[0])
            concatenate_tsv_files(motion_files, concat_motion_file)

            dcan_df_name = f"{'.'.join(concat_motion_file.split('.')[:-1])}-DCAN.hdf5"
            make_dcan_df(motion_file, dcan_df_name)

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
            outfile = concatenated_file_base + file_pattern

            if file_pattern.endswith('tsv'):
                concatenate_tsv_files(found_files, outfile)

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

        # Preprocessed BOLD files from fMRIPrep
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
            filtered_motion=concat_motion_file,
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
            f'{concatenated_filename_base}_desc-boldref_bold.svg',
        )

        preproc_file_base = os.path.basename(preproc_files[0]).split('_desc-')[0]
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
        '_outliers.tsv',
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
    denoised_bold_files = fnmatch.filter(all_func_files, '*_desc-denoised_bold.dtseries.nii')
    tasklist = [
        denoised_bold_file.split('task-')[-1].split('_')[0]
        for denoised_bold_file in denoised_bold_files
    ]
    tasklist = sorted(list(set(tasklist)))

    # Concatenate each task separately
    for task in tasklist:
        # Select unsmoothed denoised BOLD files
        denoised_task_files = natsorted(
            fnmatch.filter(all_func_files, f'*_task-{task}_*desc-denoised_bold.dtseries.nii')
        )

        # TODO: Make this robust to different output spaces.
        # Currently, different spaces will be treated like different runs.
        if len(denoised_task_files) == 0:
            # If no files found for this task, move on to the next task.
            continue
        elif len(denoised_task_files) == 1:
            # If only one file is found, there's only one run.
            # In this case we just need to make the DCAN HDF5 file from the filtered motion file.
            motion_file = _get_motion_file(denoised_task_files[0])

            dcan_df_name = f"{'.'.join(motion_file.split('.')[:-1])}-DCAN.hdf5"
            make_dcan_df([motion_file], dcan_df_name)
            continue

        # If multiple runs of motion files are found, concatenate them and make the HDF5.
        motion_files = [_get_motion_file(f) for f in denoised_task_files]

        concat_motion_file = os.path.join(
            os.path.dirname(motion_files[0]),
            re.sub("_run-[0-9]+", "", os.path.basename(motion_files[0])),
        )
        concatenate_tsv_files(motion_files, concat_motion_file)

        dcan_df_name = f"{'.'.join(concat_motion_file.split('.')[:-1])}-DCAN.hdf5"
        make_dcan_df(motion_files, dcan_df_name)

        # Remove run entity from BOLD filename to get concatenated version
        concat_denoised_task_file = os.path.join(
            os.path.dirname(denoised_task_files[0]),
            re.sub("_run-[0-9]+", "", os.path.basename(denoised_task_files[0])),
        )

        bold_filename = os.path.basename(denoised_task_files[0])
        bold_filename_before_run = bold_filename.split('run-')[0]
        bold_filename_after_run = bold_filename.split('run-')[1].partition('_')[-1]

        preproc_base_search_pattern = (
            bold_filename_before_run + '*run*_den-91k_bold.dtseries.nii'
        )
        file_search_base = (
            bold_filename_before_run + 'run-*' + bold_filename_after_run.partition('_desc')[0]
        )

        for file_pattern in datafile:
            found_files = natsorted(glob.glob(file_search_base + file_pattern))
            found_file_dir, found_file_base = os.path.split(found_files[0])

            # Remove run entity from first filename to get concatenated filename
            outfile = os.path.join(found_file_dir, re.sub("_run-[0-9]+", "", found_file_base))

            if file_pattern.endswith('ptseries.nii'):
                combinefile = " -cifti ".join(found_files)
                os.system('wb_command -cifti-merge ' + outfile + ' -cifti ' + combinefile)

            elif file_pattern.endswith('dtseries.nii'):
                combinefile = " -cifti ".join(found_files)
                os.system('wb_command -cifti-merge ' + outfile + ' -cifti ' + combinefile)

                # Calculate DVARS from the unsmoothed, denoised BOLD data
                if file_pattern.endswith('_desc-denoised_bold.dtseries.nii'):
                    regressed_dvars = []
                    for denoised_bold_file in found_files:
                        dvar = compute_dvars(read_ndata(denoised_bold_file))
                        dvar[0] = np.mean(dvar)
                        regressed_dvars.append(dvar)

        raw_dvars = []
        # Preprocessed BOLD files from fMRIPrep
        preproc_files = natsorted(glob.glob(os.path.join(func_dir, preproc_base_search_pattern)))
        for f in preproc_files:
            dvar = compute_dvars(read_ndata(f))
            dvar[0] = np.mean(dvar)
            raw_dvars.append(dvar)

        TR = _get_tr(preproc_files[0])
        rawdata = os.path.join(tempfile.mkdtemp(), 'den-91k_bold.dtseries.nii')
        combinefile = " -cifti ".join(preproc_files)
        os.system('wb_command -cifti-merge ' + rawdata + ' -cifti ' + combinefile)

        concat_preproc_base_name = os.path.basename(concat_denoised_task_file).split("_desc-")[0]
        precarpet = os.path.join(
            figures_dir,
            f'{concat_preproc_base_name}_desc-precarpetplot_bold.svg',
        )
        postcarpet = os.path.join(
            figures_dir,
            f'{concat_preproc_base_name}_desc-postcarpetplot_bold.svg',
        )

        raw_dvars = np.array(raw_dvars).flatten()
        regressed_dvars = np.array(regressed_dvars).flatten()
        plot_svgx(
            rawdata=rawdata,
            regressed_data=concat_denoised_task_file,
            residual_data=concat_denoised_task_file,
            filtered_motion=concat_motion_file,
            raw_dvars=raw_dvars,
            regressed_dvars=regressed_dvars,
            filtered_dvars=regressed_dvars,
            processed_filename=postcarpet,
            unprocessed_filename=precarpet,
            TR=TR,
            work_dir=work_dir,
        )

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
