# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Functions for concatenating scans across runs."""
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
from bids.layout import BIDSLayout
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
    """Concatenate niimgs."""
    if files[0].extension == ".nii.gz":
        concat_preproc_img = concat_imgs([f.path for f in files])
        concat_preproc_img.to_filename(out_file)
    else:
        combinefile = " -cifti ".join([f.path for f in files])
        os.system('wb_command -cifti-merge ' + out_file + ' -cifti ' + combinefile)


def concatenate_bold(fmridir, outputdir, work_dir, subjects, cifti):
    """I still need preproc_files, brain mask, segfile, TR."""
    # NOTE: The config has no effect when derivatives is True :(
    layout = BIDSLayout(outputdir, validate=False, derivatives=True)
    layout_fmriprep = BIDSLayout(fmridir, validate=False, derivatives=True)

    if cifti:
        tsv_extensions = [".ptseries.nii"]
        img_extensions = [".dtseries.nii"]
    else:
        tsv_extensions = [".tsv"]
        img_extensions = [".nii.gz"]

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
                extension=img_extensions,
                **base_entities,
            )
            for task in tasks:
                task_entities = base_entities.copy()
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
                    extension=img_extensions,
                    **task_entities,
                )

                for space in output_spaces:
                    space_entities = task_entities.copy()
                    space_entities["space"] = space

                    # Preprocessed BOLD files
                    preproc_files = layout_fmriprep.get(
                        desc=["preproc", None],
                        suffix="bold",
                        extension=img_extensions,
                        **space_entities,
                    )
                    concat_preproc_file = os.path.join(
                        tempfile.mkdtemp(),
                        f"rawdata{preproc_files[0].extension}",
                    )
                    concatenate_niimgs(preproc_files, concat_preproc_file)

                    if not cifti:
                        # Mask file
                        mask_files = layout_fmriprep.get(
                            desc=["brain"],
                            suffix="mask",
                            extension=[".nii.gz"],
                            **space_entities,
                        )
                        if len(mask_files) == 0:
                            raise ValueError(f"No mask files found for {preproc_files[0].path}")
                        elif len(mask_files) != 1:
                            print(f"Too many files found: {mask_files}")

                        mask = mask_files[0].path
                        # TODO: Use layout_fmriprep for this
                        segfile = get_segfile(preproc_files[0].path)
                    else:
                        mask = None
                        segfile = None

                    # Calculate DVARS from preprocessed BOLD
                    raw_dvars = []
                    for preproc_file in preproc_files:
                        dvar = compute_dvars(read_ndata(preproc_file.path, mask))
                        dvar[0] = np.mean(dvar)
                        raw_dvars.append(dvar)
                    raw_dvars = np.concatenate(raw_dvars)

                    TR = _get_tr(preproc_files[0].path)

                    # Denoised BOLD files
                    bold_files = layout.get(
                        desc="denoised",
                        suffix="bold",
                        extension=img_extensions,
                        **space_entities,
                    )
                    concat_bold_file = _get_concat_name(layout, bold_files[0])
                    concatenate_niimgs(bold_files, concat_bold_file)

                    # Calculate DVARS from denoised BOLD
                    regressed_dvars = []
                    for bold_file in bold_files:
                        dvar = compute_dvars(read_ndata(bold_file.path, mask))
                        dvar[0] = np.mean(dvar)
                        regressed_dvars.append(dvar)
                    regressed_dvars = np.concatenate(regressed_dvars)

                    # Concatenate smoothed BOLD files if they exist
                    smooth_bold_files = layout.get(
                        desc="denoisedSmoothed",
                        suffix="bold",
                        extension=img_extensions,
                        **space_entities,
                    )
                    if len(smooth_bold_files):
                        concat_file = _get_concat_name(layout, smooth_bold_files[0])
                        concatenate_niimgs(smooth_bold_files, concat_file)

                    # Carpet plots
                    carpet_entities = bold_files[0].get_entities()
                    carpet_entities["run"] = None
                    carpet_entities["datatype"] = "figures"
                    carpet_entities["desc"] = "precarpetplot"
                    carpet_entities["extension"] = ".svg"
                    precarpet = layout.build_path(
                        carpet_entities,
                        path_patterns=path_patterns,
                        strict=False,
                        validate=False,
                    )
                    carpet_entities["desc"] = "postcarpetplot"
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
                    in_fig_entities = preproc_files[0].get_entities()
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

                    out_fig_entities = bold_files[0].get_entities()
                    out_fig_entities["run"] = None
                    out_fig_entities["desc"] = "bbregister"
                    out_fig_entities["datatype"] = "figures"
                    out_fig_entities["extension"] = ".svg"
                    bbreg_fig_out = layout.build_path(
                        out_fig_entities,
                        path_patterns=path_patterns,
                        strict=False,
                        validate=False,
                    )
                    out_fig_entities["desc"] = "boldref"
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
                        extension=tsv_extensions,
                        **space_entities,
                    )
                    for atlas in atlases:
                        atlas_timeseries_files = layout.get(
                            atlas=atlas,
                            suffix="timeseries",
                            extension=tsv_extensions,
                            **space_entities,
                        )
                        concat_file = _get_concat_name(layout, atlas_timeseries_files[0])
                        if atlas_timeseries_files[0].extension == ".tsv":
                            concatenate_tsv_files(atlas_timeseries_files, concat_file)
                        elif atlas_timeseries_files[0].extension == ".ptseries.nii":
                            concatenate_niimgs(atlas_timeseries_files, concat_file)
                        else:
                            raise ValueError(
                                f"Unknown extension for {atlas_timeseries_files[0].path}"
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
    if tsv_files[0].path.endswith("timeseries.tsv"):
        # timeseries files have no header
        data = [np.loadtxt(tsv_file.path, delimiter="\t") for tsv_file in tsv_files]
        data = np.vstack(data)
        np.savetxt(fileout, data, fmt='%.5f', delimiter='\t')
    else:
        # other tsv files have a header
        data = [pd.read_table(tsv_file.path) for tsv_file in tsv_files]
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
