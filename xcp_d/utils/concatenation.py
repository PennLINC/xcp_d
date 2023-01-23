# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Functions for concatenating scans across runs."""
import os
import tempfile
from json import loads
from pathlib import Path

import numpy as np
import pandas as pd
from bids.layout import BIDSLayout, Query
from nilearn.image import concat_imgs
from nipype import logging
from pkg_resources import resource_filename as _pkgres

from xcp_d.utils.bids import _get_tr
from xcp_d.utils.confounds import _infer_dummy_scans, get_confounds_tsv
from xcp_d.utils.doc import fill_doc
from xcp_d.utils.modified_data import _drop_dummy_scans
from xcp_d.utils.plotting import plot_fmri_es
from xcp_d.utils.qcmetrics import compute_dvars, make_dcan_df
from xcp_d.utils.utils import get_segfile
from xcp_d.utils.write_save import read_ndata, write_ndata

_pybids_spec = loads(Path(_pkgres("xcp_d", "data/nipreps.json")).read_text())
path_patterns = _pybids_spec["default_path_patterns"]
LOGGER = logging.getLogger("nipype.interface")


@fill_doc
def concatenate_derivatives(
    fmri_dir,
    output_dir,
    work_dir,
    subjects,
    cifti,
    dcan_qc,
    dummy_scans,
    dummytime=0,
):
    """Concatenate derivatives.

    This function does a lot more than concatenate derivatives.
    It also makes DCAN QC files, creates figures, and copies figures from the preprocessed
    dataset to the post-processed dataset.

    TODO: Move concatenation to *inside* main workflow.
    That way we can feed files in directly instead of searching for them,
    and we can pass the already-initialized fMRIPrep BIDSLayout.

    Parameters
    ----------
    fmri_dir : str
        Path to preprocessed derivatives (not xcpd post-processed derivatives).
    output_dir : str
        Path to location of xcpd derivatives.
    work_dir : str
        Working directory.
    subjects : list of str
        List of subjects to run concatenation on.
    cifti : bool
        Whether xcpd was run on CIFTI files or not.
    dcan_qc : bool
        Whether to perform DCAN QC or not.
    %(dummy_scans)s
    %(dummytime)s
    """
    LOGGER.debug("Starting concatenation workflow.")

    # NOTE: The config has no effect when derivatives is True :(
    # At least for pybids ~0.15.1.
    # TODO: Find a way to support the xcpd config file in the BIDSLayout.
    layout_xcpd = BIDSLayout(
        output_dir,
        validate=False,
        derivatives=True,
    )
    layout_fmriprep = BIDSLayout(
        fmri_dir,
        validate=False,
        derivatives=True,
        config=["bids", "derivatives"],
    )

    if cifti:
        tsv_extensions = [".ptseries.nii"]
        img_extensions = [".dtseries.nii"]
    else:
        tsv_extensions = [".tsv"]
        img_extensions = [".nii.gz"]

    for subject in subjects:
        if subject.startswith("sub-"):
            subject = subject[4:]

        LOGGER.debug(f"Concatenating subject {subject}")

        sessions = layout_xcpd.get_sessions(subject=subject)
        if not sessions:
            sessions = [None]

        for session in sessions:
            LOGGER.debug(f"Concatenating session {session}")

            base_entities = {
                "subject": subject,
                "session": session,
                "datatype": "func",
            }
            tasks = layout_xcpd.get_tasks(
                desc="denoised",
                suffix="bold",
                extension=img_extensions,
                **base_entities,
            )
            for task in tasks:
                LOGGER.debug(f"Concatenating task {task}")

                task_entities = base_entities.copy()
                task_entities["task"] = task

                motion_files = layout_xcpd.get(
                    run=Query.ANY,
                    desc=["filtered", None],
                    suffix="motion",
                    extension=".tsv",
                    **task_entities,
                )
                if len(motion_files) == 0:
                    # No run-wise motion files exist, so task probably only has one run.
                    motion_files = layout_xcpd.get(
                        run=None,
                        desc=["filtered", None],
                        suffix="motion",
                        extension=".tsv",
                        **task_entities,
                    )
                    if len(motion_files) == 1:
                        LOGGER.debug(f"Only one run found for task {task}")
                        continue
                    elif len(motion_files) == 0:
                        LOGGER.warning(f"No motion files found for task {task}")
                        continue
                    else:
                        raise ValueError(
                            "Found multiple motion files when there should only be one: "
                            f"{motion_files}"
                        )

                # Infer some entities from the detected files
                runs = layout_xcpd.get_runs(absolute_paths=motion_files)

                # Get TR from one of the preproc files
                # NOTE: We're assuming that the TR is the same across runs.
                preproc_files = layout_fmriprep.get(
                    desc=["preproc", None],
                    run=runs,
                    suffix="bold",
                    extension=img_extensions,
                    **task_entities,
                )
                TR = _get_tr(preproc_files[0].path)

                # Concatenate motion files
                motion_file_names = ", ".join([motion_file.path for motion_file in motion_files])
                LOGGER.debug(f"Concatenating motion files: {motion_file_names}")
                concat_motion_file = _get_concat_name(layout_xcpd, motion_files[0])
                concatenate_tsv_files(motion_files, concat_motion_file)

                # Make DCAN HDF5 file from concatenated motion file
                if dcan_qc:
                    concat_dcan_df_file = concat_motion_file.replace(
                        "desc-filtered_motion.tsv",
                        "desc-dcan_qc.hdf5",
                    ).replace(
                        "_motion.tsv",
                        "desc-dcan_qc.hdf5",
                    )
                    make_dcan_df(concat_motion_file, concat_dcan_df_file, TR)

                # Concatenate outlier files
                outlier_files = layout_xcpd.get(
                    run=runs,
                    desc=None,
                    suffix="outliers",
                    extension=".tsv",
                    **task_entities,
                )
                outlier_file_names = ", ".join(
                    [outlier_file.path for outlier_file in outlier_files]
                )
                LOGGER.debug(f"Concatenating outlier files: {outlier_file_names}")
                concat_outlier_file = _get_concat_name(layout_xcpd, outlier_files[0])
                concat_outlier_file = concatenate_tsv_files(outlier_files, concat_outlier_file)

                # otherwise, concatenate stuff
                output_spaces = layout_xcpd.get_spaces(
                    run=runs,
                    desc="denoised",
                    suffix="bold",
                    extension=img_extensions,
                    **task_entities,
                )

                for space in output_spaces:
                    LOGGER.debug(f"Concatenating files in {space} space")
                    space_entities = task_entities.copy()
                    space_entities["space"] = space

                    # Concatenate denoised BOLD files
                    denoised_files = layout_xcpd.get(
                        run=runs,
                        desc="denoised",
                        suffix="bold",
                        extension=img_extensions,
                        **space_entities,
                    )
                    concat_denoised_file = _get_concat_name(layout_xcpd, denoised_files[0])
                    LOGGER.debug(f"Concatenating postprocessed file: {concat_denoised_file}")
                    _concatenate_niimgs(denoised_files, concat_denoised_file, dummy_scans=0)

                    # Concatenate smoothed BOLD files if they exist
                    smooth_denoised_files = layout_xcpd.get(
                        run=runs,
                        desc="denoisedSmoothed",
                        suffix="bold",
                        extension=img_extensions,
                        **space_entities,
                    )
                    if len(smooth_denoised_files):
                        concat_smooth_denoised_file = _get_concat_name(
                            layout_xcpd, smooth_denoised_files[0]
                        )
                        LOGGER.debug(
                            "Concatenating smoothed postprocessed file: "
                            f"{concat_smooth_denoised_file}"
                        )
                        _concatenate_niimgs(
                            smooth_denoised_files,
                            concat_smooth_denoised_file,
                            dummy_scans=0,
                        )

                    # Executive summary carpet plots
                    if dcan_qc:
                        tmpdir = tempfile.mkdtemp()

                        # Concatenate preprocessed BOLD files
                        preproc_files = layout_fmriprep.get(
                            run=runs,
                            desc=["preproc", None],
                            suffix="bold",
                            extension=img_extensions,
                            **space_entities,
                        )
                        concat_preproc_file = os.path.join(
                            tmpdir,
                            f"rawdata{preproc_files[0].extension}",
                        )
                        preproc_files_str = "\n\t".join(
                            [preproc_file.path for preproc_file in preproc_files]
                        )
                        LOGGER.debug(
                            f"Concatenating preprocessed file ({concat_preproc_file}) from\n"
                            f"{preproc_files_str}"
                        )

                        # Get TR for BOLD files in this query (e.g., task/acq/ses)
                        # We assume that TR is the same across runs,
                        # which isn't a solid assumption, since we don't account for "acq".
                        TR = _get_tr(preproc_files[0].path)

                        if dummy_scans == 0 and dummytime != 0:
                            dummy_scans = int(np.ceil(dummytime / TR))

                        if dummy_scans == "auto":
                            runwise_dummy_scans = []
                            for i_file, preproc_file in enumerate(preproc_files):
                                confounds_file = get_confounds_tsv(preproc_file.path)
                                if not os.path.isfile(confounds_file):
                                    raise FileNotFoundError(
                                        f"Confounds file not found: {confounds_file}"
                                    )

                                runwise_dummy_scans.append(
                                    _infer_dummy_scans(dummy_scans, confounds_file)
                                )

                        elif isinstance(dummy_scans, int):
                            runwise_dummy_scans = [dummy_scans] * len(preproc_files)

                        # Concatenate preprocessed files, but drop dummy scans from each run
                        _concatenate_niimgs(
                            preproc_files,
                            concat_preproc_file,
                            dummy_scans=runwise_dummy_scans,
                        )

                        # Get mask and dseg files for loading data and calculating DVARS.
                        if not cifti:
                            mask_files = layout_fmriprep.get(
                                run=1,
                                desc=["brain"],
                                suffix="mask",
                                extension=[".nii.gz"],
                                **space_entities,
                            )
                            if len(mask_files) == 0:
                                raise ValueError(
                                    f"No mask files found for {preproc_files[0].path}"
                                )
                            elif len(mask_files) != 1:
                                LOGGER.warning(
                                    f"More than one mask file found. Using first: {mask_files}"
                                )

                            mask = mask_files[0].path
                            # TODO: Use layout_fmriprep for this
                            segfile = get_segfile(preproc_files[0].path)
                        else:
                            mask = None
                            segfile = None

                        # Create a censored version of the denoised file,
                        # because denoised_file is from before interpolation.
                        concat_censored_file = os.path.join(
                            tmpdir,
                            f"filtereddata{preproc_files[0].extension}",
                        )
                        tmask_df = pd.read_table(concat_outlier_file)
                        tmask_arr = tmask_df["framewise_displacement"].values
                        tmask_bool = ~tmask_arr.astype(bool)
                        temp_data_arr = read_ndata(
                            datafile=concat_denoised_file,
                            maskfile=mask,
                        )
                        temp_data_arr = temp_data_arr[:, tmask_bool]
                        write_ndata(
                            data_matrix=temp_data_arr,
                            template=concat_denoised_file,
                            filename=concat_censored_file,
                            mask=mask,
                            TR=TR,
                        )

                        # Calculate DVARS from preprocessed BOLD
                        raw_dvars = []
                        for i_file, preproc_file in enumerate(preproc_files):
                            dvar = compute_dvars(read_ndata(preproc_file.path, mask))
                            dvar[0] = np.mean(dvar)
                            dvar = dvar[runwise_dummy_scans[i_file] :]
                            raw_dvars.append(dvar)

                        raw_dvars = np.concatenate(raw_dvars)

                        # Censor DVARS
                        raw_dvars = raw_dvars[tmask_bool]

                        # Calculate DVARS from denoised BOLD
                        denoised_dvars = []
                        for denoised_file in denoised_files:
                            dvar = compute_dvars(read_ndata(denoised_file.path, mask))
                            dvar[0] = np.mean(dvar)
                            denoised_dvars.append(dvar)

                        denoised_dvars = np.concatenate(denoised_dvars)

                        # Censor DVARS
                        denoised_dvars = denoised_dvars[tmask_bool]

                        # Start on carpet plots
                        LOGGER.debug("Generating carpet plots")
                        carpet_entities = denoised_files[0].get_entities()
                        carpet_entities = _sanitize_entities(carpet_entities)
                        carpet_entities["run"] = None
                        carpet_entities["datatype"] = "figures"
                        carpet_entities["extension"] = ".svg"

                        carpet_entities["desc"] = "precarpetplot"
                        precarpet = layout_xcpd.build_path(
                            carpet_entities,
                            path_patterns=path_patterns,
                            strict=False,
                            validate=False,
                        )

                        carpet_entities["desc"] = "postcarpetplot"
                        postcarpet = layout_xcpd.build_path(
                            carpet_entities,
                            path_patterns=path_patterns,
                            strict=False,
                            validate=False,
                        )

                        LOGGER.debug("Starting plot_fmri_es")
                        plot_fmri_es(
                            preprocessed_file=concat_preproc_file,
                            residuals_file=concat_censored_file,
                            denoised_file=concat_denoised_file,
                            dummy_scans=0,
                            tmask=concat_outlier_file,
                            filtered_motion=concat_motion_file,
                            raw_dvars=raw_dvars,
                            residuals_dvars=denoised_dvars,
                            denoised_dvars=denoised_dvars,
                            processed_filename=postcarpet,
                            unprocessed_filename=precarpet,
                            mask=mask,
                            seg_data=segfile,
                            TR=TR,
                            work_dir=work_dir,
                        )
                        LOGGER.debug("plot_fmri_es done")

                    # Now timeseries files
                    atlases = layout_xcpd.get_atlases(
                        suffix="timeseries",
                        extension=tsv_extensions,
                        **space_entities,
                    )
                    for atlas in atlases:
                        LOGGER.debug(f"Concatenating time series files for atlas {atlas}")
                        atlas_timeseries_files = layout_xcpd.get(
                            run=runs,
                            atlas=atlas,
                            suffix="timeseries",
                            extension=tsv_extensions,
                            **space_entities,
                        )
                        concat_file = _get_concat_name(layout_xcpd, atlas_timeseries_files[0])
                        if atlas_timeseries_files[0].extension == ".tsv":
                            concatenate_tsv_files(atlas_timeseries_files, concat_file)
                        elif atlas_timeseries_files[0].extension == ".ptseries.nii":
                            _concatenate_niimgs(
                                atlas_timeseries_files,
                                concat_file,
                                dummy_scans=0,
                            )
                        else:
                            raise ValueError(
                                f"Unknown extension for {atlas_timeseries_files[0].path}"
                            )


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
        np.savetxt(fileout, data, fmt="%.5f", delimiter="\t")
    else:
        # other tsv files have a header
        data = [pd.read_table(tsv_file.path) for tsv_file in tsv_files]
        data = pd.concat(data, axis=0)
        data.to_csv(fileout, sep="\t", index=False)
    return fileout


def _get_concat_name(layout_xcpd, in_file):
    """Drop run entity from filename to get concatenated version."""
    in_file_entities = in_file.get_entities()
    in_file_entities["run"] = None
    concat_file = layout_xcpd.build_path(
        in_file_entities,
        path_patterns=path_patterns,
        strict=False,
        validate=False,
    )
    return concat_file


def _sanitize_entities(dict_):
    """Ensure "description" isn't in dictionary keys."""
    dict_ = dict_.copy()
    if "description" in dict_.keys():
        dict_["desc"] = dict_["description"]
        del dict_["description"]

    return dict_


def _concatenate_niimgs(files, out_file, dummy_scans=0):
    """Concatenate niimgs.

    This is generally a very simple proposition (especially with niftis).
    However, sometimes we need to account for dummy scans- especially when we want to use
    the non-steady-state volume indices from fMRIPrep.

    Parameters
    ----------
    files : :obj:`list` of :obj:`bids.layout.models.BIDSImageFile`
        List of BOLD files to concatenate over the time dimension.
    out_file : :obj:`str`
        The concatenated file to write out.
    dummy_scans : int or list of int, optional
        The number of dummy scans to drop from the beginning of each file before concatenation.
        If None (default), no volumes will be dropped.
        If an integer, the same number of volumes will be dropped from each file.
        If "auto", this function will attempt to find each file's associated confounds file,
        load it, and determine the number of non-steady-state volumes estimated by the
        preprocessing workflow.
    """
    assert isinstance(dummy_scans, (int, list))

    is_nifti = files[0].extension == ".nii.gz"
    use_temp_files = False

    if isinstance(dummy_scans, list):
        assert all([isinstance(val, int) for val in dummy_scans])
        runwise_dummy_scans = dummy_scans
    else:
        runwise_dummy_scans = [dummy_scans] * len(files)

    if dummy_scans != 0:
        bold_imgs = [
            _drop_dummy_scans(f.path, dummy_scans=runwise_dummy_scans[i])
            for i, f in enumerate(files)
        ]
        if is_nifti:
            bold_files = bold_imgs
        else:
            # Create temporary files for cifti images
            use_temp_files = True
            bold_files = []
            for i_img, img in enumerate(bold_imgs):
                temporary_file = f"temp_{i_img}{files[0].extension}"
                img.to_filename(temporary_file)
                bold_files.append(temporary_file)

    else:
        bold_files = [f.path for f in files]

    if is_nifti:
        concat_preproc_img = concat_imgs(bold_files)
        concat_preproc_img.to_filename(out_file)
    else:
        os.system(f"wb_command -cifti-merge {out_file} -cifti {' -cifti '.join(bold_files)}")

        if use_temp_files:
            # Delete temporary files
            for bold_file in bold_files:
                os.remove(bold_file)
