"""Functions to convert preprocessed UK Biobank BOLD data to BIDS derivatives format."""

import glob
import json
import os

import pandas as pd
from nipype import logging
from nipype.interfaces.fsl.preprocess import ApplyWarp

from xcp_d.data import load as load_data
from xcp_d.ingression.utils import (
    collect_ukbiobank_confounds,
    copy_files_in_dict,
    write_json,
)
from xcp_d.utils.filemanip import ensure_list

LOGGER = logging.getLogger("nipype.utils")


def convert_ukb2bids(in_dir, out_dir, participant_ids=None, bids_filters={}):
    """Convert UK Biobank derivatives to BIDS-compliant derivatives.

    Parameters
    ----------
    in_dir : str
        Path to UK Biobank derivatives.
    out_dir : str
        Path to the output BIDS-compliant derivatives folder.
    participant_ids : None or list of str
        List of participant IDs to run conversion on.
        The participant IDs must not have the "sub-" prefix.
        If None, the function will search for all subjects in ``in_dir`` and convert all of them.
    bids_filters : dict
        Filters to apply to select files to convert.
        The only filter that is currently supported is {"bold": {"session": []}}.

    Returns
    -------
    participant_ids : list of str
        The list of subjects whose derivatives were converted.

    Notes
    -----
    Since the T1w is in standard space already, we use identity transforms.
    """
    LOGGER.warning("convert_ukb2bids is an experimental function.")
    in_dir = os.path.abspath(in_dir)
    out_dir = os.path.abspath(out_dir)

    if participant_ids is None:
        subject_folders = sorted(glob.glob(os.path.join(in_dir, "*_*_2_0")))
        subject_folders = [
            subject_folder for subject_folder in subject_folders if os.path.isdir(subject_folder)
        ]
        participant_ids = [
            os.path.basename(subject_folder).split("_")[0] for subject_folder in subject_folders
        ]
        all_subject_ids = []
        for subject_id in participant_ids:
            if subject_id not in all_subject_ids:
                all_subject_ids.append(subject_id)

        participant_ids = all_subject_ids

        if len(participant_ids) == 0:
            raise ValueError(f"No subject found in {in_dir}")

    else:
        participant_ids = ensure_list(participant_ids)

    for subject_id in participant_ids:
        LOGGER.info(f"Converting {subject_id}")
        session_ids = ensure_list(bids_filters.get("bold", {}).get("session", "*"))
        subject_dirs = []
        for session_id in session_ids:
            subject_dir = sorted(glob.glob(os.path.join(in_dir, f"{subject_id}_{session_id}_2_0")))
            subject_dirs += subject_dir

        for subject_dir in subject_dirs:
            session_id = os.path.basename(subject_dir).split("_")[1]
            convert_ukb_to_bids_single_subject(
                in_dir=subject_dirs[0],
                out_dir=out_dir,
                sub_id=subject_id,
                ses_id=session_id,
            )

    return participant_ids


def convert_ukb_to_bids_single_subject(in_dir, out_dir, sub_id, ses_id):
    """Convert UK Biobank derivatives to BIDS-compliant derivatives for a single subject.

    Parameters
    ----------
    in_dir : str
        Path to the subject's UK Biobank derivatives.
    out_dir : str
        Path to the output fMRIPrep-style derivatives folder.
    sub_id : str
        Subject identifier, without "sub-" prefix.
    ses_id : str
        Session identifier, without "ses-" prefix.

    Notes
    -----
    The BOLD and brain mask files are in boldref space, so they must be warped to standard
    (MNI152NLin6Asym) space with FNIRT.

    Since the T1w is in standard space already, we use identity transforms.

    .. code-block::

        <sub_id>_<ses_id>_2_0
            ├── fMRI
            │   ├── rfMRI.ica
            │   │   ├── mc
            │   │   │   ├── prefiltered_func_data_mcf_abs.rms
            │   │   │   └── prefiltered_func_data_mcf.par
            │   │   ├── reg
            │   │   │   ├── example_func2standard.mat
            │   │   │   └── example_func2standard_warp.nii.gz
            │   │   ├── filtered_func_data_clean.nii.gz
            │   │   └── mask.nii.gz
            │   ├── rfMRI_SBREF.json
            │   └── rfMRI_SBREF.nii.gz
            └── T1
                └── T1_brain_to_MNI.nii.gz
    """
    assert isinstance(in_dir, str)
    assert os.path.isdir(in_dir), f"Folder DNE: {in_dir}"
    assert isinstance(out_dir, str)
    assert isinstance(sub_id, str)
    assert isinstance(ses_id, str)
    subses_ents = f"sub-{sub_id}_ses-{ses_id}"

    task_dir_orig = os.path.join(in_dir, "fMRI", "rfMRI.ica")
    bold_file = os.path.join(task_dir_orig, "filtered_func_data_clean.nii.gz")
    assert os.path.isfile(bold_file), f"File DNE: {bold_file}"
    bold_json = os.path.join(in_dir, "fMRI", "rfMRI.json")
    assert os.path.isfile(bold_json), f"File DNE: {bold_json}"
    boldref_file = os.path.join(task_dir_orig, "example_func.nii.gz")
    assert os.path.isfile(boldref_file), f"File DNE: {boldref_file}"
    brainmask_file = os.path.join(task_dir_orig, "mask.nii.gz")
    assert os.path.isfile(brainmask_file), f"File DNE: {brainmask_file}"
    t1w = os.path.join(in_dir, "T1", "T1_brain_to_MNI.nii.gz")
    assert os.path.isfile(t1w), f"File DNE: {t1w}"
    warp_file = os.path.join(task_dir_orig, "reg", "example_func2standard_warp.nii.gz")
    assert os.path.isfile(warp_file), f"File DNE: {warp_file}"

    func_prefix = f"sub-{sub_id}_ses-{ses_id}_task-rest"
    subject_dir_bids = os.path.join(out_dir, f"sub-{sub_id}", f"ses-{ses_id}")
    anat_dir_bids = os.path.join(subject_dir_bids, "anat")
    func_dir_bids = os.path.join(subject_dir_bids, "func")
    work_dir = os.path.join(subject_dir_bids, "work")
    os.makedirs(anat_dir_bids, exist_ok=True)
    os.makedirs(func_dir_bids, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)

    collect_ukbiobank_confounds(
        task_dir_orig=task_dir_orig,
        out_dir=func_dir_bids,
        prefix=func_prefix,
        work_dir=work_dir,
        bold_file=bold_file,
        brainmask_file=brainmask_file,
    )

    dataset_description_fmriprep = os.path.join(out_dir, "dataset_description.json")

    if os.path.isfile(dataset_description_fmriprep):
        LOGGER.info("Converted dataset already exists. Skipping conversion.")
        return

    VOLSPACE = "MNI152NLin6Asym"

    # Warp BOLD, T1w, and brainmask to MNI152NLin6Asym
    # We use FSL's MNI152NLin6Asym 2 mm3 template instead of TemplateFlow's version,
    # because FSL uses LAS+ orientation, while TemplateFlow uses RAS+.
    template_file = str(load_data("MNI152_T1_2mm.nii.gz"))

    copy_dictionary = {}

    warp_bold_to_std = ApplyWarp(
        interp="spline",
        output_type="NIFTI_GZ",
        ref_file=template_file,
        in_file=bold_file,
        field_file=warp_file,
    )
    LOGGER.warning(warp_bold_to_std.cmdline)
    warp_bold_to_std_results = warp_bold_to_std.run(cwd=work_dir)
    bold_nifti_fmriprep = os.path.join(
        func_dir_bids,
        f"{func_prefix}_space-{VOLSPACE}_desc-preproc_bold.nii.gz",
    )
    copy_dictionary[warp_bold_to_std_results.outputs.out_file] = [bold_nifti_fmriprep]

    # Extract metadata for JSON file
    with open(bold_json, "r") as fo:
        bold_metadata = json.load(fo)

    # Keep only the relevant fields
    keep_keys = [
        "FlipAngle",
        "EchoTime",
        "Manufacturer",
        "ManufacturersModelName",
        "EffectiveEchoSpacing",
        "RepetitionTime",
        "PhaseEncodingDirection",
    ]
    bold_metadata = {k: bold_metadata[k] for k in keep_keys if k in bold_metadata}
    bold_metadata["TaskName"] = "resting state"
    bold_nifti_json_fmriprep = bold_nifti_fmriprep.replace(".nii.gz", ".json")
    write_json(bold_metadata, bold_nifti_json_fmriprep)

    warp_brainmask_to_std = ApplyWarp(
        interp="nn",
        output_type="NIFTI_GZ",
        ref_file=template_file,
        in_file=brainmask_file,
        field_file=warp_file,
    )
    warp_brainmask_to_std_results = warp_brainmask_to_std.run(cwd=work_dir)
    copy_dictionary[warp_brainmask_to_std_results.outputs.out_file] = [
        os.path.join(
            func_dir_bids,
            f"{func_prefix}_space-{VOLSPACE}_desc-brain_mask.nii.gz",
        )
    ]
    # Use the brain mask as the anatomical brain mask too.
    copy_dictionary[warp_brainmask_to_std_results.outputs.out_file].append(
        os.path.join(
            anat_dir_bids,
            f"{subses_ents}_space-{VOLSPACE}_desc-brain_mask.nii.gz",
        )
    )
    # Use the brain mask as the "aparcaseg" dseg too.
    copy_dictionary[warp_brainmask_to_std_results.outputs.out_file].append(
        os.path.join(
            anat_dir_bids,
            f"{subses_ents}_space-{VOLSPACE}_desc-aparcaseg_dseg.nii.gz",
        )
    )

    # Warp the reference file to MNI space.
    warp_boldref_to_std = ApplyWarp(
        interp="spline",
        output_type="NIFTI_GZ",
        ref_file=template_file,
        in_file=boldref_file,
        field_file=warp_file,
    )
    warp_boldref_to_std_results = warp_boldref_to_std.run(cwd=work_dir)
    boldref_nifti_fmriprep = os.path.join(
        func_dir_bids,
        f"{func_prefix}_space-{VOLSPACE}_boldref.nii.gz",
    )
    copy_dictionary[warp_boldref_to_std_results.outputs.out_file] = [boldref_nifti_fmriprep]

    # The MNI-space anatomical image.
    copy_dictionary[t1w] = [
        os.path.join(anat_dir_bids, f"{subses_ents}_space-{VOLSPACE}_desc-preproc_T1w.nii.gz")
    ]

    # The identity xform is used in place of any actual ones.
    identity_xfm = str(load_data("transform/itkIdentityTransform.txt"))
    copy_dictionary[identity_xfm] = []

    t1w_to_template_fmriprep = os.path.join(
        anat_dir_bids,
        f"{subses_ents}_from-T1w_to-{VOLSPACE}_mode-image_xfm.txt",
    )
    copy_dictionary[identity_xfm].append(t1w_to_template_fmriprep)

    template_to_t1w_fmriprep = os.path.join(
        anat_dir_bids,
        f"{subses_ents}_from-{VOLSPACE}_to-T1w_mode-image_xfm.txt",
    )
    copy_dictionary[identity_xfm].append(template_to_t1w_fmriprep)

    LOGGER.info("Finished collecting functional files")

    # Copy UK Biobank files to fMRIPrep folder
    LOGGER.info("Copying files")
    copy_files_in_dict(copy_dictionary)
    LOGGER.info("Finished copying files")

    # Write the dataset description out last
    dataset_description_dict = {
        "Name": "UK Biobank",
        "BIDSVersion": "1.9.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "UK Biobank",
                "Version": "unknown",
                "CodeURL": "https://github.com/ucam-department-of-psychiatry/UKB",
            },
        ],
    }

    if not os.path.isfile(dataset_description_fmriprep):
        LOGGER.info(f"Writing dataset description to {dataset_description_fmriprep}")
        write_json(dataset_description_dict, dataset_description_fmriprep)

    # Write out the mapping from UK Biobank to fMRIPrep
    scans_dict = {}
    for key, values in copy_dictionary.items():
        for item in values:
            scans_dict[item] = key

    scans_tuple = tuple(scans_dict.items())
    scans_df = pd.DataFrame(scans_tuple, columns=["filename", "source_file"])
    scans_tsv = os.path.join(subject_dir_bids, f"{subses_ents}_scans.tsv")
    scans_df.to_csv(scans_tsv, sep="\t", index=False)
    LOGGER.info("Conversion completed")
