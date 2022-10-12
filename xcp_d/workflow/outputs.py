# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Workflows for collecting and saving xcp_d outputs."""
import numpy as np
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from xcp_d.interfaces.bids import DerivativesDataSink
from xcp_d.utils.doc import fill_doc


@fill_doc
def init_writederivatives_wf(
    bold_file,
    lowpass,
    highpass,
    smoothing,
    params,
    cifti,
    dummytime,
    output_dir,
    TR,
    name='write_derivatives_wf',
):
    """Write out the xcp_d derivatives in BIDS format.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.workflow.outputs import init_writederivatives_wf
            wf = init_writederivatives_wf(
                bold_file="/path/to/file.nii.gz",
                lowpass=6.,
                highpass=60.,
                smoothing=6,
                params="36P",
                cifti=False,
                dummytime=0,
                output_dir=".",
                TR=2.,
                name="write_derivatives_wf",
            )

    Parameters
    ----------
    bold_file : str
        bold or cifti files
    lowpass : float
        low pass filter
    highpass : float
        high pass filter
    %(smoothing)s
    %(params)s
    %(cifti)s
    dummytime : float
        volume(s) removed before postprocessing in seconds
    output_dir : str
        output directory
    TR : float
        repetition time in seconds
    %(name)s
        Default is "fcons_ts_wf".

    Inputs
    ------
    %(atlas_names)s
        Used for indexing ``timeseries`` and ``correlations``.
    timeseries : list of str
        List of paths to parcellated time series files.
    correlations : list of str
        List of paths to ROI-to-ROI correlation files.
    qc_file
        quality control files
    processed_bold
        clean bold after regression and filtering
    smoothed_bold
        smoothed clean bold
    alff_out
        alff niifti
    smoothed_alff
        smoothed alff
    reho_lh
        reho left hemisphere
    reho_rh
        reho right hemisphere
    reho_out
    fd
    """
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "atlas_names",
                "timeseries",
                "correlations",
                "qc_file",
                "processed_bold",
                "smoothed_bold",
                "alff_out",
                "smoothed_alff",
                "reho_lh",
                "reho_rh",
                "reho_out",
                "fd",
            ],
        ),
        name="inputnode",
    )

    # Create dictionary of basic information
    cleaned_data_dictionary = {
        'RepetitionTime': TR,
        'Freq Band': [highpass, lowpass],
        'nuisance parameters': params,
        'dummy vols': int(np.ceil(dummytime / TR))
    }
    smoothed_data_dictionary = {'FWHM': smoothing}  # Separate dictionary for smoothing
    # Write out detivatives via DerivativesDataSink
    if not cifti:  # if Nifti
        ds_denoised_bold = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                meta_dict=cleaned_data_dictionary,
                source_file=bold_file,
                desc='denoised',
                extension='.nii.gz',
                compression=True,
            ),
            name='ds_denoised_bold',
            run_without_submitting=True,
            mem_gb=2,
        )

        ds_alff = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                source_file=bold_file,
                dismiss_entities=["desc"],
                suffix="alff",
                extension='.nii.gz',
                compression=True,
            ),
            name='ds_alff',
            run_without_submitting=True,
            mem_gb=1,
        )

        ds_qc_file = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                source_file=bold_file,
                dismiss_entities=['desc'],
                suffix='qc',
                extension='.csv',
            ),
            name='ds_qc_file',
            run_without_submitting=True,
            mem_gb=1,
        )

        ds_timeseries = pe.MapNode(
            DerivativesDataSink(
                base_directory=output_dir,
                source_file=bold_file,
                dismiss_entities=['desc'],
                suffix='timeseries',
                extension=".tsv",
            ),
            name="ds_timeseries",
            run_without_submitting=True,
            mem_gb=1,
            iterfield=["atlas", "in_file"],
        )
        ds_conmats = pe.MapNode(
            DerivativesDataSink(
                base_directory=output_dir,
                source_file=bold_file,
                dismiss_entities=['desc'],
                measure="pearsoncorrelation",
                suffix='conmat',
                extension=".tsv",
            ),
            name="ds_conmats",
            run_without_submitting=True,
            mem_gb=1,
            iterfield=["atlas", "in_file"],
        )

        ds_reho = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                source_file=bold_file,
                dismiss_entities=['desc'],
                suffix='reho',
                extension='.nii.gz',
                compression=True,
            ),
            name='ds_reho',
            run_without_submitting=True,
            mem_gb=1,
        )

        ds_fd_motion = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                source_file=bold_file,
                dismiss_entities=["atlas", "res", "space"],
                desc='framewisedisplacement',
                suffix="motion",
                extension='.tsv',
            ),
            name='ds_fd_motion',
            run_without_submitting=True,
            mem_gb=1,
        )

        workflow.connect([(inputnode, ds_reho, [('reho_out', 'in_file')])])

        if smoothing:  # if smoothed
            # Write out detivatives via DerivativesDataSink
            ds_denoised_smoothed_bold = pe.Node(
                DerivativesDataSink(
                    base_directory=output_dir,
                    meta_dict=smoothed_data_dictionary,
                    source_file=bold_file,
                    desc='denoisedSmoothed',
                    extension='.nii.gz',
                    compression=True,
                ),
                name='ds_denoised_smoothed_bold',
                run_without_submitting=True,
                mem_gb=2,
            )

            ds_smoothed_alff = pe.Node(
                DerivativesDataSink(
                    base_directory=output_dir,
                    meta_dict=smoothed_data_dictionary,
                    source_file=bold_file,
                    desc='smooth',
                    suffix="alff",
                    extension='.nii.gz',
                    compression=True,
                ),
                name='ds_smoothed_alff',
                run_without_submitting=True,
                mem_gb=1,
            )

    else:  # For cifti files
        # Write out derivatives via DerivativesDataSink
        ds_denoised_bold = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                meta_dict=cleaned_data_dictionary,
                source_file=bold_file,
                dismiss_entities=["den"],
                desc='denoised',
                density='91k',
                extension='.dtseries.nii',
            ),
            name='ds_denoised_bold',
            run_without_submitting=True,
            mem_gb=2,
        )

        ds_alff = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                source_file=bold_file,
                check_hdr=False,
                dismiss_entities=['desc', 'den'],
                density='91k',
                suffix='alff',
                extension='.dscalar.nii',
            ),
            name='ds_alff',
            run_without_submitting=True,
            mem_gb=1,
        )

        ds_qc_file = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                source_file=bold_file,
                dismiss_entities=['desc', 'den'],
                density='91k',
                suffix='qc',
                extension='.csv',
            ),
            name='ds_qc_file',
            run_without_submitting=True,
            mem_gb=1,
        )

        ds_timeseries = pe.MapNode(
            DerivativesDataSink(
                base_directory=output_dir,
                source_file=bold_file,
                check_hdr=False,
                dismiss_entities=['desc', 'den'],
                density='91k',
                suffix="timeseries",
                extension='.ptseries.nii',
            ),
            name="ds_timeseries",
            run_without_submitting=True,
            mem_gb=1,
            iterfield=["atlas", "in_file"],
        )

        ds_conmats = pe.MapNode(
            DerivativesDataSink(
                base_directory=output_dir,
                source_file=bold_file,
                check_hdr=False,
                dismiss_entities=['desc', 'den'],
                density='91k',
                measure="pearsoncorrelation",
                suffix='conmat',
                extension='.pconn.nii',
            ),
            name="ds_conmats",
            run_without_submitting=True,
            mem_gb=1,
            iterfield=["atlas", "in_file"],
        )

        ds_reho_lh = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                source_file=bold_file,
                check_hdr=False,
                dismiss_entities=['desc', 'den'],
                density='32k',
                hemi='L',
                suffix='reho',
                extension='.shape.gii',
            ),
            name='ds_reho_lh',
            run_without_submitting=True,
            mem_gb=1,
        )

        ds_reho_rh = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                source_file=bold_file,
                check_hdr=False,
                dismiss_entities=['desc', 'den'],
                density='32k',
                hemi='R',
                suffix='reho',
                extension='.shape.gii',
            ),
            name='ds_reho_rh',
            run_without_submitting=True,
            mem_gb=1,
        )

        ds_fd_motion = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                source_file=bold_file,
                dismiss_entities=["atlas", "den", "res", "space"],
                desc='framewisedisplacement',
                suffix="motion",
                extension='.tsv',
            ),
            name='ds_fd_motion',
            run_without_submitting=True,
            mem_gb=1,
        )

        workflow.connect([
            (inputnode, ds_reho_lh, [('reho_lh', 'in_file')]),
            (inputnode, ds_reho_rh, [('reho_rh', 'in_file')]),
        ])

        if smoothing:  # If smoothed
            # Write out detivatives via DerivativesDataSink
            ds_denoised_smoothed_bold = pe.Node(
                DerivativesDataSink(
                    base_directory=output_dir,
                    meta_dict=smoothed_data_dictionary,
                    source_file=bold_file,
                    dismiss_entities=["den"],
                    density='91k',
                    desc='denoisedSmoothed',
                    extension='.dtseries.nii',
                    check_hdr=False,
                ),
                name='ds_denoised_smoothed_bold',
                run_without_submitting=True,
                mem_gb=2,
            )

            ds_smoothed_alff = pe.Node(
                DerivativesDataSink(
                    base_directory=output_dir,
                    meta_dict=smoothed_data_dictionary,
                    source_file=bold_file,
                    dismiss_entities=["den"],
                    desc='smooth',
                    density='91k',
                    suffix='alff',
                    extension='.dscalar.nii',
                    check_hdr=False,
                ),
                name='ds_smoothed_alff',
                run_without_submitting=True,
                mem_gb=1,
            )

    workflow.connect([
        (inputnode, ds_denoised_bold, [('processed_bold', 'in_file')]),
        (inputnode, ds_alff, [('alff_out', 'in_file')]),
        (inputnode, ds_qc_file, [('qc_file', 'in_file')]),
        (inputnode, ds_timeseries, [('timeseries', 'in_file'), ('atlas_names', 'atlas')]),
        (inputnode, ds_conmats, [('correlations', 'in_file'), ('atlas_names', 'atlas')]),
        (inputnode, ds_fd_motion, [('fd', 'in_file')]),
    ])

    if smoothing:
        workflow.connect([
            (inputnode, ds_denoised_smoothed_bold, [('smoothed_bold', 'in_file')]),
            (inputnode, ds_smoothed_alff, [('smoothed_alff', 'in_file')]),
        ])

    return workflow
