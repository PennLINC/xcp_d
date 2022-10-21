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
    motion_filter_type,
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
    %(motion_filter_type)s
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
    filtered_motion
    tmask
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
                "filtered_motion",
                "tmask",
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

    write_derivative_tmask_wf = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            dismiss_entities=["atlas", "den", "res", "space", "desc"],
            suffix="outliers",
            extension=".tsv",
            source_file=bold_file,
        ),
        name="write_derivative_tmask_wf",
        run_without_submitting=True,
        mem_gb=1,
    )

    ds_filtered_motion = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            source_file=bold_file,
            dismiss_entities=["atlas", "den", "res", "space", "desc"],
            desc="filtered" if motion_filter_type else None,
            suffix="motion",
            extension='.tsv',
        ),
        name='ds_filtered_motion',
        run_without_submitting=True,
        mem_gb=1,
    )

    workflow.connect([
        (inputnode, write_derivative_tmask_wf, [('tmask', 'in_file')]),
        (inputnode, ds_filtered_motion, [('filtered_motion', 'in_file')]),
    ])

    # Write out detivatives via DerivativesDataSink
    if not cifti:  # if Nifti
        write_derivative_cleandata_wf = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                meta_dict=cleaned_data_dictionary,
                source_file=bold_file,
                desc='denoised',
                extension='.nii.gz',
                compression=True,
            ),
            name='write_derivative_cleandata_wf',
            run_without_submitting=True,
            mem_gb=2,
        )

        write_derivative_alff_wf = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                source_file=bold_file,
                dismiss_entities=["desc"],
                suffix="alff",
                extension='.nii.gz',
                compression=True,
            ),
            name='write_derivative_alff_wf',
            run_without_submitting=True,
            mem_gb=1,
        )

        write_derivative_qcfile_wf = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                source_file=bold_file,
                dismiss_entities=['desc'],
                suffix='qc',
                extension='.csv',
            ),
            name='write_derivative_qcfile_wf',
            run_without_submitting=True,
            mem_gb=1,
        )

        timeseries_wf = pe.MapNode(
            DerivativesDataSink(
                base_directory=output_dir,
                source_file=bold_file,
                dismiss_entities=['desc'],
                suffix='timeseries',
                extension=".tsv",
            ),
            name="timeseries_wf",
            run_without_submitting=True,
            mem_gb=1,
            iterfield=["atlas", "in_file"],
        )
        correlations_wf = pe.MapNode(
            DerivativesDataSink(
                base_directory=output_dir,
                source_file=bold_file,
                dismiss_entities=['desc'],
                measure="pearsoncorrelation",
                suffix='conmat',
                extension=".tsv",
            ),
            name="correlations_wf",
            run_without_submitting=True,
            mem_gb=1,
            iterfield=["atlas", "in_file"],
        )

        write_derivative_reho_wf = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                source_file=bold_file,
                dismiss_entities=['desc'],
                suffix='reho',
                extension='.nii.gz',
                compression=True,
            ),
            name='write_derivative_reho_wf',
            run_without_submitting=True,
            mem_gb=1,
        )

        workflow.connect([(inputnode, write_derivative_reho_wf, [('reho_out', 'in_file')])])

        if smoothing:  # if smoothed
            # Write out detivatives via DerivativesDataSink
            write_derivative_smoothcleandata_wf = pe.Node(
                DerivativesDataSink(
                    base_directory=output_dir,
                    meta_dict=smoothed_data_dictionary,
                    source_file=bold_file,
                    desc='denoisedSmoothed',
                    extension='.nii.gz',
                    compression=True,
                ),
                name='write_derivative_smoothcleandata_wf',
                run_without_submitting=True,
                mem_gb=2,
            )

            write_derivative_smoothalff_wf = pe.Node(
                DerivativesDataSink(
                    base_directory=output_dir,
                    meta_dict=smoothed_data_dictionary,
                    source_file=bold_file,
                    desc='smooth',
                    suffix="alff",
                    extension='.nii.gz',
                    compression=True,
                ),
                name='write_derivative_smoothalff_wf',
                run_without_submitting=True,
                mem_gb=1,
            )

    else:  # For cifti files
        # Write out derivatives via DerivativesDataSink
        write_derivative_cleandata_wf = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                meta_dict=cleaned_data_dictionary,
                source_file=bold_file,
                dismiss_entities=["den"],
                desc='denoised',
                den='91k',
                extension='.dtseries.nii',
            ),
            name='write_derivative_cleandata_wf',
            run_without_submitting=True,
            mem_gb=2,
        )

        write_derivative_alff_wf = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                source_file=bold_file,
                check_hdr=False,
                dismiss_entities=['desc', 'den'],
                den='91k',
                suffix='alff',
                extension='.dscalar.nii',
            ),
            name='write_derivative_alff_wf',
            run_without_submitting=True,
            mem_gb=1,
        )

        write_derivative_qcfile_wf = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                source_file=bold_file,
                dismiss_entities=['desc', 'den'],
                den='91k',
                suffix='qc',
                extension='.csv',
            ),
            name='write_derivative_qcfile_wf',
            run_without_submitting=True,
            mem_gb=1,
        )

        timeseries_wf = pe.MapNode(
            DerivativesDataSink(
                base_directory=output_dir,
                source_file=bold_file,
                check_hdr=False,
                dismiss_entities=['desc', 'den'],
                den='91k',
                suffix="timeseries",
                extension='.ptseries.nii',
            ),
            name="timeseries_wf",
            run_without_submitting=True,
            mem_gb=1,
            iterfield=["atlas", "in_file"],
        )

        correlations_wf = pe.MapNode(
            DerivativesDataSink(
                base_directory=output_dir,
                source_file=bold_file,
                check_hdr=False,
                dismiss_entities=['desc', 'den'],
                den='91k',
                measure="pearsoncorrelation",
                suffix='conmat',
                extension='.pconn.nii',
            ),
            name="correlations_wf",
            run_without_submitting=True,
            mem_gb=1,
            iterfield=["atlas", "in_file"],
        )

        write_derivative_reholh_wf = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                source_file=bold_file,
                check_hdr=False,
                dismiss_entities=['desc', 'den'],
                den='32k',
                hemi='L',
                suffix='reho',
                extension='.shape.gii',
            ),
            name='write_derivative_reholh_wf',
            run_without_submitting=True,
            mem_gb=1,
        )

        write_derivative_rehorh_wf = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                source_file=bold_file,
                check_hdr=False,
                dismiss_entities=['desc', 'den'],
                den='32k',
                hemi='R',
                suffix='reho',
                extension='.shape.gii',
            ),
            name='write_derivative_rehorh_wf',
            run_without_submitting=True,
            mem_gb=1,
        )

        workflow.connect([
            (inputnode, write_derivative_reholh_wf, [('reho_lh', 'in_file')]),
            (inputnode, write_derivative_rehorh_wf, [('reho_rh', 'in_file')]),
        ])

        if smoothing:  # If smoothed
            # Write out detivatives via DerivativesDataSink
            write_derivative_smoothcleandata_wf = pe.Node(
                DerivativesDataSink(
                    base_directory=output_dir,
                    meta_dict=smoothed_data_dictionary,
                    source_file=bold_file,
                    dismiss_entities=["den"],
                    den='91k',
                    desc='denoisedSmoothed',
                    extension='.dtseries.nii',
                    check_hdr=False,
                ),
                name='write_derivative_smoothcleandata_wf',
                run_without_submitting=True,
                mem_gb=2,
            )

            write_derivative_smoothalff_wf = pe.Node(
                DerivativesDataSink(
                    base_directory=output_dir,
                    meta_dict=smoothed_data_dictionary,
                    source_file=bold_file,
                    dismiss_entities=["den"],
                    desc='smooth',
                    den='91k',
                    suffix='alff',
                    extension='.dscalar.nii',
                    check_hdr=False,
                ),
                name='write_derivative_smoothalff_wf',
                run_without_submitting=True,
                mem_gb=1,
            )

    workflow.connect([
        (inputnode, write_derivative_cleandata_wf, [('processed_bold', 'in_file')]),
        (inputnode, write_derivative_alff_wf, [('alff_out', 'in_file')]),
        (inputnode, write_derivative_qcfile_wf, [('qc_file', 'in_file')]),
        (inputnode, timeseries_wf, [('timeseries', 'in_file'), ('atlas_names', 'atlas')]),
        (inputnode, correlations_wf, [('correlations', 'in_file'), ('atlas_names', 'atlas')]),
    ])

    if smoothing:
        workflow.connect([
            (inputnode, write_derivative_smoothcleandata_wf, [('smoothed_bold', 'in_file')]),
            (inputnode, write_derivative_smoothalff_wf, [('smoothed_alff', 'in_file')]),
        ])

    return workflow
