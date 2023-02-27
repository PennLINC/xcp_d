# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Workflows for post-processing CIFTI-format BOLD data."""
import os

import nibabel as nb
import numpy as np
from nipype import Function, logging
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from num2words import num2words

from xcp_d.interfaces.bids import DerivativesDataSink
from xcp_d.interfaces.nilearn import DenoiseCifti
from xcp_d.interfaces.prepostcleaning import (
    ConvertTo32,
    FlagMotionOutliers,
    RemoveDummyVolumes,
)
from xcp_d.interfaces.resting_state import DespikePatch
from xcp_d.interfaces.workbench import CiftiConvert
from xcp_d.utils.bids import collect_run_data
from xcp_d.utils.confounds import (
    consolidate_confounds,
    describe_censoring,
    describe_regression,
    get_customfile,
)
from xcp_d.utils.doc import fill_doc
from xcp_d.utils.plotting import plot_design_matrix
from xcp_d.utils.utils import estimate_brain_radius
from xcp_d.workflows.connectivity import init_cifti_functional_connectivity_wf
from xcp_d.workflows.execsummary import init_execsummary_functional_plots_wf
from xcp_d.workflows.outputs import init_writederivatives_wf
from xcp_d.workflows.plotting import init_qc_report_wf
from xcp_d.workflows.postprocessing import init_resd_smoothing_wf
from xcp_d.workflows.restingstate import init_cifti_reho_wf, init_compute_alff_wf

LOGGER = logging.getLogger("nipype.workflow")


@fill_doc
def init_ciftipostprocess_wf(
    bold_file,
    bandpass_filter,
    lower_bpf,
    upper_bpf,
    bpf_order,
    motion_filter_type,
    motion_filter_order,
    band_stop_min,
    band_stop_max,
    smoothing,
    head_radius,
    params,
    output_dir,
    custom_confounds_folder,
    input_type,
    dummytime,
    dummy_scans,
    fd_thresh,
    despike,
    dcan_qc,
    n_runs,
    min_coverage,
    omp_nthreads,
    layout=None,
    name="cifti_process_wf",
):
    """Organize the cifti processing workflow.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            import os

            from xcp_d.utils.bids import collect_data
            from xcp_d.workflows.cifti import init_ciftipostprocess_wf
            from xcp_d.utils.doc import download_example_data

            fmri_dir = download_example_data()

            layout, subj_data = collect_data(
                bids_dir=fmri_dir,
                input_type="fmriprep",
                participant_label="01",
                task="imagery",
                bids_validate=False,
                cifti=True,
            )

            bold_file = subj_data["bold"][0]
            custom_confounds_folder = os.path.join(fmri_dir, "sub-01/func")

            wf = init_ciftipostprocess_wf(
                bold_file=bold_file,
                bandpass_filter=True,
                lower_bpf=0.01,
                upper_bpf=0.08,
                bpf_order=2,
                motion_filter_type="notch",
                motion_filter_order=4,
                band_stop_min=12,
                band_stop_max=20,
                smoothing=6,
                head_radius=50.,
                params="27P",
                output_dir=".",
                custom_confounds_folder=custom_confounds_folder,
                input_type="fmriprep",
                dummy_scans=0,
                dummytime=0,
                fd_thresh=0.2,
                despike=True,
                dcan_qc=True,
                n_runs=1,
                min_coverage=0.5,
                omp_nthreads=1,
                layout=layout,
                name="cifti_postprocess_wf",
            )
            wf.inputs.inputnode.t1w = subj_data["t1w"]

    Parameters
    ----------
    bold_file
    input_type
    %(bandpass_filter)s
    %(lower_bpf)s
    %(upper_bpf)s
    %(bpf_order)s
    %(motion_filter_type)s
    %(motion_filter_order)s
    %(band_stop_min)s
    %(band_stop_max)s
    %(smoothing)s
    %(head_radius)s
    %(params)s
    %(output_dir)s
    custom_confounds_folder: str
        path to cusrtom nuissance regressors
    %(omp_nthreads)s
    %(dummytime)s
    %(dummy_scans)s
    %(fd_thresh)s
    despike: bool
        afni depsike
    dcan_qc : bool
        Whether to run DCAN QC or not.
    n_runs
    min_coverage
    layout : BIDSLayout object
        BIDS dataset layout
    %(name)s
        Default is 'cifti_postprocess_wf'.

    Inputs
    ------
    bold_file
        CIFTI file
    custom_confounds_file
        custom regressors
    t1w
        Preprocessed T1w image, warped to standard space.
        Fed from the subject workflow.
    t2w
        Preprocessed T2w image, warped to standard space.
        Fed from the subject workflow.
    t1w_mask
        T1w brain mask, used to estimate head/brain radius.
        Fed from the subject workflow.
    fmriprep_confounds_tsv

    References
    ----------
    .. footbibliography::
    """
    run_data = collect_run_data(layout, input_type, bold_file, cifti=True)

    TR = run_data["bold_metadata"]["RepetitionTime"]

    # Load custom confounds
    # We need to run this function directly to access information in the confounds that is
    # used for the boilerplate.
    custom_confounds_file = get_customfile(
        custom_confounds_folder,
        run_data["confounds"],
    )
    regression_description = describe_regression(params, custom_confounds_file)
    censoring_description = describe_censoring(
        motion_filter_type=motion_filter_type,
        motion_filter_order=motion_filter_order,
        band_stop_min=band_stop_min,
        band_stop_max=band_stop_max,
        head_radius=head_radius,
        fd_thresh=fd_thresh,
    )

    workflow = Workflow(name=name)

    if dummy_scans == 0 and dummytime != 0:
        dummy_scans = int(np.ceil(dummytime / TR))

    dummy_scans_str = ""
    if dummy_scans == "auto":
        dummy_scans_str = (
            "non-steady-state volumes were extracted from the preprocessed confounds "
            "and were discarded from both the BOLD data and nuisance regressors, then"
        )
    elif dummy_scans > 0:
        dummy_scans_str = (
            f"the first {num2words(dummy_scans)} of both the BOLD data and nuisance "
            "regressors were discarded, then "
        )

    despike_str = ""
    if despike:
        despike_str = (
            "After censoring, but before nuisance regression, "
            "the BOLD data were converted to NIfTI format, despiked with 3dDespike, "
            "and converted back to CIFTI format."
        )

    bandpass_str = ""
    if bandpass_filter:
        bandpass_str = (
            "The interpolated timeseries were then band-pass filtered using a(n) "
            f"{num2words(bpf_order, ordinal=True)}-order Butterworth filter, "
            f"in order to retain signals within the {lower_bpf}-{upper_bpf} Hz frequency band."
        )

    workflow.__desc__ = f"""\
For each of the {num2words(n_runs)} BOLD runs found per subject (across all tasks and sessions),
the following post-processing was performed.
First, {dummy_scans_str}outlier detection was performed.
{censoring_description}
{despike_str}
Next, the BOLD data and confounds were mean-centered and linearly detrended.
{regression_description}
Any volumes censored earlier in the workflow were then interpolated in the residual time series
produced by the regression.
{bandpass_str}
"""

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "bold_file",
                "ref_file",
                "custom_confounds_file",
                "t1w",
                "t2w",
                "t1w_mask",
                "fmriprep_confounds_tsv",
                "dummy_scans",
            ],
        ),
        name="inputnode",
    )

    inputnode.inputs.bold_file = bold_file
    inputnode.inputs.ref_file = run_data["boldref"]
    inputnode.inputs.custom_confounds_file = custom_confounds_file
    inputnode.inputs.fmriprep_confounds_tsv = run_data["confounds"]
    inputnode.inputs.dummy_scans = dummy_scans

    mem_gbx = _create_mem_gb(bold_file)

    downcast_data = pe.Node(
        ConvertTo32(),
        name="downcast_data",
        mem_gb=mem_gbx["timeseries"],
        n_procs=omp_nthreads,
    )

    # fmt:off
    workflow.connect([
        (inputnode, downcast_data, [
            ("bold_file", "bold_file"),
            ("t1w_mask", "t1w_mask"),
        ]),
    ])
    # fmt:on

    determine_head_radius = pe.Node(
        Function(
            function=estimate_brain_radius,
            input_names=["mask_file", "head_radius"],
            output_names=["head_radius"],
        ),
        name="determine_head_radius",
    )
    determine_head_radius.inputs.head_radius = head_radius

    # fmt:off
    workflow.connect([
        (downcast_data, determine_head_radius, [
            ("t1w_mask", "mask_file"),
        ]),
    ])
    # fmt:on

    fcon_ts_wf = init_cifti_functional_connectivity_wf(
        min_coverage=min_coverage,
        output_dir=output_dir,
        mem_gb=mem_gbx["timeseries"],
        name="cifti_ts_con_wf",
        omp_nthreads=omp_nthreads,
    )

    if bandpass_filter:
        alff_compute_wf = init_compute_alff_wf(
            mem_gb=mem_gbx["timeseries"],
            TR=TR,
            bold_file=bold_file,
            lowpass=upper_bpf,
            highpass=lower_bpf,
            smoothing=smoothing,
            cifti=True,
            name="compute_alff_wf",
            omp_nthreads=omp_nthreads,
        )

    reho_compute_wf = init_cifti_reho_wf(
        mem_gb=mem_gbx["timeseries"],
        bold_file=bold_file,
        name="cifti_reho_wf",
        omp_nthreads=omp_nthreads,
    )

    write_derivative_wf = init_writederivatives_wf(
        smoothing=smoothing,
        bold_file=bold_file,
        bandpass_filter=bandpass_filter,
        params=params,
        cifti=True,
        output_dir=output_dir,
        lowpass=upper_bpf,
        highpass=lower_bpf,
        motion_filter_type=motion_filter_type,
        TR=TR,
        name="write_derivative_wf",
    )

    censor_scrub = pe.Node(
        FlagMotionOutliers(
            TR=TR,
            band_stop_min=band_stop_min,
            band_stop_max=band_stop_max,
            motion_filter_type=motion_filter_type,
            motion_filter_order=motion_filter_order,
            fd_thresh=fd_thresh,
        ),
        name="censoring",
        mem_gb=mem_gbx["timeseries"],
        omp_nthreads=omp_nthreads,
    )

    resd_smoothing_wf = init_resd_smoothing_wf(
        mem_gb=mem_gbx["timeseries"],
        smoothing=smoothing,
        cifti=True,
        name="resd_smoothing_wf",
        omp_nthreads=omp_nthreads,
    )

    denoise_bold = pe.Node(
        DenoiseCifti(
            TR=TR,
            lowpass=upper_bpf,
            highpass=lower_bpf,
            filter_order=bpf_order,
            bandpass_filter=bandpass_filter,
        ),
        name="denoise_bold",
        mem_gb=mem_gbx["timeseries"],
        n_procs=omp_nthreads,
    )

    consolidate_confounds_node = pe.Node(
        Function(
            input_names=[
                "img_file",
                "custom_confounds_file",
                "params",
            ],
            output_names=["out_file"],
            function=consolidate_confounds,
        ),
        name="consolidate_confounds_node",
    )
    consolidate_confounds_node.inputs.params = params

    # Load and filter confounds
    # fmt:off
    workflow.connect([
        (inputnode, consolidate_confounds_node, [
            ("bold_file", "img_file"),
            ("custom_confounds_file", "custom_confounds_file"),
        ]),
    ])
    # fmt:on

    plot_design_matrix_node = pe.Node(
        Function(
            input_names=["design_matrix", "censoring_file"],
            output_names=["design_matrix_figure"],
            function=plot_design_matrix,
        ),
        name="plot_design_matrix_node",
    )

    qc_report_wf = init_qc_report_wf(
        output_dir=output_dir,
        TR=TR,
        motion_filter_type=motion_filter_type,
        band_stop_max=band_stop_max,
        band_stop_min=band_stop_min,
        motion_filter_order=motion_filter_order,
        fd_thresh=fd_thresh,
        mem_gb=mem_gbx["timeseries"],
        omp_nthreads=omp_nthreads,
        dcan_qc=dcan_qc,
        cifti=True,
        name="qc_report_wf",
    )

    # fmt:off
    workflow.connect([
        (inputnode, qc_report_wf, [
            ("bold_file", "inputnode.preprocessed_bold"),
        ]),
        (determine_head_radius, qc_report_wf, [
            ("head_radius", "inputnode.head_radius"),
        ]),
        (denoise_bold, qc_report_wf, [
            ("uncensored_denoised_bold", "inputnode.uncensored_denoised_bold"),
        ]),
    ])
    # fmt:on

    # Remove TR first
    if dummy_scans:
        remove_dummy_scans = pe.Node(
            RemoveDummyVolumes(),
            name="remove_dummy_scans",
            mem_gb=mem_gbx["timeseries"],
        )

        # fmt:off
        workflow.connect([
            (inputnode, remove_dummy_scans, [
                ("dummy_scans", "dummy_scans"),
                # fMRIPrep confounds file is needed for filtered motion.
                # The selected confounds are not guaranteed to include motion params.
                ("fmriprep_confounds_tsv", "fmriprep_confounds_file"),
            ]),
            (downcast_data, remove_dummy_scans, [
                ("bold_file", "bold_file"),
            ]),
            (consolidate_confounds_node, remove_dummy_scans, [
                ("out_file", "confounds_file"),
            ]),
            (remove_dummy_scans, censor_scrub, [
                # fMRIPrep confounds file is needed for filtered motion.
                # The selected confounds are not guaranteed to include motion params.
                ("fmriprep_confounds_file_dropped_TR", "fmriprep_confounds_file"),
            ]),
            (remove_dummy_scans, denoise_bold, [
                ("confounds_file_dropped_TR", "confounds_file"),
            ]),
            (remove_dummy_scans, qc_report_wf, [
                ("dummy_scans", "inputnode.dummy_scans"),
            ]),
            (remove_dummy_scans, plot_design_matrix_node, [
                ("confounds_file_dropped_TR", "design_matrix"),
            ]),
        ])
        # fmt:on

    else:
        # fmt:off
        workflow.connect([
            (inputnode, qc_report_wf, [
                ("dummy_scans", "inputnode.dummy_scans"),
            ]),
            (inputnode, censor_scrub, [
                # fMRIPrep confounds file is needed for filtered motion.
                # The selected confounds are not guaranteed to include motion params.
                ("fmriprep_confounds_tsv", "fmriprep_confounds_file"),
            ]),
            (consolidate_confounds_node, denoise_bold, [('out_file', 'confounds_file')]),
            (consolidate_confounds_node, plot_design_matrix_node, [
                ("out_file", "design_matrix"),
            ]),
        ])
        # fmt:on

    # fmt:off
    workflow.connect([
        (determine_head_radius, censor_scrub, [
            ("head_radius", "head_radius"),
        ]),
        (censor_scrub, plot_design_matrix_node, [
            ("tmask", "censoring_file"),
        ]),
    ])
    # fmt:on

    if despike:
        # first, convert the cifti to a nifti
        convert_to_nifti = pe.Node(
            CiftiConvert(target="to"),
            name="convert_to_nifti",
            mem_gb=mem_gbx["timeseries"],
            n_procs=omp_nthreads,
        )

        # next, run 3dDespike
        despike3d = pe.Node(
            DespikePatch(outputtype="NIFTI_GZ", args="-nomask -NEW"),
            name="despike3d",
            mem_gb=mem_gbx["timeseries"],
            n_procs=omp_nthreads,
        )

        # finally, convert the despiked nifti back to cifti
        convert_to_cifti = pe.Node(
            CiftiConvert(target="from", TR=TR),
            name="convert_to_cifti",
            mem_gb=mem_gbx["timeseries"],
            n_procs=omp_nthreads,
        )

        # fmt:off
        workflow.connect([
            (convert_to_nifti, despike3d, [("out_file", "in_file")]),
            (downcast_data, convert_to_cifti, [("bold_file", "cifti_template")]),
            (despike3d, convert_to_cifti, [("out_file", "in_file")]),
            (convert_to_cifti, denoise_bold, [('out_file', 'preprocessed_bold')]),
        ])

        if dummy_scans:
            workflow.connect([
                (remove_dummy_scans, convert_to_nifti, [('bold_file_dropped_TR', 'in_file')]),
            ])
        else:
            workflow.connect([(downcast_data, convert_to_nifti, [('bold_file', 'in_file')])])
        # fmt:on

    elif dummy_scans:
        # fmt:off
        workflow.connect([
            (remove_dummy_scans, denoise_bold, [('bold_file_dropped_TR', 'preprocessed_bold')]),
        ])
        # fmt:on
    else:
        # fmt:off
        workflow.connect([
            (downcast_data, denoise_bold, [('bold_file', 'preprocessed_bold')]),
        ])
        # fmt:on

    # fmt:off
    workflow.connect([
        (censor_scrub, denoise_bold, [("tmask", "censoring_file")]),
    ])

    # residual smoothing
    workflow.connect([
        (denoise_bold, resd_smoothing_wf, [('filtered_denoised_bold', 'inputnode.bold_file')]),
    ])

    # functional connectivity workflow
    workflow.connect([
        (inputnode, fcon_ts_wf, [('bold_file', 'inputnode.bold_file')]),
        (denoise_bold, fcon_ts_wf, [('filtered_denoised_bold', 'inputnode.clean_bold')]),
    ])

    # reho and alff
    workflow.connect([
        (denoise_bold, reho_compute_wf, [('filtered_denoised_bold', 'inputnode.clean_bold')]),
    ])

    if bandpass_filter:
        workflow.connect([
            (denoise_bold, alff_compute_wf, [('filtered_denoised_bold', 'inputnode.clean_bold')]),
        ])

    # qc report
    workflow.connect([
        (denoise_bold, qc_report_wf, [
            ("filtered_denoised_bold", "inputnode.filtered_denoised_bold"),
        ]),
        (censor_scrub, qc_report_wf, [
            ("tmask", "inputnode.tmask"),
            ("filtered_motion", "inputnode.filtered_motion"),
        ]),
    ])

    # write derivatives
    workflow.connect([
        (consolidate_confounds_node, write_derivative_wf, [
            ('out_file', 'inputnode.confounds_file'),
        ]),
        (denoise_bold, write_derivative_wf, [
            ('filtered_denoised_bold', 'inputnode.processed_bold'),
        ]),
        (qc_report_wf, write_derivative_wf, [
            ('outputnode.qc_file', 'inputnode.qc_file'),
        ]),
        (resd_smoothing_wf, write_derivative_wf, [
            ('outputnode.smoothed_bold', 'inputnode.smoothed_bold'),
        ]),
        (censor_scrub, write_derivative_wf, [
            ('filtered_motion', 'inputnode.filtered_motion'),
            ('filtered_motion_metadata', 'inputnode.filtered_motion_metadata'),
            ('tmask', 'inputnode.tmask'),
            ('tmask_metadata', 'inputnode.tmask_metadata'),
        ]),
        (reho_compute_wf, write_derivative_wf, [
            ('outputnode.reho_out', 'inputnode.reho_out'),
        ]),
        (fcon_ts_wf, write_derivative_wf, [
            ('outputnode.atlas_names', 'inputnode.atlas_names'),
            ('outputnode.coverage_pscalar', 'inputnode.coverage_ciftis'),
            ('outputnode.ptseries', 'inputnode.timeseries_ciftis'),
            ('outputnode.pconn', 'inputnode.correlation_ciftis'),
            ('outputnode.coverage', 'inputnode.coverage_files'),
            ('outputnode.timeseries', 'inputnode.timeseries'),
            ('outputnode.correlations', 'inputnode.correlations'),
        ]),
    ])

    if bandpass_filter:
        workflow.connect([
            (alff_compute_wf, write_derivative_wf, [
                ('outputnode.alff_out', 'inputnode.alff_out'),
                ('outputnode.smoothed_alff', 'inputnode.smoothed_alff'),
            ]),
        ])
    # fmt:on

    ds_design_matrix_plot = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            source_file=bold_file,
            dismiss_entities=["space", "res", "den", "desc"],
            datatype="figures",
            suffix="design",
            extension=".svg",
        ),
        name="ds_design_matrix_plot",
        run_without_submitting=False,
    )

    ds_report_connectivity = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            source_file=bold_file,
            desc="connectivityplot",
            datatype="figures",
        ),
        name="ds_report_connectivity",
        run_without_submitting=True,
    )

    if bandpass_filter:
        ds_report_alffplot = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                source_file=bold_file,
                desc="alffSurfacePlot",
                datatype="figures",
            ),
            name="ds_report_alffplot",
            run_without_submitting=False,
        )

    ds_report_rehoplot = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            source_file=bold_file,
            desc="rehoSurfacePlot",
            datatype="figures",
        ),
        name="ds_report_rehoplot",
        run_without_submitting=False,
    )

    # fmt:off
    workflow.connect([
        (plot_design_matrix_node, ds_design_matrix_plot, [("design_matrix_figure", "in_file")]),
        (reho_compute_wf, ds_report_rehoplot, [('outputnode.rehoplot', 'in_file')]),
        (fcon_ts_wf, ds_report_connectivity, [('outputnode.connectplot', "in_file")])
    ])
    # fmt:on

    if bandpass_filter:
        # fmt:off
        workflow.connect([
            (alff_compute_wf, ds_report_alffplot, [('outputnode.alffplot', 'in_file')])
        ])
        # fmt:on

    # executive summary workflow
    if dcan_qc:
        execsummary_functional_plots_wf = init_execsummary_functional_plots_wf(
            preproc_nifti=run_data["nifti_file"],
            t1w_available=True,
            t2w_available=False,
            output_dir=output_dir,
            layout=layout,
            name="execsummary_functional_plots_wf",
        )

        # Use inputnode for executive summary instead of downcast_data
        # because T1w is used as name source.
        # fmt:off
        workflow.connect([
            # Use inputnode for executive summary instead of downcast_data
            # because T1w is used as name source.
            (inputnode, execsummary_functional_plots_wf, [
                ("ref_file", "inputnode.boldref"),
                ("t1w", "inputnode.t1w"),
                ("t2w", "inputnode.t2w"),
            ]),
        ])
        # fmt:on

    return workflow


def _create_mem_gb(bold_fname):
    bold_size_gb = os.path.getsize(bold_fname) / (1024**3)
    bold_tlen = nb.load(bold_fname).shape[-1]
    mem_gbz = {
        "derivative": bold_size_gb,
        "resampled": bold_size_gb * 4,
        "timeseries": bold_size_gb * (max(bold_tlen / 100, 1.0) + 4),
    }

    return mem_gbz
