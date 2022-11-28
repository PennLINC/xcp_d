# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Workflows for post-processing CIFTI-format BOLD data."""
import os

import nibabel as nb
import numpy as np
import sklearn
from nipype import Function, logging
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from num2words import num2words

from xcp_d.interfaces.bids import DerivativesDataSink
from xcp_d.interfaces.filtering import FilteringData
from xcp_d.interfaces.prepostcleaning import CensorScrub, Interpolate, RemoveTR
from xcp_d.interfaces.regression import CiftiDespike, Regress
from xcp_d.utils.bids import collect_run_data
from xcp_d.utils.doc import fill_doc
from xcp_d.utils.plot import plot_design_matrix
from xcp_d.utils.utils import consolidate_confounds, get_customfile, stringforparams
from xcp_d.workflow.connectivity import init_cifti_functional_connectivity_wf
from xcp_d.workflow.execsummary import init_execsummary_wf
from xcp_d.workflow.outputs import init_writederivatives_wf
from xcp_d.workflow.plotting import init_qc_report_wf
from xcp_d.workflow.postprocessing import init_resd_smoothing_wf
from xcp_d.workflow.restingstate import init_cifti_reho_wf, init_compute_alff_wf

LOGGER = logging.getLogger("nipype.workflow")


@fill_doc
def init_ciftipostprocess_wf(
    bold_file,
    lower_bpf,
    upper_bpf,
    bpf_order,
    motion_filter_type,
    motion_filter_order,
    bandpass_filter,
    band_stop_min,
    band_stop_max,
    smoothing,
    head_radius,
    params,
    output_dir,
    custom_confounds_folder,
    omp_nthreads,
    dummytime,
    dummy_scans,
    fd_thresh,
    despike,
    dcan_qc,
    n_runs,
    layout=None,
    name="cifti_process_wf",
):
    """Organize the cifti processing workflow.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.workflow.cifti import init_ciftipostprocess_wf
            wf = init_ciftipostprocess_wf(
                bold_file="/path/to/cifti.dtseries.nii",
                bandpass_filter=True,
                lower_bpf=0.009,
                upper_bpf=0.08,
                bpf_order=2,
                motion_filter_type=None,
                motion_filter_order=4,
                band_stop_min=0,
                band_stop_max=0,
                smoothing=6,
                head_radius=50,
                params="36P",
                output_dir=".",
                custom_confounds_folder=None,
                omp_nthreads=1,
                dummytime=0,
                dummy_scans=0,
                fd_thresh=0.2,
                despike=False,
                dcan_qc=False,
                n_runs=1,
                layout=None,
                name="cifti_postprocess_wf",
            )

    Parameters
    ----------
    bold_file
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
    layout : BIDSLayout object
        BIDS dataset layout
    %(name)s
        Default is "cifti_postprocess_wf".

    Inputs
    ------
    bold_file
        CIFTI file
    custom_confounds_folder
        custom regressors
    t1w
    t1seg
    %(mni_to_t1w)s
    fmriprep_confounds_tsv

    References
    ----------
    .. footbibliography::
    """
    run_data = collect_run_data(layout, bold_file)

    TR = run_data["bold_metadata"]["RepetitionTime"]
    mem_gbx = _create_mem_gb(bold_file)

    workflow = Workflow(name=name)

    filter_str, filter_post_str = "", ""
    if motion_filter_type:
        if motion_filter_type == "notch":
            filter_sub_str = (
                f"band-stop filtered to remove signals between {band_stop_min} and "
                f"{band_stop_max} breaths-per-minute using a notch filter, based on "
                "@fair2020correction"
            )
        else:  # lp
            filter_sub_str = (
                f"low-pass filtered below {band_stop_min} breaths-per-minute, "
                "based on @fair2020correction and @gratton2020removal"
            )

        filter_str = (
            f"the six translation and rotation head motion traces were {filter_sub_str}. Next, "
        )
        filter_post_str = (
            "The filtered versions of the motion traces and framewise displacement were not used "
            "for denoising."
        )

    fd_str = (
        f"{filter_str}framewise displacement was calculated using the formula from "
        f"@power_fd_dvars, with a head radius of {head_radius} mm"
    )

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

    if despike:
        despike_str = "despiked, mean-centered, and linearly detrended"
    else:
        despike_str = "mean-centered and linearly detrended"

    workflow.__desc__ = f"""\
For each of the {num2words(n_runs)} BOLD series found per subject (across all tasks and sessions),
the following post-processing was performed.
First, {dummy_scans_str}outlier detection was performed.
In order to identify high-motion outlier volumes, {fd_str}.
Volumes with {'filtered ' if motion_filter_type else ''}framewise displacement greater than
{fd_thresh} mm were flagged as outliers and excluded from nuisance regression [@power_fd_dvars].
{filter_post_str}
Before nuisance regression, but after censoring, the BOLD data were {despike_str}.
{stringforparams(params=params)} [@benchmarkp;@satterthwaite_2013].
These nuisance regressors were regressed from the BOLD data using linear regression -
as implemented in Scikit-Learn {sklearn.__version__} [@scikit-learn].
Any volumes censored earlier in the workflow were then interpolated in the residual time series
produced by the regression.
The interpolated timeseries were then band-pass filtered to retain signals within the
{lower_bpf}-{upper_bpf} Hz frequency band.
"""

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "bold_file",
                "custom_confounds_folder",
                "t1w",
                "t1seg",
                "mni_to_t1w",
                "fmriprep_confounds_tsv",
                "dummy_scans",
            ],
        ),
        name="inputnode",
    )

    inputnode.inputs.bold_file = bold_file
    inputnode.inputs.custom_confounds_folder = custom_confounds_folder
    inputnode.inputs.fmriprep_confounds_tsv = run_data["confounds"]
    inputnode.inputs.dummy_scans = dummy_scans

    # Load and filter confounds
    get_custom_confounds_file = pe.Node(
        Function(
            input_names=["custom_confounds_folder", "fmriprep_confounds_file"],
            output_names=["custom_confounds_file"],
            function=get_customfile,
        ),
        name="get_custom_confounds_file",
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

    # fmt:off
    workflow.connect([
        (inputnode, get_custom_confounds_file, [
            ("custom_confounds_folder", "custom_confounds_folder"),
            ("fmriprep_confounds_tsv", "fmriprep_confounds_file"),
        ]),
        (inputnode, consolidate_confounds_node, [
            ("bold_file", "img_file"),
        ]),
        (get_custom_confounds_file, consolidate_confounds_node, [
            ("custom_confounds_file", "custom_confounds_file"),
        ]),
    ])
    # fmt:on

    # QC report workflow: Generate figures for derivatives
    qc_report_wf = init_qc_report_wf(
        output_dir=output_dir,
        TR=TR,
        motion_filter_type=motion_filter_type,
        band_stop_max=band_stop_max,
        band_stop_min=band_stop_min,
        motion_filter_order=motion_filter_order,
        fd_thresh=fd_thresh,
        head_radius=head_radius,
        mem_gb=mem_gbx["timeseries"],
        omp_nthreads=omp_nthreads,
        cifti=True,
        name="qc_report_wf",
    )

    # fmt:off
    workflow.connect([
        (inputnode, qc_report_wf, [
            ("bold_file", "inputnode.preprocessed_bold_file"),
        ]),
    ])
    # fmt:on

    # Filter motion parameters and flag high-motion volumes
    censor_scrub = pe.Node(
        CensorScrub(
            TR=TR,
            band_stop_min=band_stop_min,
            band_stop_max=band_stop_max,
            motion_filter_type=motion_filter_type,
            motion_filter_order=motion_filter_order,
            head_radius=head_radius,
            fd_thresh=fd_thresh,
        ),
        name="censoring",
        mem_gb=mem_gbx["timeseries"],
        omp_nthreads=omp_nthreads,
    )

    # Remove dummy scans if requested
    if dummy_scans:
        remove_dummy_scans = pe.Node(
            RemoveTR(),
            name="remove_dummy_scans",
            mem_gb=0.1 * mem_gbx["timeseries"],
        )

        # fmt:off
        workflow.connect([
            (inputnode, remove_dummy_scans, [
                ("bold_file", "bold_file"),
                ("dummy_scans", "dummy_scans"),
                # fMRIPrep confounds file is needed for filtered motion.
                # The selected confounds are not guaranteed to include motion params.
                ("fmriprep_confounds_tsv", "fmriprep_confounds_file"),
            ]),
            (consolidate_confounds_node, remove_dummy_scans, [
                ("out_file", "confounds_file"),
            ]),
            (remove_dummy_scans, censor_scrub, [
                ("bold_file_dropped_TR", "in_file"),
                ("confounds_file_dropped_TR", "confounds_file"),
                # fMRIPrep confounds file is needed for filtered motion.
                # The selected confounds are not guaranteed to include motion params.
                ("fmriprep_confounds_file_dropped_TR", "fmriprep_confounds_file"),
            ]),
            (remove_dummy_scans, qc_report_wf, [
                ("dummy_scans", "inputnode.dummy_scans"),
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
                ("bold_file", "in_file"),
                # fMRIPrep confounds file is needed for filtered motion.
                # The selected confounds are not guaranteed to include motion params.
                ("fmriprep_confounds_tsv", "fmriprep_confounds_file"),
            ]),
            (consolidate_confounds_node, censor_scrub, [
                ("out_file", "confounds_file"),
            ]),
        ])
        # fmt:on

    # Regress nuisance regressors out of censored data
    regression_wf = pe.Node(
        Regress(TR=TR, params=params),
        name="regression_wf",
        mem_gb=mem_gbx["timeseries"],
        n_procs=omp_nthreads,
    )

    if despike:  # If we despike
        despike3d = pe.Node(
            CiftiDespike(TR=TR),
            name="cifti_despike",
            mem_gb=mem_gbx["timeseries"],
            n_procs=omp_nthreads,
        )

        # fmt:off
        workflow.connect([
            (censor_scrub, despike3d, [("bold_censored", "in_file")]),
            (despike3d, regression_wf, [("des_file", "in_file")]),
        ])
        # fmt:on

    else:
        # fmt:off
        workflow.connect([
            (censor_scrub, regression_wf, [("bold_censored", "in_file")]),
        ])
        # fmt:on

    # fmt:off
    workflow.connect([
        (censor_scrub, regression_wf, [("confounds_censored", "confounds")]),
    ])
    # fmt:on

    # Interpolation workflow: Fill in censored volumes in residuals
    interpolate_wf = pe.Node(
        Interpolate(TR=TR),
        name="interpolation_wf",
        mem_gb=mem_gbx["timeseries"],
        n_procs=omp_nthreads,
    )

    # fmt:off
    workflow.connect([
        (inputnode, interpolate_wf, [("bold_file", "bold_file")]),
        (censor_scrub, interpolate_wf, [("tmask", "tmask")]),
        (regression_wf, interpolate_wf, [("res_file", "in_file")])
    ])
    # fmt:on

    # Filtering workflow: Bandpass filter the interpolated residuals
    filtering_wf = pe.Node(
        FilteringData(
            TR=TR,
            lowpass=upper_bpf,
            highpass=lower_bpf,
            filter_order=bpf_order,
            bandpass_filter=bandpass_filter,
        ),
        name="filtering_wf",
        mem_gb=mem_gbx["timeseries"],
        n_procs=omp_nthreads,
    )

    # fmt:off
    workflow.connect([
        (interpolate_wf, filtering_wf, [("bold_interpolated", "in_file")]),
    ])
    # fmt:on

    # functional connectivity workflow
    fcon_ts_wf = init_cifti_functional_connectivity_wf(
        output_dir=output_dir,
        mem_gb=mem_gbx["timeseries"],
        name="cifti_ts_con_wf",
        omp_nthreads=omp_nthreads,
    )

    # fmt:off
    workflow.connect([
        (inputnode, fcon_ts_wf, [("bold_file", "inputnode.bold_file")]),
        (filtering_wf, fcon_ts_wf, [("filtered_file", "inputnode.clean_bold")]),
    ])
    # fmt:on

    # Residual smoothing workflow: Smooth filtered, interpolated residuals
    resd_smoothing_wf = init_resd_smoothing_wf(
        mem_gb=mem_gbx["timeseries"],
        smoothing=smoothing,
        cifti=True,
        name="resd_smoothing_wf",
        omp_nthreads=omp_nthreads,
    )

    # fmt:off
    workflow.connect([
        (filtering_wf, resd_smoothing_wf, [("filtered_file", "inputnode.bold_file")]),
    ])
    # fmt:on

    # reho and alff
    compute_reho_wf = init_cifti_reho_wf(
        output_dir=output_dir,
        mem_gb=mem_gbx["timeseries"],
        omp_nthreads=omp_nthreads,
        name="compute_reho_wf",
    )

    if bandpass_filter:
        compute_alff_wf = init_compute_alff_wf(
            output_dir=output_dir,
            TR=TR,
            lowpass=upper_bpf,
            highpass=lower_bpf,
            smoothing=smoothing,
            cifti=True,
            mem_gb=mem_gbx["timeseries"],
            omp_nthreads=omp_nthreads,
            name="compute_alff_wf",
        )

    # fmt:off
    workflow.connect([
        (inputnode, compute_reho_wf, [("bold_file", "inputnode.bold_file")]),
        (filtering_wf, compute_reho_wf, [("filtered_file", "inputnode.clean_bold")]),
    ])

    if bandpass_filter:
        workflow.connect([
            (inputnode, compute_alff_wf, [("bold_file", "inputnode.bold_file")]),
            (filtering_wf, compute_alff_wf, [("filtered_file", "inputnode.clean_bold")]),
        ])

    # qc report
    workflow.connect([
        (filtering_wf, qc_report_wf, [("filtered_file", "inputnode.cleaned_file")]),
        (censor_scrub, qc_report_wf, [("tmask", "inputnode.tmask")]),
    ])
    # fmt:on

    # write derivatives
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

    # fmt:off
    workflow.connect([
        (consolidate_confounds_node, write_derivative_wf, [
            ("out_file", "inputnode.confounds_file"),
        ]),
        (filtering_wf, write_derivative_wf, [
            ("filtered_file", "inputnode.processed_bold"),
        ]),
        (qc_report_wf, write_derivative_wf, [
            ("outputnode.qc_file", "inputnode.qc_file"),
        ]),
        (resd_smoothing_wf, write_derivative_wf, [
            ("outputnode.smoothed_bold", "inputnode.smoothed_bold"),
        ]),
        (censor_scrub, write_derivative_wf, [
            ("filtered_motion", "inputnode.filtered_motion"),
            ("tmask", "inputnode.tmask"),
        ]),
        (compute_reho_wf, write_derivative_wf, [
            ("outputnode.reho_out", "inputnode.reho_out"),
        ]),
        (fcon_ts_wf, write_derivative_wf, [
            ("outputnode.atlas_names", "inputnode.atlas_names"),
            ("outputnode.correlations", "inputnode.correlations"),
            ("outputnode.timeseries", "inputnode.timeseries"),
        ]),
    ])

    if bandpass_filter:
        workflow.connect([
            (compute_alff_wf, write_derivative_wf, [
                ("outputnode.alff_out", "inputnode.alff_out"),
                ("outputnode.smoothed_alff", "inputnode.smoothed_alff"),
            ]),
        ])
    # fmt:on

    # Create and write out plots
    plot_design_matrix_node = pe.Node(
        Function(
            input_names=["design_matrix"],
            output_names=["design_matrix_figure"],
            function=plot_design_matrix,
        ),
        name="plot_design_matrix_node",
    )

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

    # fmt:off
    workflow.connect([
        (censor_scrub, plot_design_matrix_node, [
            ("confounds_censored", "design_matrix"),
        ]),
        (plot_design_matrix_node, ds_design_matrix_plot, [
            ("design_matrix_figure", "in_file"),
        ]),
    ])
    # fmt:on

    # executive summary workflow
    if dcan_qc:
        executivesummary_wf = init_execsummary_wf(
            TR=TR,
            bold_file=bold_file,
            layout=layout,
            output_dir=output_dir,
            omp_nthreads=omp_nthreads,
            mem_gb=mem_gbx["timeseries"],
        )

        # fmt:off
        workflow.connect([
            (inputnode, executivesummary_wf, [
                ("t1w", "inputnode.t1w"),
                ("t1seg", "inputnode.t1seg"),
                ("bold_file", "inputnode.bold_file"),
                ("mni_to_t1w", "inputnode.mni_to_t1w"),
            ]),
            (regression_wf, executivesummary_wf, [
                ("res_file", "inputnode.regressed_data"),
            ]),
            (filtering_wf, executivesummary_wf, [
                ("filtered_file", "inputnode.residual_data"),
            ]),
            (censor_scrub, executivesummary_wf, [
                ("filtered_motion", "inputnode.filtered_motion"),
                ("tmask", "inputnode.tmask"),
            ]),
        ])
        # fmt:on

        if dummy_scans:
            # fmt:off
            workflow.connect([
                (remove_dummy_scans, executivesummary_wf, [
                    ("dummy_scans", "inputnode.dummy_scans"),
                ]),
            ])
            # fmt:on
        else:
            # fmt:off
            workflow.connect([
                (inputnode, executivesummary_wf, [
                    ("dummy_scans", "inputnode.dummy_scans"),
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
