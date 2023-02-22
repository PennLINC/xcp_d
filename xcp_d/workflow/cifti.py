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
from xcp_d.interfaces.filtering import FilteringData
from xcp_d.interfaces.prepostcleaning import (
    Censor,
    CensorScrub,
    ConvertTo32,
    Interpolate,
    RemoveTR,
)
from xcp_d.interfaces.regression import Regress
from xcp_d.interfaces.resting_state import DespikePatch
from xcp_d.interfaces.workbench import CiftiConvert
from xcp_d.utils.confounds import (
    consolidate_confounds,
    describe_regression,
    get_customfile,
)
from xcp_d.utils.doc import fill_doc
from xcp_d.utils.plot import plot_design_matrix
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
    dummytime,
    dummy_scans,
    fd_thresh,
    despike,
    dcan_qc,
    n_runs,
    run_data,
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
            from xcp_d.workflow.cifti import init_ciftipostprocess_wf
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
            run_data = {
                "boldref": "",
                "confounds": "",
                "bold_metadata": {"RepetitionTime": 2},
            }

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
                dummy_scans=0,
                dummytime=0,
                fd_thresh=0.2,
                despike=True,
                dcan_qc=True,
                n_runs=1,
                run_data=run_data,
                omp_nthreads=1,
                layout=layout,
                name="cifti_postprocess_wf",
            )
            wf.inputs.inputnode.t1w = subj_data["t1w"]
            wf.inputs.inputnode.t1seg = subj_data["t1w_seg"]

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
        Default is 'cifti_postprocess_wf'.

    Inputs
    ------
    bold_file
        CIFTI file
    custom_confounds_file
        custom regressors
    t1w
    t1seg
    fmriprep_confounds_tsv

    References
    ----------
    .. footbibliography::
    """
    workflow = Workflow(name=name)

    TR = run_data["bold_metadata"]["RepetitionTime"]

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "bold_file",
                "custom_confounds_file",
                "t1w",
                "t1seg",
                "fmriprep_confounds_tsv",
                "dummy_scans",
            ],
        ),
        name="inputnode",
    )

    inputnode.inputs.bold_file = bold_file
    inputnode.inputs.ref_file = run_data["boldref"]
    inputnode.inputs.fmriprep_confounds_tsv = run_data["confounds"]
    inputnode.inputs.dummy_scans = dummy_scans

    # Load custom confounds
    # We need to run this function directly to access information in the confounds that is
    # used for the boilerplate.
    custom_confounds_file = get_customfile(
        custom_confounds_folder,
        run_data["confounds"],
    )
    inputnode.inputs.custom_confounds_file = custom_confounds_file
    regression_description = describe_regression(params, custom_confounds_file)

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
            "The interpolated timeseries were then band-pass filtered to retain signals within "
            f"the {lower_bpf}-{upper_bpf} Hz frequency band."
        )

    workflow.__desc__ = f"""\
For each of the {num2words(n_runs)} BOLD series found per subject (across all tasks and sessions),
the following post-processing was performed.
First, {dummy_scans_str}outlier detection was performed.
In order to identify high-motion outlier volumes, {fd_str}.
Volumes with {'filtered ' if motion_filter_type else ''}framewise displacement greater than
{fd_thresh} mm were flagged as outliers and excluded from nuisance regression [@power_fd_dvars].
{filter_post_str}
{despike_str}
Next, the BOLD data and confounds were mean-centered and linearly detrended.
{regression_description}
Any volumes censored earlier in the workflow were then interpolated in the residual time series
produced by the regression.
{bandpass_str}
"""

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
            ("t1w", "t1w"),
            ("t1seg", "t1seg"),
        ]),
    ])
    # fmt:on

    fcon_ts_wf = init_cifti_functional_connectivity_wf(
        mem_gb=mem_gbx["timeseries"], name="cifti_ts_con_wf", omp_nthreads=omp_nthreads
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

    resd_smoothing_wf = init_resd_smoothing_wf(
        mem_gb=mem_gbx["timeseries"],
        smoothing=smoothing,
        cifti=True,
        name="resd_smoothing_wf",
        omp_nthreads=omp_nthreads,
    )

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

    censor_interpolated_data = pe.Node(
        Censor(),
        name="censor_interpolated_data",
        mem_gb=mem_gbx["timeseries"],
        omp_nthreads=omp_nthreads,
    )

    # fmt:off
    workflow.connect([
        (filtering_wf, censor_interpolated_data, [("filtered_file", "in_file")]),
        (censor_scrub, censor_interpolated_data, [("tmask", "temporal_mask")]),
    ])
    # fmt:on

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
            input_names=["design_matrix"],
            output_names=["design_matrix_figure"],
            function=plot_design_matrix,
        ),
        name="plot_design_matrix_node",
    )

    regression_wf = pe.Node(
        Regress(TR=TR, params=params),
        name="regression_wf",
        mem_gb=mem_gbx["timeseries"],
        n_procs=omp_nthreads,
    )

    interpolate_wf = pe.Node(
        Interpolate(TR=TR),
        name="interpolation_wf",
        mem_gb=mem_gbx["timeseries"],
        n_procs=omp_nthreads,
    )

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
        dcan_qc=dcan_qc,
        cifti=True,
        name="qc_report_wf",
    )

    # fmt:off
    workflow.connect([
        (inputnode, qc_report_wf, [
            ("bold_file", "inputnode.preprocessed_bold_file"),
        ]),
        (regression_wf, qc_report_wf, [
            ("res_file", "inputnode.cleaned_unfiltered_file"),
        ]),
    ])
    # fmt:on

    # Remove TR first
    if dummy_scans:
        remove_dummy_scans = pe.Node(
            RemoveTR(),
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
            (downcast_data, censor_scrub, [
                ('bold_file', 'in_file'),
            ]),
            (inputnode, censor_scrub, [
                # fMRIPrep confounds file is needed for filtered motion.
                # The selected confounds are not guaranteed to include motion params.
                ("fmriprep_confounds_tsv", "fmriprep_confounds_file"),
            ]),
            (consolidate_confounds_node, censor_scrub, [
                ("out_file", "confounds_file"),
            ]),
        ])
        # fmt:on

    # fmt:off
    workflow.connect([
        (censor_scrub, plot_design_matrix_node, [
            ("confounds_censored", "design_matrix"),
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
            (censor_scrub, convert_to_nifti, [("bold_censored", "in_file")]),
            (convert_to_nifti, despike3d, [("out_file", "in_file")]),
            (censor_scrub, convert_to_cifti, [("bold_censored", "cifti_template")]),
            (despike3d, convert_to_cifti, [("out_file", "in_file")]),
            (convert_to_cifti, regression_wf, [("out_file", "in_file")]),
        ])
        # fmt:on

    else:
        # fmt:off
        workflow.connect([
            (censor_scrub, regression_wf, [('bold_censored', 'in_file')]),
        ])
        # fmt:on

    # fmt:off
    workflow.connect([
        (censor_scrub, regression_wf, [('confounds_censored', 'confounds')]),
    ])
    # fmt:on

    # interpolation workflow
    # fmt:off
    workflow.connect([
        (downcast_data, interpolate_wf, [('bold_file', 'bold_file')]),
        (censor_scrub, interpolate_wf, [('tmask', 'tmask')]),
        (regression_wf, interpolate_wf, [('res_file', 'in_file')])
    ])

    # add filtering workflow
    workflow.connect([(interpolate_wf, filtering_wf, [('bold_interpolated',
                                                       'in_file')])])

    # residual smoothing
    workflow.connect([
        (censor_interpolated_data, resd_smoothing_wf, [('bold_censored', 'inputnode.bold_file')]),
    ])

    # functional connect workflow
    workflow.connect([
        (censor_interpolated_data, fcon_ts_wf, [('bold_censored', 'inputnode.clean_bold')]),
    ])

    # reho and alff
    workflow.connect([
        (censor_interpolated_data, reho_compute_wf, [('bold_censored', 'inputnode.clean_bold')]),
    ])

    if bandpass_filter:
        workflow.connect([
            (censor_interpolated_data, alff_compute_wf, [
                ('bold_censored', 'inputnode.clean_bold'),
            ]),
        ])

    # qc report
    workflow.connect([
        (filtering_wf, qc_report_wf, [("filtered_file", "inputnode.cleaned_file")]),
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
        (censor_interpolated_data, write_derivative_wf, [
            ('bold_censored', 'inputnode.processed_bold'),
        ]),
        (qc_report_wf, write_derivative_wf, [
            ('outputnode.qc_file', 'inputnode.qc_file'),
        ]),
        (resd_smoothing_wf, write_derivative_wf, [
            ('outputnode.smoothed_bold', 'inputnode.smoothed_bold'),
        ]),
        (censor_scrub, write_derivative_wf, [
            ('filtered_motion', 'inputnode.filtered_motion'),
            ('tmask', 'inputnode.tmask'),
        ]),
        (reho_compute_wf, write_derivative_wf, [
            ('outputnode.reho_out', 'inputnode.reho_out'),
        ]),
        (fcon_ts_wf, write_derivative_wf, [
            ('outputnode.atlas_names', 'inputnode.atlas_names'),
            ('outputnode.correlations', 'inputnode.correlations'),
            ('outputnode.timeseries', 'inputnode.timeseries'),
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
        executive_summary_wf = init_execsummary_wf(
            bold_file=bold_file,
            layout=layout,
            output_dir=output_dir,
            name="executive_summary_wf",
        )

        # fmt:off
        # Use inputnode for executive summary instead of downcast_data
        # because T1w is used as name source.
        workflow.connect([
            # Use inputnode for executive summary instead of downcast_data
            # because T1w is used as name source.
            (inputnode, executive_summary_wf, [
                ('bold_file', 'inputnode.bold_file'),
                ("ref_file", "inputnode.boldref_file"),
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
