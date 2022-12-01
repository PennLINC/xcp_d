# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Workflows for post-processing the BOLD data."""
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
from xcp_d.interfaces.prepostcleaning import (
    CensorScrub,
    ConvertTo32,
    Interpolate,
    RemoveTR,
)
from xcp_d.interfaces.regression import Regress
from xcp_d.interfaces.resting_state import DespikePatch
from xcp_d.utils.bids import collect_run_data
from xcp_d.utils.doc import fill_doc
from xcp_d.utils.filemanip import check_binary_mask
from xcp_d.utils.plot import plot_design_matrix
from xcp_d.utils.utils import consolidate_confounds, get_customfile, stringforparams
from xcp_d.workflow.connectivity import init_nifti_functional_connectivity_wf
from xcp_d.workflow.execsummary import init_execsummary_wf
from xcp_d.workflow.outputs import init_writederivatives_wf
from xcp_d.workflow.plotting import init_qc_report_wf
from xcp_d.workflow.postprocessing import init_resd_smoothing_wf
from xcp_d.workflow.restingstate import init_compute_alff_wf, init_nifti_reho_wf

LOGGER = logging.getLogger("nipype.workflow")


@fill_doc
def init_boldpostprocess_wf(
    lower_bpf,
    upper_bpf,
    bpf_order,
    motion_filter_type,
    motion_filter_order,
    bandpass_filter,
    band_stop_min,
    band_stop_max,
    smoothing,
    bold_file,
    head_radius,
    params,
    custom_confounds_folder,
    omp_nthreads,
    dummytime,
    dummy_scans,
    output_dir,
    fd_thresh,
    n_runs,
    despike,
    dcan_qc,
    layout=None,
    name="bold_postprocess_wf",
):
    """Organize the bold processing workflow.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.workflow.bold import init_boldpostprocess_wf
            wf = init_boldpostprocess_wf(
                lower_bpf=0.009,
                upper_bpf=0.08,
                bpf_order=2,
                motion_filter_type=None,
                motion_filter_order=4,
                bandpass_filter=True,
                band_stop_min=0.,
                band_stop_max=0.,
                smoothing=6,
                bold_file="/path/to/file.nii.gz",
                head_radius=50.,
                params="36P",
                custom_confounds_folder=None,
                omp_nthreads=1,
                dummytime=0,
                dummy_scans=0,
                output_dir=".",
                fd_thresh=0.2,
                n_runs=1,
                despike=False,
                dcan_qc=False,
                layout=None,
                name='bold_postprocess_wf',
            )

    Parameters
    ----------
    %(bandpass_filter)s
    %(lower_bpf)s
    %(upper_bpf)s
    %(bpf_order)s
    %(motion_filter_type)s
    %(motion_filter_order)s
    %(band_stop_min)s
    %(band_stop_max)s
    %(smoothing)s
    bold_file: str
        bold file for post processing
    %(head_radius)s
    %(params)s
    custom_confounds: str
        path to cusrtom nuissance regressors
    %(omp_nthreads)s
    %(dummytime)s
    %(dummy_scans)s
    output_dir : str
        Directory in which to save xcp_d output
    %(fd_thresh)s
    n_runs
    despike: bool
        If True, run 3dDespike from AFNI
    dcan_qc : bool
        Whether to run DCAN QC or not.
    layout : BIDSLayout object
        BIDS dataset layout
    %(name)s

    Inputs
    ------
    bold_file
        BOLD series NIfTI file
    ref_file
        Bold reference file from fmriprep
        Loaded in this workflow.
    bold_mask
        bold_mask from fmriprep
        Loaded in this workflow.
    custom_confounds_folder
        custom regressors
    %(template_to_t1w)s
        MNI to T1W ants Transformation file/h5
        Fed from the subject workflow.
    t1w
        Fed from the subject workflow.
    t1seg
        Fed from the subject workflow.
    t1w_mask
        Fed from the subject workflow.
    fmriprep_confounds_tsv
        Loaded in this workflow.

    References
    ----------
    .. footbibliography::
    """
    run_data = collect_run_data(layout, bold_file)

    TR = run_data["bold_metadata"]["RepetitionTime"]

    # TODO: This is a workaround for a bug in nibabies.
    # Once https://github.com/nipreps/nibabies/issues/245 is resolved
    # and a new release is made, remove this.
    mask_file = check_binary_mask(run_data["boldmask"])

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
                "ref_file",
                "bold_mask",
                "custom_confounds_folder",
                "template_to_t1w",
                "t1w",
                "t1seg",
                "t1w_mask",
                "fmriprep_confounds_tsv",
                "t1w_to_native",
                "dummy_scans",
            ],
        ),
        name="inputnode",
    )

    inputnode.inputs.bold_file = bold_file
    inputnode.inputs.ref_file = run_data["boldref"]
    inputnode.inputs.bold_mask = mask_file
    inputnode.inputs.custom_confounds_folder = custom_confounds_folder
    inputnode.inputs.fmriprep_confounds_tsv = run_data["confounds"]
    inputnode.inputs.t1w_to_native = run_data["t1w_to_native_xform"]
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
            ("ref_file", "ref_file"),
            ("bold_mask", "bold_mask"),
            ("t1w", "t1w"),
            ("t1seg", "t1seg"),
            ("t1w_mask", "t1w_mask"),
        ]),
    ])
    # fmt:on

    get_custom_confounds_file = pe.Node(
        Function(
            input_names=["custom_confounds_folder", "fmriprep_confounds_file"],
            output_names=["custom_confounds_file"],
            function=get_customfile,
        ),
        name="get_custom_confounds_file",
    )

    fcon_ts_wf = init_nifti_functional_connectivity_wf(
        mem_gb=mem_gbx["timeseries"],
        name="fcons_ts_wf",
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
            cifti=False,
            name="compute_alff_wf",
            omp_nthreads=omp_nthreads,
        )

    reho_compute_wf = init_nifti_reho_wf(
        mem_gb=mem_gbx["timeseries"],
        bold_file=bold_file,
        name="nifti_reho_wf",
        omp_nthreads=omp_nthreads,
    )

    write_derivative_wf = init_writederivatives_wf(
        smoothing=smoothing,
        bold_file=bold_file,
        bandpass_filter=bandpass_filter,
        params=params,
        cifti=None,
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
        cifti=False,
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
        cifti=False,
        name="qc_report_wf",
    )

    # fmt:off
    workflow.connect([
        (inputnode, qc_report_wf, [
            ("bold_file", "inputnode.preprocessed_bold_file"),
            ("ref_file", "inputnode.boldref"),
            ("bold_mask", "inputnode.bold_mask"),
            ("t1w_mask", "inputnode.t1w_mask"),
            ("template_to_t1w", "inputnode.template_to_t1w"),
            ("t1w_to_native", "inputnode.t1w_to_native"),
        ]),
    ])
    # fmt:on

    # Remove TR first:
    if dummy_scans:
        remove_dummy_scans = pe.Node(
            RemoveTR(),
            name="remove_dummy_scans",
            mem_gb=2*mem_gbx["timeseries"],  # assume it takes a lot of memory
        )

        # fmt:off
        workflow.connect([
            (inputnode, remove_dummy_scans, [
                ("dummy_scans", "dummy_scans"),
                # fMRIPrep confounds file is needed for filtered motion.
                # The selected confounds are not guaranteed to include motion params.
                ("fmriprep_confounds_tsv", "fmriprep_confounds_file"),
            ]),
            (downcast_data, remove_dummy_scans, [("bold_file", "bold_file")]),
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
            (downcast_data, censor_scrub, [
                ('bold_file', 'in_file'),
            ]),
            (inputnode, qc_report_wf, [
                ("dummy_scans", "inputnode.dummy_scans"),
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

    if despike:  # If we despike
        # Despiking truncates large spikes in the BOLD times series
        # Despiking reduces/limits the amplitude or magnitude of
        # large spikes but preserves those data points with an imputed
        # reduced amplitude. Despiking is done before regression and filtering
        # to minimize the impact of spike. Despiking is applied to whole volumes
        # and data, and different from temporal censoring. It can be added to the
        # command line arguments with --despike.
        despike3d = pe.Node(
            DespikePatch(outputtype="NIFTI_GZ", args="-NEW"),
            name="despike3d",
            mem_gb=mem_gbx["timeseries"],
            n_procs=omp_nthreads,
        )

        # fmt:off
        workflow.connect([
            (censor_scrub, despike3d, [('bold_censored', 'in_file')]),
            (despike3d, regression_wf, [('out_file', 'in_file')]),
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
        (downcast_data, regression_wf, [('bold_mask', 'mask')]),
        (censor_scrub, regression_wf, [('confounds_censored', 'confounds')]),
    ])
    # fmt:on

    # interpolation workflow
    # fmt:off
    workflow.connect([
        (downcast_data, interpolate_wf, [('bold_file', 'bold_file'),
                                         ('bold_mask', 'mask_file')]),
        (censor_scrub, interpolate_wf, [('tmask', 'tmask')]),
        (regression_wf, interpolate_wf, [('res_file', 'in_file')])
    ])

    # add filtering workflow
    workflow.connect([(downcast_data, filtering_wf, [('bold_mask', 'mask')]),
                      (interpolate_wf, filtering_wf, [('bold_interpolated',
                                                       'in_file')])])

    # residual smoothing
    workflow.connect([(filtering_wf, resd_smoothing_wf,
                       [('filtered_file', 'inputnode.bold_file')])])

    # functional connect workflow
    workflow.connect([
        (downcast_data, fcon_ts_wf, [('bold_file', 'inputnode.bold_file'),
                                     ('ref_file', 'inputnode.ref_file')]),
        (inputnode, fcon_ts_wf, [('template_to_t1w', 'inputnode.template_to_t1w'),
                                 ('t1w_to_native', 'inputnode.t1w_to_native')]),
        (filtering_wf, fcon_ts_wf, [('filtered_file', 'inputnode.clean_bold')])
    ])

    # reho and alff
    workflow.connect([
        (downcast_data, reho_compute_wf, [('bold_mask', 'inputnode.bold_mask')]),
        (filtering_wf, reho_compute_wf, [('filtered_file', 'inputnode.clean_bold')]),
    ])

    if bandpass_filter:
        workflow.connect([
            (downcast_data, alff_compute_wf, [('bold_mask', 'inputnode.bold_mask')]),
            (filtering_wf, alff_compute_wf, [('filtered_file', 'inputnode.clean_bold')]),
        ])

    # qc report
    workflow.connect([
        (filtering_wf, qc_report_wf, [('filtered_file', 'inputnode.cleaned_file')]),
        (censor_scrub, qc_report_wf, [('tmask', 'inputnode.tmask')]),
    ])
    # fmt:on

    # write derivatives
    # fmt:off
    workflow.connect([
        (consolidate_confounds_node, write_derivative_wf, [
            ('out_file', 'inputnode.confounds_file'),
        ]),
        (filtering_wf, write_derivative_wf, [
            ('filtered_file', 'inputnode.processed_bold'),
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
    # fmt:on

    if bandpass_filter:
        # fmt:off
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
        run_without_submitting=False,
    )

    ds_report_rehoplot = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            source_file=bold_file,
            desc="rehoVolumetricPlot",
            datatype="figures",
        ),
        name="ds_report_rehoplot",
        run_without_submitting=False,
    )

    # fmt:off
    workflow.connect([
        (plot_design_matrix_node, ds_design_matrix_plot, [("design_matrix_figure", "in_file")]),
        (fcon_ts_wf, ds_report_connectivity, [('outputnode.connectplot', 'in_file')]),
        (reho_compute_wf, ds_report_rehoplot, [('outputnode.rehoplot', 'in_file')]),
    ])
    # fmt:on

    if bandpass_filter:
        ds_report_alffplot = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                source_file=bold_file,
                desc="alffVolumetricPlot",
                datatype="figures",
            ),
            name="ds_report_alffplot",
            run_without_submitting=False,
        )

        # fmt:off
        workflow.connect([
            (alff_compute_wf, ds_report_alffplot, [('outputnode.alffplot', 'in_file')]),
        ])
        # fmt:on

    # executive summary workflow
    if dcan_qc:
        executivesummary_wf = init_execsummary_wf(
            TR=TR,
            bold_file=bold_file,
            layout=layout,
            mem_gb=mem_gbx["timeseries"],
            output_dir=output_dir,
            omp_nthreads=omp_nthreads,
        )

        # fmt:off
        workflow.connect([
            (downcast_data, executivesummary_wf, [
                ("t1w", "inputnode.t1w"),
                ("t1seg", "inputnode.t1seg"),
                ("bold_file", "inputnode.bold_file"),
                ("bold_mask", "inputnode.mask"),
            ]),
            (inputnode, executivesummary_wf, [
                ("template_to_t1w", "inputnode.template_to_t1w"),
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

    if mem_gbz["timeseries"] < 4.0:
        mem_gbz["timeseries"] = 6.0
        mem_gbz["resampled"] = 2
    elif mem_gbz["timeseries"] > 8.0:
        mem_gbz["timeseries"] = 8.0
        mem_gbz["resampled"] = 3

    return mem_gbz
