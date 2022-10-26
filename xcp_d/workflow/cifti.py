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
from xcp_d.interfaces.prepostcleaning import CensorScrub
from xcp_d.interfaces.qc_plot import CensoringPlot, QCPlot
from xcp_d.interfaces.regression import CiftiDespike
from xcp_d.interfaces.report import FunctionalSummary
from xcp_d.utils.doc import fill_doc
from xcp_d.utils.plot import _get_tr
from xcp_d.utils.utils import consolidate_confounds, denoise_cifti_with_nilearn, stringforparams
from xcp_d.workflow.connectivity import init_cifti_functional_connectivity_wf
from xcp_d.workflow.execsummary import init_execsummary_wf
from xcp_d.workflow.outputs import init_writederivatives_wf
from xcp_d.workflow.postprocessing import init_resd_smoothing
from xcp_d.workflow.restingstate import init_cifti_reho_wf, init_compute_alff_wf

LOGGER = logging.getLogger('nipype.workflow')


@fill_doc
def init_ciftipostprocess_wf(
    bold_file,
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
    custom_confounds,
    omp_nthreads,
    dummytime,
    fd_thresh,
    despike,
    n_runs,
    layout=None,
    name='cifti_process_wf',
):
    """Organize the cifti processing workflow.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.workflow.cifti import init_ciftipostprocess_wf
            wf = init_ciftipostprocess_wf(
                bold_file="/path/to/cifti.dtseries.nii",
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
                custom_confounds=None,
                omp_nthreads=1,
                dummytime=0,
                fd_thresh=0.2,
                despike=False,
                n_runs=1,
                layout=None,
                name='cifti_postprocess_wf',
            )

    Parameters
    ----------
    bold_file
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
    custom_confounds: str
        path to cusrtom nuissance regressors
    %(omp_nthreads)s
    dummytime: float
        the first few seconds to be removed before postprocessing
    %(fd_thresh)s
    despike: bool
        afni depsike
    n_runs
    layout : BIDSLayout object
        BIDS dataset layout
    %(name)s
        Default is 'cifti_postprocess_wf'.

    Inputs
    ------
    bold_file
        CIFTI file
    custom_confounds
        custom regressors
    t1w
    t1seg
    %(mni_to_t1w)s
    fmriprep_confounds_tsv

    Outputs
    -------
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
    %(atlas_names)s
    %(timeseries)s
    %(correlations)s
    qc_file
        quality control files

    References
    ----------
    .. footbibliography::
    """
    TR = _get_tr(bold_file)
    if TR is None:
        metadata = layout.get_metadata(bold_file)
        TR = metadata['RepetitionTime']

    # Confounds file is necessary: ensure we can find it
    from xcp_d.utils.confounds import get_confounds_tsv
    try:
        confounds_tsv = get_confounds_tsv(bold_file)
    except Exception:
        raise Exception(f"Unable to find confounds file for {bold_file}.")

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
            f"the six translation and rotation head motion traces were {filter_sub_str}. "
            "Next, "
        )
        filter_post_str = (
            "The filtered versions of the motion traces and framewise displacement were not used "
            "for denoising."
        )

    fd_str = (
        f"{filter_str}framewise displacement was calculated using the formula from "
        f"@power_fd_dvars, with a head radius of {head_radius} mm"
    )

    dummytime_str = ""
    initial_volumes_to_drop = 0
    if dummytime > 0:
        initial_volumes_to_drop = int(np.ceil(dummytime / TR))
        dummytime_str = (
            f"the first {num2words(initial_volumes_to_drop)} of both the BOLD data and nuisance "
            "regressors were discarded, then "
        )

    if despike:
        despike_str = "despiked, mean-centered, and linearly detrended"
    else:
        despike_str = "mean-centered and linearly detrended"

    workflow.__desc__ = f"""\
For each of the {num2words(n_runs)} BOLD series found per subject (across all tasks and sessions),
the following post-processing was performed.
First, {dummytime_str}outlier detection was performed.
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
                'bold_file',
                'custom_confounds',
                't1w',
                't1seg',
                'mni_to_t1w',
                'fmriprep_confounds_tsv',
            ],
        ),
        name='inputnode',
    )

    inputnode.inputs.bold_file = bold_file
    inputnode.inputs.fmriprep_confounds_tsv = confounds_tsv

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'processed_bold',
                'smoothed_bold',
                'alff_out',
                'smoothed_alff',
                'reho_out',
                'atlas_names',
                'timeseries',
                'correlations',
                'qc_file',
                'filtered_motion',
                'tmask',
            ],
        ),
        name='outputnode',
    )

    mem_gbx = _create_mem_gb(bold_file)

    fcon_ts_wf = init_cifti_functional_connectivity_wf(
        mem_gb=mem_gbx['timeseries'],
        name='cifti_ts_con_wf',
        omp_nthreads=omp_nthreads)

    alff_compute_wf = init_compute_alff_wf(
        mem_gb=mem_gbx['timeseries'],
        TR=TR,
        lowpass=upper_bpf,
        highpass=lower_bpf,
        smoothing=smoothing,
        cifti=True,
        name="compute_alff_wf",
        omp_nthreads=omp_nthreads)

    reho_compute_wf = init_cifti_reho_wf(
        mem_gb=mem_gbx['timeseries'],
        name="cifti_reho_wf",
        omp_nthreads=omp_nthreads)

    write_derivative_wf = init_writederivatives_wf(
        smoothing=smoothing,
        bold_file=bold_file,
        params=params,
        cifti=True,
        output_dir=output_dir,
        dummytime=dummytime,
        lowpass=upper_bpf,
        highpass=lower_bpf,
        motion_filter_type=motion_filter_type,
        TR=TR,
        name="write_derivative_wf",
    )

    censor_scrub = pe.Node(CensorScrub(
        TR=TR,
        custom_confounds=custom_confounds,
        band_stop_min=band_stop_min,
        band_stop_max=band_stop_max,
        motion_filter_type=motion_filter_type,
        motion_filter_order=motion_filter_order,
        head_radius=head_radius,
        fd_thresh=fd_thresh),
        name='censoring',
        mem_gb=mem_gbx['timeseries'],
        omp_nthreads=omp_nthreads)

    resdsmoothing_wf = init_resd_smoothing(
        mem_gb=mem_gbx['timeseries'],
        smoothing=smoothing,
        cifti=True,
        name="resd_smoothing_wf",
        omp_nthreads=omp_nthreads)

    denoise_bold = pe.Node(
        Function(
            input_names=[
                "bold_file",
                "fmriprep_confounds_file",
                "custom_confounds_file",
                "censoring_file",
                "namesource",
                "low_pass",
                "high_pass",
                "TR",
                "params",
            ],
            output_names=["out_file"],
            function=denoise_cifti_with_nilearn,
        ),
        name="denoise_bold",
        mem_gb=mem_gbx['timeseries'],
        n_procs=omp_nthreads,
    )
    denoise_bold.inputs.high_pass = lower_bpf
    denoise_bold.inputs.low_pass = upper_bpf
    denoise_bold.inputs.TR = TR

    qcreport = pe.Node(
        QCPlot(
            TR=TR,
            dummytime=dummytime,
            head_radius=head_radius,
        ),
        name="qc_report",
        mem_gb=mem_gbx['resampled'],
        n_procs=omp_nthreads)

    censor_report = pe.Node(
        CensoringPlot(
            TR=TR,
            dummytime=dummytime,
            head_radius=head_radius,
            motion_filter_type=motion_filter_type,
            band_stop_max=band_stop_max,
            band_stop_min=band_stop_min,
            motion_filter_order=motion_filter_order,
            fd_thresh=fd_thresh,
        ),
        name="censor_report",
        mem_gb=mem_gbx["timeseries"],
        n_procs=omp_nthreads,
    )

    executivesummary_wf = init_execsummary_wf(
        TR=TR,
        bold_file=bold_file,
        layout=layout,
        output_dir=output_dir,
        omp_nthreads=omp_nthreads,
        mem_gb=mem_gbx['timeseries'],
    )

    # A node to hold outputs from either rm_dummytime or inputnode
    bold_holder_node = pe.Node(
        niu.IdentityInterface(
            fields=["bold_file", "fmriprep_confounds_tsv", "custom_confounds"],
        ),
        name="bold_holder_node",
    )

    # Combine confounds into a single file
    consolidate_confounds_node = pe.Node(
        Function(
            input_names=[
                "fmriprep_confounds_file",
                "custom_confounds_file",
                "namesource",
                "params",
            ],
            output_names=["out_file"],
            function=consolidate_confounds,
        )
    )
    consolidate_confounds_node.inputs.params = params

    # Censor Scrub:
    workflow.connect([
        (inputnode, censor_scrub, [
            ('bold_file', 'in_file'),
            ('fmriprep_confounds_tsv', 'fmriprep_confounds_file')
        ])])

    if despike:  # If we despike
        despike3d = pe.Node(CiftiDespike(TR=TR),
                            name="cifti_despike",
                            mem_gb=mem_gbx['timeseries'],
                            n_procs=omp_nthreads)

        workflow.connect([(bold_holder_node, despike3d, [('bold_file', 'in_file')]),
                          (despike3d, denoise_bold, [('out_file', 'bold_file')])])

    else:  # If we don't despike
        # regression workflow
        workflow.connect([(bold_holder_node, denoise_bold, [('bold_file', 'bold_file')])])

    workflow.connect([
        (inputnode, consolidate_confounds_node, [('bold_file', 'namesource')]),
        (inputnode, denoise_bold, [('bold_mask', 'mask_file')]),
        (bold_holder_node, consolidate_confounds_node, [
            ('fmriprep_confounds_tsv', 'fmriprep_confounds_file'),
            ('custom_confounds', 'custom_confounds_file'),
        ]),
        (consolidate_confounds_node, denoise_bold, [('confounds', 'confounds')]),
    ])

    workflow.connect([
        (censor_scrub, denoise_bold, [('tmask', 'censoring_file')]),
    ])

    # residual smoothing
    workflow.connect([
        (denoise_bold, resdsmoothing_wf, [('out_file', 'inputnode.bold_file')]),
    ])

    # functional connect workflow
    workflow.connect([(denoise_bold, fcon_ts_wf, [('out_file', 'inputnode.clean_bold')])])

    # reho and alff
    workflow.connect([(denoise_bold, alff_compute_wf, [('out_file', 'inputnode.clean_bold')]),
                      (denoise_bold, reho_compute_wf, [('out_file', 'inputnode.clean_bold')])])

    # qc report
    workflow.connect([
        (inputnode, qcreport, [("bold_file", "bold_file")]),
        (inputnode, censor_report, [("bold_file", "bold_file")]),
        (denoise_bold, qcreport, [('out_file', 'cleaned_file')]),
        (censor_scrub, qcreport, [("tmask", "tmask")]),
        (censor_scrub, censor_report, [('tmask', 'tmask')]),
        (qcreport, outputnode, [('qc_file', 'qc_file')])
    ])

    workflow.connect([
        (denoise_bold, outputnode, [('out_file', 'processed_bold')]),
        (censor_scrub, outputnode, [('filtered_motion', 'filtered_motion'),
                                    ('tmask', 'tmask')]),
        (resdsmoothing_wf, outputnode, [('outputnode.smoothed_bold',
                                         'smoothed_bold')]),
        (alff_compute_wf, outputnode, [('outputnode.alff_out', 'alff_out')]),
        (reho_compute_wf, outputnode, [('outputnode.reho_out', 'reho_out')]),
        (fcon_ts_wf, outputnode, [('outputnode.atlas_names', 'atlas_names'),
                                  ('outputnode.correlations', 'correlations'),
                                  ('outputnode.timeseries', 'timeseries')]),
    ])

    # write derivatives
    workflow.connect([
        (denoise_bold, write_derivative_wf, [('out_file', 'inputnode.processed_bold')]),
        (resdsmoothing_wf, write_derivative_wf, [
            ('outputnode.smoothed_bold', 'inputnode.smoothed_bold'),
        ]),
        (censor_scrub, write_derivative_wf, [('filtered_motion', 'inputnode.filtered_motion'),
                                             ('tmask', 'inputnode.tmask')]),
        (alff_compute_wf, write_derivative_wf, [
            ('outputnode.alff_out', 'inputnode.alff_out'),
            ('outputnode.smoothed_alff', 'inputnode.smoothed_alff'),
        ]),
        (reho_compute_wf, write_derivative_wf, [
            ('outputnode.reho_out', 'inputnode.reho_out')
        ]),
        (fcon_ts_wf, write_derivative_wf, [
            ('outputnode.atlas_names', 'inputnode.atlas_names'),
            ('outputnode.correlations', 'inputnode.correlations'),
            ('outputnode.timeseries', 'inputnode.timeseries'),
        ]),
        (qcreport, write_derivative_wf, [('qc_file', 'inputnode.qc_file')])
    ])

    functional_qc = pe.Node(FunctionalSummary(bold_file=bold_file, TR=TR),
                            name='qcsummary',
                            run_without_submitting=True)

    ds_report_qualitycontrol = pe.Node(DerivativesDataSink(
        base_directory=output_dir,
        desc='qualitycontrol',
        source_file=bold_file,
        datatype="figures"),
        name='ds_report_qualitycontrol',
        run_without_submitting=True)

    ds_report_preprocessing = pe.Node(DerivativesDataSink(
        base_directory=output_dir,
        source_file=bold_file,
        desc='preprocessing',
        datatype="figures"),
        name='ds_report_preprocessing',
        run_without_submitting=True)

    ds_report_censoring = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            source_file=bold_file,
            datatype="figures",
            desc="censoring",
            suffix="motion",
            extension=".svg",
        ),
        name='ds_report_censoring',
        run_without_submitting=False,
    )

    ds_report_postprocessing = pe.Node(DerivativesDataSink(
        base_directory=output_dir,
        source_file=bold_file,
        desc='postprocessing',
        datatype="figures"),
        name='ds_report_postprocessing',
        run_without_submitting=True)

    ds_report_connectivity = pe.Node(DerivativesDataSink(
        base_directory=output_dir,
        source_file=bold_file,
        desc='connectivityplot',
        datatype="figures"),
        name='ds_report_connectivity',
        run_without_submitting=True)

    workflow.connect([
        (qcreport, ds_report_preprocessing, [('raw_qcplot', 'in_file')]),
        (qcreport, ds_report_postprocessing, [('clean_qcplot', 'in_file')]),
        (qcreport, functional_qc, [('qc_file', 'qc_file')]),
        (censor_report, ds_report_censoring, [("out_file", "in_file")]),
        (functional_qc, ds_report_qualitycontrol, [('out_report', 'in_file')]),
        (fcon_ts_wf, ds_report_connectivity, [('outputnode.connectplot', "in_file")])
    ])

    # exexetive summary workflow
    workflow.connect([
        (inputnode, executivesummary_wf, [('t1w', 'inputnode.t1w'),
                                          ('t1seg', 'inputnode.t1seg'),
                                          ('bold_file', 'inputnode.bold_file'),
                                          ('mni_to_t1w', 'inputnode.mni_to_t1w')]),
        (denoise_bold, executivesummary_wf, [('out_file', 'inputnode.residual_data')]),
        (censor_scrub, executivesummary_wf, [('filtered_motion', 'inputnode.filtered_motion')]),
    ])

    return workflow


def _create_mem_gb(bold_fname):
    bold_size_gb = os.path.getsize(bold_fname) / (1024**3)
    bold_tlen = nb.load(bold_fname).shape[-1]
    mem_gbz = {
        'derivative': bold_size_gb,
        'resampled': bold_size_gb * 4,
        'timeseries': bold_size_gb * (max(bold_tlen / 100, 1.0) + 4),
    }

    return mem_gbz
