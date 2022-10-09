# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Workflows for post-processing the BOLD data."""
import os

import nibabel as nb
import numpy as np
import sklearn
from nipype import logging
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from niworkflows.interfaces.fixes import FixHeaderApplyTransforms as ApplyTransforms
from num2words import num2words
from templateflow.api import get as get_template

from xcp_d.interfaces.bids import DerivativesDataSink
from xcp_d.interfaces.filtering import FilteringData
from xcp_d.interfaces.prepostcleaning import CensorScrub, Interpolate, RemoveTR
from xcp_d.interfaces.qc_plot import QCPlot
from xcp_d.interfaces.regression import Regress
from xcp_d.interfaces.report import FunctionalSummary
from xcp_d.interfaces.resting_state import DespikePatch
from xcp_d.utils.concantenation import _t12native
from xcp_d.utils.doc import fill_doc
from xcp_d.utils.utils import (
    get_maskfiles,
    get_transformfile,
    get_transformfilex,
    stringforparams,
)
from xcp_d.workflow.connectivity import init_nifti_functional_connectivity_wf
from xcp_d.workflow.execsummary import init_execsummary_wf
from xcp_d.workflow.outputs import init_writederivatives_wf
from xcp_d.workflow.postprocessing import init_resd_smoothing
from xcp_d.workflow.restingstate import init_3d_reho_wf, init_compute_alff_wf

LOGGER = logging.getLogger('nipype.workflow')


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
    head_radius,
    params,
    omp_nthreads,
    dummytime,
    output_dir,
    fd_thresh,
    n_runs,
    mni_to_t1w,
    despike,
    layout=None,
    name='bold_postprocess_wf',
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
                head_radius=50.,
                params="36P",
                omp_nthreads=1,
                dummytime=0,
                output_dir=".",
                fd_thresh=0.2,
                n_runs=1,
                mni_to_t1w="identity",
                despike=False,
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
    %(head_radius)s
    %(params)s
    %(omp_nthreads)s
    dummytime: float
        the time in seconds to be removed before postprocessing
    output_dir : str
        Directory in which to save xcp_d output
    %(fd_thresh)s
    n_runs
    mni_to_t1w
    despike: bool
        If True, run 3dDespike from AFNI
    layout : BIDSLayout object
        BIDS dataset layout
    %(name)s

    Inputs
    ------
    bold_file
        BOLD series NIfTI file
    ref_file
        Bold reference file from fmriprep
    bold_mask
        bold_mask from fmriprep
    custom_confounds
        custom regressors
    mni_to_t1w
        MNI to T1W ants Transformation file/h5
    t1w
    t1seg
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
    reho_out
        reho output computed by afni.3dreho
    %(atlas_names)s
    %(timeseries)s
    %(correlations)s
    qc_file
        quality control files
    fd
    """
    # Ensure that we know the TR
    metadata = layout.get_metadata(bold_file)
    TR = metadata['RepetitionTime']
    if TR is None:
        TR = layout.get_tr(bold_file)
    if not isinstance(TR, float):
        raise Exception(f"Unable to determine TR of {bold_file}")

    # Confounds file is necessary: ensure we can find it
    from xcp_d.utils.confounds import get_confounds_tsv
    try:
        confounds_tsv = get_confounds_tsv(bold_file)
    except Exception:
        raise Exception(f"Unable to find confounds file for {bold_file}.")

    workflow = Workflow(name=name)

    workflow.__desc__ = f"""
For each of the {num2words(n_runs)} BOLD series found per subject (across all
tasks and sessions), the following post-processing was performed:
"""
    initial_volumes_to_drop = 0
    if dummytime > 0:
        initial_volumes_to_drop = int(np.ceil(dummytime / TR))
        workflow.__desc__ = workflow.__desc__ + f""" \
before nuisance regression and filtering of the data, the first
{num2words(initial_volumes_to_drop)} were discarded, then both
the nuisance regressors and volumes were demeaned and detrended. Furthermore, volumes with
framewise-displacement greater than {fd_thresh} mm [@power_fd_dvars;@satterthwaite_2013] were
flagged as outliers and excluded from nuisance regression.
"""

    else:
        workflow.__desc__ = workflow.__desc__ + f""" \
before nuisance regression and filtering of the data, both the nuisance regressors and
volumes were demean and detrended. Volumes with framewise-displacement greater than
{fd_thresh} mm [@power_fd_dvars;@satterthwaite_2013] were flagged as outliers
and excluded from nuisance regression.
"""

    workflow.__desc__ = workflow.__desc__ + f""" \
{stringforparams(params=params)} [@benchmarkp;@satterthwaite_2013]. These nuisance regressors were
regressed from the BOLD data using linear regression - as implemented in Scikit-Learn
{sklearn.__version__} [@scikit-learn].
Residual timeseries from this regression were then band-pass filtered to retain signals within the
{lower_bpf}-{upper_bpf} Hz frequency band.
 """

    # get reference and mask
    mask_file, ref_file = _get_ref_mask(fname=bold_file)
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'bold_file',
                'ref_file',
                'bold_mask',
                'custom_confounds',
                'mni_to_t1w',
                't1w',
                't1seg',
                'fmriprep_confounds_tsv',
            ],
        ),
        name='inputnode',
    )

    inputnode.inputs.ref_file = str(ref_file)
    inputnode.inputs.bold_mask = str(mask_file)
    inputnode.inputs.fmriprep_confounds_tsv = str(confounds_tsv)

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
                'fd',
            ],
        ),
        name='outputnode',
    )

    mem_gbx = _create_mem_gb(bold_file)

    fcon_ts_wf = init_nifti_functional_connectivity_wf(
        mem_gb=mem_gbx['timeseries'],
        mni_to_t1w=mni_to_t1w,
        t1w_to_native=_t12native(bold_file),
        name="fcons_ts_wf",
        omp_nthreads=omp_nthreads,
    )

    alff_compute_wf = init_compute_alff_wf(mem_gb=mem_gbx['timeseries'],
                                           TR=TR,
                                           lowpass=upper_bpf,
                                           highpass=lower_bpf,
                                           smoothing=smoothing,
                                           cifti=False,
                                           name="compute_alff_wf",
                                           omp_nthreads=omp_nthreads)

    reho_compute_wf = init_3d_reho_wf(mem_gb=mem_gbx['timeseries'],
                                      name="afni_reho_wf",
                                      omp_nthreads=omp_nthreads)

    write_derivative_wf = init_writederivatives_wf(smoothing=smoothing,
                                                   bold_file=bold_file,
                                                   params=params,
                                                   cifti=None,
                                                   output_dir=output_dir,
                                                   dummytime=dummytime,
                                                   lowpass=upper_bpf,
                                                   highpass=lower_bpf,
                                                   TR=TR,
                                                   name="write_derivative_wf")

    censor_scrub = pe.Node(CensorScrub(
        TR=TR,
        custom_confounds=custom_confounds,
        low_freq=band_stop_max,
        high_freq=band_stop_min,
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
        cifti=False,
        name="resd_smoothing_wf",
        omp_nthreads=omp_nthreads)

    filtering_wf = pe.Node(
        FilteringData(
            TR=TR,
            lowpass=upper_bpf,
            highpass=lower_bpf,
            filter_order=bpf_order,
            bandpass_filter=bandpass_filter),
        name="filtering_wf",
        mem_gb=mem_gbx['timeseries'],
        n_procs=omp_nthreads)

    regression_wf = pe.Node(
        Regress(TR=TR,
                original_file=bold_file),
        name="regression_wf",
        mem_gb=mem_gbx['timeseries'],
        n_procs=omp_nthreads)

    interpolate_wf = pe.Node(
        Interpolate(TR=TR),
        name="interpolation_wf",
        mem_gb=mem_gbx['timeseries'],
        n_procs=omp_nthreads)

    executivesummary_wf = init_execsummary_wf(
        TR=TR,
        bold_file=bold_file,
        layout=layout,
        mem_gb=mem_gbx['timeseries'],
        output_dir=output_dir,
        mni_to_t1w=mni_to_t1w,
        omp_nthreads=omp_nthreads)
    # get transform file for resampling and fcon
    transformfile = get_transformfile(bold_file=bold_file,
                                      mni_to_t1w=mni_to_t1w,
                                      t1w_to_native=_t12native(bold_file))
    t1w_mask = get_maskfiles(bold_file=bold_file, mni_to_t1w=mni_to_t1w)[1]

    bold2MNI_trans, bold2T1w_trans = get_transformfilex(
        bold_file=bold_file,
        mni_to_t1w=mni_to_t1w,
        t1w_to_native=_t12native(bold_file))

    resample_parc = pe.Node(ApplyTransforms(
        dimension=3,
        input_image=str(
            get_template('MNI152NLin2009cAsym',
                         resolution=1,
                         desc='carpet',
                         suffix='dseg',
                         extension=['.nii', '.nii.gz'])),
        interpolation='MultiLabel',
        transforms=transformfile),
        name='resample_parc',
        n_procs=omp_nthreads,
        mem_gb=mem_gbx['timeseries'])

    resample_bold2T1w = pe.Node(ApplyTransforms(
        dimension=3,
        input_image=mask_file,
        reference_image=t1w_mask,
        interpolation='NearestNeighbor',
        transforms=bold2T1w_trans),
        name='bold2t1_trans',
        n_procs=omp_nthreads,
        mem_gb=mem_gbx['timeseries'])

    resample_bold2MNI = pe.Node(ApplyTransforms(
        dimension=3,
        input_image=mask_file,
        reference_image=str(
            get_template('MNI152NLin2009cAsym',
                         resolution=2,
                         desc='brain',
                         suffix='mask',
                         extension=['.nii', '.nii.gz'])),
        interpolation='NearestNeighbor',
        transforms=bold2MNI_trans),
        name='bold2mni_trans',
        n_procs=omp_nthreads,
        mem_gb=mem_gbx['timeseries'])

    qcreport = pe.Node(
        QCPlot(
            TR=TR,
            bold_file=bold_file,
            dummytime=dummytime,
            t1w_mask=t1w_mask,
            template_mask=str(
                get_template(
                    'MNI152NLin2009cAsym',
                    resolution=2,
                    desc='brain',
                    suffix='mask',
                    extension=['.nii', '.nii.gz']
                )
            ),
            head_radius=head_radius,
            low_freq=band_stop_max,
            high_freq=band_stop_min),
        name="qc_report",
        mem_gb=mem_gbx['timeseries'],
        n_procs=omp_nthreads,
    )

    # Remove TR first:
    if dummytime > 0:
        rm_dummytime = pe.Node(
            RemoveTR(initial_volumes_to_drop=initial_volumes_to_drop,
                     custom_confounds=custom_confounds),
            name="remove_dummy_time",
            mem_gb=0.1 * mem_gbx['timeseries'])
        workflow.connect([
            (inputnode, rm_dummytime, [('fmriprep_confounds_tsv', 'fmriprep_confounds_file')]),
            (inputnode, rm_dummytime, [('bold_file', 'bold_file')]),
            (inputnode, rm_dummytime, [('custom_confounds', 'custom_confounds')])])

        workflow.connect([
            (rm_dummytime, censor_scrub, [
                ('bold_file_dropped_TR', 'in_file'),
                ('fmriprep_confounds_file_dropped_TR', 'fmriprep_confounds_file'),
                ('custom_confounds_dropped', 'custom_confounds')
            ])])

    else:  # No need to remove TR
        # Censor Scrub:
        workflow.connect([
            (inputnode, censor_scrub, [
                ('bold_file', 'in_file'),
                ('fmriprep_confounds_tsv', 'fmriprep_confounds_file')
            ])])

    if despike:  # If we despike
        # Despiking truncates large spikes in the BOLD times series
        # Despiking reduces/limits the amplitude or magnitude of
        # large spikes but preserves those data points with an imputed
        # reduced amplitude. Despiking is done before regression and filtering
        # to minimize the impact of spike. Despiking is applied to whole volumes
        # and data, and different from temporal censoring. It can be added to the
        # command line arguments with --despike.

        despike3d = pe.Node(DespikePatch(
            outputtype='NIFTI_GZ',
            args='-NEW'),
            name="despike3d",
            mem_gb=mem_gbx['timeseries'],
            n_procs=omp_nthreads)

        workflow.connect([(censor_scrub, despike3d, [('bold_censored', 'in_file')])])
        # Censor Scrub:
        workflow.connect([
            (despike3d, regression_wf, [
                ('out_file', 'in_file')]),
            (inputnode, regression_wf, [('bold_mask', 'mask')]),
            (censor_scrub, regression_wf,
             [('fmriprep_confounds_censored', 'confounds'),
              ('custom_confounds_censored', 'custom_confounds')])])

    else:  # If we don't despike
        # regression workflow
        workflow.connect([(inputnode, regression_wf, [('bold_mask', 'mask')]),
                          (censor_scrub, regression_wf,
                         [('bold_censored', 'in_file'),
                          ('fmriprep_confounds_censored', 'confounds'),
                          ('custom_confounds_censored', 'custom_confounds')])])

    # interpolation workflow
    workflow.connect([
        (inputnode, interpolate_wf, [('bold_file', 'bold_file'),
                                     ('bold_mask', 'mask_file')]),
        (censor_scrub, interpolate_wf, [('tmask', 'tmask')]),
        (regression_wf, interpolate_wf, [('res_file', 'in_file')])
    ])

    # add filtering workflow
    workflow.connect([(inputnode, filtering_wf, [('bold_mask', 'mask')]),
                      (interpolate_wf, filtering_wf, [('bold_interpolated',
                                                       'in_file')])])

    # residual smoothing
    workflow.connect([(filtering_wf, resdsmoothing_wf,
                       [('filtered_file', 'inputnode.bold_file')])])

    # functional connect workflow
    workflow.connect([
        (inputnode, fcon_ts_wf, [('bold_file', 'inputnode.bold_file'),
                                 ('ref_file', 'inputnode.ref_file')]),
        (filtering_wf, fcon_ts_wf, [('filtered_file', 'inputnode.clean_bold')])
    ])

    # reho and alff
    workflow.connect([
        (inputnode, alff_compute_wf, [('bold_mask', 'inputnode.bold_mask')]),
        (inputnode, reho_compute_wf, [('bold_mask', 'inputnode.bold_mask')]),
        (filtering_wf, alff_compute_wf, [('filtered_file', 'inputnode.clean_bold')
                                         ]),
        (filtering_wf, reho_compute_wf, [('filtered_file', 'inputnode.clean_bold')
                                         ]),
    ])

    # qc report
    workflow.connect([
        (inputnode, qcreport, [('bold_mask', 'mask_file')]),
        (filtering_wf, qcreport, [('filtered_file', 'cleaned_file')]),
        (censor_scrub, qcreport, [('tmask', 'tmask')]),
        (inputnode, resample_parc, [('ref_file', 'reference_image')]),
        (resample_parc, qcreport, [('output_image', 'seg_file')]),
        (resample_bold2T1w, qcreport, [('output_image', 'bold2T1w_mask')]),
        (resample_bold2MNI, qcreport, [('output_image', 'bold2temp_mask')]),
        (qcreport, outputnode, [('qc_file', 'qc_file')])
    ])

    # write  to the outputnode, may be use in future
    workflow.connect([
        (filtering_wf, outputnode, [('filtered_file', 'processed_bold')]),
        (censor_scrub, outputnode, [('fd_timeseries', 'fd')]),
        (resdsmoothing_wf, outputnode, [('outputnode.smoothed_bold',
                                         'smoothed_bold')]),
        (alff_compute_wf, outputnode, [('outputnode.alff_out', 'alff_out'),
                                       ('outputnode.smoothed_alff',
                                        'smoothed_alff')]),
        (reho_compute_wf, outputnode, [('outputnode.reho_out', 'reho_out')]),
        (fcon_ts_wf, outputnode, [('outputnode.atlas_names', 'atlas_names'),
                                  ('outputnode.correlations', 'correlations'),
                                  ('outputnode.timeseries', 'timeseries')]),
    ])

    # write derivatives
    workflow.connect([
        (filtering_wf, write_derivative_wf, [('filtered_file',
                                              'inputnode.processed_bold')]),
        (resdsmoothing_wf, write_derivative_wf, [('outputnode.smoothed_bold',
                                                  'inputnode.smoothed_bold')]),
        (censor_scrub, write_derivative_wf, [('fd_timeseries',
                                              'inputnode.fd')]),
        (alff_compute_wf, write_derivative_wf,
         [('outputnode.alff_out', 'inputnode.alff_out'),
          ('outputnode.smoothed_alff', 'inputnode.smoothed_alff')]),
        (reho_compute_wf, write_derivative_wf, [('outputnode.reho_out',
                                                 'inputnode.reho_out')]),
        (fcon_ts_wf, write_derivative_wf, [('outputnode.atlas_names', 'inputnode.atlas_names'),
                                           ('outputnode.correlations', 'inputnode.correlations'),
                                           ('outputnode.timeseries', 'inputnode.timeseries')]),
        (qcreport, write_derivative_wf, [('qc_file', 'inputnode.qc_file')]),
    ])

    functional_qc = pe.Node(FunctionalSummary(bold_file=bold_file, TR=TR),
                            name='qcsummary',
                            run_without_submitting=False,
                            mem_gb=mem_gbx['timeseries'])

    ds_report_qualitycontrol = pe.Node(DerivativesDataSink(
        base_directory=output_dir,
        desc='qualitycontrol',
        source_file=bold_file,
        datatype="figures"),
        name='ds_report_qualitycontrol',
        run_without_submitting=False)

    ds_report_preprocessing = pe.Node(DerivativesDataSink(
        base_directory=output_dir,
        desc='preprocessing',
        source_file=bold_file,
        datatype="figures"),
        name='ds_report_preprocessing',
        run_without_submitting=False)

    ds_report_postprocessing = pe.Node(DerivativesDataSink(
        base_directory=output_dir,
        source_file=bold_file,
        desc='postprocessing',
        datatype="figures"),
        name='ds_report_postprocessing',
        un_without_submitting=False)

    ds_report_connectivity = pe.Node(DerivativesDataSink(
        base_directory=output_dir,
        source_file=bold_file,
        desc='connectvityplot',
        datatype="figures"),
        name='ds_report_connectivity',
        run_without_submitting=False)

    ds_report_rehoplot = pe.Node(DerivativesDataSink(base_directory=output_dir,
                                                     source_file=bold_file,
                                                     desc='rehoplot',
                                                     datatype="figures"),
                                 name='ds_report_rehoplot',
                                 run_without_submitting=False)

    ds_report_afniplot = pe.Node(DerivativesDataSink(base_directory=output_dir,
                                                     source_file=bold_file,
                                                     desc='afniplot',
                                                     datatype="figures"),
                                 name='ds_report_afniplot',
                                 run_without_submitting=False)

    workflow.connect([
        (qcreport, ds_report_preprocessing, [('raw_qcplot', 'in_file')]),
        (qcreport, ds_report_postprocessing, [('clean_qcplot', 'in_file')]),
        (qcreport, functional_qc, [('qc_file', 'qc_file')]),
        (functional_qc, ds_report_qualitycontrol, [('out_report', 'in_file')]),
        (fcon_ts_wf, ds_report_connectivity, [('outputnode.connectplot', 'in_file')]),
        (reho_compute_wf, ds_report_rehoplot, [('outputnode.rehohtml', 'in_file')]),
        (alff_compute_wf, ds_report_afniplot, [('outputnode.alffhtml', 'in_file')]),
    ])

    # exexetive summary workflow
    workflow.connect([
        (inputnode, executivesummary_wf, [('t1w', 'inputnode.t1w'),
                                          ('t1seg', 'inputnode.t1seg'),
                                          ('bold_file', 'inputnode.bold_file'),
                                          ('bold_mask', 'inputnode.mask')]),
        (regression_wf, executivesummary_wf, [('res_file', 'inputnode.regressed_data')
                                              ]),
        (filtering_wf, executivesummary_wf, [('filtered_file',
                                              'inputnode.residual_data')]),
        (censor_scrub, executivesummary_wf, [('fd_timeseries',
                                              'inputnode.fd')]),
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

    if mem_gbz['timeseries'] < 4.0:
        mem_gbz['timeseries'] = 6.0
        mem_gbz['resampled'] = 2
    elif mem_gbz['timeseries'] > 8.0:
        mem_gbz['timeseries'] = 8.0
        mem_gbz['resampled'] = 3

    return mem_gbz


def _get_ref_mask(fname):
    directx = os.path.dirname(fname)
    filename = os.path.basename(fname)
    filex = filename.split('preproc_bold.nii.gz')[0] + 'brain_mask.nii.gz'
    filez = filename.split('_desc-preproc_bold.nii.gz')[0] + '_boldref.nii.gz'
    mask = directx + '/' + filex
    ref = directx + '/' + filez
    return mask, ref
