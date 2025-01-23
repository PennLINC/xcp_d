# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Workflows for post-processing BOLD data."""

import yaml
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from num2words import num2words
from templateflow.api import get as get_template

from xcp_d import config
from xcp_d.interfaces.bids import DerivativesDataSink
from xcp_d.interfaces.censoring import (
    Censor,
    GenerateConfounds,
    ProcessMotion,
    RandomCensor,
    RemoveDummyVolumes,
)
from xcp_d.interfaces.nilearn import DenoiseCifti, DenoiseNifti, Smooth
from xcp_d.interfaces.plotting import CensoringPlot
from xcp_d.interfaces.restingstate import DespikePatch
from xcp_d.interfaces.workbench import CiftiConvert, CiftiSmooth, FixCiftiIntent
from xcp_d.utils.boilerplate import (
    describe_censoring,
    describe_motion_parameters,
    describe_regression,
)
from xcp_d.utils.doc import fill_doc
from xcp_d.utils.plotting import plot_design_matrix as _plot_design_matrix
from xcp_d.utils.utils import fwhm2sigma, is_number


@fill_doc
def init_prepare_confounds_wf(
    TR,
    exact_scans,
    head_radius,
    name='prepare_confounds_wf',
):
    """Prepare confounds.

    This workflow loads and consolidates confounds, removes dummy volumes,
    filters motion parameters, calculates framewise displacement, and flags outlier volumes.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.tests.tests import mock_config
            from xcp_d import config
            from xcp_d.workflows.postprocessing import init_prepare_confounds_wf

            with mock_config():
                wf = init_prepare_confounds_wf(
                    TR=0.8,
                    exact_scans=[],
                    head_radius=70,
                    name="prepare_confounds_wf",
                )

    Parameters
    ----------
    %(TR)s
    %(exact_scans)s
    %(head_radius)s
        This will already be estimated before this workflow.
    %(name)s
        Default is "prepare_confounds_wf".

    Inputs
    ------
    %(name_source)s
    preprocessed_bold : :obj:`str`
    motion_file : :obj:`str`
        The motion parameters estimated by fMRIPrep.
    motion_json : :obj:`str`
        JSON file associated with the fMRIPrep confounds TSV.
    confounds_files : :obj:`dict`
    %(dummy_scans)s
        Set from the parameter.

    Outputs
    -------
    preprocessed_bold : :obj:`str`
    motion_file
    confounds_tsv : :obj:`str`
        The selected confounds, potentially including custom confounds, after dummy scan removal.
    confounds_images : :obj:`list` of :obj:`str`
    confounds_metadata : :obj:`dict`
    %(dummy_scans)s
        If originally set to "auto", this output will have the actual number of dummy volumes.
    motion_file
    motion_metadata : :obj:`dict`
    %(temporal_mask)s
    temporal_mask_metadata : :obj:`dict`
    """
    workflow = Workflow(name=name)

    dummy_scans = config.workflow.dummy_scans
    motion_filter_type = config.workflow.motion_filter_type
    band_stop_min = config.workflow.band_stop_min
    band_stop_max = config.workflow.band_stop_max
    motion_filter_order = config.workflow.motion_filter_order
    fd_thresh = config.workflow.fd_thresh

    dummy_scans_str = ''
    if dummy_scans == 'auto':
        dummy_scans_str = (
            'Non-steady-state volumes were extracted from the preprocessed confounds '
            'and were discarded from both the BOLD data and nuisance regressors. '
        )
    elif dummy_scans == 1:
        dummy_scans_str = (
            'The first volume of both the BOLD data and nuisance '
            "regressors was discarded as a non-steady-state volume, or 'dummy scan'. "
        )
    elif dummy_scans > 1:
        dummy_scans_str = (
            f'The first {num2words(dummy_scans)} volumes of both the BOLD data and nuisance '
            "regressors were discarded as non-steady-state volumes, or 'dummy scans'. "
        )

    motion_description = describe_motion_parameters(
        motion_filter_type=motion_filter_type,
        motion_filter_order=motion_filter_order,
        band_stop_min=band_stop_min,
        band_stop_max=band_stop_max,
        TR=TR,
    )

    if (fd_thresh > 0) or exact_scans:
        censoring_description = describe_censoring(
            motion_filter_type=motion_filter_type,
            head_radius=head_radius,
            fd_thresh=fd_thresh,
            exact_scans=exact_scans,
        )
    else:
        censoring_description = ''

    if config.execution.confounds_config is None:
        confounds_config = None
    else:
        confounds_config = yaml.safe_load(config.execution.confounds_config.read_text())

    confounds_description = describe_regression(
        confounds_config=confounds_config,
        motion_filter_type=motion_filter_type,
        motion_filter_order=motion_filter_order,
        band_stop_min=band_stop_min,
        band_stop_max=band_stop_max,
        TR=TR,
        fd_thresh=fd_thresh,
    )

    workflow.__desc__ = (
        f' {dummy_scans_str}{motion_description} {censoring_description} {confounds_description}'
    )

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'name_source',
                'preprocessed_bold',
                'dummy_scans',
                'motion_file',
                'motion_json',
                'confounds_files',
            ],
        ),
        name='inputnode',
    )
    inputnode.inputs.dummy_scans = dummy_scans

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'preprocessed_bold',
                'dummy_scans',
                'motion_file',
                'motion_metadata',
                'temporal_mask',
                'temporal_mask_metadata',
                'confounds_tsv',
                'confounds_images',
                'confounds_metadata',
            ],
        ),
        name='outputnode',
    )

    # Filter motion parameters and calculate FD for censoring
    process_motion = pe.Node(
        ProcessMotion(
            TR=TR,
            band_stop_min=band_stop_min,
            band_stop_max=band_stop_max,
            motion_filter_type=motion_filter_type,
            motion_filter_order=motion_filter_order,
            fd_thresh=fd_thresh,
            head_radius=head_radius,
        ),
        name='process_motion',
        mem_gb=1,
        n_procs=1,
    )

    workflow.connect([
        (inputnode, process_motion, [
            ('motion_file', 'motion_file'),
            ('motion_json', 'motion_json'),
        ]),
        (process_motion, outputnode, [('motion_metadata', 'motion_metadata')]),
    ])  # fmt:skip

    if config.execution.confounds_config is not None:
        confounds_config = yaml.safe_load(config.execution.confounds_config.read_text())

        generate_confounds = pe.Node(
            GenerateConfounds(
                confounds_config=confounds_config,
                TR=TR,
                dataset_links=config.execution.dataset_links,
                out_dir=str(config.execution.output_dir),
                band_stop_min=band_stop_min,
                band_stop_max=band_stop_max,
                motion_filter_type=motion_filter_type,
                motion_filter_order=motion_filter_order,
            ),
            name='generate_confounds',
            mem_gb=2,
        )

        # Load and filter confounds
        workflow.connect([
            (inputnode, generate_confounds, [
                ('name_source', 'in_file'),
                ('confounds_files', 'confounds_files'),
            ]),
            (generate_confounds, outputnode, [('confounds_metadata', 'confounds_metadata')]),
        ])  # fmt:skip

    # A buffer node to hold either the original files or the files with the first N vols removed.
    dummy_scan_buffer = pe.Node(
        niu.IdentityInterface(
            fields=[
                'preprocessed_bold',
                'dummy_scans',
                'confounds_tsv',
                'confounds_images',
                'motion_file',
                'temporal_mask',
            ]
        ),
        name='dummy_scan_buffer',
    )

    if dummy_scans:
        remove_dummy_scans = pe.Node(
            RemoveDummyVolumes(),
            name='remove_dummy_scans',
            mem_gb=4,
        )

        workflow.connect([
            (inputnode, remove_dummy_scans, [
                ('preprocessed_bold', 'bold_file'),
                ('dummy_scans', 'dummy_scans'),
                # *not* the filtered motion file, which has dummy volume columns removed
                ('motion_file', 'motion_file'),
            ]),
            (process_motion, remove_dummy_scans, [('temporal_mask', 'temporal_mask')]),
            (remove_dummy_scans, dummy_scan_buffer, [
                ('bold_file_dropped_TR', 'preprocessed_bold'),
                ('confounds_tsv_dropped_TR', 'confounds_tsv'),
                ('confounds_images_dropped_TR', 'confounds_images'),
                ('motion_file_dropped_TR', 'motion_file'),
                ('temporal_mask_dropped_TR', 'temporal_mask'),
                ('dummy_scans', 'dummy_scans'),
            ]),
        ])  # fmt:skip

        if config.execution.confounds_config is not None:
            workflow.connect([
                (generate_confounds, remove_dummy_scans, [
                    ('confounds_tsv', 'confounds_tsv'),
                    ('confounds_images', 'confounds_images'),
                ]),
            ])  # fmt:skip
    else:
        workflow.connect([
            (inputnode, dummy_scan_buffer, [
                ('dummy_scans', 'dummy_scans'),
                ('preprocessed_bold', 'preprocessed_bold'),
            ]),
            (process_motion, dummy_scan_buffer, [
                ('motion_file', 'motion_file'),
                ('temporal_mask', 'temporal_mask'),
            ]),
        ])  # fmt:skip

        if config.execution.confounds_config is not None:
            workflow.connect([
                (generate_confounds, dummy_scan_buffer, [
                    ('confounds_tsv', 'confounds_tsv'),
                    ('confounds_images', 'confounds_images'),
                ]),
            ])  # fmt:skip

    workflow.connect([
        (dummy_scan_buffer, outputnode, [
            ('preprocessed_bold', 'preprocessed_bold'),
            ('dummy_scans', 'dummy_scans'),
            ('motion_file', 'motion_file'),
        ]),
    ])  # fmt:skip

    if config.execution.confounds_config is not None:
        workflow.connect([
            (dummy_scan_buffer, outputnode, [
                ('confounds_tsv', 'confounds_tsv'),
                ('confounds_images', 'confounds_images'),
            ]),
        ])  # fmt:skip

    if any(is_number(length) for length in config.workflow.correlation_lengths):
        random_censor = pe.Node(
            RandomCensor(exact_scans=exact_scans, random_seed=config.seeds.master),
            name='random_censor',
        )

        workflow.connect([
            (process_motion, random_censor, [
                ('temporal_mask_metadata', 'temporal_mask_metadata'),
            ]),
            (dummy_scan_buffer, random_censor, [('temporal_mask', 'temporal_mask')]),
            (random_censor, outputnode, [
                ('temporal_mask', 'temporal_mask'),
                ('temporal_mask_metadata', 'temporal_mask_metadata'),
            ]),
        ])  # fmt:skip
    else:
        workflow.connect([
            (process_motion, outputnode, [('temporal_mask_metadata', 'temporal_mask_metadata')]),
            (dummy_scan_buffer, outputnode, [('temporal_mask', 'temporal_mask')]),
        ])  # fmt:skip

    if config.execution.confounds_config is not None:
        plot_design_matrix = pe.Node(
            niu.Function(
                input_names=['design_matrix', 'temporal_mask'],
                output_names=['design_matrix_figure'],
                function=_plot_design_matrix,
            ),
            name='plot_design_matrix',
        )

        workflow.connect([
            (dummy_scan_buffer, plot_design_matrix, [('confounds_tsv', 'design_matrix')]),
            (outputnode, plot_design_matrix, [('temporal_mask', 'temporal_mask')]),
        ])  # fmt:skip

        ds_report_design_matrix = pe.Node(
            DerivativesDataSink(
                dismiss_entities=['space', 'res', 'den', 'desc'],
                suffix='design',
                extension='.svg',
            ),
            name='ds_report_design_matrix',
            run_without_submitting=False,
        )

        workflow.connect([
            (inputnode, ds_report_design_matrix, [('name_source', 'source_file')]),
            (plot_design_matrix, ds_report_design_matrix, [('design_matrix_figure', 'in_file')]),
        ])  # fmt:skip

    censor_report = pe.Node(
        CensoringPlot(
            TR=TR,
            motion_filter_type=motion_filter_type,
            fd_thresh=fd_thresh,
            head_radius=head_radius,
        ),
        name='censor_report',
        mem_gb=2,
    )

    workflow.connect([
        # use the full version of the confounds, for dummy scans in the figure
        (process_motion, censor_report, [('motion_file', 'motion_file')]),
        (dummy_scan_buffer, censor_report, [('dummy_scans', 'dummy_scans')]),
        (outputnode, censor_report, [('temporal_mask', 'temporal_mask')]),
    ])  # fmt:skip

    ds_report_censoring = pe.Node(
        DerivativesDataSink(
            desc='censoring',
            suffix='motion',
            extension='.svg',
        ),
        name='ds_report_censoring',
        run_without_submitting=False,
    )

    workflow.connect([
        (inputnode, ds_report_censoring, [('name_source', 'source_file')]),
        (censor_report, ds_report_censoring, [('out_file', 'in_file')]),
    ])  # fmt:skip

    return workflow


@fill_doc
def init_despike_wf(TR, name='despike_wf'):
    """Despike BOLD data with AFNI's 3dDespike.

    Despiking truncates large spikes in the BOLD times series.
    Despiking reduces/limits the amplitude or magnitude of large spikes,
    but preserves those data points with an imputed reduced amplitude.
    Despiking is done before regression and filtering to minimize the impact of spikes.
    Despiking is applied to whole volumes and data, and different from temporal censoring.
    It can be added to the command line arguments with ``--despike``.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.tests.tests import mock_config
            from xcp_d import config
            from xcp_d.workflows.postprocessing import init_despike_wf

            with mock_config():
                wf = init_despike_wf(
                    TR=0.8,
                    name="despike_wf",
                )

    Parameters
    ----------
    %(TR)s
    %(name)s
        Default is "despike_wf".

    Inputs
    ------
    bold_file : :obj:`str`
        A NIFTI or CIFTI BOLD file to despike.

    Outputs
    -------
    bold_file : :obj:`str`
        The despiked NIFTI or CIFTI BOLD file.
    """
    workflow = Workflow(name=name)
    file_format = config.workflow.file_format
    omp_nthreads = config.nipype.omp_nthreads

    inputnode = pe.Node(niu.IdentityInterface(fields=['bold_file']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['bold_file']), name='outputnode')

    despike3d = pe.Node(
        DespikePatch(outputtype='NIFTI_GZ', args='-nomask -NEW'),
        name='despike3d',
        mem_gb=4,
        n_procs=omp_nthreads,
    )

    if file_format == 'cifti':
        workflow.__desc__ = """
The BOLD data were converted to NIfTI format, despiked with *AFNI*'s *3dDespike*,
and converted back to CIFTI format.
"""

        # first, convert the cifti to a nifti
        convert_to_nifti = pe.Node(
            CiftiConvert(target='to', num_threads=config.nipype.omp_nthreads),
            name='convert_to_nifti',
            mem_gb=4,
            n_procs=config.nipype.omp_nthreads,
        )
        workflow.connect([
            (inputnode, convert_to_nifti, [('bold_file', 'in_file')]),
            (convert_to_nifti, despike3d, [('out_file', 'in_file')]),
        ])  # fmt:skip

        # finally, convert the despiked nifti back to cifti
        convert_to_cifti = pe.Node(
            CiftiConvert(target='from', TR=TR, num_threads=config.nipype.omp_nthreads),
            name='convert_to_cifti',
            mem_gb=4,
            n_procs=config.nipype.omp_nthreads,
        )
        workflow.connect([
            (inputnode, convert_to_cifti, [('bold_file', 'cifti_template')]),
            (despike3d, convert_to_cifti, [('out_file', 'in_file')]),
            (convert_to_cifti, outputnode, [('out_file', 'bold_file')]),
        ])  # fmt:skip

    else:
        workflow.__desc__ = """
The BOLD data were despiked with *AFNI*'s *3dDespike*.
"""
        workflow.connect([
            (inputnode, despike3d, [('bold_file', 'in_file')]),
            (despike3d, outputnode, [('out_file', 'bold_file')]),
        ])  # fmt:skip

    return workflow


@fill_doc
def init_denoise_bold_wf(TR, mem_gb, name='denoise_bold_wf'):
    """Denoise BOLD data.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.tests.tests import mock_config
            from xcp_d import config
            from xcp_d.workflows.postprocessing import init_denoise_bold_wf

            with mock_config():
                wf = init_denoise_bold_wf(
                    TR=0.8,
                    name="denoise_bold_wf",
                )

    Parameters
    ----------
    %(TR)s
    mem_gb : :obj:`dict`
        Memory size in GB to use for each of the nodes.
    %(name)s
        Default is "denoise_bold_wf".

    Inputs
    ------
    preprocessed_bold
    %(temporal_mask)s
    mask
    confounds_file

    Outputs
    -------
    %(denoised_interpolated_bold)s
    %(censored_denoised_bold)s
    %(smoothed_denoised_bold)s
    denoised_bold
        The selected denoised BOLD data to write out.
    """
    workflow = Workflow(name=name)

    fd_thresh = config.workflow.fd_thresh
    low_pass = config.workflow.low_pass
    high_pass = config.workflow.high_pass
    bpf_order = config.workflow.bpf_order
    bandpass_filter = config.workflow.bandpass_filter
    smoothing = config.workflow.smoothing
    file_format = config.workflow.file_format

    workflow.__desc__ = """\

Nuisance regressors were regressed from the BOLD data using a denoising method based on *Nilearn*'s
approach.
"""
    if fd_thresh > 0:
        workflow.__desc__ += (
            'Any volumes censored earlier in the workflow were first cubic spline interpolated in '
            'the BOLD data. '
            'Outlier volumes at the beginning or end of the time series were replaced with the '
            "closest low-motion volume's values, "
            'as cubic spline interpolation can produce extreme extrapolations. '
        )

    if bandpass_filter:
        if low_pass > 0 and high_pass > 0:
            btype = 'band-pass'
            preposition = 'between'
            filt_input = f'{high_pass}-{low_pass}'
        elif high_pass > 0:
            btype = 'high-pass'
            preposition = 'above'
            filt_input = f'{high_pass}'
        elif low_pass > 0:
            btype = 'low-pass'
            preposition = 'below'
            filt_input = f'{low_pass}'

        workflow.__desc__ += (
            f'The timeseries were {btype} filtered using a(n) '
            f'{num2words(bpf_order, ordinal=True)}-order Butterworth filter, '
            f'in order to retain signals {preposition} {filt_input} Hz. '
            'The same filter was applied to the confounds.'
        )

    if fd_thresh > 0:
        workflow.__desc__ += (
            ' The resulting time series were then denoised via linear regression, '
            'in which the low-motion volumes from the BOLD time series and confounds were used to '
            'calculate parameter estimates, and then the interpolated time series were denoised '
            'using the low-motion parameter estimates. '
            'The interpolated time series were then censored using the temporal mask.'
        )
    else:
        workflow.__desc__ += (
            ' The resulting time series were then denoised using linear regression. '
        )

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'preprocessed_bold',
                'temporal_mask',
                'confounds_tsv',
                'confounds_images',
                'mask',  # only used for NIFTIs
            ],
        ),
        name='inputnode',
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'denoised_interpolated_bold',
                'censored_denoised_bold',
                'smoothed_denoised_bold',
                'denoised_bold',
            ],
        ),
        name='outputnode',
    )

    denoising_interface = DenoiseCifti if (file_format == 'cifti') else DenoiseNifti
    regress_and_filter_bold = pe.Node(
        denoising_interface(
            TR=TR,
            low_pass=low_pass,
            high_pass=high_pass,
            filter_order=bpf_order,
            bandpass_filter=bandpass_filter,
        ),
        name='regress_and_filter_bold',
        mem_gb=mem_gb['timeseries'],
    )

    workflow.connect([
        (inputnode, regress_and_filter_bold, [
            ('preprocessed_bold', 'preprocessed_bold'),
            ('confounds_tsv', 'confounds_tsv'),
            ('confounds_images', 'confounds_images'),
            ('temporal_mask', 'temporal_mask'),
        ]),
        (regress_and_filter_bold, outputnode, [
            ('denoised_interpolated_bold', 'denoised_interpolated_bold'),
        ]),
    ])  # fmt:skip
    if file_format == 'nifti':
        workflow.connect([(inputnode, regress_and_filter_bold, [('mask', 'mask')])])

    censor_interpolated_data = pe.Node(
        Censor(column='framewise_displacement'),
        name='censor_interpolated_data',
        mem_gb=mem_gb['resampled'],
    )

    workflow.connect([
        (inputnode, censor_interpolated_data, [('temporal_mask', 'temporal_mask')]),
        (regress_and_filter_bold, censor_interpolated_data, [
            ('denoised_interpolated_bold', 'in_file'),
        ]),
        (censor_interpolated_data, outputnode, [('out_file', 'censored_denoised_bold')]),
    ])  # fmt:skip

    denoised_bold_buffer = pe.Node(
        niu.IdentityInterface(fields=['denoised_bold']),
        name='denoised_bold_buffer',
    )
    if config.workflow.output_interpolated:
        workflow.connect([
            (regress_and_filter_bold, denoised_bold_buffer, [
                ('denoised_interpolated_bold', 'denoised_bold'),
            ]),
        ])  # fmt:skip
    else:
        workflow.connect([
            (censor_interpolated_data, denoised_bold_buffer, [('out_file', 'denoised_bold')]),
        ])  # fmt:skip

    workflow.connect([(denoised_bold_buffer, outputnode, [('denoised_bold', 'denoised_bold')])])

    if smoothing:
        resd_smoothing_wf = init_resd_smoothing_wf(mem_gb=mem_gb)

        workflow.connect([
            (denoised_bold_buffer, resd_smoothing_wf, [('denoised_bold', 'inputnode.bold_file')]),
            (resd_smoothing_wf, outputnode, [
                ('outputnode.smoothed_bold', 'smoothed_denoised_bold'),
            ]),
        ])  # fmt:skip

    return workflow


@fill_doc
def init_resd_smoothing_wf(mem_gb, name='resd_smoothing_wf'):
    """Smooth BOLD residuals.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.tests.tests import mock_config
            from xcp_d import config
            from xcp_d.workflows.postprocessing import init_resd_smoothing_wf

            with mock_config():
                wf = init_resd_smoothing_wf()

    Parameters
    ----------
    mem_gb : :obj:`dict`
        Memory size in GB to use for each of the nodes.
    %(name)s
        Default is "resd_smoothing_wf".

    Inputs
    ------
    bold_file

    Outputs
    -------
    smoothed_bold
    """
    workflow = Workflow(name=name)
    smoothing = config.workflow.smoothing
    file_format = config.workflow.file_format

    inputnode = pe.Node(niu.IdentityInterface(fields=['bold_file']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['smoothed_bold']), name='outputnode')

    # Turn specified FWHM (Full-Width at Half Maximum) to standard deviation.
    sigma_lx = fwhm2sigma(smoothing)
    if file_format == 'cifti':
        workflow.__desc__ = f""" \
The denoised BOLD was then smoothed using *Connectome Workbench* with a Gaussian kernel
(FWHM={str(smoothing)} mm).
"""

        # Call connectome workbench to smooth for each hemisphere
        smooth_data = pe.Node(
            CiftiSmooth(
                sigma_surf=sigma_lx,  # the size of the surface kernel
                sigma_vol=sigma_lx,  # the volume of the surface kernel
                direction='COLUMN',  # which direction to smooth along@
                # pull out atlases for each hemisphere
                right_surf=str(
                    get_template(
                        template='fsLR',
                        space=None,
                        hemi='R',
                        density='32k',
                        desc=None,
                        suffix='sphere',
                    )
                ),
                left_surf=str(
                    get_template(
                        template='fsLR',
                        space=None,
                        hemi='L',
                        density='32k',
                        desc=None,
                        suffix='sphere',
                    )
                ),
                num_threads=config.nipype.omp_nthreads,
            ),
            name='cifti_smoothing',
            mem_gb=mem_gb['timeseries'],
            n_procs=config.nipype.omp_nthreads,
        )

        # Always check the intent code in CiftiSmooth's output file
        fix_cifti_intent = pe.Node(
            FixCiftiIntent(),
            name='fix_cifti_intent',
            mem_gb=1,
        )
        workflow.connect([
            (smooth_data, fix_cifti_intent, [('out_file', 'in_file')]),
            (fix_cifti_intent, outputnode, [('out_file', 'smoothed_bold')]),
        ])  # fmt:skip

    else:
        workflow.__desc__ = f""" \
The denoised BOLD was smoothed using *Nilearn* with a Gaussian kernel (FWHM={str(smoothing)} mm).
"""
        # Use nilearn to smooth the image
        smooth_data = pe.Node(
            Smooth(fwhm=smoothing),  # FWHM = kernel size
            name='nifti_smoothing',
            mem_gb=mem_gb['timeseries'],
        )
        workflow.connect([(smooth_data, outputnode, [('out_file', 'smoothed_bold')])])

    workflow.connect([(inputnode, smooth_data, [('bold_file', 'in_file')])])

    return workflow
