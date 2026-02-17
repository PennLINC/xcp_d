# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Workflows for calculating BOLD metrics (ALFF and ReHo)."""

from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from templateflow.api import get as get_template

from xcp_d import config
from xcp_d.config import dismiss_hash
from xcp_d.interfaces.bids import DerivativesDataSink
from xcp_d.interfaces.nilearn import Smooth
from xcp_d.interfaces.plotting import PlotDenseCifti, PlotNifti
from xcp_d.interfaces.restingstate import ComputeALFF, ComputePerAF, ReHoNamePatch, SurfaceReHo
from xcp_d.interfaces.workbench import (
    CiftiCreateDenseFromTemplate,
    CiftiSeparateMetric,
    CiftiSeparateVolumeAll,
    CiftiSmooth,
    FixCiftiIntent,
)
from xcp_d.utils.doc import fill_doc
from xcp_d.utils.utils import fwhm2sigma


@fill_doc
def init_alff_wf(
    name_source,
    TR,
    mem_gb,
    name='alff_wf',
):
    """Compute alff for both nifti and cifti.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.tests.tests import mock_config
            from xcp_d import config
            from xcp_d.workflows.bold.metrics import init_alff_wf

            with mock_config():
                wf = init_alff_wf(
                    name_source="/path/to/file.nii.gz",
                    TR=2.,
                    mem_gb={"bold": 0.1},
                    name="alff_wf",
                )

    Parameters
    ----------
    name_source
    %(TR)s
    mem_gb : :obj:`dict`
        Memory allocation dictionary
    %(name)s
        Default is "compute_alff_wf".

    Inputs
    ------
    denoised_bold
       This is the ``filtered, interpolated, denoised BOLD``,
       although interpolation is not necessary if the data were not originally censored.
    bold_mask
       bold mask if bold is nifti
    temporal_mask
    name_source

    Outputs
    -------
    alff
        alff output
    smoothed_alff
        smoothed alff  output

    Notes
    -----
    The ALFF implementation is based on :footcite:t:`yu2007altered`,
    although the ALFF values are not scaled by the mean ALFF value across the brain.

    If censoring is applied (i.e., ``fd_thresh > 0``), then the power spectrum will be estimated
    using a Lomb-Scargle periodogram
    :footcite:p:`lomb1976least,scargle1982studies,townsend2010fast,taylorlomb`.

    This workflow will also generate a plot of the ALFF map.
    For CIFTI data, the plot will be overlaid on the midthickness surface-
    either the subject's surface warped to fsLR space (when the anatomical workflow is enabled)
    or the fsLR 32k midthickness surface template.

    References
    ----------
    .. footbibliography::
    """
    workflow = Workflow(name=name)

    low_pass = config.workflow.low_pass
    high_pass = config.workflow.high_pass
    fd_thresh = config.workflow.fd_thresh
    smoothing = config.workflow.smoothing
    file_format = config.workflow.file_format

    periodogram_desc = ''
    if fd_thresh > 0:
        periodogram_desc = (
            ' using the Lomb-Scargle periodogram '
            '[@lomb1976least;@scargle1982studies;@townsend2010fast;@taylorlomb]'
        )

    workflow.__desc__ = f""" \

The amplitude of low-frequency fluctuation (ALFF) [@alff] was computed by transforming
the mean-centered, standard deviation-normalized, denoised BOLD time series to the frequency
domain{periodogram_desc}.
The power spectrum was computed within the {high_pass}-{low_pass} Hz frequency band and the
mean square root of the power spectrum was calculated at each voxel to yield voxel-wise ALFF
measures.
The resulting ALFF values were then multiplied by the standard deviation of the denoised BOLD time
series to retain the original scaling.
"""

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'denoised_bold',
                'bold_mask',
                'temporal_mask',
                # only used for CIFTI data if the anatomical workflow is enabled
                'lh_midthickness',
                'rh_midthickness',
            ],
        ),
        name='inputnode',
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['alff', 'smoothed_alff']),
        name='outputnode',
    )

    # compute alff
    alff_compt = pe.Node(
        ComputeALFF(
            TR=TR,
            low_pass=low_pass,
            high_pass=high_pass,
            n_threads=config.nipype.omp_nthreads,
        ),
        mem_gb=2 * mem_gb['bold'],
        name='alff_compt',
        n_procs=config.nipype.omp_nthreads,
    )
    workflow.connect([
        (inputnode, alff_compt, [
            ('denoised_bold', 'in_file'),
            ('bold_mask', 'mask'),
            ('temporal_mask', 'temporal_mask'),
        ]),
        (alff_compt, outputnode, [('alff', 'alff')])
    ])  # fmt:skip

    # Plot the ALFF map
    ds_report_alff = pe.Node(
        DerivativesDataSink(
            dismiss_entities=dismiss_hash(),
            source_file=name_source,
        ),
        name='ds_report_alff',
        run_without_submitting=True,
    )

    if file_format == 'cifti':
        alff_plot = pe.Node(
            PlotDenseCifti(base_desc='alff'),
            name='alff_plot',
        )
        workflow.connect([
            (inputnode, alff_plot, [
                ('lh_midthickness', 'lh_underlay'),
                ('rh_midthickness', 'rh_underlay'),
            ]),
            (alff_plot, ds_report_alff, [('desc', 'desc')]),
        ])  # fmt:skip
    else:
        alff_plot = pe.Node(
            PlotNifti(name_source=name_source),
            name='alff_plot',
        )
        ds_report_alff.inputs.desc = 'alffVolumetricPlot'

    workflow.connect([
        (alff_compt, alff_plot, [('alff', 'in_file')]),
        (alff_plot, ds_report_alff, [('out_file', 'in_file')]),
    ])  # fmt:skip

    if smoothing:  # If we want to smooth
        if file_format == 'nifti':
            workflow.__desc__ = workflow.__desc__ + (
                ' The ALFF maps were smoothed with Nilearn using a Gaussian kernel '
                f'(FWHM={str(smoothing)} mm).'
            )
            # Smooth via Nilearn
            smooth_data = pe.Node(
                Smooth(fwhm=smoothing),
                name='niftismoothing',
            )
            workflow.connect([
                (alff_compt, smooth_data, [('alff', 'in_file')]),
                (smooth_data, outputnode, [('out_file', 'smoothed_alff')])
            ])  # fmt:skip

        else:  # If cifti
            workflow.__desc__ = workflow.__desc__ + (
                ' The ALFF maps were smoothed with the Connectome Workbench using a Gaussian '
                f'kernel (FWHM={str(smoothing)} mm).'
            )

            # Smooth via Connectome Workbench
            sigma_lx = fwhm2sigma(smoothing)  # Convert fwhm to standard deviation
            # Get templates for each hemisphere
            lh_midthickness = str(
                get_template('fsLR', hemi='L', suffix='sphere', density='32k', raise_empty=True)[0]
            )
            rh_midthickness = str(
                get_template('fsLR', hemi='R', suffix='sphere', density='32k', raise_empty=True)[0]
            )
            smooth_data = pe.Node(
                CiftiSmooth(
                    sigma_surf=sigma_lx,
                    sigma_vol=sigma_lx,
                    direction='COLUMN',
                    right_surf=rh_midthickness,
                    left_surf=lh_midthickness,
                    num_threads=config.nipype.omp_nthreads,
                ),
                name='ciftismoothing',
                mem_gb=mem_gb['bold'],
                n_procs=config.nipype.omp_nthreads,
            )

            # Always check the intent code in CiftiSmooth's output file
            fix_cifti_intent = pe.Node(
                FixCiftiIntent(),
                name='fix_cifti_intent',
                mem_gb=mem_gb['bold'],
            )
            workflow.connect([
                (alff_compt, smooth_data, [('alff', 'in_file')]),
                (smooth_data, fix_cifti_intent, [('out_file', 'in_file')]),
                (fix_cifti_intent, outputnode, [('out_file', 'smoothed_alff')]),
            ])  # fmt:skip

    return workflow


@fill_doc
def init_reho_cifti_wf(
    name_source,
    mem_gb,
    name='cifti_reho_wf',
):
    """Compute ReHo from surface+volumetric (CIFTI) data.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.tests.tests import mock_config
            from xcp_d import config
            from xcp_d.workflows.bold.metrics import init_reho_cifti_wf

            with mock_config():
                wf = init_reho_cifti_wf(
                    name_source="/path/to/bold.dtseries.nii",
                    mem_gb={"bold": 0.1},
                    name="cifti_reho_wf",
                )

    Parameters
    ----------
    name_source
    mem_gb : :obj:`dict`
        Memory allocation dictionary
    %(name)s
        Default is "cifti_reho_wf".

    Inputs
    ------
    denoised_bold
       residual and filtered, cifti
    name_source

    Outputs
    -------
    reho
        ReHo in a CIFTI file.

    Notes
    -----
    This workflow will also generate a plot of the ReHo map.
    The plot will be overlaid on the midthickness surface-
    either the subject's surface warped to fsLR space (when the anatomical workflow is enabled)
    or the fsLR 32k midthickness surface template.
    """
    workflow = Workflow(name=name)
    workflow.__desc__ = """

For each hemisphere, regional homogeneity (ReHo) [@jiang2016regional] was computed using
surface-based *2dReHo* [@surface_reho].
Specifically, for each vertex on the surface, the Kendall's coefficient of concordance (KCC)
was computed with nearest-neighbor vertices to yield ReHo.
For the subcortical, volumetric data, ReHo was computed with neighborhood voxels using *AFNI*'s
*3dReHo* [@taylor2013fatcat].
"""

    inputnode = pe.Node(
        niu.IdentityInterface(fields=['denoised_bold', 'lh_midthickness', 'rh_midthickness']),
        name='inputnode',
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['reho']),
        name='outputnode',
    )

    # Extract left and right hemispheres via Connectome Workbench
    lh_surf = pe.Node(
        CiftiSeparateMetric(
            metric='CORTEX_LEFT',
            direction='COLUMN',
            num_threads=config.nipype.omp_nthreads,
        ),
        name='separate_lh',
        mem_gb=mem_gb['bold'],
        n_procs=config.nipype.omp_nthreads,
    )
    rh_surf = pe.Node(
        CiftiSeparateMetric(
            metric='CORTEX_RIGHT',
            direction='COLUMN',
            num_threads=config.nipype.omp_nthreads,
        ),
        name='separate_rh',
        mem_gb=mem_gb['bold'],
        n_procs=config.nipype.omp_nthreads,
    )
    subcortical_nifti = pe.Node(
        CiftiSeparateVolumeAll(
            direction='COLUMN',
            num_threads=config.nipype.omp_nthreads,
        ),
        name='separate_subcortical',
        mem_gb=mem_gb['bold'],
        n_procs=config.nipype.omp_nthreads,
    )

    # Calculate the reho by hemisphere
    lh_reho = pe.Node(
        SurfaceReHo(surf_hemi='L'),
        name='reho_lh',
        mem_gb=2 * mem_gb['bold'],
    )
    rh_reho = pe.Node(
        SurfaceReHo(surf_hemi='R'),
        name='reho_rh',
        mem_gb=2 * mem_gb['bold'],
    )
    subcortical_reho = pe.Node(
        ReHoNamePatch(neighborhood='vertices'),
        name='reho_subcortical',
        mem_gb=mem_gb['bold'],
    )

    # Merge the surfaces and subcortical structures back into a CIFTI
    merge_cifti = pe.Node(
        CiftiCreateDenseFromTemplate(
            from_cropped=True,
            out_file='reho.dscalar.nii',
            num_threads=config.nipype.omp_nthreads,
        ),
        name='merge_cifti',
        mem_gb=mem_gb['bold'],
        n_procs=config.nipype.omp_nthreads,
    )
    reho_plot = pe.Node(
        PlotDenseCifti(base_desc='reho'),
        name='reho_cifti_plot',
    )
    workflow.connect([
        (inputnode, reho_plot, [
            ('lh_midthickness', 'lh_underlay'),
            ('rh_midthickness', 'rh_underlay'),
        ]),
    ])  # fmt:skip

    ds_report_reho = pe.Node(
        DerivativesDataSink(
            dismiss_entities=dismiss_hash(),
            source_file=name_source,
        ),
        name='ds_report_reho',
        run_without_submitting=True,
    )

    # Write out results
    workflow.connect([
        (inputnode, lh_surf, [('denoised_bold', 'in_file')]),
        (inputnode, rh_surf, [('denoised_bold', 'in_file')]),
        (inputnode, subcortical_nifti, [('denoised_bold', 'in_file')]),
        (lh_surf, lh_reho, [('out_file', 'surf_bold')]),
        (rh_surf, rh_reho, [('out_file', 'surf_bold')]),
        (subcortical_nifti, subcortical_reho, [('out_file', 'in_file')]),
        (inputnode, merge_cifti, [('denoised_bold', 'template_cifti')]),
        (lh_reho, merge_cifti, [('surf_gii', 'left_metric')]),
        (rh_reho, merge_cifti, [('surf_gii', 'right_metric')]),
        (subcortical_reho, merge_cifti, [('out_file', 'volume_all')]),
        (merge_cifti, outputnode, [('out_file', 'reho')]),
        (merge_cifti, reho_plot, [('out_file', 'in_file')]),
        (reho_plot, ds_report_reho, [
            ('out_file', 'in_file'),
            ('desc', 'desc'),
        ]),
    ])  # fmt:skip

    return workflow


@fill_doc
def init_reho_nifti_wf(name_source, mem_gb, name='reho_nifti_wf'):
    """Compute ReHo on volumetric (NIFTI) data.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.tests.tests import mock_config
            from xcp_d import config
            from xcp_d.workflows.bold.metrics import init_reho_nifti_wf

            with mock_config():
                wf = init_reho_nifti_wf(
                    name_source="/path/to/bold.nii.gz",
                    mem_gb={"bold": 0.1},
                    name="nifti_reho_wf",
                )

    Parameters
    ----------
    name_source
    mem_gb : :obj:`dict`
        Memory allocation dictionary
    %(name)s
        Default is "nifti_reho_wf".

    Inputs
    ------
    denoised_bold
       residual and filtered, nifti
    bold_mask
       bold mask
    name_source

    Outputs
    -------
    reho
        reho output
    """
    workflow = Workflow(name=name)

    workflow.__desc__ = """
Regional homogeneity (ReHo) [@jiang2016regional] was computed with neighborhood voxels using
*AFNI*'s *3dReHo* [@taylor2013fatcat].
"""

    inputnode = pe.Node(
        niu.IdentityInterface(fields=['denoised_bold', 'bold_mask']),
        name='inputnode',
    )
    outputnode = pe.Node(niu.IdentityInterface(fields=['reho']), name='outputnode')

    # Run AFNI'S 3DReHo on the data
    compute_reho = pe.Node(
        ReHoNamePatch(neighborhood='vertices'),
        name='reho_3d',
        mem_gb=2 * mem_gb['bold'],
        n_procs=1,
    )
    # Get the svg
    reho_plot = pe.Node(
        PlotNifti(name_source=name_source),
        name='reho_nifti_plot',
    )

    ds_report_reho = pe.Node(
        DerivativesDataSink(
            dismiss_entities=dismiss_hash(),
            source_file=name_source,
            desc='rehoVolumetricPlot',
        ),
        name='ds_report_reho',
        run_without_submitting=True,
    )

    # Write the results out
    workflow.connect([
        (inputnode, compute_reho, [
            ('denoised_bold', 'in_file'),
            ('bold_mask', 'mask_file'),
        ]),
        (compute_reho, outputnode, [('out_file', 'reho')]),
        (compute_reho, reho_plot, [('out_file', 'in_file')]),
        (reho_plot, ds_report_reho, [('out_file', 'in_file')]),
    ])  # fmt:skip

    return workflow


@fill_doc
def init_peraf_wf(
    name_source,
    mem_gb,
    name='peraf_wf',
):
    """Compute PerAF for both NIfTI and CIFTI.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.tests.tests import mock_config
            from xcp_d import config
            from xcp_d.workflows.bold.metrics import init_peraf_wf

            with mock_config():
                wf = init_peraf_wf(
                    name_source="/path/to/file.nii.gz",
                    mem_gb={"bold": 0.1},
                    name="peraf_wf",
                )

    Parameters
    ----------
    name_source
    mem_gb : :obj:`dict`
        Memory allocation dictionary
    %(name)s
        Default is "peraf_wf".

    Inputs
    ------
    denoised_bold
        The ``filtered, interpolated, denoised BOLD``,
        although interpolation is not necessary if the data were not originally censored.
    preprocessed_bold
        The ``preprocessed BOLD``.
        It is censored (if fd_thresh > 0) and used to calculate the mean image.
    bold_mask
        Bold mask if bold is NIfTI.
    temporal_mask
        Temporal mask.
    lh_midthickness
        Left hemisphere midthickness surface.
    rh_midthickness
        Right hemisphere midthickness surface.
        Only used for CIFTI data if the anatomical workflow is enabled.

    Outputs
    -------
    peraf
        PerAF output

    Notes
    -----
    The PerAF implementation is based on :footcite:t:`jia2020percent`.
    PerAF is calculated after denoising, bandpass filtering, and temporal masking.

    The denoised BOLD data is already mean-centered by the denoising workflow,
    but the original standard deviation is retained.
    This workflow will explicitly mean-center the denoised data, just to be safe.
    The preprocessed BOLD data is used to calculate the mean image.

    This workflow will also generate a plot of the PerAF map.
    For CIFTI data, the plot will be overlaid on the midthickness surface-
    either the subject's surface warped to fsLR space (when the anatomical workflow is enabled)
    or the fsLR 32k midthickness surface template.

    .. math::
        PerAF = \frac{1}{N} \sum_{i=1}^{N} \left( \frac{X_i - \mu}{\mu} \right) * 100%

    References
    ----------
    .. footbibliography::
    """
    workflow = Workflow(name=name)

    smoothing = config.workflow.smoothing
    file_format = config.workflow.file_format

    workflow.__desc__ = """ \

The percent amplitude of fluctuation (PerAF) [@jia2020percent] was calculated as the percent
change from the mean of the denoised BOLD time series.
"""

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'denoised_bold',  # already mean-centered by the denoising workflow
                'preprocessed_bold',  # censored and used to calculate the mean image
                'temporal_mask',
                # only used for NIfTI data
                'bold_mask',
                # only used for CIFTI data if the anatomical workflow is enabled
                'lh_midthickness',
                'rh_midthickness',
            ],
        ),
        name='inputnode',
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['peraf', 'smoothed_peraf']),
        name='outputnode',
    )

    # compute alff
    compute_peraf = pe.Node(
        ComputePerAF(),
        mem_gb=2 * mem_gb['bold'],
        name='compute_peraf',
        n_procs=config.nipype.omp_nthreads,
    )
    workflow.connect([
        (inputnode, compute_peraf, [
            ('denoised_bold', 'in_file'),
            ('preprocessed_bold', 'mean_file'),
            ('bold_mask', 'mask'),
            ('temporal_mask', 'temporal_mask'),
        ]),
        (compute_peraf, outputnode, [('peraf', 'peraf')])
    ])  # fmt:skip

    # Plot the ALFF map
    ds_report_peraf = pe.Node(
        DerivativesDataSink(
            dismiss_entities=dismiss_hash(),
            source_file=name_source,
        ),
        name='ds_report_peraf',
        run_without_submitting=True,
    )

    if file_format == 'cifti':
        peraf_plot = pe.Node(
            PlotDenseCifti(base_desc='peraf'),
            name='peraf_plot',
        )
        workflow.connect([
            (inputnode, peraf_plot, [
                ('lh_midthickness', 'lh_underlay'),
                ('rh_midthickness', 'rh_underlay'),
            ]),
            (peraf_plot, ds_report_peraf, [('desc', 'desc')]),
        ])  # fmt:skip
    else:
        peraf_plot = pe.Node(
            PlotNifti(name_source=name_source),
            name='peraf_plot',
        )
        ds_report_peraf.inputs.desc = 'perafVolumetricPlot'

    workflow.connect([
        (compute_peraf, peraf_plot, [('peraf', 'in_file')]),
        (peraf_plot, ds_report_peraf, [('out_file', 'in_file')]),
    ])  # fmt:skip

    if smoothing:  # If we want to smooth
        if file_format == 'nifti':
            workflow.__desc__ = workflow.__desc__ + (
                ' The PerAF maps were smoothed with Nilearn using a Gaussian kernel '
                f'(FWHM={str(smoothing)} mm).'
            )
            # Smooth via Nilearn
            smooth_peraf = pe.Node(
                Smooth(fwhm=smoothing),
                name='smooth_peraf',
            )
            workflow.connect([
                (compute_peraf, smooth_peraf, [('peraf', 'in_file')]),
                (smooth_peraf, outputnode, [('out_file', 'smoothed_peraf')])
            ])  # fmt:skip

        else:  # If cifti
            workflow.__desc__ = workflow.__desc__ + (
                ' The PerAF maps were smoothed with the Connectome Workbench using a Gaussian '
                f'kernel (FWHM={str(smoothing)} mm).'
            )

            # Smooth via Connectome Workbench
            sigma_lx = fwhm2sigma(smoothing)  # Convert fwhm to standard deviation
            # Get templates for each hemisphere
            lh_midthickness = str(
                get_template('fsLR', hemi='L', suffix='sphere', density='32k', raise_empty=True)[0]
            )
            rh_midthickness = str(
                get_template('fsLR', hemi='R', suffix='sphere', density='32k', raise_empty=True)[0]
            )
            smooth_peraf = pe.Node(
                CiftiSmooth(
                    sigma_surf=sigma_lx,
                    sigma_vol=sigma_lx,
                    direction='COLUMN',
                    right_surf=rh_midthickness,
                    left_surf=lh_midthickness,
                    num_threads=config.nipype.omp_nthreads,
                ),
                name='smooth_peraf',
                mem_gb=mem_gb['bold'],
                n_procs=config.nipype.omp_nthreads,
            )

            # Always check the intent code in CiftiSmooth's output file
            fix_cifti_intent = pe.Node(
                FixCiftiIntent(),
                name='fix_cifti_intent',
                mem_gb=mem_gb['bold'],
            )
            workflow.connect([
                (compute_peraf, smooth_peraf, [('peraf', 'in_file')]),
                (smooth_peraf, fix_cifti_intent, [('out_file', 'in_file')]),
                (fix_cifti_intent, outputnode, [('out_file', 'smoothed_peraf')]),
            ])  # fmt:skip

    return workflow
