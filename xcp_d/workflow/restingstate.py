# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Workflows for calculating resting state-specific metrics."""
from nipype.interfaces import utility as niu
from nipype.interfaces.workbench import CiftiSmooth
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from templateflow.api import get as get_template
from xcp_d.interfaces.nilearn import Smooth
from xcp_d.interfaces.resting_state import (
    ComputeALFF,
    ReHoNamePatch,
    SurfaceReHo,
)
from xcp_d.interfaces.workbench import (
    CiftiCreateDenseScalar,
    CiftiSeparateMetric,
    CiftiSeparateVolumeAll,
)

from xcp_d.utils.doc import fill_doc
from xcp_d.utils.utils import fwhm2sigma

from xcp_d.utils.plot import plot_alff_reho_surface, plot_alff_reho_volumetric


@fill_doc
def init_compute_alff_wf(
    mem_gb,
    TR,
    lowpass,
    highpass,
    smoothing,
    cifti,
    omp_nthreads,
    name="compute_alff_wf",
):
    """Compute alff for both nifti and cifti.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.workflow.restingstate import init_compute_alff_wf
            wf = init_compute_alff_wf(
                mem_gb=0.1,
                TR=2.,
                lowpass=6.,
                highpass=60.,
                smoothing=6,
                cifti=False,
                omp_nthreads=1,
                name="compute_alff_wf",
            )

    Parameters
    ----------
    %(mem_gb)s
    TR : float
        repetition time
    lowpass : float
        low pass filter
    highpass : float
        high pass filter
    %(smoothing)s
    %(cifti)s
    %(omp_nthreads)s
    %(name)s
        Default is "compute_alff_wf".

    Inputs
    ------
    clean_bold
       residual and filtered
    bold_mask
       bold mask if bold is nifti

    Outputs
    -------
    alff_out
        alff output
    smoothed_alff
        smoothed alff  output
    html
        alff html for nifti
    """
    workflow = Workflow(name=name)

    workflow.__desc__ = f""" \
The amplitude of low-frequency fluctuation (ALFF) [@alff] was computed by transforming
the processed BOLD timeseries  to the frequency domain. The power spectrum was computed within
the {highpass}-{lowpass} Hz frequency band and the mean square root of the power spectrum was
calculated at each voxel to yield voxel-wise ALFF measures.
"""

    inputnode = pe.Node(
        niu.IdentityInterface(fields=['clean_bold', 'bold_mask']),
        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['alff_out', 'smoothed_alff', 'alffsvg']),
        name='outputnode')

    # compute alff
    alff_compt = pe.Node(ComputeALFF(TR=TR, lowpass=lowpass,
                                     highpass=highpass),
                         mem_gb=mem_gb,
                         name='alff_compt',
                         n_procs=omp_nthreads)

    if not cifti:
        alff_plot = pe.Node(plot_alff_reho_volumetric(output_path='alff.svg'),
                            mem_gb=mem_gb,
                            name='alff_plot_nifti',
                            n_procs=omp_nthreads)
    if cifti:
        alff_plot = pe.Node(plot_alff_reho_surface(output_path='alff.svg'),
                            mem_gb=mem_gb,
                            name='alff_plot_cifti',
                            n_procs=omp_nthreads)
    workflow.connect([(inputnode, alff_compt, [('clean_bold', 'in_file'),
                                               ('bold_mask', 'mask')]),
                      (alff_compt, alff_plot, [('alff_out', 'inputnode.filename')]),
                      (alff_plot, outputnode, [('outputnode.output_path', 'alffsvg')]),
                      (alff_compt, outputnode, [('alff_out', 'alff_out')])
                      ])

    if smoothing:  # If we want to smooth
        if not cifti:  # If nifti
            workflow.__desc__ = workflow.__desc__ + (
                " The ALFF maps were smoothed with Nilearn using a gaussian kernel size of "
                f"{str(smoothing)} mm (FWHM)."
            )
            # Smooth via Nilearn
            smooth_data = pe.Node(
                Smooth(fwhm=smoothing),
                name="niftismoothing",
                n_procs=omp_nthreads,
            )
            workflow.connect([
                (alff_compt, smooth_data, [('alff_out', 'in_file')]),
                (smooth_data, outputnode, [('out_file', 'smoothed_alff')])
            ])

        else:  # If cifti
            workflow.__desc__ = workflow.__desc__ + (
                " The ALFF maps were smoothed with the Connectome Workbench using a gaussian "
                f"kernel size of {str(smoothing)} mm (FWHM)."
            )

            # Smooth via Connectome Workbench
            sigma_lx = fwhm2sigma(smoothing)   # Convert fwhm to standard deviation
            # Get templates for each hemisphere
            lh_midthickness = str(
                get_template("fsLR", hemi='L', suffix='sphere',
                             density='32k')[0])
            rh_midthickness = str(
                get_template("fsLR", hemi='R', suffix='sphere',
                             density='32k')[0])
            smooth_data = pe.Node(CiftiSmooth(sigma_surf=sigma_lx,
                                              sigma_vol=sigma_lx,
                                              direction='COLUMN',
                                              right_surf=rh_midthickness,
                                              left_surf=lh_midthickness),
                                  name="ciftismoothing",
                                  mem_gb=mem_gb,
                                  n_procs=omp_nthreads)
            workflow.connect([
                (alff_compt, smooth_data, [('alff_out', 'in_file')]),
                (smooth_data, outputnode, [('out_file', 'smoothed_alff')]),
            ])

    return workflow


@fill_doc
def init_cifti_reho_wf(
    mem_gb,
    omp_nthreads,
    name="cifti_reho_wf",
):
    """Compute ReHo from surface+volumetric (CIFTI) data.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.workflow.restingstate import init_cifti_reho_wf
            wf = init_cifti_reho_wf(
                mem_gb=0.1,
                omp_nthreads=1,
                name="cifti_reho_wf",
            )

    Parameters
    ----------
    %(mem_gb)s
    %(omp_nthreads)s
    %(name)s
        Default is "cifti_reho_wf".

    Inputs
    ------
    clean_bold
       residual and filtered, cifti

    Outputs
    -------
    reho_out
        ReHo in a CIFTI file.
    """
    workflow = Workflow(name=name)
    workflow.__desc__ = """

For each hemisphere, regional homogeneity (ReHo) was computed using surface-based
*2dReHo* [@surface_reho].
Specifically, for each vertex on the surface, the Kendall's coefficient of concordance (KCC)
was computed with nearest-neighbor vertices to yield ReHo.
For the subcortical, volumetric data, ReHo was computed with neighborhood voxels using
*3dReHo* in AFNI [@afni].
"""
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['clean_bold']),
        name='inputnode',
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['reho_out', 'rehosvg']),
        name='outputnode',
    )

    # Extract left and right hemispheres via Connectome Workbench
    lh_surf = pe.Node(
        CiftiSeparateMetric(metric='CORTEX_LEFT', direction="COLUMN"),
        name="separate_lh",
        mem_gb=mem_gb,
        n_procs=omp_nthreads,
    )
    rh_surf = pe.Node(
        CiftiSeparateMetric(metric='CORTEX_RIGHT', direction="COLUMN"),
        name="separate_rh",
        mem_gb=mem_gb,
        n_procs=omp_nthreads,
    )
    subcortical_nifti = pe.Node(
        CiftiSeparateVolumeAll(direction="COLUMN"),
        name="separate_subcortical",
        mem_gb=mem_gb,
        n_procs=omp_nthreads,
    )

    # Calculate the reho by hemipshere
    lh_reho = pe.Node(
        SurfaceReHo(surf_hemi='L'),
        name="reho_lh",
        mem_gb=mem_gb,
        n_procs=omp_nthreads,
    )
    rh_reho = pe.Node(
        SurfaceReHo(surf_hemi='R'),
        name="reho_rh",
        mem_gb=mem_gb,
        n_procs=omp_nthreads,
    )
    subcortical_reho = pe.Node(
        ReHoNamePatch(neighborhood='vertices'),
        name="reho_subcortical",
        mem_gb=mem_gb,
        n_procs=omp_nthreads,
    )

    # Merge the surfaces and subcortical structures back into a CIFTI
    merge_cifti = pe.Node(
        CiftiCreateDenseScalar(),
        name="merge_cifti",
        mem_gb=mem_gb,
        n_procs=omp_nthreads,
    )

    reho_plot = pe.Node(plot_alff_reho_surface(output_path='reho.svg'),
                        mem_gb=mem_gb,
                        name='reho_plot_cifti',
                        n_procs=omp_nthreads)
    # Write out results
    workflow.connect([
        (inputnode, lh_surf, [('clean_bold', 'in_file')]),
        (inputnode, rh_surf, [('clean_bold', 'in_file')]),
        (inputnode, subcortical_nifti, [('clean_bold', 'in_file')]),
        (lh_surf, lh_reho, [('out_file', 'surf_bold')]),
        (rh_surf, rh_reho, [('out_file', 'surf_bold')]),
        (subcortical_nifti, subcortical_reho, [('out_file', 'in_file')]),
        (subcortical_nifti, merge_cifti, [('label_file', 'structure_label_volume')]),
        (lh_reho, merge_cifti, [('surf_gii', 'left_metric')]),
        (rh_reho, merge_cifti, [('surf_gii', 'right_metric')]),
        (subcortical_reho, merge_cifti, [('out_file', 'volume_data')]),
        (merge_cifti, outputnode, [('out_file', 'reho_out')]),
        (merge_cifti, reho_plot, [('out_file', 'inputnode.filename')]),
        (reho_plot, outputnode, [('outputnode.out_file', 'rehosvg')])
    ])

    return workflow


@fill_doc
def init_nifti_reho_wf(
    mem_gb,
    omp_nthreads,
    name="nifti_reho_wf",
):
    """Compute ReHo on volumetric (NIFTI) data.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.workflow.restingstate import init_nifti_reho_wf
            wf = init_nifti_reho_wf(
                mem_gb=0.1,
                omp_nthreads=1,
                name="nifti_reho_wf",
            )

    Parameters
    ----------
    %(mem_gb)s
    %(omp_nthreads)s
    %(name)s
        Default is "nifti_reho_wf".

    Inputs
    ------
    clean_bold
       residual and filtered, nifti
    bold_mask
       bold mask

    Outputs
    -------
    reho_out
        reho output
    """
    workflow = Workflow(name=name)
    workflow.__desc__ = """
Regional homogeneity (ReHo) was computed with neighborhood voxels using *3dReHo* in AFNI [@afni].
"""

    inputnode = pe.Node(
        niu.IdentityInterface(fields=['clean_bold', 'bold_mask']),
        name='inputnode')
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['reho_out', 'rehosvg']),
        name='outputnode')

    # Run AFNI'S 3DReHo on the data
    compute_reho = pe.Node(ReHoNamePatch(neighborhood='vertices'),
                           name="reho_3d",
                           mem_gb=mem_gb,
                           n_procs=omp_nthreads)
    # Get the HTML
    reho_plot = pe.Node(plot_alff_reho_surface(output_path='reho.svg'),
                        mem_gb=mem_gb,
                        name='reho_plot_cifti',
                        n_procs=omp_nthreads)
    # Write the results out
    workflow.connect([(inputnode, compute_reho, [('clean_bold', 'in_file'),
                                                 ('bold_mask', 'mask_file')]),
                      (compute_reho, outputnode, [('out_file', 'reho_out')]),
                      (compute_reho, reho_plot, [('out_file', 'inputnode.filename')]),
                      (reho_plot, outputnode, [('outputnode.output_path', 'reho_out')]),
                      ])

    return workflow
