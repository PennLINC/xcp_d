# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Workflows for calculating resting state-specific metrics."""
from nipype import Function
from nipype.interfaces import utility as niu
from nipype.interfaces.workbench.cifti import CiftiSmooth
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from templateflow.api import get as get_template

from xcp_d.interfaces.bids import DerivativesDataSink
from xcp_d.interfaces.nilearn import Smooth
from xcp_d.interfaces.restingstate import ComputeALFF, ReHoNamePatch, SurfaceReHo
from xcp_d.interfaces.workbench import (
    CiftiCreateDenseScalar,
    CiftiSeparateMetric,
    CiftiSeparateVolumeAll,
    FixCiftiIntent,
)
from xcp_d.utils.doc import fill_doc
from xcp_d.utils.plotting import plot_alff_reho_surface, plot_alff_reho_volumetric
from xcp_d.utils.utils import fwhm2sigma


@fill_doc
def init_alff_wf(
    name_source,
    output_dir,
    TR,
    low_pass,
    high_pass,
    smoothing,
    cifti,
    mem_gb,
    omp_nthreads,
    name="alff_wf",
):
    """Compute alff for both nifti and cifti.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.workflows.restingstate import init_alff_wf
            wf = init_alff_wf(
                name_source="/path/to/file.nii.gz",
                output_dir=".",
                TR=2.,
                low_pass=0.1,
                high_pass=0.01,
                smoothing=6,
                cifti=False,
                mem_gb=0.1,
                omp_nthreads=1,
                name="alff_wf",
            )

    Parameters
    ----------
    name_source
    %(output_dir)s
    %(TR)s
    %(low_pass)s
    %(high_pass)s
    %(smoothing)s
    %(cifti)s
    %(mem_gb)s
    %(omp_nthreads)s
    %(name)s
        Default is "compute_alff_wf".

    Inputs
    ------
    denoised_bold
       residual and filtered
    bold_mask
       bold mask if bold is nifti
    name_source

    Outputs
    -------
    alff
        alff output
    smoothed_alff
        smoothed alff  output
    """
    workflow = Workflow(name=name)

    workflow.__desc__ = f""" \
The amplitude of low-frequency fluctuation (ALFF) [@alff] was computed by transforming
the processed BOLD timeseries to the frequency domain. The power spectrum was computed within
the {high_pass}-{low_pass} Hz frequency band and the mean square root of the power spectrum was
calculated at each voxel to yield voxel-wise ALFF measures.
"""

    inputnode = pe.Node(
        niu.IdentityInterface(fields=["denoised_bold", "bold_mask"]),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=["alff", "smoothed_alff", "alffplot"]),
        name="outputnode",
    )

    # compute alff
    alff_compt = pe.Node(
        ComputeALFF(TR=TR, low_pass=low_pass, high_pass=high_pass),
        mem_gb=mem_gb,
        name="alff_compt",
        n_procs=omp_nthreads,
    )

    alff_plot = pe.Node(
        Function(
            input_names=["output_path", "filename", "name_source"],
            output_names=["output_path"],
            function=plot_alff_reho_surface if cifti else plot_alff_reho_volumetric,
        ),
        name="alff_plot",
    )
    alff_plot.inputs.output_path = "alff.svg"
    alff_plot.inputs.name_source = name_source

    # fmt:off
    workflow.connect([
        (inputnode, alff_compt, [
            ("denoised_bold", "in_file"),
            ("bold_mask", "mask"),
        ]),
        (alff_compt, alff_plot, [("alff", "filename")]),
        (alff_compt, outputnode, [("alff", "alff")])
    ])
    # fmt:on

    if smoothing:  # If we want to smooth
        if not cifti:  # If nifti
            workflow.__desc__ = workflow.__desc__ + (
                " The ALFF maps were smoothed with Nilearn using a Gaussian kernel "
                f"(FWHM={str(smoothing)} mm)."
            )
            # Smooth via Nilearn
            smooth_data = pe.Node(
                Smooth(fwhm=smoothing),
                name="niftismoothing",
                n_procs=omp_nthreads,
            )
            # fmt:off
            workflow.connect([
                (alff_compt, smooth_data, [("alff", "in_file")]),
                (smooth_data, outputnode, [("out_file", "smoothed_alff")])
            ])
            # fmt:on

        else:  # If cifti
            workflow.__desc__ = workflow.__desc__ + (
                " The ALFF maps were smoothed with the Connectome Workbench using a Gaussian "
                f"kernel (FWHM={str(smoothing)} mm)."
            )

            # Smooth via Connectome Workbench
            sigma_lx = fwhm2sigma(smoothing)  # Convert fwhm to standard deviation
            # Get templates for each hemisphere
            lh_midthickness = str(
                get_template("fsLR", hemi="L", suffix="sphere", density="32k")[0]
            )
            rh_midthickness = str(
                get_template("fsLR", hemi="R", suffix="sphere", density="32k")[0]
            )
            smooth_data = pe.Node(
                CiftiSmooth(
                    sigma_surf=sigma_lx,
                    sigma_vol=sigma_lx,
                    direction="COLUMN",
                    right_surf=rh_midthickness,
                    left_surf=lh_midthickness,
                ),
                name="ciftismoothing",
                mem_gb=mem_gb,
                n_procs=omp_nthreads,
            )

            # Always check the intent code in CiftiSmooth's output file
            fix_cifti_intent = pe.Node(
                FixCiftiIntent(),
                name="fix_cifti_intent",
                mem_gb=mem_gb,
                n_procs=omp_nthreads,
            )

            # fmt:off
            workflow.connect([
                (alff_compt, smooth_data, [("alff", "in_file")]),
                (smooth_data, fix_cifti_intent, [("out_file", "in_file")]),
                (fix_cifti_intent, outputnode, [("out_file", "smoothed_alff")]),
            ])
            # fmt:on

    ds_alff_plot = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            source_file=name_source,
            desc="alffSurfacePlot" if cifti else "alffVolumetricPlot",
            datatype="figures",
        ),
        name="ds_alff_plot",
        run_without_submitting=False,
    )

    # fmt:off
    workflow.connect([(alff_plot, ds_alff_plot, [("output_path", "in_file")])])
    # fmt:on

    return workflow


@fill_doc
def init_reho_cifti_wf(
    name_source,
    output_dir,
    mem_gb,
    omp_nthreads,
    name="cifti_reho_wf",
):
    """Compute ReHo from surface+volumetric (CIFTI) data.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.workflows.restingstate import init_reho_cifti_wf

            wf = init_reho_cifti_wf(
                name_source="/path/to/bold.dtseries.nii",
                output_dir=".",
                mem_gb=0.1,
                omp_nthreads=1,
                name="cifti_reho_wf",
            )

    Parameters
    ----------
    name_source
    %(output_dir)s
    %(mem_gb)s
    %(omp_nthreads)s
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
    """
    workflow = Workflow(name=name)
    workflow.__desc__ = """

For each hemisphere, regional homogeneity (ReHo) was computed using surface-based
*2dReHo* [@surface_reho].
Specifically, for each vertex on the surface, the Kendall's coefficient of concordance (KCC)
was computed with nearest-neighbor vertices to yield ReHo.
For the subcortical, volumetric data, ReHo was computed with neighborhood voxels using AFNI's
*3dReHo* [@taylor2013fatcat].
"""
    inputnode = pe.Node(
        niu.IdentityInterface(fields=["denoised_bold"]),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=["reho"]),
        name="outputnode",
    )

    # Extract left and right hemispheres via Connectome Workbench
    lh_surf = pe.Node(
        CiftiSeparateMetric(metric="CORTEX_LEFT", direction="COLUMN"),
        name="separate_lh",
        mem_gb=mem_gb,
        n_procs=omp_nthreads,
    )
    rh_surf = pe.Node(
        CiftiSeparateMetric(metric="CORTEX_RIGHT", direction="COLUMN"),
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
        SurfaceReHo(surf_hemi="L"),
        name="reho_lh",
        mem_gb=mem_gb,
        n_procs=omp_nthreads,
    )
    rh_reho = pe.Node(
        SurfaceReHo(surf_hemi="R"),
        name="reho_rh",
        mem_gb=mem_gb,
        n_procs=omp_nthreads,
    )
    subcortical_reho = pe.Node(
        ReHoNamePatch(neighborhood="vertices"),
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
    reho_plot = pe.Node(
        Function(
            input_names=["output_path", "filename", "name_source"],
            output_names=["output_path"],
            function=plot_alff_reho_surface,
        ),
        name="reho_cifti_plot",
    )
    reho_plot.inputs.output_path = "reho.svg"
    reho_plot.inputs.name_source = name_source

    ds_reho_plot = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            source_file=name_source,
            desc="rehoSurfacePlot",
            datatype="figures",
        ),
        name="ds_reho_plot",
        run_without_submitting=False,
    )

    # Write out results
    # fmt:off
    workflow.connect([
        (inputnode, lh_surf, [("denoised_bold", "in_file")]),
        (inputnode, rh_surf, [("denoised_bold", "in_file")]),
        (inputnode, subcortical_nifti, [("denoised_bold", "in_file")]),
        (lh_surf, lh_reho, [("out_file", "surf_bold")]),
        (rh_surf, rh_reho, [("out_file", "surf_bold")]),
        (subcortical_nifti, subcortical_reho, [("out_file", "in_file")]),
        (subcortical_nifti, merge_cifti, [("label_file", "structure_label_volume")]),
        (lh_reho, merge_cifti, [("surf_gii", "left_metric")]),
        (rh_reho, merge_cifti, [("surf_gii", "right_metric")]),
        (subcortical_reho, merge_cifti, [("out_file", "volume_data")]),
        (merge_cifti, outputnode, [("out_file", "reho")]),
        (merge_cifti, reho_plot, [("out_file", "filename")]),
        (reho_plot, ds_reho_plot, [("output_path", "in_file")]),
    ])
    # fmt:on

    return workflow


@fill_doc
def init_reho_nifti_wf(
    name_source,
    output_dir,
    mem_gb,
    omp_nthreads,
    name="nifti_reho_wf",
):
    """Compute ReHo on volumetric (NIFTI) data.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.workflows.restingstate import init_reho_nifti_wf
            wf = init_reho_nifti_wf(
                name_source="/path/to/bold.nii.gz",
                output_dir=".",
                mem_gb=0.1,
                omp_nthreads=1,
                name="nifti_reho_wf",
            )

    Parameters
    ----------
    name_source
    %(output_dir)s
    %(mem_gb)s
    %(omp_nthreads)s
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
Regional homogeneity (ReHo) was computed with neighborhood voxels using AFNI's *3dReHo*
[@taylor2013fatcat].
"""

    inputnode = pe.Node(
        niu.IdentityInterface(fields=["denoised_bold", "bold_mask"]),
        name="inputnode",
    )
    outputnode = pe.Node(niu.IdentityInterface(fields=["reho"]), name="outputnode")

    # Run AFNI'S 3DReHo on the data
    compute_reho = pe.Node(
        ReHoNamePatch(neighborhood="vertices"),
        name="reho_3d",
        mem_gb=mem_gb,
        n_procs=omp_nthreads,
    )
    # Get the svg
    reho_plot = pe.Node(
        Function(
            input_names=["output_path", "filename", "name_source"],
            output_names=["output_path"],
            function=plot_alff_reho_volumetric,
        ),
        name="reho_nifti_plot",
    )
    reho_plot.inputs.output_path = "reho.svg"
    reho_plot.inputs.name_source = name_source

    ds_reho_plot = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            source_file=name_source,
            desc="rehoVolumetricPlot",
            datatype="figures",
        ),
        name="ds_reho_plot",
        run_without_submitting=False,
    )

    # Write the results out
    # fmt:off
    workflow.connect([
        (inputnode, compute_reho, [
            ("denoised_bold", "in_file"),
            ("bold_mask", "mask_file"),
        ]),
        (compute_reho, outputnode, [("out_file", "reho")]),
        (compute_reho, reho_plot, [("out_file", "filename")]),
        (reho_plot, ds_reho_plot, [("output_path", "in_file")]),
    ])
    # fmt:on

    return workflow
