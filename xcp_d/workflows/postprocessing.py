# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Workflows for post-processing BOLD data."""
from nipype.interfaces import utility as niu
from nipype.interfaces.workbench.cifti import CiftiSmooth
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from pkg_resources import resource_filename as pkgrf

from xcp_d.interfaces.nilearn import Smooth
from xcp_d.interfaces.restingstate import DespikePatch
from xcp_d.interfaces.workbench import CiftiConvert, FixCiftiIntent
from xcp_d.utils.doc import fill_doc
from xcp_d.utils.utils import fwhm2sigma


@fill_doc
def init_despike_wf(
    TR,
    cifti,
    mem_gb,
    omp_nthreads,
    name="despike_wf",
):
    """Despike BOLD data with 3dDespike.

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

            from xcp_d.workflows.postprocessing import init_despike_wf

            wf = init_despike_wf(
                TR=0.8,
                cifti=True,
                mem_gb=0.1,
                omp_nthreads=1,
                name="despike_wf",
            )

    Parameters
    ----------
    %(TR)s
    %(cifti)s
    %(mem_gb)s
    %(omp_nthreads)s
    %(name)s
        Default is "despike_wf".

    Inputs
    ------
    bold_file : str
        A NIFTI or CIFTI BOLD file to despike.

    Outputs
    -------
    bold_file : str
        The despiked NIFTI or CIFTI BOLD file.
    """
    workflow = Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=["bold_file"]), name="inputnode")
    outputnode = pe.Node(niu.IdentityInterface(fields=["bold_file"]), name="outputnode")

    despike3d = pe.Node(
        DespikePatch(outputtype="NIFTI_GZ", args="-nomask -NEW"),
        name="despike3d",
        mem_gb=mem_gb,
        n_procs=omp_nthreads,
    )

    if cifti:
        workflow.__desc__ = (
            "The BOLD data were converted to NIfTI format, despiked with 3dDespike, "
            "and converted back to CIFTI format."
        )

        # first, convert the cifti to a nifti
        convert_to_nifti = pe.Node(
            CiftiConvert(target="to"),
            name="convert_to_nifti",
            mem_gb=mem_gb,
            n_procs=omp_nthreads,
        )

        # fmt:off
        workflow.connect([
            (inputnode, convert_to_nifti, [("bold_file", "in_file")]),
            (convert_to_nifti, despike3d, [("out_file", "in_file")]),
        ])
        # fmt:on

        # finally, convert the despiked nifti back to cifti
        convert_to_cifti = pe.Node(
            CiftiConvert(target="from", TR=TR),
            name="convert_to_cifti",
            mem_gb=mem_gb,
            n_procs=omp_nthreads,
        )

        # fmt:off
        workflow.connect([
            (inputnode, convert_to_cifti, [("bold_file", "cifti_template")]),
            (despike3d, convert_to_cifti, [("out_file", "in_file")]),
            (convert_to_cifti, outputnode, [("out_file", "bold_file")]),
        ])
        # fmt:on

    else:
        workflow.__desc__ = "The BOLD data were despiked with 3dDespike."

        # fmt:off
        workflow.connect([
            (inputnode, despike3d, [("bold_file", "in_file")]),
            (despike3d, outputnode, [("out_file", "bold_file")]),
        ])
        # fmt:on

    return workflow


@fill_doc
def init_resd_smoothing_wf(
    smoothing,
    cifti,
    mem_gb,
    omp_nthreads,
    name="resd_smoothing_wf",
):
    """Smooth BOLD residuals.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.workflows.postprocessing import init_resd_smoothing_wf

            wf = init_resd_smoothing_wf(
                smoothing=6,
                cifti=False,
                mem_gb=0.1,
                omp_nthreads=1,
                name="resd_smoothing_wf",
            )

    Parameters
    ----------
    smoothing
    %(cifti)s
    %(mem_gb)s
    %(omp_nthreads)s
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
    inputnode = pe.Node(niu.IdentityInterface(fields=["bold_file"]), name="inputnode")
    outputnode = pe.Node(niu.IdentityInterface(fields=["smoothed_bold"]), name="outputnode")

    # Turn specified FWHM (Full-Width at Half Maximum) to standard deviation.
    sigma_lx = fwhm2sigma(smoothing)
    if cifti:
        workflow.__desc__ = f""" \
The processed BOLD  was smoothed using Connectome Workbench with a gaussian kernel
size of {str(smoothing)} mm  (FWHM).
"""

        # Call connectome workbench to smooth for each hemisphere
        smooth_data = pe.Node(
            CiftiSmooth(
                sigma_surf=sigma_lx,  # the size of the surface kernel
                sigma_vol=sigma_lx,  # the volume of the surface kernel
                direction="COLUMN",  # which direction to smooth along@
                right_surf=pkgrf(  # pull out atlases for each hemisphere
                    "xcp_d",
                    (
                        "data/ciftiatlas/"
                        "Q1-Q6_RelatedParcellation210.R.midthickness_32k_fs_LR.surf.gii"
                    ),
                ),
                left_surf=pkgrf(
                    "xcp_d",
                    (
                        "data/ciftiatlas/"
                        "Q1-Q6_RelatedParcellation210.L.midthickness_32k_fs_LR.surf.gii"
                    ),
                ),
            ),
            name="cifti_smoothing",
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
            (smooth_data, fix_cifti_intent, [("out_file", "in_file")]),
            (fix_cifti_intent, outputnode, [("out_file", "smoothed_bold")]),
        ])
        # fmt:on

    else:
        workflow.__desc__ = f""" \
The processed BOLD was smoothed using Nilearn with a gaussian kernel size of {str(smoothing)} mm
(FWHM).
"""
        # Use nilearn to smooth the image
        smooth_data = pe.Node(
            Smooth(fwhm=smoothing),  # FWHM = kernel size
            name="nifti_smoothing",
            mem_gb=mem_gb,
            n_procs=omp_nthreads,
        )

        # fmt:off
        workflow.connect([
            (smooth_data, outputnode, [("out_file", "smoothed_bold")]),
        ])
        # fmt:on

    # fmt:off
    workflow.connect([
        (inputnode, smooth_data, [("bold_file", "in_file")]),
    ])
    # fmt:on

    return workflow
