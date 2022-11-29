# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Workflows for post-processing BOLD data."""
from nipype.interfaces import utility as niu
from nipype.interfaces.workbench import CiftiSmooth
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from pkg_resources import resource_filename as pkgrf

from xcp_d.interfaces.nilearn import Smooth
from xcp_d.utils.doc import fill_doc
from xcp_d.utils.utils import fwhm2sigma


@fill_doc
def init_resd_smoothing_wf(
    mem_gb,
    smoothing,
    omp_nthreads,
    cifti=False,
    name="resd_smoothing_wf",
):
    """Smooth BOLD residuals.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.workflow.postprocessing import init_resd_smoothing_wf
            wf = init_resd_smoothing_wf(
                mem_gb=0.1,
                smoothing=6,
                omp_nthreads=1,
                cifti=False,
                name="qc_report_wf",
            )

    Parameters
    ----------
    %(mem_gb)s
    smoothing
    %(omp_nthreads)s
    %(cifti)s
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
        (inputnode, smooth_data, [("bold_file", "in_file")]),
        (smooth_data, outputnode, [("out_file", "smoothed_bold")]),
    ])
    # fmt:on

    return workflow
