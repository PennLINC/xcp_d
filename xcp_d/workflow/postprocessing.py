# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Workflows for post-processing BOLD data."""
from nipype.interfaces import utility as niu
from nipype.interfaces.workbench import CiftiSmooth
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from pkg_resources import resource_filename as pkgrf

from xcp_d.interfaces.nilearn import Smooth
from xcp_d.utils.utils import fwhm2sigma


def init_resd_smoothing(
    mem_gb,
    smoothing,
    omp_nthreads,
    cifti=False,
    name="smoothing",
):
    """Smooth BOLD residuals."""
    workflow = Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=["bold_file"]), name="inputnode")
    outputnode = pe.Node(niu.IdentityInterface(fields=["smoothed_bold"]), name="outputnode")

    sigma_lx = fwhm2sigma(smoothing)  # Turn specified FWHM (Full-Width at Half Maximum)
    # to standard deviation.
    if cifti:  # For ciftis
        workflow.__desc__ = f""" \
The processed BOLD  was smoothed using Connectome Workbench with a gaussian kernel
size of {str(smoothing)} mm  (FWHM).
"""

        smooth_data = pe.Node(
            CiftiSmooth(  # Call connectome workbench to smooth for each
                #  hemisphere
                sigma_surf=sigma_lx,  # the size of the surface kernel
                sigma_vol=sigma_lx,  # the volume of the surface kernel
                direction="COLUMN",  # which direction to smooth along@
                right_surf=pkgrf(  # pull out atlases for each hemisphere
                    "xcp_d",
                    "data/ciftiatlas/"
                    "Q1-Q6_RelatedParcellation210.R.midthickness_32k_fs_LR.surf.gii",
                ),
                left_surf=pkgrf(
                    "xcp_d",
                    "data/ciftiatlas/"
                    "Q1-Q6_RelatedParcellation210.L.midthickness_32k_fs_LR.surf.gii",
                ),
            ),
            name="cifti_smoothing",
            mem_gb=mem_gb,
            n_procs=omp_nthreads,
        )

        #  Connect to workflow
        # fmt:off
        workflow.connect([(inputnode, smooth_data, [('bold_file', 'in_file')]),
                          (smooth_data, outputnode, [('out_file',
                                                      'smoothed_bold')])])
        # fmt:on

    else:  # for Nifti
        workflow.__desc__ = f""" \
The processed BOLD was smoothed using Nilearn with a gaussian kernel size of {str(smoothing)} mm
(FWHM).
"""
        smooth_data = pe.Node(
            Smooth(fwhm=smoothing),  # FWHM = kernel size
            name="nifti_smoothing",
            mem_gb=mem_gb,
            n_procs=omp_nthreads,
        )  # Use nilearn to smooth the image

        #  Connect to workflow
        # fmt:off
        workflow.connect([(inputnode, smooth_data, [('bold_file', 'in_file')]),
                          (smooth_data, outputnode, [('out_file', 'smoothed_bold')])])
        # fmt:on

    return workflow
