"""Plotting workflows."""
from nipype import Function
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from niworkflows.interfaces.fixes import FixHeaderApplyTransforms as ApplyTransforms
from templateflow.api import get as get_template

from xcp_d.interfaces.bids import DerivativesDataSink
from xcp_d.interfaces.qc_plot import CensoringPlot, QCPlot
from xcp_d.interfaces.report import FunctionalSummary
from xcp_d.utils.utils import get_std2bold_xforms


def init_qc_report_wf(
    output_dir,
    TR,
    motion_filter_type,
    band_stop_max,
    band_stop_min,
    motion_filter_order,
    fd_thresh,
    head_radius,
    mem_gb,
    omp_nthreads,
    name="qc_report_wf",
):
    """Generate quality control figures and a QC file."""
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "preprocessed_bold_file",
                "mask_file",
                "t1w_mask",
                "bold_to_std_xforms",
                "bold_to_std_xforms_invert",
                "bold_to_t1w_xforms",
                "bold_to_t1w_xforms_invert",
                "dummy_scans",
                "cleaned_file",
                "tmask",
            ],
        ),
        name="inputnode",
    )

    censor_report = pe.Node(
        CensoringPlot(
            TR=TR,
            head_radius=head_radius,
            motion_filter_type=motion_filter_type,
            band_stop_max=band_stop_max,
            band_stop_min=band_stop_min,
            motion_filter_order=motion_filter_order,
            fd_thresh=fd_thresh,
        ),
        name="censor_report",
        mem_gb=mem_gb,
        n_procs=omp_nthreads,
    )

    # fmt:off
    workflow.connect([
        (inputnode, censor_report, [
            ("tmask", "tmask"),
            ("dummy_scans", "dummy_scans"),
            ("preprocessed_bold_file", "bold_file"),
        ]),
    ])
    # fmt:on

    warp_boldmask_to_t1w = pe.Node(
        ApplyTransforms(
            dimension=3,
            interpolation="NearestNeighbor",
        ),
        name="warp_boldmask_to_t1w",
        n_procs=omp_nthreads,
        mem_gb=mem_gb,
    )

    # fmt:off
    workflow.connect([
        (inputnode, warp_boldmask_to_t1w, [
            ("mask_file", "input_image"),
            ('t1w_mask', 'reference_image'),
            ('bold_to_t1w_xforms', 'transforms'),
            ("bold_to_t1w_xforms_invert", "invert_transform_flags"),
        ]),
    ])
    # fmt:on

    warp_boldmask_to_mni = pe.Node(
        ApplyTransforms(
            dimension=3,
            reference_image=str(
                get_template(
                    "MNI152NLin2009cAsym",
                    resolution=2,
                    desc="brain",
                    suffix="mask",
                    extension=[".nii", ".nii.gz"],
                ),
            ),
            interpolation="NearestNeighbor",
        ),
        name="warp_boldmask_to_mni",
        n_procs=omp_nthreads,
        mem_gb=mem_gb,
    )

    # fmt:off
    workflow.connect([
        (inputnode, warp_boldmask_to_mni, [
            ("mask_file", "input_image"),
            ('bold_to_std_xforms', 'transforms'),
            ("bold_to_std_xforms_invert", "invert_transform_flags"),
        ]),
    ])
    # fmt:on

    # Obtain transforms for QC report
    get_std2native_transform = pe.Node(
        Function(
            input_names=["bold_file", "mni_to_t1w", "t1w_to_native"],
            output_names=["transform_list"],
            function=get_std2bold_xforms,
        ),
        name="get_std2native_transform",
    )

    # fmt:off
    workflow.connect([
        (inputnode, get_std2native_transform, [
            ("bold_file", "bold_file"),
            ("mni_to_t1w", "mni_to_t1w"),
            ("t1w_to_native", "t1w_to_native"),
        ]),
    ])
    # fmt:on

    # Resample discrete segmentation for QCPlot into the appropriate space.
    resample_parc = pe.Node(
        ApplyTransforms(
            dimension=3,
            input_image=str(
                get_template(
                    "MNI152NLin2009cAsym",
                    resolution=1,
                    desc="carpet",
                    suffix="dseg",
                    extension=[".nii", ".nii.gz"],
                )
            ),
            interpolation="MultiLabel",
        ),
        name="resample_parc",
        n_procs=omp_nthreads,
        mem_gb=mem_gb,
    )

    # fmt:off
    workflow.connect([
        (inputnode, resample_parc, [('ref_file', 'reference_image')]),
        (get_std2native_transform, resample_parc, [('transform_list', 'transforms')]),
    ])
    # fmt:on

    qcreport = pe.Node(
        QCPlot(
            TR=TR,
            template_mask=str(
                get_template(
                    "MNI152NLin2009cAsym",
                    resolution=2,
                    desc="brain",
                    suffix="mask",
                    extension=[".nii", ".nii.gz"],
                )
            ),
            head_radius=head_radius,
        ),
        name="qc_report",
        mem_gb=mem_gb,
        n_procs=omp_nthreads,
    )

    # fmt:off
    workflow.connect([
        (inputnode, qcreport, [
            ("bold_file", "bold_file"),
            ("t1w_mask", "t1w_mask"),
            ('bold_mask', 'mask_file'),
        ]),
        (resample_parc, qcreport, [
            ('output_image', 'seg_file'),
        ]),
        (warp_boldmask_to_t1w, qcreport, [
            ('output_image', 'bold2T1w_mask'),
        ]),
        (warp_boldmask_to_mni, qcreport, [
            ('output_image', 'bold2temp_mask'),
        ]),
    ])
    # fmt:on

    functional_qc = pe.Node(
        FunctionalSummary(TR=TR),
        name="qcsummary",
        run_without_submitting=False,
        mem_gb=mem_gb,
    )

    # fmt:off
    workflow.connect([
        (inputnode, functional_qc, [("namesource", "bold_file")])
        (qcreport, functional_qc, [('qc_file', 'qc_file')]),
    ])
    # fmt:on

    ds_report_censoring = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            datatype="figures",
            desc="censoring",
            suffix="motion",
            extension=".svg",
        ),
        name="ds_report_censoring",
        run_without_submitting=False,
    )

    ds_report_qualitycontrol = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            desc="qualitycontrol",
            datatype="figures",
        ),
        name="ds_report_qualitycontrol",
        run_without_submitting=False,
    )

    ds_report_preprocessing = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            desc="preprocessing",
            datatype="figures",
        ),
        name="ds_report_preprocessing",
        run_without_submitting=False,
    )

    ds_report_postprocessing = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            desc="postprocessing",
            datatype="figures",
        ),
        name="ds_report_postprocessing",
        run_without_submitting=False,
    )

    # fmt:off
    workflow.connect([
        (inputnode, ds_report_censoring, [("namesource", "source_file")]),
        (inputnode, ds_report_qualitycontrol, [("namesource", "source_file")]),
        (inputnode, ds_report_preprocessing, [("namesource", "source_file")]),
        (inputnode, ds_report_postprocessing, [("namesource", "source_file")]),
        (censor_report, ds_report_censoring, [("out_file", "in_file")]),
        (functional_qc, ds_report_qualitycontrol, [('out_report', 'in_file')]),
        (qcreport, ds_report_preprocessing, [('raw_qcplot', 'in_file')]),
        (qcreport, ds_report_postprocessing, [('clean_qcplot', 'in_file')]),
    ])
    # fmt:on

    return workflow
