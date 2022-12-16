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
from xcp_d.interfaces.surfplotting import PlotSVGData
from xcp_d.utils.doc import fill_doc
from xcp_d.utils.utils import get_bold2std_and_t1w_xforms, get_std2bold_xforms


@fill_doc
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
    cifti,
    dcan_qc,
    name="qc_report_wf",
):
    """Generate quality control figures and a QC file.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.workflow.plotting import init_qc_report_wf
            wf = init_qc_report_wf(
                output_dir=".",
                TR=0.5,
                motion_filter_type=None,
                band_stop_max=0,
                band_stop_min=0,
                motion_filter_order=1,
                fd_thresh=0.2,
                head_radius=50,
                mem_gb=0.1,
                omp_nthreads=1,
                cifti=False,
                dcan_qc=True,
                name="qc_report_wf",
            )

    Parameters
    ----------
    %(output_dir)s
    TR
    %(motion_filter_type)s
    %(band_stop_max)s
    %(band_stop_min)s
    %(motion_filter_order)s
    %(fd_thresh)s
    %(head_radius)s
    %(mem_gb)s
    %(omp_nthreads)s
    %(cifti)s
    dcan_qc : bool
        Whether to generate figures for the executive summary or not.
    %(name)s
        Default is "qc_report_wf".

    Inputs
    ------
    preprocessed_bold_file
        Used for naming outputs and finding related files.
    cleaned_unfiltered_file
    cleaned_file
    boldref
        Only used with non-CIFTI data.
    bold_mask
        Only used with non-CIFTI data.
    t1w_mask
        Only used with non-CIFTI data.
    %(template_to_t1w)s
        Only used with non-CIFTI data.
    t1w_to_native
        Only used with non-CIFTI data.
    %(dummy_scans)s
    tmask
    filtered_motion

    Outputs
    -------
    qc_file
    """
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "preprocessed_bold_file",
                "cleaned_file",
                "cleaned_unfiltered_file",
                "dummy_scans",
                "filtered_motion",
                "tmask",
                # nifti-only inputs
                "bold_mask",
                "t1w_mask",
                "boldref",
                "template_to_t1w",
                "t1w_to_native",
            ],
        ),
        name="inputnode",
    )

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "qc_file",
            ],
        ),
        name="outputnode",
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

    if not cifti:
        nlin2009casym_brain_mask = str(
            get_template(
                "MNI152NLin2009cAsym",
                resolution=2,
                desc="brain",
                suffix="mask",
                extension=[".nii", ".nii.gz"],
            )
        )

        # We need the BOLD mask in T1w and standard spaces for QC metric calculation.
        # This is only possible for nifti inputs.
        get_native2space_transforms = pe.Node(
            Function(
                input_names=["bold_file", "template_to_t1w", "t1w_to_native"],
                output_names=[
                    "bold_to_std_xforms",
                    "bold_to_std_xforms_invert",
                    "bold_to_t1w_xforms",
                    "bold_to_t1w_xforms_invert",
                ],
                function=get_bold2std_and_t1w_xforms,
            ),
            name="get_native2space_transforms",
        )

        # fmt:off
        workflow.connect([
            (inputnode, get_native2space_transforms, [
                ("preprocessed_bold_file", "bold_file"),
                ("template_to_t1w", "template_to_t1w"),
                ("t1w_to_native", "t1w_to_native"),
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
                ("bold_mask", "input_image"),
                ("t1w_mask", "reference_image"),
            ]),
            (get_native2space_transforms, warp_boldmask_to_t1w, [
                ("bold_to_t1w_xforms", "transforms"),
                ("bold_to_t1w_xforms_invert", "invert_transform_flags"),
            ]),
        ])
        # fmt:on

        warp_boldmask_to_mni = pe.Node(
            ApplyTransforms(
                dimension=3,
                reference_image=nlin2009casym_brain_mask,
                interpolation="NearestNeighbor",
            ),
            name="warp_boldmask_to_mni",
            n_procs=omp_nthreads,
            mem_gb=mem_gb,
        )

        # fmt:off
        workflow.connect([
            (inputnode, warp_boldmask_to_mni, [
                ("bold_mask", "input_image"),
            ]),
            (get_native2space_transforms, warp_boldmask_to_mni, [
                ("bold_to_std_xforms", "transforms"),
                ("bold_to_std_xforms_invert", "invert_transform_flags"),
            ]),
        ])
        # fmt:on

        # NIFTI files require a tissue-type segmentation in the same space as the BOLD data.
        # Get the set of transforms from MNI152NLin6Asym (the dseg) to the BOLD space.
        # Given that xcp-d doesn't process native-space data, this transform will never be used.
        get_mni_to_bold_xforms = pe.Node(
            Function(
                input_names=["bold_file", "template_to_t1w", "t1w_to_native"],
                output_names=["transform_list"],
                function=get_std2bold_xforms,
            ),
            name="get_std2native_transform",
        )

        # fmt:off
        workflow.connect([
            (inputnode, get_mni_to_bold_xforms, [
                ("preprocessed_bold_file", "bold_file"),
                ("template_to_t1w", "template_to_t1w"),
                ("t1w_to_native_xform", "t1w_to_native"),
            ]),
        ])
        # fmt:on

        # Use MNI152NLin2009cAsym tissue-type segmentation file for carpet plots.
        dseg_file = str(
            get_template(
                "MNI152NLin2009cAsym",
                resolution=1,
                desc="carpet",
                suffix="dseg",
                extension=[".nii", ".nii.gz"],
            )
        )

        # Get MNI152NLin2009cAsym --> MNI152NLin6Asym xform.
        MNI152NLin2009cAsym_to_MNI152NLin6Asym = str(
            get_template(
                template="MNI152NLin6Asym",
                mode="image",
                suffix="xfm",
                extension=".h5",
                **{"from": "MNI152NLin2009cAsym"},
            ),
        )

        # Add the MNI152NLin2009cAsym --> MNI152NLin6Asym xform to the end of the
        # BOLD --> MNI152NLin6Asym xform list, because xforms are applied in reverse order.
        add_xform_to_nlin6asym = pe.Node(
            niu.Merge(2),
            name="add_xform_to_nlin6asym",
        )
        add_xform_to_nlin6asym.inputs.in2 = MNI152NLin2009cAsym_to_MNI152NLin6Asym

        # fmt:off
        workflow.connect([
            (get_mni_to_bold_xforms, add_xform_to_nlin6asym, [("transform_list", "in1")]),
        ])
        # fmt:on

        # Transform MNI152NLin2009cAsym dseg file to the same space as the BOLD data.
        warp_dseg_to_bold = pe.Node(
            ApplyTransforms(
                dimension=3,
                input_image=dseg_file,
                interpolation="MultiLabel",
            ),
            name="warp_dseg_to_bold",
            n_procs=omp_nthreads,
            mem_gb=mem_gb * 3 * omp_nthreads,
        )

        # fmt:off
        workflow.connect([
            (inputnode, warp_dseg_to_bold, [("boldref", "reference_image")]),
            (add_xform_to_nlin6asym, warp_dseg_to_bold, [("out", "transforms")]),
        ])
        # fmt:on

    qcreport = pe.Node(
        QCPlot(
            TR=TR,
            template_mask=nlin2009casym_brain_mask,
            head_radius=head_radius,
        ),
        name="qc_report",
        mem_gb=mem_gb,
        n_procs=omp_nthreads,
    )

    # fmt:off
    workflow.connect([
        (inputnode, qcreport, [
            ("preprocessed_bold_file", "bold_file"),
            ("cleaned_file", "cleaned_file"),
            ("tmask", "tmask"),
            ("dummy_scans", "dummy_scans"),
        ]),
        (qcreport, outputnode, [
            ("qc_file", "qc_file"),
        ]),
    ])
    # fmt:on

    if dcan_qc:
        # Generate preprocessing and postprocessing carpet plots.
        plot_executive_summary_carpets = pe.Node(
            PlotSVGData(TR=TR),
            name="plot_carpets",
            mem_gb=mem_gb,
            n_procs=omp_nthreads,
        )

        # fmt:off
        workflow.connect([
            (inputnode, plot_executive_summary_carpets, [
                ("preprocessed_bold_file", "rawdata")
                ("cleaned_unfiltered_file", "regressed_data"),  # need to get
                ("cleaned_file", "residual_data"),
                ("filtered_motion", "filtered_motion"),  # need to get
                ("tmask", "tmask"),
                ("dummy_scans", "dummy_scans"),
            ]),
        ])
        # fmt:on

        if not cifti:
            # fmt:off
            workflow.connect([
                (inputnode, plot_executive_summary_carpets, [
                    ("mask", "mask"),
                ]),
                (warp_dseg_to_bold, plot_executive_summary_carpets, [
                    ("output_image", "seg_data"),
                ]),
            ])
            # fmt:on

        ds_preproc_executive_summary_carpet = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                dismiss_entities=["den"],
                datatype="figures",
                desc="precarpetplot",
            ),
            name="ds_preproc_executive_summary_carpet",
            run_without_submitting=True,
        )

        ds_postproc_executive_summary_carpet = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                dismiss_entities=["den"],
                datatype="figures",
                desc="postcarpetplot",
            ),
            name="ds_postproc_executive_summary_carpet",
            run_without_submitting=True,
        )

        # fmt:off
        workflow.connect([
            (inputnode, ds_preproc_executive_summary_carpet, [
                ("preprocessed_bold_file", "source_file"),
            ]),
            (inputnode, ds_postproc_executive_summary_carpet, [
                ("preprocessed_bold_file", "source_file"),
            ]),
            (plot_executive_summary_carpets, ds_preproc_executive_summary_carpet, [
                ("before_process", "in_file"),
            ]),
            (plot_executive_summary_carpets, ds_postproc_executive_summary_carpet, [
                ("after_process", "in_file"),
            ]),
        ])
        # fmt:on

    if not cifti:
        # fmt:off
        workflow.connect([
            (inputnode, qcreport, [
                ("t1w_mask", "t1w_mask"),
                ("bold_mask", "mask_file"),
            ]),
            (warp_dseg_to_bold, qcreport, [
                ("output_image", "seg_file"),
            ]),
            (warp_boldmask_to_t1w, qcreport, [
                ("output_image", "bold2T1w_mask"),
            ]),
            (warp_boldmask_to_mni, qcreport, [
                ("output_image", "bold2temp_mask"),
            ]),
        ])
        # fmt:on
    else:
        qcreport.inputs.mask_file = None

    functional_qc = pe.Node(
        FunctionalSummary(TR=TR),
        name="qcsummary",
        run_without_submitting=False,
        mem_gb=mem_gb,
    )

    # fmt:off
    workflow.connect([
        (inputnode, functional_qc, [("preprocessed_bold_file", "bold_file")]),
        (qcreport, functional_qc, [("qc_file", "qc_file")]),
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
        (inputnode, ds_report_censoring, [("preprocessed_bold_file", "source_file")]),
        (inputnode, ds_report_qualitycontrol, [("preprocessed_bold_file", "source_file")]),
        (inputnode, ds_report_preprocessing, [("preprocessed_bold_file", "source_file")]),
        (inputnode, ds_report_postprocessing, [("preprocessed_bold_file", "source_file")]),
        (censor_report, ds_report_censoring, [("out_file", "in_file")]),
        (functional_qc, ds_report_qualitycontrol, [("out_report", "in_file")]),
        (qcreport, ds_report_preprocessing, [("raw_qcplot", "in_file")]),
        (qcreport, ds_report_postprocessing, [("clean_qcplot", "in_file")]),
    ])
    # fmt:on

    return workflow
