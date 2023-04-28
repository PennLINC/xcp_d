"""Plotting workflows."""
from nipype import Function
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from templateflow.api import get as get_template

from xcp_d.interfaces.ants import ApplyTransforms
from xcp_d.interfaces.bids import DerivativesDataSink
from xcp_d.interfaces.plotting import QCPlots, QCPlotsES
from xcp_d.interfaces.report import FunctionalSummary
from xcp_d.utils.doc import fill_doc
from xcp_d.utils.qcmetrics import _make_dcan_qc_file
from xcp_d.utils.utils import get_bold2std_and_t1w_xfms, get_std2bold_xfms


@fill_doc
def init_qc_report_wf(
    output_dir,
    TR,
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

            from xcp_d.workflows.plotting import init_qc_report_wf
            wf = init_qc_report_wf(
                output_dir=".",
                TR=0.5,
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
    %(TR)s
    %(head_radius)s
    %(mem_gb)s
    %(omp_nthreads)s
    %(cifti)s
    %(dcan_qc)s
    %(name)s
        Default is "qc_report_wf".

    Inputs
    ------
    %(name_source)s
    preprocessed_bold
        The preprocessed BOLD file, after dummy scan removal.
        Used for carpet plots.
    %(uncensored_denoised_bold)s
        Used for carpet plots.
        Only used if dcan_qc is True.
    %(interpolated_filtered_bold)s
        Used for DCAN carpet plots.
        Only used if dcan_qc is True.
    %(censored_denoised_bold)s
        Used for LINC carpet plots.
    %(boldref)s
        Only used with non-CIFTI data.
    bold_mask
        Only used with non-CIFTI data.
    anat_brainmask
        Only used with non-CIFTI data.
    %(template_to_anat_xfm)s
        Only used with non-CIFTI data.
    %(anat_to_native_xfm)s
        Only used with non-CIFTI data.
    %(dummy_scans)s
    %(fmriprep_confounds_file)s
    %(temporal_mask)s
    %(filtered_motion)s

    Outputs
    -------
    qc_file
    """
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "name_source",
                "preprocessed_bold",
                "uncensored_denoised_bold",
                "interpolated_filtered_bold",
                "censored_denoised_bold",
                "dummy_scans",
                "fmriprep_confounds_file",
                "filtered_motion",
                "temporal_mask",
                "run_index",  # will only be set for concatenated data
                # nifti-only inputs
                "bold_mask",
                "anat_brainmask",
                "boldref",
                "template_to_anat_xfm",
                "anat_to_native_xfm",
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

    nlin2009casym_brain_mask = str(
        get_template(
            "MNI152NLin2009cAsym",
            resolution=2,
            desc="brain",
            suffix="mask",
            extension=[".nii", ".nii.gz"],
        )
    )

    if not cifti:
        # We need the BOLD mask in T1w and standard spaces for QC metric calculation.
        # This is only possible for nifti inputs.
        get_native2space_transforms = pe.Node(
            Function(
                input_names=["bold_file", "template_to_anat_xfm", "anat_to_native_xfm"],
                output_names=[
                    "bold_to_std_xfms",
                    "bold_to_std_xfms_invert",
                    "bold_to_t1w_xfms",
                    "bold_to_t1w_xfms_invert",
                ],
                function=get_bold2std_and_t1w_xfms,
            ),
            name="get_native2space_transforms",
        )

        # fmt:off
        workflow.connect([
            (inputnode, get_native2space_transforms, [
                ("name_source", "bold_file"),
                ("template_to_anat_xfm", "template_to_anat_xfm"),
                ("anat_to_native_xfm", "anat_to_native_xfm"),
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
                ("anat_brainmask", "reference_image"),
            ]),
            (get_native2space_transforms, warp_boldmask_to_t1w, [
                ("bold_to_t1w_xfms", "transforms"),
                ("bold_to_t1w_xfms_invert", "invert_transform_flags"),
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
            (inputnode, warp_boldmask_to_mni, [("bold_mask", "input_image")]),
            (get_native2space_transforms, warp_boldmask_to_mni, [
                ("bold_to_std_xfms", "transforms"),
                ("bold_to_std_xfms_invert", "invert_transform_flags"),
            ]),
        ])
        # fmt:on

        # NIFTI files require a tissue-type segmentation in the same space as the BOLD data.
        # Get the set of transforms from MNI152NLin6Asym (the dseg) to the BOLD space.
        # Given that xcp-d doesn't process native-space data, this transform will never be used.
        get_mni_to_bold_xfms = pe.Node(
            Function(
                input_names=["bold_file", "template_to_anat_xfm", "anat_to_native_xfm"],
                output_names=["transform_list"],
                function=get_std2bold_xfms,
            ),
            name="get_std2native_transform",
        )

        # fmt:off
        workflow.connect([
            (inputnode, get_mni_to_bold_xfms, [
                ("name_source", "bold_file"),
                ("template_to_anat_xfm", "template_to_anat_xfm"),
                ("anat_to_native_xfm", "anat_to_native_xfm"),
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
        add_xfm_to_nlin6asym = pe.Node(
            niu.Merge(2),
            name="add_xfm_to_nlin6asym",
        )
        add_xfm_to_nlin6asym.inputs.in2 = MNI152NLin2009cAsym_to_MNI152NLin6Asym

        # fmt:off
        workflow.connect([
            (get_mni_to_bold_xfms, add_xfm_to_nlin6asym, [("transform_list", "in1")]),
        ])
        # fmt:on

        # Transform MNI152NLin2009cAsym dseg file to the same space as the BOLD data.
        warp_dseg_to_bold = pe.Node(
            ApplyTransforms(
                dimension=3,
                input_image=dseg_file,
                interpolation="GenericLabel",
            ),
            name="warp_dseg_to_bold",
            n_procs=omp_nthreads,
            mem_gb=mem_gb * 3 * omp_nthreads,
        )

        # fmt:off
        workflow.connect([
            (inputnode, warp_dseg_to_bold, [("boldref", "reference_image")]),
            (add_xfm_to_nlin6asym, warp_dseg_to_bold, [("out", "transforms")]),
        ])
        # fmt:on

    qcreport = pe.Node(
        QCPlots(
            TR=TR,
            head_radius=head_radius,
            template_mask=nlin2009casym_brain_mask,
        ),
        name="qc_report",
        mem_gb=mem_gb,
        n_procs=omp_nthreads,
    )

    # fmt:off
    workflow.connect([
        (inputnode, qcreport, [
            ("name_source", "name_source"),
            ("preprocessed_bold", "bold_file"),
            ("censored_denoised_bold", "cleaned_file"),
            ("fmriprep_confounds_file", "fmriprep_confounds_file"),
            ("temporal_mask", "temporal_mask"),
            ("dummy_scans", "dummy_scans"),
        ]),
        (qcreport, outputnode, [("qc_file", "qc_file")]),
    ])
    # fmt:on

    if dcan_qc:
        make_dcan_qc_file = pe.Node(
            Function(
                input_names=["filtered_motion", "TR"],
                output_names=["dcan_df_file"],
                function=_make_dcan_qc_file,
            ),
            name="make_dcan_qc_file",
        )
        make_dcan_qc_file.inputs.TR = TR

        # fmt:off
        workflow.connect([
            (inputnode, make_dcan_qc_file, [("filtered_motion", "filtered_motion")]),
        ])
        # fmt:on

        ds_dcan_qc = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                datatype="func",
                desc="dcan",
                suffix="qc",
                extension="hdf5",
            ),
            name="ds_dcan_qc",
            run_without_submitting=True,
        )

        # fmt:off
        workflow.connect([
            (inputnode, ds_dcan_qc, [("name_source", "source_file")]),
            (make_dcan_qc_file, ds_dcan_qc, [("dcan_df_file", "in_file")]),
        ])
        # fmt:on

        # Generate preprocessing and postprocessing carpet plots.
        plot_execsummary_carpets_dcan = pe.Node(
            QCPlotsES(TR=TR, standardize=False),
            name="plot_execsummary_carpets_dcan",
            mem_gb=mem_gb,
            n_procs=omp_nthreads,
        )

        # fmt:off
        workflow.connect([
            (inputnode, plot_execsummary_carpets_dcan, [
                ("preprocessed_bold", "preprocessed_bold"),
                ("uncensored_denoised_bold", "uncensored_denoised_bold"),
                ("interpolated_filtered_bold", "interpolated_filtered_bold"),
                ("filtered_motion", "filtered_motion"),
                ("run_index", "run_index"),
            ]),
        ])
        # fmt:on

        plot_execsummary_carpets_linc = pe.Node(
            QCPlotsES(TR=TR, standardize=True),
            name="plot_execsummary_carpets_linc",
            mem_gb=mem_gb,
            n_procs=omp_nthreads,
        )

        # fmt:off
        workflow.connect([
            (inputnode, plot_execsummary_carpets_linc, [
                ("preprocessed_bold", "preprocessed_bold"),
                ("uncensored_denoised_bold", "uncensored_denoised_bold"),
                ("interpolated_filtered_bold", "interpolated_filtered_bold"),
                ("filtered_motion", "filtered_motion"),
                ("run_index", "run_index"),
            ]),
        ])
        # fmt:on

        if not cifti:
            # fmt:off
            workflow.connect([
                (inputnode, plot_execsummary_carpets_dcan, [("bold_mask", "mask")]),
                (warp_dseg_to_bold, plot_execsummary_carpets_dcan, [
                    ("output_image", "seg_data"),
                ]),
                (inputnode, plot_execsummary_carpets_linc, [("bold_mask", "mask")]),
                (warp_dseg_to_bold, plot_execsummary_carpets_linc, [
                    ("output_image", "seg_data"),
                ]),
            ])
            # fmt:on

        ds_preproc_execsummary_carpet_dcan = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                dismiss_entities=["den"],
                datatype="figures",
                desc="preprocESQC",
            ),
            name="ds_preproc_execsummary_carpet_dcan",
            run_without_submitting=True,
        )

        ds_postproc_execsummary_carpet_dcan = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                dismiss_entities=["den"],
                datatype="figures",
                desc="postprocESQC",
            ),
            name="ds_postproc_execsummary_carpet_dcan",
            run_without_submitting=True,
        )

        # fmt:off
        workflow.connect([
            (inputnode, ds_preproc_execsummary_carpet_dcan, [
                ("name_source", "source_file"),
            ]),
            (inputnode, ds_postproc_execsummary_carpet_dcan, [
                ("name_source", "source_file"),
            ]),
            (plot_execsummary_carpets_dcan, ds_preproc_execsummary_carpet_dcan, [
                ("before_process", "in_file"),
            ]),
            (plot_execsummary_carpets_dcan, ds_postproc_execsummary_carpet_dcan, [
                ("after_process", "in_file"),
            ]),
        ])
        # fmt:on

        ds_preproc_execsummary_carpet_linc = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                dismiss_entities=["den"],
                datatype="figures",
                desc="preprocESQCScaled",
            ),
            name="ds_preproc_execsummary_carpet_linc",
            run_without_submitting=True,
        )

        ds_postproc_execsummary_carpet_linc = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                dismiss_entities=["den"],
                datatype="figures",
                desc="postprocESQCScaled",
            ),
            name="ds_postproc_execsummary_carpet_linc",
            run_without_submitting=True,
        )

        # fmt:off
        workflow.connect([
            (inputnode, ds_preproc_execsummary_carpet_linc, [
                ("name_source", "source_file"),
            ]),
            (inputnode, ds_postproc_execsummary_carpet_linc, [
                ("name_source", "source_file"),
            ]),
            (plot_execsummary_carpets_linc, ds_preproc_execsummary_carpet_linc, [
                ("before_process", "in_file"),
            ]),
            (plot_execsummary_carpets_linc, ds_postproc_execsummary_carpet_linc, [
                ("after_process", "in_file"),
            ]),
        ])
        # fmt:on

    if not cifti:
        # fmt:off
        workflow.connect([
            (inputnode, qcreport, [
                ("anat_brainmask", "anat_brainmask"),
                ("bold_mask", "mask_file"),
            ]),
            (warp_dseg_to_bold, qcreport, [("output_image", "seg_file")]),
            (warp_boldmask_to_t1w, qcreport, [("output_image", "bold2T1w_mask")]),
            (warp_boldmask_to_mni, qcreport, [("output_image", "bold2temp_mask")]),
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
        (inputnode, functional_qc, [("name_source", "bold_file")]),
        (qcreport, functional_qc, [("qc_file", "qc_file")]),
    ])
    # fmt:on

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
        (inputnode, ds_report_qualitycontrol, [("name_source", "source_file")]),
        (inputnode, ds_report_preprocessing, [("name_source", "source_file")]),
        (inputnode, ds_report_postprocessing, [("name_source", "source_file")]),
        (functional_qc, ds_report_qualitycontrol, [("out_report", "in_file")]),
        (qcreport, ds_report_preprocessing, [("raw_qcplot", "in_file")]),
        (qcreport, ds_report_postprocessing, [("clean_qcplot", "in_file")]),
    ])
    # fmt:on

    return workflow
