"""Plotting workflows."""

from nipype import Function
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from templateflow.api import get as get_template

from xcp_d import config
from xcp_d.interfaces.ants import ApplyTransforms
from xcp_d.interfaces.bids import DerivativesDataSink
from xcp_d.interfaces.plotting import QCPlots, QCPlotsES
from xcp_d.interfaces.report import FunctionalSummary
from xcp_d.interfaces.utils import ABCCQC, LINCQC
from xcp_d.utils.doc import fill_doc
from xcp_d.utils.utils import get_bold2std_and_t1w_xfms, get_std2bold_xfms


@fill_doc
def init_qc_report_wf(
    TR,
    head_radius,
    name="qc_report_wf",
):
    """Generate quality control figures and a QC file.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.tests.tests import mock_config
            from xcp_d import config
            from xcp_d.workflows.plotting import init_qc_report_wf

            with mock_config():
                wf = init_qc_report_wf(
                    TR=0.5,
                    head_radius=50,
                    name="qc_report_wf",
                )

    Parameters
    ----------
    %(TR)s
    %(head_radius)s
    %(name)s
        Default is "qc_report_wf".

    Inputs
    ------
    %(name_source)s
    preprocessed_bold
        The preprocessed BOLD file, after dummy scan removal.
        Used for carpet plots.
    %(denoised_interpolated_bold)s
        Used for DCAN carpet plots.
        Only used if abcc_qc is True.
    %(censored_denoised_bold)s
        Used for LINC carpet plots.
    %(boldref)s
        Only used with non-CIFTI data.
    bold_mask
        Path to the BOLD run's brain mask in the same space as ``preprocessed_bold``.
        Only used with non-CIFTI data.
    anat_brainmask
        Path to the anatomical brain mask in the same standard space as ``bold_mask``.
        Only used with non-CIFTI data.
    %(template_to_anat_xfm)s
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

    output_dir = config.execution.xcp_d_dir
    omp_nthreads = config.nipype.omp_nthreads

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "name_source",
                "preprocessed_bold",
                "denoised_interpolated_bold",
                "censored_denoised_bold",
                "dummy_scans",
                "fmriprep_confounds_file",
                "filtered_motion",
                "temporal_mask",
                "run_index",  # will only be set for concatenated data
                # nifti-only inputs
                "bold_mask",
                "anat",  # T1w/T2w image in anatomical space
                "anat_brainmask",
                "boldref",
                "template_to_anat_xfm",
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

    if config.workflow.file_format == "nifti":
        # We need the BOLD mask in T1w and standard spaces for QC metric calculation.
        # This is only possible for nifti inputs.
        get_native2space_transforms = pe.Node(
            Function(
                input_names=["bold_file", "template_to_anat_xfm"],
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

        workflow.connect([
            (inputnode, get_native2space_transforms, [
                ("name_source", "bold_file"),
                ("template_to_anat_xfm", "template_to_anat_xfm"),
            ]),
        ])  # fmt:skip

        warp_boldmask_to_t1w = pe.Node(
            ApplyTransforms(
                dimension=3,
                interpolation="NearestNeighbor",
            ),
            name="warp_boldmask_to_t1w",
            n_procs=omp_nthreads,
            mem_gb=1,
        )
        workflow.connect([
            (inputnode, warp_boldmask_to_t1w, [
                ("bold_mask", "input_image"),
                ("anat", "reference_image"),
            ]),
            (get_native2space_transforms, warp_boldmask_to_t1w, [
                ("bold_to_t1w_xfms", "transforms"),
                ("bold_to_t1w_xfms_invert", "invert_transform_flags"),
            ]),
        ])  # fmt:skip

        warp_boldmask_to_mni = pe.Node(
            ApplyTransforms(
                dimension=3,
                reference_image=nlin2009casym_brain_mask,
                interpolation="NearestNeighbor",
            ),
            name="warp_boldmask_to_mni",
            n_procs=omp_nthreads,
            mem_gb=1,
        )
        workflow.connect([
            (inputnode, warp_boldmask_to_mni, [("bold_mask", "input_image")]),
            (get_native2space_transforms, warp_boldmask_to_mni, [
                ("bold_to_std_xfms", "transforms"),
                ("bold_to_std_xfms_invert", "invert_transform_flags"),
            ]),
        ])  # fmt:skip

        # Warp the standard-space anatomical brain mask to the anatomical space
        warp_anatmask_to_t1w = pe.Node(
            ApplyTransforms(
                dimension=3,
                interpolation="NearestNeighbor",
            ),
            name="warp_anatmask_to_t1w",
            n_procs=omp_nthreads,
            mem_gb=1,
        )
        workflow.connect([
            (inputnode, warp_anatmask_to_t1w, [
                ("bold_mask", "input_image"),
                ("anat", "reference_image"),
            ]),
            (get_native2space_transforms, warp_anatmask_to_t1w, [
                ("bold_to_t1w_xfms", "transforms"),
                ("bold_to_t1w_xfms_invert", "invert_transform_flags"),
            ]),
        ])  # fmt:skip

        # NIFTI files require a tissue-type segmentation in the same space as the BOLD data.
        # Get the set of transforms from MNI152NLin6Asym (the dseg) to the BOLD space.
        # Given that xcp-d doesn't process native-space data, this transform will never be used.
        get_mni_to_bold_xfms = pe.Node(
            Function(
                input_names=["bold_file"],
                output_names=["transform_list"],
                function=get_std2bold_xfms,
            ),
            name="get_std2native_transform",
        )
        workflow.connect([(inputnode, get_mni_to_bold_xfms, [("name_source", "bold_file")])])

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

        workflow.connect([
            (get_mni_to_bold_xfms, add_xfm_to_nlin6asym, [("transform_list", "in1")]),
        ])  # fmt:skip

        # Transform MNI152NLin2009cAsym dseg file to the same space as the BOLD data.
        warp_dseg_to_bold = pe.Node(
            ApplyTransforms(
                dimension=3,
                input_image=dseg_file,
                interpolation="GenericLabel",
            ),
            name="warp_dseg_to_bold",
            n_procs=omp_nthreads,
            mem_gb=3,
        )
        workflow.connect([
            (inputnode, warp_dseg_to_bold, [("boldref", "reference_image")]),
            (add_xfm_to_nlin6asym, warp_dseg_to_bold, [("out", "transforms")]),
        ])  # fmt:skip

    if config.workflow.linc_qc:
        make_linc_qc = pe.Node(
            LINCQC(
                TR=TR,
                head_radius=head_radius,
                template_mask=nlin2009casym_brain_mask,
            ),
            name="make_linc_qc",
            mem_gb=2,
            n_procs=omp_nthreads,
        )
        workflow.connect([
            (inputnode, make_linc_qc, [
                ("name_source", "name_source"),
                ("preprocessed_bold", "bold_file"),
                ("censored_denoised_bold", "cleaned_file"),
                ("fmriprep_confounds_file", "fmriprep_confounds_file"),
                ("temporal_mask", "temporal_mask"),
                ("dummy_scans", "dummy_scans"),
            ]),
            (make_linc_qc, outputnode, [("qc_file", "qc_file")]),
        ])  # fmt:skip

        if config.workflow.file_format == "nifti":
            workflow.connect([
                (inputnode, make_linc_qc, [("bold_mask", "bold_mask_inputspace")]),
                (warp_boldmask_to_t1w, make_linc_qc, [("output_image", "bold_mask_anatspace")]),
                (warp_boldmask_to_mni, make_linc_qc, [("output_image", "bold_mask_stdspace")]),
                (warp_anatmask_to_t1w, make_linc_qc, [("output_image", "anat_mask_anatspace")]),
            ])  # fmt:skip
        else:
            make_linc_qc.inputs.bold_mask_inputspace = None

        ds_qc_metadata = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                dismiss_entities=list(DerivativesDataSink._allowed_entities),
                allowed_entities=["desc"],
                desc="linc",
                suffix="qc",
                extension=".json",
            ),
            name="ds_qc_metadata",
            run_without_submitting=True,
        )
        workflow.connect([
            (inputnode, ds_qc_metadata, [("name_source", "source_file")]),
            (make_linc_qc, ds_qc_metadata, [("qc_metadata", "in_file")]),
        ])  # fmt:skip

        make_qc_plots_nipreps = pe.Node(
            QCPlots(
                TR=TR,
                head_radius=head_radius,
                template_mask=nlin2009casym_brain_mask,
            ),
            name="make_qc_plots_nipreps",
            mem_gb=2,
            n_procs=omp_nthreads,
        )
        workflow.connect([
            (inputnode, make_qc_plots_nipreps, [
                ("name_source", "name_source"),
                ("preprocessed_bold", "bold_file"),
                ("censored_denoised_bold", "cleaned_file"),
                ("fmriprep_confounds_file", "fmriprep_confounds_file"),
                ("temporal_mask", "temporal_mask"),
                ("dummy_scans", "dummy_scans"),
            ]),
        ])  # fmt:skip

        if config.workflow.file_format == "nifti":
            workflow.connect([
                (inputnode, make_qc_plots_nipreps, [("bold_mask", "mask_file")]),
                (warp_dseg_to_bold, make_qc_plots_nipreps, [("output_image", "seg_file")]),
            ])  # fmt:skip
        else:
            make_qc_plots_nipreps.inputs.mask_file = None

        ds_preproc_qc_plot_nipreps = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                desc="preprocessing",
                datatype="figures",
            ),
            name="ds_preproc_qc_plot_nipreps",
            run_without_submitting=False,
        )
        workflow.connect([
            (inputnode, ds_preproc_qc_plot_nipreps, [("name_source", "source_file")]),
            (make_qc_plots_nipreps, ds_preproc_qc_plot_nipreps, [("raw_qcplot", "in_file")]),
        ])  # fmt:skip

        ds_postproc_qc_plot_nipreps = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                desc="postprocessing",
                datatype="figures",
            ),
            name="ds_postproc_qc_plot_nipreps",
            run_without_submitting=False,
        )
        workflow.connect([
            (inputnode, ds_postproc_qc_plot_nipreps, [("name_source", "source_file")]),
            (make_qc_plots_nipreps, ds_postproc_qc_plot_nipreps, [("clean_qcplot", "in_file")]),
        ])  # fmt:skip

        functional_qc = pe.Node(
            FunctionalSummary(TR=TR),
            name="qcsummary",
            run_without_submitting=False,
            mem_gb=2,
        )
        workflow.connect([
            (inputnode, functional_qc, [("name_source", "bold_file")]),
            (make_linc_qc, functional_qc, [("qc_file", "qc_file")]),
        ])  # fmt:skip

        ds_report_qualitycontrol = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                desc="qualitycontrol",
                datatype="figures",
            ),
            name="ds_report_qualitycontrol",
            run_without_submitting=False,
        )
        workflow.connect([
            (inputnode, ds_report_qualitycontrol, [("name_source", "source_file")]),
            (functional_qc, ds_report_qualitycontrol, [("out_report", "in_file")]),
        ])  # fmt:skip
    else:
        # Need to explicitly add the outputnode to the workflow, since it's not set otherwise.
        workflow.add_nodes([outputnode])

    if config.workflow.abcc_qc:
        make_abcc_qc = pe.Node(
            ABCCQC(TR=TR),
            name="make_abcc_qc",
            mem_gb=2,
            n_procs=omp_nthreads,
        )
        workflow.connect([(inputnode, make_abcc_qc, [("filtered_motion", "filtered_motion")])])

        ds_abcc_qc = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                datatype="func",
                desc="abcc",
                suffix="qc",
                extension="hdf5",
            ),
            name="ds_abcc_qc",
            run_without_submitting=True,
        )
        workflow.connect([
            (inputnode, ds_abcc_qc, [("name_source", "source_file")]),
            (make_abcc_qc, ds_abcc_qc, [("qc_file", "in_file")]),
        ])  # fmt:skip

        # Generate preprocessing and postprocessing carpet plots.
        make_qc_plots_es = pe.Node(
            QCPlotsES(TR=TR, standardize=config.workflow.params == "none"),
            name="make_qc_plots_es",
            mem_gb=2,
            n_procs=omp_nthreads,
        )
        workflow.connect([
            (inputnode, make_qc_plots_es, [
                ("preprocessed_bold", "preprocessed_bold"),
                ("denoised_interpolated_bold", "denoised_interpolated_bold"),
                ("filtered_motion", "filtered_motion"),
                ("temporal_mask", "temporal_mask"),
                ("run_index", "run_index"),
            ]),
        ])  # fmt:skip

        if config.workflow.file_format == "nifti":
            workflow.connect([
                (inputnode, make_qc_plots_es, [("bold_mask", "mask")]),
                (warp_dseg_to_bold, make_qc_plots_es, [("output_image", "seg_data")]),
            ])  # fmt:skip

        ds_preproc_qc_plot_es = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                dismiss_entities=["den"],
                datatype="figures",
                desc="preprocESQC",
            ),
            name="ds_preproc_qc_plot_es",
            run_without_submitting=True,
        )
        workflow.connect([
            (inputnode, ds_preproc_qc_plot_es, [("name_source", "source_file")]),
            (make_qc_plots_es, ds_preproc_qc_plot_es, [("before_process", "in_file")]),
        ])  # fmt:skip

        ds_postproc_qc_plot_es = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                dismiss_entities=["den"],
                datatype="figures",
                desc="postprocESQC",
            ),
            name="ds_postproc_qc_plot_es",
            run_without_submitting=True,
        )
        workflow.connect([
            (inputnode, ds_postproc_qc_plot_es, [("name_source", "source_file")]),
            (make_qc_plots_es, ds_postproc_qc_plot_es, [("after_process", "in_file")]),
        ])  # fmt:skip

    return workflow
