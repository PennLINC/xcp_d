# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Workflows for collecting and saving anatomical outputs."""

from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from xcp_d import config
from xcp_d.interfaces.bids import DerivativesDataSink
from xcp_d.interfaces.utils import FilterUndefined
from xcp_d.utils.doc import fill_doc


@fill_doc
def init_copy_inputs_to_outputs_wf(name="copy_inputs_to_outputs_wf"):
    """Copy files from the preprocessing derivatives to the output folder, with no modifications.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.tests.tests import mock_config
            from xcp_d import config
            from xcp_d.workflows.outputs import init_copy_inputs_to_outputs_wf

            with mock_config():
                wf = init_copy_inputs_to_outputs_wf()

    Parameters
    ----------
    %(name)s
        Default is "copy_inputs_to_outputs_wf".

    Inputs
    ------
    lh_pial_surf
    rh_pial_surf
    lh_wm_surf
    rh_wm_surf
    sulcal_depth
    sulcal_curv
    cortical_thickness
    cortical_thickness_corr
    myelin
    myelin_smoothed
    """
    workflow = Workflow(name=name)

    output_dir = config.execution.xcp_d_dir

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "lh_pial_surf",
                "rh_pial_surf",
                "lh_wm_surf",
                "rh_wm_surf",
                "sulcal_depth",
                "sulcal_curv",
                "cortical_thickness",
                "cortical_thickness_corr",
                "myelin",
                "myelin_smoothed",
            ],
        ),
        name="inputnode",
    )

    # Place the surfaces in a single node.
    collect_files = pe.Node(
        niu.Merge(10),
        name="collect_files",
    )
    workflow.connect([
        (inputnode, collect_files, [
            # fsLR-space surface mesh files
            ("lh_pial_surf", "in1"),
            ("rh_pial_surf", "in2"),
            ("lh_wm_surf", "in3"),
            ("rh_wm_surf", "in4"),
            # fsLR-space surface shape files
            ("sulcal_depth", "in5"),
            ("sulcal_curv", "in6"),
            ("cortical_thickness", "in7"),
            ("cortical_thickness_corr", "in8"),
            ("myelin", "in9"),
            ("myelin_smoothed", "in10"),
        ]),
    ])  # fmt:skip

    filter_out_undefined = pe.Node(
        FilterUndefined(),
        name="filter_out_undefined",
    )
    workflow.connect([(collect_files, filter_out_undefined, [("out", "inlist")])])

    ds_copied_outputs = pe.MapNode(
        DerivativesDataSink(
            base_directory=output_dir,
            check_hdr=False,
        ),
        name="ds_copied_outputs",
        run_without_submitting=True,
        mem_gb=1,
        iterfield=["in_file", "source_file"],
    )
    workflow.connect([
        (filter_out_undefined, ds_copied_outputs, [
            ("outlist", "in_file"),
            ("outlist", "source_file"),
        ]),
    ])  # fmt:skip

    return workflow
