"""Test xcp_d.workflows.plotting."""

import os

from nilearn import image

from xcp_d.tests.utils import get_nodes
from xcp_d.workflows import plotting


def test_init_plot_custom_slices_wf(ds001419_data, tmp_path_factory):
    """Test init_plot_custom_slices_wf."""
    tmpdir = tmp_path_factory.mktemp("test_init_plot_custom_slices_wf")

    nifti_file = ds001419_data["nifti_file"]
    nifti_3d = os.path.join(tmpdir, "img3d.nii.gz")
    img_3d = image.index_img(nifti_file, 5)
    img_3d.to_filename(nifti_3d)

    wf = plotting.init_plot_custom_slices_wf(
        output_dir=tmpdir,
        desc="SubcorticalOnAtlas",
        name="plot_custom_slices_wf",
    )
    wf.inputs.inputnode.name_source = nifti_file
    wf.inputs.inputnode.overlay_file = nifti_3d
    wf.inputs.inputnode.underlay_file = ds001419_data["t1w_mni"]
    wf.base_dir = tmpdir
    wf_res = wf.run()
    nodes = get_nodes(wf_res)
    overlay_figure = nodes["plot_custom_slices_wf.ds_overlay_figure"].get_output("out_file")
    assert os.path.isfile(overlay_figure)
