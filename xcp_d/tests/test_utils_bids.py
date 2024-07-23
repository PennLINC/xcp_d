"""Tests for the xcp_d.utils.bids module."""

import json
import os
import shutil

import pytest
from bids.layout import BIDSLayout

import xcp_d.utils.bids as xbids
from xcp_d.data import load as load_data


def test_collect_participants(datasets):
    """Test collect_participants.

    This also covers BIDSError and BIDSWarning.
    """
    bids_dir = datasets["ds001419"]

    bids_layout = BIDSLayout(bids_dir, validate=False)
    nonbids_layout = BIDSLayout(os.path.dirname(bids_dir), validate=False)

    # Pass in non-BIDS folder to get BIDSError.
    with pytest.raises(xbids.BIDSError, match="Could not find participants"):
        xbids.collect_participants(nonbids_layout, participant_label="fail")

    # Pass in BIDS folder with no matching participants to get BIDSWarning.
    with pytest.raises(xbids.BIDSError, match="Could not find participants"):
        xbids.collect_participants(bids_layout, participant_label="fail")

    # Pass in BIDS folder with only some participants to get BIDSWarning.
    with pytest.warns(xbids.BIDSWarning, match="Some participants were not found"):
        xbids.collect_participants(bids_layout, participant_label=["01", "fail"])

    # Pass in BIDS folder with only some participants to get BIDSError.
    with pytest.raises(xbids.BIDSError, match="Some participants were not found"):
        xbids.collect_participants(bids_layout, participant_label=["01", "fail"], strict=True)

    found_labels = xbids.collect_participants(bids_layout, participant_label=None)
    assert found_labels == ["01"]

    found_labels = xbids.collect_participants(bids_layout, participant_label="01")
    assert found_labels == ["01"]


def test_collect_data_ds001419(datasets):
    """Test the collect_data function."""
    bids_dir = datasets["ds001419"]
    layout = BIDSLayout(bids_dir, validate=False)

    # NIFTI workflow, but also get a BIDSLayout
    subj_data = xbids.collect_data(
        layout=layout,
        input_type="fmriprep",
        participant_label="01",
        bids_filters=None,
        file_format="nifti",
    )

    assert len(subj_data["bold"]) == 5
    assert "space-MNI152NLin2009cAsym" in subj_data["bold"][0]
    assert os.path.basename(subj_data["t1w"]) == "sub-01_desc-preproc_T1w.nii.gz"
    assert "space-" not in subj_data["t1w"]
    assert "to-MNI152NLin2009cAsym" in subj_data["anat_to_template_xfm"]
    assert "from-MNI152NLin2009cAsym" in subj_data["template_to_anat_xfm"]

    # CIFTI workflow
    subj_data = xbids.collect_data(
        layout=layout,
        input_type="fmriprep",
        participant_label="01",
        bids_filters={"bold": {"task": "rest"}},
        file_format="cifti",
    )

    assert len(subj_data["bold"]) == 1
    assert "space-fsLR" in subj_data["bold"][0]
    assert "space-" not in subj_data["t1w"]
    assert os.path.basename(subj_data["t1w"]) == "sub-01_desc-preproc_T1w.nii.gz"
    assert "to-MNI152NLin2009cAsym" in subj_data["anat_to_template_xfm"]
    assert "from-MNI152NLin2009cAsym" in subj_data["template_to_anat_xfm"]


def test_collect_data_nibabies(datasets):
    """Test the collect_data function."""
    bids_dir = datasets["nibabies"]
    xcp_d_config = str(load_data("xcp_d_bids_config2.json"))
    layout = BIDSLayout(
        bids_dir,
        validate=False,
        config=["bids", "derivatives", xcp_d_config],
    )
    cohort_files = layout.get(subject="01", cohort="1", space="MNIInfant", suffix="boldref")
    assert len(cohort_files) > 0

    # NIFTI workflow
    subj_data = xbids.collect_data(
        layout=layout,
        input_type="fmriprep",
        participant_label="01",
        bids_filters=None,
        file_format="nifti",
    )

    assert len(subj_data["bold"]) == 1
    assert "space-MNIInfant" in subj_data["bold"][0]
    assert "cohort-1" in subj_data["bold"][0]
    assert os.path.basename(subj_data["t1w"]) == "sub-01_ses-1mo_run-001_desc-preproc_T1w.nii.gz"
    assert "space-" not in subj_data["t1w"]
    assert "to-MNIInfant" in subj_data["anat_to_template_xfm"]
    assert "from-MNIInfant" in subj_data["template_to_anat_xfm"]

    # CIFTI workflow
    with pytest.raises(FileNotFoundError):
        subj_data = xbids.collect_data(
            layout=layout,
            input_type="fmriprep",
            participant_label="01",
            bids_filters=None,
            file_format="cifti",
        )


def test_collect_mesh_data(datasets, tmp_path_factory):
    """Test collect_mesh_data."""
    # Dataset without mesh files
    layout = BIDSLayout(datasets["fmriprep_without_freesurfer"], validate=False)
    mesh_available, standard_space_mesh, _, _ = xbids.collect_mesh_data(
        layout, "1648798153", bids_filters={}
    )
    assert mesh_available is False
    assert standard_space_mesh is False

    # Dataset with native-space mesh files (one file matching each query)
    layout = BIDSLayout(datasets["pnc"], validate=False)
    mesh_available, standard_space_mesh, _, _ = xbids.collect_mesh_data(
        layout, "1648798153", bids_filters={}
    )
    assert mesh_available is True
    assert standard_space_mesh is False

    # Dataset with standard-space mesh files (one file matching each query)
    std_mesh_dir = tmp_path_factory.mktemp("standard_mesh")
    shutil.copyfile(
        os.path.join(datasets["pnc"], "dataset_description.json"),
        std_mesh_dir / "dataset_description.json",
    )
    os.makedirs(std_mesh_dir / "sub-1648798153/ses-PNC1/anat", exist_ok=True)
    files = [
        "sub-1648798153_ses-PNC1_space-fsLR_den-32k_hemi-L_pial.surf.gii",
        "sub-1648798153_ses-PNC1_space-fsLR_den-32k_hemi-L_white.surf.gii",
        "sub-1648798153_ses-PNC1_space-fsLR_den-32k_hemi-R_pial.surf.gii",
        "sub-1648798153_ses-PNC1_space-fsLR_den-32k_hemi-R_white.surf.gii",
    ]
    for f in files:
        (std_mesh_dir / "sub-1648798153/ses-PNC1/anat").joinpath(f).touch()

    layout = BIDSLayout(std_mesh_dir, validate=False)
    mesh_available, standard_space_mesh, _, _ = xbids.collect_mesh_data(
        layout, "1648798153", bids_filters={}
    )
    assert mesh_available is True
    assert standard_space_mesh is True

    # Dataset with multiple files matching each query (raises an error)
    bad_mesh_dir = tmp_path_factory.mktemp("standard_mesh")
    shutil.copyfile(
        os.path.join(datasets["pnc"], "dataset_description.json"),
        bad_mesh_dir / "dataset_description.json",
    )
    os.makedirs(bad_mesh_dir / "sub-1648798153/ses-PNC1/anat", exist_ok=True)
    files = [
        "sub-1648798153_ses-PNC1_space-fsLR_den-32k_hemi-L_pial.surf.gii",
        "sub-1648798153_ses-PNC1_space-fsLR_den-32k_hemi-L_white.surf.gii",
        "sub-1648798153_ses-PNC1_space-fsLR_den-32k_hemi-R_pial.surf.gii",
        "sub-1648798153_ses-PNC1_space-fsLR_den-32k_hemi-R_white.surf.gii",
        "sub-1648798153_ses-PNC1_acq-test_space-fsLR_den-32k_hemi-L_pial.surf.gii",
        "sub-1648798153_ses-PNC1_acq-test_space-fsLR_den-32k_hemi-L_white.surf.gii",
        "sub-1648798153_ses-PNC1_acq-test_space-fsLR_den-32k_hemi-R_pial.surf.gii",
        "sub-1648798153_ses-PNC1_acq-test_space-fsLR_den-32k_hemi-R_white.surf.gii",
    ]
    for f in files:
        (std_mesh_dir / "sub-1648798153/ses-PNC1/anat").joinpath(f).touch()

    layout = BIDSLayout(std_mesh_dir, validate=False)
    with pytest.raises(ValueError, match="More than one surface found"):
        xbids.collect_mesh_data(layout, "1648798153", bids_filters={})

    # If we include BIDS filters, we should be able to ignore the existing files
    layout = BIDSLayout(datasets["pnc"], validate=False)
    mesh_available, standard_space_mesh, _, _ = xbids.collect_mesh_data(
        layout,
        "1648798153",
        bids_filters={
            "lh_pial_surf": {"acquisition": "test"},
            "rh_pial_surf": {"acquisition": "test"},
            "lh_wm_surf": {"acquisition": "test"},
            "rh_wm_surf": {"acquisition": "test"},
            "lh_subject_sphere": {"acquisition": "test"},
            "rh_subject_sphere": {"acquisition": "test"},
        },
    )
    assert mesh_available is False
    assert standard_space_mesh is False


def test_collect_morphometry_data(datasets, tmp_path_factory):
    """Test collect_morphometry_data."""
    # Dataset without morphometry files
    layout = BIDSLayout(datasets["fmriprep_without_freesurfer"], validate=False)
    morph_file_types, _ = xbids.collect_morphometry_data(layout, "1648798153", bids_filters={})
    assert morph_file_types == []

    # Dataset with morphometry files (one file matching each query)
    layout = BIDSLayout(datasets["pnc"], validate=False)
    morph_file_types, _ = xbids.collect_morphometry_data(layout, "1648798153", bids_filters={})
    assert morph_file_types == ["cortical_thickness", "sulcal_curv", "sulcal_depth"]

    # Dataset with multiple files matching each query (raises an error)
    bad_morph_dir = tmp_path_factory.mktemp("bad_morph")
    shutil.copyfile(
        os.path.join(datasets["pnc"], "dataset_description.json"),
        bad_morph_dir / "dataset_description.json",
    )
    os.makedirs(bad_morph_dir / "sub-1648798153/ses-PNC1/anat", exist_ok=True)
    files = [
        "sub-1648798153_ses-PNC1_acq-refaced_space-fsLR_den-91k_thickness.dscalar.nii",
        "sub-1648798153_ses-PNC1_acq-refaced2_space-fsLR_den-91k_thickness.dscalar.nii",
        "sub-1648798153_ses-PNC1_acq-refaced_space-fsLR_den-91k_sulc.dscalar.nii",
        "sub-1648798153_ses-PNC1_acq-refaced2_space-fsLR_den-91k_sulc.dscalar.nii",
        "sub-1648798153_ses-PNC1_acq-refaced_space-fsLR_den-91k_curv.dscalar.nii",
        "sub-1648798153_ses-PNC1_acq-refaced2_space-fsLR_den-91k_curv.dscalar.nii",
    ]
    for f in files:
        (bad_morph_dir / "sub-1648798153/ses-PNC1/anat").joinpath(f).touch()

    layout = BIDSLayout(bad_morph_dir, validate=False)
    with pytest.raises(ValueError, match="More than one .* found"):
        xbids.collect_morphometry_data(layout, "1648798153", bids_filters={})

    # If we include BIDS filters, we should be able to ignore the existing files
    layout = BIDSLayout(datasets["pnc"], validate=False)
    morph_file_types, _ = xbids.collect_morphometry_data(
        layout,
        "1648798153",
        bids_filters={
            "cortical_thickness": {"acquisition": "test"},
            "sulcal_curv": {"acquisition": "test"},
            "sulcal_depth": {"acquisition": "test"},
        },
    )
    assert morph_file_types == []


def test_write_dataset_description(datasets, tmp_path_factory, caplog):
    """Test write_dataset_description."""
    tmpdir = tmp_path_factory.mktemp("test_write_dataset_description")
    dset_description = os.path.join(tmpdir, "dataset_description.json")

    # The function expects a description file in the fmri_dir.
    with pytest.raises(FileNotFoundError, match="Dataset description DNE"):
        xbids.write_dataset_description(tmpdir, tmpdir, atlases=None, custom_confounds_folder=None)
    assert not os.path.isfile(dset_description)

    # It will work when we give it a real fmri_dir.
    fmri_dir = datasets["ds001419"]
    xbids.write_dataset_description(
        fmri_dir,
        tmpdir,
        atlases=["Gordon"],
        custom_confounds_folder="/fake/path4",
    )
    assert os.path.isfile(dset_description)

    # Now overwrite the description.
    with open(dset_description, "r") as fo:
        desc = json.load(fo)

    assert "'preprocessed' is already a dataset link" not in caplog.text
    assert "'custom_confounds' is already a dataset link" not in caplog.text
    xbids.write_dataset_description(
        tmpdir,
        tmpdir,
        atlases=["Gordon"],
        custom_confounds_folder="/fake/path4",
    )
    assert "'preprocessed' is already a dataset link" in caplog.text
    assert "'custom_confounds' is already a dataset link" in caplog.text

    # Now change the version and re-run the function.
    desc["GeneratedBy"][0]["Version"] = "0.0.1"
    with open(dset_description, "w") as fo:
        json.dump(desc, fo, indent=4)

    assert "Previous output generated by version" not in caplog.text
    xbids.write_dataset_description(fmri_dir, tmpdir, atlases=None, custom_confounds_folder=None)
    assert "Previous output generated by version" in caplog.text

    # Should raise a warning if DatasetType is not in the description
    desc.pop("DatasetType")
    with open(dset_description, "w") as fo:
        json.dump(desc, fo, indent=4)

    assert "DatasetType key not in" not in caplog.text
    xbids.write_dataset_description(tmpdir, tmpdir, atlases=None, custom_confounds_folder=None)
    assert "DatasetType key not in" in caplog.text

    # Should raise an error if DatasetType is present, but isn't "derivative"
    desc["DatasetType"] = "raw"
    with open(dset_description, "w") as fo:
        json.dump(desc, fo, indent=4)

    with pytest.raises(ValueError, match="XCP-D only works on derivative datasets."):
        xbids.write_dataset_description(
            tmpdir,
            tmpdir,
            atlases=None,
            custom_confounds_folder=None,
        )


def test_write_atlas_dataset_description(tmp_path_factory, caplog):
    """Test write_atlas_dataset_description."""
    tmpdir = tmp_path_factory.mktemp("test_write_atlas_dataset_description")
    dset_description = os.path.join(tmpdir, "dataset_description.json")
    xbids.write_atlas_dataset_description(tmpdir)
    assert os.path.isfile(dset_description)

    # Now change the version and re-run the function.
    with open(dset_description, "r") as fo:
        desc = json.load(fo)

    desc["GeneratedBy"][0]["Version"] = "0.0.1"
    with open(dset_description, "w") as fo:
        json.dump(desc, fo, indent=4)

    assert "Previous output generated by version" not in caplog.text
    xbids.write_atlas_dataset_description(tmpdir)
    assert "Previous output generated by version" in caplog.text


def test_get_preproc_pipeline_info(datasets, tmp_path_factory, caplog):
    """Test get_preproc_pipeline_info."""
    input_types = ["fmriprep", "nibabies", "hcp", "dcan", "ukb"]
    for input_type in input_types:
        info_dict = xbids.get_preproc_pipeline_info(input_type, datasets["ds001419"])
        assert "references" in info_dict.keys()

    with pytest.raises(ValueError, match="Unsupported input_type"):
        xbids.get_preproc_pipeline_info("fail", datasets["ds001419"])

    assert "Dataset description DNE" not in caplog.text
    info_dict = xbids.get_preproc_pipeline_info("fmriprep", ".")
    assert "Dataset description DNE" in caplog.text
    assert info_dict["name"] == "fmriprep"
    assert info_dict["version"] == "unknown"

    tmpdir = tmp_path_factory.mktemp("test_get_preproc_pipeline_info")
    dset_description = os.path.join(tmpdir, "dataset_description.json")
    with open(dset_description, "w") as fo:
        json.dump({}, fo)

    assert "GeneratedBy key DNE" not in caplog.text
    info_dict = xbids.get_preproc_pipeline_info("dcan", tmpdir)
    assert "GeneratedBy key DNE" in caplog.text
    assert info_dict["name"] == "dcan"
    assert info_dict["version"] == "unknown"


def test_get_tr(ds001419_data):
    """Test _get_tr."""
    t_r = xbids._get_tr(ds001419_data["nifti_file"])
    assert t_r == 3.0

    t_r = xbids._get_tr(ds001419_data["cifti_file"])
    assert t_r == 3.0


def test_get_entity(datasets):
    """Test get_entity."""
    fname = os.path.join(datasets["ds001419"], "sub-01", "anat", "sub-01_desc-preproc_T1w.nii.gz")
    entity = xbids.get_entity(fname, "space")
    assert entity == "T1w"

    fname = os.path.join(
        datasets["ds001419"],
        "sub-01",
        "func",
        "sub-01_task-rest_desc-preproc_bold.nii.gz",
    )
    entity = xbids.get_entity(fname, "space")
    assert entity == "native"
    entity = xbids.get_entity(fname, "desc")
    assert entity == "preproc"
    entity = xbids.get_entity(fname, "fail")
    assert entity is None

    fname = os.path.join(
        datasets["ds001419"],
        "sub-01",
        "fmap",
        "sub-01_fmapid-auto00001_desc-coeff1_fieldmap.nii.gz",
    )
    with pytest.raises(ValueError, match="Unknown space"):
        xbids.get_entity(fname, "space")


def test_group_across_runs():
    """Test group_across_runs."""
    in_files = [
        "/path/sub-01_task-axcpt_run-03_bold.nii.gz",
        "/path/sub-01_task-rest_run-03_bold.nii.gz",
        "/path/sub-01_task-rest_run-01_bold.nii.gz",
        "/path/sub-01_task-axcpt_run-02_bold.nii.gz",
        "/path/sub-01_task-rest_run-02_bold.nii.gz",
        "/path/sub-01_task-axcpt_run-01_bold.nii.gz",
    ]
    grouped_files = xbids.group_across_runs(in_files)
    assert isinstance(grouped_files, list)
    assert len(grouped_files[0]) == 3
    assert grouped_files[0] == [
        "/path/sub-01_task-axcpt_run-01_bold.nii.gz",
        "/path/sub-01_task-axcpt_run-02_bold.nii.gz",
        "/path/sub-01_task-axcpt_run-03_bold.nii.gz",
    ]
    assert len(grouped_files[1]) == 3
    assert grouped_files[1] == [
        "/path/sub-01_task-rest_run-01_bold.nii.gz",
        "/path/sub-01_task-rest_run-02_bold.nii.gz",
        "/path/sub-01_task-rest_run-03_bold.nii.gz",
    ]

    in_files = [
        "/path/sub-01_task-rest_dir-LR_run-2_bold.nii.gz",
        "/path/sub-01_task-rest_dir-RL_run-1_bold.nii.gz",
        "/path/sub-01_task-axcpt_dir-LR_bold.nii.gz",
        "/path/sub-01_task-rest_dir-RL_run-2_bold.nii.gz",
        "/path/sub-01_task-rest_dir-LR_run-1_bold.nii.gz",
        "/path/sub-01_task-axcpt_dir-RL_bold.nii.gz",
    ]
    grouped_files = xbids.group_across_runs(in_files)
    assert isinstance(grouped_files, list)
    assert len(grouped_files[0]) == 2
    assert grouped_files[0] == [
        "/path/sub-01_task-axcpt_dir-LR_bold.nii.gz",
        "/path/sub-01_task-axcpt_dir-RL_bold.nii.gz",
    ]
    assert len(grouped_files[1]) == 4
    assert grouped_files[1] == [
        "/path/sub-01_task-rest_dir-LR_run-1_bold.nii.gz",
        "/path/sub-01_task-rest_dir-RL_run-1_bold.nii.gz",
        "/path/sub-01_task-rest_dir-LR_run-2_bold.nii.gz",
        "/path/sub-01_task-rest_dir-RL_run-2_bold.nii.gz",
    ]


def test_make_uri():
    """Test _make_uri."""
    in_file = "/path/to/dset/sub-01/func/sub-01_task-rest_bold.nii.gz"
    dataset_name = "test"
    dataset_path = "/path/to/dset"
    uri = xbids._make_uri(in_file, dataset_name=dataset_name, dataset_path=dataset_path)
    assert uri == "bids:test:sub-01/func/sub-01_task-rest_bold.nii.gz"

    dataset_path = "/another/path/haha"
    with pytest.raises(ValueError, match="is not in the subpath of"):
        xbids._make_uri(in_file, dataset_name=dataset_name, dataset_path=dataset_path)


def test_make_xcpd_uri():
    """Test _make_xcpd_uri."""
    out_file = "/path/to/dset/xcp_d/sub-01/func/sub-01_task-rest_bold.nii.gz"
    uri = xbids._make_xcpd_uri(out_file, output_dir="/path/to/dset/xcp_d")
    assert uri == ["bids::sub-01/func/sub-01_task-rest_bold.nii.gz"]

    xbids._make_xcpd_uri([out_file], output_dir="/path/to/dset/xcp_d")
    assert uri == ["bids::sub-01/func/sub-01_task-rest_bold.nii.gz"]


def test_make_atlas_uri():
    """Test _make_atlas_uri."""
    out_file = "/path/to/dset/xcp_d/atlases/sub-01/func/sub-01_task-rest_bold.nii.gz"
    uri = xbids._make_atlas_uri(out_file, output_dir="/path/to/dset/xcp_d")
    assert uri == ["bids:atlas:sub-01/func/sub-01_task-rest_bold.nii.gz"]

    xbids._make_atlas_uri([out_file], output_dir="/path/to/dset/xcp_d")
    assert uri == ["bids:atlas:sub-01/func/sub-01_task-rest_bold.nii.gz"]


def test_make_xcpd_uri_lol():
    """Test _make_xcpd_uri_lol."""
    in_list = [
        [
            "/path/to/dset/xcp_d/sub-01/func/sub-01_task-rest_run-1_bold.nii.gz",
            "/path/to/dset/xcp_d/sub-02/func/sub-01_task-rest_run-1_bold.nii.gz",
            "/path/to/dset/xcp_d/sub-03/func/sub-01_task-rest_run-1_bold.nii.gz",
        ],
        [
            "/path/to/dset/xcp_d/sub-01/func/sub-01_task-rest_run-2_bold.nii.gz",
            "/path/to/dset/xcp_d/sub-02/func/sub-01_task-rest_run-2_bold.nii.gz",
            "/path/to/dset/xcp_d/sub-03/func/sub-01_task-rest_run-2_bold.nii.gz",
        ],
    ]
    uris = xbids._make_xcpd_uri_lol(in_list, output_dir="/path/to/dset/xcp_d")
    assert uris == [
        [
            "bids::sub-01/func/sub-01_task-rest_run-1_bold.nii.gz",
            "bids::sub-01/func/sub-01_task-rest_run-2_bold.nii.gz",
        ],
        [
            "bids::sub-02/func/sub-01_task-rest_run-1_bold.nii.gz",
            "bids::sub-02/func/sub-01_task-rest_run-2_bold.nii.gz",
        ],
        [
            "bids::sub-03/func/sub-01_task-rest_run-1_bold.nii.gz",
            "bids::sub-03/func/sub-01_task-rest_run-2_bold.nii.gz",
        ],
    ]


def test_make_preproc_uri():
    """Test _make_preproc_uri."""
    out_file = "/path/to/dset/sub-01/func/sub-01_task-rest_bold.nii.gz"
    uri = xbids._make_preproc_uri(out_file, fmri_dir="/path/to/dset")
    assert uri == ["bids:preprocessed:sub-01/func/sub-01_task-rest_bold.nii.gz"]

    xbids._make_preproc_uri([out_file], fmri_dir="/path/to/dset")
    assert uri == ["bids:preprocessed:sub-01/func/sub-01_task-rest_bold.nii.gz"]


def test_make_custom_uri():
    """Test _make_custom_uri."""
    out_file = "/path/to/dset/sub-01_task-rest_bold.nii.gz"
    uri = xbids._make_custom_uri(out_file)
    assert uri == ["bids:custom_confounds:sub-01_task-rest_bold.nii.gz"]

    xbids._make_custom_uri([out_file])
    assert uri == ["bids:custom_confounds:sub-01_task-rest_bold.nii.gz"]
