# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Workflows for extracting time series and computing functional connectivity."""

import nilearn as nl
from nipype import Function
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from xcp_d.interfaces.connectivity import ApplyTransformsx, ConnectPlot, NiftiConnect
from xcp_d.interfaces.workbench import CiftiCorrelation, CiftiParcellate
from xcp_d.utils.atlas import get_atlas_cifti, get_atlas_nifti
from xcp_d.utils.doc import fill_doc
from xcp_d.utils.utils import get_transformfile


@fill_doc
def init_fcon_ts_wf(
    mem_gb,
    t1w_to_native,
    mni_to_t1w,
    omp_nthreads,
    bold_file,
    name="fcons_ts_wf",
):
    """Extract BOLD time series and compute functional connectivity.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.workflow.connectivity import init_fcon_ts_wf
            wf = init_fcon_ts_wf(
                mem_gb=0.1,
                t1w_to_native="identity",
                mni_to_t1w="identity",
                omp_nthreads=1,
                bold_file="/path/to/file.nii.gz",
                name="fcons_ts_wf",
            )

    Parameters
    ----------
    %(mem_gb)s
    t1w_to_native: str
        transformation files from tw1 to native space ( from fmriprep)
    mni_to_t1w
    %(omp_nthreads)s
    bold_file: str
        bold file for post processing
    name

    Inputs
    ------
    clean_bold
        clean bold after filtered out nuisscance and filtering
    ref_file
        reference file
    mni_tot1w
        MNI to T1w registration files from fmriprep

    Outputs
    -------
    schaefer timeseries and func matrices, 100-1000 parcels

    gs360_ts
        glasser 360 timeseries
    gs360_fc
        glasser 360  func matrices
    gd333_ts
        gordon 333 timeseries
    gd333_fc
        gordon 333 func matrices
    qc_file
        quality control files

    """
    workflow = Workflow(name=name)

    workflow.__desc__ = f"""
Processed functional timeseries were extracted  from  the residual BOLD signal
with  *Nilearn* {nl.__version__}'s *NiftiLabelsMasker* for the following atlases
[@nilearn]: the Schaefer 200 and 400-parcel resolution atlas
[@Schaefer_2017],the Glasser atlas [@Glasser_2016], and the Gordon atlas
[@Gordon_2014] atlases.  Corresponding pair-wise functional connectivity between
all regions was computed for each atlas, which was operationalized as the
Pearson's correlation of each parcel's (unsmoothed) timeseries.
"""

    ATLAS_NAMES = [
        "Schaefer100x17",
        "Schaefer200x17",
        "Schaefer300x17",
        "Schaefer400x17",
        "Schaefer500x17",
        "Schaefer600x17",
        "Schaefer700x17",
        "Schaefer800x17",
        "Schaefer900x17",
        "Schaefer1000x17",
        "Glasser360",
        "Gordon333",
        "TianSubcortical",
    ]

    inputnode = pe.Node(
        niu.IdentityInterface(fields=['bold_file', 'clean_bold', 'ref_file', 'atlas_names']),
        name='inputnode',
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['atlas_names', 'timeseries', 'correlations', 'connectplot']),
        name='outputnode',
    )

    inputnode.inputs.bold_file = bold_file
    inputnode.inputs.atlas_names = ATLAS_NAMES

    # get atlases via pkgrf
    atlas_nifti_grabber = pe.MapNode(
        Function(input_names=["atlasname"], output_names=["out_file"], function=get_atlas_nifti),
        name="atlas_nifti_grabber",
        iterfield=["atlasname"],
    )

    # get transform file via string manipulation
    transformfile = get_transformfile(bold_file=bold_file,
                                      mni_to_t1w=mni_to_t1w,
                                      t1w_to_native=t1w_to_native)

    # Using the generated transforms, apply them to get everything in the correct MNI form
    atlas_transform = pe.MapNode(
        ApplyTransformsx(
            transforms=transformfile,
            interpolation="MultiLabel",
            input_image_type=3,
            dimension=3,
        ),
        name="atlas_mni_to_native",
        iterfield=["input_image"],
        mem_gb=mem_gb,
        n_procs=omp_nthreads,
    )

    nifti_connect = pe.MapNode(
        NiftiConnect(),
        name="nifti_connect",
        iterfield=["atlas"],
        mem_gb=mem_gb,
    )

    # Create a node to plot the matrixes
    matrix_plot = pe.Node(
        ConnectPlot(),
        name="matrix_plot_wf",
        mem_gb=mem_gb,
    )

    workflow.connect([
        # Transform Atlas to correct MNI2009 space
        (inputnode, outputnode, [('atlas_names', 'atlas_names')]),
        (inputnode, atlas_nifti_grabber, [('atlas_names', 'atlasname')]),
        (inputnode, nifti_connect, [('clean_bold', 'filtered_file')]),
        (inputnode, matrix_plot, [('bold_file', 'in_file'), ['atlas_names', 'atlas_names']]),
        (atlas_nifti_grabber, atlas_transform, [('out_file', 'input_image')]),
        (atlas_transform, nifti_connect, [('output_image', 'atlas')]),
        (nifti_connect, outputnode, [('time_series_tsv', 'timeseries'),
                                     ('fcon_matrix_tsv', 'correlations')]),
        (nifti_connect, matrix_plot, [('time_series_tsv', 'time_series_tsv')]),
        (matrix_plot, outputnode, [('connectplot', 'connectplot')]),
    ])

    return workflow


@fill_doc
def init_cifti_conts_wf(
    mem_gb,
    omp_nthreads,
    name="cifti_ts_con_wf",
):
    """Extract CIFTI time series.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.workflow.connectivity import init_cifti_conts_wf
            wf = init_fcon_ts_wf(
                mem_gb=0.1,
                omp_nthreads=1,
                name="cifti_ts_con_wf",
            )

    Parameters
    ----------
    %(mem_gb)s
    %(omp_nthreads)s

    Inputs
    ------
    clean_cifti
        clean cifti after filtered out nuisscance and filtering

    Outputs
    -------
    schaefer time series and func matrices, 100-1000 nodes

    gs360_ts
        glasser 360 timeseries
    gs360_fc
        glasser 360  func matrices
    gd333_ts
        gordon 333 timeseries
    gd333_fc
        gordon 333 func matrices
    qc_file
        quality control files
    """
    workflow = Workflow(name=name)
    workflow.__desc__ = """
Processed functional timeseries were extracted from residual BOLD  using
Connectome Workbench[@hcppipelines]: for the following atlases: the Schaefer 200
and 400-parcel resolution atlas [@Schaefer_2017], the Glasser atlas
[@Glasser_2016] and the Gordon atlas [@Gordon_2014].  Corresponding pair-wise
functional connectivity between all regions was computed for each atlas, which
was operationalized as the Pearson's correlation of each parcel's (unsmoothed)
timeseries with the Connectome Workbench.
"""
    inputnode = pe.Node(niu.IdentityInterface(fields=['clean_cifti']),
                        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=[
        'sc117_ts', 'sc117_fc', 'sc217_ts', 'sc217_fc', 'sc317_ts', 'sc317_fc',
        'sc417_ts', 'sc417_fc', 'sc517_ts', 'sc517_fc', 'sc617_ts', 'sc617_fc',
        'sc717_ts', 'sc717_fc', 'sc817_ts', 'sc817_fc', 'sc917_ts', 'sc917_fc',
        'sc1017_ts', 'sc1017_fc', 'gs360_ts', 'gs360_fc', 'gd333_ts',
        'gd333_fc', 'ts50_ts', 'ts50_fc', 'connectplot'
    ]),
        name='outputnode')

    # Get the correct atlas for each cifti
    sc117atlas = get_atlas_cifti(atlasname='schaefer100x17')
    sc217atlas = get_atlas_cifti(atlasname='schaefer200x17')
    sc317atlas = get_atlas_cifti(atlasname='schaefer300x17')
    sc417atlas = get_atlas_cifti(atlasname='schaefer400x17')
    sc517atlas = get_atlas_cifti(atlasname='schaefer500x17')
    sc617atlas = get_atlas_cifti(atlasname='schaefer600x17')
    sc717atlas = get_atlas_cifti(atlasname='schaefer700x17')
    sc817atlas = get_atlas_cifti(atlasname='schaefer800x17')
    sc917atlas = get_atlas_cifti(atlasname='schaefer900x17')
    sc1017atlas = get_atlas_cifti(atlasname='schaefer1000x17')
    gs360atlas = get_atlas_cifti(atlasname='glasser360')
    gd333atlas = get_atlas_cifti(atlasname='gordon333')
    ts50atlas = get_atlas_cifti(atlasname='tiansubcortical')

    # Use Connectome Workbench for Parcellation
    sc117parcel = pe.Node(CiftiParcellate(atlas_label=sc117atlas,
                                          direction='COLUMN'),
                          mem_gb=mem_gb,
                          name='sc117parcel',
                          n_procs=omp_nthreads)
    sc217parcel = pe.Node(CiftiParcellate(atlas_label=sc217atlas,
                                          direction='COLUMN'),
                          mem_gb=mem_gb,
                          name='sc217parcel',
                          n_procs=omp_nthreads)
    sc317parcel = pe.Node(CiftiParcellate(atlas_label=sc317atlas,
                                          direction='COLUMN'),
                          mem_gb=mem_gb,
                          name='sc317parcel',
                          n_procs=omp_nthreads)
    sc417parcel = pe.Node(CiftiParcellate(atlas_label=sc417atlas,
                                          direction='COLUMN'),
                          mem_gb=mem_gb,
                          name='sc417parcel',
                          n_procs=omp_nthreads)
    sc517parcel = pe.Node(CiftiParcellate(atlas_label=sc517atlas,
                                          direction='COLUMN'),
                          mem_gb=mem_gb,
                          name='sc517parcel',
                          n_procs=omp_nthreads)
    sc617parcel = pe.Node(CiftiParcellate(atlas_label=sc617atlas,
                                          direction='COLUMN'),
                          mem_gb=mem_gb,
                          name='sc617parcel',
                          n_procs=omp_nthreads)
    sc717parcel = pe.Node(CiftiParcellate(atlas_label=sc717atlas,
                                          direction='COLUMN'),
                          mem_gb=mem_gb,
                          name='sc717parcel',
                          n_procs=omp_nthreads)
    sc817parcel = pe.Node(CiftiParcellate(atlas_label=sc817atlas,
                                          direction='COLUMN'),
                          mem_gb=mem_gb,
                          name='sc817parcel',
                          n_procs=omp_nthreads)
    sc917parcel = pe.Node(CiftiParcellate(atlas_label=sc917atlas,
                                          direction='COLUMN'),
                          mem_gb=mem_gb,
                          name='sc917parcel',
                          n_procs=omp_nthreads)
    sc1017parcel = pe.Node(CiftiParcellate(atlas_label=sc1017atlas,
                                           direction='COLUMN'),
                           mem_gb=mem_gb,
                           name='sc1017parcel',
                           n_procs=omp_nthreads)
    gs360parcel = pe.Node(CiftiParcellate(atlas_label=gs360atlas,
                                          direction='COLUMN'),
                          mem_gb=mem_gb,
                          name='gs360parcel',
                          n_procs=omp_nthreads)
    gd333parcel = pe.Node(CiftiParcellate(atlas_label=gd333atlas,
                                          direction='COLUMN'),
                          mem_gb=mem_gb,
                          name='gd333parcel',
                          n_procs=omp_nthreads)
    ts50parcel = pe.Node(CiftiParcellate(atlas_label=ts50atlas,
                                         direction='COLUMN'),
                         mem_gb=mem_gb,
                         name='ts50parcel',
                         n_procs=omp_nthreads)

    # Create a node for plotting the matrixes
    matrix_plot = pe.Node(ConnectPlot(), name="matrix_plot_wf", mem_gb=mem_gb)

    # Compute correlation matrixes via Connectome Workbench
    sc117corr = pe.Node(CiftiCorrelation(),
                        mem_gb=mem_gb,
                        name='sc117corr',
                        n_procs=omp_nthreads)
    sc217corr = pe.Node(CiftiCorrelation(),
                        mem_gb=mem_gb,
                        name='sc217corr',
                        n_procs=omp_nthreads)
    sc317corr = pe.Node(CiftiCorrelation(),
                        mem_gb=mem_gb,
                        name='sc317corr',
                        n_procs=omp_nthreads)
    sc417corr = pe.Node(CiftiCorrelation(),
                        mem_gb=mem_gb,
                        name='sc417corr',
                        n_procs=omp_nthreads)
    sc517corr = pe.Node(CiftiCorrelation(),
                        mem_gb=mem_gb,
                        name='sc517corr',
                        n_procs=omp_nthreads)
    sc617corr = pe.Node(CiftiCorrelation(),
                        mem_gb=mem_gb,
                        name='sc617corr',
                        n_procs=omp_nthreads)
    sc717corr = pe.Node(CiftiCorrelation(),
                        mem_gb=mem_gb,
                        name='sc717corr',
                        n_procs=omp_nthreads)
    sc817corr = pe.Node(CiftiCorrelation(),
                        mem_gb=mem_gb,
                        name='sc817corr',
                        n_procs=omp_nthreads)
    sc917corr = pe.Node(CiftiCorrelation(),
                        mem_gb=mem_gb,
                        name='sc917corr',
                        n_procs=omp_nthreads)
    sc1017corr = pe.Node(CiftiCorrelation(),
                         mem_gb=mem_gb,
                         name='sc1017corr',
                         n_procs=omp_nthreads)
    gs360corr = pe.Node(CiftiCorrelation(),
                        mem_gb=mem_gb,
                        name='gs360corr',
                        n_procs=omp_nthreads)
    gd333corr = pe.Node(CiftiCorrelation(),
                        mem_gb=mem_gb,
                        name='gd333corr',
                        n_procs=omp_nthreads)
    ts50corr = pe.Node(CiftiCorrelation(),
                       mem_gb=mem_gb,
                       name='ts50corr',
                       n_procs=omp_nthreads)

    workflow.connect([  # for parcellation
        (inputnode, sc117parcel, [('clean_cifti', 'in_file')]),
        (inputnode, sc217parcel, [('clean_cifti', 'in_file')]),
        (inputnode, sc317parcel, [('clean_cifti', 'in_file')]),
        (inputnode, sc417parcel, [('clean_cifti', 'in_file')]),
        (inputnode, sc517parcel, [('clean_cifti', 'in_file')]),
        (inputnode, sc617parcel, [('clean_cifti', 'in_file')]),
        (inputnode, sc717parcel, [('clean_cifti', 'in_file')]),
        (inputnode, sc817parcel, [('clean_cifti', 'in_file')]),
        (inputnode, sc917parcel, [('clean_cifti', 'in_file')]),
        (inputnode, sc1017parcel, [('clean_cifti', 'in_file')]),
        (inputnode, gd333parcel, [('clean_cifti', 'in_file')]),
        (inputnode, gs360parcel, [('clean_cifti', 'in_file')]),
        (inputnode, ts50parcel, [('clean_cifti', 'in_file')]),
        (sc117parcel, outputnode, [('out_file', 'sc117_ts')]),
        (sc217parcel, outputnode, [('out_file', 'sc217_ts')]),
        (sc317parcel, outputnode, [('out_file', 'sc317_ts')]),
        (sc417parcel, outputnode, [('out_file', 'sc417_ts')]),
        (sc517parcel, outputnode, [('out_file', 'sc517_ts')]),
        (sc617parcel, outputnode, [('out_file', 'sc617_ts')]),
        (sc717parcel, outputnode, [('out_file', 'sc717_ts')]),
        (sc817parcel, outputnode, [('out_file', 'sc817_ts')]),
        (sc917parcel, outputnode, [('out_file', 'sc917_ts')]),
        (sc1017parcel, outputnode, [('out_file', 'sc1017_ts')]),
        (gs360parcel, outputnode, [('out_file', 'gs360_ts')]),
        (gd333parcel, outputnode, [('out_file', 'gd333_ts')]),
        (ts50parcel, outputnode, [('out_file', 'ts50_ts')]),
        (sc117parcel, sc117corr, [('out_file', 'in_file')]),  # for correlation
        (sc217parcel, sc217corr, [('out_file', 'in_file')]),
        (sc317parcel, sc317corr, [('out_file', 'in_file')]),
        (sc417parcel, sc417corr, [('out_file', 'in_file')]),
        (sc517parcel, sc517corr, [('out_file', 'in_file')]),
        (sc617parcel, sc617corr, [('out_file', 'in_file')]),
        (sc717parcel, sc717corr, [('out_file', 'in_file')]),
        (sc817parcel, sc817corr, [('out_file', 'in_file')]),
        (sc917parcel, sc917corr, [('out_file', 'in_file')]),
        (sc1017parcel, sc1017corr, [('out_file', 'in_file')]),
        (gs360parcel, gs360corr, [('out_file', 'in_file')]),
        (gd333parcel, gd333corr, [('out_file', 'in_file')]),
        (ts50parcel, ts50corr, [('out_file', 'in_file')]),
        (sc117corr, outputnode, [('out_file', 'sc117_fc')]),
        (sc217corr, outputnode, [('out_file', 'sc217_fc')]),
        (sc317corr, outputnode, [('out_file', 'sc317_fc')]),
        (sc417corr, outputnode, [('out_file', 'sc417_fc')]),
        (sc517corr, outputnode, [('out_file', 'sc517_fc')]),
        (sc617corr, outputnode, [('out_file', 'sc617_fc')]),
        (sc717corr, outputnode, [('out_file', 'sc717_fc')]),
        (sc817corr, outputnode, [('out_file', 'sc817_fc')]),
        (sc917corr, outputnode, [('out_file', 'sc917_fc')]),
        (sc1017corr, outputnode, [('out_file', 'sc1017_fc')]),
        (gs360corr, outputnode, [('out_file', 'gs360_fc')]),
        (gd333corr, outputnode, [('out_file', 'gd333_fc')]),
        (ts50corr, outputnode, [('out_file', 'ts50_fc')]),
        (inputnode, matrix_plot, [('clean_cifti', 'in_file')]),   # for plotting
        (sc217parcel, matrix_plot, [('out_file', 'sc217_timeseries')]),
        (sc417parcel, matrix_plot, [('out_file', 'sc417_timeseries')]),
        (gd333parcel, matrix_plot, [('out_file', 'gd333_timeseries')]),
        (gs360parcel, matrix_plot, [('out_file', 'gs360_timeseries')]),
        (matrix_plot, outputnode, [('connectplot', 'connectplot')])
    ])

    return workflow
