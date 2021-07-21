# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""ploting tools."""
import numpy as np
import nibabel as nb
import pandas as pd
from nilearn.signal import clean 
import matplotlib.pyplot as plt
from matplotlib import gridspec as mgs
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, Normalize
import seaborn as sns
from nilearn._utils import check_niimg_4d
from nilearn._utils.niimg import _safe_get_data
from niworkflows.viz.plots import _decimate_data
from write_save import read_ndata


def plot_svg(fdata,fd,dvars,filename,tr=1):
    '''
    plot carpetplot with fd and dvars
    ------------
    fdata:
      4D ndarray
    fd:
      framewise displacement
    dvars: 
      dvars
    filename
      filename
    tr:
    repetion time 
    '''
    fig = plt.figure(constrained_layout=False, figsize=(30, 15))
    grid = mgs.GridSpec(3, 1, wspace=0.0, hspace=0.05,
                               height_ratios=[1] * (3 - 1) + [5])
    confoundplot(fd, grid[0], tr=tr, color='b', name='FD')
    confoundplot(dvars, grid[1], tr=tr, color='r', name='DVARS')
    plot_carpet(func_data=fdata,subplot=grid[-1], tr=tr,)
    fig.savefig(filename,bbox_inches="tight", pad_inches=None)

def compute_dvars(datat):
    '''
    compute standard dvars

    datat : numpy darrays
        data matrix vertices by timepoints 
     
    '''
    firstcolumn=np.zeros((datat.shape[0]))[...,None]
    datax=np.hstack((firstcolumn,np.diff(datat)))
    datax_ss=np.sum(np.square(datax),axis=0)/datat.shape[0]
    return np.sqrt(datax_ss)


def plot_carpet(func_data,detrend=True, nskip=0, size=(950, 800),
                subplot=None, title=None, output_file=None, legend=False,
                tr=None):
    """
    Plot an image representation of voxel intensities across time also know
    as the "carpet plot"
    from Niworkflows
    Parameters
    ----------
        func_data : 
            4D ndarray
        detrend : boolean, optional
            Detrend and standardize the data prior to plotting.
        nskip : int
            Number of volumes at the beginning of the scan marked to be excluded.
        title : string, optional
            The title displayed on the figure.
        output_file : string, or None, optional
            The name of an image file to export the plot to. Valid extensions
            are .png, .pdf, .svg. If output_file is not None, the plot
            is saved to a file, and the display is closed.
        legend : bool
            Whether to render the average functional series with ``atlaslabels`` as
            overlay.
        tr : float , optional
            Specify the TR, if specified it uses this value. If left as None,
            # Frames is plotted instead of time.
    """

    # Define TR and number of frames
    notr = False
    if tr is None:
        notr = True
        tr = 1    
    ntsteps = func_data.shape[-1]

    data = func_data.reshape(-1, ntsteps)

    p_dec = 1 + data.shape[0] // size[0]
    if p_dec:
        data = data[::p_dec, :]

    t_dec = 1 + data.shape[1] // size[1]
    if t_dec:
        data = data[:, ::t_dec]

    # Detrend data
    v = (None, None)
    if detrend:
        data = clean(data.T, t_r=tr).T
        v = (-2, 2)
    # If subplot is not defined
    if subplot is None:
        subplot = mgs.GridSpec(1, 1)[0]

    # Define nested GridSpec
    wratios = [1, 100, 20]
    gs = mgs.GridSpecFromSubplotSpec(1, 2 + int(legend), subplot_spec=subplot,
                                     width_ratios=wratios[:2 + int(legend)],
                                     wspace=0.0)
    # Carpet plot
    ax1 = plt.subplot(gs[1])
    ax1.imshow(data, interpolation='nearest', aspect='auto', cmap='gray',
               vmin=v[0], vmax=v[1])
    ax1.grid(False)
    ax1.set_yticks([])
    ax1.set_yticklabels([])

    # Set 10 frame markers in X axis
    interval = max((int(data.shape[-1] + 1) //
                    10, int(data.shape[-1] + 1) // 5, 1))
    xticks = list(range(0, data.shape[-1])[::interval])
    ax1.set_xticks(xticks)
    if notr:
        ax1.set_xlabel('time (frame #)')
    else:
        ax1.set_xlabel('time (s)')
    labels = tr * (np.array(xticks)) * t_dec
    ax1.set_xticklabels(['%.02f' % t for t in labels.tolist()], fontsize=10)

    # Remove and redefine spines
    for side in ["top", "right"]:
        ax1.spines[side].set_color('none')
        ax1.spines[side].set_visible(False)

    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
    ax1.spines["bottom"].set_visible(False)
    ax1.spines["left"].set_color('none')
    ax1.spines["left"].set_visible(False)
    if output_file is not None:
        figure = plt.gcf()
        figure.savefig(output_file, bbox_inches='tight')
        plt.close(figure)
        figure = None
        return output_file

    return [ax1], gs

    

def confoundplot(tseries, gs_ts, gs_dist=None, name=None,
                 units=None, tr=None, hide_x=True, color='b', nskip=0,
                 cutoff=None, ylims=None):
    '''
    adapted from niworkflows
    tseries: 
       numpy array
    gs_ts:
       GridSpec
    name:
      file name
    units:
      tseries unit
    tr:
      repetition time
    '''

    # Define TR and number of frames
    notr = False
    if tr is None:
        notr = True
        tr = 1.
    ntsteps = len(tseries)
    tseries = np.array(tseries)

    # Define nested GridSpec
    gs = mgs.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_ts,
                                     width_ratios=[1, 100], wspace=0.0)

    ax_ts = plt.subplot(gs[1])
    ax_ts.grid(False)

    # Set 10 frame markers in X axis
    interval = max((ntsteps // 10, ntsteps // 5, 1))
    xticks = list(range(0, ntsteps)[::interval])
    ax_ts.set_xticks(xticks)

    if not hide_x:
        if notr:
            ax_ts.set_xlabel('time (frame #)')
        else:
            ax_ts.set_xlabel('time (s)')
            labels = tr * np.array(xticks)
            ax_ts.set_xticklabels(['%.02f' % t for t in labels.tolist()])
    else:
        ax_ts.set_xticklabels([])

    if name is not None:
        if units is not None:
            name += ' [%s]' % units

        ax_ts.annotate(
            name, xy=(0.0, 0.7), xytext=(0, 0), xycoords='axes fraction',
            textcoords='offset points', va='center', ha='left',
            color=color, size=20,
            bbox={'boxstyle': 'round', 'fc': 'w', 'ec': 'none',
                  'color': 'none', 'lw': 0, 'alpha': 0.8})

    for side in ["top", "right"]:
        ax_ts.spines[side].set_color('none')
        ax_ts.spines[side].set_visible(False)

    if not hide_x:
        ax_ts.spines["bottom"].set_position(('outward', 20))
        ax_ts.xaxis.set_ticks_position('bottom')
    else:
        ax_ts.spines["bottom"].set_color('none')
        ax_ts.spines["bottom"].set_visible(False)

    # ax_ts.spines["left"].set_position(('outward', 30))
    ax_ts.spines["left"].set_color('none')
    ax_ts.spines["left"].set_visible(False)
    # ax_ts.yaxis.set_ticks_position('left')

    ax_ts.set_yticks([])
    ax_ts.set_yticklabels([])

    nonnan = tseries[~np.isnan(tseries)]
    if nonnan.size > 0:
        # Calculate Y limits
        valrange = (nonnan.max() - nonnan.min())
        def_ylims = [nonnan.min() - 0.1 * valrange,
                     nonnan.max() + 0.1 * valrange]
        if ylims is not None:
            if ylims[0] is not None:
                def_ylims[0] = min([def_ylims[0], ylims[0]])
            if ylims[1] is not None:
                def_ylims[1] = max([def_ylims[1], ylims[1]])

        # Add space for plot title and mean/SD annotation
        def_ylims[0] -= 0.1 * (def_ylims[1] - def_ylims[0])

        ax_ts.set_ylim(def_ylims)

        # Annotate stats
        maxv = nonnan.max()
        mean = nonnan.mean()
        stdv = nonnan.std()
        p95 = np.percentile(nonnan, 95.0)
    else:
        maxv = 0
        mean = 0
        stdv = 0
        p95 = 0

    stats_label = (r'max: {max:.3f}{units} $\bullet$ mean: {mean:.3f}{units} '
                   r'$\bullet$ $\sigma$: {sigma:.3f}').format(
        max=maxv, mean=mean, units=units or '', sigma=stdv)
    ax_ts.annotate(
        stats_label, xy=(0.98, 0.7), xycoords='axes fraction',
        xytext=(0, 0), textcoords='offset points',
        va='center', ha='right', color=color, size=10,
        bbox={'boxstyle': 'round', 'fc': 'w', 'ec': 'none', 'color': 'none',
              'lw': 0, 'alpha': 0.8}
    )

    # Annotate percentile 95
    ax_ts.plot((0, ntsteps - 1), [p95] * 2, linewidth=.1, color='lightgray')
    ax_ts.annotate(
        '%.2f' % p95, xy=(0, p95), xytext=(-1, 0),
        textcoords='offset points', va='center', ha='right',
        color='lightgray', size=3)

    if cutoff is None:
        cutoff = []

    for thr in enumerate(cutoff):
        ax_ts.plot((0, ntsteps - 1), [thr] * 2,
                   linewidth=.2, color='dimgray')

        ax_ts.annotate(
            '%.2f' % thr, xy=(0, thr), xytext=(-1, 0),
            textcoords='offset points', va='center', ha='right',
            color='dimgray', size=3)

    ax_ts.plot(tseries, color=color, linewidth=1.5)
    ax_ts.set_xlim((0, ntsteps - 1))

    if gs_dist is not None:
        ax_dist = plt.subplot(gs_dist)
        sns.distplot(tseries, vertical=True, ax=ax_dist)
        ax_dist.set_xlabel('Timesteps')
        ax_dist.set_ylim(ax_ts.get_ylim())
        ax_dist.set_yticklabels([])

        return [ax_ts, ax_dist], gs
    return ax_ts, gs


# for executive summmary report
# Azeez Adebimpe, 2021
def plotseries(conf,gs_ts,ylim=None,ylabelx=None,hide_x=None,tr=None,ax=None):
    colums =conf.columns
    notr = False
    if tr is None:
        notr = True
        tr = 1.
        
    xtick = np.linspace(0,conf.shape[0]*tr,num=conf.shape[0])
    plt.style.use('seaborn-white')
    plt.xticks(color='k')
    plt.yticks(color='k')
    gs = mgs.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_ts,
                                     width_ratios=[1, 100], wspace=0.0)

    ax= plt.subplot(gs[1])
    ax.grid(False)
    for k in colums:
        ax.plot(xtick,conf[k],label=k,linewidth=2)
    if ylim:
        ax.set_ylim(ylim)
    else: 
        ax.set_ylim([-2*conf[k].max(),2*conf[k].max()])
    ax.set_ylabel(ylabelx,fontsize=20)    
    ax.legend(fontsize=20)
    
    last = conf.shape[0] - 1
    ax.set_xlim(0, last)
    xticks = list(range(0, last)) + [last] if not hide_x else []
    ax.set_xticks(xticks)

    if not hide_x:
        if tr is None:
            ax.set_xlabel("time (frame #)")
        else:
            ax.set_xlabel("time (s)")
            ax.set_xticklabels(["%.01f" % t for t in (tr * np.array(xticks)).tolist()])
   
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)
    
    return ax


def plot_carpetx(
    func,
    segfile,
    lut=None,
    tr=None,
    subplot=None,
    detrend=None,
    legend=True,
    size=(1800,950),
  ):
    
    
    img = nb.load(func)
    if segfile:
        atlaslabels = np.asanyarray(nb.load(segfile).dataobj)

    if isinstance(img, nb.Cifti2Image):
        assert (
            img.nifti_header.get_intent()[0] == "ConnDenseSeries"
        ), "Not a dense timeseries"

        data = img.get_fdata().T
        matrix = img.header.matrix
        struct_map = {
            "LEFT_CORTEX": 1,
            "RIGHT_CORTEX": 2,
            "SUBCORTICAL": 3,
            "CEREBELLUM": 4,
        }
        seg = np.zeros((data.shape[0],), dtype="uint32")
        for bm in matrix.get_index_map(1).brain_models:
            if "CORTEX" in bm.brain_structure:
                lidx = (1, 2)["RIGHT" in bm.brain_structure]
            elif "CEREBELLUM" in bm.brain_structure:
                lidx = 4
            else:
                lidx = 3
            index_final = bm.index_offset + bm.index_count
            seg[bm.index_offset:index_final] = lidx
        assert len(seg[seg < 1]) == 0, "Unassigned labels"

        # Decimate data
        data, seg = _decimate_data(data, seg, size)
        # preserve as much continuity as possible
        order = seg.argsort(kind="stable")

        cmap = ListedColormap([cm.get_cmap("Paired").colors[i] for i in (1, 0, 7, 3)])
        assert len(cmap.colors) == len(
            struct_map
        ), "Mismatch between expected # of structures and colors"

        # ensure no legend for CIFTI
        legend = False

    else:  # Volumetric NIfTI
        img_nii = check_niimg_4d(img, dtype="auto",)
        func_data = _safe_get_data(img_nii, ensure_finite=True)
        ntsteps = func_data.shape[-1]
        data = func_data[atlaslabels > 0].reshape(-1, ntsteps)
        oseg = atlaslabels[atlaslabels > 0].reshape(-1)

        # Map segmentation
        if lut is None:
            lut = np.zeros((256,), dtype="int")
            lut[1:11] = 1
            lut[255] = 2
            lut[30:99] = 3
            lut[100:201] = 4
        # Apply lookup table
        seg = lut[oseg.astype(int)]

        # Decimate data
        data, seg = _decimate_data(data, seg, size)
        # Order following segmentation labels
        order = np.argsort(seg)[::-1]
        # Set colormap
        cmap = ListedColormap(cm.get_cmap("tab10").colors[:4][::-1])
    
    """Common carpetplot building code for volumetric / CIFTI plots"""
    notr = False
    if tr is None:
        notr = True
        tr = 1.0

    # Detrend data
    v = (None, None)
    if detrend:
        data = clean(data.T, t_r=tr).T
        v = (-2, 2)

    # If subplot is not defined
    if subplot is None:
        subplot = mgs.GridSpec(1, 1)[0]

    # Define nested GridSpec
    wratios = [1, 100, 0]
    gs = mgs.GridSpecFromSubplotSpec(
        1,
        2 + int(legend),
        subplot_spec=subplot,
        width_ratios=wratios[: 2 + int(legend)],
        wspace=0.0,
    )

    # Segmentation colorbar
    ax0 = plt.subplot(gs[0])
    ax0.set_yticks([])
    ax0.set_xticks([])
    ax0.imshow(seg[order, np.newaxis], interpolation="none", aspect="auto", cmap=cmap)

    ax0.grid(False)
    ax0.spines["left"].set_visible(False)
    ax0.spines["bottom"].set_color("none")
    ax0.spines["bottom"].set_visible(False)

    # Carpet plot
    ax1 = plt.subplot(gs[1])
    ax1.imshow(
        data[order],
        interpolation="nearest",
        aspect="auto",
        cmap="gray",
        vmin=v[0],
        vmax=v[1],
    )

    ax1.grid(False)
    ax1.set_yticks([])
    ax1.set_yticklabels([])
    ax1.set_xticks([])

    if func.endswith('nii.gz'):
        ax0.set_ylabel('Voxels \n Blue: Cortical GM, Orange: Subcortical GM, \n Green: Cerebellum, Red: CSF and WM',
                      fontsize=20)
    elif func.endswith('.dtseries.nii'):
        ax0.set_ylabel('Grayordinates\n Blue: Left Cortex, Cyan: Right Cortex, \n Orange: Subcortical, Green: Cerebellum',
                      fontsize=20)
    # Remove and redefine spines
    for side in ["top", "right"]:
        # Toggle the spine objects
        ax0.spines[side].set_color("none")
        ax0.spines[side].set_visible(False)
        ax1.spines[side].set_color("none")
        ax1.spines[side].set_visible(False)

    ax1.yaxis.set_ticks_position("left")
    ax1.xaxis.set_ticks_position("bottom")
    ax1.spines["bottom"].set_visible(False)
    ax1.spines["left"].set_color("none")
    ax1.spines["left"].set_visible(False)

    return (ax0, ax1), gs

def plot_svgx(rawdata,regdata,resddata,fd,filenamebf,filenameaf,mask=None,seg=None,tr=1,taskname='rest'):
    '''
    generate carpet plot with dvars, fd, and WB
    ------------
    rawdata:
       nifti or cifti
    regdata: 
      nifti or cifti after nuissance regression 
    resddata: 
      nifti or cifti after regression and filtering
    mask: 
         mask for nifti if available
    seg:
        3 tissues seg files 
    tr: 
        repetition times
    fd: 
      framewise displacement
    filenamebf: 
      output file svg before processing
    filenameaf: 
      output file svg after processing
    '''
    
    rxdata = compute_dvars(read_ndata(datafile=rawdata,maskfile=mask))
    rgdata = compute_dvars(read_ndata(datafile=regdata,maskfile=mask))
    rsdata = compute_dvars(read_ndata(datafile=resddata,maskfile=mask))
    
    conf = pd.DataFrame({'Pre reg': rxdata, 'Post reg': rgdata, 'Post all': rsdata})
    fdx = pd.DataFrame({'FD':np.loadtxt(fd)})
    
    rw = read_ndata(datafile=rawdata,maskfile=mask)
    rs = read_ndata(datafile=resddata,maskfile=mask)
    
    wbbf = pd.DataFrame({'Mean':np.nanmean(rw,axis=0),'Std':np.nanstd(rw,axis=0)})
    wbaf = pd.DataFrame({'Mean':np.nanmean(rs,axis=0),'Std':np.nanstd(rs,axis=0)})
    
    # plot filex
    
    
    figx = plt.figure(constrained_layout=False, figsize=(15,30))
    grid = mgs.GridSpec(4, 1, wspace=0.0, hspace=0.05,height_ratios=[1,1,1.5,1])
    plotseries(conf=conf,gs_ts=grid[0],tr=tr,ylabelx='DVARS',hide_x=True,ylim=[0,500])
    plotseries(conf=wbbf,gs_ts=grid[1],tr=tr,ylabelx='WB',hide_x=True,ylim=[-400,800])
    plot_carpetx(func=rawdata,seg=seg,tr=tr,subplot=grid[2])
    plotseries(conf=fdx,gs_ts=grid[3],tr=tr,ylabelx='FD',hide_x=False,ylim=[0,0.8])
    figx.savefig(filenamebf,bbox_inches="tight", pad_inches=None)
    
    plt.cla()
    plt.clf()
    
    figy = plt.figure(constrained_layout=False, figsize=(15,30))
   
    grid = mgs.GridSpec(4, 1, wspace=0.0, hspace=0.05,height_ratios=[1,1,1.5,1])
    plotseries(conf=conf,gs_ts=grid[0],tr=tr,ylabelx='DVARS',hide_x=True,ylim=[0,500])
    plotseries(conf=wbaf,gs_ts=grid[1],tr=tr,ylabelx='WB',hide_x=True,ylim=[-400,800])
    plot_carpetx(func=resddata,seg=seg,tr=tr,subplot=grid[2])
    plotseries(conf=fdx,gs_ts=grid[3],tr=tr,ylabelx='FD',hide_x=False,ylim=[0,0.8])
    figy.savefig(filenameaf,bbox_inches="tight", pad_inches=None)
    
    return filenamebf,filenameaf
