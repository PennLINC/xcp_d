# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Plotting tools."""
import os

import matplotlib.pyplot as plt
import nibabel as nb
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec as mgs
from matplotlib.colors import ListedColormap
from nilearn._utils import check_niimg_4d
from nilearn._utils.niimg import safe_get_data
from nilearn.signal import clean

from xcp_d.utils.bids import _get_tr
from xcp_d.utils.doc import fill_doc
from xcp_d.utils.qcmetrics import compute_dvars
from xcp_d.utils.write_save import read_ndata, write_ndata


def _decimate_data(data, seg_data, temporal_mask, size):
    """Decimate timeseries data.

    Parameters
    ----------
    data : ndarray
        2D array of timepoints and samples
    seg_data : ndarray
        1D array of samples
    temporal_mask : ndarray or None
        1D array of timepoints. May be None.
    size : tuple
        2 element for P/T decimation
    """
    # Decimate the data in the spatial dimension
    p_dec = 1 + data.shape[0] // size[0]
    if p_dec:
        data = data[::p_dec, :]
        seg_data = seg_data[::p_dec]

    # Decimate the data in the temporal dimension
    t_dec = 1 + data.shape[1] // size[1]
    if t_dec:
        data = data[:, ::t_dec]
        if temporal_mask is not None:
            temporal_mask = temporal_mask[::t_dec]

    return data, seg_data, temporal_mask


def plot_confounds(
    time_series,
    grid_spec_ts,
    gs_dist=None,
    name=None,
    units=None,
    TR=None,
    hide_x=True,
    color="b",
    cutoff=None,
    ylims=None,
):
    """Create a time series plot for confounds.

    Adapted from niworkflows.

    Parameters
    ----------
    time_series : numpy.ndarray
       Time series to plot in the figure.
    grid_spec_ts : GridSpec
       The GridSpec object in which the time series plot will be stored.
    name :
      file name
    units :
      time_series unit
    TR : float or None, optional
      Repetition time for the time series. Default is None.

    Returns
    -------
    time_series_axis
    grid_specification
    """
    # Define TR and number of frames
    no_repetition_time = False
    if TR is None:  # Set default Repetition Time
        no_repetition_time = True
        TR = 1.0
    ntsteps = len(time_series)
    time_series = np.array(time_series)

    # Define nested GridSpec
    grid_specification = mgs.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=grid_spec_ts, width_ratios=[1, 100], wspace=0.0
    )

    time_series_axis = plt.subplot(grid_specification[1])
    time_series_axis.grid(False)

    # Set 10 frame markers in X axis
    interval = max((ntsteps // 10, ntsteps // 5, 1))
    xticks = list(range(0, ntsteps)[::interval])
    time_series_axis.set_xticks(xticks)

    # Set x_axis
    if not hide_x:
        if no_repetition_time:
            time_series_axis.set_xlabel("time (frame #)")
        else:
            time_series_axis.set_xlabel("time (s)")
            labels = TR * np.array(xticks)
            time_series_axis.set_xticklabels([f"{t:.02f}" for t in labels.tolist()])
    else:
        time_series_axis.set_xticklabels([])

    if name is not None:
        if units is not None:
            name += f" [{units}]"
        #   Formatting
        time_series_axis.annotate(
            name,
            xy=(0.0, 0.7),
            xytext=(0, 0),
            xycoords="axes fraction",
            textcoords="offset points",
            va="center",
            ha="left",
            color=color,
            size=16,
            bbox={
                "boxstyle": "round",
                "fc": "w",
                "ec": "none",
                "color": "none",
                "lw": 0,
                "alpha": 0.8,
            },
        )
    for side in ["top", "right"]:
        time_series_axis.spines[side].set_color("none")
        time_series_axis.spines[side].set_visible(False)

    if not hide_x:
        time_series_axis.spines["bottom"].set_position(("outward", 20))
        time_series_axis.xaxis.set_ticks_position("bottom")
    else:
        time_series_axis.spines["bottom"].set_color("none")
        time_series_axis.spines["bottom"].set_visible(False)

    time_series_axis.spines["left"].set_color("none")
    time_series_axis.spines["left"].set_visible(False)
    time_series_axis.set_yticks([])
    time_series_axis.set_yticklabels([])

    nonnan = time_series[~np.isnan(time_series)]
    if nonnan.size > 0:
        # Calculate Y limits
        valrange = nonnan.max() - nonnan.min()
        def_ylims = [nonnan.min() - 0.1 * valrange, nonnan.max() + 0.1 * valrange]
        if ylims is not None:
            if ylims[0] is not None:
                def_ylims[0] = min([def_ylims[0], ylims[0]])
            if ylims[1] is not None:
                def_ylims[1] = max([def_ylims[1], ylims[1]])

        # Add space for plot title and mean/SD annotation
        def_ylims[0] -= 0.1 * (def_ylims[1] - def_ylims[0])

        time_series_axis.set_ylim(def_ylims)

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

    stats_label = (
        r"max: {max:.3f}{units} $\bullet$ mean: {mean:.3f}{units} "
        r"$\bullet$ $\sigma$: {sigma:.3f}"
    ).format(max=maxv, mean=mean, units=units or "", sigma=stdv)
    time_series_axis.annotate(
        stats_label,
        xy=(0.98, 0.7),
        xycoords="axes fraction",
        xytext=(0, 0),
        textcoords="offset points",
        va="center",
        ha="right",
        color=color,
        size=14,
        bbox={
            "boxstyle": "round",
            "fc": "w",
            "ec": "none",
            "color": "none",
            "lw": 0,
            "alpha": 0.8,
        },
    )

    # Annotate percentile 95
    time_series_axis.plot((0, ntsteps - 1), [p95] * 2, linewidth=0.1, color="lightgray")
    time_series_axis.annotate(
        f"{p95:.2f}",
        xy=(0, p95),
        xytext=(-1, 0),
        textcoords="offset points",
        va="center",
        ha="right",
        color="lightgray",
        size=3,
    )

    if cutoff is None:
        cutoff = []

    for threshold in enumerate(cutoff):
        time_series_axis.plot((0, ntsteps - 1), [threshold] * 2, linewidth=0.2, color="dimgray")

        time_series_axis.annotate(
            f"{threshold:.2f}",
            xy=(0, threshold),
            xytext=(-1, 0),
            textcoords="offset points",
            va="center",
            ha="right",
            color="dimgray",
            size=3,
        )

    time_series_axis.plot(time_series, color=color, linewidth=2.5)
    time_series_axis.set_xlim((0, ntsteps - 1))

    # Plotting
    if gs_dist is not None:
        ax_dist = plt.subplot(gs_dist)
        sns.distplot(time_series, vertical=True, ax=ax_dist)
        ax_dist.set_xlabel("Timesteps")
        ax_dist.set_ylim(time_series_axis.get_ylim())
        ax_dist.set_yticklabels([])

        return [time_series_axis, ax_dist], grid_specification

    return time_series_axis, grid_specification


def plot_dvars_es(time_series, ax, run_index=None):
    """Create DVARS plot for the executive summary."""
    ax.grid(False)

    ntsteps = time_series.shape[0]

    # Set 10 frame markers in X axis
    interval = max((ntsteps // 10, ntsteps // 5, 1))
    xticks = list(range(0, ntsteps)[::interval])
    ax.set_xticks(xticks)
    ax.set_xticklabels([])

    # Set y-axis labels
    ax.set_ylabel("DVARS")

    columns = time_series.columns
    maximum_values = []
    minimum_values = []

    colors = {
        "Pre regression": "#68AC57",
        "Post all": "#EF8532",
    }
    for c in columns:
        color = colors[c]
        ax.plot(time_series[c], label=c, linewidth=2, alpha=1, color=color)
        maximum_values.append(max(time_series[c]))
        minimum_values.append(min(time_series[c]))

    if run_index is not None:
        for run_location in run_index:
            ax.axvline(run_location, color="black", linestyle="--")

    # Set limits and format
    minimum_x_value = [abs(x) for x in minimum_values]

    ax.set_xlim((0, ntsteps - 1))

    ax.legend(fontsize=30)
    ax.set_ylim([-1.5 * max(minimum_x_value), 1.5 * max(maximum_values)])

    for item in (
        [ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()
    ):
        item.set_fontsize(30)

    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(4)
    sns.despine()

    return ax


def plot_global_signal_es(time_series, ax, run_index=None):
    """Create global signal plot for the executive summary."""
    ntsteps = time_series.shape[0]

    ax.grid(False)

    # Set 10 frame markers in X axis
    interval = max((ntsteps // 10, ntsteps // 5, 1))
    xticks = list(range(0, ntsteps)[::interval])
    ax.set_xticks(xticks)
    ax.set_xticklabels([])

    # Set y-axis labels
    ax.set_ylabel("WB")

    # Plot the whole brain mean and std.
    # Mean scale on the left, std scale on the right.
    mean_line = ax.plot(
        time_series["Mean"],
        label="Mean",
        linewidth=2,
        alpha=1,
        color="#D1352B",
    )
    ax_right = ax.twinx()
    ax_right.set_ylabel("Standard Deviation")
    std_line = ax_right.plot(
        time_series["Std"],
        label="Std",
        linewidth=2,
        alpha=1,
        color="#497DB3",
    )

    std_mean = np.mean(time_series["Std"])
    ax_right.set_ylim(
        (1.5 * np.min(time_series["Std"] - std_mean)) + std_mean,
        (1.5 * np.max(time_series["Std"] - std_mean)) + std_mean,
    )
    ax_right.yaxis.label.set_fontsize(30)
    for item in ax_right.get_yticklabels():
        item.set_fontsize(30)

    lines = mean_line + std_line
    line_labels = [line.get_label() for line in lines]
    ax.legend(lines, line_labels, fontsize=30)

    if run_index is not None:
        for run_location in run_index:
            ax.axvline(run_location, color="black", linestyle="--")

    ax.set_xlim((0, ntsteps - 1))

    mean_mean = np.mean(time_series["Mean"])
    ax.set_ylim(
        (1.5 * np.min(time_series["Mean"] - mean_mean)) + mean_mean,
        (1.5 * np.max(time_series["Mean"] - mean_mean)) + mean_mean,
    )

    for item in (
        [ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()
    ):
        item.set_fontsize(30)

    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(4)
    sns.despine()

    return ax


def plot_framewise_displacement_es(
    time_series,
    ax,
    TR,
    run_index=None,
):
    """Create framewise displacement plot for the executive summary."""
    ntsteps = time_series.shape[0]
    ax.grid(axis="y")

    # Set 10 frame markers in X axis
    interval = max((ntsteps // 10, ntsteps // 5, 1))
    xticks = list(range(0, ntsteps)[::interval])
    ax.set_xticks(xticks)

    # Set the x-axis labels based on time, not index
    ax.set_xlabel("Time (s)")
    labels = TR * np.array(xticks)
    labels = labels.astype(int)
    ax.set_xticklabels(labels)

    # Set y-axis labels
    ax.set_ylabel("FD (mm)")
    ax.plot(time_series, label="FD", linewidth=3, color="black")

    # Threshold fd at 0.1, 0.2 and 0.5 and plot
    # Plot zero line
    fd_dots = time_series.copy()  # dots in line with FD time series
    fd_line = time_series.copy()  # dots on top of axis
    top_line = 0.8
    ymax = 0.85

    fd_dots[fd_dots < 0] = np.nan

    THRESHOLDS = [0.05, 0.1, 0.2, 0.5]
    COLORS = ["#969696", "#377C21", "#EF8532", "#EB392A"]
    for i_thresh, threshold in enumerate(THRESHOLDS):
        color = COLORS[i_thresh]

        ax.axhline(
            y=threshold,
            color=color,
            linestyle="-",
            linewidth=3,
            alpha=1,
        )

        fd_dots[fd_dots < threshold] = np.nan
        ax.plot(fd_dots, ".", color=color, markersize=10)

        fd_line = time_series.copy()
        fd_line[fd_line >= threshold] = top_line
        fd_line[fd_line < threshold] = np.nan
        ax.plot(fd_line, ".", color=color, markersize=10)

        # Plot the good volumes, i.e: thresholded at 0.1, 0.2, 0.5
        good_vols = len(time_series[time_series < threshold])
        ax.text(
            1.01,
            threshold / ymax,
            good_vols,
            c=color,
            verticalalignment="center",
            horizontalalignment="left",
            transform=ax.transAxes,
            fontsize=20,
        )

    # Log the total number of volumes as well
    ax.text(
        1.01,
        top_line / ymax,
        time_series.size,
        c="black",
        verticalalignment="center",
        horizontalalignment="left",
        transform=ax.transAxes,
        fontsize=20,
    )

    if run_index is not None:
        # FD plots use time series index, not time, as x-axis
        for run_location in run_index:
            ax.axvline(run_location, color="black", linestyle="--")

    ax.set_xlim((0, ntsteps - 1))
    ax.set_ylim(0, ymax)
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

    for item in (
        [ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()
    ):
        item.set_fontsize(30)

    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(4)
    sns.despine()

    return ax


@fill_doc
def plot_fmri_es(
    *,
    preprocessed_bold,
    denoised_interpolated_bold,
    TR,
    filtered_motion,
    temporal_mask,
    preprocessed_figure,
    denoised_figure,
    standardize,
    temporary_file_dir,
    mask=None,
    seg_data=None,
    run_index=None,
):
    """Generate carpet plot with DVARS, FD, and WB for the executive summary.

    Parameters
    ----------
    preprocessed_bold : :obj:`str`
        Preprocessed BOLD file, dummy scan removal.
    %(denoised_interpolated_bold)s
    %(TR)s
    %(filtered_motion)s
    %(temporal_mask)s
        Only non-outlier (low-motion) volumes in the temporal mask will be used to scale
        the carpet plot.
    preprocessed_figure : :obj:`str`
        output file svg before processing
    denoised_figure : :obj:`str`
        output file svg after processing
    standardize : :obj:`bool`
        Whether to standardize the data or not.
        If False, then the preferred DCAN version of the plot will be generated,
        where the BOLD data are not rescaled, and the carpet plot has color limits from the
        2.5th percentile to the 97.5th percentile.
        If True, then the BOLD data will be z-scored and the color limits will be -2 and 2.
    temporary_file_dir : :obj:`str`
        Path in which to store temporary files.
    mask : :obj:`str`, optional
        Brain mask file. Used only when the pre- and post-processed BOLD data are NIFTIs.
    seg_data : :obj:`str`, optional
        Three-tissue segmentation file. This is only used for NIFTI inputs.
        With CIFTI inputs, the tissue types are inferred directly from the CIFTI file.
    run_index : None or array_like, optional
        An index indicating splits between runs, for concatenated data.
        If not None, this should be an array/list of integers, indicating the volumes.
    """
    # Compute dvars correctly if not already done
    preprocessed_arr = read_ndata(datafile=preprocessed_bold, maskfile=mask)
    denoised_interpolated_arr = read_ndata(datafile=denoised_interpolated_bold, maskfile=mask)

    preprocessed_dvars = compute_dvars(datat=preprocessed_arr)[1]
    denoised_interpolated_dvars = compute_dvars(datat=denoised_interpolated_arr)[1]

    if preprocessed_arr.shape != denoised_interpolated_arr.shape:
        raise ValueError(
            "Shapes do not match:\n"
            f"\t{preprocessed_bold}: {preprocessed_arr.shape}\n"
            f"\t{denoised_interpolated_bold}: {denoised_interpolated_arr.shape}\n\n"
        )

    # Create dataframes for the bold_data DVARS, FD
    dvars_regressors = pd.DataFrame(
        {
            "Pre regression": preprocessed_dvars,
            "Post all": denoised_interpolated_dvars,
        }
    )

    fd_regressor = pd.read_table(filtered_motion)["framewise_displacement"].values
    if temporal_mask:
        tmask_arr = pd.read_table(temporal_mask)["framewise_displacement"].values.astype(bool)
    else:
        tmask_arr = np.zeros(fd_regressor.shape, dtype=bool)

    # The mean and standard deviation of the preprocessed data,
    # after mean-centering and detrending.
    preprocessed_timeseries = pd.DataFrame(
        {
            "Mean": np.nanmean(preprocessed_arr, axis=0),
            "Std": np.nanstd(preprocessed_arr, axis=0),
        }
    )

    # The mean and standard deviation of the denoised data, with bad volumes included.
    denoised_interpolated_timeseries = pd.DataFrame(
        {
            "Mean": np.nanmean(denoised_interpolated_arr, axis=0),
            "Std": np.nanstd(denoised_interpolated_arr, axis=0),
        }
    )

    atlaslabels = None
    if seg_data is not None:
        atlaslabels = nb.load(seg_data).get_fdata()

    rm_temp_file = False
    temp_preprocessed_file = preprocessed_bold
    if not standardize:
        # The plot going to carpet plot will be mean-centered and detrended,
        # but will not otherwise be rescaled.
        detrended_preprocessed_arr = clean(
            preprocessed_arr.T,
            t_r=TR,
            detrend=True,
            filter=False,
            standardize=False,
        ).T

        # Make a temporary file for niftis and ciftis
        rm_temp_file = True
        if preprocessed_bold.endswith(".nii.gz"):
            temp_preprocessed_file = os.path.join(temporary_file_dir, "filex_raw.nii.gz")
        else:
            temp_preprocessed_file = os.path.join(temporary_file_dir, "filex_raw.dtseries.nii")

        # Write out the scaled data
        temp_preprocessed_file = write_ndata(
            data_matrix=detrended_preprocessed_arr,
            template=preprocessed_bold,
            filename=temp_preprocessed_file,
            mask=mask,
            TR=TR,
        )

    files_for_carpet = [temp_preprocessed_file, denoised_interpolated_bold]
    figure_names = [preprocessed_figure, denoised_figure]
    data_arrays = [preprocessed_timeseries, denoised_interpolated_timeseries]
    for i_fig, figure_name in enumerate(figure_names):
        file_for_carpet = files_for_carpet[i_fig]
        data_arr = data_arrays[i_fig]

        # Plot the data and confounds, plus the carpet plot
        plt.cla()
        plt.clf()

        fig = plt.figure(constrained_layout=False, figsize=(22.5, 30))
        grid = fig.add_gridspec(
            nrows=4,
            ncols=1,
            wspace=0.0,
            hspace=0.1,
            height_ratios=[1, 1, 2.5, 1.3],
        )

        # The DVARS plot in the first row
        gridspec0 = mgs.GridSpecFromSubplotSpec(
            1,
            3,
            subplot_spec=grid[0],
            width_ratios=[1, 100, 3],
            wspace=0.0,
        )
        ax0 = plt.subplot(gridspec0[1])
        plot_dvars_es(dvars_regressors, ax0, run_index=run_index)

        # The WB plot in the second row
        gridspec1 = mgs.GridSpecFromSubplotSpec(
            1,
            3,
            subplot_spec=grid[1],
            width_ratios=[1, 100, 3],
            wspace=0.0,
        )
        ax1 = plt.subplot(gridspec1[1])
        plot_global_signal_es(data_arr, ax1, run_index=run_index)

        # The carpet plot in the third row
        plot_carpet(
            func=file_for_carpet,
            atlaslabels=atlaslabels,
            standardize=standardize,  # Data are already detrended if standardize is False
            size=(950, 800),
            labelsize=30,
            subplot=grid[2],  # Use grid for now.
            output_file=None,
            TR=TR,
            colorbar=True,
            lut=None,
            temporal_mask=tmask_arr,
        )

        # The FD plot at the bottom
        gridspec3 = mgs.GridSpecFromSubplotSpec(
            1,
            3,
            subplot_spec=grid[3],
            width_ratios=[1, 100, 3],
            wspace=0.0,
        )
        ax3 = plt.subplot(gridspec3[1])
        plot_framewise_displacement_es(fd_regressor, ax3, TR=TR, run_index=run_index)

        # Save out the before processing file
        fig.savefig(figure_name, bbox_inches="tight", pad_inches=None, dpi=300)
        plt.close(fig)

    # Remove temporary files
    if rm_temp_file:
        os.remove(temp_preprocessed_file)

    # Save out the after processing file
    return preprocessed_figure, denoised_figure


@fill_doc
class FMRIPlot:
    """Generates the fMRI Summary Plot.

    Parameters
    ----------
    func_file
    mask_file
    data
    confound_file
    seg_file
    %(TR)s
    usecols
    units
    vlines
    spikes_files
    """

    __slots__ = ("func_file", "mask_data", "TR", "seg_data", "confounds", "spikes")

    def __init__(
        self,
        func_file,
        mask_file=None,
        data=None,
        confound_file=None,
        seg_file=None,
        TR=None,
        usecols=None,
        units=None,
        vlines=None,
        spikes_files=None,
    ):
        #  Load in the necessary information
        func_img = nb.load(func_file)
        self.func_file = func_file
        self.TR = TR or _get_tr(func_img)
        self.mask_data = None
        self.seg_data = None

        if not isinstance(func_img, nb.Cifti2Image):  # If Nifti
            self.mask_data = nb.fileslice.strided_scalar(func_img.shape[:3], np.uint8(1))
            if mask_file:
                self.mask_data = np.asanyarray(nb.load(mask_file).dataobj).astype("uint8")
            if seg_file:
                self.seg_data = np.asanyarray(nb.load(seg_file).dataobj)

        if units is None:
            units = {}
        if vlines is None:
            vlines = {}
        self.confounds = {}
        if data is None and confound_file:
            data = pd.read_csv(confound_file, sep=r"[\t\s]+", usecols=usecols, index_col=False)
        # Confounds information
        if data is not None:
            for name in data.columns.ravel():
                self.confounds[name] = {
                    "values": data[[name]].values.ravel().tolist(),
                    "units": units.get(name),
                    "cutoff": vlines.get(name),
                }
        #  Spike information
        self.spikes = []
        if spikes_files:
            for sp_file in spikes_files:
                self.spikes.append((np.loadtxt(sp_file), None, False))

    def plot(self, labelsize, figure=None):
        """Perform main plotting step."""
        # Layout settings
        sns.set_context("paper", font_scale=1)

        if figure is None:
            figure = plt.gcf()

        n_confounds = len(self.confounds)
        n_spikes = len(self.spikes)
        n_rows = 1 + n_confounds + n_spikes

        # Create grid specification
        grid = mgs.GridSpec(
            n_rows, 1, wspace=0.0, hspace=0.05, height_ratios=[1] * (n_rows - 1) + [5]
        )

        grid_id = 0
        for _, name, _ in self.spikes:
            # RF: What is this?
            # spikesplot(tsz,
            #            title=name,
            #            outer_gs=grid[grid_id],
            #            TR=self.TR,
            #            zscored=iszs)
            grid_id += 1

        # Plot confounds
        if self.confounds:
            from seaborn import color_palette

            palette = color_palette("husl", n_confounds)

        for i, (name, kwargs) in enumerate(self.confounds.items()):
            time_series = kwargs.pop("values")
            plot_confounds(
                time_series, grid[grid_id], TR=self.TR, color=palette[i], name=name, **kwargs
            )
            grid_id += 1

        # Carpet plot
        plot_carpet(
            func=self.func_file,
            atlaslabels=self.seg_data,
            standardize=True,
            size=(950, 800),
            labelsize=labelsize,
            subplot=grid[-1],
            output_file=None,
            TR=self.TR,
            colorbar=False,
            lut=None,
            temporal_mask=None,
        )
        return figure


def plot_carpet(
    *,
    func,
    atlaslabels,
    TR,
    standardize,
    temporal_mask=None,
    size=(950, 800),
    labelsize=30,
    subplot=None,
    lut=None,
    colorbar=False,
    output_file=None,
):
    """Plot an image representation of voxel intensities across time.

    This is also known as the "carpet plot" or "Power plot".
    See Jonathan Power Neuroimage 2017 Jul 1; 154:150-158.

    Parameters
    ----------
    func : :obj:`str`
        Path to NIfTI or CIFTI BOLD image
    atlaslabels : numpy.ndarray, optional
        A 3D array of integer labels from an atlas, resampled into ``img`` space.
        Required if ``func`` is a NIfTI image.
        Unused if ``func`` is a CIFTI.
    standardize : bool, optional
        Detrend and standardize the data prior to plotting.
    size : tuple, optional
        Size of figure.
    labelsize : int, optional
    subplot : matplotlib Subplot, optional
        Subplot to plot figure on.
    output_file : :obj:`str` or None, optional
        The name of an image file to export the plot to. Valid extensions
        are .png, .pdf, .svg. If output_file is not None, the plot
        is saved to a file, and the display is closed.
    legend : bool
        Whether to render the average functional series with ``atlaslabels`` as overlay.
    TR : float, optional
        Specify the TR, if specified it uses this value.
        If left as None, # of frames is plotted instead of time.
    lut : numpy.ndarray, optional
        Look up table for segmentations
    colorbar : bool, optional
        Default is False.
    """
    img = nb.load(func)

    if isinstance(img, nb.Cifti2Image):  # CIFTI
        assert (
            img.nifti_header.get_intent()[0] == "ConnDenseSeries"
        ), f"Not a dense timeseries: {img.nifti_header.get_intent()[0]}, {func}"

        # Get required information
        data = img.get_fdata().T
        matrix = img.header.matrix
        struct_map = {
            "LEFT_CORTEX": 1,
            "RIGHT_CORTEX": 2,
            "SUBCORTICAL": 3,
            "CEREBELLUM": 4,
        }
        seg_data = np.zeros((data.shape[0],), dtype="uint32")
        # Get brain model information
        for brain_model in matrix.get_index_map(1).brain_models:
            if "CORTEX" in brain_model.brain_structure:
                lidx = (1, 2)["RIGHT" in brain_model.brain_structure]
            elif "CEREBELLUM" in brain_model.brain_structure:
                lidx = 4
            else:
                lidx = 3
            index_final = brain_model.index_offset + brain_model.index_count
            seg_data[brain_model.index_offset : index_final] = lidx
        assert len(seg_data[seg_data < 1]) == 0, "Unassigned labels"

    else:  # Volumetric NIfTI
        img_nii = check_niimg_4d(img, dtype="auto")  # Check the image is in nifti format
        func_data = safe_get_data(img_nii, ensure_finite=True)
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
        seg_data = lut[oseg.astype(int)]

    # Decimate data
    data, seg_data, temporal_mask = _decimate_data(data, seg_data, temporal_mask, size)

    if isinstance(img, nb.Cifti2Image):
        # Preserve continuity
        order = seg_data.argsort(kind="stable")
        # Get color maps
        cmap = ListedColormap([plt.get_cmap("Paired").colors[i] for i in (1, 0, 7, 3)])
        assert len(cmap.colors) == len(
            struct_map
        ), "Mismatch between expected # of structures and colors"
    else:
        # Order following segmentation labels
        order = np.argsort(seg_data)[::-1]
        # Set colormap
        cmap = ListedColormap(plt.get_cmap("tab10").colors[:4][::-1])

    # Detrend and z-score data
    if standardize:
        # This does not account for the temporal mask.
        data = clean(data.T, t_r=TR, detrend=True, filter=False, standardize="zscore_sample").T
        vlimits = (-2, 2)
    elif temporal_mask is not None:
        # If standardize is False and a temporal mask is provided,
        # then we use only low-motion timepoints to define the vlimits.
        # The executive summary uses the following range for native BOLD units.
        vlimits = tuple(np.percentile(data[:, ~temporal_mask], q=(2.5, 97.5)))
    else:
        # If standardize is False, then the data are assumed to have native BOLD units.
        # The executive summary uses the following range for native BOLD units.
        vlimits = tuple(np.percentile(data, q=(2.5, 97.5)))

    # If subplot is not defined
    if subplot is None:
        subplot = mgs.GridSpec(1, 1)[0]

    # Define nested GridSpec
    if colorbar:
        wratios = [1, 100, 1, 2]
        grid_specification = mgs.GridSpecFromSubplotSpec(
            1,
            4,
            subplot_spec=subplot,
            width_ratios=wratios,
            wspace=0.0,
        )
        ax0 = plt.subplot(grid_specification[0])
        ax1 = plt.subplot(grid_specification[1])
        ax2 = plt.subplot(grid_specification[3])
    else:
        wratios = [1, 100]
        grid_specification = mgs.GridSpecFromSubplotSpec(
            1,
            2,
            subplot_spec=subplot,
            width_ratios=wratios,
            wspace=0.0,
        )
        ax0 = plt.subplot(grid_specification[0])
        ax1 = plt.subplot(grid_specification[1])
        ax2 = None

    # Segmentation colorbar
    ax0.set_xticks([])
    ax0.imshow(seg_data[order, np.newaxis], interpolation="none", aspect="auto", cmap=cmap)

    if func.endswith("nii.gz"):  # Nifti
        labels = ["Cortical GM", "Subcortical GM", "Cerebellum", "CSF and WM"]
    else:  # Cifti
        labels = ["Left Cortex", "Right Cortex", "Subcortical", "Cerebellum"]

    # Formatting the plot
    tick_locs = []
    for y in np.unique(seg_data[order]):
        tick_locs.append(np.argwhere(seg_data[order] == y).mean())

    ax0.set_yticks(tick_locs)
    ax0.set_yticklabels(labels, fontdict={"fontsize": labelsize}, rotation=0, va="center")
    ax0.grid(False)
    ax0.spines["left"].set_visible(False)
    ax0.spines["bottom"].set_color("none")
    ax0.spines["bottom"].set_visible(False)
    ax0.set_xticks([])
    ax0.set_xticklabels([])

    # Carpet plot
    pos = ax1.imshow(
        data[order],
        interpolation="nearest",
        aspect="auto",
        cmap="gray",
        vmin=vlimits[0],
        vmax=vlimits[1],
    )
    ax1.grid(False)
    ax1.set_yticks([])
    ax1.set_yticklabels([])
    ax1.set_xticks([])
    ax1.set_xticklabels([])

    if temporal_mask is not None:
        # Add color bands to the carpet plot corresponding to censored volumes
        outlier_idx = list(np.where(temporal_mask)[0])
        gaps = [
            [start, end] for start, end in zip(outlier_idx, outlier_idx[1:]) if start + 1 < end
        ]
        edges = iter(outlier_idx[:1] + sum(gaps, []) + outlier_idx[-1:])
        consecutive_outliers_idx = list(zip(edges, edges))
        for band in consecutive_outliers_idx:
            start = band[0] - 0.5
            end = band[1] + 0.5
            ax1.axvspan(start, end, color="red", alpha=0.5)

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

    # Use the last axis for a colorbar
    if colorbar:
        ax2.set_xticks([])
        ax2.set_yticks([])
        fig = ax2.get_figure()
        cbar = fig.colorbar(pos, cax=ax2, ticks=vlimits)
        cbar.ax.tick_params(size=0, labelsize=20)

    #  Write out file
    if output_file is not None:
        figure = plt.gcf()
        figure.savefig(output_file, bbox_inches="tight")
        plt.close(figure)
        figure = None
        return output_file

    return (ax0, ax1, ax2), grid_specification


def surf_data_from_cifti(data, axis, surf_name):
    """From https://neurostars.org/t/separate-cifti-by-structure-in-python/17301/2.

    https://nbviewer.org/github/neurohackademy/nh2020-curriculum/blob/master/\
    we-nibabel-markiewicz/NiBabel.ipynb
    """
    assert isinstance(axis, (nb.cifti2.BrainModelAxis, nb.cifti2.ParcelsAxis))
    if isinstance(axis, nb.cifti2.BrainModelAxis):
        for name, data_indices, model in axis.iter_structures():
            # Iterates over volumetric and surface structures
            if name == surf_name:  # Just looking for a surface
                data = data.T[data_indices]
                # Assume brainmodels axis is last, move it to front
                vtx_indices = model.vertex
                # Generally 1-N, except medial wall vertices
                surf_data = np.zeros((vtx_indices.max() + 1,) + data.shape[1:], dtype=data.dtype)
                surf_data[vtx_indices] = data
                return surf_data
    else:
        if surf_name not in axis.nvertices:
            raise ValueError(
                f"No structure named {surf_name}.\n\n"
                f"Available structures are {list(axis.name.keys())}"
            )
        nvertices = axis.nvertices[surf_name]
        surf_data = np.zeros(nvertices)
        for i_label in range(len(axis.name)):
            element_dict = axis.get_element(i_label)[2]
            if surf_name in element_dict:
                element_idx = element_dict[surf_name]
                surf_data[element_idx] = data[0, i_label]

        return surf_data

    raise ValueError(f"No structure named {surf_name}")


def plot_design_matrix(design_matrix, temporal_mask=None):
    """Plot design matrix TSV with Nilearn.

    NOTE: This is a Node function.

    Parameters
    ----------
    design_matrix : :obj:`str`
        Path to TSV file containing the design matrix.
    temporal_mask : :obj:`str`, optional
        Path to TSV file containing a list of volumes to censor.

    Returns
    -------
    design_matrix_figure : :obj:`str`
        Path to SVG figure file.
    """
    import os

    import numpy as np
    import pandas as pd
    from nilearn import plotting

    design_matrix_df = pd.read_table(design_matrix)
    if temporal_mask:
        censoring_df = pd.read_table(temporal_mask)
        n_motion_outliers = censoring_df["framewise_displacement"].sum()
        motion_outliers_df = pd.DataFrame(
            data=np.zeros((censoring_df.shape[0], n_motion_outliers), dtype=np.int16),
            columns=[f"outlier{i}" for i in range(1, n_motion_outliers + 1)],
        )
        motion_outlier_idx = np.where(censoring_df["framewise_displacement"])[0]
        for i_outlier, outlier_col in enumerate(motion_outliers_df.columns):
            outlier_row = motion_outlier_idx[i_outlier]
            motion_outliers_df.loc[outlier_row, outlier_col] = 1

        design_matrix_df = pd.concat(
            (design_matrix_df, motion_outliers_df),
            axis=1,
        )

    design_matrix_figure = os.path.abspath("design_matrix.svg")
    plotting.plot_design_matrix(design_matrix_df, output_file=design_matrix_figure)

    return design_matrix_figure
