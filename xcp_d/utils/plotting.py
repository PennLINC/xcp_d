# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Plotting tools."""
import os
import tempfile

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import nibabel as nb
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec as mgs
from matplotlib.colors import ListedColormap
from nilearn._utils import check_niimg_4d
from nilearn._utils.niimg import _safe_get_data
from nilearn.signal import clean

from xcp_d.utils.bids import _get_tr
from xcp_d.utils.doc import fill_doc
from xcp_d.utils.qcmetrics import compute_dvars
from xcp_d.utils.write_save import read_ndata, write_ndata


def _decimate_data(data, seg_data, size):
    """Decimate timeseries data.

    Parameters
    ----------
    data : ndarray
        2 element array of timepoints and samples
    seg_data : ndarray
        1 element array of samples
    size : tuple
        2 element for P/T decimation
    """
    p_dec = 1 + data.shape[0] // size[0]
    if p_dec:
        data = data[::p_dec, :]
        seg_data = seg_data[::p_dec]
    t_dec = 1 + data.shape[1] // size[1]
    if t_dec:
        data = data[:, ::t_dec]
    return data, seg_data


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
    sns.set_style("whitegrid")
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
    sns.set_style("whitegrid")

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
        "Post regression": "#8E549F",
        "Post all": "#EF8532",
    }
    for c in columns:
        color = colors[c]
        ax.plot(time_series[c], label=c, linewidth=2, alpha=1, color=color)
        maximum_values.append(max(time_series[c]))
        minimum_values.append(min(time_series[c]))

    if run_index:
        ax.axvline(run_index, color="yellow")

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
    sns.set_style("whitegrid")

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

    if run_index:
        ax.axvline(run_index, color="yellow")

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
    sns.set_style("whitegrid")

    ntsteps = time_series.shape[0]
    ax.grid(axis="y")

    # Set 10 frame markers in X axis
    interval = max((ntsteps // 10, ntsteps // 5, 1))
    xticks = list(range(0, ntsteps)[::interval])
    ax.set_xticks(xticks)

    # Set the x-axis labels
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

    if run_index:
        ax.axvline([idx * TR for idx in run_index], color="yellow")

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
    preprocessed_bold,
    uncensored_denoised_bold,
    interpolated_filtered_bold,
    TR,
    filtered_motion,
    preprocessed_bold_figure,
    denoised_bold_figure,
    standardize,
    mask=None,
    seg_data=None,
    run_index=None,
):
    """Generate carpet plot with DVARS, FD, and WB for the executive summary.

    Parameters
    ----------
    preprocessed_bold : :obj:`str`
        Preprocessed BOLD file, dummy scan removal.
    %(uncensored_denoised_bold)s
    %(interpolated_filtered_bold)s
    %(TR)s
    %(filtered_motion)s
    preprocessed_bold_figure : :obj:`str`
        output file svg before processing
    denoised_bold_figure : :obj:`str`
        output file svg after processing
    standardize : :obj:`bool`
        Whether to standardize the data or not.
        If False, then the preferred DCAN version of the plot will be generated,
        where the BOLD data are not rescaled, and the carpet plot has color limits of -600 and 600.
        If True, then the BOLD data will be z-scored and the color limits will be -2 and 2.
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
    preprocessed_bold_arr = read_ndata(datafile=preprocessed_bold, maskfile=mask)
    uncensored_denoised_bold_arr = read_ndata(datafile=uncensored_denoised_bold, maskfile=mask)
    filtered_denoised_bold_arr = read_ndata(datafile=interpolated_filtered_bold, maskfile=mask)

    preprocessed_bold_dvars = compute_dvars(preprocessed_bold_arr)
    uncensored_denoised_bold_dvars = compute_dvars(uncensored_denoised_bold_arr)
    filtered_denoised_bold_dvars = compute_dvars(filtered_denoised_bold_arr)

    if not (
        preprocessed_bold_dvars.shape
        == uncensored_denoised_bold_dvars.shape
        == filtered_denoised_bold_dvars.shape
    ):
        raise ValueError(
            "Shapes do not match:\n"
            f"\t{preprocessed_bold}: {preprocessed_bold_arr.shape}\n"
            f"\t{uncensored_denoised_bold}: {uncensored_denoised_bold_arr.shape}\n"
            f"\t{interpolated_filtered_bold}: {filtered_denoised_bold_arr.shape}\n\n"
        )

    if not (
        preprocessed_bold_arr.shape
        == uncensored_denoised_bold_arr.shape
        == filtered_denoised_bold_arr.shape
    ):
        raise ValueError(
            "Shapes do not match:\n"
            f"\t{preprocessed_bold}: {preprocessed_bold_arr.shape}\n"
            f"\t{uncensored_denoised_bold}: {uncensored_denoised_bold_arr.shape}\n"
            f"\t{interpolated_filtered_bold}: {filtered_denoised_bold_arr.shape}\n\n"
        )

    # Formatting & setting of files
    sns.set_style("whitegrid")

    # Create dataframes for the bold_data DVARS, FD
    dvars_regressors = pd.DataFrame(
        {
            "Pre regression": preprocessed_bold_dvars,
            "Post regression": uncensored_denoised_bold_dvars,
            "Post all": filtered_denoised_bold_dvars,
        }
    )

    fd_regressor = pd.read_table(filtered_motion)["framewise_displacement"].values

    # The mean and standard deviation of the preprocessed data,
    # after mean-centering and detrending.
    preprocessed_bold_timeseries = pd.DataFrame(
        {
            "Mean": np.nanmean(preprocessed_bold_arr, axis=0),
            "Std": np.nanstd(preprocessed_bold_arr, axis=0),
        }
    )

    # The mean and standard deviation of the denoised data, with bad volumes included.
    uncensored_denoised_bold_timeseries = pd.DataFrame(
        {
            "Mean": np.nanmean(uncensored_denoised_bold_arr, axis=0),
            "Std": np.nanstd(uncensored_denoised_bold_arr, axis=0),
        }
    )

    if seg_data is not None:
        atlaslabels = nb.load(seg_data).get_fdata()
    else:
        atlaslabels = None

    if not standardize:
        # The plot going to carpet plot will be mean-centered and detrended,
        # but will not otherwise be rescaled.
        detrended_preprocessed_bold_arr = clean(
            preprocessed_bold_arr.T,
            t_r=TR,
            detrend=True,
            filter=False,
            standardize=False,
        ).T

        # Make a temporary file for niftis and ciftis
        if preprocessed_bold.endswith(".nii.gz"):
            temp_preprocessed_file = os.path.join(tempfile.mkdtemp(), "filex_raw.nii.gz")
        else:
            temp_preprocessed_file = os.path.join(tempfile.mkdtemp(), "filex_raw.dtseries.nii")

        # Write out the scaled data
        temp_preprocessed_file = write_ndata(
            data_matrix=detrended_preprocessed_bold_arr,
            template=uncensored_denoised_bold,  # residuals file is censored, so length matches
            filename=temp_preprocessed_file,
            mask=mask,
            TR=TR,
        )
    else:
        temp_preprocessed_file = preprocessed_bold

    files_for_carpet = [temp_preprocessed_file, uncensored_denoised_bold]
    figure_names = [preprocessed_bold_figure, denoised_bold_figure]
    data_arrays = [preprocessed_bold_timeseries, uncensored_denoised_bold_timeseries]
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
            TR=TR,
            subplot=grid[2],  # Use grid for now.
            detrend=standardize,  # Data are already detrended if standardize is False
            legend=False,
            colorbar=True,
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
        plot_framewise_displacement_es(fd_regressor, ax3, run_index=run_index, TR=TR)

        # Save out the before processing file
        fig.savefig(figure_name, bbox_inches="tight", pad_inches=None, dpi=300)

    # Save out the after processing file
    return preprocessed_bold_figure, denoised_bold_figure


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
        sns.set_style("whitegrid")

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
        sns.set_style("whitegrid")
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
            self.func_file,
            atlaslabels=self.seg_data,
            subplot=grid[-1],
            detrend=True,
            TR=self.TR,
            labelsize=labelsize,
            colorbar=False,
        )
        return figure


def plot_carpet(
    func,
    atlaslabels=None,
    detrend=True,
    size=(950, 800),
    labelsize=30,
    subplot=None,
    output_file=None,
    legend=True,
    TR=None,
    lut=None,
    colorbar=False,
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
    detrend : bool, optional
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
    epinii = None
    segnii = None
    img = nb.load(func)
    sns.set_style("whitegrid")
    if isinstance(img, nb.Cifti2Image):  # Cifti
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

        # Decimate data
        data, seg_data = _decimate_data(data, seg_data, size)
        # Preserve continuity
        order = seg_data.argsort(kind="stable")
        # Get color maps
        cmap = ListedColormap([cm.get_cmap("Paired").colors[i] for i in (1, 0, 7, 3)])
        assert len(cmap.colors) == len(
            struct_map
        ), "Mismatch between expected # of structures and colors"

        # ensure no legend for CIFTI
        legend = False

    else:  # Volumetric NIfTI
        img_nii = check_niimg_4d(
            img,
            dtype="auto",
        )  # Check the image is in nifti format
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
        seg_data = lut[oseg.astype(int)]

        # Decimate data
        data, seg_data = _decimate_data(data, seg_data, size)
        # Order following segmentation labels
        order = np.argsort(seg_data)[::-1]
        # Set colormap
        cmap = ListedColormap(cm.get_cmap("tab10").colors[:4][::-1])

        if legend:
            epiavg = func_data.mean(3)
            epinii = nb.Nifti1Image(epiavg, img_nii.affine, img_nii.header)
            segnii = nb.Nifti1Image(lut[atlaslabels.astype(int)], epinii.affine, epinii.header)
            segnii.set_data_dtype("uint8")

    return _carpet(
        func,
        data,
        seg_data,
        order,
        cmap,
        labelsize,
        TR=TR,
        detrend=detrend,
        colorbar=colorbar,
        subplot=subplot,
        output_file=output_file,
    )


def _carpet(
    func,
    data,
    seg_data,
    order,
    cmap,
    labelsize,
    TR=None,
    detrend=True,
    colorbar=False,
    subplot=None,
    output_file=None,
):
    """Build carpetplot for volumetric / CIFTI plots."""
    if TR is None:
        TR = 1.0  # Default TR

    sns.set_style("white")

    # Detrend and z-score data
    if detrend:
        data = clean(data.T, t_r=TR, detrend=True, filter=False).T
        vlimits = (-2, 2)
    else:
        # If detrend is False, then the data are assumed to have native BOLD units.
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
        cbar = fig.colorbar(
            pos,
            cax=ax2,
            ticks=vlimits,
        )
        cbar.ax.tick_params(size=0, labelsize=20)

    #  Write out file
    if output_file is not None:
        figure = plt.gcf()
        figure.savefig(output_file, bbox_inches="tight")
        plt.close(figure)
        figure = None
        return output_file

    return (ax0, ax1, ax2), grid_specification


def plot_alff_reho_volumetric(output_path, filename, name_source):
    """Plot ReHo and ALFF mosaics for niftis.

    NOTE: This is a Node function.

    Parameters
    ----------
    output_path : :obj:`str`
        path to save plot
    filename : :obj:`str`
        surface file
    name_source : :obj:`str`
        original input bold file

    Returns
    ----------
    output_path : :obj:`str`
        path to plot

    """
    import os

    from bids.layout import parse_file_entities
    from nilearn import plotting as plott
    from templateflow.api import get as get_template

    ENTITIES_TO_USE = ["cohort", "den", "res"]

    # templateflow uses the full entity names in its BIDSLayout config,
    # so we need to map the abbreviated names used by xcpd and pybids to the full ones.
    ENTITY_NAMES_MAPPER = {"den": "density", "res": "resolution"}
    space = parse_file_entities(name_source)["space"]
    file_entities = parse_file_entities(name_source)
    entities_to_use = {f: file_entities[f] for f in file_entities if f in ENTITIES_TO_USE}
    entities_to_use = {ENTITY_NAMES_MAPPER.get(k, k): v for k, v in entities_to_use.items()}

    template_file = get_template(template=space, **entities_to_use, suffix="T1w", desc=None)
    if isinstance(template_file, list):
        template_file = template_file[0]

    template = str(template_file)
    output_path = os.path.abspath(output_path)
    plott.plot_stat_map(
        filename, bg_img=template, display_mode="mosaic", cut_coords=8, output_file=output_path
    )
    return output_path


def surf_data_from_cifti(data, axis, surf_name):
    """From https://neurostars.org/t/separate-cifti-by-structure-in-python/17301/2.

    https://nbviewer.org/github/neurohackademy/nh2020-curriculum/blob/master/\
    we-nibabel-markiewicz/NiBabel.ipynb
    """
    assert isinstance(axis, nb.cifti2.BrainModelAxis)
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

    raise ValueError(f"No structure named {surf_name}")


def plot_alff_reho_surface(output_path, filename, name_source):
    """Plot ReHo and ALFF for ciftis on surface.

    NOTE: This is a Node function.

    Parameters
    ----------
    output_path : :obj:`str`
        path to save plot
    filename : :obj:`str`
        surface file
    name_source : :obj:`str`
        original input bold file

    Returns
    ----------
    output_path : :obj:`str`
        path to plot

    """
    import os

    import matplotlib.pyplot as plt
    import nibabel as nb
    import numpy as np
    from bids.layout import parse_file_entities
    from nilearn import plotting as plott
    from templateflow.api import get as get_template

    from xcp_d.utils.plotting import surf_data_from_cifti

    density = parse_file_entities(name_source).get("den", "32k")
    if density == "91k":
        density = "32k"
    rh = str(
        get_template(
            template="fsLR", hemi="R", density="32k", suffix="midthickness", extension=".surf.gii"
        )
    )
    lh = str(
        get_template(
            template="fsLR", hemi="L", density="32k", suffix="midthickness", extension=".surf.gii"
        )
    )

    cifti = nb.load(filename)
    cifti_data = cifti.get_fdata()
    cifti_axes = [cifti.header.get_axis(i) for i in range(cifti.ndim)]

    fig, axes = plt.subplots(figsize=(4, 4), ncols=2, nrows=2, subplot_kw={"projection": "3d"})
    output_path = os.path.abspath(output_path)
    lh_surf_data = surf_data_from_cifti(
        cifti_data,
        cifti_axes[1],
        "CIFTI_STRUCTURE_CORTEX_LEFT",
    )
    rh_surf_data = surf_data_from_cifti(
        cifti_data,
        cifti_axes[1],
        "CIFTI_STRUCTURE_CORTEX_RIGHT",
    )

    v_max = np.max([np.max(lh_surf_data), np.max(rh_surf_data)])
    v_min = np.min([np.min(lh_surf_data), np.min(rh_surf_data)])

    plott.plot_surf_stat_map(
        lh,
        lh_surf_data,
        v_min=v_min,
        v_max=v_max,
        hemi="left",
        view="lateral",
        engine="matplotlib",
        colorbar=False,
        axes=axes[0, 0],
        figure=fig,
    )
    plott.plot_surf_stat_map(
        lh,
        lh_surf_data,
        v_min=v_min,
        v_max=v_max,
        hemi="left",
        view="medial",
        engine="matplotlib",
        colorbar=False,
        axes=axes[1, 0],
        figure=fig,
    )
    plott.plot_surf_stat_map(
        rh,
        rh_surf_data,
        v_min=v_min,
        v_max=v_max,
        hemi="right",
        view="lateral",
        engine="matplotlib",
        colorbar=False,
        axes=axes[0, 1],
        figure=fig,
    )
    plott.plot_surf_stat_map(
        rh,
        rh_surf_data,
        v_min=v_min,
        v_max=v_max,
        hemi="right",
        view="medial",
        engine="matplotlib",
        colorbar=False,
        axes=axes[1, 1],
        figure=fig,
    )
    axes[0, 0].set_title("Left Hemisphere", fontsize=10)
    axes[0, 1].set_title("Right Hemisphere", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path)
    return output_path


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
        n_outliers = censoring_df["framewise_displacement"].sum()
        new_df = pd.DataFrame(
            data=np.zeros((censoring_df.shape[0], n_outliers), dtype=np.int16),
            columns=[f"outlier{i}" for i in range(1, n_outliers + 1)],
        )
        outlier_idx = np.where(censoring_df["framewise_displacement"])[0]
        for i_outlier, outlier_col in enumerate(new_df.columns):
            outlier_row = outlier_idx[i_outlier]
            new_df.loc[outlier_row, outlier_col] = 1

        design_matrix_df = pd.concat((design_matrix_df, new_df), axis=1)

    design_matrix_figure = os.path.abspath("design_matrix.svg")
    plotting.plot_design_matrix(design_matrix_df, output_file=design_matrix_figure)

    return design_matrix_figure
