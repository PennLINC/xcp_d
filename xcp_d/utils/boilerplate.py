"""Functions for generating boilerplate code."""

from xcp_d.utils.confounds import _modify_motion_filter
from xcp_d.utils.doc import fill_doc
from xcp_d.utils.utils import list_to_str


@fill_doc
def describe_motion_parameters(
    *,
    motion_filter_type,
    motion_filter_order,
    band_stop_min,
    band_stop_max,
    TR,
):
    """Build a text description of the motion parameter filtering and FD calculation process.

    Parameters
    ----------
    %(motion_filter_type)s
    %(motion_filter_order)s
    %(band_stop_min)s
    %(band_stop_max)s
    %(TR)s

    Returns
    -------
    desc : :obj:`str`
        A text description of the motion parameters.
    """
    import numpy as np
    from num2words import num2words

    desc = ""
    if motion_filter_type:
        band_stop_min_adjusted, band_stop_max_adjusted, is_modified = _modify_motion_filter(
            motion_filter_type=motion_filter_type,
            band_stop_min=band_stop_min,
            band_stop_max=band_stop_max,
            TR=TR,
        )
        if motion_filter_type == "notch":
            n_filter_applications = int(np.floor(motion_filter_order / 4))
            if is_modified:
                desc = (
                    "The six translation and rotation head motion traces were "
                    f"band-stop filtered to remove signals between {band_stop_min_adjusted} and "
                    f"{band_stop_max_adjusted} breaths-per-minute "
                    f"(automatically modified from {band_stop_min} and {band_stop_max} BPM due "
                    "to Nyquist frequency constraints) using a(n) "
                    f"{num2words(n_filter_applications, ordinal=True)}-order notch filter, "
                    "based on @fair2020correction. "
                )
            else:
                desc = (
                    "The six translation and rotation head motion traces were "
                    f"band-stop filtered to remove signals between {band_stop_min} and "
                    f"{band_stop_max} breaths-per-minute using a(n) "
                    f"{num2words(n_filter_applications, ordinal=True)}-order notch filter, "
                    "based on @fair2020correction. "
                )
        else:  # lp
            n_filter_applications = int(np.floor(motion_filter_order / 2))
            if is_modified:
                desc = (
                    "The six translation and rotation head motion traces were "
                    f"low-pass filtered below {band_stop_min_adjusted} breaths-per-minute "
                    f"(automatically modified from {band_stop_min} BPM due to Nyquist frequency "
                    "constraints) using a(n) "
                    f"{num2words(n_filter_applications, ordinal=True)}-order Butterworth filter, "
                    "based on @gratton2020removal. "
                )
            else:
                desc = (
                    "The six translation and rotation head motion traces were "
                    f"low-pass filtered below {band_stop_min} breaths-per-minute "
                    "using a(n) "
                    f"{num2words(n_filter_applications, ordinal=True)}-order Butterworth filter, "
                    "based on @gratton2020removal. "
                )

        desc += "The Volterra expansion of these filtered motion parameters was then calculated. "

    return desc


@fill_doc
def describe_censoring(*, motion_filter_type, head_radius, fd_thresh, exact_scans):
    """Build a text description of the FD censoring process.

    Parameters
    ----------
    %(motion_filter_type)s
    %(head_radius)s
    %(fd_thresh)s
    %(exact_scans)s

    Returns
    -------
    desc : :obj:`str`
        A text description of the censoring procedure.
    """
    desc = ""
    if fd_thresh > 0:
        desc += (
            "Framewise displacement was calculated from the "
            f"{'filtered ' if motion_filter_type else ''}motion parameters using the formula from "
            f"@power_fd_dvars, with a head radius of {head_radius} mm. "
            f"Volumes with {'filtered ' if motion_filter_type else ''}framewise displacement "
            f"greater than {fd_thresh} mm were flagged as high-motion outliers for the sake of "
            "later censoring [@power_fd_dvars]."
        )

    if exact_scans and (fd_thresh > 0):
        desc += (
            " Additional sets of censoring volumes were randomly selected to produce additional "
            f"correlation matrices limited to {list_to_str(exact_scans)} volumes."
        )
    elif exact_scans:
        desc += (
            "Volumes were randomly selected for censoring, to produce additional correlation "
            f"matrices limited to {list_to_str(exact_scans)} volumes."
        )

    return desc


@fill_doc
def describe_regression(
    *,
    confounds_config,
    motion_filter_type,
    motion_filter_order,
    band_stop_min,
    band_stop_max,
    TR,
    fd_thresh,
):
    """Build a text description of the regression that will be performed.

    Parameters
    ----------
    confounds_config
    %(motion_filter_type)s

    Returns
    -------
    desc : :obj:`str`
        A text description of the regression.
    """
    if confounds_config is None:
        return "No nuisance regression was performed."

    desc = confounds_config["description"]

    if (fd_thresh > 0) and motion_filter_type:
        # Censoring was done, so just refer back to the earlier description of the filter
        desc += (
            " Any motion parameters in the confounds file were filtered using the same "
            "parameters as described above and the Volterra expansion was calculated."
        )
    elif motion_filter_type:
        # Censoring was not done, so describe the filter here
        desc += " " + describe_motion_parameters(
            motion_filter_type=motion_filter_type,
            motion_filter_order=motion_filter_order,
            band_stop_min=band_stop_min,
            band_stop_max=band_stop_max,
            TR=TR,
        )

    return desc


def describe_atlases(atlases):
    """Build a text description of the atlases that will be used."""
    atlas_descriptions = {
        "Glasser": "the Glasser atlas [@Glasser_2016]",
        "Gordon": "the Gordon atlas [@Gordon_2014]",
        "Tian": "the Tian subcortical atlas [@tian2020topographic]",
        "HCP": "the HCP CIFTI subcortical atlas [@glasser2013minimal]",
        "MIDB": (
            "the MIDB precision brain atlas derived from ABCD data and thresholded at 75% "
            "probability [@hermosillo2022precision]"
        ),
        "MyersLabonte": (
            "the Myers-Labonte infant atlas thresholded at 50% probability [@myers2023functional]"
        ),
    }

    atlas_strings = []
    described_atlases = []
    atlases_4s = [atlas for atlas in atlases if str(atlas).startswith("4S")]
    described_atlases += atlases_4s
    if atlases_4s:
        parcels = [int(str(atlas[2:-7])) for atlas in atlases_4s]
        s = (
            "the Schaefer Supplemented with Subcortical Structures (4S) atlas "
            "[@Schaefer_2017;@pauli2018high;@king2019functional;@najdenovska2018vivo;"
            "@glasser2013minimal] "
            f"at {len(atlases_4s)} different resolutions ({list_to_str(parcels)} parcels)"
        )
        atlas_strings.append(s)

    for k, v in atlas_descriptions.items():
        if k in atlases:
            atlas_strings.append(v)
            described_atlases.append(k)

    undescribed_atlases = [atlas for atlas in atlases if atlas not in described_atlases]
    for atlas in undescribed_atlases:
        atlas_strings.append(f"the {atlas} atlas")

    return list_to_str(atlas_strings)
