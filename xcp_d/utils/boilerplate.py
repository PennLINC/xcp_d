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
    head_radius,
    TR,
):
    """Build a text description of the motion parameter filtering and FD calculation process.

    Parameters
    ----------
    %(motion_filter_type)s
    %(motion_filter_order)s
    %(band_stop_min)s
    %(band_stop_max)s
    %(head_radius)s
    %(TR)s

    Returns
    -------
    desc : :obj:`str`
        A text description of the motion parameters.
    """
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
            if is_modified:
                desc = (
                    "The six translation and rotation head motion traces were "
                    f"band-stop filtered to remove signals between {band_stop_min_adjusted} and "
                    f"{band_stop_max_adjusted} breaths-per-minute "
                    f"(automatically modified from {band_stop_min} and {band_stop_max} BPM due "
                    "to Nyquist frequency constraints) using a(n) "
                    f"{num2words(motion_filter_order, ordinal=True)}-order notch filter, "
                    "based on @fair2020correction. "
                )
            else:
                desc = (
                    "The six translation and rotation head motion traces were "
                    f"band-stop filtered to remove signals between {band_stop_min} and "
                    f"{band_stop_max} breaths-per-minute using a(n) "
                    f"{num2words(motion_filter_order, ordinal=True)}-order notch filter, "
                    "based on @fair2020correction. "
                )
        else:  # lp
            if is_modified:
                desc = (
                    "The six translation and rotation head motion traces were "
                    f"low-pass filtered below {band_stop_min_adjusted} breaths-per-minute "
                    f"(automatically modified from {band_stop_min} BPM due to Nyquist frequency "
                    "constraints) using a(n) "
                    f"{num2words(motion_filter_order, ordinal=True)}-order Butterworth filter, "
                    "based on @gratton2020removal. "
                )
            else:
                desc = (
                    "The six translation and rotation head motion traces were "
                    f"low-pass filtered below {band_stop_min} breaths-per-minute "
                    "using a(n) "
                    f"{num2words(motion_filter_order, ordinal=True)}-order Butterworth filter, "
                    "based on @gratton2020removal. "
                )

        desc += "The Volterra expansion of these filtered motion parameters was then calculated. "

    desc += (
        "Framewise displacement was calculated from the "
        f"{'filtered ' if motion_filter_type else ''}motion parameters using the formula from "
        f"@power_fd_dvars, with a head radius of {head_radius} mm. "
    )

    return desc


@fill_doc
def describe_censoring(motion_filter_type, fd_thresh, exact_scans):
    """Build a text description of the FD censoring process.

    Parameters
    ----------
    %(motion_filter_type)s
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
def describe_regression(params, custom_confounds_file, motion_filter_type):
    """Build a text description of the regression that will be performed.

    Parameters
    ----------
    %(params)s
    %(custom_confounds_file)s
    %(motion_filter_type)s

    Returns
    -------
    desc : :obj:`str`
        A text description of the regression.
    """
    import pandas as pd

    use_custom_confounds, orth = False, False
    if custom_confounds_file is not None:
        use_custom_confounds = True
        custom_confounds = pd.read_table(custom_confounds_file)
        orth = any([c.startswith("signal__") for c in custom_confounds.columns])

    fstr = "filtered " if motion_filter_type else ""

    BASE_DESCRIPTIONS = {
        "custom": "A custom set of regressors was used, with no other regressors from XCP-D.",
        "none": "No nuisance regression was performed.",
        "24P": (
            "In total, 24 nuisance regressors were selected from the preprocessing confounds, "
            "according to the '24P' strategy. "
            "These nuisance regressors included "
            f"six {fstr}motion parameters with their temporal derivatives, "
            "and their quadratic expansion of those six motion parameters and their "
            "temporal derivatives [@benchmarkp;@satterthwaite_2013]."
        ),
        "27P": (
            "In total, 27 nuisance regressors were selected from the preprocessing confounds, "
            "according to the '27P' strategy. "
            "These nuisance regressors included "
            f"six {fstr}motion parameters with their temporal derivatives, "
            "quadratic expansion of those six motion parameters and their derivatives, "
            "mean global signal, mean white matter signal, and mean cerebrospinal fluid signal "
            "[@benchmarkp;@satterthwaite_2013]."
        ),
        "36P": (
            "In total, 36 nuisance regressors were selected from the preprocessing confounds, "
            "according to the '36P' strategy. "
            "These nuisance regressors included "
            f"six {fstr}motion parameters, mean global signal, mean white matter signal, "
            "mean cerebrospinal fluid signal with their temporal derivatives, "
            "and quadratic expansion of six motion parameters, tissue signals and "
            "their temporal derivatives [@benchmarkp;@satterthwaite_2013]."
        ),
        "acompcor": (
            "Nuisance regressors were selected according to the 'acompcor' strategy. "
            "The top 5 aCompCor principal components from the white matter and "
            "cerebrospinal fluid compartments were selected as nuisance regressors "
            "[@behzadi2007component], "
            f"along with the six {fstr}motion parameters and their temporal derivatives "
            "[@benchmarkp;@satterthwaite_2013]. "
            "As the aCompCor regressors were generated on high-pass filtered data, "
            "the associated cosine basis regressors were included. "
            "This has the effect of high-pass filtering the data as well."
        ),
        "acompcor_gsr": (
            "Nuisance regressors were selected according to the 'acompcor_gsr' strategy. "
            "The top 5 aCompCor principal components from the white matter and "
            "cerebrospinal fluid compartments were selected as nuisance regressors "
            "[@behzadi2007component], "
            f"along with the six {fstr}motion parameters and their temporal derivatives, "
            "mean white matter signal, mean cerebrospinal fluid signal, and mean global signal "
            "[@benchmarkp;@satterthwaite_2013]. "
            "As the aCompCor regressors were generated on high-pass filtered data, "
            "the associated cosine basis regressors were included. "
            "This has the effect of high-pass filtering the data as well."
        ),
        "aroma": (
            "Nuisance regressors were selected according to the 'aroma' strategy. "
            "AROMA motion-labeled components [@pruim2015ica], mean white matter signal, "
            "and mean cerebrospinal fluid signal were selected as nuisance regressors "
            "[@benchmarkp;@satterthwaite_2013]."
        ),
        "aroma_gsr": (
            "Nuisance regressors were selected according to the 'aroma_gsr' strategy. "
            "AROMA motion-labeled components [@pruim2015ica], mean white matter signal, "
            "mean cerebrospinal fluid signal, and mean global signal were selected as "
            "nuisance regressors [@benchmarkp;@satterthwaite_2013]."
        ),
        "gsr_only": (
            "Nuisance regressors were selected according to the 'gsr_only' strategy. "
            "Mean global signal was selected as the only nuisance regressor."
        ),
    }

    if params not in BASE_DESCRIPTIONS.keys():
        raise ValueError(f"Unrecognized parameter string '{params}'")

    desc = BASE_DESCRIPTIONS[params]
    if use_custom_confounds and params != "custom":
        desc += " Additionally, custom confounds were also included as nuisance regressors."

    if "aroma" not in params and orth:
        desc += (
            " Custom confounds prefixed with 'signal__' were used to account for variance "
            "explained by known signals. "
            "Prior to denoising the BOLD data, the nuisance confounds were orthogonalized "
            "with respect to the signal regressors."
        )
    elif "aroma" in params and not orth:
        desc += (
            " AROMA non-motion components (i.e., ones assumed to reflect signal) were used to "
            "account for variance by known signals. "
            "Prior to denoising the BOLD data, the nuisance confounds were orthogonalized "
            "with respect to the non-motion components."
        )

    if "aroma" in params or orth:
        desc += (
            " In this way, the confound regressors were orthogonalized to produce regressors "
            "without variance explained by known signals, so that signal would not be removed "
            "from the BOLD data in the later regression."
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
    atlases_4s = [atlas for atlas in atlases if atlas.startswith("4S")]
    described_atlases += atlases_4s
    if atlases_4s:
        parcels = [int(atlas[2:-7]) for atlas in atlases_4s]
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
    if undescribed_atlases:
        raise ValueError(f"Unrecognized atlas(es) in the list: {', '.join(undescribed_atlases)}.")

    return list_to_str(atlas_strings)
