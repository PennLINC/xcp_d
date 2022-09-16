# from xcp_d.interfaces.filtering import butter_bandpass  # Butterworth
# from xcp_d.utils.confounds import motion_regression_filter  # lowpass and notch
# from xcp_d.utils import read_ndata
# import pandas as pd
# import numpy as np


# #  Read in raw data and filtered data to compare with as df --> update ASAP to put this on circleCI so we can test there too
# raw_data = '/Users/kahinim/Desktop/filter/raw_data.csv'
# raw_data_df = pd.read_table(raw_data, header=None)
# low_pass = '/Users/kahinim/Desktop/filter/low_passed_MR_data.csv'
# lowpass_data_df = pd.read_table(low_pass, header=None)
# notch = '/Users/kahinim/Desktop/filter/notched_MR_data.csv'
# notch_data_df = pd.read_table(notch, header=None)
# butterworth = '/Users/kahinim/Desktop/filter/band_passed_bold_data.csv'
# butterworth_data_df = pd.read_table(butterworth, header=None)


# def test_lp_notch():
#     """
#     Run LP/Notch on toy data, compare to results that have been verified
#     """
#     freqband = [12, 20]
#     # Confirm the LP filter runs with reasonable parameters
#     LP_data = motion_regression_filter(raw_data_df.to_numpy().T.copy(), TR=0.8, motion_filter_type='lp',
#                                        motion_filter_order=2, freqband=freqband, cutoff=12)

#     # What's the difference from the verified data?
#     lp_data_comparator = lowpass_data_df.to_numpy().T
#     diff_LP = np.sum(abs(np.array(LP_data)-np.array(lp_data_comparator)))

#     # Repeat for notch filter
#     notch_data = motion_regression_filter(raw_data_df.to_numpy().T.copy(), TR=0.8,
#                                           motion_filter_type='notch',
#                                           freqband=freqband, motion_filter_order=2,
#                                           cutoff=12)
#     notch_data_comparator = notch_data_df.to_numpy().T
#     diff_notch = np.sum(abs(np.array(notch_data)-np.array(notch_data_comparator)))

#     # Assert a high correlation
#     print(notch_data.shape, notch_data_comparator.shape)
#     assert np.mean(np.correlate(notch_data.squeeze(), notch_data_comparator.squeeze())) > 0.95
#     # Assert a high correlation
#     assert np.mean(np.correlate(LP_data.squeeze(), lp_data_comparator.squeeze())) > 0.95
#     return (diff_LP), (diff_notch)


# def test_butterworth():
#     """
#     Run Butterworth on toy data, compare to results that have been verified
#     """
#     # Confirm the butterworth filter runs with reasonable parameters
#     butterworth_data = butter_bandpass(raw_data_df.to_numpy().T.copy(), fs=1/0.8, highpass=0.009,
#                                        lowpass=0.080, order=2)
#     butterworth_data_comparator = butterworth_data_df.to_numpy().T

#     # What's the difference with verified data?
#     diff = np.sum(abs(np.array(butterworth_data)-np.array(butterworth_data_comparator)))

#     # Assert a high correlation
#     assert np.mean(np.correlate(butterworth_data.squeeze(),
#                    butterworth_data_comparator.squeeze())) > 0.95
#     return (diff)
