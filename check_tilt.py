__author__ = 'christina'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # version 0.18
import pvsyst_sims2 as pvs2

# define pvsyst project to study:
PROJECT_LOCATION = 'Cedar City Municipal Ap'
PROJECT_PARAM_LIST = [60, 45, 50, 55]
PROJECT_VARIANT_LIST = ['0', 'A', '9', 'B']
PARAM = 'track'
PARAM_VARIANT_MAP = dict(zip(PROJECT_PARAM_LIST, PROJECT_VARIANT_LIST))
VARIANT_PARAM_MAP = dict(zip(PROJECT_VARIANT_LIST, PROJECT_PARAM_LIST))

project = pvs2.BatchPVSystResults(PROJECT_LOCATION, PARAM, PROJECT_PARAM_LIST, PROJECT_VARIANT_LIST)


##  THESE METHODS ARE NOW IN THE BASE CLASS - BATCHPVSYSTRESULTS
# def compare_single_factor_to_baseline(project, single_factor, resample_rate='H'):
#
#     project_diff_tilt = pd.DataFrame({(str(VARIANT_PARAM_MAP[p[0]])+PARAM):
#                                       project.results_dict[PARAM_VARIANT_MAP[60]][single_factor].resample(resample_rate).sum() -
#                                       p[1][single_factor].resample(resample_rate).sum()
#                                       for p in project.results_dict.iteritems()})
#
#     return project_diff_tilt
#
# def compare_single_factor_to_60(project, pvs2.ENERGY_ARRAY, resample_rate='A'):
# # refine dataframe when tilt is different
# project_ANG_TRACK = project.get_single_factor_df(pvs2.ANG_TRACK)
# project_diff_tilt = project_ANG_TRACK[project_ANG_TRACK['50tilt'] != project_ANG_TRACK['60tilt']]
#
# # find irradiances during hours when tilt is different
# project_diff_tilt_E_GLOBAL = project.get_single_factor_df(pvs2.E_GLOBAL).loc[project_diff_tilt.index, :]
# project_diff_tilt_E_DIRECT = project.get_single_factor_df(pvs2.E_DIRECT).loc[project_diff_tilt.index, :]
# project_diff_tilt_E_DIFF_S = project.get_single_factor_df(pvs2.E_DIFF_S).loc[project_diff_tilt.index, :]
#
# # plot tilt angles to verify
# project.plot_single_factor_multiple_days(pvs2.ANG_TRACK)