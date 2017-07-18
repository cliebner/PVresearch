__author__ = 'christina'

# import matplotlib  # version 1.5.1
# matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # version 0.18
import pvlib as pvl

'''
Goals:
1. Handle any known format for raw or processed weather data.
    For now includes: tmy, nrel, pvsyst, pvlib
2. Calculate useful data from fundamental weather data.
    For now, includes: diffuse ratio, sunny days, cloudy days, weather classifier

Basic methods:
1. get weather data (e.g. load from csv or web api)
2. process into a known format
3. calculate
4. (translate if needed, &) export to csv
'''

MORNING_ANGLE = 0.0


def convert_weather_text_to_dataframe(weather_text, sep=',', ix_meta_name=0, ix_meta_data=1, ix_cols=2, ix_start=3):
    '''
    meant to convert any request result into a pretty dataset

    only allows for one line for column headers for now -- this will not work for reading PVSyst results, for example
    but it does work for NREL files
    '''

    # split into lines:
    lines = weather_text.split('\n')
    if len(lines) <= 1:  # then there are no newline chars and may not be useful data
        print 'request returned no data or data in bad format'
        return None

    meta_df = None
    # get metadata dataframe:
    if ix_meta_name is None or ix_meta_data is None:
        pass
    else:
        meta_names = lines[ix_meta_name].split(sep)
        meta_data = lines[ix_meta_data].split(sep)
        meta_df = pd.DataFrame.from_dict(dict(zip(meta_names, meta_data)), orient='index')

    # get main data dataframe:
    col_names = lines[ix_cols].split(sep)
    data_dict = {c: [] for c in col_names}
    for l in range(ix_start, len(lines)):
        this_line = lines[l].split(sep)
        if len(this_line) != len(col_names):
            break
        for c in range(len(col_names)):
            data_dict[col_names[c]].append(this_line[c])
    data_df = pd.DataFrame.from_dict(data_dict, orient='columns')
    # this is how we need to do it for PVsyst results
    # for i in range(i_col + 3, len(all_lines)):
    #     for u in usable_names:
    #         value = all_lines[i].split(self.separator)[name_index_map[u]]
    #         if u is pvsyst_name_lookup[DATE]:
    #             result_dict[u].append(dt.datetime.strptime(value, '%d/%m/%y %H:%M'))
    #         else:
    #             result_dict[u].append(float(value))
    # result_df = pd.DataFrame.from_dict(result_dict, orient='columns')
    return meta_df, data_df


def make_pvlib_object_from_weather(weather_obj, user_gcr):
    pvlib_loc = pvl.location.Location(weather_obj.actual_lat, weather_obj.actual_lon)
    pvlib_solpos = pvlib_loc.get_solarposition(weather_obj.ts_data.index)
    single_tracker_obj = pvl.tracking.SingleAxisTracker(axis_tilt=0, axis_azimuth=180, max_angle=60.0, backtrack=True,
                                                        gcr=user_gcr)
    with np.errstate(invalid='ignore'):
        single_tracker_df = single_tracker_obj.singleaxis(apparent_azimuth=pvlib_solpos.loc[:, 'azimuth'],
                                                          apparent_zenith=pvlib_solpos.loc[:, 'apparent_zenith'])
    return single_tracker_df


def get_backtracking_hrs(tracker_angles):
    '''
    input: Series with DateTimeIndex consistent with rest of data and representing tracker angle
    output: Series with same index as input and representing whether tracker is backtracking -- values are True or False
    '''
    diffs = tracker_angles.fillna(MORNING_ANGLE).diff()
    return pd.Series(((tracker_angles != np.max(tracker_angles)) &
                      (tracker_angles != np.min(tracker_angles)) &
                      (np.isnan(tracker_angles) != True) &
                      (diffs < 0)),
                     index=tracker_angles.index,
                     name='is backtracking?')


def calculate_DFI(ghi=None, dhi=None):
    return pd.Series(dhi/ghi, name='DFI')


def calculate_scaling_factor(scale_map=None, dfi=None):
    # use 0 as default scaling factor to be conservative:
    scaling_factors = pd.Series(len(dfi) * [0.0], index=dfi.index, name='Scaling Factor')
    for s in range(len(scale_map['bins'])):
        scaling_factors.mask(
            (dfi > scale_map['bins'][s][0]) &
            (dfi <= scale_map['bins'][s][1]),
            other=scale_map['values'][s], inplace=True)
    return scaling_factors


class AbstractWeatherData(object):
    def __init__(self):
        self.lat = None
        self.lon = None
        self.name = None
        self.source = None
        self.tz = None
