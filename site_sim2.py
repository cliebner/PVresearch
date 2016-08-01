__author__ = 'christina'

# import matplotlib  # version 1.5.1
# matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # version 0.18
import datetime as dt
# When generating a pvsyst results file, keep these project parameters in mind:
# Export all parameters listed below
# Weather file of interest: Hayward, Seattle, Reno
# 3D shading file
# GCR in range [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
# with and without backtracking
# azimuth facing south, zero tilt
# LOCATION = 'Seattle'
LOCATION = 'Hayward Air Term'

WINTER = '1990-12-21'
SUMMER = '1990-06-21'
SPRING = '1990-03-22'

ANG_ELEV = 'Sun Elevation'
ANG_AZ = 'Sun Azimuth'  # pvlib - 180 = pvsyst
ANG_INC = 'Incident Angle'
ANG_PROF = 'Profile Angle'
ANG_SURF_AZ = 'Panel Azimuth'  # pvlib - 180 = pvsyst
ANG_SURF_TILT = 'Panel Tilt'
ANG_TRACK = 'Tracker Angle'
ANG_ZENITH = 'Sun Zenith'
H_GLOBAL = 'Global Horizontal'  # available global irradiance on an unshaded horizontal surface (from weather data)
H_DIFFUSE = 'Diffuse Horizontal'  # available diffuse irradiance on an unshaded horizontal surface
H_DIRECT = 'Direct Horizontal'  # available direct beam irradiance on an unshaded horizontal surface
I_GLOBAL = 'Global Incident'  # available global irradiance in POA (no losses accounted)
I_DIRECT = 'Direct Incident'  # available direct beam irradiance in POA (no losses accounted)
I_DIFF_G = 'Albedo Incident'  # available albedo/ground diffuse irradiance in POA (no losses accounted)
I_DIFF_S = 'Sky Diffuse Incident'  # available sky diffuse irradiance in POA (no losses accounted)
I_DIFF_T = 'Total Diffuse Incident'  # available total diffuse irradiance in POA (no losses accounted)
E_GLOBAL = 'Effective Global'  # effective global irradiance, accounting for hourly shading factor + IAM
E_DIRECT = 'Effective Direct'  # effective direct beam irradiance, accounting for hourly shading factor + IAM
E_DIFF_S = 'Effective Sky Diffuse'  # effective sky diffuse irradiance, accounting for hourly shading factor + IAM
E_DIFF_G = 'Albedo Effective'  # effective albedo irradiance, accounting for hourly shading factor + IAM
S_GLOBAL = 'Global Incident w/Shade'  # effective global irradiance, accounting for hourly shading factor
SLOSS_GLOBAL = 'Global Shade Loss'  # available global irradiance loss due to hourly shading factor
SLOSS_DIRECT = 'Direct Shade Loss'  # available direct beam irradiance loss due to hourly shading factor
SLOSS_DIFFUSE = 'Diffuse Shade Loss'  # available diffuse irradiance loss due to hourly shading factor
SF_GLOBAL = 'Global Shade Factor'
SF_DIRECT = 'Direct Shade Factor'
SF_DIFF_S = 'Sky Diffuse Shade Factor'
SF_DIFF_G = 'Albedo Shade Factor'
IAMF_GLOBAL = 'Global IAM Factor'
IAMF_DIRECT = 'Direct IAM Factor'
IAMF_DIFF_S = 'Sky Diffuse IAM Factor'
IAMF_DIFF_G = 'Albedo IAM Factor'
IS_DT = 'Diffuse Tracking?'
DT_GAIN = 'DiffTrack Gain'
DT_GAIN_FACTOR = 'DiffTrack Gain Factor'
DT_ANG_TRACK = 'DiffTrack Tracker Angle'

factors_list = [SF_GLOBAL, SF_DIRECT, SF_DIFF_S, SF_DIFF_G, IAMF_GLOBAL, IAMF_DIRECT, IAMF_DIFF_S, IAMF_DIFF_G,
                DT_GAIN_FACTOR]
factors_range = [0.0, 1.1]
angles_list = [ANG_ELEV, ANG_AZ, ANG_INC, ANG_PROF, ANG_SURF_AZ, ANG_SURF_TILT, ANG_TRACK, ANG_ZENITH,
               DT_ANG_TRACK]
angles_range = [-90.0, 90.0]
irradiance_list = [H_GLOBAL, H_DIFFUSE, H_DIRECT, I_GLOBAL, I_DIRECT, I_DIFF_G, I_DIFF_S, I_DIFF_T,
                   E_GLOBAL, E_DIRECT, E_DIFF_S, E_DIFF_G, S_GLOBAL, SLOSS_GLOBAL, SLOSS_DIRECT, SLOSS_DIFFUSE, DT_GAIN]
irradiance_range = [0.0, 1000.0]
boolean_list = [IS_DT]
boolean_range = [-1.5, 1.5]

pvsyst_rename = {
    'HSol': ANG_ELEV,
    'AzSol': ANG_AZ,  # 180 deg out of phase with pvlib
    'AngInc': ANG_INC,
    'AngProf': ANG_PROF,
    'PlTilt': ANG_SURF_TILT,
    'PlAzim': ANG_SURF_AZ,  # 180 deg out of phase with pvlib
    'PhiAng': ANG_TRACK,
    'GlobHor': H_GLOBAL,
    'DiffHor': H_DIFFUSE,
    'BeamHor': H_DIRECT,
    'GlobInc': I_GLOBAL,
    'BeamInc': I_DIRECT,
    'DifSInc': I_DIFF_S,
    'Alb Inc': I_DIFF_G,
    'GlobEff': E_GLOBAL,
    'BeamEff': E_DIRECT,
    'DiffEff': E_DIFF_S,
    'Alb_Eff': E_DIFF_G,
    'GlobShd': S_GLOBAL,
    'ShdLoss': SLOSS_GLOBAL,
    'ShdBLss': SLOSS_DIRECT,
    'ShdDLss': SLOSS_DIFFUSE,
    'FShdGl': SF_GLOBAL,
    'FShdBm': SF_DIRECT,
    'FShdDif': SF_DIFF_S,
    'FShdAlb': SF_DIFF_G,
    'FIAMGl': IAMF_GLOBAL,
    'FIAMBm': IAMF_DIRECT,
    'FIAMDif': IAMF_DIFF_S,
    'FIAMAlb': IAMF_DIFF_G,
}
pvsyst_variant_map = {
    '0': '2GCR',
    '1': '2GCR_b',
    '2': '3GCR',
    '3': '3GCR_b',
    '4': '4GCR',
    '5': '4GCR_b',
    '6': '5GCR',
    '7': '5GCR_b',
    '8': '6GCR',
    '9': '6GCR_b',
    'A': '7GCR',
    'B': '7GCR_b',
    'C': '8GCR',
    'D': '8GCR_b',
    'E': '2GCR_flat',
}  # maps filename suffix (key) to pvsyst variant name (value)

gcr_map = {
    '0': 0.2,
    '1': 0.2,
    '2': 0.3,
    '3': 0.3,
    '4': 0.4,
    '5': 0.4,
    '6': 0.5,
    '7': 0.5,
    '8': 0.6,
    '9': 0.6,
    'A': 0.7,
    'B': 0.7,
    'C': 0.8,
    'D': 0.8,
}


def get_pvsyst_results(pvsyst_hourly_results_file):
    # Unpack results from file:
    # Read in PVSyst results file as data frame:
    # columns: HSol (sun height), AzSol (sun azimuth), AngInc (incidence angle), AngProf (profile angle),
    # PlTilt (plane tilt: tracking), PlAzim (plane azimuth: tracking), PhiAng (phi angle: tracking),
    # DiffHor, BeamHor, GlobHor, T Amb, GlobInc, BeamInc, DifSInc, Alb Inc,
    # GlobIAM, GlobEFf, BeamEff, DiffEff, Alb_Eff
    pvsyst_DF = pd.read_csv(pvsyst_hourly_results_file, sep=';', header=0, skiprows=[0,1,2,3,4,5,6,7,8,9,11],
                            index_col=0, parse_dates=True, dayfirst=True)
    pvsyst_DF.rename(columns=pvsyst_rename, inplace=True)
    return pvsyst_DF


def batch_pvresults(variant_list=pvsyst_variant_map.keys()):
    return {v: get_pvsyst_results('../PVresearch/' + LOCATION + '_Project_HourlyRes_' + v + '.CSV')
            for v in variant_list}


def get_diffuse_tracking(tracking_df, flat_df):
    # Compare "effective" irradiance from two simulations, where "effective" accounts for linear effect of 3D near
    # shadings construction & for IAM (impt at low sun angles!):
    # Outputs: a dataframe with diffuse metrics concatenated onto input tracking_df
    # Inputs: tracking_df = results df of simulation of arrays on N/S axis with E/W tracking on 0 deg tilt
    #         flat_df = results df of simulation of arrays w/o tracking on 0 deg tilt

    diffuse_tracking_gain = flat_df[E_GLOBAL] - tracking_df[E_GLOBAL]
    is_diffuse_tracking_better = (diffuse_tracking_gain > 1.0) & (tracking_df[E_GLOBAL] > 1.0)
    diffuse_tracking_gain_ratio = diffuse_tracking_gain / tracking_df[E_GLOBAL]
    when_flat = pd.Series([0]*8760, index=flat_df.index)[is_diffuse_tracking_better]
    when_tracking = tracking_df.loc[(is_diffuse_tracking_better == False), ANG_TRACK]
    effective_tracker_angle = pd.concat([when_flat, when_tracking])
    effective_tracker_angle.sort_index

    return pd.concat([tracking_df,
                      pd.DataFrame({IS_DT: is_diffuse_tracking_better,
                                    DT_ANG_TRACK: effective_tracker_angle,
                                    DT_GAIN: diffuse_tracking_gain,
                                    DT_GAIN_FACTOR: diffuse_tracking_gain_ratio})], axis=1)


def resample_diffuse_only(diffuse_tracking_df, resample_str):

    # Only want to resample values when diffuse makes sense (i.e. "gain" is positive values).
    # Output: 2-tuple of (absolute gain, gain factor)
    # Input: df with diffuse values, string representing resample interval ('A' = annual, 'M' = monthly, etc.)
    # resample_gain = diffuse_tracking_df[DT_GAIN][diffuse_tracking_df[IS_DIFF]].resample(resample_str).sum()
    resample_gain = (diffuse_tracking_df.loc[diffuse_tracking_df[IS_DT], DT_GAIN]).resample(resample_str).sum()
    resample_factor = resample_gain / diffuse_tracking_df[E_GLOBAL].resample(resample_str).sum()
    return resample_gain, resample_factor


def get_single_factor_df(df_dict, single_factor):
    # Creates a dataframe by pulling out one column from each df in the df_dict.  Plotting is a bit easier when all
    # columns are the same factor.
    # E.G. get_single_factor_df(df_dict, ANG_INC) copies angle of incidence values from each df in df_dict and
    # creates a new df.  Column names are the simulation variant names.
    single_factor_df = pd.DataFrame({pvsyst_variant_map[d[0]]: d[1][single_factor] for d in df_dict.iteritems()})
    return single_factor_df


def sort_diffuse_days(df, is_diffuse=True, diffuse_ratio_threshold=0.7, min_day_fraction=0.75):
    # Sorts days of the year based on SKY diffuse ratio (ignoring albedo).  Use keyword args to select for high
    # diffuse (cloudy) or low diffuse (clear) days.
    # Outputs: pandas series for days where diffuse ratio meets a limit
    # Inputs: dataframe with the effective sky diffuse irradiance and
    #         optional kwargs: diffuse boolean (set to False to find clear days),
    #                          diffuse ratio threshold,
    #                          minimum number of hours at or above threshold
    diffuse_ratio = df[E_DIFF_S] / df[E_GLOBAL]
    day_count = diffuse_ratio.resample('D').count()
    if is_diffuse is True:
        diffuse_count = diffuse_ratio[diffuse_ratio > diffuse_ratio_threshold].resample('D').count()
    else:
        diffuse_count = diffuse_ratio[diffuse_ratio < diffuse_ratio_threshold].resample('D').count()
    day_fraction = diffuse_count / day_count

    return day_fraction[day_fraction > min_day_fraction].sort_values(ascending=False)


def plot_many_days(df_dict, single_factor, days_list, flat_df_dict=None):
    # Output: graph with subplots for each day
    # Inputs: df_dict: dict of dataframes of PVsyst result
    #        single_factor: keyword name of data to plot
    #        days_list: list of datetimes as strings in format %Y-%m-%d; used for subplots' titles
    #        (optional) flat_df_dict: if you also want to plot results from "flat" simulation
    #        (optional) units: string for y axis label
    #        (optional) title: string for graph title (not subplot)

    # to handle input dict of dataframes or a single dataframe:
    if type(df_dict) is dict:
        single_factor_df = get_single_factor_df(df_dict, single_factor)
        legend_list = list(single_factor_df.columns)
    else:
        single_factor_df = df_dict[single_factor]
        legend_list = [single_factor]
    if flat_df_dict is not None:
        flat_df = get_single_factor_df(flat_df_dict, single_factor)
        legend_list.append(list(flat_df.columns)[0])
    plt.figure(figsize=(8, 15))
    plt.suptitle(LOCATION + ': ' + single_factor, fontsize=16)

    i = 1
    ax = []
    for d in days_list:
        if i == 1:
            ax.append(plt.subplot(len(days_list), 1, i))
            plt.ylabel(get_units(single_factor), fontsize=14)
        else:
            ax.append(plt.subplot(len(days_list), 1, i, sharex=ax[0], sharey=ax[0]))
        plt.grid(True, which='major', axis='x')
        ax[i-1].plot(single_factor_df[d].index.hour, single_factor_df[d])
        if flat_df_dict is not None:
            ax[i - 1].plot(flat_df[d].index.hour, flat_df[d], 'k--')
        ax[i-1].legend(legend_list, loc='lower right', fontsize=9)
        ax[i-1].set_title(d, fontsize=12)
        i += 1

    plt.xlabel('Hour of day', fontsize=14)
    ax[0].set_xlim([0, 24])
    ax[0].set_xticks([0, 6, 12, 18, 24])
    ax[0].set_ylim(get_ylim(single_factor))


def get_ylim(single_factor):
    if single_factor in factors_list:
        new_ylim = factors_range
    elif single_factor in irradiance_list:
        new_ylim = irradiance_range
    elif single_factor in angles_list:
        new_ylim = angles_range
    elif single_factor in boolean_list:
        new_ylim = boolean_range
    else:
        pass
    return new_ylim

def get_units(single_factor):
    if single_factor in factors_list:
        units = 'factor'
    elif single_factor in irradiance_list:
        units = 'W/m2'
    elif single_factor in angles_list:
        units = 'Angle (deg)'
    elif single_factor in boolean_list:
        units = '1 = True'
    else:
        pass
    return units

# def plot_solar_position_diagram(df, day):
#     fig, ax1 = plt.subplots()
#     pvsyst_days = ['1990-06-22', '1990-05-22', '1990-04-22', '1990-03-20', '1990-02-21', '1990-01-19', '1990-12-22']
#     for d in pvsyst_days:
#         ax1.plot(df.loc[d, ANG_AZ], df.loc[d, ANG_ELEV], 'k-')


def get_daytime_df(df, threshold=10.0):
    # Use this method to get only the hours when Global Incident Irradiance > threshold W/m2.
    daytime_df = df[df[I_GLOBAL] > float(threshold)]
    return daytime_df


def plot_annual_by_hours(df_dict, df_list, single_factor):
    # plots any PVsyst result series throughout the year for one dataframe

    morning_hrs = [5, 6, 7, 8, 9, 10, 11]
    evening_hrs = [12, 13, 14, 15, 16, 17, 18]

    if '_b' in pvsyst_variant_map[df_list[0]]:
        backtrack_label = 'w/ Backtracking'
    else:
        backtrack_label = 'w/o Backtracking'
    plt.figure(figsize=(15, 15))
    plt.suptitle(LOCATION + ': ' + single_factor + ': ' + backtrack_label, fontsize=16)

    i = 1
    ax = []
    for df in df_list:
        if i == 1:
            ax.append(plt.subplot(len(df_list), 1, i))
            plt.ylabel(get_units(single_factor), fontsize=14)
        else:
            ax.append(plt.subplot(len(df_list), 1, i, sharex=ax[0], sharey=ax[0]))

        daytime_df = get_daytime_df(df_dict[df], threshold=0.1)

        shade_factors_by_hour_am = [daytime_df[daytime_df.index.hour == d].loc[:, single_factor] for d in morning_hrs]
        for j in range(len(morning_hrs)):
            ax[i - 1].plot(shade_factors_by_hour_am[j], 'b.')
            text_pos_y = shade_factors_by_hour_am[j][0:int(len(shade_factors_by_hour_am[j])/2)].min()
            text_pos_x = shade_factors_by_hour_am[j][0:int(len(shade_factors_by_hour_am[j])/2)].idxmin()
            ax[i - 1].annotate(str(morning_hrs[j])+'h',
                               xy=(text_pos_x, text_pos_y), xytext=(text_pos_x, text_pos_y), fontsize=14)

        shade_factors_by_hour_pm = [daytime_df[daytime_df.index.hour == d].loc[:, single_factor] for d in evening_hrs]
        for j in range(len(evening_hrs)):
            ax[i - 1].plot(shade_factors_by_hour_pm[j], 'r.')
            text_pos_y = shade_factors_by_hour_pm[j][int(len(shade_factors_by_hour_pm[j])/2):].min()
            text_pos_x = shade_factors_by_hour_pm[j][int(len(shade_factors_by_hour_pm[j])/2):].idxmin()
            ax[i - 1].annotate(str(evening_hrs[j]) + 'h',
                               xy=(text_pos_x, text_pos_y), xytext=(text_pos_x, text_pos_y), fontsize=14)

        ax[i - 1].set_title(pvsyst_variant_map[df], fontsize=12)
        i += 1
    ax[0].set_ylim(get_ylim(single_factor))

print LOCATION
print "loading no_backtrack files..."
no_backtrack = batch_pvresults(['0', '2', '4', '6', '8', 'A', 'C'])

print "loading backtrack files..."
backtrack = batch_pvresults(['1', '3', '5', '7', '9', 'B', 'D'])

print "loading fixed tilt file..."
flat = batch_pvresults(['E'])

no_backtrack_diffuse = {t[0]: get_diffuse_tracking(t[1], flat['E']) for t in no_backtrack.iteritems()}


backtrack_diffuse = {t[0]: get_diffuse_tracking(t[1], flat['E']) for t in backtrack.iteritems()}

# # To export to csv:
# print "saving no backtrack diffuse to .csv files..."
# for d in no_backtrack_diffuse.iteritems():
#     d[1].to_csv('no_backtrack_diffuse_' + LOCATION + '_' + pvsyst_variant_map[d[0]])
#
# print "saving backtrack diffuse to .csv files..."
# for d in backtrack_diffuse.iteritems():
#     d[1].to_csv('backtrack_diffuse_' + LOCATION + '_' + pvsyst_variant_map[d[0]])
#
# # now trying to load it back to dataframe:
# reloaded = pd.read_csv('backtrack_diffuse_' + LOCATION + '_' + pvsyst_variant_map['1'], sep=',', header=0,
#                        index_col=0, parse_dates=True, dayfirst=True)


# # PLOTS: Annual improvement vs GCR for backtrack and no backtracking
# backtrack_annual_DT_factors = {gcr_map[d[0]]: resample_diffuse_only(d[1], 'A')[1]
#                                for d in backtrack_diffuse.iteritems()}
# no_backtrack_annual_DT_factors = {gcr_map[d[0]]: resample_diffuse_only(d[1], 'A')[1]
#                                   for d in no_backtrack_diffuse.iteritems()}
# plt.plot(backtrack_annual_DT_factors.keys(), backtrack_annual_DT_factors.values(), 'bo')
# plt.plot(no_backtrack_annual_DT_factors.keys(), no_backtrack_annual_DT_factors.values(), 'ro')
# plt.xlabel('Ground coverage ratio, GCR', fontsize=12)
# plt.xlim([0.0, 1.0])
# plt.ylabel('Improvement ratio', fontsize=12)
# plt.ylim([0.0, 0.05])
# plt.legend(['w/ Backtracking', 'w/o Backtracking'], loc='upper left', fontsize=9)
# plt.title(LOCATION + ': Annual Effective Irradiance Improvement\ndue to Flat Orientation during Diffuse Hours',
#           fontsize=16)
#
# # PLOTS: Monthly improvement
# backtrack_monthly_DT_factors = pd.DataFrame({pvsyst_variant_map[d[0]]: resample_diffuse_only(d[1], 'M')[1]
#                                             for d in backtrack_diffuse.iteritems()})
# no_backtrack_monthly_DT_factors = pd.DataFrame({pvsyst_variant_map[d[0]]: resample_diffuse_only(d[1], 'M')[1]
#                                                for d in no_backtrack_diffuse.iteritems()})
# f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True)
# plt.ylim(0, 0.10)
# ax1.plot(backtrack_monthly_DT_factors)
# ax1.legend(backtrack_monthly_DT_factors.columns, fontsize=9)
# ax1.set_title('w/ Backtracking', fontsize=12)
# ax2.plot(no_backtrack_monthly_DT_factors)
# ax2.legend(no_backtrack_monthly_DT_factors.columns, fontsize=9)
# ax2.set_title('w/o Backtracking', fontsize=12)
# plt.suptitle(LOCATION + ': Monthly Effective Irradiance Improvement due to Flat Orientation on Diffuse Days',
#              fontsize=16)

# # PLOTS: ANG_TRACK: standard tracking angle vs hour of day for 3 key days of the year (winter, summer, spring),
# # family of GCR curves
# # backtracking
# plot_many_days(backtrack, ANG_TRACK, [SUMMER, SPRING, WINTER],
#                units=ANG_TRACK, title='Tracker Angle w/ Backtracking')
# # no backtracking
# plot_many_days(no_backtrack, ANG_TRACK, [SUMMER, SPRING, WINTER],
#                units=ANG_TRACK, title='Tracker Angle w/o Backtracking')

# # PLOTS: SF_DIRECT: beam shade factor (no shade: SF = 1.0) vs hour of day for 3 keys days of the year, family of GCR
# # backtracking
# plot_many_days(backtrack, SF_DIRECT, [SUMMER, SPRING, WINTER],
#                flat_df_dict=flat)
# # no backtracking
# plot_many_days(no_backtrack, SF_DIRECT, [SUMMER, SPRING, WINTER],
#                flat_df_dict=flat)
#
# # SF_DIFF_S: sky diffuse shade factor (no shade: SF = 1.0) vs hour of day for 3 keys days of the year, family of GCR
# # backtracking
# plot_many_days(backtrack, SF_DIFF_S, [SUMMER, SPRING, WINTER],
#                flat_df_dict=flat)
# # no backtracking
# plot_many_days(no_backtrack, SF_DIFF_S, [SUMMER, SPRING, WINTER],
#                flat_df_dict=flat)


# # PLOTS: E_GLOBAL:
# diffuse_days_2GCR_backtrack = sort_diffuse_days(backtrack['1'], is_diffuse=True)
# plot_many_days(backtrack, E_GLOBAL, diffuse_days_2GCR_backtrack.index[:4].strftime('%Y-%m-%d'),
#                flat_df_dict=flat, units='kWh/m2', title='Effective Global Irradiance w/ Backtracking\non Cloudy Days')
#
# diffuse_days_2GCR_no_backtrack = sort_diffuse_days(no_backtrack['0'], is_diffuse=True)
# plot_many_days(no_backtrack, E_GLOBAL, diffuse_days_2GCR_no_backtrack.index[:4].strftime('%Y-%m-%d'),
#                flat_df_dict=flat, units='kWh/m2', title='Effective Global Irradiance w/o Backtracking\non Cloudy Days')
#
# clear_days_2GCR_backtrack = sort_diffuse_days(backtrack['1'], is_diffuse=False, diffuse_ratio_threshold=0.3)
# plot_many_days(backtrack, E_GLOBAL, clear_days_2GCR_backtrack.index[:4].strftime('%Y-%m-%d'),
#                flat_df_dict=flat, units='kWh/m2', title='Effective Global Irradiance w/ Backtracking\non Clear Days')
#
# clear_days_2GCR_no_backtrack = sort_diffuse_days(no_backtrack['0'], is_diffuse=False, diffuse_ratio_threshold=0.3)
# plot_many_days(no_backtrack, E_GLOBAL, clear_days_2GCR_no_backtrack.index[:4].strftime('%Y-%m-%d'),
#                flat_df_dict=flat, units='kWh/m2', title='Effective Global Irradiance w/o Backtracking\non Clear Days')

# # PLOTS: Effective tracking angle on partly cloudy days:
# partly_cloudy = sort_diffuse_days(backtrack['1'], is_diffuse=True, min_day_fraction=0.6)
# plot_many_days(backtrack_diffuse['5'], DT_ANG_TRACK, partly_cloudy.index[-4:].strftime('%Y-%m-%d'),
#                units='deg', title='Effective Tracker Angle w/ Optimal Diffuse Tracking\n '
#                                   'and Backtracking on Partly Cloudy Days')
# plot_many_days(backtrack_diffuse['5'], ANG_TRACK, partly_cloudy.index[-4:].strftime('%Y-%m-%d'),
#                units='deg', title='Tracker Angle w/ Backtracking\n '
#                                   'on Partly Cloudy Days')

# PLOTS: Annual effective incident for various GCR: flat, no backtrack, backtrack, backtrack+diffuse tracking
backtrack_annual = {gcr_map[d[0]]: float(d[1][E_GLOBAL].resample('A').sum())
                    for d in backtrack.iteritems()}
no_backtrack_annual = {gcr_map[d[0]]: float(d[1][E_GLOBAL].resample('A').sum())
                       for d in no_backtrack.iteritems()}
flat_annual = dict(zip(backtrack_annual.keys(),
                       len(backtrack_annual.keys())*[float(flat['E'][E_GLOBAL].resample('A').sum())]))

backtrack_annual_DT_gain = {gcr_map[d[0]]: float(resample_diffuse_only(d[1], 'A')[0])
                               for d in backtrack_diffuse.iteritems()}
no_backtrack_annual_DT_gain = {gcr_map[d[0]]: float(resample_diffuse_only(d[1], 'A')[0])
                                  for d in no_backtrack_diffuse.iteritems()}
backtrack_with_DT_annual = {d[0]: d[1] + backtrack_annual_DT_gain[d[0]]
                            for d in backtrack_annual.iteritems()}
no_backtrack_with_DT_annual = {d[0]: d[1] + no_backtrack_annual_DT_gain[d[0]]
                               for d in no_backtrack_annual.iteritems()}
plt.plot(flat_annual.keys(), flat_annual.values(), 'ko')
plt.plot(backtrack_annual.keys(), backtrack_annual.values(), 'ro')
plt.plot(backtrack_with_DT_annual.keys(), backtrack_with_DT_annual.values(), 'r^')
plt.plot(no_backtrack_annual.keys(), no_backtrack_annual.values(), 'bo')
plt.plot(no_backtrack_with_DT_annual.keys(), no_backtrack_with_DT_annual.values(), 'b^')
plt.legend(['Flat', 'Backtrack', 'Backtrack w/ DT', 'No backtrack', 'No backtrack w/ DT'])
plt.title(LOCATION + ': Annual Effective POA Irradiance with Different Tracking Schemes')
plt.xlim(0.0, 1.0)
plt.xlabel('GCR')
plt.ylabel('kWh/m2')

# PLOTS: Direct/Beam Shade Factor throughout year:
# plot_annual_by_hours(no_backtrack, ['0', '4', 'C'], SF_DIRECT)
# daytime_df = get_daytime_df(no_backtrack['6'], threshold=0.1)
# plt.plot(daytime_df.loc[:, ANG_ELEV], daytime_df.loc[:, SF_DIRECT], 'b.')
# six_pm_df = daytime_df[daytime_df.index.hour == 18]
# plt.plot(six_pm_df.loc['1990-04-08':'1990-04-20', ANG_ELEV], six_pm_df.loc['1990-04-08':'1990-04-20', SF_DIRECT], 'r.')
# # plt.plot(daytime_df.loc[:, ANG_TRACK], daytime_df.loc[:, SF_DIRECT], 'r.')
# # plt.plot(daytime_df.loc[:, ANG_PROF], daytime_df.loc[:, SF_DIRECT], 'g.')
# # plt.plot(daytime_df.loc[:, ANG_INC], daytime_df.loc[:, SF_DIRECT], 'k.')
# plt.xlim(0, 90)
# plt.ylim(0, 1.1)
# plt.title(LOCATION + ': ' + pvsyst_variant_map['6'])
# plt.xlabel(ANG_ELEV)
# plt.ylabel(SF_DIRECT)
# TODO: characterize diffuse-ness of location

