__author__ = 'christina'

# import matplotlib  # version 1.5.1
# matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # version 0.18
import datetime as dt


# global variables:
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
H_DIFF_RATIO = 'Horizontal Diffuse Ratio'
I_DIFF_RATIO = 'Incident Diffuse Ratio'
E_DIFF_RATIO = 'Effective Diffuse Ratio'

SOLSTICES = [SUMMER, SPRING, WINTER]

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
resample_rate = {
    'A': 'Annual',
    'M': 'Monthly',
    'D': 'Daily',
    'H': 'Hourly',
}
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


class BatchPVSystResults(object):
    # an instance of this class creates a dict that stores unpacked pvsyst results in the format:
    # dict key = variant number
    # dict value = pandas dataframe of pvsyst results for that variant
    #
    # PVSyst differentiates variants of a project by appending the project name with '0', '1', '2', ... etc.
    # The user should define the list of variant suffixes to be considered.
    # The user should also define a list of values that correspond to the project variants.
    # The value might indicate the parameter that changes from one variant to another.
    #
    # For example, if a user has created pvsyst variants to study the effect of changing GCR,
    # then the user should define the variant list as ['0', '1', '2'] and the parameter list as the
    # range of GCRs studied [0.2, 0.5, 0.8].

    def __init__(self, location, parameter_name, parameter_list, variant_list):
        self.location = location
        self.parameter_name = parameter_name
        self.parameter_list = parameter_list
        self.variant_list = variant_list
        self.parameter_variant_map = dict(zip(parameter_list, variant_list))
        self.variant_parameter_map = dict(zip(variant_list, parameter_list))

        self.results_dict = {}
        for v in variant_list:
            print "loading variant: " + v
            self.results_dict[v] = self.unpack_single_pvsyst_result(v)
        print "done!"

        self.get_site_diffuseness()
        daytime_df_index = get_daytime_df(self.results_dict[variant_list[0]]).index
        self.avg_diffuse = self.results_dict[variant_list[0]].loc[daytime_df_index, H_DIFF_RATIO].mean()

    def unpack_single_pvsyst_result(self, v):
        filename = '../PVresearch/' + self.location + '_Project_HourlyRes_' + v + '.CSV'
        # Unpack results from file:
        # Read in PVSyst results file as data frame:
        # columns: HSol (sun height), AzSol (sun azimuth), AngInc (incidence angle), AngProf (profile angle),
        # PlTilt (plane tilt: tracking), PlAzim (plane azimuth: tracking), PhiAng (phi angle: tracking),
        # DiffHor, BeamHor, GlobHor, T Amb, GlobInc, BeamInc, DifSInc, Alb Inc,
        # GlobIAM, GlobEFf, BeamEff, DiffEff, Alb_Eff
        result_df = pd.read_csv(filename, sep=';', header=0, skiprows=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11],
                                index_col=0, parse_dates=True, dayfirst=True)
        result_df.rename(columns=pvsyst_rename, inplace=True)
        return result_df

    def export_as_csv(self):
        pvsyst_rename_inv = {p[1]: p[0] for p in pvsyst_rename.iteritems()}
        for d in self.results_dict.iteritems():
            # Convert names back to PVSyst convention names:
            d[1].rename(columns=pvsyst_rename_inv, inplace=True)
            print "saving to .csv file: " + d[0]
            d[1].to_csv(self.location + '_' + d[0] + '_' + self.parameter_name)
            print "done!"

    def get_site_diffuseness(self):
        for r in self.results_dict.iteritems():
            h_diffuse_ratio = r[1][H_DIFFUSE] / r[1][H_GLOBAL]
            i_diffuse_ratio = r[1][I_DIFF_S] / r[1][I_GLOBAL]
            e_diffuse_ratio = r[1][E_DIFF_S] / r[1][E_GLOBAL]
            self.results_dict[r[0]] = pd.concat([r[1],
                                                 pd.DataFrame({
                                                     H_DIFF_RATIO: h_diffuse_ratio,
                                                     I_DIFF_RATIO: i_diffuse_ratio,
                                                     E_DIFF_RATIO: e_diffuse_ratio,
                                                 })], axis=1)

    def plot_site_diffuseness(self):
        for r in self.results_dict.iteritems():
            r[1][E_DIFF_RATIO].plot.hist(bins=20)
            plt.title(self.location, fontsize=16)
            plt.xlabel(E_DIFF_RATIO)
            plt.ylim([0, 1000])

    def get_diffuse_tracking(self, flat_result):
        # Compare "effective" irradiance from two simulations, where "effective" accounts for linear effect of 3D near
        # shadings construction & for IAM (impt at low sun angles!):
        # Outputs: a dataframe with diffuse metrics concatenated onto input tracking_df
        # Inputs: tracking_df = results df of simulation of arrays on N/S axis with E/W tracking on 0 deg tilt
        #         flat_df = results df of simulation of arrays w/o tracking on 0 deg tilt
        flat_df = flat_result.results_dict[flat_result.results_dict.keys()[0]]
        for r in self.results_dict.iteritems():

            diffuse_tracking_gain = flat_df[E_GLOBAL] - r[1][E_GLOBAL]
            is_diffuse_tracking_better = (diffuse_tracking_gain > 1.0) & (r[1][E_GLOBAL] > 1.0)
            diffuse_tracking_gain_ratio = diffuse_tracking_gain / r[1][E_GLOBAL]
            when_flat = pd.Series([0]*8760, index=flat_df.index)[is_diffuse_tracking_better]
            when_tracking = r[1].loc[(is_diffuse_tracking_better == False), ANG_TRACK]
            effective_tracker_angle = pd.concat([when_flat, when_tracking])
            effective_tracker_angle.sort_index()
            self.results_dict[r[0]] = pd.concat([r[1],
                                                 pd.DataFrame({
                                                     IS_DT: is_diffuse_tracking_better,
                                                     DT_ANG_TRACK: effective_tracker_angle,
                                                     DT_GAIN: diffuse_tracking_gain,
                                                     DT_GAIN_FACTOR: diffuse_tracking_gain_ratio,
                                                 })], axis=1)

    def get_diffuse_tracking_improvement(self, resample_str):
        # Only want to resample values when diffuse makes sense (i.e. "gain" is positive values).
        # Output: 2-tuple of (absolute gain, gain factor)
        # Input: df with diffuse values, string representing resample interval ('A' = annual, 'M' = monthly, etc.)
        # Setting resample interval to 'H' should result in same as DT_GAIN and DT_FACTOR.
        diffuse_track_gain = {r[0]: (r[1].loc[r[1][IS_DT], DT_GAIN]).resample(resample_str).sum()
                              for r in self.results_dict.iteritems()}
        diffuse_track_factor = pd.DataFrame(
            {r[0]: diffuse_track_gain[r[0]] / r[1][E_GLOBAL].resample(resample_str).sum()
             for r in self.results_dict.iteritems()})
        return diffuse_track_factor

    def plot_diffuse_tracking_improvement(self, resample_str, title_str=''):
        # Only want to resample values when diffuse makes sense (i.e. "gain" is positive values).
        # Output: 2-tuple of (absolute gain, gain factor)
        # Input: df with diffuse values, string representing resample interval ('A' = annual, 'M' = monthly, etc.)
        # Setting resample interval to 'H' should result in same as DT_GAIN and DT_FACTOR.

        diffuse_track_gain = {r[0]: (r[1].loc[r[1][IS_DT], DT_GAIN]).resample(resample_str).sum()
                              for r in self.results_dict.iteritems()}

        if resample_str == 'A':
            parameter = []
            diffuse_track_factor = []

            for r in self.results_dict.iteritems():
                parameter.append(self.variant_parameter_map[r[0]])
                diffuse_track_factor.append(diffuse_track_gain[r[0]] / r[1][E_GLOBAL].resample(resample_str).sum())
            plt.figure(figsize=(8, 8))
            plt.plot(parameter, diffuse_track_factor, 'bo')
            plt.xlabel('Ground coverage ratio, GCR', fontsize=12)
            plt.xlim([0.0, 1.0])

        else:
            diffuse_track_factor = pd.DataFrame(
                {(str(self.variant_parameter_map[r[0]]) + self.parameter_name):
                     diffuse_track_gain[r[0]] / r[1][E_GLOBAL].resample(resample_str).sum()
                 for r in self.results_dict.iteritems()})
            plt.figure(figsize=(15, 8))
            plt.plot(diffuse_track_factor)
            plt.legend(diffuse_track_factor.columns)

        plt.ylabel('Improvement ratio', fontsize=12)
        plt.ylim([0.0, 0.05])
        plt.title(self.location + ': ' + '\nIrradiance Improvement with Diffuse Tracking ' + title_str, fontsize=16)

    def get_single_factor_df(self, single_factor):
        # Creates a dataframe by pulling out one column from each df in the df_dict.  Plotting is a bit easier when all
        # columns are the same factor.
        # E.G. get_single_factor_df(df_dict, ANG_INC) copies angle of incidence values from each df in df_dict and
        # creates a new df.  Column names are the simulation variant names.
        single_factor_df = pd.DataFrame(
            {(str(self.variant_parameter_map[r[0]]) + self.parameter_name): r[1][single_factor]
             for r in self.results_dict.iteritems()})
        return single_factor_df

    def plot_single_factor_multiple_days(self, single_factor, days_list=SOLSTICES, title_str=''):
        # Output: graph with subplots for each day
        # Inputs: df_dict: dict of dataframes of PVsyst result
        #        single_factor: keyword name of data to plot
        #        days_list: list of datetimes as strings in format %Y-%m-%d; used for subplots' titles
        single_factor_df = self.get_single_factor_df(single_factor)
        legend_list = list(single_factor_df.columns)

        plt.figure(figsize=(8, 15))
        plt.suptitle(self.location + ': ' + single_factor + '\n' + title_str, fontsize=16)

        ax = []
        for d in range(1, len(days_list) + 1):
            if d == 1:
                ax.append(plt.subplot(len(days_list), 1, d))
                plt.ylabel(get_units(single_factor), fontsize=14)
            else:
                ax.append(plt.subplot(len(days_list), 1, d, sharex=ax[0], sharey=ax[0]))
            plt.grid(True, which='major', axis='x')
            ax[d - 1].plot(single_factor_df[days_list[d - 1]].index.hour, single_factor_df[days_list[d - 1]])
            # if show_flat == True:
            #     ax[d - 1].plot(flat_df[days_list[d - 1]].index.hour, flat_df[days_list[d - 1]], 'k--')
            ax[d - 1].legend(legend_list, loc='lower right', fontsize=9)
            ax[d - 1].set_title(days_list[d - 1], fontsize=12)

        plt.xlabel('Hour of day', fontsize=14)
        ax[0].set_xlim([0, 24])
        ax[0].set_xticks([0, 6, 12, 18, 24])
        ax[0].set_ylim(get_ylim(single_factor))

    def plot_single_factor_hourly(self, single_factor, variant, title_str=''):
        # Output: graph plotting hourly values of single_factor
        # Inputs: single_factor: keyword name of data to plot
        #        (optional) backtracking: set False if you want no backtacking
        #        (optional) show_flat: set True if you also want to plot results from "flat" simulation
        plt.plot(self.results_dict[variant][single_factor], 'b.')
        # plt.figure(figsize=(15, 8))
        plt.title(self.location + ': ' + single_factor + '\n' +
                  title_str, fontsize=16)


def compare_single_factor_across_batches(batches_list, variants_list, single_factor, days_list=SOLSTICES):
    results_df_list = []
    legend_list = []
    for b in range(len(batches_list)):
        results_df_list.append(batches_list[b].results_dict.get(variants_list[b]))
        legend_list.append(batches_list[b].location[:11] +
                           str(batches_list[b].variant_parameter_map[variants_list[b]]) +
                           batches_list[b].parameter_name)

    plt.figure(figsize=(8, 15))
    plt.suptitle(single_factor, fontsize=16)

    ax = []
    for d in range(1, len(days_list) + 1):
        if d == 1:
            ax.append(plt.subplot(len(days_list), 1, d))
            plt.ylabel(get_units(single_factor), fontsize=14)
        else:
            ax.append(plt.subplot(len(days_list), 1, d, sharex=ax[0], sharey=ax[0]))
        plt.grid(True, which='major', axis='x')
        [ax[d - 1].plot(r[days_list[d - 1]].index.hour, r.loc[days_list[d - 1], single_factor]) for r in results_df_list]
        ax[d - 1].set_title(days_list[d - 1], fontsize=12)

    plt.xlabel('Hour of day', fontsize=14)
    ax[0].set_xlim([0, 24])
    ax[0].set_xticks([0, 6, 12, 18, 24])
    ax[0].set_ylim(get_ylim(single_factor))
    ax[0].legend(legend_list, loc='lower right', fontsize=9)


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
        new_ylim = [0, 1]
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
        units = 'units'
    return units


def get_cloudy_days(results, num_days=3, is_diffuse=True, diffuse_ratio_threshold=0.7, min_day_fraction=0.75):
    # Sorts days of the year based on SKY diffuse ratio (ignoring albedo).  Use keyword args to select for high
    # diffuse (cloudy) or low diffuse (clear) days.
    # Outputs: pandas series for days where diffuse ratio meets a limit
    # Inputs: dataframe with the effective sky diffuse irradiance and
    #         optional kwargs: diffuse boolean (set to False to find clear days),
    #                          diffuse ratio threshold,
    #                          minimum number of hours at or above threshold
    diffuse_ratio = results[E_DIFF_S] / results[E_GLOBAL]
    day_count = diffuse_ratio.resample('D').count()
    if is_diffuse is True:
        diffuse_count = diffuse_ratio[diffuse_ratio > diffuse_ratio_threshold].resample('D').count()
    else:
        diffuse_count = diffuse_ratio[diffuse_ratio < diffuse_ratio_threshold].resample('D').count()
    day_fraction = diffuse_count / day_count
    sorted_by_diffuse_ratio = day_fraction[day_fraction > min_day_fraction].sort_values(ascending=False)
    return sorted_by_diffuse_ratio.index[:num_days].strftime('%Y-%m-%d')


def get_daytime_df(df, threshold=10.0):
    # Use this method on any dataframe to get only the hours when Global Incident Irradiance > threshold W/m2.
    daytime_df = df[df[I_GLOBAL] > float(threshold)]
    return daytime_df







# # now trying to load it back to dataframe:
# reloaded = pd.read_csv('backtrack_diffuse_' + LOCATION + '_' + pvsyst_variant_map['1'], sep=',', header=0,
#                        index_col=0, parse_dates=True, dayfirst=True)


# pvsyst_results = BatchPVSystResults('Hayward Air Term')
# pvsyst_results.plot_single_factor_multiple_days(ANG_TRACK, backtracking=True, show_flat=False)
# pvsyst_results.plot_single_factor_multiple_days(SF_DIRECT, backtracking=True, show_flat=True)
# pvsyst_results.plot_single_factor_multiple_days(SF_DIFF_S, backtracking=True, show_flat=True)
# cloudy_days = pvsyst_results.sort_diffuse_days(pvsyst_results.backtrack_results['1'],
#                                                is_diffuse=True).index[:4].strftime('%Y-%m-%d')
# partly_cloudy_days = pvsyst_results.sort_diffuse_days(pvsyst_results.backtrack_results['1'],
#                                                       min_day_fraction=0.6,
#                                                       is_diffuse=True).index[:4].strftime('%Y-%m-%d')
# clear_days = pvsyst_results.sort_diffuse_days(pvsyst_results.backtrack_results['1'],
#                                               is_diffuse=False).index[:4].strftime('%Y-%m-%d')
# pvsyst_results.plot_single_factor_multiple_days(E_GLOBAL, days_list=cloudy_days,
#                                                 backtracking=True, show_flat=True, title_str='on Cloudy Days')
# pvsyst_results.plot_single_factor_multiple_days(E_GLOBAL, days_list=clear_days,
#                                                 backtracking=True, show_flat=True, title_str='on Clear Days')
# pvsyst_results.plot_single_factor_multiple_days(DT_ANG_TRACK, days_list=partly_cloudy_days,
#                                                 backtracking=True, show_flat=False, title_str='on Partly Cloudy Days')
#
# pvsyst_results.plot_diffuse_tracking_improvement('A', backtracking=True)
# pvsyst_results.plot_diffuse_tracking_improvement('M', backtracking=True)
#
# pvsyst_results.plot_single_factor_hourly(SF_DIRECT, backtracking=False)