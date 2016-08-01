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

no_backtrack_variants = ['0', '2', '4', '6', '8', 'A', 'C']
no_backtrack_inv_gcr_map = dict(zip([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], no_backtrack_variants))
backtrack_variants = ['1', '3', '5', '7', '9', 'B', 'D']
backtrack_inv_gcr_map = dict(zip([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], backtrack_variants))


class BatchPVSystResults(object):

    def __init__(self, location):
        self.location = location

        print "loading no_backtrack files..."
        self.no_backtrack_results = {v: self.unpack_single_pvsyst_result(v) for v in no_backtrack_variants}

        print "loading backtrack files..."
        self.backtrack_results = {v: self.unpack_single_pvsyst_result(v) for v in backtrack_variants}

        print "loading flat file..."
        self.flat_result = self.unpack_single_pvsyst_result('E')

        print "calculating diffuse tracking..."
        self.no_backtrack_diffuse_results = {v[0]: self.get_diffuse_tracking(v[1]) for v in
                                             self.no_backtrack_results.iteritems()}
        self.backtrack_diffuse_results = {v[0]: self.get_diffuse_tracking(v[1]) for v in
                                          self.backtrack_results.iteritems()}
        print "done!"

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

    def get_diffuse_tracking(self, tracking_df):
        # Compare "effective" irradiance from two simulations, where "effective" accounts for linear effect of 3D near
        # shadings construction & for IAM (impt at low sun angles!):
        # Outputs: a dataframe with diffuse metrics concatenated onto input tracking_df
        # Inputs: tracking_df = results df of simulation of arrays on N/S axis with E/W tracking on 0 deg tilt
        #         flat_df = results df of simulation of arrays w/o tracking on 0 deg tilt

        diffuse_tracking_gain = self.flat_result[E_GLOBAL] - tracking_df[E_GLOBAL]
        is_diffuse_tracking_better = (diffuse_tracking_gain > 1.0) & (tracking_df[E_GLOBAL] > 1.0)
        diffuse_tracking_gain_ratio = diffuse_tracking_gain / tracking_df[E_GLOBAL]
        when_flat = pd.Series([0]*8760, index=self.flat_result.index)[is_diffuse_tracking_better]
        when_tracking = tracking_df.loc[(is_diffuse_tracking_better == False), ANG_TRACK]
        effective_tracker_angle = pd.concat([when_flat, when_tracking])
        effective_tracker_angle.sort_index()

        return pd.concat([tracking_df,
                          pd.DataFrame({IS_DT: is_diffuse_tracking_better,
                                        DT_ANG_TRACK: effective_tracker_angle,
                                        DT_GAIN: diffuse_tracking_gain,
                                        DT_GAIN_FACTOR: diffuse_tracking_gain_ratio})], axis=1)

    def get_diffuse_tracking_improvement(self, resample_str, backtracking=True):
        # Only want to resample values when diffuse makes sense (i.e. "gain" is positive values).
        # Output: 2-tuple of (absolute gain, gain factor)
        # Input: df with diffuse values, string representing resample interval ('A' = annual, 'M' = monthly, etc.)
        # Setting resample interval to 'H' should result in same as DT_GAIN and DT_FACTOR.
        if backtracking == True:
            results = self.backtrack_diffuse_results
        else:
            results = self.no_backtrack_diffuse_results

        resample_gain = {r[0]: (r[1].loc[r[1][IS_DT], DT_GAIN]).resample(resample_str).sum()
                         for r in results.iteritems()}
        resample_factor = {gcr_map[r[0]]: resample_gain[r[0]] / r[1][E_GLOBAL].resample(resample_str).sum()
                           for r in results.iteritems()}

        # if resample_str == 'A':
        #     resample_factor = pd.DataFrame(
        #         {gcr_map[r[0]]: resample_gain[r[0]] / r[1][E_GLOBAL].resample(resample_str).sum()
        #          for r in results.iteritems()})
        # else:
        #     resample_factor = pd.DataFrame(
        #         {pvsyst_variant_map[r[0]]: resample_gain[r[0]] / r[1][E_GLOBAL].resample(resample_str).sum()
        #          for r in results.iteritems()})
        return resample_factor

    def plot_diffuse_tracking_improvement(self, resample_str, backtracking=True,
                                          title_str='Irradiance Improvement with Diffuse Tracking'):
        # Only want to resample values when diffuse makes sense (i.e. "gain" is positive values).
        # Output: 2-tuple of (absolute gain, gain factor)
        # Input: df with diffuse values, string representing resample interval ('A' = annual, 'M' = monthly, etc.)
        # Setting resample interval to 'H' should result in same as DT_GAIN and DT_FACTOR.
        if backtracking == True:
            results = self.backtrack_diffuse_results
            title_str += '\n' + 'w/ Backtracking'
        else:
            results = self.no_backtrack_diffuse_results
            title_str += '\n' + ' w/o Backtracking'
        title_str += (' (' + resample_rate[resample_str] + ')')

        resample_gain = {r[0]: (r[1].loc[r[1][IS_DT], DT_GAIN]).resample(resample_str).sum()
                         for r in results.iteritems()}

        if resample_str == 'A':
            resample_factor = {gcr_map[r[0]]: resample_gain[r[0]] / r[1][E_GLOBAL].resample(resample_str).sum()
                 for r in results.iteritems()}
            plt.figure(figsize=(8, 8))
            plt.plot(resample_factor.keys(), resample_factor.values(), 'bo')
            plt.xlabel('Ground coverage ratio, GCR', fontsize=12)
            plt.xlim([0.0, 1.0])

        else:
            resample_factor = pd.DataFrame(
                {pvsyst_variant_map[r[0]]: resample_gain[r[0]] / r[1][E_GLOBAL].resample(resample_str).sum()
                 for r in results.iteritems()})
            plt.figure(figsize=(15, 8))
            plt.plot(resample_factor)

        plt.ylabel('Improvement ratio', fontsize=12)
        plt.ylim([0.0, 0.05])
        plt.title(self.location + ': ' + title_str, fontsize=16)

    def plot_single_factor_multiple_days(self, single_factor, days_list=SOLSTICES,
                                         backtracking=True, show_flat=False, title_str=''):
        # Output: graph with subplots for each day
        # Inputs: df_dict: dict of dataframes of PVsyst result
        #        single_factor: keyword name of data to plot
        #        days_list: list of datetimes as strings in format %Y-%m-%d; used for subplots' titles
        #        (optional) backtracking: set False if you want no backtacking
        #        (optional) show_flat: set True if you also want to plot results from "flat" simulation

        if backtracking == True:
            results = self.backtrack_diffuse_results
            title_str += ' w/ Backtracking'
        else:
            results = self.no_backtrack_diffuse_results
            title_str += ' w/o Backtracking'

        single_factor_df = self.get_single_factor_df(results, single_factor)
        legend_list = list(single_factor_df.columns)

        if show_flat == True:
            flat_df = self.flat_result[single_factor]
            legend_list.append(pvsyst_variant_map['E'])
        plt.figure(figsize=(8, 15))
        plt.suptitle(self.location + ': ' + single_factor + '\n' + title_str, fontsize=16)

        ax = []
        for d in range(1, len(days_list) + 1):
            if d == 1:
                ax.append(plt.subplot(len(days_list), 1, d))
                plt.ylabel(self.get_units(single_factor), fontsize=14)
            else:
                ax.append(plt.subplot(len(days_list), 1, d, sharex=ax[0], sharey=ax[0]))
            plt.grid(True, which='major', axis='x')
            ax[d - 1].plot(single_factor_df[days_list[d - 1]].index.hour, single_factor_df[days_list[d - 1]])
            if show_flat == True:
                ax[d - 1].plot(flat_df[days_list[d - 1]].index.hour, flat_df[days_list[d - 1]], 'k--')
            ax[d - 1].legend(legend_list, loc='lower right', fontsize=9)
            ax[d - 1].set_title(days_list[d - 1], fontsize=12)

        plt.xlabel('Hour of day', fontsize=14)
        ax[0].set_xlim([0, 24])
        ax[0].set_xticks([0, 6, 12, 18, 24])
        ax[0].set_ylim(self.get_ylim(single_factor))

    def plot_single_factor_hourly(self, single_factor, gcr=0.4, backtracking=True, title_str=''):
        # Output: graph plotting hourly values of single_factor
        # Inputs: single_factor: keyword name of data to plot
        #        (optional) backtracking: set False if you want no backtacking
        #        (optional) show_flat: set True if you also want to plot results from "flat" simulation

        if backtracking == True:
            results_df = self.backtrack_diffuse_results[backtrack_inv_gcr_map[gcr]]
            title_str += ' w/ Backtracking'
        else:
            results_df = self.no_backtrack_diffuse_results[no_backtrack_inv_gcr_map[gcr]]
            title_str += ' w/o Backtracking'

        plt.plot(results_df[single_factor], 'b.')
        # plt.figure(figsize=(15, 8))
        plt.title(self.location + ': ' + single_factor + '\n' + title_str, fontsize=16)

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def get_single_factor_df(df_dict, single_factor):
        # Creates a dataframe by pulling out one column from each df in the df_dict.  Plotting is a bit easier when all
        # columns are the same factor.
        # E.G. get_single_factor_df(df_dict, ANG_INC) copies angle of incidence values from each df in df_dict and
        # creates a new df.  Column names are the simulation variant names.
        single_factor_df = pd.DataFrame({pvsyst_variant_map[d[0]]: d[1][single_factor] for d in df_dict.iteritems()})
        return single_factor_df

    @staticmethod
    def get_cloudy_days(results, num_days = 3, is_diffuse=True, diffuse_ratio_threshold=0.7, min_day_fraction=0.75):
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

    @staticmethod
    def get_daytime_df(df, threshold=10.0):
        # Use this method on any dataframe to get only the hours when Global Incident Irradiance > threshold W/m2.
        daytime_df = df[df[I_GLOBAL] > float(threshold)]
        return daytime_df





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