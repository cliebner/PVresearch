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
H_GLOBAL = 'Global Horizontal Irradiance'  # available global irradiance on an unshaded horizontal surface (from weather data)
H_DIFFUSE = 'Diffuse Horizontal Irradiance'  # available diffuse irradiance on an unshaded horizontal surface
H_DIRECT = 'Direct Horizontal Irradiance'  # available direct beam irradiance on an unshaded horizontal surface
I_GLOBAL = 'Global Incident Irradiance'  # available global irradiance in POA (no losses accounted)
I_DIRECT = 'Direct Incident Irradiance'  # available direct beam irradiance in POA (no losses accounted)
I_DIFF_G = 'Albedo Incident Irradiance'  # available albedo/ground diffuse irradiance in POA (no losses accounted)
I_DIFF_S = 'Sky Diffuse Incident Irradiance'  # available sky diffuse irradiance in POA (no losses accounted)
I_DIFF_T = 'Total Diffuse Incident Irradiance'  # available total diffuse irradiance in POA (no losses accounted)
E_GLOBAL = 'Effective Global Irradiance'  # effective global irradiance, accounting for hourly shading output + IAM
E_DIRECT = 'Effective Direct Irradiance'  # effective direct beam irradiance, accounting for hourly shading output + IAM
E_DIFF_S = 'Effective Sky Diffuse Irradiance'  # effective sky diffuse irradiance, accounting for hourly shading output + IAM
E_DIFF_G = 'Albedo Effective Irradiance'  # effective albedo irradiance, accounting for hourly shading output + IAM
S_GLOBAL = 'Global Incident w/Shade'  # effective global irradiance, accounting for hourly shading output
SLOSS_GLOBAL = 'Global Shade Loss'  # available global irradiance loss due to hourly shading output
SLOSS_DIRECT = 'Direct Shade Loss'  # available direct beam irradiance loss due to hourly shading output
SLOSS_DIFFUSE = 'Diffuse Shade Loss'  # available diffuse irradiance loss due to hourly shading output
SF_GLOBAL = 'Global Shade Factor'
SF_DIRECT = 'Direct Shade Factor'
SF_DIFF_S = 'Sky Diffuse Shade Factor'
SF_DIFF_G = 'Albedo Shade Factor'
IAM_GLOBAL = 'Global corr for IAM'
ILOSS_GLOBAL = 'Global IAM Loss'
ILOSS_DIRECT = 'Direct IAM Loss'
ILOSS_DIFF_S = 'Sky Diffuse IAM Loss'  # unclear if this is total diffuse or just sky
ILOSS_DIFF_G = 'Albedo IAM Loss'
IAMF_GLOBAL = 'Global IAM Factor'
IAMF_DIRECT = 'Direct IAM Factor'
IAMF_DIFF_S = 'Sky Diffuse IAM Factor'
IAMF_DIFF_G = 'Albedo IAM Factor'
IS_DT = 'Diffuse Tracking?'
DT_GAIN = 'DiffTrack Gain'
DT_GAIN_FACTOR = 'DiffTrack Gain Factor'
DT_ANG_TRACK = 'DiffTrack Tracker Angle'
DT_E_GLOBAL = 'DiffTrack Eff Global Irradiance'
DT_ENERGY_ARRAY = 'DiffTrack Array Output Energy'
DT_ENERGY_GRID = 'DiffTrack Energy to Grid'
H_DIFF_RATIO = 'Horizontal Diffuse Ratio'
I_DIFF_RATIO = 'Incident Diffuse Ratio'
E_DIFF_RATIO = 'Effective Diffuse Ratio'
RATIO_DIFF_S = 'Incident Sky Diffuse Ratio'
ENERGY_ARRAY = 'Array Output Energy'
PVLOSS_IRRADIANCE = 'PV Loss due to Irradiance'
PVLOSS_TEMP = 'PV Loss due to Temp'
ENERGY_GRID = 'Energy to Grid'
SCALE = 'scale output'
LIM = 'limits'
UNITS = 'units'

SOLSTICES = [SUMMER, SPRING, WINTER]

outputs_list = [SF_GLOBAL, SF_DIRECT, SF_DIFF_S, SF_DIFF_G, IAMF_GLOBAL, IAMF_DIRECT, IAMF_DIFF_S, IAMF_DIFF_G,
                DT_GAIN_FACTOR]
outputs_range = [0.0, 1.1]
angles_list = [ANG_ELEV, ANG_AZ, ANG_INC, ANG_PROF, ANG_SURF_AZ, ANG_SURF_TILT, ANG_TRACK, ANG_ZENITH,
               DT_ANG_TRACK]
angles_range = [-90.0, 90.0]
irradiance_list = [H_GLOBAL, H_DIFFUSE, H_DIRECT, I_GLOBAL, I_DIRECT, I_DIFF_G, I_DIFF_S, I_DIFF_T,
                   E_GLOBAL, E_DIRECT, E_DIFF_S, E_DIFF_G, S_GLOBAL, SLOSS_GLOBAL, SLOSS_DIRECT, SLOSS_DIFFUSE, DT_GAIN]
irradiance_range = [0.0, 1000.0]
energy_list = [ENERGY_ARRAY, ENERGY_GRID, PVLOSS_IRRADIANCE, PVLOSS_TEMP]
energy_range = [0.0, 1000.0]
boolean_list = [IS_DT]
boolean_range = [-1.5, 1.5]
resample_rate_dict = {
    'A': 'Annual',
    'M': 'Monthly',
    'W': 'Weekly',
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
    'DifS/Gl': RATIO_DIFF_S,
    'EArray': ENERGY_ARRAY,  # W
    'GIncLss': PVLOSS_IRRADIANCE,  # W
    'TempLss': PVLOSS_TEMP,  # W
    'E_Grid': ENERGY_GRID,  # W
    'GlobIAM': IAM_GLOBAL,  # W/m2
    'IAMLoss': ILOSS_GLOBAL,  #W/m2
    'IAMBLss': ILOSS_DIRECT,  #W/m2
    'IAMDLss': ILOSS_DIFF_S,  # W/m2
    'IAMALss': ILOSS_DIFF_G,  # W/m2
}


class BatchPVSystResults(object):
    # an instance of this class creates a dict that stores unpacked pvsyst results in the format:
    # dict key = variant number
    # dict value = pandas dataframe of pvsyst results for that variant
    #
    # PVSyst differentiates variants of a project by appending the project name with '0', '1', '2', ... etc.
    # The user should define the list of variant suffixes to be considered.
    # The user should also define a list of values that correspond to the project variants.
    # The value might indicate the output that changes from one variant to another.
    #
    # For example, if a user has created pvsyst variants to study the effect of changing GCR,
    # then the user should define the variant list as ['0', '1', '2'] and the output list as the
    # range of GCRs studied [0.2, 0.5, 0.8].

    def __init__(self, location, output_name, output_list, variant_list):
        self.location = location
        self.output_name = output_name
        self.output_list = output_list
        self.variant_list = variant_list
        self.output_variant_map = dict(zip(output_list, variant_list))
        self.variant_output_map = dict(zip(variant_list, output_list))
        self.name_list = [(str(p) + self.output_name) for p in self.output_list]

        self.results_dict = {}
        for v in variant_list:
            print "loading variant: " + v
            self.results_dict[v] = self.unpack_single_pvsyst_result(v)
        print "done!"

        self.get_site_diffuseness()
        self.daytime_df_index = get_daytime_df(self.results_dict[variant_list[0]]).index
        self.avg_diffuse = {r[0]: r[1].loc[self.daytime_df_index, H_DIFF_RATIO].mean()
                            for r in self.results_dict.iteritems()}

    def unpack_single_pvsyst_result(self, v):
        filename = '../PVresearch/' + self.location + '_Project_HourlyRes_' + v + '.CSV'
        # Unpack results from file:
        # Read in PVSyst results file as data frame:
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
            d[1].to_csv(self.location + '_' + d[0] + '_' + self.output_name)
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
        # Compare "effective" irradiance, energy from array, and energy to grid from two simulations,
        # where "effective" accounts for linear effect of 3D near shadings construction and
        # for IAM (impt at low sun angles!).
        #
        # Outputs: a dataframe with diffuse metrics concatenated onto self.results_dict
        # Inputs: flat_result = BatchPVSyst object for a simulation of arrays w/o tracking on 0 deg tilt
        #
        flat_df = flat_result.results_dict[flat_result.results_dict.keys()[0]]
        for r in self.results_dict.iteritems():
            # find when diffuse tracking makes sense:
            diffuse_tracking_gain_eff_irrad = flat_df[E_GLOBAL] - r[1][E_GLOBAL]
            is_diffuse_tracking_better = (diffuse_tracking_gain_eff_irrad > 1.0) & (r[1][E_GLOBAL] > 1.0)
            diffuse_tracking_gain_ratio = diffuse_tracking_gain_eff_irrad / r[1][E_GLOBAL]

            # pull values from tracking and flat df's as appropriate:
            track_angle_when_flat = pd.Series([0]*8760, index=flat_df.index)[is_diffuse_tracking_better]
            track_angle_when_tracking = r[1].loc[(is_diffuse_tracking_better == False), ANG_TRACK]
            eff_irrad_when_flat = r[1].loc[(is_diffuse_tracking_better == True), E_GLOBAL]
            eff_irrad_when_tracking = r[1].loc[(is_diffuse_tracking_better == False), E_GLOBAL]
            energy_array_when_flat = r[1].loc[(is_diffuse_tracking_better == True), ENERGY_ARRAY]
            energy_array_when_tracking = r[1].loc[(is_diffuse_tracking_better == False), ENERGY_ARRAY]
            energy_grid_when_flat = r[1].loc[(is_diffuse_tracking_better == True), ENERGY_GRID]
            energy_grid_when_tracking = r[1].loc[(is_diffuse_tracking_better == False), ENERGY_GRID]

            # merge series:
            dt_track_angle = pd.concat([track_angle_when_flat, track_angle_when_tracking])
            dt_track_angle.sort_index()
            dt_eff_irrad = pd.concat([eff_irrad_when_flat, eff_irrad_when_tracking])
            dt_eff_irrad.sort_index()
            dt_energy_array = pd.concat([energy_array_when_flat, energy_array_when_tracking])
            dt_energy_array.sort_index()
            dt_energy_grid = pd.concat([energy_grid_when_flat, energy_grid_when_tracking])
            dt_energy_grid.sort_index()

            # throw it all at the end of the existing df!
            self.results_dict[r[0]] = pd.concat([r[1],
                                                 pd.DataFrame({
                                                     IS_DT: is_diffuse_tracking_better,
                                                     DT_ANG_TRACK: dt_track_angle,
                                                     DT_GAIN: diffuse_tracking_gain_eff_irrad,
                                                     DT_GAIN_FACTOR: diffuse_tracking_gain_ratio,
                                                     DT_E_GLOBAL: dt_eff_irrad,
                                                     DT_ENERGY_ARRAY: dt_energy_array,
                                                     DT_ENERGY_GRID: dt_energy_grid,
                                                 })], axis=1)

    def get_diffuse_tracking_improvement(self, resample_str):
        # Only want to resample values when diffuse makes sense (i.e. "gain" is positive values).
        # Output: 2-tuple of (absolute gain, gain output)
        # Input: df with diffuse values, string representing resample interval ('A' = annual, 'M' = monthly, etc.)
        # Setting resample interval to 'H' should result in same as DT_GAIN and DT_FACTOR.
        diffuse_track_gain = {r[0]: (r[1].loc[r[1][IS_DT], DT_GAIN]).resample(resample_str).sum()
                              for r in self.results_dict.iteritems()}
        diffuse_track_output = pd.DataFrame(
            {r[0]: diffuse_track_gain[r[0]] / r[1][E_GLOBAL].resample(resample_str).sum()
             for r in self.results_dict.iteritems()})
        return diffuse_track_output

    def plot_diffuse_tracking_improvement(self, resample_str, title_str=''):
        # Only want to resample values when diffuse makes sense (i.e. "gain" is positive values).
        # Output: 2-tuple of (absolute gain, gain output)
        # Input: df with diffuse values, string representing resample interval ('A' = annual, 'M' = monthly, etc.)
        # Setting resample interval to 'H' should result in same as DT_GAIN and DT_FACTOR.

        diffuse_track_gain = {r[0]: (r[1].loc[r[1][IS_DT], DT_GAIN]).resample(resample_str).sum()
                              for r in self.results_dict.iteritems()}

        if resample_str == 'A':
            output = []
            diffuse_track_output = []

            for r in self.results_dict.iteritems():
                output.append(self.variant_output_map[r[0]])
                diffuse_track_output.append(diffuse_track_gain[r[0]] / r[1][E_GLOBAL].resample(resample_str).sum())
            plt.figure(figsize=(8, 8))
            plt.plot(output, diffuse_track_output, 'bo')
            plt.xlabel('Ground coverage ratio, GCR', fontsize=12)
            plt.xlim([0.0, 1.0])

        else:
            diffuse_track_output = pd.DataFrame(
                {(str(self.variant_output_map[r[0]]) + self.output_name):
                     diffuse_track_gain[r[0]] / r[1][E_GLOBAL].resample(resample_str).sum()
                 for r in self.results_dict.iteritems()})
            plt.figure(figsize=(15, 8))
            plt.plot(diffuse_track_output)
            plt.legend(diffuse_track_output.columns)

        plt.ylabel('Improvement ratio', fontsize=12)
        plt.ylim([0.0, 0.05])
        plt.title(self.location + ': ' + '\nIrradiance Improvement with Diffuse Tracking ' + title_str, fontsize=16)

    def get_single_output_df(self, single_output, resample_rate='H'):
        # Creates a dataframe by pulling out one column from each df in the df_dict.  Plotting is a bit easier when all
        # columns are the same output.
        # E.G. get_single_output_df(df_dict, ANG_INC) copies angle of incidence values from each df in df_dict and
        # creates a new df.  Column names are the simulation variant names.
        single_output_df = pd.DataFrame(
            {(str(self.variant_output_map[r[0]]) + self.output_name): r[1][single_output].resample(resample_rate).sum()
             for r in self.results_dict.iteritems()})
        return single_output_df

    def plot_single_output_multiple_days(self, single_output, days_list=SOLSTICES, title_str=''):
        # Output: graph with subplots for each day
        # Inputs: df_dict: dict of dataframes of PVsyst result
        #        single_output: keyword name of data to plot
        #        days_list: list of datetimes as strings in format %Y-%m-%d; used for subplots' titles
        single_output_df = self.get_single_output_df(single_output)
        legend_list = list(single_output_df.columns)
        y_info = get_y_axis_info(single_output)
        plt.figure(figsize=(8, 15))
        plt.suptitle(self.location + ': ' + single_output + '\n' + title_str, fontsize=16)

        ax = []
        for d in range(1, len(days_list) + 1):
            if d == 1:
                ax.append(plt.subplot(len(days_list), 1, d))
                plt.ylabel(y_info[UNITS], fontsize=14)
            else:
                ax.append(plt.subplot(len(days_list), 1, d, sharex=ax[0], sharey=ax[0]))
            plt.grid(True, which='major', axis='x')
            ax[d - 1].plot(single_output_df[days_list[d - 1]].index.hour, single_output_df[days_list[d - 1]]/y_info[SCALE],
                           marker='.')
            # if show_flat == True:
            #     ax[d - 1].plot(flat_df[days_list[d - 1]].index.hour, flat_df[days_list[d - 1]], 'k--')
            ax[d - 1].legend(legend_list, loc='lower right', fontsize=9)
            ax[d - 1].set_title(days_list[d - 1], fontsize=12)

        plt.xlabel('Hour of day', fontsize=14)
        ax[0].set_xlim([0, 24])
        ax[0].set_xticks([0, 6, 12, 18, 24])
        ax[0].set_ylim(y_info[LIM])

    def plot_single_output_hourly(self, single_output, variant, title_str=''):
        # Output: graph plotting hourly values of single_output
        # Inputs: single_output: keyword name of data to plot
        #        (optional) backtracking: set False if you want no backtacking
        #        (optional) show_flat: set True if you also want to plot results from "flat" simulation
        plt.plot(self.results_dict[variant][single_output], 'b.')
        # plt.figure(figsize=(15, 8))
        plt.title(self.location + ': ' + single_output + '\n' +
                  title_str, fontsize=16)

    def compare_single_output_to_baseline(self, single_output, baseline_var=None, resample_rate='H', diff='rel'):
        if baseline_var is None:
            baseline_var = self.variant_list[0]
        if diff == 'abs':  # find absolute difference (i.e. variant - baseline)
            project_diff = pd.DataFrame({(str(self.variant_output_map[r[0]])+self.output_name):
                                              r[1][single_output].resample(resample_rate).sum() -
                                              self.results_dict[baseline_var][single_output].resample(resample_rate).sum()
                                            for r in self.results_dict.iteritems()})
        elif diff == 'rel':  # find relative difference (i.e. ratio)
            project_diff = pd.DataFrame({(str(self.variant_output_map[r[0]]) + self.output_name):
                                        (r[1][single_output].resample(resample_rate).sum() -
                                            self.results_dict[baseline_var][single_output].resample(resample_rate).sum()) /
                                        self.results_dict[baseline_var][single_output].resample(resample_rate).sum()
                                            for r in self.results_dict.iteritems()})
        else:
            print 'diff arg must be rel or abs'
        # title_dict = {
        #     'abs': '\n' + '(Variant - Baseline)',
        #     'rel': ' (Variant - Baseline) / Baseline'
        # }
        # # plot:
        # plot_generic_df(project_diff, single_output, is_ratio=True,
        #                 title_str=title_dict[diff] + '\n' + 'Baseline: ' +
        #                           str(self.variant_output_map[baseline_var]) + self.output_name)

        return project_diff


def plot_generic_df(df, single_output, is_ratio=False, title_str='', show_neg=False):
    y_info = get_y_axis_info(single_output, is_ratio)
    for c in df.columns:
        plt.plot(df.index, df[c]/y_info[SCALE], marker='.')
    plt.legend(df.columns, loc='best')
    # if show_neg is True:
    #     plt.ylim([-1*y_info[LIM][1], -1*y_info[LIM][0]])
    # else:
    #     plt.ylim(y_info[LIM])
    plt.ylabel(y_info[UNITS])
    plt.title(single_output + ' ' + title_str)


def compare_batches_to_baseline(batches_list, variants_list, params_list, names_list, baseline_series=None, resample_rate='H', diff='rel'):
    # Compare any result in any batch to any baseline batch.
    # User's responsbility to compare results that mean something.
    # ORDER MATTERS for batches_list, outputs_list, variants_list, and params_list!!
    # e.g. this method will compare the output[i] of variant[i] of the batch[i] with
    # the output[i+1] of variant[i+1] of batch[i+1], etc.

    if baseline_series is None:
        baseline_series = batches_list[0].results_dict[variants_list[0]][params_list[0]]  # df
    if diff == 'abs':  # find absolute difference (i.e. variant - baseline)
        project_diff = pd.DataFrame({(str(batches_list[b].location[:6]) + '_' + str(names_list[b])):
                                         batches_list[b].results_dict[variants_list[b]][params_list[b]].resample(resample_rate).sum() -
                                         baseline_series.resample(resample_rate).sum()
                                     for b in range(len(batches_list))})
    elif diff == 'rel':  # find relative difference (i.e. ratio)
        project_diff = pd.DataFrame({(str(batches_list[b].location[:6]) + '_' + str(names_list[b])):
                                         (batches_list[b].results_dict[variants_list[b]][params_list[b]].resample(
                                             resample_rate).sum() -
                                          baseline_series.resample(resample_rate).sum()) /
                                         baseline_series.resample(resample_rate).sum()
                                     for b in range(len(batches_list))})
    else:
        print 'diff arg must be rel or abs'
    title_dict = {
        'abs': '\n' + '(Variant - Baseline)',
        'rel': '\n' + ' (Variant - Baseline) / Baseline',
    }
    # # plot:
    # plot_generic_df(project_diff, outputs_list, is_ratio=True,
    #                 title_str=title_dict[diff] + '\n' + 'Baseline: ' +
    #                           str(self.variant_output_map[baseline_series]) + self.output_name)
    plt.plot(project_diff, marker='o', mec='None')
    plt.legend(project_diff.columns, loc='best')
    plt.ylabel(params_list[0])
    # plt.title(title_dict[diff] + '\n' + 'Baseline: ' +


    return project_diff


def compare_single_output_across_batches(batches_list, variants_list, params_list, single_output, days_list=SOLSTICES):
    # ORDER MATTERS for batches_list, variants_list, and params_list!!
    # e.g. this method will compare the param[i] of variant[i] of the batch[i] with
    # the param[i+1] of variant[i+1] of batch[i+1], etc.
    results_df_list = []
    legend_list = []
    for b in range(len(batches_list)):
        results_df_list.append(batches_list[b].results_dict.get(variants_list[b]))
        # legend_list.append(batches_list[b].location[:11] +
        #                    str(batches_list[b].variant_output_map[variants_list[b]]) +
        #                    batches_list[b].output_name)

    plt.figure(figsize=(8, 15))
    plt.suptitle(single_output, fontsize=16)
    y_info = get_y_axis_info(single_output)

    ax = []
    for d in range(1, len(days_list) + 1):
        if d == 1:
            ax.append(plt.subplot(len(days_list), 1, d))
            plt.ylabel(y_info[UNITS], fontsize=14)
        else:
            ax.append(plt.subplot(len(days_list), 1, d, sharex=ax[0], sharey=ax[0]))
        plt.grid(True, which='major', axis='x')
        [ax[d - 1].plot(r[days_list[d - 1]].index.hour, r.loc[days_list[d - 1], single_output]/y_info[SCALE])
         for r in results_df_list]
        ax[d - 1].set_title(days_list[d - 1], fontsize=12)

    plt.xlabel('Hour of day', fontsize=14)
    ax[0].set_xlim([0, 24])
    ax[0].set_xticks([0, 6, 12, 18, 24])
    ax[0].set_ylim(y_info[LIM])
    ax[0].legend(params_list, loc='lower right', fontsize=9)


def get_y_axis_info(single_output):
    if single_output in outputs_list:
        scale_by = 1.0
        ylim = outputs_range
        units = 'output'
    elif single_output in irradiance_list:
        scale_by = 1.0
        ylim = irradiance_range
        units = 'Wh/m2'
    elif single_output in angles_list:
        scale_by = 1.0
        ylim = angles_range
        units = 'Angle (deg)'
    elif single_output in boolean_list:
        scale_by = 1.0
        ylim = boolean_range
        units = '1 = True'
    elif single_output in energy_list:
        scale_by = 1000.0
        ylim = energy_range
        units = 'kWh'
    else:
        scale_by = 1.0
        ylim = [0, 1]
        units = 'units'
    return {SCALE: scale_by, LIM: ylim, UNITS: units}


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



#test:
# cedar_test = BatchPVSystResults('Cedar City Municipal Ap', 'track', [60, 45, 50, 55], ['0', 'A', '9', 'B'])
# plot_generic_df(cedar_test.compare_single_output_to_baseline(ENERGY_ARRAY, baseline_var='0', resample_rate='M'), ENERGY_ARRAY)



# # now trying to load it back to dataframe:
# reloaded = pd.read_csv('backtrack_diffuse_' + LOCATION + '_' + pvsyst_variant_map['1'], sep=',', header=0,
#                        index_col=0, parse_dates=True, dayfirst=True)


# pvsyst_results = BatchPVSystResults('Hayward Air Term')
# pvsyst_results.plot_single_output_multiple_days(ANG_TRACK, backtracking=True, show_flat=False)
# pvsyst_results.plot_single_output_multiple_days(SF_DIRECT, backtracking=True, show_flat=True)
# pvsyst_results.plot_single_output_multiple_days(SF_DIFF_S, backtracking=True, show_flat=True)
# cloudy_days = pvsyst_results.sort_diffuse_days(pvsyst_results.backtrack_results['1'],
#                                                is_diffuse=True).index[:4].strftime('%Y-%m-%d')
# partly_cloudy_days = pvsyst_results.sort_diffuse_days(pvsyst_results.backtrack_results['1'],
#                                                       min_day_fraction=0.6,
#                                                       is_diffuse=True).index[:4].strftime('%Y-%m-%d')
# clear_days = pvsyst_results.sort_diffuse_days(pvsyst_results.backtrack_results['1'],
#                                               is_diffuse=False).index[:4].strftime('%Y-%m-%d')
# pvsyst_results.plot_single_output_multiple_days(E_GLOBAL, days_list=cloudy_days,
#                                                 backtracking=True, show_flat=True, title_str='on Cloudy Days')
# pvsyst_results.plot_single_output_multiple_days(E_GLOBAL, days_list=clear_days,
#                                                 backtracking=True, show_flat=True, title_str='on Clear Days')
# pvsyst_results.plot_single_output_multiple_days(DT_ANG_TRACK, days_list=partly_cloudy_days,
#                                                 backtracking=True, show_flat=False, title_str='on Partly Cloudy Days')
#
# pvsyst_results.plot_diffuse_tracking_improvement('A', backtracking=True)
# pvsyst_results.plot_diffuse_tracking_improvement('M', backtracking=True)
#
# pvsyst_results.plot_single_output_hourly(SF_DIRECT, backtracking=False)