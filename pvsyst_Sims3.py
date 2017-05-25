__author__ = 'christina'

# import matplotlib  # version 1.5.1
# matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # version 0.18
import datetime as dt



class PVSystResult(object):
    '''
    This is a class that defines general methods for interacting with a single PVSyst 'User Hourly" csv file.
    It defines:
    * global variables that map to PVSyst output naming convention
    * method for importing a csv and converting to pandas data frame
    * method for
    '''

    # global variables:
    WINTER = '1990-12-21'
    SUMMER = '1990-06-21'
    SPRING = '1990-03-22'

    # output names:
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

    factors_list = [SF_GLOBAL, SF_DIRECT, SF_DIFF_S, SF_DIFF_G, IAMF_GLOBAL, IAMF_DIRECT, IAMF_DIFF_S, IAMF_DIFF_G,
                    DT_GAIN_FACTOR]
    factors_range = [0.0, 1.1]
    angles_list = [ANG_ELEV, ANG_AZ, ANG_INC, ANG_PROF, ANG_SURF_AZ, ANG_SURF_TILT, ANG_TRACK, ANG_ZENITH,
                   DT_ANG_TRACK]
    angles_range = [-90.0, 90.0]
    irradiance_list = [H_GLOBAL, H_DIFFUSE, H_DIRECT, I_GLOBAL, I_DIRECT, I_DIFF_G, I_DIFF_S, I_DIFF_T,
                       E_GLOBAL, E_DIRECT, E_DIFF_S, E_DIFF_G, S_GLOBAL, SLOSS_GLOBAL, SLOSS_DIRECT, SLOSS_DIFFUSE,
                       DT_GAIN]
    energy_list = [ENERGY_ARRAY, ENERGY_GRID, PVLOSS_IRRADIANCE, PVLOSS_TEMP]
    energy_range = [0.0, 1000.0]
    energy_scales = {
        'A': 0.000001,
        'M': 0.000001,
        'W': 0.001,
        'D': 0.001,
        'H': 1.0,
    }
    energy_units = {
        'A': 'MWh',
        'M': 'MWh',
        'W': 'kWh',
        'D': 'kWh',
        'H': 'Wh',
    }
    boolean_list = [IS_DT]
    boolean_range = [-1.5, 1.5]
    percent_range = [-25.0, 25.0]
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
        'IAMLoss': ILOSS_GLOBAL,  # W/m2
        'IAMBLss': ILOSS_DIRECT,  # W/m2
        'IAMDLss': ILOSS_DIFF_S,  # W/m2
        'IAMALss': ILOSS_DIFF_G,  # W/m2
    }

    def __init__(self, location, variant):
        self.location = location
        self.variant = variant
        self.result_df = self.unpack_single_pvsyst_result(self.variant)

        self.get_site_diffuseness()
        self.daytime_df_index = self.get_daytime_df().index
        self.avg_diffuse = self.result_df.loc[self.daytime_df_index, self.H_DIFF_RATIO].mean()
        self.cloudy_days = self.get_cloudy_days()

    def __str__(self):
        return self.location + ', variant ' + self.variant

    def unpack_single_pvsyst_result(self, v):
        filename = '../PVresearch/' + self.location + '_Project_HourlyRes_' + v + '.CSV'
        # Read in PVSyst results file as data frame:
        result_df = pd.read_csv(filename, sep=';', header=0, skiprows=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11],
                                index_col=0, parse_dates=True, dayfirst=True)
        result_df.rename(columns=self.pvsyst_rename, inplace=True)
        return result_df

    def get_site_diffuseness(self):
        h_diffuse_ratio = self.result_df.loc[:, self.H_DIFFUSE] / self.result_df[self.H_GLOBAL]
        i_diffuse_ratio = self.result_df[self.I_DIFF_S] / self.result_df[self.I_GLOBAL]
        e_diffuse_ratio = self.result_df[self.E_DIFF_S] / self.result_df[self.E_GLOBAL]
        self.result_df = pd.concat([self.result_df,
                                             pd.DataFrame({
                                                 self.H_DIFF_RATIO: h_diffuse_ratio,
                                                 self.I_DIFF_RATIO: i_diffuse_ratio,
                                                 self.E_DIFF_RATIO: e_diffuse_ratio,
                                             })], axis=1)

    def plot_site_diffuseness(self):
        self.result_df[self.E_DIFF_RATIO].plot.hist(bins=20)
        plt.title(self.location, fontsize=16)
        plt.xlabel(self.E_DIFF_RATIO)
        plt.ylim([0, 1000])

    def export_as_csv(self):
        pvsyst_rename_inv = {p[1]: p[0] for p in self.pvsyst_rename.iteritems()}
        # Convert names back to PVSyst convention names:
        self.result_df.rename(columns=pvsyst_rename_inv, inplace=True)
        print "saving to .csv file"
        self.result_df.to_csv(self.location + '_' + self.variant)
        print "done!"

    def plot_single_output_hourly(self, single_output, title_str=''):
        # Output: graph plotting hourly values of single_output
        # Inputs: single_output: keyword name of data to plot
        #        (optional) backtracking: set False if you want no backtacking
        #        (optional) show_flat: set True if you also want to plot results from "flat" simulation
        plt.plot(self.result_df[single_output], 'b.')
        # plt.figure(figsize=(15, 8))
        plt.title(self.location + ': ' + single_output + '\n' +
                  title_str, fontsize=16)

    def get_y_axis_info(self, y_data, resample_rate):
        if y_data in self.factors_list:
            scale_by = 1.0
            ylim = self.factors_range
            units = 'factor'
        elif y_data in self.irradiance_list:
            scale_by = self.energy_scales[resample_rate]
            ylim = self.energy_range
            units = self.energy_units[resample_rate] +'/m2'
        elif y_data in self.energy_list:
            scale_by = self.energy_scales[resample_rate]
            ylim = self.energy_range
            units = self.energy_units[resample_rate]
        elif y_data in self.angles_list:
            scale_by = 1.0
            ylim = self.angles_range
            units = 'Angle (deg)'
        elif y_data in self.boolean_list:
            scale_by = 1.0
            ylim = self.boolean_range
            units = '1 = True'
        elif y_data is 'rel':
            scale_by = 100.0
            ylim = self.percent_range
            units = '%'
        else:
            scale_by = 1.0
            ylim = [0, 1]
            units = 'units'
        return {self.SCALE: scale_by, self.LIM: ylim, self.UNITS: units}

    def get_cloudy_days(self, num_days=3, is_diffuse=True, diffuse_ratio_threshold=0.7, min_day_fraction=0.75):
        ''''
        Sorts days of the year based on SKY diffuse ratio (ignoring albedo).  Use keyword args to select for high
        diffuse (cloudy) or low diffuse (clear) days.
        Outputs: pandas series for days where diffuse ratio meets a limit
        Inputs: dataframe with the effective sky diffuse irradiance and
                optional kwargs: diffuse boolean (set to False to find clear days),
                                 diffuse ratio threshold,
                                 minimum number of hours at or above threshold
        '''
        diffuse_ratio = self.result_df[self.E_DIFF_S] / self.result_df[self.E_GLOBAL]
        day_count = diffuse_ratio.resample('D').count()
        if is_diffuse is True:
            diffuse_count = diffuse_ratio[diffuse_ratio > diffuse_ratio_threshold].resample('D').count()
        else:
            diffuse_count = diffuse_ratio[diffuse_ratio < diffuse_ratio_threshold].resample('D').count()
        day_fraction = diffuse_count / day_count
        sorted_by_diffuse_ratio = day_fraction[day_fraction > min_day_fraction].sort_values(ascending=False)
        return sorted_by_diffuse_ratio.index[:num_days].strftime('%Y-%m-%d')

    def get_daytime_df(self, threshold=10.0):
        # Use this method on any dataframe to get only the hours when Global Incident Irradiance > threshold W/m2.
        daytime_df = self.result_df[self.result_df[self.I_GLOBAL] > float(threshold)]
        return daytime_df


class BatchPVSystResults(PVSystResult):
    '''
    an instance of this class creates a dict that stores unpacked pvsyst results in the format:
    dict key = variant number
    dict value = pandas dataframe of pvsyst results for that variant

    PVSyst differentiates variants of a project by appending the project's file name with '0', '1', '2', ... etc.
    The user should define the list of variant suffixes to be considered.
    The user should also define a list of values that correspond to the project variants.
    The value might indicate the output that changes from one variant to another.

    For example, if a user has created pvsyst variants to study the effect of changing GCR,
    then the user should define the variant list as ['0', '1', '2'] and the parameter value list as the
    range of GCRs studied [0.2, 0.5, 0.8] and the parameter name list as a list with the same length as the others and
    containing 'GCR'.  Parameter name list is used to create meaningful dataframe column headings and legend keys in
    the graphs.

    This class includes methods that must operate on multiple PVSystResults objects -- such as calculating percent
    change or difference between results.  Any methods that can operate only on one result object are contained in the
    PVSystResult class.
    '''

    def __init__(self, locations, variant_files, parameter_names, parameter_values):
        if len(locations) != len(variant_files) != len(parameter_names) != len(parameter_values):
            raise ValueError('input lists are unequal length')
        # need to raise error if variant names are repeated in variant list
        self.num_results = len(locations)
        self.locations = locations
        self.parameter_names = parameter_names
        self.parameter_values = parameter_values
        self.variant_files = variant_files
        self.pvalue_variant_map = dict(zip(parameter_values, variant_files))
        self.variant_pvalue_map = dict(zip(variant_files, parameter_values))
        self.labels_map = dict(zip(variant_files, [z[1] + str(z[0]) for z in zip(parameter_values, parameter_names)]))

        self.results_dict = {}  # a dict of PVSystResult objects
        for i in range(self.num_results):
            print "loading variant: " + variant_files[i]
            self.results_dict[self.variant_pvalue_map[variant_files[i]]] = PVSystResult(locations[i], variant_files[i])
        print "done!"

    def __str__(self):
        a_str = ''
        for d in self.results_dict.iteritems():
            a_str += d[1].__str__() + '\n'
        return a_str

    def plot_generic_df(self, df_to_plot, single_output, y_info, title_str=''):
        plt.clf()

        if len(df_to_plot) == 1:  # single row df -- each coln is scalar-like, want to plot wrt column values
            plt.plot(df_to_plot.columns, df_to_plot.iloc[0, :] * y_info[self.SCALE], marker='.')
        else:  # multi-row df -- each column is array-like
            legend_list = []
            for c in df_to_plot.columns:
                plt.plot(df_to_plot.index, df_to_plot[c] * y_info[self.SCALE], marker='.')
                legend_list.append(self.labels_map[self.pvalue_variant_map[c]])
            plt.legend(legend_list, loc='best')
        plt.ylabel(y_info[self.UNITS])
        plt.xlabel(self.parameter_names[0])
        plt.title(single_output + title_str)
        plt.show()

    def compare_batch__resampling(self, single_output, baseline=None, resample_rate='H', diff=None, plot=True):
        '''
        Method (for now) assumes that there is only one parameter type for the batch.
        A more general method would allow for different parameters in the same batch, where each parameter is on a
        subplot, since each parameter could have different value ranges and units.
        '''
        if baseline is None and (diff is 'rel' or diff is 'abs'):
            baseline = self.results_dict[
                self.variant_pvalue_map[self.variant_files[0]]]  # baseline should be a PVSystResult object

        if baseline in self.results_dict.values():
            baseline_label = self.labels_map[baseline.variant]
        else:
            baseline_label = baseline.__str__()
        # and if a baseline object is provided but diff = 'total', we don't use the baseline at all.

        # resample:
        if single_output in self.energy_list + self.irradiance_list:
            resampled_df = pd.DataFrame({r[0]: r[1].result_df[single_output].resample(resample_rate).sum()
                                         for r in self.results_dict.iteritems()})
            if baseline is not None:
                baseline_resampled = baseline.result_df[single_output].resample(resample_rate).sum()
        else:  # includes: factors, angles, booleans
            resampled_df = pd.DataFrame({r[0]: r[1].result_df[single_output].resample(resample_rate).mean()
                                         for r in self.results_dict.iteritems()})
            if baseline is not None:
                baseline_resampled = baseline.result_df[single_output].resample(resample_rate).mean()

        if diff is 'abs':  # find absolute difference (i.e. variant - baseline)
            compare_df = pd.DataFrame({c: resampled_df[c] - baseline_resampled for c in resampled_df.columns})
            y_info = self.get_y_axis_info(single_output, resample_rate)
            add_title = '\n' + '(Variant - Baseline)' + '\n' + 'Baseline: ' + baseline_label

        elif diff is 'rel':  # find relative difference (i.e. ratio)
            compare_df = pd.DataFrame({c: (resampled_df[c] - baseline_resampled)/baseline_resampled
                                       for c in resampled_df.columns})
            y_info = self.get_y_axis_info(diff, resample_rate)
            add_title = '\n' + '(Variant - Baseline) / Baseline' + '\n' + 'Baseline: ' + baseline_label

        else:  # just find the resample total. includes the case where diff = 'total'
            compare_df = resampled_df
            y_info = self.get_y_axis_info(single_output, resample_rate)
            add_title = ''

        if plot:
            self.plot_generic_df(compare_df, single_output, y_info, title_str=add_title)

        return compare_df

    def compare_batch__specific_days(self, single_output, days_list=None):
        plt.clf()
        if days_list is None:
            days_list = self.SOLSTICES

        compare_df = self.compare_batch__resampling(single_output, plot=False)
        legend_list = [self.labels_map[self.pvalue_variant_map[c]] for c in compare_df.columns]

        plt.figure(figsize=(8, 15))
        plt.suptitle(single_output, fontsize=16)
        y_info = self.get_y_axis_info(single_output, 'H')
        ax = []
        for d in range(1, len(days_list) + 1):
            if d == 1:
                ax.append(plt.subplot(len(days_list), 1, d))
                plt.ylabel(y_info[self.UNITS], fontsize=14)
            else:
                ax.append(plt.subplot(len(days_list), 1, d, sharex=ax[0], sharey=ax[0]))
            plt.grid(True, which='major', axis='x')
            [ax[d - 1].plot(compare_df[days_list[d - 1]].index.hour, compare_df.loc[days_list[d - 1], c] * y_info[self.SCALE])
             for c in compare_df.columns]
            ax[d - 1].set_title(days_list[d - 1], fontsize=12)
        plt.xlabel('Hour of day', fontsize=14)
        ax[0].set_xlim([0, 24])
        ax[0].set_xticks([0, 6, 12, 18, 24])
        ax[0].set_ylim(y_info[self.LIM])
        ax[0].legend(legend_list, loc='lower right', fontsize=9)
        plt.show()

    def compare_batch__single_output(self, single_output, baseline=None):
        # creates a Panel where items = PVSyst result column names and minor_axis = variants
        batch_panel = pd.Panel.from_dict({r[0]: r[1].result_df for r in self.results_dict.iteritems()}, orient='minor')

        # creates a single dataframe where columns are PVSyst variants
        return batch_panel[single_output]


def resample_max(df):

    if single_output in self.energy_list + self.irradiance_list:
        maximized = single_output_slice.apply(np.max, axis=1).sum()
        if baseline is not None:
            baseline_agg = single_output_slice[baseline].sum()
    else:  # includes: factors, angles, booleans
        maximized = single_output_slice.apply(np.max, axis=1).mean()
        if baseline is not None:
            baseline_agg = single_output_slice[baseline].mean()

    if baseline is not None:
        return (maximized - baseline_agg)/baseline_agg
    else:
        return maximized

def compare_batch__get_best_angle(self, single_output):

    # creates a Panel where items = PVSyst result column names and minor_axis = variants
    batch_panel = pd.Panel.from_dict({r[0]: r[1].result_df for r in self.results_dict.iteritems()}, orient='minor')

    # creates a single dataframe where columns are PVSyst variants
    single_output_df = batch_panel[single_output]




# def get_diffuse_tracking(self, flat_result):
#     # Compare "effective" irradiance, energy from array, and energy to grid from two simulations,
#     # where "effective" accounts for linear effect of 3D near shadings construction and
#     # for IAM (impt at low sun angles!).
#     #
#     # Outputs: a dataframe with diffuse metrics concatenated onto self.results_dict
#     # Inputs: flat_result = BatchPVSyst object for a simulation of arrays w/o tracking on 0 deg tilt
#     #
#     flat_df = flat_result.results_dict[flat_result.results_dict.keys()[0]]
#     for r in self.results_dict.iteritems():
#         # find when diffuse tracking makes sense:
#         diffuse_tracking_gain_eff_irrad = flat_df[E_GLOBAL] - r[1][E_GLOBAL]
#         is_diffuse_tracking_better = (diffuse_tracking_gain_eff_irrad > 1.0) & (r[1][E_GLOBAL] > 1.0)
#         diffuse_tracking_gain_ratio = diffuse_tracking_gain_eff_irrad / r[1][E_GLOBAL]
#
#         # pull values from tracking and flat df's as appropriate:
#         track_angle_when_flat = pd.Series([0]*8760, index=flat_df.index)[is_diffuse_tracking_better]
#         track_angle_when_tracking = r[1].loc[(is_diffuse_tracking_better == False), ANG_TRACK]
#         eff_irrad_when_flat = r[1].loc[(is_diffuse_tracking_better == True), E_GLOBAL]
#         eff_irrad_when_tracking = r[1].loc[(is_diffuse_tracking_better == False), E_GLOBAL]
#         energy_array_when_flat = r[1].loc[(is_diffuse_tracking_better == True), ENERGY_ARRAY]
#         energy_array_when_tracking = r[1].loc[(is_diffuse_tracking_better == False), ENERGY_ARRAY]
#         energy_grid_when_flat = r[1].loc[(is_diffuse_tracking_better == True), ENERGY_GRID]
#         energy_grid_when_tracking = r[1].loc[(is_diffuse_tracking_better == False), ENERGY_GRID]
#
#         # merge series:
#         dt_track_angle = pd.concat([track_angle_when_flat, track_angle_when_tracking])
#         dt_track_angle.sort_index()
#         dt_eff_irrad = pd.concat([eff_irrad_when_flat, eff_irrad_when_tracking])
#         dt_eff_irrad.sort_index()
#         dt_energy_array = pd.concat([energy_array_when_flat, energy_array_when_tracking])
#         dt_energy_array.sort_index()
#         dt_energy_grid = pd.concat([energy_grid_when_flat, energy_grid_when_tracking])
#         dt_energy_grid.sort_index()
#
#         # throw it all at the end of the existing df!
#         self.results_dict[r[0]] = pd.concat([r[1],
#                                              pd.DataFrame({
#                                                  IS_DT: is_diffuse_tracking_better,
#                                                  DT_ANG_TRACK: dt_track_angle,
#                                                  DT_GAIN: diffuse_tracking_gain_eff_irrad,
#                                                  DT_GAIN_FACTOR: diffuse_tracking_gain_ratio,
#                                                  DT_E_GLOBAL: dt_eff_irrad,
#                                                  DT_ENERGY_ARRAY: dt_energy_array,
#                                                  DT_ENERGY_GRID: dt_energy_grid,
#                                              })], axis=1)
#
# def get_diffuse_tracking_improvement(self, resample_str):
#     # Only want to resample values when diffuse makes sense (i.e. "gain" is positive values).
#     # Output: 2-tuple of (absolute gain, gain output)
#     # Input: df with diffuse values, string representing resample interval ('A' = annual, 'M' = monthly, etc.)
#     # Setting resample interval to 'H' should result in same as DT_GAIN and DT_FACTOR.
#     diffuse_track_gain = {r[0]: (r[1].loc[r[1][IS_DT], DT_GAIN]).resample(resample_str).sum()
#                           for r in self.results_dict.iteritems()}
#     diffuse_track_output = pd.DataFrame(
#         {r[0]: diffuse_track_gain[r[0]] / r[1][E_GLOBAL].resample(resample_str).sum()
#          for r in self.results_dict.iteritems()})
#     return diffuse_track_output
#
# def plot_diffuse_tracking_improvement(self, resample_str, title_str=''):
#     # Only want to resample values when diffuse makes sense (i.e. "gain" is positive values).
#     # Output: 2-tuple of (absolute gain, gain output)
#     # Input: df with diffuse values, string representing resample interval ('A' = annual, 'M' = monthly, etc.)
#     # Setting resample interval to 'H' should result in same as DT_GAIN and DT_FACTOR.
#
#     diffuse_track_gain = {r[0]: (r[1].loc[r[1][IS_DT], DT_GAIN]).resample(resample_str).sum()
#                           for r in self.results_dict.iteritems()}
#
#     if resample_str == 'A':
#         output = []
#         diffuse_track_output = []
#
#         for r in self.results_dict.iteritems():
#             output.append(self.variant_output_map[r[0]])
#             diffuse_track_output.append(diffuse_track_gain[r[0]] / r[1][E_GLOBAL].resample(resample_str).sum())
#         plt.figure(figsize=(8, 8))
#         plt.plot(output, diffuse_track_output, 'bo')
#         plt.xlabel('Ground coverage ratio, GCR', fontsize=12)
#         plt.xlim([0.0, 1.0])
#
#     else:
#         diffuse_track_output = pd.DataFrame(
#             {(str(self.variant_output_map[r[0]]) + self.output_name):
#                  diffuse_track_gain[r[0]] / r[1][E_GLOBAL].resample(resample_str).sum()
#              for r in self.results_dict.iteritems()})
#         plt.figure(figsize=(15, 8))
#         plt.plot(diffuse_track_output)
#         plt.legend(diffuse_track_output.columns)
#
#     plt.ylabel('Improvement ratio', fontsize=12)
#     plt.ylim([0.0, 0.05])
#     plt.title(self.location + ': ' + '\nIrradiance Improvement with Diffuse Tracking ' + title_str, fontsize=16)


# hay_batch = BatchPVSystResults(12*['Hayward Air Term'], ['O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'], 12*['num_rows'], [4, 7, 8, 10, 13, 16, 19, 22, 25, 28, 3, 2])
# hay_batch = BatchPVSystResults(6*['Hayward Air Term'], ['Z', 'Y', 'O', 'P', 'Q', 'R'], 6*['num_rows'], [2, 3, 4, 7, 8, 10])
# hay_batch.compare_single_output_to_baseline(hay_batch.ENERGY_ARRAY, baseline=hay_batch.results_dict['num_rows8'], resample_rate='M', diff='rel')
# consider = hay_batch.compare_batch__resampling(hay_batch.E_DIRECT, resample_rate='A', diff='rel', plot=True)
# cloudy = hay_batch.results_dict[2].get_cloudy_days()
# hay_batch.compare_batch__specific_days(hay_batch.E_DIFF_S)












