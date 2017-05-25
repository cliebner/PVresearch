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

# output names:
ANG_ELEV = 'Sun Elevation'
ANG_AZ = 'Sun Azimuth'  # pvlib - 180 = pvsyst
ANG_INC = 'Incident Angle'
ANG_PROF = 'Profile Angle'
ANG_SURF_AZ = 'Panel Azimuth'  # pvlib - 180 = pvsyst
ANG_SURF_TILT = 'Panel Tilt'
ANG_FIXED = 'Plane tilt'
ANG_TRACK = 'Tracker Angle'
ANG_TOP = 'TOP Tracker Angle'
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
IS_BTRK = 'Backtracking?'
IS_TRK = 'Standard tracking?'
DT_GAIN = 'DiffTrack Gain'
DT_GAIN_FACTOR = 'DiffTrack Gain Factor'
DT_ANG_TRACK = 'DiffTrack Tracker Angle'
DT_DELTA = 'Diffuse Offset'
DT_DELTA_ABS = 'Absolute Diffuse Offset'
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
pvsyst_name_lookup = dict(zip(pvsyst_rename.values(), pvsyst_rename.keys()))


class PVSystResult(object):
    '''
    This is a class that defines general methods for interacting with a single PVSyst 'User Hourly" csv file.
    It defines:
    * global variables that map to PVSyst output naming convention
    * method for importing a csv and converting to pandas data frame
    * method for
    '''

    def __init__(self, location, hourly_results_file):
        self.location = location
        self.result_file = hourly_results_file
        self.separator = ';'  # for single PVsyst results
        self.result_df = self.unpack_single_pvsyst_result()
        self.is_tracking = ANG_TRACK in self.result_df.columns
        self.get_site_diffuseness()
        # self.daytime_df_index = self.get_daytime_df().index
        # self.avg_diffuse = self.result_df.loc[self.daytime_df_index, self.H_DIFF_RATIO].mean()
        # self.cloudy_days = self.get_cloudy_days()

    def __str__(self):
        return self.result_file

    def unpack_single_pvsyst_result(self):
        # Read in PVSyst results file as data frame:
        result_df = pd.read_csv(self.result_file, sep=self.separator, header=0,
                                skiprows=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11],
                                index_col=0, parse_dates=True, dayfirst=True)
        result_df.rename(columns=pvsyst_rename, inplace=True)
        return result_df

    def get_site_diffuseness(self):
        # h_diffuse_ratio = self.result_df.loc[:, self.H_DIFFUSE] / self.result_df[self.H_GLOBAL]
        h_diffuse_ratio = self.result_df[H_DIFFUSE] / self.result_df[H_GLOBAL]
        i_diffuse_ratio = self.result_df[I_DIFF_S] / self.result_df[I_GLOBAL]
        e_diffuse_ratio = self.result_df[E_DIFF_S] / self.result_df[E_GLOBAL]
        self.result_df = pd.concat([self.result_df,
                                             pd.DataFrame({
                                                 H_DIFF_RATIO: h_diffuse_ratio,
                                                 I_DIFF_RATIO: i_diffuse_ratio,
                                                 E_DIFF_RATIO: e_diffuse_ratio,
                                             })], axis=1)

    def plot_site_diffuseness(self):
        self.result_df[E_DIFF_RATIO].plot.hist(bins=50)
        plt.title(self.location, fontsize=16)
        plt.xlabel(E_DIFF_RATIO)
        plt.ylim([0, 1000])

    def export_as_csv(self):
        pvsyst_rename_inv = {p[1]: p[0] for p in pvsyst_rename.iteritems()}
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
        if y_data in factors_list:
            scale_by = 1.0
            ylim = factors_range
            units = 'factor'
        elif y_data in irradiance_list:
            scale_by = energy_scales[resample_rate]
            ylim = energy_range
            units = energy_units[resample_rate] +'/m2'
        elif y_data in energy_list:
            scale_by = energy_scales[resample_rate]
            ylim = energy_range
            units = energy_units[resample_rate]
        elif y_data in angles_list:
            scale_by = 1.0
            ylim = angles_range
            units = 'Angle (deg)'
        elif y_data in boolean_list:
            scale_by = 1.0
            ylim = boolean_range
            units = '1 = True'
        elif y_data is 'rel':
            scale_by = 100.0
            ylim = percent_range
            units = '%'
        else:
            scale_by = 1.0
            ylim = [0, 1]
            units = 'units'
        return {SCALE: scale_by, LIM: ylim, UNITS: units}

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
        diffuse_ratio = self.result_df[E_DIFF_S] / self.result_df[E_GLOBAL]
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
        daytime_df = self.result_df[self.result_df[I_GLOBAL] > float(threshold)]
        return daytime_df


class PVSystBatchSim(PVSystResult):
    '''
    Uses a PVSyst batch .csv file to create an object of many individual PVSyst hourly results

    Assume this format of Batch_Params csv file from PVSyst:
    * ignore first 10 lines
    * column names start with the line that includes "Ident"
    * 3 consecutive lines required to complete column names
    * general format is:
    ** col 0: simulation ID
    ** col 1: base variant
    ** col 2: hourly results filename
    ** cols 3 to N-1: parameters changed in each simulation
    ** col N: description of simulation
    '''

    def __init__(self, location, batch_param_filename, directory=None):
        '''
        all simulation results and the batch parameter file should be in the same directory
        '''
        self.location = location
        self.separator = ';'
        if directory is None:
            self.batch_param_filename = batch_param_filename
        else:
            self.batch_param_filename = directory + '/' + batch_param_filename

        with open(self.batch_param_filename, 'r', 0) as f:
            batch_param_file = f.read().splitlines()

        # read column headings:
        i = ['Ident' in p for p in batch_param_file].index(True)
        col_names = [b.rstrip(';').split(';') for b in batch_param_file[i: i+2]]
        sims_list_all = [b.rstrip(';').split(';') for b in batch_param_file if 'SIM' in b[0:3]]
        sims_list = [s for s in sims_list_all if 'N' not in s]

        if directory is None:
            self.hourly_results_filenames = [s[2] for s in sims_list]
        else:
            self.hourly_results_filenames = [directory + '/' + (s[2]) for s in sims_list]

        self.sims_descrip = [s[col_names[1].index('Comment')] for s in sims_list]
        self.params_dict = {}
        for j in range(3, len(col_names[0])-1):  # ignore 'Ident', 'Create hourly', and 'Simul' columns
            param_name = ' '.join([c[j] for c in col_names])
            param_vals = []
            for s in range(len(sims_list)):
                try:
                    param_vals.append(float(sims_list[s][j]))
                except ValueError:
                    param_vals.append(99.0)
            self.params_dict[param_name] = dict(zip(self.sims_descrip, param_vals))
        self.num_results = len(sims_list)
        self.parameter_names = self.params_dict.keys()

        self.results_dict = {}  # a dict of PVSystResult objects
        for i in range(len(self.hourly_results_filenames)):
            print "loading sim: " + self.sims_descrip[i]
            try:
                self.results_dict[self.sims_descrip[i]] = PVSystResult(self.location, self.hourly_results_filenames[i])
            except IOError:
                print self.sims_descrip[i] + ' does not exist'
        self.tracking_model = self.find_tracking_model()
        print "tracking model(s) found: " + str(self.tracking_model.keys()[0])
        print "done!"

    def __str__(self):
        a_str = ''
        for d in self.results_dict.iteritems():
            a_str += d[1].__str__() + '\n'
        return a_str

    def find_tracking_model(self):
        tracking_models = {r[0]: r[1] for r in self.results_dict.iteritems() if r[1].is_tracking is True}
        return tracking_models

    def get_TOP_model(self, opt_param=E_GLOBAL):

        '''
        This method sifts through many PVSyst simulations to optimize for a given factor (e.g.  POA irradiance,
        energy to grid).
        Assume: each PVSyst simulation represents a fixed tilt array. By selecting the simulation that optimizes
        for the given factor in that hour, we assume that the fixed angle represented by the selected simulation
        is the "optimal" tracking angle.

        The TOP model should be limited by backtracking.  To be safe, the model should default to normal tracking.
        '''

        try:
            self.tracking_model.keys()[0]
        except KeyError:
            print 'no tracking model loaded'
            return None

        # create a Panel where items = PVSyst result column names, major_axis= datetime, minor_axis = variants
        # to slice a Panel: panel.loc[item, major_axis, minor_axis]
        batch_panel = pd.Panel.from_dict({r[0]: r[1].result_df for r in self.results_dict.iteritems()}, orient='minor')

        # only look at daytime hours
        day_index = self.results_dict.values()[0].result_df[self.results_dict.values()[0].result_df[E_GLOBAL] > 1.0].index

        # find max value and its index
        max_value = batch_panel.loc[opt_param, day_index, :].apply(np.max, axis=1)
        max_index = batch_panel.loc[opt_param, day_index, :].idxmax(axis=1)

        # map series name to panel tilt angle
        # map fixed tilt angle to series name
        angle_to_name_map = dict(zip(self.params_dict[ANG_FIXED].values(), self.params_dict[ANG_FIXED].keys()))

        TOP_model = pd.DataFrame(index=day_index,
                                 columns=[H_DIFF_RATIO, opt_param, ANG_FIXED, ANG_TRACK, IS_BTRK, ANG_TOP, ANG_INC,
                                          DT_DELTA, DT_DELTA_ABS])
        TOP_model[H_DIFF_RATIO] = batch_panel.loc[H_DIFF_RATIO, day_index, self.results_dict.keys()[0]]
        TOP_model[opt_param] = max_value
        TOP_model[ANG_FIXED] = max_index.map(self.params_dict[ANG_FIXED])
        TOP_model[ANG_TRACK] = batch_panel.loc[ANG_TRACK, day_index, self.tracking_model.keys()[0]]

        # initialize other columns assuming backtracking is default (will apply mask later):
        TOP_model[IS_BTRK] = True
        TOP_model[ANG_TOP] = TOP_model[ANG_TRACK]
        TOP_model[ANG_INC] = batch_panel.loc[ANG_INC, day_index, self.tracking_model.keys()[0]]

        # conditionals for mask:
        # don't let TOP pick an angle more steep than backtracking to prevent unwanted shading
        # TODO: check if this is necessary if we optimize on E_ARRAY
        is_outside_btrk_range = abs(TOP_model[ANG_FIXED]) > abs(self.tracking_model.values()[0].result_df[ANG_TRACK])
        # don't let TOP pick an angle inappropriate for time of day, e.g. +10 deg for 10am.
        is_facing_wrong_azimuth = ((TOP_model.index.hour < 12) & (TOP_model[ANG_FIXED] > 0.0)) | \
                              ((TOP_model.index.hour > 12) & (TOP_model[ANG_FIXED] < 0.0))

        TOP_model[IS_BTRK].where(is_outside_btrk_range, other=False, inplace=True)
        TOP_model[ANG_TOP].where(is_outside_btrk_range, other=TOP_model[ANG_FIXED], inplace=True)
        TOP_model[ANG_TOP].mask(is_facing_wrong_azimuth, other=0.0, inplace=True)
        TOP_model[ANG_INC].where(is_outside_btrk_range, other=batch_panel.loc[ANG_INC,:,angle_to_name_map], inplace=True)

        # for t in TOP_model.index:
        #     if abs(TOP_model.loc[t, ANG_FIXED]) > abs(self.tracking_model.values()[0].result_df.loc[t, ANG_TRACK]):
        #         # TOP_model.loc[t, IS_BTRK] = True
        #         # TOP_model.loc[t, ANG_TOP] = self.tracking_model.values()[0].result_df.loc[t, ANG_TRACK]
        #         TOP_model.loc[t, ANG_INC] = self.tracking_model.values()[0].result_df.loc[t, ANG_INC]
        #     elif ((t.hour < 12) & (TOP_model.loc[t, ANG_FIXED] > 0.0)) | \
        #             ((t.hour > 12) & (TOP_model.loc[t, ANG_FIXED] < 0.0)):
        #         # TOP_model.loc[t, IS_BTRK] = False
        #         # TOP_model.loc[t, ANG_TOP] = 0.0  # could also default to regular tracking here?
        #         TOP_model.loc[t, ANG_INC] = batch_ANG_INC.loc[t, reverse_map[0.0]]
        #     else:
        #         # TOP_model.loc[t, IS_BTRK] = False
        #         # TOP_model.loc[t, ANG_TOP] = TOP_model.loc[t,ANG_FIXED]
        #         TOP_model.loc[t, ANG_INC] = batch_ANG_INC.loc[t, reverse_map[TOP_model.loc[t,ANG_FIXED]]]
        TOP_model[DT_DELTA] = TOP_model.loc[:, ANG_TOP] - \
                                               TOP_model.loc[:, ANG_TRACK]
        TOP_model[DT_DELTA_ABS] = TOP_model[DT_DELTA].apply(np.abs)
        plt.plot(TOP_model.index.hour, TOP_model[ANG_FIXED], mec='b', color='None', marker='o')
        plt.plot(TOP_model.index.hour, TOP_model[ANG_TOP], mec='r', color='None', marker='o')
        return TOP_model

    def abc(self):
        '''
        1. separate results into bins based on PROFILE ANGLE, then plot diffuse ratio vs tracking angle adjustment
        develop algorithm, e.g. for each hour, given a diffuse fraction, do SOMETHING
        2. use the algortihm to predict how the tracker will perform
        3. compare to TOP
        '''

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


# test = PVSystBatchSim('seattle', 'Seattle_Project_BatchParams_1.CSV', directory='test folder')
# test_TOP = test.get_TOP_model(E_GLOBAL)













