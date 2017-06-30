__author__ = 'christina'

# import matplotlib  # version 1.5.1
# matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # version 0.18
import datetime as dt
import itertools
import os

# global variables:
WINTER = '1990-12-21'
SUMMER = '1990-06-21'
SPRING = '1990-03-22'

# output names:
DATE = 'Datetime'
ANG_ELEV = 'Sun Elevation'
ANG_AZ = 'Sun Azimuth'  # pvlib - 180 = pvsyst
ANG_INC = 'Incident Angle'
ANG_PROF = 'Profile Angle'
ANG_SURF_AZ = 'Panel Azimuth'  # pvlib - 180 = pvsyst
ANG_SURF_TILT = 'Panel Tilt'
ANG_FIXED = 'Plane tilt'
ANG_TRACK = 'Tracker Angle'
ANG_TOP = 'TOP Tracker Angle'
ANG_TOP_0 = 'TOP: 0 constraint'
ANG_TOP_1 = 'TOP: 1 constraint'
ANG_TOP_2 = 'TOP: 2 constraints'
ANG_TOP_INC = 'TOP Incident Angle'
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
SLOSS_ELEC = 'Electrical Shade Loss'
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
IS_STD_TRK = 'Std tracking best?'
IS_FORCED_BTRK = 'Forced backtracking?'
IS_FORCED_ZERO = 'Forced to zero?'
STD_TRK = 'Standard tracking'
SIM_LOOKUP = 'Sim lookup'
DT_GAIN = 'DiffTrack Gain'
DT_GAIN_FACTOR = 'DiffTrack Gain Factor'
DT_ANG_TRACK = 'DiffTrack Tracker Angle'
DT_DELTA = 'Diffuse Offset'
DT_DELTA_ABS = 'Absolute Diffuse Offset'
DT_RATIO = 'DiffTrack Ratio'
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
ENERGY_INV = 'Energy from Inverter'
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
                   SLOSS_ELEC, DT_GAIN]
energy_list = [ENERGY_ARRAY, ENERGY_INV, ENERGY_GRID, PVLOSS_IRRADIANCE, PVLOSS_TEMP]
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
    'date': DATE,
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
    'ShdElec': SLOSS_ELEC,
    'FShdGl': SF_GLOBAL,
    'FShdBm': SF_DIRECT,
    'FShdDif': SF_DIFF_S,
    'FShdAlb': SF_DIFF_G,
    'FIAMGl': IAMF_GLOBAL,
    'FIAMBm': IAMF_DIRECT,
    'FIAMDif': IAMF_DIFF_S,
    'FIAMAlb': IAMF_DIFF_G,
    'DifS/Gl': RATIO_DIFF_S,
    'EArray': ENERGY_ARRAY,
    'EOutInv': ENERGY_INV, # W
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
useful_names = [pvsyst_name_lookup[i] for i in [DATE, ANG_ELEV, ANG_AZ, ANG_INC, ANG_TRACK,
                                                H_GLOBAL, H_DIFFUSE, E_GLOBAL, E_DIRECT, E_DIFF_S, E_DIFF_G,
                                                ENERGY_ARRAY, ENERGY_INV, ENERGY_GRID, SLOSS_ELEC]
                ]
PVSYST_BATCH_TEMPLATE_FILE = 'pvsyst_batch_template.CSV'
BATCH_IDENT = 'Ident\n\n'  # e.g. list of SIM_1, SIM_2, etc.
BATCH_VC = 'Simul.\nident\n'  # list of variant files to use in simulation, e.g. VC0, VC1, VC2, etc.
BATCH_SITE = 'Project site\nSite name\n'
BATCH_MET = 'Meteo data\n*.MET file\n'
BATCH_FILENAME = 'Create hourly\nfile\nFile name'
BATCH_PLANE_TILT = 'Plane\ntilt\n[deg]'  # must be empty -- not zero -- if backtracking is used in simulation
BATCH_PLANE_AZIM = 'Plane\nazim\n[deg]'  # must be empty -- not zero -- if backtracking is used in simulation
BATCH_PITCH_S = 'Sheds 3D\npitch\n[m]'
BATCH_PITCH_T = 'Trackers\nPitch EW\n[m]'
BATCH_PHI_MAX = 'Trackers\nPhi Max\n[deg]'  # must be empty -- not zero -- if backtracking NOT used in simulation
BATCH_PHI_MIN = 'Trackers\nPhi Min\n[deg]'  # must be empty -- not zero -- if backtracking NOT used in simulation
BATCH_COMMENT = 'Simul\nComment\n'
PVSYST_BATCH_ORDER = [BATCH_IDENT, BATCH_SITE, BATCH_MET, BATCH_VC, BATCH_FILENAME,
                      BATCH_PLANE_TILT, BATCH_PLANE_AZIM, BATCH_PITCH_S,
                      BATCH_PITCH_T, BATCH_PHI_MAX, BATCH_PHI_MIN,
                      BATCH_COMMENT]
PVSYST_BATCH_VARIABLES = [BATCH_VC, BATCH_MET, BATCH_SITE,
                          BATCH_PLANE_TILT, BATCH_PLANE_AZIM, BATCH_PITCH_S,
                          BATCH_PITCH_T, BATCH_PHI_MAX, BATCH_PHI_MIN]
PVSYST_BATCH_CONSTANTS = [BATCH_IDENT, BATCH_FILENAME, BATCH_COMMENT]
PVSYST_BATCH_PARAMS_DICT = {x: [] for x in PVSYST_BATCH_ORDER}
PVSYST_BATCH_PARAMS_DICT_FORMATTER = {
    BATCH_IDENT: lambda x: str(x),
    BATCH_VC: lambda x: str(x),
    BATCH_SITE: lambda x: str(x),
    BATCH_MET: lambda x: str(x),
    BATCH_FILENAME: lambda x: str(x),
    BATCH_PLANE_TILT: lambda x: float(x),
    BATCH_PLANE_AZIM: lambda x: float(x),
    BATCH_PITCH_S: lambda x: float(x),
    BATCH_PITCH_T: lambda x: float(x),
    BATCH_PHI_MAX: lambda x: float(x),
    BATCH_PHI_MIN: lambda x: float(x),
    BATCH_COMMENT: lambda x: str(x),
}
PVSYST_BATCH_PARAMS_DICT_COMMENTER = {
    BATCH_IDENT: '',
    BATCH_VC: '',
    BATCH_SITE: '',
    BATCH_MET: '',
    BATCH_FILENAME: '',
    BATCH_PLANE_TILT: 't',
    BATCH_PLANE_AZIM: 'az',
    BATCH_PITCH_S: 'p',
    BATCH_PITCH_T: 'p',
    BATCH_PHI_MAX: 'px',
    BATCH_PHI_MIN: 'pn',
    BATCH_COMMENT: '',
}


def get_tracker_band(panel, low, high):
    band_index = panel.major_axis[
        (panel.loc[ANG_TRACK, :, STD_TRK] < high) & (panel.loc[ANG_TRACK, :, STD_TRK] >= low)]
    return panel[:, band_index, :]


def plot_tracker_band(panel, constraint, start, end):
    x_data_am = get_tracker_band(panel, min([-1*abs(start), -1*abs(end)]),
                                 max([-1*abs(start), -1*abs(end)])).loc[H_DIFF_RATIO, :, constraint]
    y_data_am = get_tracker_band(panel, min([-1*abs(start), -1*abs(end)]),
                                 max([-1*abs(start), -1*abs(end)])).loc[DT_RATIO, :, constraint]
    x_data_pm = get_tracker_band(panel, min([abs(start), abs(end)]),
                                 max([abs(start), abs(end)])).loc[H_DIFF_RATIO,:, constraint]
    y_data_pm = get_tracker_band(panel, min([abs(start), abs(end)]),
                                 max([abs(start), abs(end)])).loc[DT_RATIO,:, constraint]
    fit_am = np.polyfit(x_data_am, y_data_am, 2)
    fit_pm = np.polyfit(x_data_pm, y_data_pm, 2)
    x_array = np.arange(0.1,1.01, 0.01)
    y_fit_am = fit_am[0]*x_array**2 + fit_am[1]*x_array**1 + fit_am[2]*x_array**0
    y_fit_pm = fit_pm[0] * x_array ** 2 + fit_pm[1] * x_array ** 1 + fit_pm[2] * x_array ** 0
    plt.plot(x_data_am, y_data_am, mec='r', marker='o', color='None')
    plt.plot(x_array, y_fit_am, 'r:')
    plt.plot(x_data_pm, y_data_pm, mec='g', marker='o', color='None')
    plt.plot(x_array, y_fit_pm, 'g:')
    plt.legend([str([min([-1*abs(start), -1*abs(end)]), max([-1*abs(start), -1*abs(end)])]), '',
                str([min([abs(start), abs(end)]), max([abs(start), abs(end)])]), ''], loc='lower left')


class PVSystResult(object):
    '''
    This is a class that defines general methods for interacting with a single PVSyst 'User Hourly" csv file.
    It defines:
    * global variables that map to PVSyst output naming convention
    * method for importing a csv and converting to pandas data frame
    * method for
    '''

    def __init__(self, descrip, hourly_results_file):
        self.description = descrip
        self.MET_file = None
        self.SIT_file = None
        self.sim_variant = None
        self.result_file = hourly_results_file
        self.separator = ';'  # for single PVsyst results
        self.is_tracking = False
        self.result_df = self.unpack_single_pvsyst_result()

        # TODO: implement general weather analyzer -- parent class of tmy analyzer...
        self.get_site_diffuseness()
        # self.daytime_df_index = self.get_daytime_df().index
        # self.avg_diffuse = self.result_df.loc[self.daytime_df_index, self.H_DIFF_RATIO].mean()
        self.cloudy_days, self.cloudy_days_eqx = self.get_cloudy_days(is_diffuse=True)
        self.sunny_days, self.sunny_days_eqx = self.get_cloudy_days(is_diffuse=False)
        self.export_as_csv()

    def __str__(self):
        return self.result_file

    def unpack_single_pvsyst_result(self):
        # Read top few lines to understand data column names and order:
        with open(self.result_file, 'r', 0) as f:
            all_lines = f.readlines()

        # check the separator (always positioned at zeroth character of first line in file):
        self.separator = all_lines[1][0]

        # get simulation-specific data:
        i_met = ['Meteo data' in a[:12] for a in all_lines[:15]].index(True)
        i_sit = ['Geographical Site' in a[:18] for a in all_lines[:15]].index(True)
        i_vc = ['Simulation variant' in a[:19] for a in all_lines[:15]].index(True)
        i_col = ['date' in a[:5] for a in all_lines[:15]].index(True)

        self.MET_file = all_lines[i_met].split(self.separator)[1]
        self.SIT_file = all_lines[i_sit].split(self.separator)[1]
        self.sim_variant = all_lines[i_vc].split(self.separator)[1]
        col_names = all_lines[i_col].strip().split(self.separator)

        name_index_map = dict(zip(col_names, range(len(col_names))))
        usable_names = [u for u in useful_names if u in col_names]
        result_dict = {u: [] for u in usable_names}
        for i in range(i_col+3, len(all_lines)):
            for u in usable_names:
                value = all_lines[i].split(self.separator)[name_index_map[u]]
                if u is pvsyst_name_lookup[DATE]:
                    result_dict[u].append(dt.datetime.strptime(value, '%d/%m/%y %H:%M'))
                else:
                    result_dict[u].append(float(value))
        result_df = pd.DataFrame.from_dict(result_dict, orient='columns')
        useful_rename = {p[0]: p[1] for p in pvsyst_rename.iteritems() if p[0] in usable_names}
        result_df.rename(columns=useful_rename, inplace=True)
        result_df.set_index(DATE, inplace=True)
        self.is_tracking = ANG_TRACK in result_df.columns

        # Read in PVSyst results file as data frame:  DON'T DO THIS -- PANDAS HAS A LOT OF OVERHEAD
        # result_df = pd.read_csv(self.result_file, sep=self.separator, header=0,
        #                         skiprows=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11],
        #                         usecols=usable_names,
        #                         index_col=0, parse_dates=True, dayfirst=True)

        return result_df

    def get_site_diffuseness(self):
        # h_diffuse_ratio = self.result_df.loc[:, self.H_DIFFUSE] / self.result_df[self.H_GLOBAL]
        h_diffuse_ratio = self.result_df[H_DIFFUSE] / self.result_df[H_GLOBAL]
        # i_diffuse_ratio = self.result_df[I_DIFF_S] / self.result_df[I_GLOBAL]
        # e_diffuse_ratio = self.result_df[E_DIFF_S] / self.result_df[E_GLOBAL]
        self.result_df = pd.concat([self.result_df,
                                             pd.DataFrame({
                                                 H_DIFF_RATIO: h_diffuse_ratio,
                                                 # I_DIFF_RATIO: i_diffuse_ratio,
                                                 # E_DIFF_RATIO: e_diffuse_ratio,
                                             })], axis=1)

    def plot_site_diffuseness(self):
        self.result_df[H_DIFF_RATIO].plot.hist(bins=50)
        plt.title(self.description, fontsize=16)
        plt.xlabel(H_DIFF_RATIO)
        plt.ylim([0, 1000])

    def export_as_csv(self):
        pvsyst_rename_inv = {p[1]: p[0] for p in pvsyst_rename.iteritems()}
        # Convert names back to PVSyst convention names:
        export_df = self.result_df.rename(columns=pvsyst_rename_inv, inplace=False)
        print "saving to .csv file"
        export_df.to_csv(self.description + '_pyexport.csv')
        print "done!"

    def plot_single_output_hourly(self, single_output, title_str=''):
        # Output: graph plotting hourly values of single_output
        # Inputs: single_output: keyword name of data to plot
        #        (optional) backtracking: set False if you want no backtacking
        #        (optional) show_flat: set True if you also want to plot results from "flat" simulation
        plt.plot(self.result_df[single_output], 'b.')
        # plt.figure(figsize=(15, 8))
        plt.title(self.description + ': ' + single_output + '\n' +
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

    def get_cloudy_days(self, num_days=3, is_diffuse=True, diffuse_ratio_threshold=0.7):
        ''''
        Sorts days of the year based on DAILY MEAN HORIZONTAL diffuse ratio (ignoring albedo).
        Use keyword args to select for high diffuse (cloudy) or low diffuse (clear) days.
        Outputs: pandas series for days where diffuse ratio meets a limit
        Inputs: dataframe with the horiz diffuse ratio and
                optional kwargs: diffuse boolean (set to False to find clear days),
                                 diffuse ratio threshold
        '''
        # diffuse_ratio = self.result_df[E_DIFF_S] / self.result_df[E_GLOBAL]
        daily_diffuse = self.result_df[H_DIFF_RATIO].resample('D').mean()
        if is_diffuse is True:
            threshold_daily_diffuse = daily_diffuse[daily_diffuse > diffuse_ratio_threshold].sort_values(ascending=False)
            # sorted_by_diffuse_ratio = day_fraction[day_fraction > min_day_fraction].sort_values(ascending=False)
        else:
            threshold_daily_diffuse = daily_diffuse[daily_diffuse < diffuse_ratio_threshold].sort_values(ascending=True)
            # sorted_by_diffuse_ratio = day_fraction[day_fraction > min_day_fraction].sort_values(ascending=False)

        around_equinox = threshold_daily_diffuse[(threshold_daily_diffuse.index.month == 3) |
                                                 (threshold_daily_diffuse.index.month == 4) |
                                                 (threshold_daily_diffuse.index.month == 9) |
                                                 (threshold_daily_diffuse.index.month == 10)]

        return threshold_daily_diffuse[:num_days], around_equinox[:num_days]

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

    def __init__(self, location, directory=None):
        '''
        initializing a PVsystBatch object does not load data
        Typical modeling steps:
        1. initialize a new object
        2. create a batch file if needed
        3. load data by reading a batch file
        4. run TOP model in desired mode (diffuse, row to row, both, or none)
        all simulation results and the batch parameter file should be in the same directory
        '''
        self.location = location
        self.separator = ';'
        self.directory = directory
        self.batch_filename = None
        self.batch_file_created = False
        self.batch_file_read = False
        # self.batch_dict = {x: [] for x in PVSYST_BATCH_ORDER}
        self.batch_dict = {}
        self.read_variables = []
        self.base_variant = None
        self.tracking_model = None
        self.results_dict = {}  # a dict of PVSystResult objects
        self.results_panel = None
        self.top_panel = None

    def __str__(self):
        a_str = ''
        for d in self.results_dict.iteritems():
            a_str += d[1].__str__() + '\n'
        return a_str

    def load_data(self):

        if self.batch_file_read is True:
            pass
        else:
            self.read_batch_file()

        if self.directory is None:
            pass
        else:
            self.batch_dict[BATCH_FILENAME] = [self.directory + '/' + f for f in self.batch_dict[BATCH_FILENAME]]

        for h in range(len(self.batch_dict[BATCH_FILENAME])):
            print "loading sim: " + self.batch_dict[BATCH_COMMENT][h]
            try:

                self.results_dict[self.batch_dict[BATCH_COMMENT][h]] = \
                    PVSystResult(self.batch_dict[BATCH_COMMENT][h], self.batch_dict[BATCH_FILENAME][h])

                # to each result df, add columns to describe the variables in play for that simulation result
                # e.g. if panel tilt is a variable, this adds a column showing constant panel tilt for any simulation df
                for v in self.read_variables:
                    self.results_dict[self.batch_dict[BATCH_COMMENT][h]].result_df[v] = self.batch_dict[v][h]

            except IOError:
                print self.batch_dict[BATCH_COMMENT][h] + ' does not exist'

        self.results_panel = pd.Panel.from_dict({r[0]: r[1].result_df for r in self.results_dict.iteritems()}, orient='minor')

        self.tracking_model = self.find_tracking_model()
        if len(self.tracking_model) == 0:
            print "tracking model(s) found: none"
        else:
            print "tracking model(s) found: " + ', '.join([str(t) for t in self.tracking_model.keys()])
        print "done!"


    def find_tracking_model(self):
        return {r[0]: r[1] for r in self.results_dict.iteritems() if r[1].is_tracking is True}

    def read_batch_file(self):

        # check that a batch_file is defined:
        if self.batch_filename is None:
            print 'batch_file is undefined.\nRun self.create_batch_file or set self.batch_file explicitly.'
            return None

        if self.directory is None or self.batch_file_created is True:
            pass
        else:
            self.batch_filename = self.directory + '/' + self.batch_filename

        # read file that lists PVsyst batch simulations:
        with open(self.batch_filename, 'r', 0) as f:
            file_lines = f.read().splitlines()

        # identify base variant, if not already known:
        if self.base_variant is None:
            i_vn = ['Variants based on' in f for f in file_lines[0:14]].index(True)
            variant_name_line = file_lines[i_vn].split(self.separator)
            self.base_variant = variant_name_line[2]

        # count (& read) column headings:
        i = ['Ident' in f for f in file_lines].index(True)
        col_lines = [b.split(';') for b in file_lines[i: i + 3]]
        num_cols = len([c for c in col_lines[0] if len(c) > 0])
        col_names = ['\n'.join([l[c] for l in col_lines]) for c in range(num_cols)]
        self.batch_dict = {c: [] for c in col_names}

        # count number of simulations (rows):
        sim_rows = [b.split(';')[0:num_cols] for b in file_lines if 'SIM_' in b[0:4]]

        # for each row of simulations, skipping the first one bc PVsyst does something weird with the first SIM
        for r in range(1, len(sim_rows)):
            for c in range(len(col_names)):  # for each column
                value = sim_rows[r][c]
                if value == '':
                    pass
                else:
                    value = PVSYST_BATCH_PARAMS_DICT_FORMATTER[col_names[c]](value)
                self.batch_dict[col_names[c]].append(value)

        self.batch_file_read = True
        self.read_variables = set(self.batch_dict.keys()).intersection(set(PVSYST_BATCH_VARIABLES))

    def create_batch_file(self, base_variant=None, save_as='batch_filename', description='None',
                          variables=PVSYST_BATCH_PARAMS_DICT):

        self.base_variant = base_variant
        num_sims = max([len(x[1]) for x in variables.iteritems()])

        # add the SIM_, filename, and comment columns to the variables columns:
        columns = variables.copy()
        if BATCH_IDENT not in columns.keys():
            columns[BATCH_IDENT] = ['SIM_' + str(x) for x in
                                      range(2, num_sims + 2)]  # start at SIM_2 bc PVsyst overwrites SIM_1
        if BATCH_COMMENT not in columns.keys():
            columns[BATCH_COMMENT] = ['_'.join([str(v[1][n]) + PVSYST_BATCH_PARAMS_DICT_COMMENTER[v[0]]
                                            for v in variables.iteritems()
                                                if v[0] not in [BATCH_COMMENT, BATCH_FILENAME]])
                                      for n in range(0, num_sims)]
        if BATCH_FILENAME not in columns.keys():
            columns[BATCH_FILENAME] = [m + '.csv' for m in columns[BATCH_COMMENT]]

        # get columns sorted in PV syst order
        cols_in_use = [c for c in PVSYST_BATCH_ORDER if c in columns.keys()]

        with open(PVSYST_BATCH_TEMPLATE_FILE, 'r', 0) as f:
            batch_template_file = f.read().splitlines(1)

        header = batch_template_file[0:14]
        # replace some data in template header
        i_fc = ['File Created on ' in h for h in header].index(True)
        header[i_fc] = 'File Created on ' + dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        i_pn = ['Project' in h for h in header].index(True)
        project_name_line = header[i_pn].split(self.separator)
        project_name = project_name_line[3]
        project_name_line.remove(project_name)
        project_name_line.insert(3, self.location)
        header[i_pn] = self.separator.join(project_name_line)

        i_vn = ['Variants based on' in h for h in header].index(True)
        variant_name_line = header[i_vn].split(self.separator)
        variant_name = variant_name_line[2]
        descrip = variant_name_line[3]
        variant_name_line.remove(variant_name)
        variant_name_line.insert(2, self.base_variant)  #TODO: fix: should user have to enter BATCH_VC?
        variant_name_line.remove(descrip)
        variant_name_line.insert(3, description)
        header[i_vn] = self.separator.join(variant_name_line)

        if self.directory is None:
            new_filename = save_as
        else:
            new_filename = self.directory + '/' + save_as
        with open(new_filename, 'w') as f:

            # write header:
            for h in header:
                f.write(h)
            f.write(';\n')

            # write correct column names.  template requires column title on 2 lines, units on 3rd line, e.g.
            # Plane
            # tilt
            # [deg]
            f.write(self.separator.join([c.split('\n')[0] for c in cols_in_use]) + ';\n')
            f.write(self.separator.join([c.split('\n')[1] for c in cols_in_use]) + ';\n')
            f.write(self.separator.join([c.split('\n')[2] for c in cols_in_use]) + ';\n')
            f.write(';\n')

            # write data:
            # PVSyst will overwrite whatever is in the first line (SIM_1) with the base variant parameters, so write to
            # 2nd line
            f.write('SIM_1' + self.separator * len(cols_in_use) + '\n')
            for n in range(0, num_sims):
                f.write(self.separator.join([str(columns[c][n]) for c in cols_in_use]) + ';\n')
            f.write(';\n')

        # set the object variables:
        self.batch_file_created = True
        # self.batch_dict = batch_dict
        self.batch_filename = new_filename

    def get_TOP_model(self, top_param=E_GLOBAL, how='max'):

        '''
        This method sifts through many PVSyst simulations to optimize for a given factor (e.g.  POA irradiance,
        energy to grid).
        Assume: each PVSyst simulation represents a fixed tilt array. By selecting the simulation that optimizes
        for the given factor in that hour, we assume that the fixed angle represented by the selected simulation
        is the "optimal" tracking angle.

        The TOP model should be limited by backtracking.  To be safe, the model should default to normal tracking.
        '''

        # if mode = 'r2r':
        # select tracking angle that maximizes effective irradiance and minimizes beam shading loss
        # if mode = 'diffuse':
        # include backtracking in the analysis (default "do nothing" operation is regular backtracking)

        try:
            self.tracking_model.keys()[0]
        except KeyError:
            print 'no tracking model loaded'
            return None

        # only look at daytime hours
        day_index = self.results_dict.values()[0].result_df[self.results_dict.values()[0].result_df[H_GLOBAL] > 1.0].index

        # initialize a dataframe template that all TOP models will use:
        # ASSUMES: MET file and SITE files are held constant for all simulations in the batch :(
        # irradiance data: should not change in each model
        # booleans: flags to show if tracking is best, if tracking is forced, and if flat is forced.
        #           shows when masks are applied
        # angles: only ANG_TOP and ANG_TOP_INC change each time
        # effective irradiance: changes with each TOP
        # DT_xx: changes with each TOP
        top_template = pd.DataFrame(index=day_index,
                                   columns=list({H_DIFF_RATIO, H_GLOBAL, H_DIFFUSE,
                                                 top_param,
                                                 IS_STD_TRK, IS_FORCED_BTRK, IS_FORCED_ZERO, SIM_LOOKUP,
                                                 ANG_FIXED, ANG_TRACK, ANG_TOP, ANG_INC,
                                                 E_GLOBAL, E_DIRECT, E_DIFF_S, E_DIFF_G,
                                                 DT_DELTA, DT_DELTA_ABS, DT_RATIO}))
        # TOP_btrk:
        top_btrk = top_template.copy()
        for c in top_template.columns:
            try:
                top_btrk[c] = self.tracking_model.values()[0].result_df.loc[day_index, c]
            except KeyError:
                continue
        # top_0: find optimum value and its index
        top_0 = top_btrk.copy()
        top_0[top_param] = {
            'max': self.results_panel.loc[top_param, day_index, :].apply(np.max, axis=1),
            'min': self.results_panel.loc[top_param, day_index, :].apply(np.min, axis=1)
        }[how]
        top_0[SIM_LOOKUP] = {
            'max': self.results_panel.loc[top_param, day_index, :].idxmax(axis=1),
            'min': self.results_panel.loc[top_param, day_index, :].idxmin(axis=1)
        }[how]
        top_0[ANG_FIXED] = pd.Series(self.results_panel[BATCH_PLANE_TILT, day_index, :].lookup(top_0.index, top_0[SIM_LOOKUP]),
                               index=day_index)
        # if ANG_FIXED cannot be found in lookup (bc standard tracking is used), use a filler float value:
        top_0[ANG_FIXED].mask(top_0[ANG_FIXED] == '', other=99.0, inplace=True)

        constraints = {
            IS_STD_TRK: {
                'series': lambda x: x[ANG_FIXED],
                'conditional': lambda x: x[SIM_LOOKUP] == self.tracking_model.keys()[0], # may break if there are more than 1 backtracking models in the batch!
                'other': top_btrk[ANG_TRACK],
                'sim': self.tracking_model.keys()[0]
            },
            IS_FORCED_BTRK: {
                'series': lambda x: x[ANG_TOP],
                'conditional': lambda x: (x[IS_STD_TRK] == False) & (abs(x[ANG_FIXED]) > abs(x[ANG_TRACK])),
                'other': top_btrk[ANG_TRACK],
                'sim': self.tracking_model.keys()[0]
            },
            IS_FORCED_ZERO: {
                'series': lambda x: x[ANG_TOP],
                # 'conditional': lambda x: ((x.index.hour < 12) & (x[ANG_TOP] > 0.0)) | ((x.index.hour > 12) & (x[ANG_TOP] < 0.0)),
                'conditional': lambda x: x[DT_RATIO] < 0.0,
                'other': 0.0,
                'sim': self.batch_dict[BATCH_COMMENT][self.batch_dict[BATCH_PLANE_TILT].index(0.0)]
            }
        }
        constraint_order = [IS_STD_TRK, IS_FORCED_BTRK, IS_FORCED_ZERO]
        top_dict = {c: top_0.copy() for c in constraint_order}
        top_dict[STD_TRK] = top_btrk.copy()
        i = 0
        for c in constraint_order:
            top_dict[c].loc[:, c] = constraints[c]['conditional'](top_dict[c])
            top_dict[c].loc[:, ANG_TOP] = \
                constraints[c]['series'](top_dict[c]).mask(top_dict[c].loc[:, c],
                                                           other=constraints[c]['other'], inplace=False).apply(float)
            top_dict[c].loc[:, SIM_LOOKUP] = top_dict[c].loc[:, SIM_LOOKUP].mask(top_dict[c].loc[:, c],
                                                other=constraints[c]['sim'], inplace=False)
            for p in list({top_param, ANG_INC, E_GLOBAL, E_DIRECT, E_DIFF_S, E_DIFF_G}):
                top_dict[c].loc[:, p] = pd.Series(self.results_panel.loc[p,:,:].lookup(
                    day_index, top_dict[c].loc[:, SIM_LOOKUP]), index=day_index)
            top_dict[c].loc[:, DT_DELTA] = top_dict[c].loc[:, ANG_TOP] - top_btrk.loc[:, ANG_TRACK]
            top_dict[c].loc[:, DT_DELTA_ABS] = top_dict[c].loc[:, DT_DELTA].apply(np.abs)
            top_dict[c].loc[:, DT_RATIO] = top_dict[c].loc[:, ANG_TOP] / top_btrk.loc[:, ANG_TRACK]

            # initialize next df as a copy of this df:
            if i < (len(constraint_order) - 1):
                top_dict[constraint_order[i + 1]] = top_dict[constraint_order[i]].copy()
                i += 1

        # trim away instances where DIFF_RATIO = 1.0 and where DT_RATIO = 1.0
        top_dict[IS_DT] = top_dict[c][(top_dict[c][DT_RATIO] < 1.0) & (top_dict[c][H_DIFF_RATIO] < 1.0)]

        self.top_panel = pd.Panel.from_dict(top_dict, orient='minor')

        # print % improvement results:
        pure_TOP = (self.top_panel.loc[top_param,:,IS_STD_TRK].sum() - self.top_panel.loc[top_param,:,STD_TRK].sum()) \
                   / self.top_panel.loc[top_param,:,STD_TRK].sum()
        forced_btrk = (
                   self.top_panel.loc[top_param, :, IS_FORCED_BTRK].sum() - self.top_panel.loc[top_param, :, STD_TRK].sum()) \
                   / self.top_panel.loc[top_param, :, STD_TRK].sum()
        forced_zero = (
                   self.top_panel.loc[top_param, :, IS_FORCED_ZERO].sum() - self.top_panel.loc[top_param, :, STD_TRK].sum()) \
                   / self.top_panel.loc[top_param, :, STD_TRK].sum()

        print '*** Improvement by TOP ***'
        print 'pure TOP:\t\t\t\t', '{0:0.2f}'.format(pure_TOP*100.0) + '%'
        print 'force backtracking:\t', '{0:0.2f}'.format(forced_btrk*100.0) + '%'
        print 'force flat:\t\t\t\t', '{0:0.2f}'.format(forced_zero*100.0) + '%'


        # plt.figure()
        # plt.plot(top_dict[IS_BTRK].index.hour, top_dict[IS_BTRK].loc[:, ANG_TOP], mec='k', color='None', marker='o')
        # plt.plot(top_dict[IS_FORCED_BTRK].index.hour, top_dict[IS_FORCED_BTRK].loc[:, ANG_TOP], mec='b', color='None', marker='o')
        # plt.plot(top_dict[IS_FORCED_ZERO].index.hour, top_dict[IS_FORCED_ZERO].loc[:, ANG_TOP], mec='r', color='None', marker='o')
        # plt.title(self.location)
        # plt.ylabel(ANG_TOP)
        # plt.plot(TOP_btrk.index.hour, TOP_btrk[ANG_TRACK], mec='r', color='None', marker='o')
        #
        plt.figure()
        # plt.plot(top_dict[IS_BTRK].loc[:, H_DIFF_RATIO], top_dict[IS_BTRK].loc[:, DT_RATIO], mec='k', color='None', marker='o')
        plt.plot(top_dict[IS_FORCED_BTRK].loc[:, H_DIFF_RATIO], top_dict[IS_FORCED_BTRK].loc[:, DT_RATIO], mec='b', color='None', marker='o')
        plt.plot(top_dict[IS_FORCED_ZERO].loc[:, H_DIFF_RATIO], top_dict[IS_FORCED_ZERO].loc[:, DT_RATIO], mec='r', color='None', marker='o')
        plt.title(self.location + ': Optimized for ' + top_param)
        plt.xlabel(H_DIFF_RATIO)
        plt.ylabel(DT_RATIO)

        # plt.figure()
        # plt.plot()

    def get_forced_zero(self):
        weird = self.top_panel.loc[IS_FORCED_ZERO, :, IS_FORCED_ZERO][self.top_panel.loc[IS_FORCED_ZERO, :, IS_FORCED_ZERO]]
        not_weird = self.top_panel.loc[IS_FORCED_ZERO, :, IS_FORCED_ZERO][self.top_panel.loc[IS_FORCED_ZERO, :, IS_FORCED_ZERO] == False]
        forced_panel = self.top_panel[:, weird.index, :]
        not_forced_panel = self.top_panel[:, not_weird.index, :]

        plt.figure()
        # plt.plot(top_panel.major_axis.hour, top_panel[ANG_TOP, :, 0], mec='k', color='None', marker='o')
        plt.plot(self.top_panel.major_axis.hour, self.top_panel[ANG_TOP, :, IS_FORCED_BTRK], mec='b', color='None', marker='o')
        plt.plot(forced_panel.major_axis.hour, forced_panel[ANG_TOP, :, IS_FORCED_BTRK], mec='y', mew=3, color='None', marker='o')

        plt.figure()
        # plt.plot(top_panel[H_DIFF_RATIO, :, 0], top_panel[DT_RATIO, :, 0], mec='k', color='None', marker='o')
        plt.plot(self.top_panel[H_DIFF_RATIO, :, 1], self.top_panel[DT_RATIO, :, IS_FORCED_BTRK], mec='b', color='None', marker='o')
        plt.plot(forced_panel[H_DIFF_RATIO, :, 1], forced_panel[DT_RATIO, :, IS_FORCED_BTRK], mec='y', mew=3, color='None', marker='o')

        plt.figure()
        plt.plot(not_forced_panel[H_DIFFUSE, :, 1], not_forced_panel[H_DIFF_RATIO, :, IS_FORCED_BTRK], mec='b', color='None', marker='o')
        plt.plot(forced_panel[H_DIFFUSE, :, 1], forced_panel[H_DIFF_RATIO, :, IS_FORCED_BTRK], mec='y', color='None', marker='o')

        return {True: forced_panel, False: not_forced_panel}

    def get_tracker_band(self, low, high):
        band_index = self.top_panel.major_axis[
            (self.top_panel[ANG_TRACK, :, STD_TRK] < high) & (self.top_panel[ANG_TRACK, :, STD_TRK] >= low)]
        return self.top_panel[:, band_index, :]

    def plot_very_diffuse_top(self):
        index_diffuse = self.top_panel.major_axis[self.top_panel.loc[H_DIFF_RATIO, :, STD_TRK] > 0.99]
        plt.figure()
        plt.plot(index_diffuse.hour, self.top_panel.loc[DT_RATIO, index_diffuse, IS_FORCED_BTRK], mec='b',
                 color='None', marker='o')
        plt.plot(index_diffuse.hour, self.top_panel.loc[DT_RATIO, index_diffuse, IS_FORCED_ZERO], mec='r',
                 color='None', marker='o')
        plt.title(self.location + ': DFI > 0.99')
        plt.ylabel(DT_RATIO)
        plt.xlabel('Hour of day')
        plt.legend([IS_FORCED_BTRK, IS_FORCED_ZERO])

    def plot_ang_inc_top(self, constraint=IS_FORCED_ZERO):
        plt.figure()
        x_data = ANG_INC
        y_data = ANG_TRACK
        plt.plot(self.top_panel.loc[x_data,:,STD_TRK], self.top_panel.loc[y_data,:,STD_TRK], mec='k',
                 color='None', marker='o')
        plt.plot(self.top_panel.loc[x_data, :, constraint], self.top_panel.loc[y_data, :, constraint], mec='b',
                 color='None', marker='o')
        plt.title(self.location)
        plt.ylabel(y_data)
        plt.xlabel(x_data)
        plt.legend([STD_TRK, constraint])

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

# test batch run of METEO files

panel_width = 2.01
gcr_range = [0.4, 0.5]
pitch_range = [round(panel_width/g,2) for g in gcr_range]
MET_range = ['Coyhaique_MN71_SYN.MET', 'Hobart_MN71_SYN.MET', 'Kalgoorlie_MN71_SYN.MET',
             'Pilar Obs__MN71_SYN.MET', 'Mendoza Airp__MN71_SYN.MET', 'Willis Is_MN71_SYN.MET',
             'Brasilia_MN71_SYN.MET', 'Cuiab__MN71_SYN.MET', 'Thiruvananthapuram_MN71_SYN.MET',
             'Belem_MN71_SYN.MET', 'S_o Lu_s_MN71_SYN.MET', 'Manaus_MN71_SYN.MET',
             'Goa_MN71_SYN.MET', 'New Delhi_MN71_SYN.MET', 'New Orleans_MN71_SYN.MET',
             'Port Arthur_MN71_SYN.MET', 'Alpena_MN71_SYN.MET', 'Salem_MN71_SYN.MET']
pitch_range_x = [x[0] for x in list(itertools.product(pitch_range, MET_range))]
MET_range_x = [x[1] for x in list(itertools.product(pitch_range, MET_range))]
filename_range = [str(p) +'p_' + m.rstrip('_SYN.MET') + '.csv' for p in pitch_range for m in MET_range]
var_dict = {
    BATCH_PITCH_T: pitch_range_x,
    BATCH_MET: MET_range_x,
    BATCH_FILENAME: filename_range
}
# world_batch = PVSystBatchSim('world', directory='world')
# world_batch.create_batch_file(base_variant='VC5', save_as='Napa_TMY3_BatchParams_0.CSV', description='varying lat, MET', variables=var_dict)
# world_batch.batch_filename = 'Napa_TMY3_BatchParams_0.CSV'
# world_batch.load_data()
# ***********************************

# generate files for Francesco
# ************************************************
# SITE SPECIFIC!  CHECK PVSYST!
projects_library = {
    'AUS': {
        'location': 'Townsville',
        'directory': 'Townsville AUS',
        'filename': 'Townsville_AUS_BatchParams_0.CSV',
        'descrip': 'test',
        'create batch file': False,
        'variables': {
            'fixed vc': ['VCH'],
            'fixed tilt': list(np.arange(-60.0, 61.0, 1.0)),
            'azimuth': [90.0],
            'backtracking vc': ['VC1'], # what happens if this is set to None?
            'phi min': [-60.0],
            'phi max': [60.0],
            },
        'TOP model': (ENERGY_ARRAY, 'max'),
    },
    'Napa': {
        'location': 'Napa',
        'directory': 'Napa',
        'filename': 'Napa_TMY3_BatchParams_0.CSV',
        'descrip': 'DRES Quarry model',
        'create batch file': False,
        'variables': {
            'fixed vc': ['VC3'],
            'fixed tilt': list(np.arange(-60.0, 61.0, 1.0)),
            'azimuth': [90.0],
            'backtracking vc': ['VC5'],  # what happens if this is set to None?
            'phi min': [-60.0],
            'phi max': [60.0],
        },
        'TOP model': (ENERGY_ARRAY, 'max'),
    },
    'Seattle': {
        'location': 'Seattle_TMY3',
        'directory': 'Seattle',
        'filename': 'Seattle_TMY3_Project_BatchParams_0.CSV',
        'descrip': 'test',
        'create batch file': True,
        'variables': {
            'fixed vc': ['VCF'],
            'fixed tilt': list(np.arange(-60.0, 61.0, 1.0)),
            'azimuth': [90.0],
            'backtracking vc': ['VCG'],  # what happens if this is set to None?
            'phi min': [-60.0],
            'phi max': [60.0],
        },
        'TOP model': (ENERGY_ARRAY, 'max'),
    }
}

projects_to_run = []
projects = {}

for p in projects_to_run:
    projects[p] = PVSystBatchSim(projects_library[p]['location'], directory=projects_library[p]['directory'])

    if projects_library[p]['create batch file'] is True:
        # if you need to create a batch file:
        num_sims = max([len(v[1]) for v in projects_library[p]['variables'].iteritems()])
        tilt = {True: projects_library[p]['variables']['fixed tilt'],
                False: projects_library[p]['variables']['fixed tilt'] * num_sims
                }[len(projects_library[p]['variables']['fixed tilt']) == num_sims]
        azim = {True: projects_library[p]['variables']['azimuth'],
                False: projects_library[p]['variables']['azimuth'] * num_sims
                }[len(projects_library[p]['variables']['azimuth']) == num_sims]
        vc = {True: projects_library[p]['variables']['fixed vc'],
              False: projects_library[p]['variables']['fixed vc'] * num_sims
              }[len(projects_library[p]['variables']['fixed vc']) == num_sims]
        phi_max = {True: projects_library[p]['variables']['phi max'],
              False: projects_library[p]['variables']['phi max'] * num_sims
              }[len(projects_library[p]['variables']['phi max']) == num_sims]
        phi_min = {True: projects_library[p]['variables']['phi min'],
              False: projects_library[p]['variables']['phi min'] * num_sims
              }[len(projects_library[p]['variables']['phi min']) == num_sims]
        if projects_library[p]['variables']['backtracking vc'] is not None:
            tilt.insert(0, '')
            azim.insert(0, '')
            vc.insert(0, projects_library[p]['variables']['backtracking vc'][0])
            phi_max.insert(0, 60.0)
            phi_min.insert(0, -60.0)
        variables_dict = dict(zip([BATCH_VC, BATCH_PLANE_TILT, BATCH_PLANE_AZIM, BATCH_PHI_MAX, BATCH_PHI_MIN],
                                  [vc, tilt, azim, phi_max, phi_min]))
        projects[p].create_batch_file(save_as=projects_library[p]['filename'],
                                      description=projects_library[p]['descrip'],
                                      variables=variables_dict)
    else:
        projects[p].batch_filename = projects_library[p]['filename']
        projects[p].load_data()
        projects[p].get_TOP_model(top_param=projects_library[p]['TOP model'][0],
                                  how=projects_library[p]['TOP model'][1])

    # if projects_library[p]['variables']['backtracking vc'] is None:
    #     # this check may be useful later
    #     pass









