__author__ = 'christina'

# import matplotlib  # version 1.5.1
# matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # version 0.18
import datetime as dt
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
IS_BTRK = 'Backtracking?'
IS_FORCED_BTRK = 'Forced backtracking?'
IS_FORCED_ZERO = 'Forced to zero?'
IS_TRK = 'Standard tracking?'
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
useful_names = [pvsyst_name_lookup[i] for i in [DATE, ANG_ELEV, ANG_AZ, ANG_INC, ANG_TRACK,
                                                H_GLOBAL, H_DIFFUSE, E_GLOBAL, E_DIRECT, E_DIFF_S, E_DIFF_G,
                                                ENERGY_ARRAY, ENERGY_GRID, SLOSS_ELEC]
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
PVSYST_BATCH_ORDER = [BATCH_IDENT, BATCH_VC, BATCH_FILENAME,
                      BATCH_SITE, BATCH_MET,
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
        self.is_tracking = False
        self.result_df = self.unpack_single_pvsyst_result()
        self.get_site_diffuseness()
        # self.daytime_df_index = self.get_daytime_df().index
        # self.avg_diffuse = self.result_df.loc[self.daytime_df_index, self.H_DIFF_RATIO].mean()
        # self.cloudy_days = self.get_cloudy_days()

    def __str__(self):
        return self.result_file

    def unpack_single_pvsyst_result(self):
        # Read top few lines to understand data column names and order:
        with open(self.result_file, 'r', 0) as f:
            # header_lines = [next(f) for x in xrange(11)]
            all_lines = f.readlines()
        # check the separator (always positioned at zeroth character of first line in file):
        self.separator = all_lines[1][0]
        # header_lines = all_lines[:11]
        col_names = all_lines[10].strip().split(self.separator)
        name_index_map = dict(zip(col_names, range(len(col_names))))
        # col_names = header_lines[10].strip().split(self.separator)
        usable_names = [u for u in useful_names if u in col_names]
        result_dict = {u: [] for u in usable_names}
        for i in range(13, len(all_lines)):
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

        # Read in PVSyst results file as data frame:
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
        plt.title(self.location, fontsize=16)
        plt.xlabel(H_DIFF_RATIO)
        plt.ylim([0, 1000])

    def export_as_csv(self):
        pvsyst_rename_inv = {p[1]: p[0] for p in pvsyst_rename.iteritems()}
        # Convert names back to PVSyst convention names:
        self.result_df.rename(columns=pvsyst_rename_inv, inplace=True)
        print "saving to .csv file"
        self.result_df.to_csv(self.location)
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
                    PVSystResult(self.location, self.batch_dict[BATCH_FILENAME][h])
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

    def create_batch_file(self, save_as='batch_filename', base_variant='VC', description='None',
                          variables=PVSYST_BATCH_PARAMS_DICT):

        separator = ';'
        num_sims = max([len(x[1]) for x in variables.iteritems()])

        # add the SIM_, filename, and comment columns to the variables columns:
        columns = variables.copy()
        columns[BATCH_IDENT] = ['SIM_' + str(x) for x in
                                  range(2, num_sims + 2)]  # start at SIM_2 bc PVsyst overwrites SIM_1
        columns[BATCH_COMMENT] = ['_'.join([str(v[1][n]) + PVSYST_BATCH_PARAMS_DICT_COMMENTER[v[0]]
                                            for v in variables.iteritems()]) for n in range(0, num_sims)]
        columns[BATCH_FILENAME] = [m + '.csv' for m in columns[BATCH_COMMENT]]

        # get columns sorted in PV syst order
        cols_in_use = [c for c in PVSYST_BATCH_ORDER if c in columns.keys()]

        with open(PVSYST_BATCH_TEMPLATE_FILE, 'r', 0) as f:
            batch_template_file = f.read().splitlines(1)

        header_w = batch_template_file[0:14]
        # replace some data in template header
        project_name_line = header_w[4].split(separator)
        project_name = project_name_line[3]
        project_name_line.remove(project_name)
        project_name_line.insert(3, self.location)
        header_w[4] = separator.join(project_name_line)

        variant_name_line = header_w[5].split(separator)
        variant_name = variant_name_line[2]
        descrip = variant_name_line[3]
        variant_name_line.remove(variant_name)
        variant_name_line.insert(2, base_variant)
        variant_name_line.remove(descrip)
        variant_name_line.insert(3, description)
        header_w[5] = separator.join(variant_name_line)

        if self.directory is None:
            new_filename = save_as
        else:
            new_filename = self.directory + '/' + save_as
        with open(new_filename, 'w') as f:

            # write header:
            for h in header_w:
                f.write(h)
            f.write(';\n')

            # write correct column names.  template requires column title on 2 lines, units on 3rd line, e.g.
            # Plane
            # tilt
            # [deg]
            f.write(separator.join([c.split('\n')[0] for c in cols_in_use]) + ';\n')
            f.write(separator.join([c.split('\n')[1] for c in cols_in_use]) + ';\n')
            f.write(separator.join([c.split('\n')[2] for c in cols_in_use]) + ';\n')
            f.write(';\n')

            # write data:
            # PVSyst will overwrite whatever is in the first line (SIM_1) with the base variant parameters, so write to
            # 2nd line
            f.write('SIM_1' + separator * len(cols_in_use) + '\n')
            for n in range(0, num_sims):
                f.write(separator.join([str(columns[c][n]) for c in cols_in_use]) + ';\n')
            f.write(';\n')

        # set the object variables:
        self.batch_file_created = True
        # self.batch_dict = batch_dict
        self.batch_filename = new_filename

    def get_TOP_model(self, opt_param=E_GLOBAL):

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

        # create a Panel where items = PVSyst result column names, major_axis= datetime, minor_axis = variants
        # to slice a Panel: panel.loc[item, major_axis, minor_axis]
        # batch_panel = pd.Panel.from_dict({r[0]: r[1].result_df for r in self.results_dict.iteritems()}, orient='minor')

        # only look at daytime hours
        day_index = self.results_dict.values()[0].result_df[self.results_dict.values()[0].result_df[H_GLOBAL] > 1.0].index

        # find max value and its index
        max_value = self.results_panel.loc[opt_param, day_index, :].apply(np.max, axis=1)
        max_minoraxis = self.results_panel.loc[opt_param, day_index, :].idxmax(axis=1)
        is_btrk = max_minoraxis == self.tracking_model.keys()[0]

        fixed_tilt = pd.Series(self.results_panel[BATCH_PLANE_TILT, :, :].lookup(max_minoraxis.index, max_minoraxis),
                               index=self.results_panel.major_axis)
        tracking = pd.Series(self.results_panel[ANG_TRACK, :, :].lookup(max_minoraxis.index, max_minoraxis),
                               index=self.results_panel.major_axis)
        tracker_angle = fixed_tilt.mask(is_btrk, other=tracking).apply(float)


        # create map of simulation names to panel tilt.
        # for backtracking simulation, panel tilt will be empty, so force it to a bogus number we can mask out later:
        panel_tilt_angles = [99.0 if t is '' else t for t in self.batch_dict[BATCH_PLANE_TILT]]
        panel_tilt_map = dict(zip(self.batch_dict[BATCH_COMMENT], panel_tilt_angles))

        TOP_btrk = self.tracking_model.values()[0].result_df.loc[day_index, :]
        TOP_0_model = pd.DataFrame(index=day_index,
                                 columns=list({H_DIFF_RATIO, H_GLOBAL, H_DIFFUSE,
                                               opt_param, ANG_FIXED, ANG_TRACK,
                                               IS_BTRK, IS_FORCED_BTRK, IS_FORCED_ZERO,
                                               ANG_TOP, ANG_TOP_INC,
                                               E_GLOBAL, E_DIRECT, E_DIFF_S, E_DIFF_G,
                                               DT_DELTA, DT_DELTA_ABS, DT_RATIO}))

        TOP_0_model[H_DIFF_RATIO] = self.results_panel.loc[H_DIFF_RATIO, day_index, self.results_dict.keys()[0]]
        TOP_0_model[H_GLOBAL] = self.results_panel.loc[H_GLOBAL, day_index, self.results_dict.keys()[0]]
        TOP_0_model[H_DIFFUSE] = self.results_panel.loc[H_DIFFUSE, day_index, self.results_dict.keys()[0]]
        TOP_0_model[opt_param] = max_value
        # 1st pass: just pick the angle that maximizes the opt_param (apply constraints later):
        TOP_0_model[ANG_FIXED] = max_minoraxis.map(panel_tilt_map)
        TOP_0_model[ANG_TRACK] = self.tracking_model.values()[0].result_df.loc[day_index, ANG_TRACK]
        TOP_0_model[IS_BTRK] = max_minoraxis == self.tracking_model.keys()[0]
        TOP_0_model[ANG_TOP] = TOP_0_model[ANG_FIXED].mask(TOP_0_model[IS_BTRK],
                                                         other=TOP_0_model[ANG_TRACK], inplace=False)
        TOP_0_model[ANG_TOP_INC] = pd.Series(self.results_panel[ANG_INC].lookup(max_minoraxis.index, max_minoraxis),
                                             index=day_index)
        TOP_0_model[E_GLOBAL] = pd.Series(self.results_panel[E_GLOBAL].lookup(max_minoraxis.index, max_minoraxis),
                                             index=day_index)
        TOP_0_model[E_DIRECT] = pd.Series(self.results_panel[E_DIRECT].lookup(max_minoraxis.index, max_minoraxis),
                                          index=day_index)
        TOP_0_model[E_DIFF_S] = pd.Series(self.results_panel[E_DIFF_S].lookup(max_minoraxis.index, max_minoraxis),
                                             index=day_index)
        TOP_0_model[E_DIFF_G] = pd.Series(self.results_panel[E_DIFF_G].lookup(max_minoraxis.index, max_minoraxis),
                                          index=day_index)
        TOP_0_model[DT_DELTA] = TOP_0_model.loc[:, ANG_TOP] - TOP_btrk.loc[:, ANG_TRACK]
        TOP_0_model[DT_DELTA_ABS] = TOP_0_model[DT_DELTA].apply(np.abs)
        TOP_0_model[DT_RATIO] = TOP_0_model[ANG_TOP] / TOP_btrk[ANG_TRACK]

        # constraint 1:
        # don't let TOP pick an angle more steep than backtracking to prevent unwanted shading
        # force to backtracking angle
        # TODO: check if this is necessary if we optimize on E_ARRAY
        TOP_1_model = TOP_0_model.copy()
        # is_outside_btrk_range =
        TOP_1_model[IS_FORCED_BTRK] = ~TOP_0_model[IS_BTRK] & (abs(TOP_0_model[ANG_FIXED]) > abs(TOP_0_model[ANG_TRACK]))
        TOP_1_model[ANG_TOP] = TOP_0_model[ANG_TOP].mask(TOP_1_model[IS_FORCED_BTRK], other=TOP_btrk[ANG_TRACK], inplace=False)

        TOP_1_model[ANG_TOP_INC] = TOP_0_model[ANG_TOP_INC].mask(TOP_1_model[IS_FORCED_BTRK], other=TOP_btrk[ANG_INC],
                                                                 inplace=False)
        TOP_1_model[E_GLOBAL] = TOP_0_model[E_GLOBAL].mask(TOP_1_model[IS_FORCED_BTRK], other=TOP_btrk[E_GLOBAL],
                                                                 inplace=False)
        TOP_1_model[E_DIRECT] = TOP_0_model[E_DIRECT].mask(TOP_1_model[IS_FORCED_BTRK], other=TOP_btrk[E_DIRECT],
                                                           inplace=False)
        TOP_1_model[E_DIFF_S] = TOP_0_model[E_DIFF_S].mask(TOP_1_model[IS_FORCED_BTRK], other=TOP_btrk[E_DIFF_S],
                                                           inplace=False)
        TOP_1_model[E_DIFF_G] = TOP_0_model[E_DIFF_G].mask(TOP_1_model[IS_FORCED_BTRK], other=TOP_btrk[E_DIFF_G],
                                                           inplace=False)
        TOP_1_model[DT_DELTA] = TOP_1_model.loc[:, ANG_TOP] - TOP_btrk.loc[:, ANG_TRACK]
        TOP_1_model[DT_DELTA_ABS] = TOP_1_model[DT_DELTA].apply(np.abs)
        TOP_1_model[DT_RATIO] = TOP_1_model.loc[:, ANG_TOP] / TOP_btrk.loc[:, ANG_TRACK]

        # constraint 2:
        # don't let TOP pick an angle inappropriate for time of day, e.g. +10 deg for 10am.
        # force to 0.0 deg
        TOP_2_model = TOP_1_model.copy()
        minoraxis_0deg = dict(zip(panel_tilt_map.values(), panel_tilt_map.keys()))[0.0]
        # is_facing_wrong_azimuth = (TOP_model[IS_BTRK] == False) \
        #                           & (((TOP_model.index.hour < 12) & (TOP_model[ANG_FIXED] > 0.0)) |
        #                              ((TOP_model.index.hour > 12) & (TOP_model[ANG_FIXED] < 0.0)))
        TOP_2_model[IS_FORCED_ZERO] = (~TOP_0_model[IS_BTRK] | ~TOP_1_model[IS_FORCED_BTRK]) \
                                  & (((TOP_1_model.index.hour < 12) & (TOP_1_model[ANG_TOP] > 0.0)) |
                                     ((TOP_1_model.index.hour > 12) & (TOP_1_model[ANG_TOP] < 0.0)))
        # TOP_model[IS_FORCED_ZERO] = is_facing_wrong_azimuth
        TOP_2_model[ANG_TOP] = TOP_1_model[ANG_TOP].mask(TOP_2_model[IS_FORCED_ZERO], other=0.0, inplace=False)
        # TOP_model[ANG_TOP] = TOP_model[ANG_TOP_2].copy()
        TOP_2_model[ANG_TOP_INC].mask(TOP_2_model[IS_FORCED_ZERO],
                                other=self.results_panel.loc[ANG_INC, day_index, minoraxis_0deg], inplace=True)
        TOP_2_model[E_GLOBAL] = TOP_1_model[E_GLOBAL].mask(TOP_2_model[IS_FORCED_ZERO],
                                                           other=self.results_panel.loc[E_GLOBAL, day_index, minoraxis_0deg],
                                                           inplace=False)
        TOP_2_model[E_DIRECT] = TOP_1_model[E_DIRECT].mask(TOP_2_model[IS_FORCED_ZERO],
                                                           other=self.results_panel.loc[E_DIRECT, day_index, minoraxis_0deg],
                                                           inplace=False)
        TOP_2_model[E_DIFF_S] = TOP_1_model[E_DIFF_S].mask(TOP_2_model[IS_FORCED_ZERO],
                                                           other=self.results_panel.loc[E_DIFF_S, day_index, minoraxis_0deg],
                                                           inplace=False)
        TOP_2_model[E_DIFF_G] = TOP_1_model[E_DIFF_G].mask(TOP_2_model[IS_FORCED_ZERO],
                                                           other=self.results_panel.loc[E_DIFF_G, day_index, minoraxis_0deg],
                                                           inplace=False)
        TOP_2_model[DT_DELTA] = TOP_2_model.loc[:, ANG_TOP] - TOP_btrk.loc[:, ANG_TRACK]
        TOP_2_model[DT_DELTA_ABS] = TOP_2_model[DT_DELTA].apply(np.abs)
        TOP_2_model[DT_RATIO] = TOP_2_model.loc[:, ANG_TOP] / TOP_btrk.loc[:, ANG_TRACK]


        plt.figure()
        plt.plot(TOP_0_model.index.hour, TOP_0_model[ANG_TOP], mec='k', color='None', marker='o')
        plt.plot(TOP_1_model.index.hour, TOP_1_model[ANG_TOP], mec='b', color='None', marker='o')
        plt.plot(TOP_2_model.index.hour, TOP_2_model[ANG_TOP], mec='r', color='None', marker='o')
        plt.title(self.location)
        plt.ylabel(ANG_TOP)
        # plt.plot(TOP_btrk.index.hour, TOP_btrk[ANG_TRACK], mec='r', color='None', marker='o')
        #
        plt.figure()
        # plt.plot(TOP_0_model[H_DIFF_RATIO], TOP_0_model[DT_RATIO], mec='k', color='None', marker='o')
        plt.plot(TOP_1_model[H_DIFF_RATIO], TOP_1_model[DT_RATIO], mec='b', color='None', marker='o')
        plt.plot(TOP_2_model[H_DIFF_RATIO], TOP_2_model[DT_RATIO], mec='r', color='None', marker='o')
        plt.title(self.location)
        plt.xlabel(H_DIFF_RATIO)
        plt.ylabel(DT_RATIO)
        # inc_70 = TOP_model[(TOP_model[ANG_INC] >= 70.0) & (TOP_model[ANG_INC] < 72.0)]
        # plt.plot(inc_70[H_DIFF_RATIO], inc_70[DT_DELTA_ABS], 'ro')

        # plt.figure(3)
        # plt.plot(TOP_model[ANG_TRACK], mec='k', color='None', marker='o')
        # plt.plot(TOP_model)
        self.top_panel = pd.Panel.from_dict(
            {'btrk': TOP_btrk, 0: TOP_0_model, 1: TOP_1_model, 2: TOP_2_model}, orient='minor')

    def get_forced_zero(self):
        weird = self.top_panel.loc[IS_FORCED_ZERO, :, 2][self.top_panel.loc[IS_FORCED_ZERO, :, 2]]
        not_weird = self.top_panel.loc[IS_FORCED_ZERO, :, 2][self.top_panel.loc[IS_FORCED_ZERO, :, 2] == False]
        forced_panel = self.top_panel[:, weird.index, :]
        not_forced_panel = self.top_panel[:, not_weird.index, :]

        plt.figure()
        # plt.plot(top_panel.major_axis.hour, top_panel[ANG_TOP, :, 0], mec='k', color='None', marker='o')
        plt.plot(self.top_panel.major_axis.hour, self.top_panel[ANG_TOP, :, 1], mec='b', color='None', marker='o')
        plt.plot(forced_panel.major_axis.hour, forced_panel[ANG_TOP, :, 1], mec='y', mew=3, color='None', marker='o')

        plt.figure()
        # plt.plot(top_panel[H_DIFF_RATIO, :, 0], top_panel[DT_RATIO, :, 0], mec='k', color='None', marker='o')
        plt.plot(self.top_panel[H_DIFF_RATIO, :, 1], self.top_panel[DT_RATIO, :, 1], mec='b', color='None', marker='o')
        plt.plot(forced_panel[H_DIFF_RATIO, :, 1], forced_panel[DT_RATIO, :, 1], mec='y', mew=3, color='None', marker='o')

        plt.figure()
        plt.plot(not_forced_panel[H_DIFFUSE, :, 1], not_forced_panel[H_DIFF_RATIO, :, 1], mec='b', color='None', marker='o')
        plt.plot(forced_panel[H_DIFFUSE, :, 1], forced_panel[H_DIFF_RATIO, :, 1], mec='y', color='None', marker='o')

        return {True: forced_panel, False: not_forced_panel}

    def get_tracker_band(self, low, high):
        band = self.top_panel[ANG_TRACK, :, 'btrk'][
            (self.top_panel[ANG_TRACK, :, 'btrk'] < high) & (self.top_panel[ANG_TRACK, :, 'btrk'] >= low)]
        return self.top_panel[:, band.index, :]

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

# test = PVSystResult('seattle', 'test folder/0 E.csv')
# test = PVSystBatchSim('napa', directory='test folder')
# test.batch_filename = 'Napa_TMY3_BatchParams_1.CSV'
# test.read_batch_file()
# test.load_data()
# test_2 = PVSystBatchSim('seattle', directory='test folder')
# test_2_dict = PVSYST_BATCH_PARAMS_DICT
# test_2_dict[BATCH_PLANE_TILT] = list(np.arange(-60,60,10.0))
# test_2_dict[BATCH_PLANE_AZIM] = [-90] * len(PVSYST_BATCH_PARAMS_DICT[BATCH_PLANE_TILT])
# test_2.create_batch_file(base_variant='VC2', description='testtest', batch_dict=test_2_dict)
# test_TOP = test.get_TOP_model(E_GLOBAL)

# params_dict = PVSYST_BATCH_PARAMS_DICT
# params_dict[BATCH_PLANE_TILT] = list(np.arange(-60,60.5,0.5))
# params_dict[BATCH_PLANE_AZIM] = [90] * len(params_dict[BATCH_PLANE_TILT])
#
# ground_1w = PVSystBatchSim('seattle', directory='Seattle')
# ground_1w.create_batch_file(save_as='Seattle_TMY3_Project_BatchParams_2.CSV', base_variant='VC2', description='slope_1W', batch_dict=params_dict)
#
# ground_0w = PVSystBatchSim('seattle', directory='Seattle')
# ground_0w.create_batch_file(save_as='Seattle_TMY3_Project_BatchParams_3.CSV', base_variant='VC3', description='slope_0W', batch_dict=params_dict)
#
# ground_6w = PVSystBatchSim('seattle', directory='Seattle')
# ground_6w.create_batch_file(save_as='Seattle_TMY3_Project_BatchParams_4.CSV', base_variant='VC4', description='slope_6W', batch_dict=params_dict)
#
# ground_5w = PVSystBatchSim('seattle', directory='Seattle')
# ground_5w.create_batch_file(save_as='Seattle_TMY3_Project_BatchParams_5.CSV', base_variant='VC5', description='slope_5W', batch_dict=params_dict)


# generate files for Francesco
# ************************************************
# SITE SPECIFIC!  CHECK PVSYST!
# *** AUSTRALIA ***
# f_fixed_vc = 'VCH'
# f_backtracking_vc = 'VC1'
# f_directory = 'Townsville AUS'
# f_save_as = 'Townsville_AUS_BatchParams_0.CSV'
# f_descrip = 'test'

# *** NAPA ***
f_fixed_vc = 'VC0'
f_backtracking_vc = 'VC1'
f_directory = 'Napa'
f_save_as = 'Napa_TMY3_BatchParams_1.CSV'
f_descrip = 'test'

# *** SEATTLE ***
# f_fixed_vc = 'VCF'
# f_backtracking_vc = 'VCG'
# f_directory = 'Seattle'
# f_save_as = 'Seattle_Project_BatchParams_2.CSV'
# f_descrip = 'test'
# *************************************************
f_tilt = list(np.arange(-60, 61, 1))
f_azim = len(f_tilt) * [90]
f_vc = len(f_tilt) * [f_fixed_vc]
f_phi_max = len(f_tilt) * ['']
f_phi_min = len(f_tilt) * ['']
# now insert the backtracking VC at the top (not actually sure this matters?)
f_tilt.insert(0, '')
f_azim.insert(0, '')
f_vc.insert(0, f_backtracking_vc)
f_phi_max.insert(0, 60)
f_phi_min.insert(0, -60)
# # or append it:
# f_tilt.append('')
# f_azim.append('')
# f_vc.append(backtracking_vc)
# f_phi_max.append(60)
# f_phi_min.append(-60)
variables_dict = dict(zip([BATCH_VC, BATCH_PLANE_TILT, BATCH_PLANE_AZIM, BATCH_PHI_MAX, BATCH_PHI_MIN],
                          [f_vc, f_tilt, f_azim, f_phi_max, f_phi_min]))
francesco = PVSystBatchSim(f_directory, directory=f_directory)
# francesco.create_batch_file(save_as=f_save_as, base_variant=f_backtracking_vc, description=f_descrip, variables=variables_dict)
francesco.batch_filename = f_save_as
francesco.load_data()
# f_legend = []
# for f in francesco.results_dict.iteritems():
#     plt.plot(f[1].result_df.loc['1990-03', E_GLOBAL])
#     f_legend.append(f[0])
#
# plt.legend(f_legend)
# plt.show()

# generate files for Yudong
# ************************************************
# SITE SPECIFIC!  CHECK PVSYST!
y_fixed_vc = 'VCH'
y_backtracking_vc = 'VC1'
y_directory = 'Townsville AUS'
y_save_as = 'Townsville_AUS_BatchParams_1.CSV'
y_descrip = 'test'
# *************************************************








