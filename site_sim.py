__author__ = 'christina'

import matplotlib  # version 1.5.1
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # version 0.17.1
import pvlib as pv  # version 0.2.2

# RESOURCES:
# Interesting thread on daylight savings, timezones, and capturing errors in irradiance sensor alignment by peak finding:
# https://github.com/pvlib/pvlib-python/issues/47

# MODELING STEPS:
# (1) Lat/long, times of year (series) -> solar position (series)
# (2) solar position (series) -> theoretical solar resource available on horizontal surface (series)
# (3) array position (series) + solar position (series) -> angle of incidence (series)
# (4) angle of incidence (series) + solar resource (theoretical or TMY3) -> plane of array irradiance
# (5) plane of array irradiance + PV performance -> energy generation
# (6) Look at irradiance variability data and compare to TYM3 data.

# TODO: why is error > 100% does correcting sunrise/sunset hours resolve this?
# TODO: why does model fail for southern hemisphere?
# TODO: compare errors when drop sunrise/set hours
# TODO: backtracking
# TODO: gcr -- how does PVSyst model & account for gcr? nextracker installs at gcr = 0.4, pvlib assumes 0.29

# USER STEPS:
# (1) Define a Location and import a weather file
# (2) Pick an analysis to run
# (3) Pick some plots to make

# Site-specific parameters
MODEL_YEAR = 2010
reno = pv.location.Location(latitude=39.50,
                            longitude=-119.78,
                            tz='US/Pacific',
                            altitude=1347,
                            name='Reno')
seattle = pv.location.Location(latitude=47.65,
                               longitude=-122.30,
                               tz='US/Pacific',
                               altitude=13,
                               name='Seattle')
san_francisco = pv.location.Location(latitude=37.62,
                                     longitude=-122.37,
                                     tz='US/Pacific',
                                     altitude=2,
                                     name='San Francisco')
hayward = pv.location.Location(latitude=37.67,
                               longitude=-122.12,
                               tz='US/Pacific',
                               altitude=1,
                               name='Hayward')
vallenar = pv.location.Location(latitude=-28.6,
                               longitude=-70.77,
                               tz='US/Eastern',
                               altitude=524,
                               name='Vallenar')  # pvsyst file: tz = -4.0,
                               # but pytz.timezone says Chile uses America/Santiago, which is -3.0.
                               # possible DST issue?

DEFAULT_SITE = hayward
DEFAULT_TMY_FILE = '../PVresearch/724935TYA.CSV'
DEFAULT_PVSYST_FILE = '../PVresearch/Hayward Air Term_Project_HourlyRes_0.CSV'
# DEFAULT_PVSYST_FILE = '../PVresearch/Reno_Project_HourlyRes_0.CSV'
# DEFAULT_PVSYST_FILE = '../PVresearch/San Francisco_Project_HourlyRes_1.csv'
# DEFAULT_PVSYST_FILE = '../PVresearch/Seattle_Project_HourlyRes_0.csv'

SITE_ALBEDO = 0.2  # PVSyst default = 0.2
# Array parameters
ARRAY_TILT = 0
ARRAY_AZ = 180
ARRAY_SWEEP = 60
TRACKER_ERROR = 5

ANG_ELEV = 'Sun Elevation'
ANG_AZ = 'Sun Azimuth'  # pvlib - 180 = pvsyst
ANG_INC = 'Incident Angle'
ANG_SURF_AZ = 'Panel Azimuth'  # pvlib - 180 = pvsyst
ANG_SURF_TILT = 'Panel Tilt'
ANG_TRACK = 'Tracker Angle'
ANG_ZENITH = 'Sun Zenith'
H_GLOBAL = 'Global Horizontal'
H_DIFFUSE = 'Diffuse Horizontal'
H_DIRECT = 'Direct Horizontal'
I_GLOBAL = 'Global Incident'
I_DIRECT = 'Direct Incident'
I_DIFF_G = 'Ground Diffuse Incident'
I_DIFF_S = 'Sky Diffuse Incident'
I_DIFF_T = 'Total Diffuse Incident'

pvsyst_rename = {
    'HSol': ANG_ELEV,
    'AzSol': ANG_AZ,  # 180 deg out of phase with pvlib
    'AngInc': ANG_INC,
    'PlTilt': ANG_SURF_TILT,
    'PlAzim': ANG_SURF_AZ,  # 180 deg out of phase with pvlib
    'PhiAng': ANG_TRACK,
    'GlobHor': H_GLOBAL,
    'DiffHor': H_DIFFUSE,
    'BeamHor': H_DIRECT,
    'GlobInc': I_GLOBAL,
    'BeamInc': I_DIRECT,
    'DifSInc': I_DIFF_S,
    'Alb Inc': I_DIFF_G
}

pvlib_rename = {
    'elevation': ANG_ELEV,
    'azimuth': ANG_AZ,  # 180 deg out of phase with pvlib
    'zenith': ANG_ZENITH,
    'tracker_theta': ANG_TRACK,  # negative before noon (facing east)
    'aoi': ANG_INC,
    'surface_azimuth': ANG_SURF_AZ,  # 180 deg out of phase with PVSyst
    'surface_tilt': ANG_SURF_TILT,  # always positive value
    'poa_global': I_GLOBAL,
    'poa_diffuse': I_DIFF_T,
    'poa_direct': I_DIRECT
}

def get_pvlib_results(site=DEFAULT_SITE, filename=DEFAULT_PVSYST_FILE, az=ARRAY_AZ, tilt=ARRAY_TILT, source='PVSYST'):

    # which is worse: elevation or azimuth error?

    # TODO: make custom exception class for this:
    if source not in ['TMY3', 'PVSYST']:
        print 'Accepted sources are "PVSYST" and "TMY3".'

    # create tz-aware date_range with hourly intervals:
    # TODO: how to define timezone with float instead of string? then can read from project file?
    hourly_range = pd.date_range(start=pd.datetime(MODEL_YEAR, 1, 1),
                                 periods=8760, freq='1H', tz=site.tz)
    daily_range = pd.date_range(start=pd.datetime(MODEL_YEAR, 1, 1),
                                 periods=365, freq='1D', tz=site.tz)


    if source == 'PVSYST':
        # PVSyst sun position values are calculated on the 0:30 of each hour but associated with the 0:00 timestamp,
        # so we need to compare PVSyst values to the pvlib values calculated at the 0:30.
        hourly_range_30shift = hourly_range.shift(30, freq='Min')
        pvsyst_DF = pd.read_csv(filename, sep=';', header=0, skiprows=[0,1,2,3,4,5,6,7,8,9,11],
                                index_col=0, parse_dates=True)
        pvsyst_DF.index = hourly_range_30shift

        # *** PVLIB SOLAR POSITION: ***
        # columns: apparent_elevation, elevation, apparent_zenith, zenith, azimuth, equation of time
        solpos_DF = pv.solarposition.get_solarposition(hourly_range_30shift, site, method='nrel_numpy')

        # change solpos to match PVSyst method:
        # TODO: account for sun position at sunrise/set hours as described in pvsyst forum:
        # http://forum.pvsyst.com/viewtopic.php?f=4&t=2098:
        # The times when this arises are very close to sunrise or sunset.
        # For these hours including the sunrise (or sunset), PVsyst considers the time between sunrise and the end of the hour,
        # and computes the solar geometry at the middle of this interval.
        # The Direct normal is computed on the basis of the GlobHor, DiffHor and divided by the sin(sun height). The cut on
        # Sun height is rather low (I dont remember the value).
        # But there is a cut of 10 W/m2 on the GlobInc value for the rest of the simulation.
        # Now for your tracking situation (probably with backtracking), the plane is almost horizontal (1-2 degrees) so
        # that the GlobInc value is very low at sunrise. On the other hand, there is a cut on the incidence angle (>88 deg
        # doesn't make much sense due to the IAM loss). This explains the null values of BeamInc.

        sunrise_set = pv.solarposition.get_sun_rise_set_transit(daily_range, site)

        GHI = pvsyst_DF['GlobHor']
        DHI = pvsyst_DF['DiffHor']
        DNI = pvsyst_DF['BeamHor'] / pv.tools.cosd(solpos_DF['zenith'])  # TODO: apparent_zenith?

    else:  # source == 'TMY3':
        (tmy_DF, tmy_dict) = pv.tmy.readtmy3(filename)
        # check that location object is consistent with TMY3 file:
        lat_error = abs(tmy_dict['latitude'] - site.latitude)
        long_error = abs(tmy_dict['longitude'] - site.longitude)
        if (lat_error > 1) | (long_error > 1):
            print 'Location attributes are inconsistent.'
            print site
            print 'TMY3 lat = ' + str(tmy_dict['latitude']) + '; TMY3 long = ' + str(tmy_dict['longitude'])

        # *** PVLIB SOLAR POSITION: ***
        # columns: apparent_elevation, elevation, apparent_zenith, zenith, azimuth, equation of time
        solpos_DF = pv.solarposition.get_solarposition(hourly_range, site, method='nrel_numpy')

        GHI = tmy_DF['GHI']
        DHI = tmy_DF['DHI']
        DNI = tmy_DF['DNI']


    # *** PVLIB TRACKING: ***
    # columns: tracker_theta, aoi, surface_tilt, surface_azimuth
    # Note: PVSyst azimuth = 0 means south. pvlib azimuth = 0 means north, so pvlib model should use axis_azimuth = 180
    # to approx PVSyst set-up.
    # tODO: does it matter if azimuth is 180 out of phase?
    # if site.latitude < 0:
    #     ARRAY_AZ = 0  # in pvlib, az = 0 is North..
    # else:
    #     ARRAY_AZ = 180
    tracking_DF = pv.tracking.singleaxis(solpos_DF['apparent_zenith'],
                                         solpos_DF['azimuth'],
                                         axis_azimuth=az, axis_tilt=tilt, max_angle=ARRAY_SWEEP,
                                         backtrack=False)
    # *** PVLIB IRRADIANCE: ***
    # available methods are: asce, spencer, pyephem
    # PVSyst assumes solar constant = 1367; using this value in the model only changes annual POA sky diffuse by 0.1%
    dni_extra = pd.Series(pv.irradiance.extraradiation(solpos_DF.index.dayofyear,
                                                       solar_constant=1367,
                                                       method='asce'), index=solpos_DF.index)  # type = pd.Series
    airmass = pv.atmosphere.relativeairmass(solpos_DF['apparent_zenith'])
    # *** PVLIB GROUND AND SKY IRRADIANCE: ***
    ground_diffuse = pv.irradiance.grounddiffuse(tracking_DF['surface_tilt'],
                                                 GHI,
                                                 albedo=SITE_ALBEDO)  # type = pd.Series
    ground_diffuse.name = 'Ground'
    haydavies_diffuse = pv.irradiance.haydavies(tracking_DF['surface_tilt'],
                                                tracking_DF['surface_azimuth'],
                                                DHI,
                                                DNI,
                                                dni_extra,
                                                solpos_DF['apparent_zenith'],
                                                solpos_DF['azimuth'])  # type = pd.Series
    haydavies_diffuse.name = 'H-D'
    perez_diffuse = pv.irradiance.perez(tracking_DF['surface_tilt'],
                                        tracking_DF['surface_azimuth'],
                                        DHI,
                                        DNI,
                                        dni_extra,
                                        solpos_DF['apparent_zenith'],
                                        solpos_DF['azimuth'],
                                        airmass)  # type = pd.Series
    perez_diffuse.name = 'Perez'
    # Toggle sky diffuse model method:
    sky_diffuse = perez_diffuse
    # sky_diffuse = haydavies_diffuse

    # ***PVLIB INCIDENT IRRADIANCE: ***
    # returns: DF with columns poa_global, poa_direct, poa_diffuse
    poa_DF = pv.irradiance.globalinplane(tracking_DF['aoi'],
                                         DNI,
                                         sky_diffuse,
                                         ground_diffuse)  # type = pd.DataFrame
    # force pvlib azimuth angles into pvsyst sign convention:
    solpos_DF['azimuth'] -= 180
    tracking_DF['surface_azimuth'] -= 180

    # return only columns that are interesting:
    result = pd.concat([solpos_DF[['elevation', 'azimuth']],
                        tracking_DF[['surface_azimuth', 'surface_tilt', 'aoi', 'tracker_theta']],
                        poa_DF], axis=1)
    result.rename(columns=pvlib_rename, inplace=True)
    result[I_DIFF_S] = sky_diffuse
    result[I_DIFF_G] = ground_diffuse
    result[H_GLOBAL] = GHI
    result[H_DIFFUSE] = DHI
    result[H_DIRECT] = DNI * pv.tools.cosd(solpos_DF['zenith'])

    return result


def get_pvsyst_results(site=DEFAULT_SITE, filename=DEFAULT_PVSYST_FILE):
    # PVSyst sun position values are calculated on the 0:30 of each hour but associated with the 0:00 timestamp,
    # so we need to compare PVSyst values to the pvlib values calculated at the 0:30.
    hourly_range_30shift = pd.date_range(start=pd.datetime(MODEL_YEAR, 1, 1,0, 30),
                                 periods=8760, freq='1H', tz=site.tz)

    # Unpack results from file:
    # Read in PVSyst results file as data frame:
    # columns: HSol (sun height), AzSol (sun azimuth), AngInc (incidence angle), AngProf (profile angle),
    # PlTilt (plane tilt: tracking), PlAzim (plane azimuth: tracking), PhiAng (phi angle: tracking),
    # DiffHor, BeamHor, GlobHor, T Amb, GlobInc, BeamInc, DifSInc, Alb Inc,
    # GlobIAM, GlobEFf, BeamEff, DiffEff, Alb_Eff
    pvsyst_DF = pd.read_csv(filename, sep=';', header=0, skiprows=[0,1,2,3,4,5,6,7,8,9,11],
                            index_col=0, parse_dates=True)
    pvsyst_DF.index = hourly_range_30shift
    result = pvsyst_DF.copy()[['HSol', 'AzSol', 'AngInc', 'PhiAng', 'PlTilt', 'PlAzim', 'GlobHor', 'DiffHor', 'BeamHor',
                               'GlobInc', 'BeamInc', 'DifSInc', 'Alb Inc']]

    # result = pd.concat([pvsyst_DF[['HSol', 'AzSol', 'AngInc', 'PhiAng']], poa_DF], axis=1)
    result.rename(columns=pvsyst_rename, inplace=True)
    result[I_DIFF_T] = result[I_DIFF_S] + result[I_DIFF_G]

    # result_cleaned = drop_negs(result)

    return result

def drop_negs(df, set_to=0):
    # could also set_to=np.nan
    # find the columns that have negative irradiation values:
    min_val = df.describe().loc['min', :].drop([ANG_TRACK, ANG_AZ])
    drop_cols = min_val[min_val < 0]
    df_clean = df.copy()
    for d in drop_cols.index:
        df_clean.loc[:, d] = [set_to if x < 0 else x for x in df.loc[:, d]]
    return df_clean


def get_annual_error(hourly_DF_1, hourly_DF_2):
    DF_1_annual = hourly_DF_1.resample('A', how='sum').dropna().squeeze()  # Wh/m2
    DF_2_annual = hourly_DF_2.resample('A', how='sum').dropna().squeeze()  # Wh/m2
    percent_error_dict = {
        'global': (DF_1_annual[I_GLOBAL] - DF_2_annual[I_GLOBAL]) / DF_1_annual[I_GLOBAL] * 100.0,
        'direct': (DF_1_annual[I_DIRECT] - DF_2_annual[I_DIRECT]) / DF_1_annual[I_DIRECT] * 100.0,
        'sky': (DF_1_annual[I_DIFF_S] - DF_2_annual[I_DIFF_S]) / DF_1_annual[I_DIFF_S] * 100.0,
        'ground': (DF_1_annual[I_DIFF_G] - DF_2_annual[I_DIFF_G]) / DF_1_annual[I_DIFF_G] * 100.0
    }
    return percent_error_dict


def when_large_error(pvsyst_df, pvlib_df, threshold=0.2):
    pvsyst_dict = {}
    pvlib_dict = {}
    error_dict = {}
    error_df = get_error(pvsyst_df, pvlib_df)
    for c in error_df.columns:
        if error_df.describe().loc['std', c] >= 0.5:  # we only care if the series has large standard deviation
            error_dict[c] = error_df[abs(error_df.loc[:, c]) > threshold]
            pvsyst_dict[c] = pvsyst_df.loc[error_dict[c].index, :]
            pvlib_dict[c] = pvlib_df.loc[error_dict[c].index, :]
            plt.figure(figsize=(20,5))
            plt.plot(pvsyst_dict[c].index, pvsyst_dict[c].loc[:, c].values, 'b.-')
            plt.plot(pvlib_dict[c].index, pvlib_dict[c].loc[:, c].values, 'g.-')
            plt.legend(['PVSyst', 'pvlib'])
            plt.title(c)
            print c + ' (len = ' + str(len(error_dict[c].index)) + '): std = ' + str(error_df.describe().loc['std', c])
            print pd.unique(error_dict[c].index.hour)
    return pvsyst_dict, pvlib_dict, error_dict


def get_error(hourly_DF_1, hourly_DF_2):
    error_df = ((hourly_DF_1 - hourly_DF_2) / hourly_DF_1).replace([np.inf, -np.inf], np.nan).dropna()

    # return (error_df, min_max_df)
    return error_df

def plot_error(error_df, series=[ANG_ELEV]):
    # series must be provided as a list, not a string!
    # TODO: make assertion that series is a list
    if type(series) != list:
        series = [series]

    plt.figure(figsize=(20,5))
    for s in series:
        plt.plot(error_df.index, error_df[s])
    plt.legend(series, loc='upper left')

def plot_annual_comparison(hourly_DF_1, name_1, hourly_DF_2, name_2, site=DEFAULT_SITE):
    # Plot annual POA irradiation, stacked by component:
    annual_irrad_1 = hourly_DF_1[[I_DIRECT, I_DIFF_S, I_DIFF_G]].resample('A', how='sum') / 1000.0  # kWh/m2
    annual_irrad_2 = hourly_DF_2[[I_DIRECT, I_DIFF_S, I_DIFF_G]].resample('A', how='sum') / 1000.0  # kWh/m2
    annual_irrad = pd.concat([annual_irrad_1, annual_irrad_2], axis=0)
    annual_irrad.index = [name_1, name_2]
    annual_irrad.plot(kind='bar', stacked=True, figsize=(10,10))
    plt.title(name_1 + ' vs. ' + name_2 + '\n' + 'Annual Irradiation Flux by Component for ' + site.name)
    plt.ylabel('kWh/m2')
    return (annual_irrad_1, annual_irrad_2)

def plot_compare(pvsyst_df, pvlib_df, param, title_str):
    plt.figure(figsize=(18, 3))
    plt.plot(pvsyst_df.loc['2010-03-01':'2010-03-07'].index, pvsyst_df.loc['2010-03-01':'2010-03-07', param], 'k.-')
    plt.plot(pvlib_df.loc['2010-03-01':'2010-03-07'].index, pvlib_df.loc['2010-03-01':'2010-03-07', param], 'r.-')
    plt.legend(['PVSyst', 'pvlib'])
    plt.title(title_str)

# *** DOES DIFFUSE TRACKING MAKE SENSE ?? ******************************************************************************
def analyze_diffuse(poa_DF, site=DEFAULT_SITE):
    # When global horiz > global incident, it's better to be flat!
    better_flat_DF = poa_DF[(poa_DF[H_GLOBAL] > poa_DF[I_GLOBAL]) & (poa_DF[I_GLOBAL] > 1.0)]
    better_flat_DF['Flat_Frac'] = (better_flat_DF[H_GLOBAL] -
                                       better_flat_DF[I_GLOBAL]) / better_flat_DF[I_GLOBAL]

    better_flat_annual = float(better_flat_DF[H_GLOBAL].resample('A', how='sum') -
                               better_flat_DF[I_GLOBAL].resample('A', how='sum'))  # Wh/m2/yr
    better_flat_annual_fraction = float(better_flat_annual / poa_DF[I_GLOBAL].resample('A', how='sum'))

    print 'Hourly diffuse tracking could increase annual POA irradiation in ' + site.name + ' by ' \
          + '{0:.2%}'.format(better_flat_annual_fraction) + ', or ' + '{0:.0f}'.format(better_flat_annual) + ' Wh/m2.'

    # Plot average additional irradiance available for each hour in the day:
    plt.figure(figsize=(8, 8))
    plt.xlim(0,24)
    plt.xticks([0, 6, 12, 18, 24])
    plt.ylim(0, 1.0)
    plt.plot(better_flat_DF['Flat_Frac'].groupby(better_flat_DF.index.hour).mean(), 'bo',
             markeredgecolor='None')
    plt.title('Average Additional Irradiance Available on Flat Panel ' + '\n' +
              'as Fraction of Incident Irradiance on Tracking Panel' + '\n' + ' for ' + site.name)
    plt.xlabel('Hour of Day')

    # Show better_flat in scatter plot wrt diff-hor / glob-hor.
    better_flat_DF['Horiz-Diff/Global'] = better_flat_DF[H_DIFFUSE] / better_flat_DF[H_GLOBAL]
    better_flat_DF.plot.scatter(x='Horiz-Diff/Global', y='Flat_Frac', c='None', edgecolors='b')

    plt.show()
    plt.clf()
    return better_flat_DF