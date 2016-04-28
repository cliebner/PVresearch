__author__ = 'christina'

import matplotlib  # version 1.5.1
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import pandas as pd  # version 0.17.1
import pvlib as pv  # version 0.2.2

# *** EDIT THIS SECTION ONLY TO TEST OTHER LOCATIONS *******************************************************************
# Site-specific parameters
MODEL_YEAR = 2010
# SITE_NAME = 'San Francisco, CA'
# PVSYST_RES_FILE = '../PVresearch/San Francisco_Project_HourlyRes_1.csv'
# LAT = 37.62
# LONG = -122.37
# ALT = 2

SITE_NAME = 'Reno, NV'
PVSYST_RES_FILE = '../PVresearch/Reno_Project_HourlyRes_0.csv'
LAT = 39.50
LONG = -119.78
ALT = 1347

# SITE_NAME = 'Seattle, WA'
# PVSYST_RES_FILE = '../PVresearch/Seattle_Project_HourlyRes_0.csv'
# LAT = 47.65
# LONG = -122.30
# ALT = 13
TIMEZONE = 'US/Pacific'
SITE_ALBEDO = 0.2  # PVSyst default = 0.2
# Array parameters
PANEL_AZ = 180  # in pvlib, az = 0 is North
PANEL_TILT = 0
PANEL_SWEEP = 60
TRACKER_ERROR = 5
# *** END OF SITE-SPECIFIC PARAMETERS **********************************************************************************

mar20 = str(MODEL_YEAR) + '-03-20'
jun21 = str(MODEL_YEAR) + '-06-21'
sept23 = str(MODEL_YEAR) + '-09-23'
dec21 = str(MODEL_YEAR) + '-12-21'

# create tz-aware date_range with hourly intervals:
hourly_range = pd.date_range(start=pd.datetime(MODEL_YEAR, 1, 1),
                             periods=8760, freq='1H', tz=TIMEZONE)
# PVSyst sun position values are calculated on the 0:30 of each hour but associated with the 0:00 timestamp, so we need
# to compare PVSyst values to the pvlib values calculated at the 0:30.
hourly_range_30shift = hourly_range.shift(30, freq='Min')

# Read in PVSyst results file as data frame:
# columns: HSol (sun height), AzSol (sun azimuth), AngInc (incidence angle), AngProf (profile angle),
# PlTilt (plane tilt: tracking), PlAzim (plane azimuth: tracking), PhiAng (phi angle: tracking),
# DiffHor, BeamHor, GlobHor, T Amb, GlobInc, BeamInc, DifSInc, Alb Inc,
# GlobIAM, GlobEFf, BeamEff, DiffEff, Alb_Eff
pvsyst_DF = pd.read_csv(PVSYST_RES_FILE, sep=';', header=0, skiprows=[0,1,2,3,4,5,6,7,8,9,11],
                        index_col=0, parse_dates=True)
pvsyst_DF['DifTInc'] = pvsyst_DF['DifSInc'] + pvsyst_DF['Alb Inc']
pvsyst_DF.index = hourly_range

# Create pvlib Location object for pvlib model:
pvlib_Loc = pv.location.Location(LAT, LONG, tz='US/Pacific', altitude=ALT)

# *** PVLIB SOLAR POSITION: ***
# columns: apparent_elevation, elevation, apparent_zenith, zenith, azimuth, equation of time
pvlib_solpos_30shift_DF = pv.solarposition.get_solarposition(hourly_range_30shift, pvlib_Loc, method='nrel_numpy')
pvlib_solpos_30shift_DF.index = hourly_range  # Force pvlib dataframe to same index as pvsyst

# *** PVLIB TRACKING: ***
# columns: tracker_theta, aoi, surface_tilt, surface_azimuth
# Note: PVSyst azimuth = 0 means south. pvlib azimuth = 0 means north, so pvlib model should use axis_azimuth = 180
# to approx PVSyst set-up.
pvlib_tracking_DF = pv.tracking.singleaxis(pvlib_solpos_30shift_DF['apparent_zenith'],
                                           pvlib_solpos_30shift_DF['azimuth'],
                                           axis_azimuth=PANEL_AZ, axis_tilt=PANEL_TILT, max_angle=PANEL_SWEEP,
                                           backtrack=False)

# Introduce a tracking position error:
# pvlib_tracking_DF['tracker_theta_w_error'] = [(x + TRACKER_ERROR) if x < PANEL_SWEEP else PANEL_SWEEP for x in pvlib_tracking_DF['tracker_theta'].dropna()]
# pvlib_tracking_DF['surface_tilt_w_error'] = [abs(x) for x in pvlib_tracking_DF['tracker_theta_w_error']]
# pvlib_tracking_DF['aoi_w_error'] = pv.irradiance.aoi(pvlib_tracking_DF['surface_tilt_w_error'],
#                                                      pvlib_tracking_DF['surface_azimuth'],
#                                                      pvlib_solpos_30shift_DF['apparent_zenith'],
#                                                      pvlib_solpos_30shift_DF['azimuth'])

# *** PVLIB IRRADIANCE: ***
# available methods are: asce, spencer, pyephem
# PVSyst assumes solar constant = 1367; using this value in the model only changes annual POA sky diffuse by 0.1%
dni_extra_pvsyst = pd.Series(pv.irradiance.extraradiation(hourly_range.dayofyear,
                                                          solar_constant=1367,
                                                          method='spencer'), index=hourly_range)  # type = pd.Series
dni_extra_pvlib = pd.Series(pv.irradiance.extraradiation(hourly_range.dayofyear,
                                                         solar_constant=1366.1,
                                                         method='asce'), index=hourly_range)  # type = pd.Series
airmass_pvlib = pv.atmosphere.relativeairmass(pvlib_solpos_30shift_DF['apparent_zenith'])
airmass_pvsyst = pv.atmosphere.relativeairmass(90 - pvsyst_DF['HSol'])

# *** PVLIB CLEARSKY irradiance: (for completeness; not needed to compare to PVSyst) **********************************
# columns: ghi, dni, dhi
# pvlib available methods are: haurwitz (simple; input = zenith, output = GHI), ineichen
# Linke turbidity PVSyst: default = 3.5 (range = 2.0 to 5.0)
# Linke turbidity pvlib: default None = automatic table look-up, where value depends on month of year and location.
# turbidity value can make a big difference!
# ghi = dhi + cos(zenith) * dni
clearsky_DF = pv.clearsky.ineichen(hourly_range, pvlib_Loc,
                                   linke_turbidity=None,
                                   solarposition_method='nrel_numpy',
                                   zenith_data=pvlib_solpos_30shift_DF['zenith'],
                                   airmass_model='young1994',
                                   airmass_data=None,
                                   interp_turbidity=True)  # type = pd.DataFrame

# Transform DHI into plane-of-array (POA) ground and sky components.
# Since DHI assumes flat panel, we need to create a plane-of-array ground diffuse model,
# since array will view ground as it tracks.
# Based on theoretical clear sky model:
ground_diffuse_clearsky = pv.irradiance.grounddiffuse(pvlib_tracking_DF['surface_tilt'],
                                                      clearsky_DF['ghi'],
                                                      albedo=SITE_ALBEDO)  # type = pd.Series
ground_diffuse_clearsky.name = 'Ground'

# pvlib sky diffuse methods are: isotropic, Klucher, Reindl, Hay-Davies, Perez
# isotropic and Klucher are similar and do not vary with surface azimuth (so not useful for tracking systems)
# Reindl and Hay-Davies are similar (also use DNI and extraterrestrial, which accounts for scattering around sun)
# Perez includes 'horizon band' approach based on existing data and airmass coefficient.
# PVSyst sky diffuse methods are: Hay-Davies, Perez
haydavies_diffuse_clearsky = pv.irradiance.haydavies(pvlib_tracking_DF['surface_tilt'],
                                                     pvlib_tracking_DF['surface_azimuth'],
                                                     clearsky_DF['dhi'],
                                                     clearsky_DF['dni'],
                                                     dni_extra_pvlib,
                                                     pvlib_solpos_30shift_DF['apparent_zenith'],
                                                     pvlib_solpos_30shift_DF['azimuth'])  # type = pd.Series
haydavies_diffuse_clearsky.name = 'H-D'
perez_diffuse_clearsky = pv.irradiance.perez(pvlib_tracking_DF['surface_tilt'],
                                             pvlib_tracking_DF['surface_azimuth'],
                                             clearsky_DF['dhi'],
                                             clearsky_DF['dni'],
                                             dni_extra_pvlib,
                                             pvlib_solpos_30shift_DF['apparent_zenith'],
                                             pvlib_solpos_30shift_DF['azimuth'],
                                             airmass_pvlib)  # type = pd.Series
perez_diffuse_clearsky.name = 'Perez'

# Transform DNI and diffuse into plane of array irradiance.
# POA global = POA beam + POA diffuse
# columns: poa_global, poa_direct, poa_diffuse
pvlib_POA_clearsky = pv.irradiance.globalinplane(pvlib_tracking_DF['aoi'],
                                                 clearsky_DF['dni'],
                                                 haydavies_diffuse_clearsky,
                                                 ground_diffuse_clearsky)  # type = pd.DataFrame
pvlib_POA_clearsky = pd.concat([pvlib_POA_clearsky, haydavies_diffuse_clearsky, perez_diffuse_clearsky,
                                ground_diffuse_clearsky], axis=1)

# *** Now use horizontal irradiance data from PVSyst Meteonorm file: **************************************************
ground_diffuse_meteo = pv.irradiance.grounddiffuse(pvlib_tracking_DF['surface_tilt'],
                                                   pvsyst_DF['GlobHor'],
                                                   albedo=SITE_ALBEDO)  # type = pd.Series
ground_diffuse_meteo.name = 'Ground'
# PVSys 'BeamHor' is NOT the same as DNI (direct normal irrad). DNI = 'BeamHor' / cos(elevation)
haydavies_diffuse_meteo = pv.irradiance.haydavies(pvlib_tracking_DF['surface_tilt'],
                                                  pvlib_tracking_DF['surface_azimuth'],
                                                  pvsyst_DF['DiffHor'],
                                                  pvsyst_DF['BeamHor'] / pv.tools.cosd(pvlib_solpos_30shift_DF['zenith']),
                                                  dni_extra_pvlib,
                                                  pvlib_solpos_30shift_DF['apparent_zenith'],
                                                  pvlib_solpos_30shift_DF['azimuth'])  # type = pd.Series
haydavies_diffuse_meteo.name = 'H-D'
perez_diffuse_meteo = pv.irradiance.perez(pvlib_tracking_DF['surface_tilt'],
                                          pvlib_tracking_DF['surface_azimuth'],
                                          pvsyst_DF['DiffHor'],
                                          pvsyst_DF['BeamHor'] / pv.tools.cosd(pvlib_solpos_30shift_DF['zenith']),
                                          dni_extra_pvlib,
                                          pvlib_solpos_30shift_DF['apparent_zenith'],
                                          pvlib_solpos_30shift_DF['azimuth'],
                                          airmass_pvlib)  # type = pd.Series
perez_diffuse_meteo.name = 'Perez'
pvlib_POA_meteo = pv.irradiance.globalinplane(pvlib_tracking_DF['aoi'],
                                              pvsyst_DF['BeamHor'] / pv.tools.cosd(pvlib_solpos_30shift_DF['zenith']),
                                              haydavies_diffuse_meteo,
                                              ground_diffuse_meteo)  # type = pd.DataFrame
pvlib_POA_meteo = pd.concat([pvlib_POA_meteo, haydavies_diffuse_meteo, perez_diffuse_meteo,
                             ground_diffuse_meteo], axis=1)

# Account for diffraction in PV panel protective glass with incident angle modifier (IAM).
# PVSyst uses ASHRAE model, and default 'b' value of 0.05:
# do not do this yet

# *** DATA CRUNCHING! *************************************************************************************************
# Accumulate irradiances each hour into annual irradiation (kWh/m**2):
pvlib_POA_clear_yearly = pvlib_POA_clearsky.resample('A', how='sum').dropna().squeeze() / 1000.0
pvlib_POA_meteo_yearly = pvlib_POA_meteo.resample('A', how='sum').dropna().squeeze() / 1000.0
pvsyst_POA_yearly = pvsyst_DF[['GlobHor', 'BeamHor', 'DiffHor', 'GlobInc', 'BeamInc', 'DifSInc', 'Alb Inc']].\
                    resample('A', how='sum').dropna().squeeze() / 1000.0

# Accumulate irradiances each hour into monthly irradiation (kWh/m**2):
pvlib_POA_clear_monthly = pvlib_POA_clearsky.resample('M', how='sum', kind='period') / 1000.0
pvlib_POA_meteo_monthly = pvlib_POA_meteo.resample('M', how='sum', kind='period') / 1000.0
pvsyst_POA_monthly = pvsyst_DF[['GlobHor', 'BeamHor', 'DiffHor', 'GlobInc', 'BeamInc', 'DifSInc', 'Alb Inc']].\
                     resample('M', how='sum', kind='period') / 1000.0
# pvsyst_POA_monthly[['BeamInc', 'DifSInc', 'Alb Inc']].plot(kind='bar', stacked=True)

# Accumulate irradiances each hour into daily irradiation (kWh/m**2):
pvlib_POA_clear_daily = pvlib_POA_clearsky.resample('D', how='sum') / 1000.0
pvlib_POA_meteo_daily = pvlib_POA_meteo.resample('D', how='sum') / 1000.0
pvsyst_POA_daily = pvsyst_DF[['GlobHor', 'BeamHor', 'DiffHor', 'GlobInc', 'BeamInc', 'DifSInc', 'Alb Inc']].\
                    resample('D', how='sum') / 1000.0

# *** CALCULATE DIFFERENCE ERRORS BETWEEN PVSYST AND PVLIB AND PLOT ***************************************************
# Template to plot annual error:
def plot_annual_error(error_series, error_string, pvsyst_col, pvlib_col, units='units'):
    plt.figure(figsize=(20,5))
    plt.plot(error_series.index, error_series.values)
    plt.title(error_string + ': PVSyst ' + pvsyst_col + ' - pvlib ' + pvlib_col)
    plt.ylabel(units)

# Template to plot error on solstice + equinox:
def plot_solstice_error(error_series, error_string, units='units'):
    plt.figure()
    plt.xlim(0,24)
    # When plotting multiple trends against the same x-axis, must use index.hour to avoid DST confusion.
    # matplotlib will return 'ValueError: ordinal must be >= 1' because it's trying to force datetime indices with
    # different localized timezones to the same x-axis.
    plt.plot(error_series[jun21].index.hour, error_series[jun21], 'g-',
             error_series[dec21].index.hour, error_series[dec21], 'b-',
             error_series[mar20].index.hour, error_series[mar20], 'r-')
    plt.legend(['June 21', 'Dec 21', 'Mar 20'])
    plt.title(error_string + ' during key days of the year')
    plt.xlabel('Hour of day')
    plt.ylabel(units)

# Template to plot values on solstice + equinox:
def plot_solstice_values(values_string, pvsyst_DF, pvsyst_col, pvlib_DF, pvlib_col, units='units'):
    plt.figure(figsize=(20, 15))
    plt.subplot(311)
    plt.xlim(0,24)
    plt.ylabel(units)
    # When plotting multiple trends against the same x-axis, must use index.hour to avoid DST confusion.
    # matplotlib will return 'ValueError: ordinal must be >= 1' because it's trying to force datetime indices with
    # different localized timezones to the same x-axis.
    plt.plot(pvsyst_DF[jun21].index.hour, pvsyst_DF[pvsyst_col][jun21],'k-')
    plt.plot(pvsyst_DF[jun21].index.hour, pvlib_DF[pvlib_col][jun21], 'g-')
    plt.title(values_string)
    plt.legend(['June 21-pvsyst', 'June 21-pvlib'], loc='lower right')

    plt.subplot(312)
    plt.xlim(0,24)
    plt.ylabel(units)
    plt.plot(pvsyst_DF[mar20].index.hour, pvsyst_DF[pvsyst_col][mar20], 'k-')
    plt.plot(pvsyst_DF[mar20].index.hour, pvlib_DF[pvlib_col][mar20], 'r-')
    plt.legend(['Mar 20-pvsyst', 'Mar 20-pvlib'], loc='lower right')

    plt.subplot(313)
    plt.xlim(0,24)
    plt.ylabel(units)
    plt.plot(pvsyst_DF[dec21].index.hour, pvsyst_DF[pvsyst_col][dec21], 'k-')
    plt.plot(pvsyst_DF[dec21].index.hour, pvlib_DF[pvlib_col][dec21],'b-')
    plt.legend(['Dec 21-pvsyst', 'Dec 21-pvlib'], loc='lower right')
    plt.xlabel('Hour of day')

# Plot sun path diagram:
def plot_sun_path():
    plt.figure()
    plt.plot(pvsyst_DF['AzSol'][jun21], pvsyst_DF['HSol'][jun21], 'k-',
             pvlib_solpos_30shift_DF['azimuth_180'][jun21], pvlib_solpos_30shift_DF['elevation_0'][jun21], 'g-')
    plt.plot(pvsyst_DF['AzSol'][dec21], pvsyst_DF['HSol'][dec21], 'k-',
             pvlib_solpos_30shift_DF['azimuth_180'][dec21], pvlib_solpos_30shift_DF['elevation_0'][dec21], 'b-')
    plt.plot(pvsyst_DF['AzSol'][mar20], pvsyst_DF['HSol'][mar20], 'k-',
             pvlib_solpos_30shift_DF['azimuth_180'][mar20], pvlib_solpos_30shift_DF['elevation_0'][mar20], 'r-')
    plt.legend(['June 21-pvsyst', 'June 21-pvlib', 'Dec 21-pvsyst', 'Dec 21-pvlib', 'Mar 20-pvsyst', 'Mar 20-pvlib'])
    plt.xlabel('Azimuth')
    plt.ylabel('Elevation')
    plt.title('Sun Path Diagram for ' + SITE_NAME)

# To calculate difference error, transform pvlib results to PVSyst sign conventions.
# # Plot Solar Position: Elevation comparisons:
# elevation = 0: pvsyst - all negative elevations set to 0; pvlib - negative elevation ok
pvlib_solpos_30shift_DF['elevation_0'] = [0 if x < 0 else x for x in pvlib_solpos_30shift_DF['elevation']]
elevation_error = pvsyst_DF['HSol'] - pvlib_solpos_30shift_DF['elevation_0']
# plot_annual_error(elevation_error, 'Sun Elevation Error', 'HSol', 'elevation', 'degrees')
# plot_solstice_error(elevation_error, 'Sun Elevation Error')
# plot_solstice_values('Sun Elevation', pvsyst_DF, 'HSol', pvlib_solpos_30shift_DF, 'elevation_0')

# # Plot Solar Position: Azimuth comparisons:
# azimuth = 0:  = True South (pvsyst); = True North (pvlib)
pvlib_solpos_30shift_DF['azimuth_180'] = map(lambda x: (x-180), pvlib_solpos_30shift_DF['azimuth'])
azimuth_error = pvsyst_DF['AzSol'] - pvlib_solpos_30shift_DF['azimuth_180']
azimuth_error_0 = azimuth_error[pvsyst_DF['HSol'] > 0]  # Only considering the azimuth error when sun elevation angle > 0
# plot_annual_error(azimuth_error_0, 'Azimuth Error', 'AzSol', 'azimuth_180', 'degrees')
# plot_solstice_error(azimuth_error_0, 'Azimuth Error')
# plot_solstice_values('Azimuth', pvsyst_DF, 'AzSol', pvlib_solpos_30shift_DF, 'azimuth_180')
#
# # Plot Sun Path Diagram
# plot_sun_path()

# # Plot Tracking Angle Error:
tracking_error = pvsyst_DF['PhiAng'] - pvlib_tracking_DF['tracker_theta']
# plot_annual_error(tracking_error, 'Track Angle Error', 'PhiAng', 'tracker_theta_flip', 'degrees')
# plot_solstice_error(tracking_error, 'Track Angle Error')
# plot_solstice_values('Track Angle', pvsyst_DF, 'PhiAng', pvlib_tracking_DF, 'tracker_theta_flip')

# Plot Angle of Incidence error:
# angle of incidence: = 90 (PVSyst); = NaN (pvlib) when sun is below horizon
aoi_error = pvsyst_DF['AngInc'] - pvlib_tracking_DF['aoi']
# plot_annual_error(aoi_error, 'Angle of Incidence Error', 'AngInc', 'aoi', 'degrees')
# plot_solstice_error(aoi_error, 'Angle of Incidence Error')
# plot_solstice_values('Angle of Incidence', pvsyst_DF, 'AngInc', pvlib_tracking_DF, 'aoi')

# # Plot daily POA global irradiation:
# plt.figure(figsize=(20,5))
# plt.plot(pvlib_POA_meteo_daily.index, pvlib_POA_meteo_daily['poa_global'], 'b')
# plt.plot(pvsyst_POA_daily.index, pvsyst_POA_daily['GlobInc'], 'g')
# plt.title('Daily Irradiation for ' + SITE_NAME)
# plt.ylabel('kWh/m**2')
# plt.legend(['poa_global - pvlib', 'GlobInc - PVSyst'], loc='upper right')
#
# # Plot monthly POA global:
# monthlies = pd.concat([pvlib_POA_meteo_monthly['poa_global'], pvsyst_POA_monthly['GlobInc']], axis=1)
# ax_monthly = monthlies.plot(kind='bar', title='Monthly Irradiation for ' + SITE_NAME)
# ylabel = ax_monthly.set_ylabel('kWh/m**2')

# Plot annual POA irradiation, stacked by component:
irrad_components_pvlib = pvlib_POA_meteo_yearly[['poa_direct', 'H-D', 'Ground']]
irrad_components_pvlib.index = ['Direct', 'Sky Diffuse', 'Ground Diffuse']
irrad_components_pvsyst = pvsyst_POA_yearly[['BeamInc', 'DifSInc', 'Alb Inc']]
irrad_components_pvsyst.index = ['Direct', 'Sky Diffuse', 'Ground Diffuse']
irrad_components = pd.concat([irrad_components_pvsyst, irrad_components_pvlib], axis=1)
irrad_components.columns = ['PVSyst', 'pvlib']
irrad_components.transpose().plot(kind='bar', stacked=True, figsize=(10,10))
plt.title('PVSyst vs. pvlib: ' + '\n' + 'Annual Irradiation Flux by Component for ' + SITE_NAME)
plt.ylabel('kWh/m2')

# Find POA difference errors and plot:
error_POA_global = pvsyst_DF['GlobInc'] - pvlib_POA_meteo['poa_global']
error_POA_direct = pvsyst_DF['BeamInc'] - pvlib_POA_meteo['poa_direct']
error_POA_diff = (pvsyst_DF['DifSInc'] + pvsyst_DF['Alb Inc']) - pvlib_POA_meteo['poa_diffuse']
error_sky_diff = pvsyst_DF['DifSInc'] - pvlib_POA_meteo['H-D']
error_ground_diff = pvsyst_DF['Alb Inc'] - pvlib_POA_meteo['Ground']

error_POA_global_annual = error_POA_global.resample('A', how='sum')  # Wh/m2
# plot_annual_error(error_POA_direct, 'POA Direct Irradiance Error', 'BeamInc', 'poa_direct', 'W/m2')
# # plot_solstice_values('POA Direct Irradiance', pvsyst_DF, 'BeamInc', pvlib_POA_meteo, 'poa_direct', 'W/m2')
error_POA_direct_annual = error_POA_direct.resample('A', how='sum')  # Wh

# plot_annual_error(error_sky_diff, 'POA Sky Diffuse Irradiance Error', 'DifSInc', 'H-D', 'W/m2')
# plot_solstice_values('POA Sky Diffuse Irradiance', pvsyst_DF, 'DifSInc', pvlib_POA_meteo, 'H-D', 'W/m2')
error_sky_diff_annual = error_sky_diff.resample('A', how='sum')  # Wh

# plot_annual_error(error_ground_diff, 'POA Ground Diffuse Irradiance Error', 'Alb Inc', 'Ground', 'W/m2')
# # plot_solstice_values('POA Ground Diffuse Irradiance', pvsyst_DF, 'Alb Inc', pvlib_POA_meteo, 'Ground', 'W/m2')
error_ground_diff_annual = error_ground_diff.resample('A', how='sum')  # Wh

percent_error_dict = {
    'global': float(error_POA_global_annual / pvsyst_POA_yearly['GlobInc'] * (100.0/1000.0)),
    'direct': float(error_POA_direct_annual / pvsyst_POA_yearly['BeamInc'] * (100.0/1000.0)),
    'sky': float(error_sky_diff_annual / pvsyst_POA_yearly['DifSInc'] * (100.0/1000.0)),
    'ground': float(error_ground_diff_annual / pvsyst_POA_yearly['Alb Inc'] * (100.0/1000.0))
}

# # Plot irradiance difference errors by component:
# plt.figure(figsize=(25,5))
# plt.plot(error_POA_direct.index, error_POA_direct.values)
# plt.plot(error_sky_diff.index, error_sky_diff.values)
# plt.plot(error_ground_diff.index, error_ground_diff.values)
# plt.title('Irradiance errors (PVSyst - pvlib) by component')
# plt.ylabel('Difference in irradiance W/m2')
# plt.legend(['Sky Diffuse', 'Direct', 'Ground Diffuse'], loc='lower right')

# *** DOES DIFFUSE TRACKING MAKE SENSE ?? ******************************************************************************
# When global horiz > global incident, it's better to be flat!
global_diff = pvsyst_DF['GlobHor'] - pvsyst_DF['GlobInc']
global_diff_fraction = global_diff / pvsyst_DF['GlobInc']
better_flat = pd.Series([x if x > 0 else 0 for x in global_diff], index=global_diff.index)
better_flat_fraction = pd.Series([x if ((x > 0.0) & (x <= 1.0)) else 0.0 for x in global_diff_fraction],
                                index=global_diff.index)
better_flat_annual = better_flat.resample('A', how='sum') / 1000.0  # kWh/m2
better_flat_annual_fraction = better_flat_annual / pvsyst_POA_yearly['GlobInc']

# Plot difference between global horizontal and global incident irradiance throughout year:
plt.figure(figsize=(25,5))
plt.plot(global_diff.index, global_diff)
plt.title('PVSyst GlobHor minus GlobInc for ' + SITE_NAME)
plt.ylabel('W/m2')

# Plot fraction of addition available irradiance throughout year:
plt.figure(figsize=(25,5))
plt.plot(better_flat_fraction.index, better_flat_fraction.values, 'b.-')
plt.title('Additional Irradiance Available on Flat Panel as Fraction of Incident Irradiance on ' + '\n' +
          'Single-Axis Tracking Panel' + '\n' + ' for ' + SITE_NAME)

# Plot average additional irradiance available for each hour in the day:
better_flat_fraction_trimmed = better_flat_fraction[better_flat_fraction > 0]
plt.figure()
plt.xlim(0,24)
plt.ylim(0,1.0)
plt.plot(better_flat_fraction_trimmed.groupby(better_flat_fraction_trimmed.index.hour).mean(), 'b.-')
plt.title('Average Additional Irradiance Available on Flat Panel ' + '\n' +
          'as Fraction of Incident Irradiance on Tracking Panel'  + '\n' + ' for ' + SITE_NAME)
plt.xlabel('Hour of Day')

# Plot monthly irradiation for global horizontal (flat panel) and global incident (tracking panel):
# pvsyst_POA_monthly[['GlobInc','GlobHor']].plot(kind='bar')

plt.show()
plt.clf()
