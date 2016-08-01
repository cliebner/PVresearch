__author__ = 'yudong'

import pandas as pd
import pytz
import pvlib
import numpy as np

#
# IS_DT = 'Diffuse Tracking?'
# DT_GAIN = 'DiffTrack Gain'
# DT_GAIN_FACTOR = 'DiffTrack Gain Factor'
# DT_ANG_TRACK = 'DiffTrack Tracker Angle'


class AbstractResultReader(object):
    pass

BHI = 'beam horizon irradiance'
DNI = 'Direct_norm_irradiance'
GHI = 'global_horizon_irradiance'
DHI = 'diffuse_horizon_irraidance'
TOTAL_INC = 'total_incidence'
BEAM_INC = 'beam_incidence'
SKY_INC = 'sky_incidence'
GROUND_INC = 'ground_incidence'
SOLAR_ELEVATION = 'solar_elevation'
SOLAR_ZENITH = 'solar_zenith'
SOLAR_AZIMUTH = 'solar_azimuth'

PANEL_TILT = 'panel_tilt_angle'
PANEL_AZIMUTH = 'panel_azimuth_angle'
PANEL_PHI = 'panel_rotation_angle'
INC_ANGLE = 'incident_angle'

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

############### PVSYST naming convention
PVSYST_CONVENTION = {
    BHI: 'BeamHor',
    GHI: 'GlobHor',
    DHI: 'DiffHor',

    INC_ANGLE: 'AngInc',
    TOTAL_INC: 'GlobInc',
    BEAM_INC: 'BeamInc',
    SKY_INC: 'DifSInc',
    GROUND_INC: 'Alb Inc',
    SOLAR_ELEVATION: 'HSol',
    SOLAR_AZIMUTH: 'AzSol',

    PANEL_PHI: 'PhiAng',
    PANEL_AZIMUTH: 'PlAzim',
    PANEL_TILT: 'PlTilt',

    E_GLOBAL: 'GlobEff',
    E_DIRECT: 'BeamEff',
    E_DIFF_S: 'DiffEff',
    E_DIFF_G: 'Alb_Eff',
    S_GLOBAL: 'GlobShd',
    SLOSS_GLOBAL: 'ShdLoss',
    SLOSS_DIRECT: 'ShdBLss',
    SLOSS_DIFFUSE: 'ShdDLss',
    SF_GLOBAL: 'FShdGl',
    SF_DIRECT: 'FShdBm',
    SF_DIFF_S: 'FShdDif',
    SF_DIFF_G: 'FShdAlb',
}

class PVsystResultReader(AbstractResultReader):

    def __init__(self, sim_res_file, solar_site=None):
        self.file = sim_res_file
        self.data_frame = pd.read_csv(self.file, sep=';', header=0, skiprows=[0,1,2,3,4,5,6,7,8,9,11],
                        index_col=0, parse_dates=True)
        # ### TODO the date time of PVSyst cannot handle daylight saving modes #TODO
        #
        # # PVSyst sun position values are calculated on the 0:30 of each hour but associated with the 0:00 timestamp, so we need
        # # to compare PVSyst values to the pvlib values calculated at the 0:30.
        # tzinfo = solar_site.time_zone
        # tz = pytz.timezone(tzinfo)
        # start = tz.localize(self.data_frame.index[0])
        # end = tz.localize(self.data_frame.index[-1])
        # start_next  = tz.localize(self.data_frame.index[1])
        # freq = start_next-start
        # start_offset = start + pd.DateOffset(minutes=30)
        # end_offset = end + pd.DateOffset(minutes=30)
        # self.data_frame.index = pd.date_range(start=start_offset, end=end_offset, freq=freq)

    def get_solar_elevation(self, tvec):
        return self._get_series_with_name(tvec, SOLAR_ELEVATION)

    def get_solar_zenith(self, tvec):
        return 90-self.get_solar_elevation(tvec)

    def get_solar_azimuth(self, tvec):
        return self._get_series_with_name(tvec, SOLAR_AZIMUTH)+180

    def get_bhi(self, tvec):
        return self._get_series_with_name(tvec, BHI)

    def get_dni(self, tvec):
        beam_hor = self.get_bhi(tvec)
        elevation = self.get_solar_elevation(tvec)
        dni =  beam_hor/pvlib.tools.sind(elevation)
        index = np.logical_or(np.isnan(dni.values) , elevation.values<=7)
        dni[index] = 0
        return dni

    def get_dhi(self, tvec):
        return self._get_series_with_name(tvec, DHI)

    def get_ghi(self, tvec):
        return self._get_series_with_name(tvec, GHI)

    def get_beam_incidence(self, tvec):
        return self._get_series_with_name(tvec, BEAM_INC)

    def get_sky_incidence(self, tvec):
        return self._get_series_with_name(tvec, SKY_INC)

    def get_ground_incidence(self, tvec):
        return self._get_series_with_name(tvec, GROUND_INC)

    def get_total_incidence(self, tvec):
        return self._get_series_with_name(tvec, TOTAL_INC)

    def get_incident_angle(self, tvec):
        return self._get_series_with_name(tvec, INC_ANGLE)

    def get_panel_tilt(self, tvec):
        return self._get_series_with_name(tvec, PANEL_TILT)

    def get_panel_azimuth(self, tvec):
        return self._get_series_with_name(tvec, PANEL_AZIMUTH)+180
        # return self._get_series_with_name(tvec, PANEL_AZIMUTH)

    def get_panel_phi(self, tvec):
        return self._get_series_with_name(tvec, PANEL_PHI)

    def get_effective_total_incidence(self, tvec):
        return self._get_series_with_name(tvec, E_GLOBAL)

    def get_effective_beam_incidence(self, tvec):
        return self._get_series_with_name(tvec, E_DIRECT)

    def get_effective_sky_incidence(self, tvec):
        return self._get_series_with_name(tvec, E_DIFF_S)

    def get_effective_ground_incidence(self, tvec):
        return self._get_series_with_name(tvec, E_DIFF_G)

    def get_total_shading_factor(self, tvec):
        return self._get_series_with_name(tvec, SF_GLOBAL)

    def get_beam_shading_factor(self, tvec):
        return self._get_series_with_name(tvec, SF_DIRECT)

    def get_sky_shading_factor(self, tvec):
        return self._get_series_with_name(tvec, SF_DIFF_S)

    def get_ground_shading_factor(self, tvec):
        return self._get_series_with_name(tvec, SF_DIFF_G)

    def get_total_with_global_shading(self, tvec):
        return self._get_series_with_name(tvec, S_GLOBAL)

    def get_total_shading_loss(self, tvec):
        return self._get_series_with_name(tvec, SLOSS_GLOBAL)

    def get_beam_shading_loss(self, tvec):
        return self._get_series_with_name(tvec, SLOSS_DIRECT)

    def get_diffuse_shading_loss(self, tvec):
        return self._get_series_with_name(tvec, SLOSS_DIFFUSE)

    def _get_series_with_name(self, tvec, name_string):
        var_name = PVSYST_CONVENTION.get(name_string, None)
        if not var_name:
            return None
        series1 = self.data_frame[var_name]
        return self._interp_series(series1, tvec)

    def _interp_series(self, series, tvec, method='linear'):
        dict1 = series.to_dict()
        dict2 = pd.Series(index=tvec, data=len(tvec)*[np.nan, ]).to_dict()
        merged_dict = dict(dict2, **dict1)
        merged_series_raw = pd.Series(merged_dict)
        merged_series = merged_series_raw.interpolate(method=method)
        return merged_series[tvec]

