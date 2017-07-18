__author__ = 'christina'

import matplotlib.pyplot as plt
import nrel_solar as ns
import weather_tools

'''
Use this module as the quoting and marketing tool

inputs:
* lat / long (or range for wide band study)
* GCR (if range, same GCR held for entire range

outputs:
* annual R2R scaling factor
* annual average DHI/GHI


Steps:
1. DONE  get design data from user -- lat/long, GCR
2. DONE  query NREL db for best weather file
3. WON'T DO  confirm site selection with user
4.  (if no weather file, load PVSyst result file?)
5. conform whatever weather file with weather_tools module. include shifting time by 30 mins for for PVsyst files
6. DONE - get backtracking hours from pvlib
7. DONE - characterize weather during backtracking hours
8. characterize weather during all daylight hours
9. DONE - report annual summary
10. make interesting plots
'''

# GLOBALS
SCALE_MAP_R2R = {
    'bins': [(0.0, 0.3), (0.3, 0.7), (0.7, 1.0)],
    'values': [1.0, 0.5, 0.0],
}
IS_BTRK = 'is backtracking'


def get_scaling_R2R(weather, gcr):

    # create pvlib object:
    pv_lib_tracker = weather_tools.make_pvlib_object_from_weather(weather, gcr)

    # determine backtracking hours:
    pv_lib_tracker[IS_BTRK] = weather_tools.get_backtracking_hrs(pv_lib_tracker.loc[:, 'tracker_theta'])

    # calc DFI:
    dfi_series = weather_tools.calculate_DFI(ghi=weather.ts_data.loc[:, ns.GHI], dhi=weather.ts_data.loc[:, ns.DHI])

    # calc scaling factors:
    hourly_scaling = weather_tools.calculate_scaling_factor(SCALE_MAP_R2R, dfi_series[pv_lib_tracker[IS_BTRK] == True])
    annual_scaling = hourly_scaling.resample('A').mean()

    return annual_scaling, hourly_scaling

user_lat = input('Enter latitude: ')
user_lon = input('Enter longitude: ')
user_gcr = input('Enter site GCR: ')
# user_lat = -32.3182
# user_lon = -86.9023
# user_gcr = 0.45


# Find appropriate NSRDB weather station
closest_site = ns.NrelData.get_data_by_query(user_lat, user_lon, export=False, which='best')

# calculate distance away:
closest_site.calculate_distance()

# results:
if closest_site.data is None:
    pass
else:
    annual, hourly = get_scaling_R2R(closest_site, user_gcr)
    print annual

# Iterate over a range:
# Get list of sites from a random selection of available data:
# closest_site_range = ns.NrelData.get_data_range_by_query(user_lat, user_lon,  include_tmy=False, how='random', size=6)
# annual_range = [get_scaling_R2R(s, user_gcr)[0] for s in closest_site_range]
# print annual_range
# plt.plot(annual_range, 'bo')
# plt.show()







