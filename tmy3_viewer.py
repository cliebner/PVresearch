__author__ = 'christina'

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pvlib as pv

filename = '690150TYA.CSV'

(tmyDF, tmyDict) = pv.tmy.readtmy3(filename)
location_name = tmyDict['Name']
latitude = float(tmyDict['latitude'])
longitude = float(tmyDict['longitude'])
timezone = float(tmyDict['TZ'])
altitude = float(tmyDict['altitude'])

# Remove 'Source' and 'Uncertainty' columns from dataframe
tmyDF_thin = tmyDF.copy()[['ETR', 'ETRN', 'GHI', 'DNI', 'DHI','TotCld','OpqCld', 'DryBulb', 'DewPoint', 'RHum',
                         'Pressure', 'Wdir', 'Wspd', 'Hvis', 'CeilHgt', 'Pwat', 'AOD', 'Alb', 'Lprecipdepth',
                         'PresWth']]
tmyDF_thin['Hour'] = tmyDF_thin.index.hour

# Consider only hours when sun is over horizon (extraterrestrial horiz irradiance > 0)
tmyDF_daytime = tmyDF_thin[tmyDF_thin['ETR'] > 0]

# Spring equinox: month = 3, day = 22
spring = tmyDF_thin[tmyDF_thin.index.month == 3]
spring_eq = spring[spring.index.day == 22]
plt.plot(spring_eq.Hour, spring_eq.ETR, 'g:')
# plt.plot(spring_eq['Hour'], spring_eq['ETR'], 'b')
# Summer solstice: month = 6, day = 21
summer = tmyDF_thin[tmyDF_thin.index.month == 6]
summer_sol = summer[summer.index.day == 21]
plt.plot(summer_sol.Hour, summer_sol.ETR,'y-')
# Fall equinox: month = 9, day = 22
fall = tmyDF_thin[tmyDF_thin.index.month == 9]
fall_eq = fall[fall.index.day == 22]
plt.plot(fall_eq.Hour, fall_eq.ETR, 'r:')
# Winter solstice: month = 12, day = 21
winter = tmyDF_thin[tmyDF_thin.index.month == 12]
winter_sol = winter[winter.index.day == 21]
plt.plot(winter_sol.Hour, winter_sol.ETR,'b-')
plt.show()
plt.clf()
