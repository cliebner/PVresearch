__author__ = 'christina'

# import matplotlib  # version 1.5.1
# matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # version 0.18
import datetime as dt
import time as time


class siteData(object):
    def __init__(self, filename, skiprows_f):
        self.filename = filename
        self.skiprows_f = skiprows_f

        with open(self.filename, 'r', 0) as f:
            file = f.read().splitlines()
        lastline = len(file) - 1
        print 'last line: ' + str(lastline)

        # start = time.clock()
        # self.df = pd.read_csv(self.filename, sep=',', header=0, skiprows=skiprows_f, skipfooter=1, index_col=0,
        #                       parse_dates=True, dayfirst=False, na_values=' NaN', usecols=[0,1,2,3,5,7,9], engine='python')
        # end = time.clock()
        # print 'python engine: ' + str(end - start)

        self.skiprows_f += [lastline]
        start = time.clock()
        self.df = pd.read_csv(self.filename, sep=',', header=0, skiprows=skiprows_f, index_col=0,
                              parse_dates=True, dayfirst=False, na_values=' NaN', usecols=[0, 1, 2, 3, 5, 7, 15], engine='c')
        end = time.clock()
        print 'c engine: ' + str(end - start)

        self.df_clean = self.cleanup(yr=2016, mode='mid')
        self.data_health_raw = self.get_data_health(which='raw', when='all')
        self.data_health_clean = self.get_data_health(which='clean', when='all')

    def cleanup(self, yr=2016, mode='mean'):

        # limit data set to 1 yr for pvsyst
        df_trim = self.df[self.df.index.year == yr]

        # select sunposition at minute 30 to associate with each hour (to match pvyst method)
        df_sunposition = df_trim[df_trim.index.minute == 1][['el','az']]

        # get average of minutely results to associate with each hour for pvsyst
        if mode is 'mean':
            df_irrad = df_trim[['ghi', 'dhi', 'dni', 'airtemp']].resample('H').mean()
            df_sunposition.index = df_irrad.index
        else:
            df_irrad = df_trim[df_trim.index.minute == 1][['ghi', 'dhi', 'dni', 'airtemp']]

        # flag non-NaN data as "valid": if value is NaN, flag = False
        valid_flags = ~(df_irrad['ghi'].isnull() | df_irrad['dhi'].isnull())
        valid_flags.name = 'Valid flag'

        # collate
        df_merged = pd.concat([df_sunposition, df_irrad, valid_flags], axis=1)
        df_merged.index = df_trim.resample('H').mean().index

        # export
        df_merged.to_csv(self.filename.partition('.')[0] + '_hourly_' + mode)

        return df_merged

    def get_data_health(self, which='clean', when='day'):

        which_df = {
            'clean': self.df_clean,
            'raw': self.df
        }

        when_df = {
            'all': which_df[which],
            'day': which_df[which][which_df[which]['el'] > 0.0]
        }

        counts = when_df[when].count()
        return {counts.index[i]: (1.0 - counts[i]/float(len(when_df[when]))) for i in range(0, len(counts))}