__author__ = 'christina'

# import matplotlib  # version 1.5.1
# matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # version 0.18
import datetime as dt
import time as time


# read list of all TMY3 sites

# create a class which includes an object with TMY3 data for a single site
# and a methods to operate on a single weather file

# create a child class that inherits from the single class and can operate on many objects at once

# feed a list of some number of files to the batch class and perform some operations

# global variables
TMY_DATE = 'Date (MM/DD/YYYY)'
TMY_TIME = 'Time (HH:MM)'
FLAT_YEAR = 1990
FLAT_DATETIME = 'Flat DateTime'
# by NREL definitions, GHI, DNI, DHI are total radiation received for the hour ending in the TMY timestamp
GHI = 'GHI (W/m^2)'  # global horizontal
GHI_u = 'GHI uncert (%)'
DNI = 'DNI (W/m^2)'  # direct normal
DNI_u = 'DNI uncert (%)'
DHI = 'DHI (W/m^2)'  # diffuse horizontal
DHI_u = 'DHI uncert (%)'
DHI_r = 'DHI random gen'
DB = 'Dry-bulb (C)'
DP = 'Dew-point (C)'
DIFF_RATIO = 'DHI/GHI'
FILENAME_SUFFIX = 'TYA.CSV'
STATION_LIST_FILENAME = 'list of TMY3 stations.csv'
ID = 'Station ID'
U_CLASS = 'Class'
CITY = 'City'
STATE = 'State'
LOCATION = 'City, State'
LAT = 'Latitude'
LONG = 'Longitude'
TZ = 'Timezone'
ELEV = 'Elevation (m)'


def get_stations_from_file(filename):
    df = pd.read_csv(filename, sep=',', header=None, names=[ID, LOCATION, U_CLASS])
    df[U_CLASS] = [c.partition(U_CLASS+' ')[2] for c in df[U_CLASS]]
    # df.set_index(ID, inplace=True)  # why use station ID as indexer?
    return df

STATIONS_DF = get_stations_from_file(STATION_LIST_FILENAME)


def query_by_location(location, df=STATIONS_DF):
    return df[[location in s for s in df[LOCATION]]]

def query_by_class(u_class, df=STATIONS_DF):
    return df[df[U_CLASS] == u_class]

def query_by_station_id(station_id, df=STATIONS_DF):
    # can take int or string
    if type(station_id) == float or type(station_id) == int:
        return df[df[ID] == station_id]
    else:
        return wildcard_search(df, station_id)

def wildcard_search(df, search_term, wildcard='*'):
    wildcard_dict = {n: 10**n-1 for n in range(1,5)}

    if type(search_term) is not str:
        print 'Station id must be float, int, or str'
        return None
    else:
        wildcard_count = search_term.count(wildcard)
        search_lim = wildcard_dict[wildcard_count]
        search_term_root = int(search_term.rstrip(wildcard))*10**wildcard_count
        return df[(df[ID] >= search_term_root) & (df[ID] <= search_term_root + search_lim)]

def query_by_latlong(df, lat, long, radius=0.5):
    return df[(df[LAT] <= (float(lat)+float(radius))) & (df[LAT] >= (float(lat)-float(radius))) &
              (df[LONG] <= (float(long)+float(radius))) & (df[LONG] >= (float(long)-float(radius)))]

def save_df_to_file(df, filename, usecols=None):
    if usecols is None:
        usecols = df.columns
    df.to_csv(filename+'.csv', sep=',', columns=usecols, index_label='DateTimeIndex')

def read_site_info(filename):
    return pd.read_csv(filename, sep=',')


class tmy3Data(object):
    def __init__(self, filename):
        self.filename = filename
        self.separator = ','

        # read the first line to get the site ID info:
        self.station_id = None
        self.city = None
        self.state = None
        self.location = None
        self.timezone = None
        self.latitude = None
        self.longitude = None
        self.elevation = None
        self.get_site_info()
        self.uncert_class = STATIONS_DF.get_value(self.station_id, U_CLASS)

        self.tmy_df = pd.read_csv(self.filename, sep=self.separator, skiprows=[0],
                                  parse_dates=False, usecols=[0,1,4,6,7,9,10,12,31,34])

        # replace TMY3 hours (01:00 to 24:00) with pythonic hours (00:00 to 23:00):
        new_hours_str = 365 * (['0' + str(p) + ':00' for p in range(0, 10)] + [str(q) + ':00' for q in range(10, 24)])
        new_datetime = [dt.datetime.strptime((self.tmy_df[TMY_DATE][i] + ' ' + new_hours_str[i]), '%m/%d/%Y %H:%M')
                            for i in range(len(new_hours_str))]
        self.tmy_df.index = new_datetime
        self.tmy_df[FLAT_DATETIME] = [t.replace(year=FLAT_YEAR) for t in new_datetime]
        self.get_diffuse_ratio()
        print "loaded: " + self.location

    def get_site_info(self):
        f = open(self.filename, 'r')
        info_list = f.readline().strip().split(',')
        f.close()
        self.station_id = int(info_list[0])
        self.city = info_list[1].strip('"')
        self.state = info_list[2]
        self.location = self.city + ', ' + self.state
        self.timezone = float(info_list[3])
        self.latitude = float(info_list[4])
        self.longitude = float(info_list[5])
        self.elevation = float(info_list[6])

    def get_diffuse_ratio(self):
        daytime_df = self.tmy_df[self.tmy_df[GHI] > 1.0]
        h_diffuse_ratio = daytime_df.loc[:, DHI] / daytime_df.loc[:, GHI]

        self.tmy_df = pd.concat([self.tmy_df,
                                 pd.DataFrame({
                                        DIFF_RATIO: h_diffuse_ratio,
                                    })], axis=1)

    def make_fake_DHI(self):
            self.tmy_df[DHI_r] = [np.random.uniform(0.1, 1.0) * g for g in self.tmy_df[GHI]]

    def export_for_pvsyst(self, filename):
        self.make_fake_DHI()
        self.tmy_df.to_csv(filename + '.csv', sep=',', columns=[FLAT_DATETIME, GHI, DNI, DHI_r, DB, DP])

    def count_over_threshold(self, factor, threshold):
        over_threshold_df = self.tmy_df[self.tmy_df[factor] >= threshold]
        return len(over_threshold_df)

    def count_under_threshold(self, factor, threshold):
        under_threshold_df = self.tmy_df[self.tmy_df[factor] <= threshold]
        return len(under_threshold_df)

    def get_diffuse_ratio_histo(self):
        bins = list(np.arange(0.0,1.1,0.1))
        diff_bins = pd.cut(self.tmy_df[DIFF_RATIO], bins, right=False, include_lowest=True)
        return pd.value_counts(diff_bins, sort=True, ascending=True, normalize=False, dropna=True)

    def plot_site_diffuseness(self):
        self.tmy_df[DIFF_RATIO].plot.hist(bins=10)
        plt.title(self.city + ', ' + self.state, fontsize=16)
        plt.xlabel(DIFF_RATIO)
        plt.ylim([0, 1000])


class tmy3Batch(tmy3Data):
    def __init__(self, stations_list, directory=None):
        self.stations_list = stations_list  # list of TMY3 stations to run in batch
        self.batch_dict = {}
        self.directory = directory
        self.batch_site_info_df = pd.concat([(STATIONS_DF[STATIONS_DF.index.isin(stations_list)]),
                                            pd.DataFrame(columns=[CITY, STATE, TZ, LAT, LONG, ELEV, DIFF_RATIO])])
        self.load_data()

    def load_data(self):
        if self.directory is None:
            prefix = ''
        else:
            prefix = self.directory + '/'
        for s in self.stations_list:
            data = tmy3Data(prefix + str(s) + FILENAME_SUFFIX)
            self.batch_dict[data.location] = data  # tmy3Data object
            self.batch_site_info_df.loc[data.station_id, CITY] = data.city
            self.batch_site_info_df.loc[data.station_id, STATE] = data.state
            self.batch_site_info_df.loc[data.station_id, TZ] = data.timezone
            self.batch_site_info_df.loc[data.station_id, LAT] = data.latitude
            self.batch_site_info_df.loc[data.station_id, LONG] = data.longitude
            self.batch_site_info_df.loc[data.station_id, ELEV] = data.elevation

    def compare_diffuse_ratios(self):
        self.batch_site_info_df.loc[:, DIFF_RATIO] = [self.batch_dict[i].tmy_df[DIFF_RATIO].mean() for i in self.batch_site_info_df.index]



    def order_stations_by_factor(self, factor, sort_ascending=False):
        mean_df = pd.DataFrame.from_dict({b[1].location: b[1].tmy_df[factor].mean() for b in self.batch_dict.iteritems()},
                                         orient='index')
        mean_df.columns = ['mean']
        mean_df.sort_values('mean', axis=0, inplace=True, ascending=sort_ascending)
        return mean_df



# ordered_batch = batch.order_stations_by_factor(DIFF_RATIO, sort_ascending=False)
