__author__ = 'christina'

import json
import requests
import datetime as dt
import pandas as pd
import numpy as np
import weather_tools as wt

NREL_URL = 'https://developer.nrel.gov'
NREL_API_KEY = '4oZFFfj6MOX6WeEz2k4AYMpVrK9hNVWGFJ5VRXRv'
MY_NAME = 'Christina+Liebner'
USE_REASON = 'research'
AFFILIATION = 'nextracker'
EMAIL = 'cliebner@nextracker.com'
JSON = '.json?'
CSV = '.csv?'
R_EQ = 7926.0/2  # earth radius at equator in statute miles

# NSRDB naming conventions:
AVAIL_INT = 'interval'
LINK = 'link'
AVAIL_YR = 'year'
NAME = 'name'
DISP_NAME = 'displayName'
LINKS = 'links'
MDATA = 'metadataLink'
API_DOCS = 'apiDocs'
MEASUREMENT = 'type'

NSRDB = 'NSRDB'  # typ returns timeseries data in .csv
SOLAR = 'Solar Resource'  # typ returns text or summary data in .json

# DATASET SHORT NAMES (matches 'name' attribute from NSRDB query)
NSRDB_MTS1 = 'mts1'
NSRDB_MTS1_TMY = 'mts1-tmy'
NSRDB_MTS2 = 'mts2'
NSRDB_MTS2_TMY = 'mts2-tmy'
NSRDB_MTS3 = 'mts3'
NSRDB_MTS3_TMY = 'mts3-tmy'
NSRDB_QUERY = 'NSRDB Data Query'
NSRDB_PSM = 'psm'
NSRDB_INTL = 'suny-international'
NSRDB_COUNT = 'Site Count'
NSRDB_SPECTRAL = 'spectral-tmy'
SOLAR_RESOURCE = 'Solar Resource Data'
SOLAR_QUERY = 'Solar Dataset Query'

# DATASET_HIERARCHY = [NSRDB_PSM, NSRDB_MTS3, NSRDB_MTS3_TMY, NSRDB_MTS2, NSRDB_MTS2_TMY, NSRDB_INTL]
DATASET_HIERARCHY = [NSRDB_PSM, NSRDB_INTL]
#MTS1 data has a UTC offset bug that is unresolved by NREL as of 7/17/2017. so, don't let users return any MTS1 datasets

API_LINK = {
    NSRDB_MTS1: '/api/solar/nsrdb_6190_download',
    NSRDB_MTS2: '/api/solar/nsrdb_9105_download',
    NSRDB_QUERY: '/api/solar/nsrdb_data_query',
    NSRDB_PSM: '/api/solar/nsrdb_0512_download',
    NSRDB_INTL: '/api/solar/suny_india_download',
    NSRDB_COUNT: '/api/solar/nsrdb/site_count',
    NSRDB_SPECTRAL: '/api/solar/nsrdb_download',
    SOLAR_RESOURCE: '/api/solar/solar_resource/v1',
    SOLAR_QUERY: '/api/solar/data_query/v1',
}
DATASET_RESULT_FORMAT = {
    NSRDB_MTS1: CSV,
    NSRDB_MTS2: CSV,
    NSRDB_QUERY: JSON,
    NSRDB_PSM: CSV,
    NSRDB_INTL: CSV,
    NSRDB_COUNT: JSON,
    NSRDB_SPECTRAL: CSV,
    SOLAR_RESOURCE: JSON,
    SOLAR_QUERY: JSON,
}
DISPLAY_NAME_KEY = {
    NSRDB_MTS1: 'Meteorological Statistical Model 1',
    NSRDB_MTS2: 'Meteorological Statistical Model 2',
    NSRDB_QUERY: 'NSRB_DQ',
    NSRDB_PSM: 'Physical Solar Model',
    NSRDB_INTL: 'SUNY International',
    NSRDB_COUNT: 'Site count',
    NSRDB_SPECTRAL: 'Spectral',
    SOLAR_RESOURCE: 'Solar Resource',
    SOLAR_QUERY: 'Dataset Query'
}
AVAILABLE_ATTRIBUTES = {
    NSRDB_MTS1: ['dhi', 'dni', 'ghi', 'etr', 'etrn', 'tot_sky',
           'op_sky', 'dew_point', 'temp_dryb', 'atm_pres', 'rel_hum',
           'wind_dir', 'wind_spd', 'vis', 'ceil_ht', 'prec_wat',
           'aer_opt_dpth', 'tot_sky_source', 'op_sky_source', 'dew_point_source', 'temp_dryb_source',
           'atm_pres_source', 'rel_hum_source', 'wind_dir_source', 'wind_spd_source', 'vis_source',
           'ceil_ht_source', 'prec_wat_source', 'aer_opt_dpth_source', 'dhi_uncertainty', 'dni_uncertainty',
                 'ghi_uncertainty', 'tot_sky_uncertainty', 'op_sky_uncertainty', 'dew_point_uncertainty',
                 'temp_dryb_uncertainty',
                 'atm_pres_uncertainty', 'rel_hum_uncertainty', 'wind_dir_uncertainty', 'wind_spd_uncertainty',
                 'vis_uncertainty', 'ceil_ht_uncertainty', 'prec_wat_uncertainty', 'aer_opt_dpth_uncertainty'],
    NSRDB_MTS2: ['dhi', 'dni', 'ghi', 'etr', 'etrn',
           'dew_point', 'temp_dryb', 'atm_pres', 'rel_hum',
           'wind_spd', 'precip_wat', 'aer_opt_dpth'],
    NSRDB_QUERY: '',
    NSRDB_PSM: ['dhi', 'dni', 'ghi', 'clearsky_dhi', 'clearsky_dni',
          'clearsky_ghi', 'cloud_type', 'dew_point', 'surface_air_temperature_nwp',
          'surface_pressure_background', 'surface_relative_humidity_nwp', 'solar_zenith_angle',
          'total_precipitable_water_nwp', 'wind_direction_10m_nwp', 'wind_speed_10m_nwp', 'fill_flag'],
    NSRDB_INTL: ['dhi', 'dni', 'ghi', 'clearsky_dhi', 'clearsky_dni',
                 'clearsky_ghi', 'cloud_type', 'dew_point', 'surface_air_temperature',
                 'surface_pressure', 'relative_humidity',
                 'solar_zenith_angle',
                 'total_precipitable_water', 'snow_depth', 'wdir', 'wspd',
                 'fill_flag'],
    NSRDB_COUNT: '',
    NSRDB_SPECTRAL: '',
    SOLAR_RESOURCE: '',
    SOLAR_QUERY: '',
}
AVAILABLE_YEARS = {
    NSRDB_MTS1: [str(y) for y in range(1961, 1991)] + ['tmy'],
    NSRDB_MTS2: [str(y) for y in range(1991, 2006)] + ['tmy'],
    NSRDB_QUERY: '',
    NSRDB_PSM: [str(y) for y in range(1998, 2016)] + ['tmy'],
    NSRDB_INTL: [str(y) for y in range(2000, 2015)] + ['tmy'],
    NSRDB_COUNT: '',
    NSRDB_SPECTRAL: '',
    SOLAR_RESOURCE: None,
    SOLAR_QUERY: None,
}

MONTHS = {
    'jan': dt.datetime(2000, 1, 15),
    'feb': dt.datetime(2000, 2, 15),
    'mar': dt.datetime(2000, 3, 15),
    'apr': dt.datetime(2000, 4, 15),
    'may': dt.datetime(2000, 5, 15),
    'jun': dt.datetime(2000, 6, 15),
    'jul': dt.datetime(2000, 7, 15),
    'aug': dt.datetime(2000, 8, 15),
    'sep': dt.datetime(2000, 9, 15),
    'oct': dt.datetime(2000, 10, 15),
    'nov': dt.datetime(2000, 11, 15),
    'dec': dt.datetime(2000, 12, 15),
}

STATUS_CODES = {
    200: '200: request completed',
    400: '400: HTTPS required',
    403: '403: API key error',
    404: '404: not found',
    429: '429: overrate limit',
    504: '504: timeout error',
}
# Common hourly weather data column names:
GHI = 'GHI'
DHI = 'DHI'
DNI = 'DNI'

class NrelData(object):

    def __init__(self):
        self.dataset = None
        self.result_format = None

        self.url = None
        self.status_code = None

        self.query_lat = None
        self.query_lon = None
        self.query_results = None

        self.actual_lat = None
        self.actual_lon = None
        self.info = None  # reserved for summary info about a site
        self.data = None  # reserved for more detailed info about a site
        self.ts_data = None  # reserved for timeseries data from a site
        self.data_id = None
        self.location_id = None

        self.city = None
        self.state = None
        self.country = None
        self.timezone = None
        self.year = None

        self.export_filename = None

    def check_location(self):
        if self.query_lat is None:
            self.query_lat = float(input('Enter latitude (float) = '))
        if self.query_lon is None:
            self.query_lon = float(input('Enter longitude (float) = '))

    def check_url(self):
        boilerplate = 'api_key=yourapikey&email=youremail'
        if self.url is None:
            print 'no url'  # throw an exception?
        else:
            if boilerplate in self.url:
                url_to_keep = self.url.partition(boilerplate)[0]
                correction = 'api_key={api_key}&email={email}'.format(api_key=NREL_API_KEY, email=EMAIL)
                self.url = url_to_keep + correction

    def set_url(self, url):
        self.url = url
        self.check_url()

    def make_request(self):

        num_tries = 0
        max_tries = 3

        while num_tries < max_tries and self.status_code != 200:
            print 'attempt #{num}: '.format(num=num_tries+1)
            r = requests.get(self.url)
            self.status_code = r.status_code
            print STATUS_CODES[self.status_code]
            num_tries += 1
        return r

    def calculate_distance(self):
        if self.actual_lat is not None and self.actual_lon is not None:
            # assumes earth is sphere
            one_lat = 2 * np.pi * R_EQ / 360.0
            one_lon = one_lat * np.cos(abs(self.actual_lat) * np.pi/180.0)
            del_x = (self.actual_lon - self.query_lon) * one_lon
            del_y = (self.actual_lat - self.query_lat) * one_lat
            distance = (del_x**2 + del_y**2)**0.5
            if self.city == '-':
                print 'Your closest site is {distance} miles away at: {lat} deg latitude, {lon} deg longitude'.format(
                    distance=str(round(distance, 2)),
                    lat=self.actual_lat,
                    lon=self.actual_lon)
            else:
                print 'Your closest site is {distance} miles away at: {city}, {state}, {country}'.format(
                    distance=str(round(distance, 2)),
                    city=self.city,
                    state=self.state,
                    country=self.country)
        else:
            distance = None
        return distance

    def export_timeseries(self):
        self.export_filename = '{dataset}_{loc_id}_{year}.csv'.format(dataset=self.dataset,
                                                                      loc_id=self.location_id,
                                                                      year=self.year)
        if len(self.ts_data) > 0:
            print 'exporting to {filename} . . .'.format(filename=self.export_filename)
            self.ts_data.to_csv(self.export_filename)
        else:
            print 'no data to export'

    @classmethod
    def get_data_from_url(cls, url):
        #TODO: this is not completed as written
        cls.set_url(url)

    @classmethod
    def get_data_directly(cls, lat, lon, year, dataset):
        # TODO: make sure there's an elegant way to catch when no data is returned from the url request (no site is found)
        dataset_obj = {
            NSRDB_MTS1: NsrdbMts1(lat, lon, year),
            NSRDB_MTS1_TMY: NsrdbMts1(lat, lon, year),
            NSRDB_MTS2: NsrdbMts2(lat, lon, year),
            NSRDB_MTS2_TMY: NsrdbMts2(lat, lon, year),
            NSRDB_PSM: NsrdbPsm(lat, lon, year),
            NSRDB_INTL: NsrdbInternational(lat, lon, year),
            NSRDB_SPECTRAL: NsrdbSpectral(lat, lon, year),
        }[dataset]
        dataset_obj.get_data()
        print 'done!'
        return dataset_obj

    @classmethod
    def get_query(cls, lat, lon):
        query_obj = NsrdbQuery(lat, lon)
        query_obj.get_data()
        query_obj.prettify_data()

        # check if query returned any data:
        if query_obj.data is None:
            print 'No sites found'
        return query_obj

    @classmethod
    def get_data_by_query(cls, lat, lon, which='best', export=False):

        query_obj = cls.get_query(lat, lon)

        # check if query returned any data:
        if query_obj.data is None:
            data_obj = NrelData()
        else:
            if which == 'best':
                selected_year = query_obj.get_most_recent_data()
                # TODO: if there are multiple datasets available, which one to choose?
                # for now, just define a hierarchical order and pick the highest ranking one:
                avail_sets = list(set(query_obj.data[query_obj.data.loc[:, AVAIL_YR] == selected_year].loc[:, NAME]))
                best_ix = [a in avail_sets for a in DATASET_HIERARCHY].index(True)
                dataset = DATASET_HIERARCHY[best_ix]

                # there are 2 ways to get data specifically:
                # 1) select link url from NsrdbQuery dataframe;
                # data_obj = NrelData.get_data_from_url(str(query_obj.data.get_value((dataset, year), 'link')))
                # 2) build up data directly with lat, lon, year --> go with this method for now, since we have the methods
                data_obj = NrelData.get_data_directly(lat, lon, selected_year, dataset)

                if export is True:
                    data_obj.export_timeseries()
                print 'done!'

            else:  # what other ways might we want to get data?
                pass

        return data_obj

    @classmethod
    def get_data_range_by_query(cls, lat, lon, how='random', include_tmy=False, size=1):

        # TODO: this appears to fail with mts2 included. hmm...
        query_obj = cls.get_query(lat, lon)

        # check if query returned any data:
        if query_obj.data is None:
            data_objs = NrelData()
        else:

            index_trimmed1 = query_obj.data.index[query_obj.data.index.isin(DATASET_HIERARCHY, level=0)]

            if include_tmy is True:
                index_trimmed2 = index_trimmed1.copy()
            else:
                index_trimmed2 = index_trimmed1.drop(['tmy', 'tmy2', 'tmy3'], level=1, errors='ignore')

            if how == 'random':
                size = min(size, len(index_trimmed2))
                selected_index = [index_trimmed2[r] for r in np.random.choice(len(index_trimmed2),
                                                                              size=size, replace=False)]

            elif how == 'all':
                selected_index = index_trimmed2

            elif how in DATASET_HIERARCHY:
                selected_index = query_obj.data[query_obj.data[NAME] == how].index

            else:  # what other ways might we want to get data?
                selected_index = 1
                pass

            data_objs = [NrelData.get_data_directly(lat, lon, s[1], s[0]) for s in selected_index]

        return data_objs


class NsrdbCsv(NrelData):

    URL_TEMPLATE = 'wkt=POINT({lon}+{lat})&names={year}&leap_day={leap}&interval={interval}&utc={utc}' \
                   '&full_name={name}&email={email}&affiliation={affiliation}&mailing_list={mailing_list}' \
                   '&reason={reason}&api_key={api}'

    def __init__(self):
        super(NsrdbCsv, self).__init__()
        self.attributes = None

        # Set leap year to true or false. True will return leap day data if present, false will not.
        self.leap_year = 'false'
        # Set time interval in minutes, i.e., '30' is half hour intervals. Valid intervals are 30 & 60.
        self.interval = '60'
        # Specify Coordinated Universal Time (UTC), 'true' will use UTC, 'false' will use the local time zone of the data.
        # NOTE: In order to use the NSRDB data in SAM, you must specify UTC as 'false'. SAM requires the data to be in the
        # local time zone.
        self.utc = 'true'
        # Your full name, use '+' instead of spaces.
        self.your_name = MY_NAME
        # Your reason for using the NSRDB.
        self.reason_for_use = USE_REASON
        # Your affiliation
        self.your_affiliation = AFFILIATION
        # Your email address
        self.your_email = EMAIL
        # Please join our mailing list so we can keep you up-to-date on new developments.
        self.mailing_list = 'true'

        self.status_code = None

    def form_url(self):
        self.check_location()
        self.url = NREL_URL + API_LINK[self.dataset] + CSV + NsrdbCsv.URL_TEMPLATE.format(
            year=str(self.year), lat=str(self.query_lat), lon=str(self.query_lon), api=NREL_API_KEY,
            leap=self.leap_year, interval=self.interval, utc=self.utc, name=self.your_name,
            email=self.your_email,
            mailing_list=self.mailing_list, affiliation=self.your_affiliation, reason=self.reason_for_use)

    def get_data(self):

        self.form_url()

        print 'requesting {dataset} for: LAT={lat}, LON={lon}, YEAR={year} . . .'.format(dataset=self.dataset,
                                                                                   lat=str(self.query_lat),
                                                                                   lon=str(self.query_lon),
                                                                                   year=self.year)
        print 'please wait ~ 1 min . . .'

        r = self.make_request()

        if self.status_code == 200:
            t = r.text
            self.info, self.ts_data = wt.convert_weather_text_to_dataframe(t, sep=',',
                                                                        ix_meta_name=0, ix_meta_data=1,
                                                                        ix_cols=2, ix_start=3)
            self.location_id = int(self.info.loc['Location ID', 0])
            self.actual_lat = float(self.info.loc['Latitude', 0])
            self.actual_lon = float(self.info.loc['Longitude', 0])
            self.city = str(self.info.loc['City', 0])
            self.state = str(self.info.loc['State', 0])
            self.country = str(self.info.loc['Country', 0])
            self.timezone = float(self.info.loc['Time Zone', 0])

            # Read the rest into a dataframe:
            print 'fetching timeseries data for ' + str(self.location_id) + ' ...'
            self.ts_data = pd.read_csv(self.url, skiprows=[0, 1])
            dt_index = [dt.datetime(int(self.ts_data.loc[d, 'Year']), int(self.ts_data.loc[d, 'Month']),
                                           int(self.ts_data.loc[d, 'Day']), int(self.ts_data.loc[d, 'Hour']),
                                           int(self.ts_data.loc[d, 'Minute'])) for d in self.ts_data.index]
            self.ts_data.index = dt_index
        else:
            # request was not successful:
            pass



class NsrdbMts1(NsrdbCsv):
    def __init__(self, lat, lon, year):
        super(NsrdbMts1, self).__init__()
        self.dataset = NSRDB_MTS1
        self.query_lat = lat
        self.query_lon = lon
        self.year = year
        # setting utc = 'true' in url returns timestamps in local time, rather than UTC.
        # found another comment about this on NREL user forum -- appears to be a known bug
        self.utc = 'false'


class NsrdbMts2(NsrdbCsv):
    def __init__(self, lat, lon, year):
        super(NsrdbMts2, self).__init__()
        self.dataset = NSRDB_MTS2
        self.query_lat = lat
        self.query_lon = lon
        self.year = year


class NsrdbPsm(NsrdbCsv):
    def __init__(self, lat, lon, year):
        super(NsrdbPsm, self).__init__()
        self.query_lat = lat
        self.query_lon = lon
        self.dataset = NSRDB_PSM
        self.year = year


class NsrdbInternational(NsrdbCsv):
    def __init__(self, lat, lon, year):
        super(NsrdbInternational, self).__init__()
        self.dataset = NSRDB_INTL
        self.query_lat = lat
        self.query_lon = lon
        self.year = year


class NsrdbSpectral(NsrdbCsv):
    def __init__(self, lat, lon, year):
        super(NsrdbSpectral, self).__init__()
        self.dataset = NSRDB_SPECTRAL
        self.query_lat = lat
        self.query_lon = lon
        self.year = year


class NsrdbJson(NrelData):
    def __init__(self):
        super(NsrdbJson, self).__init__()
        self.inputs = None
        self.errors = None
        self.warnings = None
        self.version = None
        self.metadata = None
        self.outputs = {}

    def get_data(self):
        self.check_location()
        self.check_url()

        print 'querying NSRDB . . .'
        r = self.make_request()
        a = json.loads(r.text)

        self.inputs = a['inputs']
        try:
            self.warnings = a['warnings']
        except KeyError:
            pass
        try:
            self.errors = a['errors']
        except KeyError:
            pass
        try:
            self.metadata = a['metadata']
        except KeyError:
            pass
        self.info = a['outputs']

        print 'done query'


class SolarQuery(NsrdbJson):

    def __init__(self, lat, lon):
        super(SolarQuery, self).__init__()
        self.dataset = SOLAR_QUERY
        self.query_lat = lat
        self.query_lon = lon

        self.form_url()

    def form_url(self):
        # Solar Dataset Query API also will take search radius as an optional argument.
        # TODO: Need to decide whether we want user to control search radius outside of this module...
        search_radius = 100
        self.url = NREL_URL + API_LINK[self.dataset] + JSON + \
               'api_key={api}&lat={lat}&lon={lon}&radius={radius}&all=1'.format(
                   lat=self.query_lat, lon=self.query_lon, api=NREL_API_KEY, radius=search_radius)

    def prettify_data(self):
        # 1. check if there is any data
        if len(self.info) == 0:
            print 'No sites found'
        elif len(self.info['all_stations']) > 0:
            # it appears that 'all_stations' only includes tmy2 + tmy3, not IN or intl datasets
            # as long as we are asking the API for all=1, then we should expect the json to include an 'all' key
            # in the outputs dict.  We only care about this one for now.
            self.data = pd.DataFrame(self.info['all_stations'])
        elif len(self.info['IN']) > 0:
            self.data = pd.DataFrame(self.info['IN'])
        elif len(self.info['intl']) > 0:
            self.data = pd.DataFrame(self.info['intl'])


class SolarResource(NsrdbJson):

    def __init__(self, lat, lon):
        super(SolarResource, self).__init__()
        self.dataset = SOLAR_RESOURCE
        self.query_lat = lat
        self.query_lon = lon

        self.form_url()

    def form_url(self):
        self.url = NREL_URL + API_LINK[self.dataset] + JSON + \
               'api_key={api}&lat={lat}&lon={lon}'.format(lat=self.query_lat, lon=self.query_lon, api=NREL_API_KEY)

    def prettify_data(self):
        pass


class NsrdbQuery(NsrdbJson):

    def __init__(self, lat, lon):
        super(NsrdbQuery, self).__init__()
        self.dataset = NSRDB_QUERY
        self.query_lat = lat
        self.query_lon = lon
        self.avail_years = None
        self.avail_datasets = None

        self.form_url()

    def form_url(self):
        self.url = NREL_URL + API_LINK[self.dataset] + JSON + \
               'wkt=POINT({lon}+{lat})&api_key={api}'.format(
                   lat=str(self.query_lat), lon=str(self.query_lon), api=NREL_API_KEY)

    def prettify_data(self):
        # 1. check if there is any data
        if len(self.info) == 0:
            pass
        else:

            dfs = [pd.concat(
                [pd.DataFrame(i[LINKS]),
                 pd.Series([i[NAME]] * len(pd.DataFrame(i[LINKS])), name=NAME),
                 pd.Series([i[MEASUREMENT]] * len(pd.DataFrame(i[LINKS])), name=MEASUREMENT),
                 pd.Series([i[DISP_NAME]] * len(pd.DataFrame(i[LINKS])), name=DISP_NAME)
                 ], axis=1)
                   for i in self.info]
            self.data = pd.concat(dfs, axis=0)
            multi_ix = pd.MultiIndex.from_arrays([self.data[NAME], self.data[AVAIL_YR]])
            self.data.set_index(multi_ix, inplace=True, drop=True)

            self.avail_datasets = list(set(self.data.loc[:, NAME]))
            self.avail_years = list(set(self.data.loc[:, AVAIL_YR]))

    def get_dataset_with_year(self, year):
        ix = [year in self.data[d][AVAIL_YR] for d in self.disp_names].index(True)
        return self.disp_names[ix]

    def get_most_recent_data(self):
        '''
        input: a single NsrdbQuery instance
        returns: tuple of (model name, year)
        '''
        all_years = [y for y in self.avail_years if y not in [u'tmy', u'tmy2', u'tmy3']]
        if len(all_years) > 0:
            most_recent_year = max(all_years)
        else:  # if no year-specific data is available, only tmy data is available
            most_recent_year = all_years[0]
        # disp_name = self.get_dataset_with_year(most_recent_year)
        # return disp_name, most_recent_year
        return most_recent_year

    @classmethod
    def get_url_for_selected_data(cls, query_result, disp_name=None, year='tmy', interval=60):
        # TODO: how should user interact with results of query? Which data model(s) to choose? Which year(s)?
        # For now, Yudong just wants the most recent year available
        '''
        (let user select model?)
        years: 'latest', 'tmy', an integer year, or a list of viable years
        interval: can only be 30 or 60.  some datasets don't actually provide 30 min interval data.
        if ds_names and years are a list, then they must be of the same size
        if interval is not a list but the others are a list, then it will be assumed to be a list of the same size
        '''

        # so, are we doing a list?:
        if type(year) is list:
            # not doing it now
            pass

        # if not, we're just doing one:
        requested_data = query_result.data.get(disp_name, None)
        if requested_data is None:
            print 'data not available'  # maybe make your own exception?  or just use KeyError?
        else:
            return str(requested_data[LINKS][requested_data[LINKS][AVAIL_INT] == interval].loc[year, 'link'])



# test each dataset individually
# fremont_mts2 = NrelData.get_data_directly(37.5483, -121.9886, 1996, NSRDB_MTS2)
# test a variety of sites:
# merced = NrelData.get_data_by_query(37.32, -120.48, export=False, which='best')
# print merced.calculate_distance()
# alabama = NrelData.get_data_by_query(32.78, -86.65, export=False, which='best')
# print alabama.calculate_distance()
# hyderabad = NrelData.get_data_by_query(17.4, 78.44, export=False, which='best')
# print hyderabad.calculate_distance()
# geneva = NrelData.get_data_by_query(46.2, 6.14, export=False, which='best')
# print geneva.calculate_distance()
# fremont = NrelData.get_data_by_query(37.5483, -121.9886, export=False, which='best')
# print fremont.calculate_distance()
# alabama_q = NrelData.get_query(32.3182, -86.9023)
# fremont_q = NrelData.get_query(37.5483, -121.9886)
# fremont = NrelData.get_data_range_by_query(37.5483, -121.9886, include_tmy=False, how='psm')

