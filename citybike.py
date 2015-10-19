#!usr/bin/env python
import requests
import pandas as pd
import numpy as np
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
import sqlite3 as lite
import time
from dateutil.parser import parse
import collections


def load_dataset(loc):
    r = requests.get(loc)
    return r.json()


def preprocess(rjson, **kwargs):
    keys = rjson.keys()
    df = json_normalize(rjson[kwargs['column']])
    return keys, df


def drop_table(name):
    return 'DROP TABLE IF EXISTS' + ' ' + name


def create_table(con, **kwargs):
    cur = con.cursor()
    with con:
        cur.execute(kwargs["drop_table"])
        cur.execute(kwargs["sql_string"])
    print('Database created')


def ref_data(con, rjson, **kwargs):
    cur = con.cursor()
    var = kwargs['ref_vars']
    insert_sql = 'INSERT INTO ' + \
        kwargs['table'] + \
        ' (' + ', '.join(var) + ') VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)'
    with con:
        for station in rjson[kwargs['column']]:
            cur.execute(
                insert_sql, tuple([station[var[i]] for i in range(len(var))]))


def time_data(con, rjson, **kwargs):
    cur = con.cursor()
    exec_time = pd.to_datetime(rjson[kwargs['column']])
    insert_sql = 'insert into ' + \
        kwargs['table'] + ' (' + kwargs['column'] + ') values (?)'
    id_bikes = collections.defaultdict(int)
    for station in rjson['stationBeanList']:
        id_bikes[station['id']] = station['availableBikes']
    with con:
        cur.execute(insert_sql, (exec_time.strftime('%s'),))
        for k, v in id_bikes.iteritems():
            cur.execute('update available_bikes set _' + str(k) + ' = ' + str(v) + ' where executionTime = '
                        + exec_time.strftime('%s') + ';')

if __name__ == '__main__':

    loc = 'https://www.divvybikes.com/stations/json'

    # JSON output using requests
    rjson = load_dataset(loc)
    keys, df = preprocess(rjson, column='stationBeanList')

    ref_table_sql = '''CREATE TABLE citibike_reference (
    id INT PRIMARY KEY,
    totalDocks INT,
    city TEXT,
    altitude INT, 
    stAddress2 TEXT,
    longitude NUMERIC,
    postalCode TEXT,
    testStation TEXT,
    stAddress1 TEXT,
    stationName TEXT,
    landMark TEXT,
    latitude NUMERIC,
    location TEXT
    )'''

    ref_vars = '''id, totalDocks, city, altitude, stAddress2, longitude, 
    postalCode, testStation, stAddress1, stationName, landMark, 
    latitude, location'''.replace('\n', '').replace('    ', '').replace(' ', '').split(',')

    db_name = 'divvy_bike.db'

    # Create a database
    con = lite.connect(db_name)

    # Reference Table
    create_table(
        con, sql_string=ref_table_sql, drop_table=drop_table('citibike_reference'))
    ref_data(con, rjson, table='citibike_reference',
             ref_vars=ref_vars, column='stationBeanList')

    # Time Table
    station_ids = df['id'].tolist()
    station_ids = ['_' + str(x) + ' INT' for x in station_ids]

    time_table_sql = "create table available_bikes (executionTime INT, " + ", ".join(
        station_ids) + ");"

    create_table(
        con, sql_string=time_table_sql, drop_table=drop_table('available_bikes'))

    for i in range(6):
        rjson = load_dataset(loc)
        time_data(con, rjson, table='available_bikes', column='executionTime')
        time.sleep(6)
    # create_table(con,sql_string=time_table,drop_table=drop_table('available_bikes'))
    df1 = pd.read_sql('SELECT * from available_bikes', con)
    con.close()
