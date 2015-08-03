
#!usr/bin/env python

import sqlite3 as lite
import pandas as pd

#Add more cities to text if needed

raw_weather_data = '''New York City   2013    July        January     62
          Boston          2013    July        January     59
          Chicago         2013    July        January     59
          Miami           2013    August      January     84
          Dallas          2013    July        January     77
          Seattle         2013    July        January     61
          Portland        2013    July        December    63
          San Francisco   2013    September   December    64
          Los Angeles     2013    September   December    75'''

def db_io_weather(s):
    """
    Returns a list of tuples, each element has form
    ('city', year ,'warm_month','cold_month, average)
    """
    s = s.split('\n')
    items=[]
    for i,line in enumerate(s):
        """splitting by spaces(in this case greater than one) and 
        filtering out spaces
        """
        items_in_s = filter(bool,line.strip().split('  '))
        i = items_in_s
        items.append((i[0],int(i[1]),i[2],i[3],int(i[4])))
    return items



con = lite.connect('getting_started.db')
with con:
    cur = con.cursor()
    cur.execute("""drop table if exists cities""")
    cur.execute("""create table cities (name text, state text)""")
    cur.execute("""drop table if exists weather""")
    cur.execute("""create table weather 
    (city text,year integer,warm_month text,cold_month text,average integer)""")
    
    #Entering city data into cities table
    cur.execute('''insert into cities values 
                   ('Washington','DC'),
                   ('New York City', 'NY'),
                   ('Boston', 'MA'),
                   ('Chicago', 'IL'),
                   ('Miami', 'FL'),
                   ('Dallas', 'TX'),
                   ('Seattle', 'WA'),
                   ('Portland', 'OR'),
                   ('San Francisco', 'CA'),
                   ('Los Angeles', 'CA');''')
    
    #Entering weather data into weather table
    con.executemany("insert into weather values (?,?,?,?,?)",
                    tuple(db_io_weather(raw_weather_data)))
    cur.execute('''
    select name,state,year,warm_month,cold_month 
                from weather
                left outer join cities
                on name = city
    ''')
    output = cur.fetchall()
    cols = [col_name[0] for col_name in cur.description]
    df = pd.DataFrame(output,columns=cols)
    grouped = df.groupby('warm_month')
    grouped = dict(list(grouped))
    grouped = grouped['July']
    grouped = grouped[['name','state']]

print("The warmest cities in July are:\n{}".format(grouped.to_csv(index=False,header=False)))