---
layout: post
title: Using Plotly Express to Visualize the Palmer Penguins Dataset
---



## Intro

This post details the use of a sqlite3 database and the plotly package to create engaging and interactive visuals from a number of .csv files related to global tempuerature. To start off we first must import all relivant packages.


```python
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from plotly import express as px
import sqlite3
```

## Creating the database

Once packages have been imported, we start by opening a connecting to a sqlite3 database. Since we have not created this database before, this action also creates the database in the same folder where our notebook is saved. We then read in our csv files and rename the FIPS 10-4 column to FIPS to make is easier to query later from our database.


```python
#creat a temps database 
conn = sqlite3.connect("temps.db")

#read in the file
tempurature= pd.read_csv("temps.csv")
stations= pd.read_csv("station-metadata.csv")
countries= pd.read_csv("countries.csv")

#rename the column
countries['FIPS']=countries['FIPS 10-4']
countries.drop('FIPS 10-4', inplace=True, axis=1)
```

Next, we define a cleaning function and use it to clean the tempurature section of our data. This makes it easier to work with later on in the database.


```python
#this function cleans the data
def prepare_df(df):
    df = df.set_index(keys=["ID", "Year"])
    #restack the data from each column to single column
    df = df.stack()
    df = df.reset_index()
    #rename the columns
    df = df.rename(columns = {"level_2"  : "Month" , 0 : "Temp"})
    df["Month"] = df["Month"].str[5:].astype(int)
    df["Temp"]  = df["Temp"] / 100
    df["FIPS"] = df["ID"].str[0:2]
    return(df)
tempurature=prepare_df(tempurature)
```

Now that our data has been cleaned, we are ready to populate our database. Using the connection that we opened earlier, we populate our database with the df.to_sql() method and the close our connection as a matter of good coding practice.


```python
tempurature.to_sql("temperature", conn, index =False, if_exists = "replace")
countries.to_sql("countries", conn, index =False,  if_exists = "replace")
stations.to_sql("stations", conn, if_exists = "replace", index =False)
conn.close()
```

    /Users/conormccaulley/opt/anaconda3/lib/python3.8/site-packages/pandas/core/generic.py:2779: UserWarning: The spaces in these column names will not be changed. In pandas versions < 0.14, spaces were converted to underscores.
      sql.to_sql(


## Creating a Geographic Scatter Function from the Database

We have a populated database, what next? Queries! Thats how we get specific information that we want back out of the database. Our first query function will take in a country, beginning year, ending year, and month and returns a pandas dataframe containing latitude, longitude, year, month, temp, station name, country, and fips 10-4 id data from all stations in the specified country within the specified month and timeframe.


```python
def query_climate_database(country, year_begin, year_end, month):
    '''
    This function queries the database based on the time frame, month,
    and country specified in the args.
    Returns a pandas dataframe containing lat, long, year, month, temp, 
    name, country, and fips 10-4 data from all stations.
    
    @args: 
    country(str): the counry to be queiried
    month(int): the specified 
    year_begin(int): starting year
    year_end(int):ending year
    '''
    #open the connection to the database
    conn = sqlite3.connect("temps.db")
    cursor = conn.cursor()
    
    #retreive country code from the database
    cursor.execute("SELECT FIPS FROM countries WHERE NAME = ?",(country,))
    count_code=cursor.fetchall()[0][0]
    print(count_code)
    
    #create the query command
    cmd= \
    """
    SELECT S.Name, S.LATITUDE, S.LONGITUDE, T.Year, T.Month, T.Temp
    FROM STATIONS S
    LEFT JOIN 'temperature' T ON T.id=S.id
    WHERE T.FIPS = ? AND T.Month =? AND T.year >= ? AND T.year <= ?
    """
    df=pd.read_sql_query(cmd, conn,params=(count_code, month,year_begin, year_end,),)
    df['Country']=country
    return df
df=query_climate_database("India", 1980,2020, 2)
```




We then define a short function which takes in a data group and computes the line though the data points using the sklearn linear regressor. This data allows us to see how the average temperature changes at each station as predicted by the year.


```python
from sklearn.linear_model import LinearRegression
def coef(data_group):
    x = data_group[["Year"]] # 2 brackets because X should be a df
    y = data_group["Temp"]   # 1 bracket because y should be a series
    LR = LinearRegression()
    LR.fit(x, y)
    return LR.coef_[0]
```

We're finally ready for the fun part! Actually creating our graphic! For this we write a new function which creates a map with scatter points corresponding to each station in the country we specified. The scatter points are colored to represent average yearly temperature change so that we can visualize the yearly change at each station more clearly. The graphic resulting from the px.scatter_mapbox is interactive and allows the user to zoom in and out or view information related to a specific station.


```python
color_map = px.colors.diverging.RdGy_r # choose a colormap

def temperature_coefficient_plot(country, year_begin, year_end, month, min_obs, **kwargs):
    '''
    This function creates a map with scatter points corresponding to each station in that
    country. scatter points are colored to represent ave. yearly temp change
    
    @args: 
    country(string): the country to query
    year_begin(int): starting year
    year_end(int):ending year
    month(int): the month to query
    min_obs(int): the minimum number of observations a station must have to be included
    **kwargs: additional args to pass to the scatter plot
    '''
    
    #queiry the database to get a pandas dataframe with the specified information
    df=query_climate_database(country, year_begin,year_end, month)

    #groupby name and and use .size to find out how many observations there are for each station
    observations = df.groupby("NAME", as_index=False).size()
    
    #merge this new observation number df with our origional fram and drop the indexes with less than the 
    #minimun number of observations
    df=df.merge(observations, on="NAME")
    index=df[df["size"]<min_obs].index
    df.drop(index, inplace=True)
    
    #compute the rate at which tempurature is changing using the linear regression function and convert the 
    #resulting series to a dataframe
    coefs = df.groupby(["NAME"]).apply(coef).to_frame()
    
    #rename the column and reset the index
    coefs = coefs.rename(columns= {0: 'Estimated Yearly <br> Increase (°C)'})
    coefs=coefs.reset_index()
    
    #merge df and coefs
    df=df.merge(coefs, on ="NAME")
    df['Estimated Yearly <br> Increase (°C)']=df['Estimated Yearly <br> Increase (°C)'].round(4)
    
    #create the plot using px.scatter
    fig = px.scatter_mapbox(df, 
                        lat = "LATITUDE",
                        lon = "LONGITUDE", 
                        hover_name = "NAME",
                        color="Estimated Yearly <br> Increase (°C)",
                        **kwargs)
   
    return fig


fig=temperature_coefficient_plot("India",
                                    1980,
                                    2020,
                                    1,
                                    10,
                                    zoom = 2,
                                    height = 500,
                                    mapbox_style="open-street-map",
                                    color_continuous_scale=color_map,
                                    color_continuous_midpoint=0
                                    )
fig.update_layout(title_text='Estimates of yearly increase in temperature in January <br> for stations in India, years 1980-2020', title_x=0.5)
fig.show()
```
{% include geo_scatter.html %}

## Creating a Temperature range Choropleth

Now that we have our first graphic we need to create two more. The next one that we make will be a choropleth that shows how much averagae monthly temperatures vary through the year in each country. This makes seeing which countries have stable vs variable climates interesting and interactive for the user. We start by defining a query function like the one used in the last example exept this one returns a pandas dataframe containing latitude, longitude, year, month, temp, name, country, and fips 10-4 data from all stations around the world for the specified time frame.


```python
def query_climate_database_world(year_begin, year_end):
    '''
    This function queries the database based on the time frame specified in the args 
    and returns a pandas dataframe containing lat, long, year, month, temp, name, country,
    and fips 10-4 data from all stations.
    
    @args: 
    year_begin(int): starting year
    year_end(int):ending year
    '''
    #open the connection to the database
    conn = sqlite3.connect("temps.db")
    cursor = conn.cursor()
    
    #create the query command to retreive lat, long, year, month, temp, name, and fips
    #data from the database entries whos years match the arguments provided
    cmd= \
    """
    SELECT S.Name, S.LATITUDE, S.LONGITUDE, T.Year, T.Month, T.Temp, T.FIPS
    FROM STATIONS S
    LEFT JOIN 'temperature' T ON T.id=S.id
    WHERE T.year >= ? AND T.year <= ?
    """
    #create the dataframe using the above command
    df=pd.read_sql_query(cmd, conn,params=(year_begin, year_end,),)
    
    
    #create a new command to retreive all fips and name info from the countries table
    cmd= \
    """
    SELECT C.FIPS, C.Name
    FROM COUNTRIES C
    """
    #create a countries df
    country=pd.read_sql_query(cmd, conn,)
    #merge counries and df to add country name to the dataframe
    df=df.merge(country, on="FIPS")
    #close the connection
    conn.close()
    
    return df
df=query_climate_database_world(2010,2020)
df
```


We now have the data and we're ready to plot. Just like in the first example, we now take the query function defined above and use it to retreive the information we want from the database. We can then group the data and take the average temperature in each country (specific steps are described in the code comments below) and use that data to plot our choropleth.


```python
def plot_temp_range_choropleth(year_begin, year_end, **kwargs):
    '''
    This function creates choropleth map of average yearly temurature ranges by country
    (max monthly average - min monthly average).
    
    @args: 
    year_begin(int): starting year
    year_end(int):ending year
    **kwargs: additional args to pass to the scatter plot
    '''
    #import required librarys and open the geojson file
    from urllib.request import urlopen
    import json
    #retreive url
    countries_gj_url = "https://raw.githubusercontent.com/PhilChodrow/PIC16B/master/datasets/countries.geojson"

    with urlopen(countries_gj_url) as response:
        countries_gj = json.load(response)
        
    #retreive features from the geojson
    countries_gj["features"][1]
    
    #create a database using the read in arguments
    df=query_climate_database_world(year_begin, year_end)
    
    #groupby country, station, and month, and compute the average temperature
    #for each month for each station across the years selected in the database queiry
    averages=df.groupby(['Name', "NAME", 'Month'])[["Temp"]].mean().reset_index()

    #groupby country and station and compute the range in monthly average temperatures for 
    #each station (max-min) to find the yearly temp fluctuation
    averages=averages.groupby(['Name','NAME'])[['Temp']].agg(np.ptp).reset_index()

    #groupby country and take the average fluctuation from all the stations in each country to
    #get that countries average yearly tempeture fluctuation
    averages=averages.groupby(['Name'])[['Temp']].mean().reset_index()

    #rename the temp column
    averages['Temperature <br> Fluctuation (°C)']=averages['Temp'].round(4)
    
    


    fig = px.choropleth(averages, 
                        geojson=countries_gj,
                        locations = "Name",
                        locationmode = "country names",
                        color = "Temperature <br> Fluctuation (°C)", 
                        **kwargs)
    return fig
```

Finally we use the function we defined above to plot the choropleth.


```python
fig=plot_temp_range_choropleth(2010,2020,height=500)
fig.update_layout(title_text='Average annual temperature fluctuation by country 2010-2020<br> (Difference in hottest month vs coldest month)', title_x=0.5)
fig.show()
```
{% include Choropleth.html %}

## Creating a 3D Scatter Plot of Temperature and Elevation Data

At this point you know the drill, we're going to write a function to extract the relivant data from the database and then write another to use that data to create our plot. For this plot, we want to create a 3d scatter plot of the average temuratures at each station in a given country in a given month between the specified years. Latitude and Longitude will be displayed on the x and y axis while station elevation is displayed on the z axis. This allows the user to visualize not only a rough map of how tempurature changes across a country but also how it changes as you rise in elevation. In this case we want our query function to returns a pandas dataframe containing latitude, longitude, year, month, temp, elevation, name, country, and fips 10-4 data from all stations in the specified country.


```python
def query_climate_database_elev(country, year_begin, year_end, month):
    '''
    This function queries the database based on the time frame, month,
    and country specified in the args.
    Returns a pandas dataframe containing lat, long, year, month, temp, elevation,
    name, country, and fips 10-4 data from all stations in the specified country.
    
    @args: 
    country(str): the counry to be queiried
    month(int): the specified 
    year_begin(int): starting year
    year_end(int):ending year
    '''
    
    #open the connection to the database
    conn = sqlite3.connect("temps.db")
    cursor = conn.cursor()
    
    #retreive country code from the database
    cursor.execute("SELECT FIPS FROM countries WHERE NAME = ?",(country,))
    #fetch the code from the cursor
    count_code=cursor.fetchall()[0][0]
    print(count_code)
    
    #create the query command based on the function arguments
    cmd= \
    """
    SELECT S.Name, S.LATITUDE, S.LONGITUDE, S.STNELEV, T.Year, T.Month, T.Temp
    FROM STATIONS S
    LEFT JOIN 'temperature' T ON T.id=S.id
    WHERE T.FIPS = ? AND T.Month =? AND T.year >= ? AND T.year <= ?
    """
    #create a dataframe using the above command 
    df=pd.read_sql_query(cmd, conn,params=(count_code, month,year_begin, year_end,),)
    #add the appropriate country to the dataframe
    df['Country']=country
    
    return df
df=query_climate_database_elev("India", 1980,2020, 1)
df
```


Now that were have our the function to retreive our dataframe, we want to use that to plot our 3D scatter. To do this we take the datafram and take the average temperature at each station and use that as an imput for our 3d scatter function. The points on the graph are colored by average temperature to make them easy to visualize.


```python
def plot_avetemp_elevation(country, year_begin, year_end, month, **kwargs):
    '''
    This function creates a 3d scatter plot of the average temuratures at each 
    station in a given country in a given month between the specified years. 
    Latitude and Longitude are displayed on the x and y axis while station elevation is 
    displayed on the z axis. This allows the user to visualize not only a rough map of
    how tempurature changes across a country but also how it changes as you rise in 
    elevation. 
    @args: country(string): the country to query
    year_begin(int): starting year
    year_end(int):ending year
    month(int): the month to query
    **kwargs: additional args to pass to the scatter plot
    '''
    #call the query_climate_database_elev function to create a dataframe
    #with the requirements specified by the arguments
    df=query_climate_database_elev(country, year_begin, year_end, month)
    
    #get the average temp from each station for the specified month
    avetemp=df.groupby(['NAME'])[["Temp"]].mean().reset_index()
    avetemp.rename(columns= {"Temp": 'Average <br> Temperature (°C)'}, inplace=True)
    avetemp['Average <br> Temperature (°C)']=avetemp['Average <br> Temperature (°C)'].round(4)


    #merge the dataframes
    df=df.merge(avetemp, on='NAME')
    
    #rename columns for clarity
    df.rename(columns={
                     'LATITUDE': 'Latitude (°)',
                     'LONGITUDE': 'Longitude (°)',
                     'STNELEV': 'Station Elevation (m)'
                    }, inplace=True)
    
    #groupby on name to get the average temp, log/lat, and elivation in a convenient format
    df = df.groupby(['NAME']).mean().reset_index()
    fig = px.scatter_3d(df, x='Latitude (°)', y='Longitude (°)', z='Station Elevation (m)',
              color='Average <br> Temperature (°C)', **kwargs)
    return fig
    

```

Now we just have to plot our figure and we're done!


```python
fig=plot_avetemp_elevation("China",
                                    1980,
                                    2020,
                                    1, 
                           height=600)

fig.update_layout(title_text='Average temperature in China in January<br> plotted by location and elevation', title_x=0.5)


fig.show()
```

{% include 3d.html %}