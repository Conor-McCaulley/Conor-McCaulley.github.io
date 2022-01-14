---
layout: post
title: Using Plotly Express to Visualize the Palmer Penguins Dataset
---

## Intro

This post details how to use the plotly express package to produce a simple and engaging visualization of the Palmer Penguins dataset. To start off we will read the dataset into our computers memory so we can access and manipulate it. To do this we will use pd.read_csv to turn the data into a locally stored pandas dataframe.


```python
import pandas as pd
url = "https://raw.githubusercontent.com/PhilChodrow/PIC16B/master/datasets/palmer_penguins.csv"
penguins = pd.read_csv(url)
```

## Data Cleaning

Now that we have the data read in, the next step is to clean it. To do this we write a function which recodes string data to integers in order to make it easier to plot and then drops any NaN values which might be in the data and which could disrupt our later attemps to plot the data. 


```python
def clean(df):
    #copy the read in df object
    penguins=df.copy()
    #create recode dicts for sex, island, and species
    #to turn string information into ints
    sexRecode={'MALE': 0, 'FEMALE': 1,'.': 0}
    islandRecode={'Torgersen':0,'Biscoe':1,'Dream':2}
    speciesRecode={'Adelie Penguin (Pygoscelis adeliae)':0,'Chinstrap penguin (Pygoscelis antarctica)':1, 'Gentoo penguin (Pygoscelis papua)':2}
    
    #recode string information to ints
    penguins['Species']=penguins['Species'].map(speciesRecode)
    penguins['Island'] = penguins['Island'].map(islandRecode)
    penguins['Sex'] = penguins['Sex'].map(sexRecode)

    #select Relevant features for feature testing
    cols=["Species", "Flipper Length (mm)", "Body Mass (g)",'Island','Culmen Length (mm)','Culmen Depth (mm)','Sex']
    penguins=penguins[cols]

    #drop NaN values and return the dataframe
    penguins=penguins.dropna()
    return penguins

penguins=clean(penguins)


```

Below we can see what the cleaned dataset looks like.


```python
penguins
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Species</th>
      <th>Flipper Length (mm)</th>
      <th>Body Mass (g)</th>
      <th>Island</th>
      <th>Culmen Length (mm)</th>
      <th>Culmen Depth (mm)</th>
      <th>Sex</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>181.0</td>
      <td>3750.0</td>
      <td>0</td>
      <td>39.1</td>
      <td>18.7</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>186.0</td>
      <td>3800.0</td>
      <td>0</td>
      <td>39.5</td>
      <td>17.4</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>195.0</td>
      <td>3250.0</td>
      <td>0</td>
      <td>40.3</td>
      <td>18.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>193.0</td>
      <td>3450.0</td>
      <td>0</td>
      <td>36.7</td>
      <td>19.3</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>190.0</td>
      <td>3650.0</td>
      <td>0</td>
      <td>39.3</td>
      <td>20.6</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>338</th>
      <td>2</td>
      <td>214.0</td>
      <td>4925.0</td>
      <td>1</td>
      <td>47.2</td>
      <td>13.7</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>340</th>
      <td>2</td>
      <td>215.0</td>
      <td>4850.0</td>
      <td>1</td>
      <td>46.8</td>
      <td>14.3</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>341</th>
      <td>2</td>
      <td>222.0</td>
      <td>5750.0</td>
      <td>1</td>
      <td>50.4</td>
      <td>15.7</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>342</th>
      <td>2</td>
      <td>212.0</td>
      <td>5200.0</td>
      <td>1</td>
      <td>45.2</td>
      <td>14.8</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>343</th>
      <td>2</td>
      <td>213.0</td>
      <td>5400.0</td>
      <td>1</td>
      <td>49.9</td>
      <td>16.1</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>334 rows × 7 columns</p>
</div>



## Plotting

Now that we have our cleaned dataset, were almost ready to start plotting. The last thing we need to do is install the plotly express package on our computers. If you already have pip installed, this can easily be accomplished by running "pip install plotly_express==0.4.0" in a terminal window. Finally, we are ready to plot the data. In this example we will be creating a 3d scatter plot using the plotly express scatter_3d function. First we restrict the columns in the dataframe to those we want to sort our data points by, in this case, Culmen Depth, Culmen Length, Island, and Species. Then we use scatter_3d to scatter our datapoints in a 3d plot with Culmen Length on the x axis, Culmen Depth on the y axis, and Island on the z axis. The data points are color coded by species. 


```python
import plotly.express as px
df = penguins[['Culmen Depth (mm)', 'Culmen Length (mm)','Island','Species']]
fig = px.scatter_3d(df, x='Culmen Depth (mm)', y='Culmen Length (mm)', z='Island',
              color='Species', width=800, height=400)

fig.show()
```

