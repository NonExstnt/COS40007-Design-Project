import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import h3 as h3
import geopandas as gp
import contextily as cx
from shapely.geometry import Polygon
import folium
import branca.colormap as cm
import ast

GRAPHING_DATA_PATH = "./Graphing.csv"

#Define centre point of plot as fixed value
lat = -37.7
long = 144.85

#display all dataframe columns
pd.options.display.max_columns = None

# display all dataframe rows
pd.options.display.max_rows = None

##############
#Count processing
##############

graphing_data = pd.read_csv(GRAPHING_DATA_PATH)
 
#Calculate distrubution of data across indexes
graphing_data_counts = graphing_data.groupby('H3_Index').size().reset_index(name='Count')

#graphing_data_merge = pd.read_csv('./Graphing.csv')

#Calculate average latancy across indexes
average_values = graphing_data.groupby('H3_Index')['average_svr'].mean().reset_index(name='Average Svr')

#Merge the boundaries back with the counts
graphing_data_merge = graphing_data_counts.merge(
    graphing_data[['H3_Index', 'Boundaries']].drop_duplicates(), on='H3_Index', how='left'
 )

#Merge the boundaries back with the averages
graphing_data_merge = average_values.merge(
    graphing_data[['H3_Index', 'Boundaries']].drop_duplicates(), on='H3_Index', how='left'
 )

m = folium.Map(location=[lat, long], zoom_start=11, tiles='cartodbpositron')

#Apply heatmap of distribution of data
#colormap = cm.LinearColormap(colors=['#99ccff', '#0000ff'], vmin=graphing_data_merge['Count'].min(), vmax=graphing_data_merge['Count'].max())

#Apply heatmap of average latancy
colormap = cm.LinearColormap(colors=['#ff0000', '#33cc33'], vmin=graphing_data_merge['Average Svr'].min(), vmax=graphing_data_merge['Average Svr'].max())


# Add the H3 cells to the map with color based on count
for _, row in graphing_data_merge.iterrows():
    boundaries = row['Boundaries']
    #count = row['Count']
    count=row['Average Svr']
    cords = ast.literal_eval(boundaries)
    #print(cords)
    color = colormap(count)
    folium.Polygon(
        locations = cords,
        color=color,
        weight=1,
        fill=True,
        fill_opacity=0.7,
        popup=f'Count: {count}'
).add_to(m)

# Add the colormap to the map
colormap.add_to(m)

# Display the map
m.save("map.html")






