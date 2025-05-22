import pandas as pd
import h3 as h3


CLEAN_DATA_PATH = "./Data/Raw_data/clean_data.csv"
DEFAULT_TEST_PATH = "./Data/Raw_data/2022-07-04-garbo01-combined-kml.csv"
GRAPHING_DATA_PATH = "./Graphing.csv"

#display all dataframe columns
pd.options.display.max_columns = None

# display all dataframe rows
pd.options.display.max_rows = None

clean_data = pd.read_csv(GRAPHING_DATA_PATH)
default_data = pd.read_csv(DEFAULT_TEST_PATH)

#############################
## Plotting in Matplotlib ##
#############################
# upload = clean_data["upload_bitrate_mbits/sec"]
# download = clean_data["download_bitrate_rx_mbits/sec"]

#upload_colour = upload
#colour = upload_colour

# download_colour = download
# colour = download_colour

# scatter = plt.scatter(x=x, y=y, s=0.01, c=colour)
# plt.colorbar(scatter)
# plt.title("Upload Bitrate Heatmap")
# plt.show()

# scatter = plt.scatter(x=x, y=y, s=0.01, c=colour)
# plt.colorbar(scatter)
# plt.title("Download Bitrate Heatmap")
# plt.show()

#############################
##### Analysis using H3 #####
#############################
def compute_h3_and_boundaries(row, res=12):
    lat = row["latitude"]
    long = row["longitude"]
    
    h3_index = h3.latlng_to_cell(lat, long, res=10)
    boundary = h3.cell_to_boundary(h3_index)
    print(row)
    return pd.Series([h3_index, boundary])

clean_data[['H3_Index', 'Boundaries']] = clean_data.apply(compute_h3_and_boundaries, axis = 1)
print(clean_data[['H3_Index', 'Boundaries']])

#############################
##### Calculate Avg Svr #####
#############################
svr1 = clean_data["svr1"]
svr2 = clean_data["svr2"]
svr3 = clean_data["svr3"]
svr4 = clean_data["svr4"]

clean_data["average_svr"] = ((svr1+svr2+svr3+svr4)/4)


#Write all data to csv
clean_data.to_csv("./Graphing.csv")





























#latitude
#min = -37.828817
#max = -37.631222
#range = 0.197595

#longitude
#min = 144.817058
#max = 144.920503
#range = 0.103455

#upload speed
#min = 0
#max = 83.9

#download speed
#min 0
#max 182.0