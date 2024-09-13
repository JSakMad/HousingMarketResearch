import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import matplotlib.gridspec as gridspec
import geopandas as gpd
import datetime
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates
from scipy import stats
import math
import random
import os
import time

# Cost of living index for each state for normalization of data (linked in notes)
cost_of_living_index = {
    'MS': 86.0,   # Mississippi
    'OK': 86.9,   # Oklahoma
    'KS': 87.2,   # Kansas
    'AL': 88.2,   # Alabama
    'WV': 89.3,   # West Virginia
    'GA': 89.3,   # Georgia
    'MO': 89.9,   # Missouri
    'IA': 89.9,   # Iowa
    'AR': 90.1,   # Arkansas
    'TN': 90.4,   # Tennessee
    'NE': 91.3,   # Nebraska
    'OH': 91.4,   # Ohio
    'IN': 91.5,   # Indiana
    'WY': 92.1,   # Wyoming
    'MI': 92.1,   # Michigan
    'IL': 92.1,   # Illinois
    'LA': 92.2,   # Louisiana
    'TX': 92.9,   # Texas
    'SD': 93.4,   # South Dakota
    'NM': 94.1,   # New Mexico
    'KY': 94.5,   # Kentucky
    'WI': 95.1,   # Wisconsin
    'MN': 95.6,   # Minnesota
    'NC': 95.8,   # North Carolina
    'ND': 96.0,   # North Dakota
    'SC': 96.4,   # South Carolina
    'PA': 97.0,   # Pennsylvania 
    'PR': 97.9,   # Puerto Rico 
    'ID': 99.2 ,     #"Idaho"
    "NV":101.7 ,     #"Nevada"
    "FL":101.9 ,     #"Florida"
    "VA":102.6 ,     #"Virginia"
    "UT":102.7 ,     #"Utah"
    "MT":103.0 ,     #"Montana"
    "DE":103.3 ,     #"Delaware"
    "CO":104.8 ,     #"Colorado"
    "AZ":107.1 ,     #"Arizona"
    "NJ":111.7 ,     #"New Jersey"
    "RI":111.8 ,     #"Rhode Island"
    "ME":112.5 ,     #"Maine"
    "CT":114.4 ,     #"Connecticut"
    "NH":114.6 ,     #"New Hampshire"
    "WA":115.5 ,     #"Washington"
    "VT":115.6 ,     #"Vermont"
    "OR":116.2 ,     #"Oregon"
    "MD":120.7 ,     #"Maryland"
    "AK":125.3 ,     #"Alaska"
    "NY":126.6 ,      #"New York" 
    "CA":139.7 ,      #"California" 
    "MA":143.1 ,      #"Massachusetts" 
    "DC":149.7 ,      #"District of Columbia" 
    "HI":181.5        #"Hawaii" 
}

# To convert dataset into pandas
data = pd.read_csv('NEWstate_market_tracker.tsv000', sep='\t')

# Map the dictionary to your DataFrame
data['cost_of_living_index'] = data['state_code'].map(cost_of_living_index)

#region Normalization Process
# Create Normalized Price Data
data['price_adjusted'] = data['median_ppsf'] / data['cost_of_living_index']

# Create Normalized Month to Month Price
data['price_adjusted_mom'] = data['median_ppsf_mom'] / data['cost_of_living_index']

# Cumulative inflation index (based on .gov data)
cumulative_inflation_index = {
    2019: 1.0,        # Base year, no inflation
    2020: 1.014,      # 1.4% inflation rate in 2020
    2021: 1.014*1.07, # Cumulative inflation up to 2021
    2022: 1.014*1.07*1.065, # Cumulative inflation up to 2022
    2023: 1.014*1.07*1.065*1.037 # Projected cumulative inflation up to 2023
}

# Convert 'period_begin' to datetime and extract the year
data['year'] = pd.to_datetime(data['period_begin']).dt.year

# Map the cumulative inflation index to your DataFrame
data['cumulative_inflation_index'] = data['year'].map(cumulative_inflation_index)

# Adjust the price for inflation
data['price_adjusted'] = data['price_adjusted'] / data['cumulative_inflation_index']

# Create Normalized Month to Month Price for Inflation
data['price_adjusted_mom'] = data['median_ppsf_mom'] / data['cumulative_inflation_index']
#endregion

# To remove null columns
data = data.dropna(axis=1, how='all')

# To remove undeeded columns
data = data.drop(['period_duration', 'region_type','region_type_id','is_seasonally_adjusted','last_updated','off_market_in_two_weeks_yoy','off_market_in_two_weeks_mom','off_market_in_two_weeks','pending_sales','pending_sales_mom','state','property_type','median_ppsf_yoy','median_list_ppsf',	'median_list_ppsf_mom',	'median_list_ppsf_yoy'], axis=1)

# To view the first 5 rows of the data
print(data.head())
print(data.shape)
data.info()

# Missing data
total = data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(6)

# To describe the data 
print(data.info())
print(data.describe())

# Converting
data.isnull().sum()

#region Date Based Data Creation
# Convert 'period_begin' to datetime
data['period_begin'] = pd.to_datetime(data['period_begin'])

# Filter rows for year 2019
data_2019 = data[data['period_begin'].dt.year == 2019]

# Filter rows for year 2021
data_2021 = data[data['period_begin'].dt.year == 2021]

# Filter rows for year 2023
data_2023 = data[data['period_begin'].dt.year == 2023]
#endregion

#region MASTER COUNTRY MAPS
#region Creating Heat Maps
# Load US States geometry
us_states = gpd.read_file('https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json')

# Merge with the dataframe
merged = us_states.set_index('name').join(data.set_index('region'))

# Handle outliers
Q1 = merged['price_adjusted'].quantile(0.25)
Q3 = merged['price_adjusted'].quantile(0.75)
IQR = Q3 - Q1
filtered = merged.query('(@Q1 - 1.5 * @IQR) <= price_adjusted <= (@Q3 + 1.5 * @IQR)')

# Determine min and max values for colormap
min_value = filtered['price_adjusted'].min()
max_value = filtered['price_adjusted'].max()

#region 2019 Map
# Merge with the 2019 dataframe
merged1 = us_states.set_index('name').join(data_2019.set_index('region'))
# Plot the DataFrame
fig, ax = plt.subplots(1, 1)
merged1.plot(column='price_adjusted', ax=ax, legend=True, vmin=min_value, vmax=max_value)

# Hide the x and y axis
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)

# Save the figure
plt.savefig('NewStatemap2019.png')  
# Display the figure
plt.show()
#endregion

#region 2021 Map
# Merge with the 2021 dataframe
merged2 = us_states.set_index('name').join(data_2021.set_index('region'))
# Plot the DataFrame
fig, ax = plt.subplots(1, 1)
merged2.plot(column='price_adjusted', ax=ax, legend=True, vmin=min_value, vmax=max_value)

# Hide the x and y axis
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)

# Save the figure
plt.savefig('NewStatemap2021.png')  
# Display the figure
plt.show()
#endregion

#region 2023 Map
# Merge with the 2021 dataframe
merged2 = us_states.set_index('name').join(data_2023.set_index('region'))
# Plot the DataFrame
fig, ax = plt.subplots(1, 1)
merged2.plot(column='price_adjusted', ax=ax, legend=True, vmin=min_value, vmax=max_value)

# Hide the x and y axis
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)

# Save the figure
plt.savefig('NewStatemap2023.png')  
# Display the figure
plt.show()
#endregion
#endregion

#region MASTER 2023 GRAPHS
#region Graph 2023
# Filter data for the years 2019 to 2023
start_date = '2019-01-01'
end_date = '2023-12-31'
mask = (data['period_begin'] >= start_date) & (data['period_begin'] <= end_date)
filtered_data = data.loc[mask]

# Sort filtered_data by 'period_begin'
filtered_data = filtered_data.sort_values('period_begin')

#region Cumulative Change 2023
# Calculate the running total of 'price_adjusted_mom'
filtered_data['cumulative_price_adjusted_mom'] = filtered_data['price_adjusted_mom'].cumsum()

# Plot the running total over time for the years 2019 to 2023
plt.figure(figsize=(10,6))
plt.plot(filtered_data['period_begin'], filtered_data['cumulative_price_adjusted_mom'])
plt.title('Cumulative Month-to-Month Adjusted Price Over Time (2019-2023)')
plt.xlabel('Time')
plt.ylabel('Cumulative Month-to-Month Adjusted Price')
plt.savefig('NewMonth to Month Cumulative Price 2019-2023.png')
plt.show()
#endregion

# Extracting the year from the 'period_begin' column
filtered_data['year'] = pd.to_datetime(filtered_data['period_begin']).dt.year

# Grouping the data by year and calculating the cumulative sum
grouped_data = filtered_data.groupby('year')['price_adjusted_mom'].sum()

# Calculate cumulative sum manually
cumulative_sum = grouped_data.cumsum()

# Plotting the cumulative sum over time
plt.figure(figsize=(10,6))

# Creating a bar chart with cumulative sum
plt.bar(grouped_data.index, cumulative_sum)

plt.title('Cumulative Month-to-Month Adjusted Price Over Time (2019-2023)')
plt.xlabel('Year')
plt.ylabel('Cumulative Month-to-Month Adjusted Price')
plt.xticks(grouped_data.index, rotation=45)  # Set x-axis labels as years
plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.savefig('NewMonth to Month Cumulative Price 2019-2023.png')
plt.show()

#region Line of Best Fit for Graph 2023
# Calculate the running total of 'price_adjusted_mom'
filtered_data['cumulative_price_adjusted_mom'] = filtered_data['price_adjusted_mom'].cumsum()
filtered_data = filtered_data.dropna(subset=['period_begin', 'price_adjusted_mom'])

# Get x values (time) and y values (cumulative_price_adjusted_mom)
x = filtered_data['period_begin'].map(datetime.datetime.toordinal)  # Convert dates to integers
y = filtered_data['cumulative_price_adjusted_mom']

# Calculate coefficients for the quadratic curve of best fit
coefficients = np.polyfit(x, y, 2)

# Generate y values for the curve of best fit
polynomial = np.poly1d(coefficients)
y_fit = polynomial(x)

# Plot the running total and curve of best fit over time for the years 2019 to 2021
plt.figure(figsize=(10,6))
plt.plot(filtered_data['period_begin'], filtered_data['cumulative_price_adjusted_mom'], label='Data')
plt.plot(filtered_data['period_begin'], y_fit, color='red', label='Fit: {0:.2f} + {1:.2f}x + {2:.2f}x^2'.format(coefficients[2], coefficients[1], coefficients[0]))
plt.title('Cumulative Month-to-Month Adjusted Price Over Time (2019-2023)')
plt.xlabel('Time')
plt.ylabel('Cumulative Month-to-Month Adjusted Price')
plt.legend()
plt.savefig('NewMonth to Month Cumulative Price Best Fit 2019-2023.png')
plt.show()
#endregion

#region Actual Equation 2023
# Convert 'period_begin' to datetime if not already done
filtered_data['period_begin'] = pd.to_datetime(filtered_data['period_begin'])

# Calculate the number of days since the start of your data
filtered_data['days'] = (filtered_data['period_begin'] - filtered_data['period_begin'].min()).dt.days

# Get x values (days) and y values (cumulative_price_adjusted_mom)
x = filtered_data['days']
y = filtered_data['cumulative_price_adjusted_mom']

# Calculate coefficients for the quadratic curve of best fit
coefficients = np.polyfit(x, y, 2)

# Generate y values for the curve of best fit
polynomial = np.poly1d(coefficients)
y_fit = polynomial(x)

# Plot the running total and curve of best fit over time for the years 2019 to 2021
plt.figure(figsize=(10,6))
plt.plot(filtered_data['days'], filtered_data['cumulative_price_adjusted_mom'], label='Data')
plt.plot(filtered_data['days'], y_fit, color='red', label='Fit: {0:.10f} + {1:.10f}x + {2:.10f}x^2'.format(coefficients[2], coefficients[1], coefficients[0]))
plt.title('Cumulative Month-to-Month Adjusted Price Over Time (2019-2023)')
plt.xlabel('Days Since Start')
plt.ylabel('Cumulative Month-to-Month Adjusted Price')
plt.legend()
plt.savefig('NewMonth to Month Cumulative Price Best Fit Adjusted 2019-2023.png')
plt.show()
#endregion
#endregion


