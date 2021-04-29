#!/usr/bin/env python
# coding: utf-8

# =================================================================================================
# Filename:     geopandas-location-distance.py
# Date:         4/29/2021
# Description:  Playbook for using geopandas to calculate distance.
#               Based on an actual coding interview (removed company name).
# =================================================================================================
# 
# ### References
# * Covid-19 Data location.  https://github.com/nytimes/covid-19-data/blob/master/live/us-counties.csv
# * Covid-19 Data location (raw).  https://raw.githubusercontent.com/nytimes/covid-19-data/master/live/us-counties.csv
# * Geocode data.  https://data.healthcare.gov/dataset/Geocodes-USA-with-Counties/52wv-g36k/data
# * Paging through data.  https://dev.socrata.com/docs/paging.html#2.1
# * Read into pandas from CSV.  https://stackoverflow.com/questions/32400867/pandas-read-csv-from-url
# * Pandas error bad lines.  https://stackoverflow.com/questions/18039057/python-pandas-error-tokenizing-data
# * Python combinations in list. https://www.geeksforgeeks.org/python-all-possible-pairs-in-list/
# * Python permutations in list.  https://www.geeksforgeeks.org/python-itertools-permutations/
# * Convert pandas dataframe to dictionary.  https://stackoverflow.com/questions/26716616/convert-a-pandas-dataframe-to-a-dictionary

# In[ ]:





# ### Command to convert Jupyter Notebook to Python File
# 
# See below for syntax to run at command line
# ``` shell
# ! jupyter nbconvert --to script *.ipynb
# ```
# 
# **Reference:**
# * https://stackoverflow.com/questions/17077494/how-do-i-convert-a-ipython-notebook-into-a-python-file-via-commandline

# In[3]:


# get_ipython().system(' jupyter nbconvert --to script *.ipynb')


# In[ ]:





# ### Below are package installations using `pip install` and import statements

# In[1]:


# ! pip install sodapy


# In[2]:


# ! pip install geopy


# In[3]:


# ! pip install tqdm


# In[4]:


# # ordinarily this file would be included in a .gitignore file, so it doesn't get pushed to remote repo
# from secret import app_token


# In[5]:


import pandas as pd
import io
import requests

# pandas display settings
pd.set_option("display.max_columns", 999)


# In[6]:


from itertools import combinations 


# In[7]:


from geopy import distance


# In[8]:


from tqdm import tqdm


# In[ ]:





# ### Step 1: gather the two-letter abbreviations for states
# * The COVID-19 dataset has County and State 
#     1. plan to concatenate, to account for the possibility of same county name in multiple states
#     1. The `state` is spelled out in full
# * The GeoCodes dataset has County and State
#     1. The `state` uses two-letter abbreviations
# * In order to merge the COVID-19 dataset with the GeoCodes dataset, gather a reference table of states spelled out in full vs. two-letter abbreviations

# In[9]:


def get_state_abbrev():
    '''
    Get the two-letter abbreviations for U.S. States.
    '''
    
    # read table from html page, grab first table
    df_states = pd.read_html("https://www.ssa.gov/international/coc-docs/states.html")[0]

    # add columns
    df_states.columns = ['state', 'abbrev']

    # convert state to lowercase
    df_states['state'] = df_states['state'].str.lower()

    # convert abbrev to lowercase
    df_states['abbrev'] = df_states['abbrev'].str.lower()
    
    print(f"Shape of dataframe: {df_states.shape}")
    
    return df_states


# In[10]:


get_ipython().run_cell_magic('time', '', "'''\nShape of dataframe: (56, 2)\nCPU times: user 212 ms, sys: 8 ms, total: 220 ms\nWall time: 876 ms\n'''\n\ndf_states = get_state_abbrev()\n\ndf_states.head()")


# In[ ]:





# ### Step 2: gather the COVID-19 data

# In[11]:


def get_covid19_data():
    '''
    Get the COVID-19 data.  Make sure to use the raw data, not the html version of the page.
    '''
    
    # read dataframe from the *raw* location within github
    df_covid19 = pd.read_csv("https://raw.githubusercontent.com/nytimes/covid-19-data/master/live/us-counties.csv")

    # convert state to lowercase
    df_covid19['state'] = df_covid19['state'].str.lower()

    # convert state to lowercase
    df_covid19['county'] = df_covid19['county'].str.lower()

    print(f"Shape of dataframe: {df_covid19.shape}\n")
    print(f"Number of unique values per column: \n{df_covid19.nunique()}\n")
    
    return df_covid19


# In[12]:


get_ipython().run_cell_magic('time', '', "'''\nCPU times: user 44 ms, sys: 8 ms, total: 52 ms\nWall time: 128 ms\n'''\n\ndf_covid19 = get_covid19_data()\n\ndf_covid19.head()")


# In[ ]:





# ### Step 3: join the COVID-19 data with the state abbreviation data

# In[13]:


# merge the state abbreviations
df_covid19_merge = df_covid19.merge(df_states, on='state')

# create column concatenating county + abbrev
df_covid19_merge['county_state'] = df_covid19_merge['county'] + '_' + df_covid19_merge['abbrev']

print(f"Shape of dataframe: {df_covid19_merge.shape}\n")
print(f"Number of unique values per column: \n{df_covid19_merge.nunique()}\n")

df_covid19_merge.head()


# In[14]:


# confirm unique number of county_state values

print(f"Number of Unique `county_state` values: {df_covid19_merge['county_state'].nunique()}")
print(f"Total `county_state` values: {df_covid19_merge['county_state'].shape[0]}")
print()

print(f"Value counts head of `county_state`: ")
df_covid19_merge['county_state'].value_counts().head()


# In[ ]:





# ### Step 4: gather the County Geocode data

# In[15]:


def get_county_geocode_data():
    '''
    Get county geocode data.  
    
    While loop until the end of the API has been reached, i.e. no records retrieved.
    Approximately 50K records, retrieved 1K records at a time.
    
    Public Rest API seems to work, but if necessary also retrieved developer API key.
    '''
    
    # create empty dataframe to store geocodes from API
    df_all_geocodes = pd.DataFrame()

    # create variables to page through the API data
    int_offset = 0
    int_limit = 1000
    count_iter = 0
    count_geocodes = 1

    while count_geocodes != 0:

        response = requests.get(f"https://data.healthcare.gov/resource/geocodes-usa-with-counties.json?$offset={int_offset}&$limit={int_limit}")
        ls_geocodes = response.json()

        # create dataframe from API results
        temp_df = pd.DataFrame.from_records(ls_geocodes)

        # concatenate dataframe
        df_all_geocodes = pd.concat([df_all_geocodes, temp_df])

        print(f"Iteration: {count_iter}, Offset: {int_offset}, Limit: {int_limit}, Number of keys: {len(ls_geocodes)}")

        # increment the offset by the length of the limit
        int_offset += int_limit
        
        # increment iterations
        count_iter += 1
        
        # exit statement
        count_geocodes = len(ls_geocodes)
    
    # convert to lower-case to make merging easier
    df_all_geocodes['county'] = df_all_geocodes['county'].str.lower()
    df_all_geocodes['state'] = df_all_geocodes['state'].str.lower()

    # concatenate county + state
    df_all_geocodes['county_state'] = df_all_geocodes['county'] + '_' + df_all_geocodes['state']

    print()
    print(f"Shape of dataframe: {df_all_geocodes.shape}\n")
    print(f"Number of unique values per column: \n{df_all_geocodes.nunique()}")
    print()
    print(f"Number of Unique `county_state` values: {df_all_geocodes['county_state'].nunique()}")
    print()
    print(f"Total `county_state` values: {df_all_geocodes['county_state'].shape[0]}")
    print()
    print(f"Value counts head of `county_state`: ")
    print(df_all_geocodes['county_state'].value_counts().head())
    print()
    print(f"Minimum latitude: {df_all_geocodes['latitude'].min()}")
    print(f"Maximum latitude: {df_all_geocodes['latitude'].max()}")
    print(f"Minimum longitude: {df_all_geocodes['longitude'].min()}")
    print(f"Maximum longitude: {df_all_geocodes['longitude'].max()}")
    print()
    
    return df_all_geocodes


# In[16]:


get_ipython().run_cell_magic('time', '', "'''\nCPU times: user 3.82 s, sys: 176 ms, total: 4 s\nWall time: 23 s\n'''\n\ndf_all_geocodes = get_county_geocode_data()\n\ndf_all_geocodes.head()")


# In[17]:


# count the differences: how many in covid19 dataset but not in geocodes
st_geocodes_county_state = set(df_all_geocodes['county_state'])
st_covid19_county_state = set(df_covid19_merge['county_state'])

print(f"Number of counties in covid19 but not in geocodes: {len(st_covid19_county_state - st_geocodes_county_state)}")


# In[18]:


df_all_geocodes.head()


# In[19]:


# create smaller dataframe for merging
df_all_geocodes_excerpt = df_all_geocodes[['latitude', 'longitude', 'county_state']]


# In[20]:


# convert drop duplicates of `county_state` and .set_index()
di_all_geocodes_index = df_all_geocodes_excerpt.drop_duplicates('county_state').set_index('county_state')

di_all_geocodes_index.head()


# In[21]:


# just sample one record
di_all_geocodes_index.loc['suffolk_ny']


# In[ ]:





# ### Step 5: final merge COVID-19 and Geocodes 
# * The COVID-19 dataset has duplicates of `county` and `state`, which slightly different `fips`, `latitude`, and `longitude` ... but rest of data the same
# * Run a test for a subset of the columns, to count the number of unique values per county (e.g. are the number of deaths the same for all duplicate rows of a given county?)
#     1. Test if we can drop duplicates
#     1. If so, then drop duplicates

# In[22]:


# merge on geocodes to get latitude and longitude
df_covid19_final = df_covid19_merge.merge(df_all_geocodes_excerpt, on='county_state')

print(f"Shape of dataframe: {df_covid19_final.shape}\n")
print(f"Number of unique values per column: \n{df_covid19_final.nunique()}\n")

df_covid19_final.head()


# In[23]:


# confirm that the numbers are unique per county
ls_str_col = ['fips', 'cases', 'deaths', 'confirmed_cases', 'confirmed_deaths', 'probable_cases', 'probable_deaths']
for each_col in ls_str_col:
    temp_df = df_covid19_final.groupby('county_state')[each_col].nunique().to_frame().reset_index().sort_values(by=[each_col, 'county_state'], ascending=False).head()
    
    print(f"Column: {each_col}")
    print(temp_df)
    print()


# In[24]:


# since we've confirmed the numbers are unique per county_state, we can drop duplicates
df_covid19_drop = df_covid19_final.drop_duplicates(subset='county_state').reset_index(drop=True)

print(f"Shape of dataframe: {df_covid19_drop.shape}\n")
print(f"Number of unique values per column: \n{df_covid19_drop.nunique()}\n")

df_covid19_drop.head()


# In[25]:


df_covid19_drop['county_state'].value_counts().head()


# In[26]:


# how many counties per state
df_covid19_drop['state'].value_counts()


# In[27]:


# ========================================================================
# Run a simple calculation of unique combinations of counties per state.
# Each combination pair of counties will include a distance calculation.
# ========================================================================

# store the number of combinations of each state counties
int_count_combinations = 0

# run a for-loop of each state
for each_state in df_covid19_drop['state'].unique():
    
    # filter dataframes by state
    df_each_state = df_covid19_drop.query(" state==@each_state ")
    
    # count the number of combinations in this filtered dataframe
    num_combinations = len(list(combinations(range(df_each_state.shape[0]), 2)))
    
    # add to the total count of combinations
    int_count_combinations += num_combinations
    
    print(f"Dataframe shape for {each_state} is {df_each_state.shape}")
    
print()
print("#####################################")
print("Number of combinations: ", int_count_combinations)
print("#####################################")


# In[ ]:





# ### Step 6: calculate distance between every combination of county_state (start and end)
# * Limit the number of combinations of counties only within the same state: 145,943 combinations
# * Otherwise, the combinatorial scale is too compute-intensive to calculate distance between all counties across states

# In[28]:


def calc_distance_per_df(idx_start, idx_end, df, flag_print=False):
    '''
    Calculate the distance between two counties in the `df_covid19_drop` dataframe,
    based on the dataframe index.
    
    Dependencies:
        * geopy - Python library for calculating geographic distance.
        * df - custom dataframe for this exercise, 
            (a) assumes unique index
            (b) assumes column named `county_state`
            (c) assumes column named `latitude`
            (d) assumes column named `longitude`            
        
    Return:
        start_county - str, name of county and state
        end_county - str, name of county and state
        distance_miles_clean - float, rounded to 2 decimal points
    '''
    
    # retrieve the start county
    start_county = df.loc[idx_start, 'county_state']
    start_latitude = df.loc[idx_start, 'latitude']
    start_longitude = df.loc[idx_start, 'longitude']
    start_geocode = (start_latitude, start_longitude)

    # retrieve the end county
    end_county = df.loc[idx_end, 'county_state']
    end_latitude = df.loc[idx_end, 'latitude']
    end_longitude = df.loc[idx_end, 'longitude']
    end_geocode = (end_latitude, end_longitude)
    
    # calculate the distance in miles, rounded 2 decimal points
    distance_miles_raw = distance.distance(start_geocode, end_geocode).miles
    distance_miles_clean = float(round(distance_miles_raw, 2))
    
    if flag_print:
        print('Start: ', start_county, start_geocode)
        print('End: ', end_county, end_geocode)
        print('Distance in miles: ', distance_miles_clean)
    
    return start_county, end_county, distance_miles_clean


# In[29]:


# run a sample for calculating the distance between 2 counties
start_county, end_county, distance_miles_clean = calc_distance_per_df(0, 1, df_covid19_drop, True)


# In[30]:


get_ipython().run_cell_magic('time', '', '\'\'\'\nNumber of records in list of county combinations:  145943\nShape of dataframe with distance between counties:  (145943, 3)\n\nCPU times: user 35.8 s, sys: 72 ms, total: 35.9 s\nWall time: 35.9 s\n\'\'\'\n\n# =====================================================\n# Create dataframe of distance between all counties.\n# =====================================================\n\n\n# store distances between counties\nls_counties_distance = []\n\nfor each_state in tqdm(df_covid19_drop[\'state\'].unique()):\n    \n    # filter dataframes by state\n    df_each_state = df_covid19_drop.query(" state==@each_state ")\n    print(each_state, df_each_state.shape)\n    \n    # All possible pairs in dataframe, using combinations() \n    ls_shape_df = list(combinations(df_each_state.index, 2))\n    \n    # this for-loop is on a per-state basis\n    for each_tuple in ls_shape_df:\n        \n        tuple_result = calc_distance_per_df(each_tuple[0], each_tuple[1], df_covid19_drop)\n        ls_counties_distance.append(tuple_result)\n\n# create a dataframe of distances between counties\ndf_counties_distance = pd.DataFrame.from_records(ls_counties_distance)\ndf_counties_distance.columns = [\'start_county\', \'end_county\', \'distance_miles\']\n\nprint("Number of records in list of county combinations: ", len(ls_counties_distance))\nprint("Shape of dataframe with distance between counties: ", df_counties_distance.shape)\nprint()\n\ndf_counties_distance.head()')


# In[31]:


df_counties_distance['start_county'].value_counts()


# In[32]:


# ============================================================
# The distance between combinations of counties is same,
# whether starting at one and ending at other, or vice versa.
# Copy dataframe and reverse direction.
# ============================================================

# create reverse copy of dataframe with distances between counties
df_counties_distance_reverse = df_counties_distance.copy()
df_counties_distance_reverse.columns = ['end_county', 'start_county', 'distance_miles']
df_counties_distance_reverse = df_counties_distance_reverse[['start_county', 'end_county', 'distance_miles']]

print("Shape of distance reverse dataframe: ", df_counties_distance_reverse.shape)
print(df_counties_distance_reverse.head())
print()

# merge the dataframe of distances between counties
df_counties_distance_merge = pd.concat([df_counties_distance, df_counties_distance_reverse])

print("Shape of distance merge dataframe: ", df_counties_distance_merge.shape)

df_counties_distance_merge.head()


# In[33]:


df_counties_distance_merge.query(" start_county == 'baldwin_al' ").sort_values("distance_miles")


# In[ ]:





# ### Step 7: final application
# * Show how many active cases of Covid or Covid-related deaths are currently reported within any given radius from any given US county.
# * Input: US county name (string)
# * Input: radius in miles (integer)
# * Input: statistic ("confirmed_cases" or "confirmed_deaths")

# In[34]:


# =====================================================
# Input: US county name and state
# =====================================================

# input county name, convert to lowercase
in_county_name = input("Enter county name: ").lower()
print()
assert len(in_county_name) > 0

# input state abbreviation, convert to lowercase
in_state_abbrev = input("Enter two-letter state abbreviation: ").lower()
print()
assert len(in_state_abbrev) == 2

# concatenate county and state
str_county_state = f"{in_county_name}_{in_state_abbrev}"

# try to match the input county in list of counties
mask_match_county = df_counties_distance_merge['start_county'].str.contains(str_county_state)
df_match_county = df_counties_distance_merge.loc[mask_match_county]
assert df_match_county['start_county'].nunique()==1

# =====================================================
# Input: radius in miles (integer)
# =====================================================

# input radius in miles
in_radius_miles = float(input("Enter radius in miles: "))
print()
assert in_radius_miles > 0

# =====================================================
# Input: statistic ("confirmed_cases" or "confirmed_deaths")
# =====================================================

# input "confirmed_cases" or "confirmed_deaths"
input_cases_deaths_raw = input("Enter either (A) for confirmed cases, or (B) for confirmed deaths: ").lower()
input_cases_deaths = input_cases_deaths_raw.replace("(", "").replace(")", "")
print()
assert input_cases_deaths in ('a', 'b')
if input_cases_deaths == 'a':
    str_col_covid_stats = 'confirmed_cases'
else:
    str_col_covid_stats = 'confirmed_deaths'
    

# =====================================================
# Match list of counties within distance radius
# =====================================================

# filter on counties within radius distance in miles
df_match_radius = df_match_county.query(" distance_miles < @in_radius_miles ").sort_values("distance_miles")
    
# create list of counties within distance radius
ls_counties_within_radius = df_match_radius['end_county']

# filter covid-19 data
mask_covid_match_radius = df_covid19_drop['county_state'].isin(ls_counties_within_radius)
df_covid_match_radius = df_covid19_drop.loc[mask_covid_match_radius, ['date', 'county_state', str_col_covid_stats]]

print(f"Here is a table of counties within {in_radius_miles} mile radius of {in_county_name.title()} county in {in_state_abbrev.upper()}")
print()
print(df_match_radius.merge(df_covid_match_radius, left_on='end_county', right_on='county_state').drop(columns=['county_state']))
print()


# In[ ]:




