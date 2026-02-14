# Market Rankings Application
# Interactive Streamlit GUI

# This file provides an interactive interface displaying historical market rankings and forecasts. Users may query data on markets, property types, and quarterly time periods.
# Machine learning modeling is provided. Some code is disabled due to data sensitivity. Replacement code (if any) can be found below each instance of disabled code.

import streamlit as st
import numpy as np
import pandas as pd
from snowflake.connector import connect


CONNECTION_PARAMETERS = {
    "account": 'REDACTED',
    "user": 'REDACTED',
    "password": 'REDACTED',
    "database": 'REDACTED',
    "warehouse": 'REDACTED'
}


## sidebar navigation menu ##
def nav_menu():
    # Link to Python files in the pages folder
    page_labels = ['Home', 'Single Market View', 'Dual Market View', 'Filtered Market Rankings', 'Modeling Information']
    pages = {
        page_labels[0]: "main.py",
        page_labels[1]: "pages/mkt_view.py",
        page_labels[2]: "pages/dual_view.py",
        page_labels[3]: "pages/ranks.py",
        page_labels[4]: "pages/info.py"
    }
    for label in page_labels:
        st.sidebar.page_link(pages[label], label=f'{label}')


## fetch excel data for a single property type. Accepts a list of excel paths for one property type (DISABLED) ##
# @st.cache_data
# def get_excel(list_excel_paths):
#     # list of DataFrames from excel files. Files with multiple sheets will return a dictionary with DataFrames as values
#     data = []
#     for path in list_excel_paths:
#         data.append(pd.read_excel(path, sheet_name=None))
    
#     return data
###################################################################################################################


# get NASA industrial model data
@st.cache_data
def get_industrial():

    ########## industrial excel file paths (DISABLED) ############
    # xcel_ranks_ovtm = './data/ranktime_RENT_WH_OVERPERFORM.xlsx'
    # xcel_current_ranks = './data/Industrial Rankings 2024Q1.xlsx'
    # industrial_paths = [xcel_ranks_ovtm, xcel_current_ranks]
    # industrial_data = get_excel(industrial_paths)

    # # modify first industrial sheet: historical rankings and forecasts ##
    # wh_ranks_ovtm = pd.DataFrame(industrial_data[0]['ranktime_RENT_WH_OVERPERFORM'])
    # # define columns to keep
    # col_keep = [wh_ranks_ovtm.columns[0]]
    # col_keep.extend([col for col in wh_ranks_ovtm.columns if (('RANK_' in col and 'CHANGE' not in col) or 'CORRECT_' in col)])
    # # add accuracy per market
    # col_keep.append(wh_ranks_ovtm.columns[-1])

    # # keep columns relevant to display and sort by market
    # wh_ranks_ovtm = wh_ranks_ovtm.drop(columns=set(wh_ranks_ovtm.columns) - set(col_keep))
    # wh_ranks_ovtm = wh_ranks_ovtm.rename(columns={'CBSA_DISPLAY_NAME': 'Market'})
    # wh_ranks_ovtm = wh_ranks_ovtm.sort_values('Market')
    ###############################################################

    ## REPLACEMENT: create artificial industrial market rankings data ##
    industrial_mkts = sorted(['Riverside', 'Charlotte', 'Oakland', 'Orange County', 'Nashville', 'Los Angeles',
                       'Central New Jersey', 'Raleigh', 'Memphis', 'Northern New Jersey', 'Salt Lake City',
                       'Jacksonville', 'Wilmington, DE', 'Seattle', 'Tampa', 'Orlando', 'Miami', 'Dallas',
                       'Houston', 'San Francisco', 'Columbus', 'Las Vegas', 'Austin', 'Atlanta', 'Indianapolis',
                       'Baltimore', 'San Diego', 'Washington, DC', 'Chicago', 'Allentown', 'Fort Worth',
                       'San Jose', 'Minneapolis', 'Fort Lauderdale', 'Vallejo', 'Boston', 'West Palm Beach',
                       'St. Louis', 'Denver', 'Sacramento', 'Philadelphia', 'Ventura', 'Portland', 'Cincinnati',
                       'Hartford', 'Tucson', 'Kansas City', 'Long Island', 'Phoenix', 'Albuquerque', 'Dayton',
                       'Stamford', 'Detroit', 'Cleveland', 'Pittsburgh', 'Toledo'])
    wh_ranks_ovtm = pd.DataFrame({'Market': industrial_mkts})

    # generate random ranks (column order goes all RANK_ columns before CORRECT_ columns)
    for year in range(2014, 2025):
        wh_ranks_ovtm.loc[:, f'RANK_{str(year)}'] = pd.Series(np.random.choice(np.arange(1, len(industrial_mkts) + 1), size=len(industrial_mkts), replace=False))
    # generate random correct booleans
    for year in range(2014, 2022):
        wh_ranks_ovtm.loc[:, f'CORRECT_{str(year)}'] = pd.Series(np.random.choice([0, 1], size=len(industrial_mkts), replace=True))
    # calculate accuracy for each market
    accuracy = []
    correct_cols = [col for col in wh_ranks_ovtm.columns if 'CORRECT_' in col]
    for row in wh_ranks_ovtm.index:
        acc = wh_ranks_ovtm.loc[row, correct_cols].mean()
        accuracy.append(acc)
    wh_ranks_ovtm.loc[:, 'AVG_CORRECT'] = accuracy
    ###################################################################

    # ## modify second industrial sheet: latest forecast (DISABLED) ##
    # latest_forecast = pd.DataFrame(industrial_data[1]['Warehouse Rankings'])
    # latest_forecast_period = str(latest_forecast.loc[0, 'DATE'].year) + 'Q' + str(latest_forecast.loc[0, 'DATE'].quarter)
    # col_keep = ['Market', 'Rank']
    # latest_forecast = latest_forecast.drop(columns=set(latest_forecast.columns) - set(col_keep))
    # latest_forecast = latest_forecast.sort_values('Market')

    # # insert column for boolean outcomes
    # threshold = np.ceil(len(latest_forecast) / 2)
    # latest_forecast.loc[:, 'OUTPERFORM_PREDICTED'] = ['Outperform' if pred <= threshold else 'Underperform' for pred in latest_forecast.loc[:, 'Rank']]

    # ## skip CBRE cap rate v. ranking sheet ##
    ###################################################################

    ## REPLACEMENT: create artificial latest forecast data ##
    latest_ranks = pd.Series(np.random.choice(np.arange(1, len(industrial_mkts) + 1), size=len(industrial_mkts), replace=False))
    threshold = np.ceil(len(latest_ranks) / 2)
    outperform_predicted = ['Outperform' if pred <= threshold else 'Underperform' for pred in latest_ranks.values]
    latest_forecast = pd.DataFrame({'Rank': latest_ranks,
                                    'Market': industrial_mkts,
                                    'OUTPERFORM_PREDICTED': outperform_predicted})
    latest_forecast_period = '2024Q1'
    ##########################################################

    ## modify fourth industrial sheet: model (DISABLED)###
    # model = pd.DataFrame(industrial_data[1]['Model'])
    # # create single confidence interval column
    # model.loc[:, 'Confidence Interval'] = [[model.loc[row, 'Conf. Int. Low'], model.loc[row, 'Conf. Int. High']] for row in range(len(model))]
    # model = model.rename(columns={'Unnamed: 0': 'Variable'})
    # model = model.drop(columns=['Conf. Int. Low', 'Conf. Int. High'])
    #######################################################

    ## REPLACEMENT: create fake model ###
    variables = ['UNEMPLOYMENT_RATE', 'INT_RATE_PCT_CHNG', 'PCT_CHNG_NET_EXP', 'IS_COASTAL', 'CHNG_NASDAQ_100']
    coefficients = (np.random.rand(5) - 0.5) * 20
    conf_int = [[coef - np.random.rand() * 3, coef + np.random.rand() * 3] for coef in coefficients]
    model = pd.DataFrame({'Variable': variables,
                          'Coefficient': coefficients,
                          'Confidence Interval': conf_int})
    ######################################

    ## modify fifth industrial sheet: backtesting accuracy metrics (DISABLED) ##
    # backtesting = pd.DataFrame(industrial_data[1]['Backtesting'])
    # backtesting = backtesting.drop(columns='Unnamed: 3')
    # backtesting = backtesting.drop(index=8)
    #############################################################################

    ## REPLACEMENT: calculate backtesting accuracy from fake data ##
    correct_col = [col for col in wh_ranks_ovtm.columns if 'CORRECT_' in col]
    nan_arr = np.full(len(correct_col) + 1, np.nan)
    backtesting = pd.DataFrame({'Latest data': np.array_str(nan_arr),
                                '% of top half of rankings that actually outperformed rent benchmark': nan_arr,
                                '% of bottom half that actually underperformed': nan_arr,
                                'Average': nan_arr})
    backtesting.loc[:, 'Latest data'] = backtesting.loc[:, 'Latest data'].astype('str')
    backtesting.loc[list(backtesting.index[:-1]), 'Latest data'] = [str(int(col[-4:]) - 1) + 'Q4' for col in correct_col]
    backtesting.loc[backtesting.index[-1], 'Latest data'] = 'Average'

    # fill in % outperformance/underperformance for top/bottom half
    for row in backtesting.index[:-1]:
        year_sorted_ranks = np.array(wh_ranks_ovtm.sort_values('RANK_' + correct_col[row][-4:]).loc[:, correct_col[row]])
        backtesting.iloc[row, 1] = 100 * np.mean(year_sorted_ranks[: int(np.ceil(len(wh_ranks_ovtm) / 2))])
        backtesting.iloc[row, 2] = 100 * np.mean(year_sorted_ranks[int(np.ceil(len(wh_ranks_ovtm) / 2)):])
        backtesting.iloc[row, 3] = np.mean(np.array([backtesting.iloc[row, 1], backtesting.iloc[row, 2]]))
    # get average of each column
    backtesting.iloc[-1, 1] = backtesting.iloc[: -1, 1].mean()
    backtesting.iloc[-1, 2] = backtesting.iloc[: -1, 2].mean()
    backtesting.iloc[-1, 3] = backtesting.iloc[: -1, 3].mean()
    #################################################################

    ## return dictionary of industrial information ##
    industrial = {
        'historical rankings': wh_ranks_ovtm,
        'latest forecast': [latest_forecast, latest_forecast_period],
        'model': model,
        'backtesting': backtesting
    }
       
    return industrial


# get NASA apartment model data (similar to industrial function)
@st.cache_data
def get_apartment():

    # ## apartment excel file paths (DISABLED) ##
    # xcel_ranks_ovtm = './data/RENT_OVERPERFORMranktime_df.xlsx'
    # xcel_current_ranks = './data/RENT_OVERPERFORMfinal_rankings.xlsx'
    # xcel_model_summary = './data/RENT_OVERPERFORMsummary.xlsx'
    # apartment_paths = [xcel_ranks_ovtm, xcel_current_ranks, xcel_model_summary]
    # apartment_data = get_excel(apartment_paths)

    # ## modify ranks overtime dataframe ##
    # apt_ranks_ovtm = pd.DataFrame(apartment_data[0]['RENT_OVERPERFORMranktime_df'])
    # # define columns to keep
    # col_keep = [apt_ranks_ovtm.columns[0]]
    # col_keep.extend([col for col in apt_ranks_ovtm.columns if (('RANK_' in col and 'CHANGE' not in col) or 'CORRECT_' in col)])
    # # add accuracy per market
    # col_keep.append(apt_ranks_ovtm.columns[-1])     

    # # keep columns relevant to display and sort by market
    # apt_ranks_ovtm = apt_ranks_ovtm.drop(columns=set(apt_ranks_ovtm.columns) - set(col_keep))
    # apt_ranks_ovtm = apt_ranks_ovtm.rename(columns={'CBSA_DISPLAY_NAME': 'Market'})
    # apt_ranks_ovtm = apt_ranks_ovtm.sort_values('Market')
    # #############################################

    ## REPLACEMENT: create artificial industrial market rankings data ##
    apartment_mkts = sorted(['San Francisco', 'Houston', 'West Palm Beach', 'Lexington', 
                              'Nashville', 'Chicago', 'Washington, DC', 'Omaha', 'New York', 
                              'Oklahoma City', 'San Jose', 'Dallas', 'Orlando', 'Tulsa', 
                              'Columbus', 'Fort Lauderdale', 'Boston', 'Denver', 'Seattle', 
                              'Indianapolis', 'Minneapolis', 'Corpus Christi', 'Kansas City', 
                              'Greenville', 'Miami', 'Raleigh', 'Cincinnati', 'Louisville', 
                              'Austin', 'Hartford', 'San Antonio', 'Fort Worth', 'Philadelphia', 
                              'El Paso', 'Oakland', 'Northern New Jersey', 'Pittsburgh', 
                              'Providence', 'Dayton', 'Las Vegas', 'Los Angeles', 'St. Louis', 
                              'Tampa', 'Charlotte', 'Cleveland', 'Baltimore', 'Salt Lake City', 
                              'Atlanta', 'Norfolk', 'Birmingham', 'Orange County', 'Greensboro', 
                              'Richmond', 'Long Island', 'Jacksonville', 'San Diego', 'Detroit', 
                              'Phoenix', 'Tucson', 'Portland', 'Ventura', 'Riverside',
                              'Sacramento', 'Colorado Springs', 'Albuquerque', 'Memphis'])
    apt_ranks_ovtm = pd.DataFrame({'Market': apartment_mkts})

    # generate random ranks (column order goes all RANK_ columns before CORRECT_ columns)
    for year in range(2014, 2025):
        apt_ranks_ovtm.loc[:, f'RANK_{str(year)}'] = pd.Series(np.random.choice(np.arange(1, len(apartment_mkts) + 1), size=len(apartment_mkts), replace=False))
    # generate random correct booleans
    for year in range(2014, 2022):
        apt_ranks_ovtm.loc[:, f'CORRECT_{str(year)}'] = pd.Series(np.random.choice([0, 1], size=len(apartment_mkts), replace=True))
    # calculate accuracy for each market
    accuracy = []
    correct_cols = [col for col in apt_ranks_ovtm.columns if 'CORRECT_' in col]
    for row in apt_ranks_ovtm.index:
        acc = apt_ranks_ovtm.loc[row, correct_cols].mean()
        accuracy.append(acc)
    apt_ranks_ovtm.loc[:, 'AVG_CORRECT'] = accuracy
    ###################################################################

    # ## modify current forecast dataframe (DISABLED) ##
    # latest_forecast = pd.DataFrame(apartment_data[1]['RENT_OVERPERFORMfinal_rankings'])
    # latest_forecast_period = str(latest_forecast.loc[0, 'DATE'].year) + 'Q' + str(latest_forecast.loc[0, 'DATE'].quarter)
    # latest_forecast = latest_forecast.rename(columns={'CBSA_DISPLAY_NAME': 'Market'})
    # latest_forecast.loc[:, 'Rank'] = [i + 1 for i in range(len(latest_forecast))]
    # col_keep = ['Market', 'Rank']
    # latest_forecast = latest_forecast.drop(columns=set(latest_forecast.columns) - set(col_keep))
    # latest_forecast = latest_forecast[['Rank', 'Market']]
    # latest_forecast = latest_forecast.sort_values('Market')

    # # insert column for boolean outcomes
    # threshold = np.ceil(len(latest_forecast) / 2)
    # latest_forecast.loc[:, 'OUTPERFORM_PREDICTED'] = ['Outperform' if pred <= threshold else 'Underperform' for pred in latest_forecast.loc[:, 'Rank']]
    #####################################################

    ## REPLACEMENT: create artificial latest forecast data ##
    latest_ranks = pd.Series(np.random.choice(np.arange(1, len(apartment_mkts) + 1), size=len(apartment_mkts), replace=False))
    threshold = np.ceil(len(latest_ranks) / 2)
    outperform_predicted = ['Outperform' if pred <= threshold else 'Underperform' for pred in latest_ranks.values]
    latest_forecast = pd.DataFrame({'Rank': latest_ranks,
                                    'Market': apartment_mkts,
                                    'OUTPERFORM_PREDICTED': outperform_predicted})
    latest_forecast_period = '2024Q1'
    ##########################################################

    # ## modify model dataframe (DISABLED) ##
    # model = pd.DataFrame(apartment_data[2]['RENT_OVERPERFORMsummary'])
    # # create single confidence interval column
    # model.loc[:, 'Confidence Interval'] = [[float(model.loc[row, 'Conf. Int. Low']), float(model.loc[row, 'Conf. Int. High'])] for row in range(len(model))]
    # model = model.rename(columns={'Unnamed: 0': 'Variable'})
    # model = model.drop(columns=['Conf. Int. Low', 'Conf. Int. High'])
    ###########################################

    ## REPLACEMENT: create fake model ###
    variables = ['POP_RATE_CHNG', 'RENT_INDEX', 'MORTGAGE_RATE', 'IS_URBAN', 'CPI_LESS_FOOD_ENERGY', 'POP_DENSITY']
    coefficients = (np.random.rand(6) - 0.5) * 20
    conf_int = [[coef - np.random.rand() * 3, coef + np.random.rand() * 3] for coef in coefficients]
    model = pd.DataFrame({'Variable': variables,
                          'Coefficient': coefficients,
                          'Confidence Interval': conf_int})
    ######################################

    ## create backtesting accuracy metrics dataframe ##
    correct_col = [col for col in apt_ranks_ovtm.columns if 'CORRECT_' in col]
    # match warehouse ranking format
    nan_arr = np.full(len(correct_col) + 1, np.nan)
    backtesting = pd.DataFrame({'Latest data': np.array_str(nan_arr),
                                '% of top half of rankings that actually outperformed rent benchmark': nan_arr,
                                '% of bottom half that actually underperformed': nan_arr,
                                'Average': nan_arr})
    backtesting.loc[:, 'Latest data'] = backtesting.loc[:, 'Latest data'].astype('str')
    backtesting.loc[list(backtesting.index[:-1]), 'Latest data'] = [str(int(col[-4:]) - 1) + 'Q4' for col in correct_col]
    backtesting.loc[backtesting.index[-1], 'Latest data'] = 'Average'

    # fill in % outperformance/underperformance for top/bottom half
    for row in backtesting.index[:-1]:
        year_sorted_ranks = np.array(apt_ranks_ovtm.sort_values('RANK_' + correct_col[row][-4:]).loc[:, correct_col[row]])
        backtesting.iloc[row, 1] = 100 * np.mean(year_sorted_ranks[: int(np.ceil(len(apt_ranks_ovtm) / 2))])
        backtesting.iloc[row, 2] = 100 * np.mean(year_sorted_ranks[int(np.ceil(len(apt_ranks_ovtm) / 2)):])
        backtesting.iloc[row, 3] = np.mean(np.array([backtesting.iloc[row, 1], backtesting.iloc[row, 2]]))
    # get average of each column
    backtesting.iloc[-1, 1] = backtesting.iloc[: -1, 1].mean()
    backtesting.iloc[-1, 2] = backtesting.iloc[: -1, 2].mean()
    backtesting.iloc[-1, 3] = backtesting.iloc[: -1, 3].mean()

    ## return dictionary of apartment information ##
    apartment = {
        'historical rankings': apt_ranks_ovtm,
        'latest forecast': [latest_forecast, latest_forecast_period],
        'model': model,
        'backtesting': backtesting
    }
       
    return apartment


# property selectbox
def property_select(property, avail_types):
    # if property is not None
    if property:
        # index used for showing current selection in the box
        index = int(np.where(avail_types == property)[0][0])
        st.selectbox('**Property Type**',
                     (avail_types),
                     index=index, 
                     key='property_dummy')
    else:
        st.selectbox('**Property Type**', 
                     (avail_types),
                     index=None,
                     placeholder='Select...',
                     key='property_dummy')


# initializes session state keys for a given property type based on NASA data
def nasa_init(property, data):

    # initialize available markets and their data
    market_data = data['historical rankings']
    if 'Markets' not in st.session_state:
        st.session_state['Markets'] = {
            property: market_data
        }
    else:
        st.session_state['Markets'][property] = market_data

    # initialize time range
    time_range = [str(int(col[-4:]) - 1) + 'Q4' for col in market_data.columns if 'RANK' in col]
    if 'Time Range' not in st.session_state:
        st.session_state['Time Range'] = {
            property: time_range
        }
    else:
        st.session_state['Time Range'][property] = time_range

    # initialize latest forecasting data
    latest_forecast = data['latest forecast']
    if 'Latest Forecast' not in st.session_state:
        st.session_state['Latest Forecast'] = {
            property: latest_forecast
        }
    else:
        st.session_state['Latest Forecast'][property] = latest_forecast

    # intialize backtesting accuracy
    acc = data['backtesting']
    if 'Accuracy' not in st.session_state:
        st.session_state['Accuracy'] = {
            property: acc
        }
    else:
        st.session_state['Accuracy'][property] = acc

    # initialize model statistics
    model = data['model']
    if 'Models' not in st.session_state:
        st.session_state['Models'] = {
            property: model
        }
    else:
        st.session_state['Models'][property] = model


# retrieve snowflake data (CBRE, Green Street)
@st.cache_data
def query_snowflake(query):

    # establish connection to snowflake
    conn = connect(**CONNECTION_PARAMETERS)
    curs = conn.cursor()
    
    # execute query
    try:
        curs.execute(query)
        rows = curs.fetchall()
        field_names = [i[0] for i in curs.description]
    finally:
        curs.close()
        
    conn.close()
    
    # retrieve query results
    data = pd.DataFrame(rows)
    data.columns = field_names
    data.columns = data.columns.str.lower()

    return data


# get cleaned and NASA formatted forecasting data from CBRE
# nasa_data: dictionary with keys as available property types from NASA's end
# period: latest forecasting period from NASA's end (should be expected as the previous quarter)
@st.cache_data
def get_cbre(nasa_data, period):
    prop_type_keys = list(nasa_data.keys())
    ############## DISABLED ###############################################
    # # format latest period as a date for use in query
    # latest_forecast_date = period[:-2]
    # if period[-2:] == 'Q1':
    #     latest_forecast_date = pd.to_datetime(latest_forecast_date + '-01-01')
    # elif period[-2:] == 'Q2':
    #     latest_forecast_date = pd.to_datetime(latest_forecast_date + '-03-01')
    # elif period[-2:] == 'Q3':
    #     latest_forecast_date = pd.to_datetime(latest_forecast_date + '-07-01')
    # else:
    #     latest_forecast_date = pd.to_datetime(latest_forecast_date + '-10-01')

    # # retrieve CBRE data from snowflake - available annual 4th quarter data
    # cbre_query = f'''
    # REDACTED
    # '''
    # cbre_data = query_snowflake(cbre_query)
    # cbre = pd.DataFrame(cbre_data)
    #######################################################################
    
    # separate into dataframes based on property_type
    cbre_dict = {}
    for key in prop_type_keys:
        ######################## DISABLED ##############################
        # cbre_dict[key] = cbre[cbre['property_type'] == key]
        # cbre_dict[key] = cbre_dict[key].drop(columns='property_type')

        # # filter to rows where market in NASA data
        # cbre_dict[key] = cbre_dict[key][cbre_dict[key]['market_name'].isin(nasa_data[key]['Market'])]

        # # create columns for each unique date and create new dataframes to match NASA format
        # unique_mkts = cbre_dict[key]['market_name'].unique()
        # cbre_dict[key]['date'] = pd.to_datetime(cbre_dict[key]['date'])
        # unique_dates = np.sort(cbre_dict[key]['date'].dt.year.unique())

        # # create new dataframe to hold reformatted data
        # new_df = pd.DataFrame({'Market': unique_mkts})
        # for col in unique_dates:
        #     new_df[col] = None

        # # insert rent index into respective year column and market row
        # for row in cbre_dict[key].index:
        #     year = cbre_dict[key].loc[row, 'date'].year
        #     if year in new_df.columns:
        #         mkt = new_df.loc[new_df['Market'] == cbre_dict[key].loc[row, 'market_name']]
        #         new_df.loc[mkt.index[0], year] = cbre_dict[key].loc[row, 'rent_index_amt']

        # # create rankings from rent index
        # for col in new_df.columns[1:]:
        #     new_df = new_df.sort_values(col)
        #     exists_idx = new_df[new_df[col].notnull()].index
        #     counter = 0
        #     for i in exists_idx:
        #         new_df.loc[i, col] = len(exists_idx) - counter
        #         counter += 1

        # # adjust year columns to match NASA format
        # mapper = dict([(col, f'RANK_{str(col + 1)}') for col in new_df.columns[1:]])
        # new_df = new_df.rename(columns=mapper)
        # cbre_dict[key] = new_df.sort_values('Market')
        ######################################################

        ## REPLACEMENT: generate random ranks as CBRE data ##
        cbre_dict[key] = pd.DataFrame({'Market': nasa_data[key]['Market']})
        for year in range(2014, 2026):
            cbre_dict[key].loc[:, 'RANK_' + str(year)] = pd.Series(np.random.choice(np.arange(1, len(nasa_data[key]['Market']) + 1), size=len(nasa_data[key]['Market']), replace=False))
        #####################################################

    return cbre_dict


# initialize session state
def session_init():
    st.session_state['Types'] = np.sort(np.array(['Industrial', 'Apartment']))
    
    ## get excel data and initialize session state with NASA data ##
    industrial_dict = get_industrial()
    apartment_dict = get_apartment()
    # implement following functions in the future; add dictionaries to data variable
    # retail_dict = get_retail()
    # office_dict = get_office()
    
    # make sure data is in alphabetical order
    data = [apartment_dict, industrial_dict]
    for i in range(len(data)):
        nasa_init(st.session_state['Types'][i], data[i])
    st.session_state['Markets'] = dict(sorted(st.session_state['Markets'].items()))

    # store the latest forecasting period available
    latest_period = industrial_dict['latest forecast'][1]

    ## initialize CBRE snowflake data ###
    st.session_state['Snowflake'] = {
        'CBRE': get_cbre(st.session_state['Markets'], latest_period)
        }
    
    ## initializing dynamic session variables ##
    st.session_state['Property'] = None
    st.session_state['Time Period'] = None
    st.session_state['Market 1'] = None
    st.session_state['Market 2'] = None


################ APP MAIN FUNCTION ######################
def main():
    st.set_page_config(page_title="NASA Market Rankings Tool", page_icon='pages/Invesco_Global_Favicon_Blue_Pos_RGB_32px.png', layout='wide')
    st.logo(image='pages/logo.png', link=None)
    # update application parameters
    app_update = {'date': 7, 'month': 'August', 'year': 2024}

    ## initialize available property types ##
    if 'Types' not in st.session_state:
        session_init()

    ############## Home page #####################
    st.title('Market Rankings Tool', anchor=False)    

    ## app information ##
    st.caption(f'Available Property Types: {", ".join(str(i) for i in st.session_state["Types"])}')
    # shows available backtesting data
    avail_time = []
    # case when home page is ran more than once
    if 'property_dummy' in st.session_state:
        if st.session_state['property_dummy']:
            avail_time.append(st.session_state["Time Range"][st.session_state['property_dummy']][0])
            avail_time.append(st.session_state["Time Range"][st.session_state['property_dummy']][-1])
            st.caption(f'Available Backtesting Data: {avail_time[0]} - {avail_time[1]}')
    # case when page is first ran from start up or after switching pages
    elif st.session_state['Property']:           
        avail_time.append(st.session_state["Time Range"][st.session_state['Property']][0])
        avail_time.append(st.session_state["Time Range"][st.session_state['Property']][-1])
        st.caption(f'Available Backtesting Data: {avail_time[0]} - {avail_time[1]}')
    st.caption(f'Last Application Update: {" ".join(str(app_update[i]) for i in app_update.keys())}')

    # space padding
    st.text('')

    # property selectbox generated
    if 'property_dummy' in st.session_state:
        property_select(st.session_state['property_dummy'], st.session_state['Types']) 
    else:
        property_select(st.session_state['Property'], st.session_state['Types'])

    # if property type is changed
    if st.session_state['property_dummy'] != st.session_state['Property']:
        st.session_state['Market 1'] = None
        st.session_state['Market 2'] = None
    st.session_state['Property'] = st.session_state['property_dummy']

    st.divider()
    nav_menu()

    return 0


if __name__ == '__main__':
    main()
