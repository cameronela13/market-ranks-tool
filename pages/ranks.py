# CRE Market Rankings Application
# Interactive Streamlit GUI -- Ranks Page

# This file displays actual and predictive outcomes for markets of a given property type.
# If the current quarter is selected, only predictions will be shown.

import streamlit as st
import numpy as np
import pandas as pd


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


## property selectbox  ##
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
    

# time period selectbox
def period_select(period, avail_time):
    # if a period has been selected
    if period:
        # index used to show current selection in the selectbox
        index = int(np.where(avail_time == period)[0][0])
        st.selectbox('**Time Period**',
                     (avail_time),
                     index=index,
                     key='time_dummy')
    # no period has been selected
    else:
        st.selectbox('**Time Period**',
                    (avail_time),
                    index=None,
                    placeholder='Select...',
                    key='time_dummy')


# get market and property type
def get_selections():

    # if page has been called at least once, session variables will be updated
    if 'property_dummy' in st.session_state:
        property_select(st.session_state['property_dummy'], st.session_state['Types']) 
    else:
        property_select(st.session_state['Property'], st.session_state['Types'])

    # if the property type has changed
    if st.session_state['property_dummy'] != st.session_state['Property']:
        st.session_state['Market 1'] = None
        st.session_state['Market 2'] = None

    # if a property has been selected
    if st.session_state['property_dummy']:
        if 'time_dummy' in st.session_state:
            st.session_state['Time Period'] = st.session_state['time_dummy']
        latest = st.session_state['Latest Forecast'][st.session_state['property_dummy']][1]
        avail_time = st.session_state['Time Range'][st.session_state['property_dummy']][:]
        avail_time.append(latest)
        avail_time = np.array(avail_time)
        period_select(st.session_state['Time Period'], avail_time)

    st.session_state['Property'] = st.session_state['property_dummy']


# display market rankings by property type and time period
def ranking_table(property, period):

    # set local variables as session for easy reuse
    markets = st.session_state['Markets'][property]
    rank_cols = st.session_state['Time Range'][property]
    forecast_cols = rank_cols[-3:]
    forecast_cols.append(st.session_state['Latest Forecast'][property][1])

    # display for quarterly data where actual performance data exists
    if period not in forecast_cols:
        
        # print subheader and label definitions
        st.caption('**True Outperformance**: predicted outperformance, actual outperformance')
        st.caption('**False Outperformance**: predicted outperformance, actual underperformance')
        st.caption('**True Underperformance**: predicted underperformance, actual underperformance')
        st.caption('**False Underperformance**: predicted underperformance, actual outperformance')
        st.subheader(f'**{period} - {str(int(period[:4]) + 3) + period[-2:]}**, *{property}*', anchor=False)

        # create dataframe for period's rankings
        ranks_period = markets.loc[:, 'RANK_' + str(int(period[0:4]) + 1)]
        labels_bool = markets.loc[:, 'CORRECT_' + str(int(period[0:4]) + 1)]
        rankings = pd.DataFrame({'Market': markets['Market'],
                                 'Actual Rank': ranks_period})
        
        # add True/False Out/Underperformance labels
        for i in markets.index:
            half_mkts = int(np.ceil(len(st.session_state['Markets'][st.session_state['Property']]) / 2))
            rank = ranks_period[i]
            label = ''
            # if prediction and actual do not match
            if labels_bool[i] == 0:
                label += 'False'
                if rank <= half_mkts:
                    label += ' Underperformance'
                else:
                    label += ' Outperformance'
            # if prediction and actual match
            else:
                label += 'True'
                if rank <= half_mkts:
                    label += ' Outperformance'
                else:
                    label += ' Underperformance'   
            rankings.loc[i, 'Label'] = label 

        # display the dataframe
        st.dataframe(rankings, column_config={
            'Market': st.column_config.Column(
                width=None
            )
            },
            width=250*len(rankings.columns),
            height=550,
            hide_index=True)
    
    # displaying forecasting only data
    else:
        st.subheader(f'**{period} - {str(int(period[:4]) + 3) + period[-2:]} (forecast)**, *{property}*', anchor=False)
        ranks_period = []
        if period != forecast_cols[-1]:
            ranks_period = markets.loc[:, 'RANK_' + str(int(period[0:4]) + 1)]
        else:
            ranks_period = st.session_state['Latest Forecast'][property][0].loc[:, 'Rank']
        rankings = pd.DataFrame({'Market': markets['Market'],
                                'Forecasted Rank': ranks_period})    

        # display the dataframe
        st.dataframe(rankings,
            width=300*len(rankings.columns),
            height=550,
            hide_index=True)        


# format ranks page
def ranks_page():
    st.logo(image='pages/logo.png', link=None)

    nav_menu()
    st.title('NASA Market Rankings', anchor=False)

    # pick property type and time period
    get_selections()

    # visual page divider
    st.divider()
    
    # assign widget keys
    if st.session_state['property_dummy'] and st.session_state['time_dummy']:
        ranking_table(st.session_state['property_dummy'], st.session_state['time_dummy'])
    
    return 0


# required be streamlit API to be first code called before any other streamlit code
st.set_page_config(page_title="Single Market View", page_icon='pages/Invesco_Global_Favicon_Blue_Pos_RGB_32px.png', layout='wide')

# handles session state cache being cleared
if 'Types' not in st.session_state:
    st.write("""<meta http-equiv="refresh" content="0; url='/'">""", unsafe_allow_html=True)
# load ranks page
ranks_page()
