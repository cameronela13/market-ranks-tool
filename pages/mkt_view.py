# CRE Market Rankings Application
# Interactive Streamlit GUI -- Market View Page

# This file shows how a market's ranking has changed overtime according to NASA, CBRE, and Green Street forecasts.

import streamlit as st
import numpy as np
import pandas as pd
import altair as alt


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
    

##  market selectbox ##
def market_select(market, avail_mkts):
    if market:
        # index used to show current selection in the selectbox
        index = int(np.where(avail_mkts == market)[0][0])
        st.selectbox('**Regional Market**',
                    avail_mkts,
                    index=index,
                    key='mkt_dummy_1')
    else:
        if avail_mkts is not None:
            st.selectbox('**Regional Market**',
                        avail_mkts,
                        index=None, placeholder='Select...',
                        key='mkt_dummy_1')


## get market and property type ##
def get_selections():

    # if page has been called at least once, session variables will be updated
    if 'property_dummy' in st.session_state:
        property_select(st.session_state['property_dummy'], st.session_state['Types']) 
    else:
        property_select(st.session_state['Property'], st.session_state['Types']) 

    # if a property has been selected
    if st.session_state['property_dummy']:
        avail_mkts = st.session_state['Markets'][st.session_state['property_dummy']].loc[:, 'Market']
        # the market selectbox has ran at least once
        if 'mkt_dummy_1' in st.session_state:
            st.session_state['Market 1'] = st.session_state['mkt_dummy_1']
        #  if the property type has changed (not all properties have the same markets)
        if st.session_state['property_dummy'] != st.session_state['Property']:
            st.session_state['Market 1'] = None
            st.session_state['Market 2'] = None
        market_select(st.session_state['Market 1'], avail_mkts)

    st.session_state['Property'] = st.session_state['property_dummy']


# display line chart for a given market for a certain property type
def line_chart(property, market):
    st.subheader(f'**{market}**, *{property}*', anchor=False)
    
    # retrieve market data
    avail_mkts_data = st.session_state['Markets'][property].reset_index(drop=True)
    nasa_row = avail_mkts_data[avail_mkts_data['Market'] == market]

    # show accuracy under market
    accuracy = nasa_row.loc[nasa_row.index[0], 'AVG_CORRECT']
    st.caption(f'NASA Backtesting Accuracy for **{market}**: {accuracy:.3f}')

    # filter for rank columns and correct columns
    rank_range = [i for i in nasa_row.columns if 'RANK_' in i]
    correct_range = [i for i in nasa_row.columns if 'CORRECT_' in i]

    # get nasa ranks and forecasts
    nasa_ranks = nasa_row.loc[:, rank_range].reset_index(drop=True)
    # add forecasted data that has no real data counterpart
    forecast_data = st.session_state['Latest Forecast'][property]
    forecast_rank = forecast_data[0].reset_index(drop=True)
    forecast_rank = forecast_rank[forecast_rank['Market'] == market].iloc[:, 0].reset_index(drop=True)
    nasa_ranks = pd.concat([nasa_ranks, forecast_rank], axis=1).squeeze(axis=0)
    nasa_forecasts = np.array(nasa_row.loc[:, correct_range]).squeeze(axis=0)

    # get cbre ranks
    cbre_ranks = st.session_state['Snowflake']['CBRE'][property]
    cbre_date_range = list(set(rank_range) - (set(rank_range) - set(cbre_ranks.columns)))
    cbre_missing_date = pd.DataFrame(dict([[key, [None]] for key in set(rank_range) - set(cbre_ranks.columns)]))
    cbre_ranks = cbre_ranks[cbre_ranks['Market'] == market][cbre_date_range].reset_index(drop=True)
    cbre_ranks = pd.concat([cbre_ranks, cbre_missing_date], axis=1)
    # add dates where NASA has no real data
    forecast_cbre = st.session_state['Snowflake']['CBRE'][property]
    forecast_cbre = forecast_cbre[forecast_cbre['Market'] == market].loc[:, 'RANK_' + str(int(forecast_data[1][:4]) + 1)].reset_index(drop=True)
    cbre_ranks = pd.concat([cbre_ranks, forecast_cbre], axis=1).squeeze(axis=0).sort_index().reset_index(drop=True)

    # put all backtesting data into DataFrames
    date_range = st.session_state['Time Range'][property][:]
    date_range.append(forecast_data[1])
    line_data = pd.DataFrame({
        'Period': date_range,
        'NASA Rank': nasa_ranks,
        },
        index=None).reset_index(drop=True)
    if len(cbre_ranks) > 0:
        line_data = pd.concat([line_data, pd.DataFrame({'CBRE Rank': cbre_ranks})], axis=1)
    areas = pd.DataFrame({'Period': date_range,
                        'Bottom': np.full(shape=len(date_range), fill_value=int(len(st.session_state['Markets'][property]))),
                        'Half': np.full(shape=len(date_range), fill_value=int(len(st.session_state['Markets'][property]) * 0.5))})

    # color scheme for lines and points
    line_color = {
        'NASA Rank': '#000AD2',
        'CBRE Rank': 'orange',
        'False Prediction': '#D22B2B',
        'True Prediction': '#50C878',
        'NASA Forecast': '#000000'
    }

    # plot lines
    lines = alt.Chart(line_data).transform_fold(line_data.columns[1:].tolist(), as_=['Label', 'Rank']).mark_line().encode(
        x=alt.X('Period:N', scale=alt.Scale(padding=0)),
        y=alt.Y('Rank:Q', scale=alt.Scale(domain=[0, len(st.session_state['Markets'][property])], reverse=True, nice=False)),
        color='Label:N'
        ).properties(height=675, width=2025)

    # plot outperform area
    outperform_area = alt.Chart(areas).mark_area(opacity=0.3, color=alt.Gradient(
        gradient="linear",
        stops=[
        alt.GradientStop(color="lightgray", offset=0),
        alt.GradientStop(color="lime", offset=1),
        ],
        x1=1,
        x2=1,
        y1=1,
        y2=0)
        ).encode(
            x='Period:N',
            y=alt.Y('Half:Q', title='Rank'),
            tooltip=alt.value(None)
            )

    # plot underperform area
    underperform_area = alt.Chart(areas).mark_area(opacity=0.3, color=alt.Gradient(
        gradient="linear",
        stops=[
        alt.GradientStop(color="lightgray", offset=1),
        alt.GradientStop(color="firebrick", offset=0),
        ],
        x1=1,
        x2=1,
        y1=1,
        y2=0)
        ).encode(
            x='Period:N',
            y=alt.Y('Bottom:Q', title='Rank'),
            y2='Half:N',
            tooltip=alt.value(None)
            )

    # create dataframe for nasa points (exclude points without real data)
    nasa_points_df = pd.DataFrame(line_data.loc[:, :'NASA Rank'])
    nasa_points_df.loc[:, 'pred_val'] = pd.Series(nasa_forecasts)
    # create new columns for each class and input ranks according to class
    nasa_points_df.loc[:, 'False Prediction'] = np.full(shape=len(nasa_points_df), fill_value=np.nan)
    nasa_points_df.loc[:, 'True Prediction'] = np.full(shape=len(nasa_points_df), fill_value=np.nan)
    nasa_points_df.loc[:, 'NASA Forecast'] = np.full(shape=len(nasa_points_df), fill_value=np.nan)
    for row in nasa_points_df.index[:-4]:
        rank = nasa_points_df.loc[row, 'NASA Rank']
        col_name = ''
        if nasa_points_df.loc[row, 'pred_val'] == 0:
            col_name += 'False Prediction'
        else:
            col_name += 'True Prediction'
        nasa_points_df.loc[row, col_name] = rank
    for row in nasa_points_df.index[-4:]:
        nasa_points_df.loc[row, 'NASA Forecast'] = nasa_points_df.loc[row, 'NASA Rank']
    # drop all other columns
    nasa_points_df = nasa_points_df.drop(columns=['pred_val', 'NASA Rank'])

    # plot the nasa points
    nasa_points = alt.Chart(nasa_points_df).mark_point(filled=True, size=50, opacity=1).transform_fold(nasa_points_df.columns[1:].tolist(), as_=['Label', 'Rank']).encode(
        x='Period:N',
        y='Rank:Q',
        color=alt.Color(
            'Label:N',
            scale=alt.Scale(
                domain=list(line_color.keys()),
                range=list(line_color.values())
                ),
            )
        )
    
    # plot cbre points
    other_points_df = line_data.loc[:, line_data.columns != 'NASA Rank']
    other_points = alt.Chart(other_points_df).mark_point(filled=True, size=50, opacity=1).transform_fold(other_points_df.columns[1:].tolist(), as_=['Label', 'Rank']).encode(
        x='Period:N',
        y='Rank:Q',
        color=alt.Color(
            'Label:N',
            scale=alt.Scale(
                domain=list(line_color.keys()),
                range=list(line_color.values())
                ),
            )
        )
    # plot vertical line to show backend vs forecasts
    vline = alt.Chart(pd.DataFrame({
        'Period': [date_range[len(rank_range) - 3]],
        'Label': ['black']
        })).mark_rule().encode(
            x='Period:N',
            color=alt.Color('Label:N', scale=None),
            strokeDash=alt.value([10, 5]),
            tooltip=alt.value(None)
            )
    

    # formatting altair chart
    chart = alt.layer(lines, outperform_area, underperform_area, other_points, nasa_points, vline).configure_axis(
        titleFontSize=20,
        titlePadding=20,
        titleFont='Arial',
        titleColor='#000000',
        labelColor='#000000',
        labelFontSize=14,
        grid=True,
        gridColor='#000000',
        gridOpacity=0.15,
        ).configure_legend(
            fillColor='#F0F0F0',
            titleFontSize=16,
            titleFont='Arial',
            titleColor='#000000',
            labelFontSize=12,
            titlePadding=15,
            padding=10,
            rowPadding=10,
            symbolSize=50,
            symbolStrokeWidth=5
        ).configure_view(
            fill='#F0F0F0'
        )
    # display chart
    st.altair_chart(chart, theme=None)


## format market view page ##
def mkt_view():
    st.logo(image='pages/logo.png', link=None)

    nav_menu()
    st.title('Multi-model, Single Market View', anchor=False)

    # pick property type and market
    get_selections()
    st.divider()      

    # plots a chart if market and property selections have both been made
    if st.session_state['property_dummy'] and 'mkt_dummy_1' in st.session_state:
        if st.session_state['mkt_dummy_1']:
            line_chart(st.session_state['property_dummy'], st.session_state['mkt_dummy_1'])
        
    return 0


# required be streamlit API to be first code called before any other streamlit code
st.set_page_config(page_title="Single Market View", page_icon='pages/Invesco_Global_Favicon_Blue_Pos_RGB_32px.png', layout='wide')

# handles session state cache being cleared
if 'Types' not in st.session_state:
    st.write("""<meta http-equiv="refresh" content="0; url='/'">""", unsafe_allow_html=True)
# load market view page
mkt_view()
