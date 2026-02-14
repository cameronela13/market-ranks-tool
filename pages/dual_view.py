# CRE Market Rankings Application
# Interactive Streamlit GUI -- Dual Market View Page

# This file compares how two markets' rankings have changed over time according to NASA forecasting.

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
def market_select(market, avail_mkts, box_num):
    # selectbox 1
    if box_num == 1:
        # if a market has been selected
        if market:
            # if a second market has been selected at any time
            if 'mkt_dummy_2' in st.session_state:
                # if the second selectbox has been called, set the Market 2 value
                st.session_state['Market 2'] = st.session_state['mkt_dummy_2']
            # if Market 1 and Market 2 are in session and they aren't equal
            # handles case where the second market is selected as the primary market in the single mkt view page
            if st.session_state['Market 1'] != st.session_state['Market 2']:
                # delete the option selected for market 1 for the second market's options
                avail_mkts = np.delete(avail_mkts, np.where(avail_mkts == st.session_state['Market 2'])) 
            # index used to show current selection in the selectbox
            index = int(np.where(avail_mkts == market)[0][0])
            st.selectbox('**Regional Market 1**',
                        avail_mkts,
                        index=index,
                        key='mkt_dummy_1')
        # no market selected (selectbox 2 should have no selection and not display)
        else:
            if avail_mkts is not None:
                st.selectbox('**Regional Market 1**',
                            avail_mkts,
                            index=None, placeholder='Select...',
                            key='mkt_dummy_1')
    
    # selectbox 2
    else:
        # remove first selected market from the market 2 options
        # market 2 selection only exists if there is a market selected for market 1
        avail_mkts = np.delete(avail_mkts, np.where(avail_mkts == st.session_state['mkt_dummy_1']))
        # if a market has been selected
        if market:
            index = int(np.where(avail_mkts == market)[0][0])
            st.selectbox('**Regional Market 2**',
                        avail_mkts,
                        index=index,
                        key='mkt_dummy_2')
        else:
            if avail_mkts is not None:
                st.selectbox('**Regional Market 2**',
                            avail_mkts,
                            index=None, placeholder='Select...',
                            key='mkt_dummy_2')


## get market and property type ##
def get_selections():

    # if page has been called at least once, session variables will be updated
    if 'property_dummy' in st.session_state:
        property_select(st.session_state['property_dummy'], st.session_state['Types']) 
    else:
        property_select(st.session_state['Property'], st.session_state['Types'])

    # if a property has been selected, call the first selectbox
    if st.session_state['property_dummy']:
        # if the first selectbox has loaded, set the Market 1 session value
        if 'mkt_dummy_1' in st.session_state:
            st.session_state['Market 1'] = st.session_state['mkt_dummy_1']
        avail_mkts = st.session_state['Markets'][st.session_state['property_dummy']].loc[:, 'Market']
        # if the property type has changed (not all properties have the same markets)
        if st.session_state['property_dummy'] != st.session_state['Property']:
            st.session_state['Market 1'] = None
            st.session_state['Market 2'] = None
        market_select(st.session_state['Market 1'], avail_mkts, 1)

    # if the first market has been selected, call the second selectbox
    if 'mkt_dummy_1' in st.session_state:
        if st.session_state['mkt_dummy_1']:
            # if the current first and previous second market selection are the same, or if property type changed
            if st.session_state['Market 1'] == st.session_state['Market 2']:
                st.session_state['Market 2'] = None
            # if the second selectbox has already been called once
            if 'mkt_dummy_2' in st.session_state:
                st.session_state['Market 2'] = st.session_state['mkt_dummy_2']
            market_select(st.session_state['Market 2'], avail_mkts, 2)

    st.session_state['Property'] = st.session_state['property_dummy']



# display line chart for a given market for a certain property type
def line_chart(property, market1, market2):

    st.subheader(f'**{market1}** and **{market2}**, *{property}*', anchor=False)

    # retrieve market data
    avail_mkts_data = st.session_state['Markets'][st.session_state['Property']].reset_index(drop=True)
    mkt1_row = avail_mkts_data[avail_mkts_data['Market'] == market1]
    mkt2_row = avail_mkts_data[avail_mkts_data['Market'] == market2]

    # filter for rank columns and correct columns
    rank_range = [i for i in mkt1_row.columns if 'RANK_' in i]
    correct_range = [i for i in mkt1_row.columns if 'CORRECT_' in i]

    # get nasa ranks and forecasts
    mkt1_ranks = mkt1_row.loc[:, rank_range].reset_index(drop=True)
    mkt2_ranks = mkt2_row.loc[:, rank_range].reset_index(drop=True)
    # add forecasted data that has no real data counterpart
    forecast_data = st.session_state['Latest Forecast'][property]
    forecast_rank = forecast_data[0].reset_index(drop=True)
    forecast_rank1 = forecast_rank[forecast_rank['Market'] == market1].iloc[:, 0].reset_index(drop=True)
    forecast_rank2 = forecast_rank[forecast_rank['Market'] == market2].iloc[:, 0].reset_index(drop=True)
    mkt1_ranks = pd.concat([mkt1_ranks, forecast_rank1], axis=1).squeeze(axis=0)
    mkt2_ranks = pd.concat([mkt2_ranks, forecast_rank2], axis=1).squeeze(axis=0)
    mkt1_forecasts = np.array(mkt1_row.loc[:, correct_range]).squeeze(axis=0)
    mkt2_forecasts = np.array(mkt2_row.loc[:, correct_range]).squeeze(axis=0)

    # put all backtesting data into DataFrames
    date_range = st.session_state['Time Range'][property][:]
    date_range.append(forecast_data[1])
    line_data = pd.DataFrame({
        'Period': date_range,
        market1: mkt1_ranks,
        market2: mkt2_ranks
        },
        index=None).reset_index(drop=True)
    
    # lines for outperform and underperform areas 
    areas = pd.DataFrame({'Period': date_range,
                        'Bottom': np.full(shape=len(date_range), fill_value=int(len(st.session_state['Markets'][st.session_state['Property']]))),
                        'Half': np.full(shape=len(date_range), fill_value=int(len(st.session_state['Markets'][st.session_state['Property']]) * 0.5))})

    # color scheme for lines and points
    line_color = {
        market1: '#000AD2',
        market2: '#0598FA',
        'False Prediction': '#D22B2B',
        'True Prediction': '#50C878',
        'NASA Forecast': '#000000'
    }

    # plot lines
    lines = alt.Chart(line_data).transform_fold(line_data.columns[1:].tolist(), as_=['Label', 'Rank']).mark_line().encode(
        x=alt.X('Period:N', scale=alt.Scale(padding=0)),
        y=alt.Y('Rank:Q', scale=alt.Scale(domain=[0, len(st.session_state['Markets'][st.session_state['Property']])], reverse=True, nice=False)),
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

    # create dataframes for points
    mkt1_points_df = pd.DataFrame(line_data.loc[:, :market1]).rename(columns={market1: 'Rank'})
    mkt1_points_df.loc[:, 'pred_val'] = pd.Series(mkt1_forecasts).reset_index(drop=True)
    mkt2_points_df = pd.DataFrame(line_data[['Period', market2]]).rename(columns={market2: 'Rank'})
    mkt2_points_df.loc[:, 'pred_val'] = pd.Series(mkt2_forecasts).reset_index(drop=True)

    # create new columns for each class and input ranks according to class
    mkt1_points_df.loc[:, 'False Prediction'] = pd.Series(np.full(shape=len(mkt1_points_df), fill_value=np.nan))
    mkt1_points_df.loc[:, 'True Prediction'] = pd.Series(np.full(shape=len(mkt1_points_df), fill_value=np.nan))
    mkt1_points_df.loc[:, 'NASA Forecast'] = pd.Series(np.full(shape=len(mkt1_points_df), fill_value=np.nan))
    mkt2_points_df.loc[:, 'False Prediction'] = pd.Series(np.full(shape=len(mkt2_points_df), fill_value=np.nan))
    mkt2_points_df.loc[:, 'True Prediction'] = pd.Series(np.full(shape=len(mkt2_points_df), fill_value=np.nan))
    mkt2_points_df.loc[:, 'NASA Forecast'] = pd.Series(np.full(shape=len(mkt2_points_df), fill_value=np.nan))

    for df in [mkt1_points_df, mkt2_points_df]:
        for row in df.index[:-1]:
            rank = df.loc[row, 'Rank']
            col_name = ''
            if df.loc[row, 'pred_val'] == 0:
                col_name += 'False Prediction'
            else:
                col_name += 'True Prediction'
            df.loc[row, col_name] = rank
        df.loc[len(rank_range) - 3: len(rank_range), 'NASA Forecast'] = df.loc[len(rank_range) - 3: len(rank_range), 'Rank']
        # drop all other columns
        df = df.drop(columns=['Rank', 'pred_val'])

    # plot the nasa points
    points_df = pd.concat([mkt1_points_df, mkt2_points_df]).reset_index(drop=True)
    points = alt.Chart(points_df).mark_point(filled=True, size=50, opacity=1).transform_fold(points_df.columns[1:].tolist(), as_=['Label', 'Rank']).encode(
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
    chart = alt.layer(lines, outperform_area, underperform_area, points, vline).configure_axis(
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
def dual_view():
    st.logo(image='pages/logo.png', link=None)

    nav_menu()
    st.title('NASA Model, Dual Market View', anchor=False)

    # pick property type and market
    get_selections()
    st.divider()      

    # plots a chart if market and property selections have both been made
    if st.session_state['property_dummy'] and 'mkt_dummy_1' in st.session_state:
        if st.session_state['mkt_dummy_1'] and st.session_state['mkt_dummy_2']:
            line_chart(st.session_state['property_dummy'], st.session_state['mkt_dummy_1'], st.session_state['mkt_dummy_2'])
        
    return 0


# required be streamlit API to be first code called before any other streamlit code
st.set_page_config(page_title="Single Market View", page_icon='pages/Invesco_Global_Favicon_Blue_Pos_RGB_32px.png', layout='wide')

# handles session state cache being cleared
if 'Types' not in st.session_state:
    st.write("""<meta http-equiv="refresh" content="0; url='/'">""", unsafe_allow_html=True)
# load market view page
dual_view()
