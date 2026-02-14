<h1>Market Rankings Tool</h1>

An interactive commercial real estate market analysis tool.<br>
**DISCLAIMER:** As the coder of the original application, I have stripped all sensitive and proprietary information from this build in order to publicly present my work. All data used in this version is either randomly generated or made up by me. Any reference to real data or modeling in this README is retained to preserve basic understanding of the application's framework and purpose.

<h2>Basics and Motivation</h2>

This project displays market rankings based on their performance compared to a benchmark. The target metric is a binary performance variable, and a market's probability to outperform a time period's benchmark determines its ranking. Rankings are available for multiple property types and from multiple sources. This application focuses on the models built by members of the Invesco Real Estate NASA team. This tool will help **[REDACTED]** using historical data pulled from **[REDACTED]**, other commercial real estate firms, and the Snowflake cloud service.<br>**NOTE:** Rankings **do not** indicate magnitude of out/underperformance.

<h2>App Features/Uses</h2>

The goal of this app is to show how markets have performed or will perform. The main features of this app include:
- Home page: gives information about the app's current build
- Market view: displays past rankings and forecasts for a selected market and property type using NASA's models and other firm models
- Dual market view: displays past rankings and forecasts for two selected markets under a particular property type using NASA's models
- Filtered rankings: an interactive DataFrame with all market rankings for a selected time period and property type
    - Past rankings include labels for each market indicating True/False Out/underperformance
- Model information page: gives a NASA model's variables, coefficients, and statistics for a selected property type

<h2>Code Style</h2>

This tool frequently uses pandas, numpy, streamlit, and altair. Any issues in code involving the former three packages are likely fixable using their respective online API references. Unfortunately, the altair online API reference is not incredibly helpful. Because of this, however, it seems many people have turned to trusty Stack Overflow for help. Familiarity with streamlit's session_state and selectbox behavior is important. For implementing future property types, the format of the data used in the application should at least follow the output of the get_industrial() function in main.py. Integrating new property types is even easier if their associated data follows the excel files called in the aforementioned function (the get_industrial() function could be easily modified to accomodate such formatted future data).

<h2>Build Status</h2>

<h3>7 August 2024</h3>

**Property types available:** Apartment, Industrial<br>
**Backtesting periods available:** 2013Q4-2016Q4 to 2020Q4-2023Q4<br>
**Latest Forecast Period:** 2024Q1-2027Q1

<h4>NOTES</h4>

Any data or features regarding Green Street are disabled since accurate past Green Street data does not exist in Snowflake at this time. Green Street's past data must be pulled from Green Street itself.<br>
**Fixes:**
- App would crash whenever cached data was cleared. Temporary fix: redirect to home page and reset session state. Future goal: keep the same page and retain session state.


<h2>Coding Framework</h2>

Built using VSCode. Imported packages:
- pandas
- numpy
- streamlit
- altair
- snowflake

<h2>Running the Market Rankings Tool</h2>

With the above dependencies installed, open a terminal window, or use the in-IDE terminal. While in the 'Market Rankings' directory (use 'cd' command to navigate the directory), run the command 'streamlit run main.py'. If an error message says streamlit is not recognized, you may need to run main.py once (doing this will produce an error, but the CLI streamlit command will now work). This will open the app in the default browser. From here, navigate the application's widgets with the left tab. Use the drop-down selectboxes on each page to display the data you would like to see.