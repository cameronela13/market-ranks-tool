# CRE Market Rankings Application
# Interactive Streamlit GUI -- Information Page

# This file displays information about a selected property ranking model and data sources.

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


def get_info(property):

    # model statistics
    variables = np.array(st.session_state['Models'][property].loc[:, 'Variable'])
    # change variables for proper latex formatting
    var_latex = [var.replace('_', r'\_') for var in variables]

    coef = st.session_state['Models'][property].loc[:, 'Coefficient']
    coef = ['%.3f' % i for i in coef]
    # create list of model terms
    terms = []
    for i in range(0, len(coef), 2):
        # first line
        if i == 0:
            # y-intercept
            line = [f'{coef[i]}']
            # first and second terms
            line.extend([r'{} \cdot {}'.format(coef[i + 1], var_latex[i + 1]), r'{} \cdot {}'.format(coef[i + 2], var_latex[i + 2])])
            if line[1][0] != '-':
                line[1] = ' + ' + line[1]
                line[0] = ''.join(line[:2])
            else:
                line[1] = line[1][0] + ' ' + line[1][1:]
                line[0] = ' '.join(line[:2])
            del line[1]
            # y-int and first term are now joined; line variable has len 2 now
            if line[1][0] != '-':
                line[1] = ' + ' + line[1]
                terms.append(''.join(line))
            else:
                line[1] = line[1][0] + ' ' + line[1][1:]
                terms.append(' '.join(line))
        
        # any line after with two terms
        elif len(coef[i + 1:]) >= 2:
            line = [r'{} \cdot {}'.format(coef[i + 1], var_latex[i + 1]), r'{} \cdot {}'.format(coef[i + 2], var_latex[i + 2])]
            if line[0][0] != '-':
                line[0] = '+ ' + line[0]
            else:
                line[0] = line[0][0] + ' ' + line[0][1:]
            if line[1][0] != '-':
                line[1] = ' + ' + line[1]
                terms.append(''.join(line))
            else:
                line[1] = line[1][0] + ' ' + line[1][1:]
                terms.append(' '.join(line))

        # if the last line only has one term
        elif len(coef[i + 1:]) > 0:
            term = r'{} \cdot {}'.format(coef[i + 1], var_latex[i + 1])
            if term[0] != '-':
                term = ' + ' + term
            else:
                term = term[0] + ' ' + term[1:]
            terms.append(term)
    
    # present model in latex
    model_equation = r'\newline'.join(terms)
    accuracy = st.session_state['Accuracy'][property].iloc[-1, -1]
    conf_int = st.session_state['Models'][property].loc[:, 'Confidence Interval']
    conf_int = [['%.3f' % i[0], '%.3f' % i[1]] for i in conf_int]

    # model information section
    st.latex(f'P(outperformance) = {model_equation}')
    st.markdown(f'Average Backtesting Accuracy = **{accuracy:.3f}%**')  # display accuracy

    st.dataframe(pd.DataFrame({'Variable': variables,
                               'Coefficient': coef,
                               '95% Confidence Interval': conf_int}),
                               hide_index=True,
                               width=1000
                )  # display table with coefficients, means, variances, 95% conf. interval


# format info page
def info_page():
    st.logo(image='pages/logo.png', link=None)

    nav_menu()
    st.title('Modeling Information', anchor=False)
    
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
    
    if st.session_state['property_dummy']:
        get_info(st.session_state['property_dummy'])

    return 0


# required be streamlit API to be first code called before any other streamlit code
st.set_page_config(page_title="Single Market View", page_icon='pages/Invesco_Global_Favicon_Blue_Pos_RGB_32px.png', layout='wide')

# handles session state cache being cleared
if 'Types' not in st.session_state:
    st.write("""<meta http-equiv="refresh" content="0; url='/'">""", unsafe_allow_html=True)
# load info page
info_page()
