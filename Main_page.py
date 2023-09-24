import time
import pandas as pd
import streamlit as st
import regex as re
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as plotl
import seaborn as sns
import scipy.stats as sp
from tabulate import tabulate
import preprocess


colours = ['blue', 'orange', 'green', 'red', 'yellow', 'pink']

st.set_page_config(page_title = 'CLK Program Evaluation', layout = 'wide')
st.markdown('<div style="text-align: center;"> <font size = "8"><b>Christie Lake Kids Program Evaluation</b></font></div>', unsafe_allow_html=True)
st.divider()
st.markdown(""" <style> .font {
font-size:50px;} 
</style> """, unsafe_allow_html=True)

uploaded_file = st.file_uploader(label= 'Upload a File', type='xlsx')
try:
    num_sheets = int(st.text_input(label = 'Input the number of assessment periods to observe \n(value of 0 indicates that only the first assessment period is selected)'))
except:
    pass
def gen_data(uploaded_file):
    test_data = preprocess.preprocess(uploaded_file, num_sheets)
    test_data.to_csv('tester.csv')
    general_data = pd.read_csv('tester.csv')
    return general_data

def get_agg_cols(data):
        agg_data = ['Program Name']
        cols = data.iloc[:,-6:]
        agg_data = agg_data + cols
        return agg_data


def show_item(data, checkbox):
    if st.checkbox(checkbox):
        st.dataframe(data.style.highlight_max(axis = 0, subset=['Total Weighted Score', 'Total Weighted Score.1', 'Total Weighted Score.2', 'Total Weighted Score.3', 'Total Weighted Score.4', 'QUEST 2 Total Score']))

def select_box(data, options_list, name):
    options = st.sidebar.selectbox(name, options_list)
    return options

def filter_data(data, year, session):
    if year == 'All':
        year = data['Assessment Year'].unique()
    if session == 'All':
        session = data['Session'].unique()
    filtered = data[data['Assessment Year'].isin(year)] 
    filtered2 = filtered[filtered['Session'].isin(session)] 
    return filtered2 

if uploaded_file is not None:

    data = gen_data(uploaded_file)

    filtered_data = data
    
    year = st.sidebar.multiselect(label = "Year", options = data['Assessment Year'].unique(), default = 2022)
    
    session = st.sidebar.multiselect(label='Session', options = data['Session'].unique(), default = 'Fall')

    column_names = {'Total Weighted Score':'Leader/Child Interactions', 'Total Weighted Score.1':'Supervision and Saftey', 'Total Weighted Score.2':'Child/Child Interactions', 'Total Weighted Score.3':'Leader Behaviour & Interactions', 'Total Weighted Score.4':'Program Characteristics and Support'}
    std_vals = {'Leader/Child Interactions': 108.36, 'Supervision & Safety':64.8, 'Child/Child Interactions':48.24, 'Leader Behaviour & Interactions':34.02, 'Program Characteristics & Support':72}


    if st.sidebar.checkbox("Show Aggregate Options"):
        filtered_data = filter_data(data, year, session)
        select_agg = st.sidebar.multiselect(options = data.iloc[:,-6:].columns, label = 'Aggregate Column of Interest', default=['QUEST 2 Total Score'])
        select_agg = ['Program Name']+select_agg
        filtered_data = filtered_data[select_agg].groupby('Program Name').aggregate('mean')
    else:
        filtered_data = filter_data(data, year, session)
        agg_cols = filtered_data.iloc[:, -6:].columns
        names = filtered_data['Program Name']
        filtered_data = filtered_data[agg_cols]
        filtered_data['Program Name'] = names
        columns = filtered_data.columns.tolist()
        filtered_cols = columns[-1:] + columns[:-1]
        filtered_data = filtered_data[filtered_cols].groupby('Program Name').aggregate('mean')

    alt_data = filter_data(data, year, session)
    agg_cols = alt_data.iloc[:, -6:].columns
    agg_cols = agg_cols.insert(0, 'Program Name')
    alt_data = alt_data[agg_cols]


    if st.sidebar.checkbox('Normalized Values'): 
        for i in filtered_data.columns[1:]:
            if i == 'QUEST 2 Total Score':
                pass
            else:
                filtered_data[i] = filtered_data[i]/std_vals[i]

        for i in alt_data.columns:
            if i == 'QUEST 2 Total Score' or i == 'Program Name':
                pass
            else:
                alt_data[i] = (alt_data[i]/std_vals[i])


    tab1, tab2 = st.tabs(["Raw Data", "Summary Statistics"])
    with tab2:
        summary_cont = st.container()
        selected_item = summary_cont.selectbox(label = 'Select Subset of Interest', options=['Program Name', 'Session', 'Site', 'Assessor Name', 'Program Type', 'Participant Age', 'Program Supervisor Name'])
        radio_min = st.radio(label = 'Highlight Max', options = ['Max', 'Min'])
        cols = filtered_data.columns
        agg = data.groupby(selected_item)[cols].aggregate(['median'])
        if radio_min== 'Max':
            summary_cont.table(agg.round(2).style.highlight_max(axis=0))
        else:
            summary_cont.table(agg.round(2).style.highlight_min(axis=0))

    with tab1:
        raw_cont = st.container()
        max_cols = len(filtered_data)
        num_rows = raw_cont.slider(max_value=max_cols, label = "Select Number of rows to display")
        raw_cont.table(filtered_data.iloc[0:num_rows, :].round(2).style.highlight_max(axis =0).set_properties(**{'height': '30px'}))

    st.divider()


    col1, col2 = st.columns([1,1])
    with col1:
        hist_data = alt_data.drop('QUEST 2 Total Score', axis = 1)
        selected_cols = list(st.multiselect(label='Select Variables to Display', options=hist_data.columns, default='Leader/Child Interactions'))
        fig = plt.figure()
        for i, col in enumerate(selected_cols):
            sns.histplot(hist_data[col], color=colours[i], kde = True)
        plt.xlabel('Score')
        fig.legend(selected_cols)
        st.pyplot(fig=fig)

    with col2:
        multi_hist_data = alt_data['QUEST 2 Total Score']
        hist_data = filter_data(data, year = year, session=session)
        metric = st.selectbox(label = 'Select Item to Group on', options=['Session', 'Site', 'Assessor Name', 'Program Type', 'Participant Age', 'Program Supervisor Name'])
        cols_list = {}
        for item in data[metric].unique():
            col = hist_data[hist_data[metric] == item]
            cols_list[item] = col['QUEST 2 Total Score']
        merged_dat = pd.DataFrame(cols_list)
        figure2 = plt.figure()
        for i, column in enumerate(merged_dat.columns):
            sns.histplot(merged_dat[column], color=colours[i], kde = True)
        plt.xlabel('Quest 2 Score')
        figure2.legend(hist_data[metric].unique())
        st.pyplot(fig=figure2)


    st.divider()

    col3, col4 = st.columns([1,1])

    with col3:
        x_ax = st.selectbox(label = 'X-Axis Value', options=alt_data.columns[1:])
        y_ax = st.selectbox(label = 'Y-Axis Value', options = alt_data.columns[1:])
        fig = plotl.scatter(data_frame=alt_data, x = alt_data[x_ax], y = alt_data[y_ax],hover_name = 'Program Name', hover_data=[x_ax, y_ax], trendline = 'ols')
        st.plotly_chart(fig, use_container_width=True)


    with col4:
        metric_scatter = st.selectbox(label = 'Select Metric to Display', options = filtered_data.iloc[:,1:].columns)
        grouper = st.selectbox(label = 'Select Grouping Item', options = ['Leader Count', 'Child Count', 'Ratio Child:Leader'])
        if grouper == 'Leader Count':
            data['Leader Count'] = np.where(data['Total Number of Staff/Volunteers in Program']>3, 'Greater than 3', 'Less than 3')
            fig = plt.figure()
            sns.barplot(data, x = data['Leader Count'], y = metric_scatter)
            st.pyplot(fig = fig)
        elif grouper == 'Child Count':
            data['Child Count'] = np.where(data['Total Number of Children in Program']>8, 'Greater than 8', 'Less than 8')
            fig = plt.figure()
            sns.barplot(data, x = 'Child Count' , y = metric_scatter)
            st.pyplot(fig = fig)
        elif grouper == 'Ratio Child:Leader':
            data['ratio'] = (data['Total Number of Children in Program']/data['Total Number of Staff/Volunteers in Program'])
            data['Child:Leader Ratio'] = np.where(data['Total Number of Children in Program']>3,'Greater than 3', 'Less than 3')
            fig = plt.figure()
            sns.barplot(data, x = 'Child:Leader Ratio' , y = metric_scatter)
            st.pyplot(fig = fig)
        










