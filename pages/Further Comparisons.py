import time
import pandas as pd
import streamlit as st
import regex as re
import matplotlib.pyplot as plt
import plotly.express as plotl
import seaborn as sns
from datetime import datetime
import Main_page

data = pd.read_csv('tester.csv')

std_vals = {'Leader/Child Interactions': 108.36, 'Supervision & Safety':64.8, 'Child/Child Interactions':48.24, 'Leader Behaviour & Interactions':34.02, 'Program Characteristics & Support':72, 'QUEST 2 Total Score':327.42}

def filter_data(data, year, session):
    filtered = data[data['Assessment Year'].isin(year)] 
    filtered2 = filtered[filtered['Session'].isin(session)] 
    return filtered2 

year_options = data['Assessment Year'].unique()
year = st.sidebar.multiselect(label = "Year", options = data['Assessment Year'].unique(), default = year_options[0])

session = st.sidebar.multiselect(label='Session', options = data['Session'].unique(), default = 'Fall')

agg_cols = data.iloc[:,-6:].columns

for col in agg_cols:
    data[col] = data[col] /std_vals[col]

data_filt = filter_data(data, year, session)


cat_groups = data_filt.iloc[:, [3,5,6,8,9,12]].columns
score_groups = data_filt.iloc[:, -6:-1].columns

cat = st.selectbox(label = 'Select variable to group by', options = cat_groups)
cola, colb = st.columns([1,1])
with colb:
    dfs_list = []
    for i, col in enumerate(score_groups):
        item = data_filt[[col, cat]]
        item.rename({col: 'score'}, axis=1, inplace=True)
        item['Quest 2 Category'] = col
        dfs_list.append(item)
    plot_dat = pd.concat(dfs_list)
    fig = plt.figure()
    sns.boxplot(plot_dat, x = 'score', y ='Quest 2 Category', hue=cat, orient = 'h')
    plt.title('Summary of Evaluation Scores, Stratified by Program Characteristic of Interest')
    st.pyplot(fig)
with cola:
    grouped = plot_dat.groupby(['Quest 2 Category', cat]).aggregate(Score_mean=pd.NamedAgg(column="score", aggfunc="mean"))
    grouped = grouped.reset_index()
    fig = plt.figure()
    sns.barplot(data = grouped, x =  'Score_mean', y ='Quest 2 Category', hue = cat, orient='h')
    plt.title('Comparison of Evaluation Scores, Stratified by Program Characteristic of Interest')
    plt.xlabel('Mean Score')
    st.pyplot(fig)

st.divider()

score_groups2 = data_filt.iloc[:, -6:].columns

prog1 = st.selectbox(label = 'Select Program of Interest', options = data['Program Name'].unique())
score_data = data_filt[data_filt['Program Name'] == prog1]

fig = plt.figure()
dfs_list = []
for i, col in enumerate(score_groups2):
    item = score_data[[col, 'Assessment Year']]
    item.rename({col: 'score'}, axis=1, inplace=True)
    item['Quest 2 Category'] = col
    dfs_list.append(item)
plot_dat = pd.concat(dfs_list)
plot_dat['Assessment Year'] = plot_dat['Assessment Year'].astype('category')
fig = plotl.bar(plot_dat, x = 'score', y = 'Quest 2 Category', color = 'Assessment Year', barmode = 'group', title = f'Scores Across Evaluation Categories for {prog1}')
st.plotly_chart(fig, use_container_width=True)

st.divider()

cat = st.selectbox(label = "Select Quest 2 Category of Interst", options = score_groups2)
prog1 = st.selectbox(label = 'Select First Program of Interest', options = data['Program Name'].unique())
 #data_mid = data.drop([prog1], axis = 0)
prog2 = st.selectbox(label = 'Select Second Program of Interest', options = data['Program Name'].unique())
if st.checkbox('Show all programs'):
    data_all = data.groupby(['Assessment Year', 'Assessment Month', 'Program Name']).aggregate('mean').reset_index()
    cols=['Assessment Month', "Assessment Year"]
    data_all['Date'] = data[cols].apply(lambda x: '-'.join(x.values.astype(str)), axis="columns")
    data_all['Date'] = pd.to_datetime(data_all['Date'], format='%m-%Y').dt.date
    fig = plotl.line(data, x = 'Assessment Date', y = cat, color = 'Program Name', title =f'Change in {cat} by Year Across all Programs')
    st.plotly_chart(fig, use_container_width=True)
else:
    data_fin = data[data['Program Name'].isin([prog1, prog2])]
    data_fin = data_fin.groupby(['Assessment Year', 'Assessment Month', 'Program Name']).aggregate('mean').reset_index()
    cols=['Assessment Month', "Assessment Year"]
    data_fin['Date'] = data_fin[cols].apply(lambda x: '-'.join(x.values.astype(str)), axis="columns")
    data_fin['Date'] = pd.to_datetime(data_fin['Date'], format='%m-%Y').dt.date
    fig =plotl.line(data_fin, x = 'Date', y = cat, color = 'Program Name', title =f'Change in {cat} by Year Across {prog1} and {prog2}')
    st.plotly_chart(fig, use_container_width=True)

st.divider()

cole, colf = st.columns([1,1])
with cole:
        Assessor = st.multiselect('Input Assessor Name', options = data['Assessor Name'].unique())
        Type = st.multiselect('Input Program Type', options = data['Program Type'].unique())
        Age = st.multiselect('Input Partiticpant Age', options = data['Participant Age'].unique())
with colf:
        Supervisor = st.multiselect('Input Supervisor Name', options = data['Program Supervisor Name'].unique())
        num_child = st.multiselect('Input Number of Children', options = data['Total Number of Children in Program'].unique())
        num_super = st.multiselect('Input Number of Staff/Volunteers', options = data['Total Number of Staff/Volunteers in Program'].unique())

datafilt1 = data.drop('Unnamed: 0', axis = 1)
if len(Assessor)>0:
    datafilt1 = datafilt1[datafilt1['Assessor Name'].isin(Assessor)]
if len(Type)>0:
    datafilt1 = datafilt1[datafilt1['Program Type'].isin(Type)]
if len(Age)>0:
    datafilt1 = datafilt1[datafilt1['Participant Age'].isin(Age)]
if len(Supervisor)>0:
    datafilt1 = datafilt1[datafilt1['Program Supervisor Name'].isin(Supervisor)]
if len(num_child)>0:
    datafilt1 = datafilt1[datafilt1['Total Number of Children in Program'].isin(num_child)]
if len(num_super)>0:
    datafilt1 = datafilt1[datafilt1['Total Number of Staff/Volunteers in Program'].isin(num_super)]

st.dataframe(datafilt1)
















