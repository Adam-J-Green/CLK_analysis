import time
import pandas as pd
import streamlit as st
import regex as re
import matplotlib.pyplot as plt
import plotly 
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as sp
import statsmodels.formula.api as smf
import Main_page
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from random import choices
import Main_page
from sklearn import tree


st.set_page_config(layout = 'wide')
colours = ['blue', 'orange', 'green', 'red', 'yellow', 'pink']

st.divider()
st.markdown(""" <style> .font {
font-size:50px;} 
</style> """, unsafe_allow_html=True)


data = pd.read_csv('tester.csv')
lin_reg_data = data.iloc[:, [3,5,6,7,8,9,10,11,12,47,48,49,50,51,52]]

def arrange_lin_model(data):
    response_selection = st.selectbox(label = "Select response variable", options = lin_reg_data.columns[-6:])
    predictor_selection = st.multiselect(label = 'Select Predictor Set or All Variables', options = lin_reg_data.columns[0:9], default='Total Number of Children in Program')

    x = lin_reg_data[predictor_selection]
    y = lin_reg_data[response_selection]
    le = preprocessing.LabelEncoder()
    try:
        x['Participant Age'] = le.fit_transform(x[['Participant Age']])
        x['Assessment Year'] = le.fit_transform(x[['Assessment Year']])
    except:
        pass
    dummy_cols = []
    for i in x.columns:
        if x[i].dtype == 'object': 
            dummy_cols.append(i)
    x = pd.get_dummies(data=x, columns = dummy_cols)
    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()
    summary = model.summary()
    rsquare = model.rsquared
    st.text(f"The R-Square Value for the Displayed Model is {round(rsquare, 2)}")
    st.table(summary.tables[1])



def arrange_dec_tree(data):
    response = st.selectbox(label = "Select response variable for Decision Tree", options = lin_reg_data.columns[-6:])
    mean = data[response].median()
    data['ind'] = np.where(data[response]>mean, 1, 0)
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = data[col].str.lower()
    col1, col2 = st.columns([1,1])
    with col1:
        Assessor = st.selectbox(label = 'Input Assessor Name', options = data['Assessor Name'].unique()).lower()
        Type = st.selectbox('Input Program Type', options = data['Program Type'].unique()).lower()
        Age = st.selectbox('Input Participant Age', options = data['Participant Age'].unique()).lower()
    with col2:
        Supervisor = st.selectbox('Input Supervisor Name', options = data['Program Supervisor Name'].unique()).lower()
        num_child = st.text_input('Input Number of Children')
        num_super = st.text_input('Input Number of Staff/Volunteers')
    
    try:
        num_child = int(num_child)
        num_super = int(num_super)
    except:
        pass
    new_dict = {'Assessor Name':Assessor, 'Program Type':Type, 'Participant Age':Age, 
                'Program Supervisor Name':Supervisor, 'Total Number of Children in Program':num_child, 'Total Number of Staff/Volunteers in Program':num_super}
    df2 = data.append(new_dict, ignore_index = True)
    response = data['ind']
    cat_cols = df2[['Assessor Name', 'Program Type', 'Participant Age', 'Program Supervisor Name']]
    encoded_data = pd.get_dummies(cat_cols)
    encoded_data['Total Number of Children in Program'] = df2['Total Number of Children in Program']
    encoded_data['Total Number of Staff/Volunteers in Program'] = df2['Total Number of Staff/Volunteers in Program']
    new_data = encoded_data.iloc[-1]
    encoded_data = encoded_data.iloc[0:-1]
    dect = DecisionTreeClassifier()
    dect.fit(X = encoded_data, y = response)
    try:
        preds = dect.predict_proba([new_data])
        st.write("Predicted probability that the program will score above average overall:",preds[:,1][-1]*100)
    except:
        pass
    

col1, col2 = st.columns([1,1])
with col1:
    arrange_lin_model(lin_reg_data)

with col2:
    arrange_dec_tree(data)


def bootstrap_dat(data):
    indicator_var = st.selectbox(label = 'Select Variable of Interest', options = data.columns[-6:]) 
    cat_var = st.selectbox(label = 'Select Variable to Categorize results', options = data.columns[:-6])
    uniques = data[cat_var].unique()
    dfs_list = []
    arrays_list = []
    for uni in uniques:
        data2 = data[data[cat_var] == uni]
        ind_data = list(data2[indicator_var])
        means = {'means': []}
        for i in range(100):
            sample = choices(ind_data,k= 200)
            sample_av = np.mean(sample)
            means['means'].append(sample_av)
        arrays_list.append(means['means'])
        df = pd.DataFrame(means)
        df['ind'] = str(uni)
        dfs_list.append(df)
    return pd.concat(dfs_list), arrays_list, uniques, indicator_var, cat_var

boot_strap_data = data.iloc[:, [3,5,6,7,8,9,12,47,48,49,50,51,52]]
booted_data, array_list, uniques, indic, cat_var= bootstrap_dat(boot_strap_data)

if len(array_list) == 2:
    krusk = sp.kruskal(array_list[0], array_list[1])
    ind = ['no difference' if krusk.pvalue>0.05 else 'a difference']
    st.text(f'Kruskal-Wallis testing difference in {indic} between {uniques[0]} and {uniques[1]} returns pvalue: {krusk.pvalue}\n meaning there is {ind[0]} between groups')
elif len(array_list) == 3:
    krusk = sp.kruskal(array_list[0], array_list[1], array_list[2])
    ind = ['no difference' if krusk.pvalue>0.05 else 'a difference']
    st.text(f'Kruskal-Wallis testing difference in {indic} between {uniques[0]}, {uniques[1]} and {uniques[2]} returns pvalue: {krusk.pvalue}\n meaning there is {ind[0]} between groups')
elif len(array_list) == 4:
    krusk = sp.kruskal(array_list[0], array_list[1], array_list[2], array_list[3])
    ind = ['no difference' if krusk.pvalue>0.05 else 'a difference']
    st.text(f'Kruskal-Wallis testing difference in {indic} between {uniques[0]}, {uniques[1]}, {uniques[2]} and {uniques[3]} returns pvalue: {krusk.pvalue}\n meaning there is {ind[0]} between groups')


if len(uniques) == 2 and cat_var != 'Assessment Year':
    cat1 = booted_data[booted_data['ind'] == uniques[0]]
    cat2 = booted_data[booted_data['ind'] == uniques[1]]
    ttest_stat, ttest_pval = sp.ttest_ind(list(cat1['means']), list(cat2['means']))
    ind = ["a difference" if ttest_pval <0.05 else 'no difference']
    st.text(f"Test for difference of means between groups returned pvalue: {round(int(ttest_pval), 3)} meaning there is {ind[0]}")
if len(uniques) == 3:
    lists = []
    grouped = booted_data.groupby('ind')
    uniq = len(uniques)
    for i in uniques:
        group = grouped.get_group(i)
        lists.append(group['means'])
    lists = np.transpose(lists)
    l1,l2,l3 = map(list, zip(*lists))
    stat, pval = sp.f_oneway(l1,l2,l3)
    ind = ["a difference" if pval <0.05 else 'no difference']
    st.text(f"Test for difference of means between groups returned pvalue: {round(int(pval), 3)} meaning there is {ind[0]}")
if len(uniques) == 4:
    lists = []
    grouped = booted_data.groupby('ind')
    uniq = len(uniques)
    for i in uniques:
        group = grouped.get_group(i)
        lists.append(list(group['means']))
    lists = np.transpose(lists)
    l1,l2,l3,l4 = map(list, zip(*lists))
    stat, pval = sp.f_oneway(l1,l2,l3,l4)
    ind = ["a difference" if pval <0.05 else 'no difference']
    st.text(f"Test for difference of means between groups returned pvalue: {round(int(pval), 3)} meaning there is {ind[0]}")
if len(uniques) == 5:
    lists = []
    grouped = booted_data.groupby('ind')
    uniq = len(uniques)
    for i in uniques:
        group = grouped.get_group(i)
        lists.append(list(group['means']))
    lists = np.transpose(lists)
    l1,l2,l3,l4,l5 = map(list, zip(*lists))
    stat, pval = sp.f_oneway(l1,l2,l3,l4,l5)
    ind = ["a difference" if pval <0.05 else 'no difference']
    st.text(f"Test for difference of means between groups returned pvalue: {round(int(pval), 3)} meaning there is {ind[0]}")

fig = plt.figure()
sns.violinplot(data = booted_data, x = 'means', y= 'ind', orient='h')
st.pyplot(fig = fig)

st.divider()

value_of_interest = st.selectbox(label = 'Please select the assessment category of interest', options = lin_reg_data.columns[-6:])
metric_of_interest = st.selectbox(label = "Please Select a variable of interest", options = ['Assessment Year', 'Participant Age', 'Program Supervisor Name', 'Assessor Name'])

comparator1 = st.selectbox(label='Please select the first subset of interest', options = data[metric_of_interest].unique())
comparator2 = st.selectbox(label='Please select the second subset of interest', options = data[metric_of_interest].unique())

def simple_random(data, assess_category, metric, comparator):
    """function takes data, a metric of interest and a attribute within that metric and returns a list of means generated from simple random sampling with replacement"""
    filter_1 = data[data[metric] == comparator]
    sample_means = []
    for i in range(200):
        sample = choices(list(filter_1[assess_category]), k =200)
        mean = np.mean(sample)
        sample_means.append(mean)
    samples_dict = {'means' : sample_means}
    means_df = pd.DataFrame(samples_dict)
    means_df['ind'] = comparator
    return sample_means, means_df

cola, colb = st.columns(2)
with cola:
    col1, comp1_df = simple_random(data, assess_category = value_of_interest, metric = metric_of_interest, comparator=comparator1)
    col2, comp2_df = simple_random(data, assess_category = value_of_interest, metric = metric_of_interest, comparator=comparator2)
    alternative_hypothesis = st.selectbox(label = 'Select the alternative hypothesis you would like to test', options = ['two-sided', 'one-sided'])
    if alternative_hypothesis == 'one-sided':
        alternative_hypothesis = 'greater'
    ttest, pval = sp.ttest_ind(list(col1), list(col2), alternative=alternative_hypothesis)
    if alternative_hypothesis == 'two-sided':
        if pval<0.05:
            conclusion = 'difference'
        else:
            conclusion = 'no difference'
    else:
        if pval<0.05:
            conclusion = 'greater'
        else:
            conclusion = 'not greater'
    st.write(f'T-value for the test: {round(ttest, 2)}\nP-value for the test: {round(pval, 2)}')
    if alternative_hypothesis == 'two-sided':
        st.write(f'For the {alternative_hypothesis} test to observe a difference \nin mean {value_of_interest} between {comparator1} and {comparator2}, there was {conclusion}\n in the means of the two groups at a 95% confidence level')
    if alternative_hypothesis == 'greater':
        st.write(f'For the {alternative_hypothesis}, it can be concluded that the mean {value_of_interest} for {comparator1} is {conclusion} than the mean for {comparator2} at a 95% confidence level')
with colb:
    dfs = pd.concat([comp1_df, comp2_df])
    fig = plt.figure()
    sns.boxplot(x = dfs['means'], y = dfs['ind'], orient='h')
    st.pyplot(fig = fig)

        

















