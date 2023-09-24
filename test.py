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


path = '.\MASTER Copy Quest 2 Data - LM Edits 2.9.23.xlsx'
colours = ['blue', 'orange', 'green', 'red', 'yellow', 'pink']

def gen_data(csv):
    return pd.read_csv(csv)

#data = gen_data('general_data.csv')
data = pd.read_excel(path, sheet_name=1, header =1)



lin_reg_data = data.iloc[:, [3,5,6,7,8,9,12,47,48,49,50,51,52]]


