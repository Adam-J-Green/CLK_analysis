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

colours = ['blue', 'orange', 'green', 'red', 'yellow', 'pink']

def gen_data(csv):
    return pd.read_csv(csv)

data = gen_data('general_data.csv')

lin_reg_data = data.iloc[:, [3,5,6,7,8,9,12,47,48,49,50,51,52]]
print(lin_reg_data)
print(data.info())

