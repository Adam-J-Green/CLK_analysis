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
    st.success('Data Upload Complete. Proceed to the Exploration Pages')
except:
    pass


def gen_data(uploaded_file):
    test_data = preprocess.preprocess(uploaded_file, num_sheets)
    test_data.to_csv('tester.csv')
    general_data = pd.read_csv('tester.csv')
    return general_data

try:
    gen_data(uploaded_file)
except:
    pass