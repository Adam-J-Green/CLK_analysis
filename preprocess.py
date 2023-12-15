import pandas as pd
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import re
import streamlit as st
import openpyxl

def preprocess(file, num_sheets):
  dfs_list = []
  for i in range(num_sheets+1):
    try:
      mid_dat = pd.read_excel(file, sheet_name=i, header=1)
      num_rows = mid_dat['Assessment Date\n(DD/MM/YYYY)'].notna().sum()
      mid_data = mid_dat.iloc[0:num_rows, :]
      
      mid_data['Assessment Date\n(DD/MM/YYYY)'] = pd.to_datetime(mid_data['Assessment Date\n(DD/MM/YYYY)'], dayfirst=True)
      
      dfs_list.append(mid_data)
    except Exception as e:
      st.write(e)
      st.write('The number of sheets selected exceeds the number of sheets available, please enter an alternate value')
  general_data = pd.concat(dfs_list, axis=0)
  general_data = general_data.rename({'Assessment Date\n(DD/MM/YYYY)':'Assessment Date'}, axis = 1)
  staff = []
  comments = []
  for col in general_data.columns:
      if re.match(col[0:4], "Staff"):
          staff.append(col)
      if re.match(col[0:5], "Commen"):
          comments.append(col)
  nas = []
  df = general_data.loc[:,staff]
  for index, row in df.iterrows():
    Num_volunteers = 9 - (row.isna().sum())
    nas.append(Num_volunteers) 
  general_data['volunteer_count'] = nas
  staff = staff[2:] +comments
  general_data = general_data.drop(staff, axis = 1)
  
  #general_data['Assessment Date'] = general_data['Assessment Date'].dt.strftime('%Y-%M-%d')
  
  #general_data['Assessment Date']=general_data['Assessment Date'].astype(str).str.pad(width=10, side='left', fillchar='0')
  
  #general_data['Assessment Date'] = pd.to_datetime(general_data['Assessment Date']).dt.strftime('%Y-%m-%d')
  
  general_data['Assessment Date'] = pd.to_datetime(general_data['Assessment Date'])
  
  #generate Simplified date columns
  general_data['Assessment Year'] = general_data['Assessment Date'].apply(lambda y :  int(y.strftime('%Y')))
  general_data['Assessment Month'] = general_data['Assessment Date'].apply(lambda x : int(x.strftime('%m'))) 
  
  #Generate Sessions column
  session_dict = {11: 'Fall', 10: 'Fall', 5 : 'Spring', 8 :'Summer'}
  general_data['Session'] = general_data['Assessment Month'].map(session_dict)

  #rearrange columns
  general_data.rename({"Activity Risk Awareness'": 'Activity Risk Awareness'}, axis = 1, inplace=True)
  general_data = general_data[['Program Name', 'Assessment Date', 'Assessment Year','Assessment Month', 'Session','Assessor Name', 'Site', 'Program Type', 'Participant Age', 'Total Number of Children in Program','Total Number of Staff/Volunteers in Program',
  'Program Supervisor Name', 'Staff/Volunteer #2 Name',
  'Staff/Volunteer #3 Name', 'Warmth', 'Interest', 'Respect',
  'Individualized Approach', 'Involvement', 'Positive Leadership',
  'Children Have Priority', 'Total Weighted Score', 'Awareness',
  'Age/stage Appropriate', 'Activity Risk Awareness','Supervision in Transition Areas', 'Site Safety',
  'Total Weighted Score.1', 'Familiarity Among Children',
  'Respect and Cooperation', 'Inclusionary Behaviour', 'Atmosphere',
  'Total Weighted Score.2', 'Appropriate Behaviour & Language',
  'Discretion with Confidential Matters', 'Team Effort',
  'Total Weighted Score.3', 'Program Planning',
  'Activity Appropriate Space', 'Welcoming Environment',
  'Developmentally Appropriate Equipment',
  'Access and Quantity of Equipment', 'Balance Variety and Choice',
  'Pace of Activities', 'Individual Growth through Group Involvement',
  'Total Weighted Score.4', 'Leader/Child Interactions',
  'Supervision & Safety', 'Child/Child Interactions',
  'Leader Behaviour & Interactions', 'Program Characteristics & Support','QUEST 2 Total Score']]

  general_data.columns.str.lower()
  #general_data.to_csv('general_data.csv')
  return general_data



