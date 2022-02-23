# -*- coding: utf-8 -*-
"""
221HS-256F-1 : Healthcare Data Analytics and Data Mining (Spring 2022)

@author: Group 2
"""

import pandas as pd
from scipy.stats import fisher_exact
import seaborn as sns
pd.set_option('max_colwidth',200)
pd.set_option('display.max_columns', 10)

#%% Q1
df1 = pd.read_csv('npidata_pfile_20050523-20220109.csv', dtype=str, usecols=['NPI', 'Provider First Name', 'Provider Last Name (Legal Name)', 'Provider License Number State Code_1'])

def find_doctor(df, npi, patient_name):
    doctor_state = df.loc[df['NPI']==str(npi)]
    first_name = doctor_state['Provider First Name'].values[0]
    last_name = doctor_state['Provider Last Name (Legal Name)'].values[0]
    state_code = doctor_state['Provider License Number State Code_1'].values[0]
    return f'The healthcare provider ({first_name} {last_name}) of {patient_name} was first licensed in {state_code}.'

find_doctor(df1, 1811001167, 'Chitra')
find_doctor(df1, 1316033871, 'Qingqiu')
find_doctor(df1, 1003917626 , 'Ruifeng')
find_doctor(df1, 1891767885, 'Wilfred')
find_doctor(df1, 1588667539, 'Skye')

#%% Q2
# Load the data.
# Since we only need three columns for this question, we don’t import all the data.
df = pd.read_csv('npidata_pfile_20050523-20220109.csv',
                usecols = ['Provider Business Practice Location Address State Name',
                           'Provider Gender Code',
                           'Is Sole Proprietor'])

# Filter out the rows we don't need.
sub = df.loc[((df['Provider Business Practice Location Address State Name'] == 'AZ') |
              (df['Provider Business Practice Location Address State Name'] == 'GA') |
              (df['Provider Business Practice Location Address State Name'] == 'NV') |
              (df['Provider Business Practice Location Address State Name'] == 'RI') |
              (df['Provider Business Practice Location Address State Name'] == 'TX') |
              (df['Provider Business Practice Location Address State Name'] == 'VT') |
              (df['Provider Business Practice Location Address State Name'] == 'WV') |
              (df['Provider Business Practice Location Address State Name'] == 'WI')) &
              ((df['Provider Gender Code'] == 'M') |
               (df['Provider Gender Code'] == 'F')) &
              ((df['Is Sole Proprietor'] == 'Y') |
               (df['Is Sole Proprietor'] == 'N'))]

# Coount numbers of male and femal separately who is a sole proprietor and who is not.
group_num = sub.groupby(['Provider Gender Code', 'Is Sole Proprietor'], as_index=False).size()

# Extract the numbers of each group.
count_MY = group_num.loc[(group_num['Provider Gender Code'] =='M') &
                         (group_num['Is Sole Proprietor'] == 'Y'), 'size'].values[0]
count_MN = group_num.loc[(group_num['Provider Gender Code'] =='M') &
                         (group_num['Is Sole Proprietor'] == 'N'), 'size'].values[0]
count_FY = group_num.loc[(group_num['Provider Gender Code'] =='F') &
                         (group_num['Is Sole Proprietor'] == 'Y'), 'size'].values[0]
count_FN = group_num.loc[(group_num['Provider Gender Code'] =='F') &
                         (group_num['Is Sole Proprietor'] == 'N'), 'size'].values[0]

# Run Fisher’s Exact Test
df_2 = [[count_MY, count_MN],
        [count_FY, count_FN]]
from scipy.stats import fisher_exact
oddsratio, pvalue = fisher_exact(df_2)
print(f'p-value is: {pvalue}')
'''
The p-value for the tests is about 0.1697, which is greater than 0.05.
So we can not reject the Null Hypothesis.
In other words, we don’t have sufficient evidence to say that there is a significant association between gender and sole proprietor preference.
'''

#%% Q3
df3 = pd.read_csv('npidata_pfile_20050523-20220109.csv', dtype=str, usecols=['Provider Gender Code', 'Healthcare Provider Taxonomy Code_1', 'Provider Business Practice Location Address State Name'])

#slice with States
df3a = df3[(df3['Provider Business Practice Location Address State Name'] == 'WI') | (df3['Provider Business Practice Location Address State Name'] == 'AZ') | (df3['Provider Business Practice Location Address State Name'] == 'GA') | (df3['Provider Business Practice Location Address State Name'] == 'KY') | (df3['Provider Business Practice Location Address State Name'] == 'NV') | (df3['Provider Business Practice Location Address State Name'] == 'RI') | (df3['Provider Business Practice Location Address State Name'] == 'TX') | (df3['Provider Business Practice Location Address State Name'] == 'VT') | (df3['Provider Business Practice Location Address State Name'] == 'WV')].copy()

df3a.info()
df3a.isna().sum()

#drop NaNs from the dataframe
df3a.dropna(inplace=True, axis=0)

#Obstetrics & Gynecology: 207V00000X
#Pediatrics: 208000000X
#Surgery: 208600000X
#Orthopaedic Surgery: 207X00000X

#low risk & male
df3_lm = df3a[((df3a['Healthcare Provider Taxonomy Code_1']=='207V00000X')|(df3a['Healthcare Provider Taxonomy Code_1']=='208000000X'))&(df3a['Provider Gender Code']=='M')].copy()

#low risk & female
df3_lf = df3a[((df3a['Healthcare Provider Taxonomy Code_1']=='207V00000X')|(df3a['Healthcare Provider Taxonomy Code_1']=='208000000X'))&(df3a['Provider Gender Code']=='F')].copy()

#high risk & male
df3_hm = df3a[((df3a['Healthcare Provider Taxonomy Code_1']=='208600000X')|(df3a['Healthcare Provider Taxonomy Code_1']=='207X00000X'))&(df3a['Provider Gender Code']=='M')].copy()

#high risk & female
df3_hf = df3a[((df3a['Healthcare Provider Taxonomy Code_1']=='208600000X')|(df3a['Healthcare Provider Taxonomy Code_1']=='207X00000X'))&(df3a['Provider Gender Code']=='F')].copy()

#Fisher Exact test - H0: males are less or equal likely to do high-risk practices, compared to their female peers
oddsratio, pvalue = fisher_exact([[df3_hm.shape[0], df3_lm.shape[0]], [df3_hf.shape[0], df3_lf.shape[0]]])
print('{0:.4f}'.format(pvalue)) # reject the null hypothesis
#we can conclude that our observed imbalance is statistically significant; males prefer to choose the high-risk practices, females prefer to do low-risk practice.

#%% Q4
df = pd.read_csv('npidata_pfile_20050523-20220109.csv',dtype=str,
                 usecols=('Healthcare Provider Taxonomy Code_1', 'Entity Type Code', 'Provider Business Mailing Address State Name'))
facilities = df[df['Entity Type Code'] == '2']
MRI = facilities[facilities['Healthcare Provider Taxonomy Code_1'] == '261QM1200X'][['Healthcare Provider Taxonomy Code_1', 'Provider Business Mailing Address State Name']]
# df[df['Healthcare Provider Taxonomy Code_1'] == '261QM1200X'].shape
MRI_per_state = MRI.groupby('Provider Business Mailing Address State Name').count()
MRI_per_state = MRI_per_state.sort_values(by=['Provider Business Mailing Address State Name']).iloc[:,0:2]
MRI_per_state = MRI_per_state.rename(columns={'Provider Business Mailing Address State Name':'Provider State Name',
                                              'Healthcare Provider Taxonomy Code_1':'Count of MRI'})
MRI_per_state.to_csv(r'MRI_per_state.csv')
