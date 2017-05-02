import numpy as np 
import pandas as pd 
import seaborn as sns
import sklearn.linear_model as linreg 

dfW = pd.read_csv('../Data/NOAAWeatherDataDIA.csv')
dfOClean = pd.read_csv('../Data/OffensesData-Clean-EssVars.csv')
dfL = pd.read_csv('../Data/LiquorLicenses.csv')

dfOW = pd.merge(dfOClean,dfW,left_on='Occ Date',right_on='DATE')
dfL_district_totals = pd.DataFrame({'District':dfL['POLICE_DIST'].value_counts().sort_index().index,
                                'Total_Dist_Liq_Licenses':dfL['POLICE_DIST'].value_counts().sort_index().values})
df = pd.merge(dfOW,dfL_district_totals,on='District')  
df.drop('DATE',axis=1,inplace=True)
df.to_csv('../Data/OffensesData-Clean-EssVars-WithWeather-WithLicenseTotals.csv',index=False)
df_sample = df.sample(frac=.025,random_state=101).dropna()
df_sample.to_csv('../Data/OffensesData-Clean-EssVars-WithWeather-WithLicenseTotals-Sample.csv',index=False)

