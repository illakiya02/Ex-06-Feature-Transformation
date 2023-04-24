# Ex-06-Feature-Transformation

import pandas as pd
df=pd.read_csv('/content/Data_to_Transform.csv')
df.head()

df.isnull().sum()

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Highly Negative Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Moderate Negative Skew'],fit=True,line='45')
plt.show()

df['Highly Positive Skew']=np.log(df['Highly Positive Skew'])
sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

df['Highly Positive Skew']=1/df['Highly Positive Skew']
sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

df['Highly Positive Skew']=np.sqrt(df['Highly Positive Skew'])
sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

from sklearn.preprocessing import PowerTransformer
pt=PowerTransformer("yeo-johnson")
df['Moderate Negative Skew']=pd.DataFrame(pt.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['Moderate Negative Skew'],fit=True,line='45')
plt.show()

from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df['Moderate Negative Skew']=pd.DataFrame(pt.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['Moderate Negative Skew'],fit=True,line='45')

plt.show()

![image](https://user-images.githubusercontent.com/112244898/233904422-4ba53c78-e304-4977-af9a-8f42cada9ff7.png)

![image](https://user-images.githubusercontent.com/112244898/233904401-dbd97eca-8456-4cab-8536-7b1900f67672.png)

![image](https://user-images.githubusercontent.com/112244898/233904463-95eff791-07b7-44cd-9b11-3832d7d05eca.png)


![image](https://user-images.githubusercontent.com/112244898/233904489-1b9e7e47-63ef-4b05-902e-b93eb9bd2246.png)

![image](https://user-images.githubusercontent.com/112244898/233904529-de979922-0fef-40a2-89f1-75e0169a2767.png)


![image](https://user-images.githubusercontent.com/112244898/233904548-1b34e827-2348-4398-b7bd-2bcaa1e518c0.png)

![image](https://user-images.githubusercontent.com/112244898/233904561-4ffc0831-d052-459a-a57d-75d1cb3ccfdd.png)

![image](https://user-images.githubusercontent.com/112244898/233904570-a1ac706f-9d8b-44bd-bd3c-2a67350f5c8d.png)




