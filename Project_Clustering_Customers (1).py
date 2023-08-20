#!/usr/bin/env python
# coding: utf-8

# ## Customer Personality Analysis
# 
# Customer Personality Analysis is a detailed analysis of a company’s ideal customers. It helps a business to better understand its customers and makes it easier for them to modify products according to the specific needs, behaviors and concerns of different types of customers.
# Customer personality analysis helps a business to modify its product based on its target customers from different types of customer segments. For example, instead of spending money to market a new product to every customer in the company’s database, a company can analyze which customer segment is most likely to buy the product and then market the product only on that particular segment.
# Attributes
# 
# People
# ID: Customer's unique identifier
# 
# Year_Birth: Customer's birth year
# 
# Education: Customer's education level
# 
# Marital_Status: Customer's marital status
# 
# Income: Customer's yearly household income
# 
# Kidhome: Number of children in customer's household
# 
# Teenhome: Number of teenagers in customer's household
# 
# Dt_Customer: Date of customer's enrollment with the company
# 
# Recency: Number of days since customer's last purchase
# 
# Complain: 1 if the customer complained in the last 2 years, 0 otherwise
# Products
# 
# MntWines: Amount spent on wine in last 2 years
# 
# MntFruits: Amount spent on fruits in last 2 years
# 
# MntMeatProducts: Amount spent on meat in last 2 years
# 
# MntFishProducts: Amount spent on fish in last 2 years
# 
# MntSweetProducts: Amount spent on sweets in last 2 years
# 
# MntGoldProds: Amount spent on gold in last 2 years
# Promotion
# 
# NumDealsPurchases: Number of purchases made with a discount
# 
# AcceptedCmp1: 1 if customer accepted the offer in the 1st campaign, 0 otherwise
# 
# AcceptedCmp2: 1 if customer accepted the offer in the 2nd campaign, 0 otherwise
# 
# AcceptedCmp3: 1 if customer accepted the offer in the 3rd campaign, 0 otherwise
# 
# AcceptedCmp4: 1 if customer accepted the offer in the 4th campaign, 0 otherwise
# 
# AcceptedCmp5: 1 if customer accepted the offer in the 5th campaign, 0 otherwise
# 
# Response: 1 if customer accepted the offer in the last campaign, 0 otherwise
# Place
# 
# NumWebPurchases: Number of purchases made through the company’s website
# 
# NumCatalogPurchases: Number of purchases made using a catalogue
# 
# NumStorePurchases: Number of purchases made directly in stores
# 
# NumWebVisitsMonth: Number of visits to company’s website in the last month
# 
# ## Target
# Need to perform clustering to summarize customer segments.

# ## Data Exploratory Analysis

# In[ ]:


#installing libraries
get_ipython().system('pip install kmodes')
get_ipython().system('pip install kneed')


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sn
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime,timedelta
from sklearn.preprocessing import MinMaxScaler,StandardScaler,OneHotEncoder
from sklearn.cluster import KMeans,AgglomerativeClustering,DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score,silhouette_samples
from yellowbrick.cluster import KElbowVisualizer
import matplotlib.cm as cm
import scipy.cluster.hierarchy as sch
from itertools import product
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


from google.colab import files
uploaded = files.upload()


# In[ ]:


df_cust = pd.read_excel("marketing_campaign3.xlsx")


# In[ ]:


df_cust.head()


# In[ ]:


df_cust.dtypes


# In[ ]:


#checking shape of data
df_cust.shape


# In[ ]:


#checking with descriptive statistics
df_cust.describe().T.style.background_gradient(cmap='Greens')


# In[ ]:


# Missing values percentage in 'Income' column of datatset
missing_percentage= df_cust["Income"].isnull().sum()*100 /len(df_cust)
print("Percentage of missing values in 'income' column: {:.2f}%".format(missing_percentage))


# In[ ]:


#checking fro any null values
df_cust.isnull().sum()


# In[ ]:


#plotting with heatmap for null values
sn.heatmap(df_cust.isnull(),cbar=False,cmap='hot')


# In[ ]:


#checking with information of data
df_cust.info()


# In[ ]:


#checking with the datatypes
df_cust.dtypes


# In[ ]:


#checking for any duplicate records
df_cust[df_cust.duplicated()]


# # Univariate Analysis

# In[ ]:


# checking with the count of education
sn.set_style('whitegrid')
sn.countplot(x ='Education',data=df_cust)
plt.show()


#  - Observation
# 
#  We observe that most of the customers come from a good background of education ,most of them are having post garduation and graduation degrees
#    
# 

# In[ ]:


#checking with feature marital_status
sn.set_style('whitegrid')
sn.countplot(x ='Marital_Status',data=df_cust)
plt.show()


# - Observation
# 
# We observe that most of them are in a association compared to bachelors and singles

# In[ ]:


#checking count with customers accepted campaigns
fig,ax = plt.subplots(1,5,figsize = (20,8))
colors =['#9EF8EE','#348888']
fig.suptitle('Campaign Acceptance',fontsize = 20)
sn.countplot(ax = ax[0],x = 'AcceptedCmp1',data=df_cust)
sn.countplot(ax = ax[1],x = 'AcceptedCmp2',data=df_cust)
sn.countplot(ax = ax[2],x = 'AcceptedCmp3',data=df_cust)
sn.countplot(ax = ax[3],x = 'AcceptedCmp4',data=df_cust)
sn.countplot(ax = ax[4],x = 'AcceptedCmp5',data=df_cust)
plt.tight_layout()


# - Observation
# 
# We observe that most of the people have not participated in any of the campaigns and accepted offers  
# 
# 

# In[ ]:


#checking with the count of children
fig,ax = plt.subplots(1,2,figsize = (20,8))
colors =['#9EF8EE','#348888']
fig.suptitle('Kids and teens',fontsize = 20)
sn.countplot(ax = ax[0],x = 'Kidhome',data=df_cust)
sn.countplot(ax = ax[1],x = 'Teenhome',data=df_cust)

plt.tight_layout()


#  - observations
#  We observe that the ratio is equally shared by kids and teens, most of the customers have no kids or teens

# In[ ]:


#plotting with complain and response
fig,ax = plt.subplots(1,2,figsize = (20,8))
colors =['#9EF8EE','#348888']
fig.suptitle('Complain and Response',fontsize = 20)
sn.countplot(ax = ax[0],x = 'Complain',data=df_cust)
sn.countplot(ax = ax[1],x = 'Response',data=df_cust)

plt.tight_layout()


#  - Observations
# 
#  - We observe that most of the customers have no complains
#  - In the response plot most have not well reponded to the recent campaign or last campaign

# In[ ]:


fig,ax = plt.subplots(nrows = 3,ncols = 2,figsize = (15,7))
fig.suptitle('Amount Spent on products',fontsize = 20)
sn.histplot(ax = ax[0,0],x = 'MntFishProducts',data=df_cust)
sn.histplot(ax = ax[0,1],x = 'MntFruits',data=df_cust)
sn.histplot(ax = ax[1,0],x = 'MntGoldProds',data=df_cust)
sn.histplot(ax = ax[1,1],x = 'MntMeatProducts',data=df_cust)
sn.histplot(ax = ax[2,0],x = 'MntSweetProducts',data=df_cust)
sn.histplot(ax = ax[2,1],x = 'MntWines',data=df_cust)
plt.tight_layout()


# - Observations
# 
# - We observe that most of the customer have highest expenditure over wines and meatproducts and least engaged with fruits

# In[ ]:


#plotting with purchases at different locations
fig,ax = plt.subplots(nrows = 2,ncols = 2,figsize = (15,7))
fig.suptitle('Purchases at Locations',fontsize = 20)
sn.histplot(ax = ax[0,0],x = 'NumDealsPurchases',data=df_cust)
sn.histplot(ax = ax[0,1],x = 'NumCatalogPurchases',data=df_cust)
sn.histplot(ax = ax[1,0],x = 'NumStorePurchases',data=df_cust)
sn.histplot(ax = ax[1,1],x = 'NumWebPurchases',data=df_cust)

plt.tight_layout()


#  - Observations
# 
#  -  We observe most of the customers have reached to catalogs and purchased items and have selected particular item
# 
#  - we can also see most of the customers have prefeered online shopping and have purchased from website
# 
#  - most of customers have reach out to store and we can infer that customers purchase rate through store is less compared to other locations

# In[ ]:


#checking distribution for numerical data
numerical_feature = df_cust.select_dtypes(include = 'integer')

for n in numerical_feature.columns:
    print(n)

    sn.distplot(numerical_feature[n])
    plt.figure(figsize=(20,10))
    plt.show()


# In[ ]:


#checking with value counts of categorical features
categorical_feature = df_cust.select_dtypes(include = 'object')

for var in categorical_feature:

    print(df_cust[var].value_counts())


# In[ ]:


#checking with any float value
float_feature = df_cust.select_dtypes(include = 'float')

for n in float_feature.columns:
    print(n)

    sn.distplot(float_feature[n])
    plt.figure(figsize=(20,10))
    plt.show()


#  Observations
# 
# - We observe that  our numerical dataset is not normally distributed, this tells us that the data has skewness and needs to be treated
# 
# - We also observe that the Income feature contains null values which needs to be treated
# 
# - we also observe that the distribution of feeature Z_cost contact and Z_cost revenue is very similiar.
# 
# - We also have categorical columns which needs to be converted to numerical columns by feature engineering
# 
# - We observe that in the income column our data is highly skewed and does not support a normal distribution so to fill null values we go with median,moreover median also reduces the effect of outliers  

# 
# ## Feature analysis

# In[ ]:


df_cust.head()


# In[ ]:


#dropping our ID,Z_CostContact,Z_Revenue
df_cust2 = df_cust.drop(['ID','Z_CostContact','Z_Revenue'],axis = 1)


# In[ ]:


df_cust2


# - Analysis On Products feature

# In[ ]:


#checking the unique labels of products
products = [df_cust2['MntWines'],df_cust2['MntFishProducts'],df_cust2['MntGoldProds'],df_cust2['MntMeatProducts'],df_cust2['MntSweetProducts'],df_cust2['MntFruits']]

for items in products:
  print(items.unique())


# In[ ]:


df_cust2['Expenses'] = df_cust2['MntWines'] + df_cust2['MntFruits'] + df_cust2['MntMeatProducts'] + df_cust2['MntFishProducts'] + df_cust2['MntSweetProducts'] + df_cust2['MntGoldProds']


df_cust2['Expenses'].head(10)


# In[ ]:


df_cust2.drop(['MntWines','MntFruits','MntFishProducts','MntGoldProds','MntMeatProducts','MntSweetProducts'],axis = 1,inplace=True)


# In[ ]:


df_cust2.head()


# In[ ]:


#checking with the mean of expenses on total products
df_cust2.Expenses.mean()


# - Purchases Feature

# In[ ]:


Purchases = df_cust2['NumCatalogPurchases'],df_cust2['NumDealsPurchases'],df_cust2['NumStorePurchases'],df_cust2['NumWebPurchases']

for items in Purchases:
  print(items.unique())


# In[ ]:


#adding Purchases features
df_cust2['Purchases'] = df_cust2['NumCatalogPurchases']+df_cust2['NumDealsPurchases']+df_cust2['NumStorePurchases']+df_cust2['NumWebPurchases']

df_cust2['Purchases'].head(10)


# In[ ]:


#dropping the columns
df_cust2.drop(['NumCatalogPurchases','NumDealsPurchases','NumStorePurchases','NumWebPurchases'],axis = 1 , inplace = True)


# In[ ]:


df_cust2


# In[ ]:


#checking with the mean of purchase
df_cust2.Purchases.mean()


# - Acceptance Feature

# In[ ]:


Acceptance = [df_cust2['AcceptedCmp1'],df_cust2['AcceptedCmp2'],df_cust2['AcceptedCmp3'],df_cust2['AcceptedCmp4'],df_cust2['AcceptedCmp5']]

for items in Acceptance:
  print(items.unique)


# In[ ]:


#combining our acceptance column to one feature
df_cust2['Acceptance'] = df_cust2['AcceptedCmp1']+df_cust2['AcceptedCmp2']+df_cust2['AcceptedCmp3']+df_cust2['AcceptedCmp4']+df_cust2['AcceptedCmp5']

df_cust2['Acceptance'].head(100)


# In[ ]:


df_cust.AcceptedCmp1.value_counts()


# In[ ]:


#dropping campaign acceptance columns from main dataframe
df_cust2.drop(['AcceptedCmp1','AcceptedCmp2','AcceptedCmp3','AcceptedCmp4','AcceptedCmp5'],axis=1,inplace = True)


# In[ ]:


df_cust2.head()


# - Children Feature

# In[ ]:


#checking with value counts with children
Children = [df_cust2['Kidhome'],df_cust2['Teenhome']]

for items in Children:
  print(items.value_counts())


# In[ ]:


#combining children columns into one column
df_cust2['Children'] = df_cust2['Kidhome']+df_cust2['Teenhome']

df_cust2['Children'].head(10)


# In[ ]:


#dropping our kids and teen column
df_cust2.drop(['Kidhome','Teenhome'],axis = 1,inplace = True)


# In[ ]:


df_cust2.head()


# Customer Age Feature

# In[ ]:


#extracting the year,day name and month name from date customer feature
df_cust2['Year_Enr'] = df_cust2['Dt_Customer'].dt.year
df_cust2['Day'] = df_cust2['Dt_Customer'].dt.day_name()
df_cust2['month'] = df_cust2['Dt_Customer'].dt.month_name()


# In[ ]:


df_cust2.head()


# In[ ]:


df_cust2.Dt_Customer.value_counts()


# In[ ]:


print(df_cust2.Dt_Customer.max(),df_cust2.Dt_Customer.min())


# In[ ]:


#Extracting Day engaged with the company using the schedule of first day enrollment program of next year
df_cust2['Dt_Customer'] = pd.to_datetime(df_cust2.Dt_Customer)
df_cust2['First_day'] = '01-01-2015'
df_cust2['First_day'] = pd.to_datetime(df_cust2.First_day)
df_cust2['day_engaged'] = (df_cust2['First_day'] - df_cust2['Dt_Customer']).dt.days


# In[ ]:


df_cust2.head()


# In[ ]:


df_cust2['Customer_Age'] = 2023 - df_cust2['Year_Birth']


# In[ ]:


df_cust2.head()


# In[ ]:


df_cust2 = df_cust2.drop(['Dt_Customer','Year_Birth','First_day'],axis=1)


# In[ ]:


#extracting age frequency of customers enrolled from the year joined till today
df_cust2['Enr_Freq'] = 2023 - df_cust2['Year_Enr']


# In[ ]:


df_cust2.head()


# In[ ]:


#checking value counts with Day enrolled
df_cust2.Day.value_counts()


# In[ ]:


#checking value counts with month enrolled
df_cust2.month.value_counts()


# In[ ]:


#plotting enrollment analysis with the company
fig,ax = plt.subplots(1,2,figsize = (15,10))
colors =['#9EF8EE','#348888']
fig.suptitle('Enrollment Analytics',fontsize = 40)
df_cust2.month.value_counts().sort_values(ascending = False).plot(ax = ax [0],kind ='barh')
df_cust2.Day.value_counts().plot(ax = ax [1],kind ='pie', autopct= '%.2f%%',shadow=True,explode = [0.1,0.0,0.0,0.0,0.0,0.0,0.0],colors=colors)


# Obervations
# - In the customer enrollment analytics we observe that most of the customer have enrolled in the month of August, and least in th month of july
# 
# - In the customer enrollment analytics We observe that on daily analysis that most customers have enrolled on Monday and least on tuesday
# 
# 

# In[ ]:


#computing the counts of years of relationship with company
df_cust2.Enr_Freq.value_counts()


# In[ ]:


#dropping unecessary columns with a new dataframe
df_cust3  = df_cust2.copy()
df_cust3 = df_cust3.drop(['Day','month','Year_Enr','NumWebVisitsMonth','Complain','Response'],axis = 1)


# In[ ]:


df_cust3.head()


# - Marital_Status Feature

# In[ ]:


df_cust3.Marital_Status.value_counts()


# In[ ]:


#replacing values of marital status
df_cust3['Marital_Status'] = df_cust3['Marital_Status'].replace(['Married','Together'],'Couple')
df_cust3['Marital_Status'] = df_cust3['Marital_Status'].replace(['Single','Divorced','Widow','Alone','Absurd','YOLO'],'Solo')


# In[ ]:


print(df_cust3.Marital_Status.value_counts())


# In[ ]:


df_cust3.head()


# - Education Feature

# In[ ]:


df_cust3.Education.value_counts()


# In[ ]:


#replacing values of Education
df_cust3['Education'] = df_cust3['Education'].replace(['PhD','Master','2n Cycle'],'PostGrad')
df_cust3['Education'] = df_cust3['Education'].replace(['Graduation'],'Grad')
df_cust3['Education'] = df_cust3['Education'].replace(['Basic'],'Undergrad')


# In[ ]:


df_cust3.Education.value_counts()


# In[ ]:


df_cust3.head()


# In[ ]:


#dropping the recency columns as we dont require
df_cust3 = df_cust3.drop(['Recency'],axis = 1)


# In[ ]:


df_cust3.head()


# # EDA And Visualizations

# - Income Feature

# In[ ]:


#checking with the descriptive statistics of income
df_cust3.Income.describe()


# In[ ]:


#filling null values for Income value
median = df_cust3.Income.median()
df_cust3['Income'] = df_cust3['Income'].fillna(median)


# In[ ]:


df_cust3.Income.isnull().sum()


# In[ ]:


#checking with descriptive statistics for Income feature
df_cust3.Income.describe()


# In[ ]:


#checking for outliers in Income column
sn.boxplot(df_cust3['Income'],orient='h',width = 0.5)


# In[ ]:


#locating outliers in the income column
df_cust3[df_cust3['Income']>600000]


# In[ ]:


#since we have observed that our income column is skewed we will use the interquartile method
#locating iqr
percentile25 = df_cust3['Income'].quantile(0.25)
percentile75 = df_cust3['Income'].quantile(0.75)

iqr = percentile75 - percentile25


# In[ ]:


upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr
print(upper_limit,lower_limit)


# In[ ]:


df_cust3[df_cust3['Income']>upper_limit]
df_cust3[df_cust3['Income']<lower_limit]


# In[ ]:


#trimming the outliers
df_cust4 = df_cust3[df_cust3['Income']< upper_limit]
df_cust4.shape


# In[ ]:


#capping or winsorization method
df_cust4['Income'] = np.where(df_cust4['Income']>upper_limit,
                      upper_limit,
                      np.where(df_cust4['Income']<lower_limit,
                              lower_limit,
                           df_cust4['Income']))


# In[ ]:


#checking the Income feature after treating outliers
sn.boxplot(df_cust4['Income'],orient = 'h', width = 0.5)


# In[ ]:


#checking with descriptive statistics with income  feature for binning
df_cust4.Income.describe()


# In[ ]:


#binning for Income column
min_value = df_cust4['Income'].min()
max_value = df_cust4['Income'].max()
income_range = (np.round(max_value - min_value))
bins = 4
bin_width = int(np.round(income_range/bins))


# In[ ]:


#printing our values
print("the max range is ,",max_value)
print("the min range is ,",min_value)
print("the income range is ,",income_range)
print("the bin width is ,",bin_width)


# In[ ]:


#plotting for histogram before binning  and afte binning
values = df_cust4.Income.values
fig,ax = plt.subplots(ncols = 2,figsize = (15,7))
fig.suptitle("Income Binning plots",fontsize = 30)
ax[0].set_title('Before Bining Distribution',fontsize = 15)
ax[1].set_title('After binning Distribution',fontsize = 15)
plt.hist( values,edgecolor="red", bins= 5)
sn.histplot(ax=ax[0],x = 'Income',data=df_cust4)


# - Observation
#  - Before binnning we observe that we have peaks with irregular intervals of data points.
# 
#  - after binning our intervals have been generalised and uneven peaks have been reduced  with a normal distribution

# In[ ]:


#locating the range for binning using numpy library
np.linspace(min_value,max_value,5)


# In[ ]:


#creating bins for income columns by choosing bins= 5 according to histogram plots shown above

df_cust4['Income_Category'] = pd.cut(df_cust4['Income'],bins = [0,29732,57733,85734,113735],labels = ['lower','middle','upper_middle','upper'],right=False,include_lowest=True)
df_cust4.head()


# In[ ]:


df_cust4.Income_Category.value_counts()


# In[ ]:


#dropping our income feature as we have created bins
df_cust4.drop(['Income'],axis=1,inplace=True)


# In[ ]:


df_cust4.head()


# In[ ]:


#checking with outliers in our total dataframe
plt.figure(figsize=(20,10))
sn.boxplot(data=df_cust4)


# In[ ]:


#subplots to check the distance of outliers for Expenses and customer Age
fig,ax = plt.subplots(1,2,figsize = (15,10))
fig.suptitle("Feature Boxplots Analytics",fontsize = 30)
ax[0].set_title('Expenses',fontsize = 20)
ax[1].set_title('Customer Age',fontsize = 20)
sn.boxplot(ax=ax[0],data=df_cust4['Expenses'])
sn.boxplot(ax=ax[1],data=df_cust4['Customer_Age'])


# - In the expenses plot we observe that we have very minimal distance of outliers, hence we leave it without treating
# 
# - In the Customer Age plot we observe that we have outliers with extreme values of distance hence we choose the Customer Age to treat the outliers

# Outlier Analysis with Customer Age Feature

# In[ ]:


#checking with boxenplot for the customer Age feature
sn.boxplot(df_cust4['Customer_Age'],orient = 'h',)


# In[ ]:


#checking with outliers with age feature
df_cust4[df_cust4['Customer_Age']>100]


# In[ ]:


#since we have observed that our Customer Age column is skewed we will use the interquartile method
#locating iqr
percentile25 = df_cust4['Customer_Age'].quantile(0.25)
percentile75 = df_cust4['Customer_Age'].quantile(0.75)

iqr = percentile75 - percentile25
print(iqr)


# In[ ]:


#locating our upper and lower limit
upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr
print(upper_limit,lower_limit)


# In[ ]:


#locating the outliers
df_cust4[df_cust4['Customer_Age']>upper_limit]
df_cust4[df_cust4['Customer_Age']<lower_limit]


# In[ ]:


#trimming the outliers
df_cust5 = df_cust4[df_cust4['Customer_Age']< upper_limit]
df_cust5.shape


# In[ ]:


#capping or winsorization method
df_cust5['Customer_Age'] = np.where(df_cust5['Customer_Age']>upper_limit,
                      upper_limit,
                      np.where(df_cust5['Customer_Age']<lower_limit,
                              lower_limit,
                           df_cust5['Customer_Age']))


# In[ ]:


#checking the Customer Age after treating outliers
sn.boxplot(df_cust5['Customer_Age'],orient = 'h', width = 0.5)


# In[ ]:


#checking with descriptive statistics with Customer Age
df_cust5.Customer_Age.describe()


# In[ ]:


df_cust5.head()


# In[ ]:


#checking with histogram with our final dataset
df_cust5.hist(figsize = (10,8))
plt.show()


# In[ ]:


#checking with Dtypes
df_cust5.dtypes


# - Combining Marital and Children To Family Feature

# In[ ]:


#checking with marital status and children
print(df_cust5.Marital_Status.value_counts())
print(df_cust5.Children.value_counts())


# In[ ]:


# Mapping Marital status for combination with children to create a new feature Family
df_cust5.Marital_Status = df_cust5.Marital_Status.map({'Solo':1,'Couple':2})


# In[ ]:


df_cust5.head()


# In[ ]:


#adding our Marital feature and children feature to create one feature family
df_cust5['Family'] = df_cust5['Marital_Status']+ df_cust5['Children']


# In[ ]:


df_cust5.head()


# In[ ]:


#dropping our Marital Status and Children feature
df_cust5.drop(['Marital_Status','Children'],axis=1,inplace = True)


# In[ ]:


df_cust5.head()


# In[ ]:


print(df_cust5.Family.value_counts())


# # Bivariate Analysis

# In[ ]:


#plotting with Family Distribution and their expenses
fig,ax = plt.subplots(1,2,figsize = (12,8))
fig.suptitle("Family Expenses Distribution",fontsize = 30)
ax[0].set_title('Family Expenses Distribution',fontsize = 15)
ax[1].set_title('Family Distribution',fontsize = 15)
plt.pie(df_cust5.Family.value_counts(),
      labels=['1','2','3','4','5'], autopct= '%.2f%%',shadow=True,
      colors = ['#9EF8EE','#348888'],explode= [0.1,0.0,0.0,0.0,0.0],
      textprops={'size':'large',
                 'fontweight':'bold'})
sn.barplot(ax=ax[0],data = df_cust5,x='Family',y = 'Expenses',palette="mako",)


# - Observations
# 
# - We observe people who are single comprise around 39.75% and have the    highest expenses compared to others
# 
# - Family of 4 or 5 have very less  expenses compared to singles and they comprise of 11.35% and 1.44% respectively
# 
# - Family of 2 and 3 have moderate expenses and spend very judiciously compared to other categories,they comprise 34.01% and 13.46% respectively

# In[ ]:


df_cust5.head()


# In[ ]:


#plotting Education with Expenses
fig,ax = plt.subplots(1,2,figsize =(12,8))
fig.suptitle("Education Expenses Distribution",fontsize = 30)
ax[0].set_title('Stripplot Distribution',fontsize = 15)
ax[1].set_title('Bar Distribution',fontsize = 15)
sn.stripplot(ax=ax[0],data= df_cust5,x = 'Education',y = 'Expenses',palette='cividis',jitter=True)
sn.barplot(ax = ax [1],data = df_cust5,x = 'Education',y = 'Expenses',palette='winter')


# - Observations
# 
# - We clearly observe that people  with Post graduation and graduation degree have higher expenses compared to people with under graduation and also we observe that the majority with post graduation and graduation are greater in number compared to undergrads

# In[ ]:


df_cust5.head()


# In[ ]:


#plotting with Expenses and purchases
fig,ax = plt.subplots(nrows=2,ncols=2,figsize = (15,10))
fig.suptitle("Expenses-Purchases Plot",fontsize = 30)
sn.regplot(ax=ax[0,0],x='Expenses',y = 'Purchases',data=df_cust5)
sn.scatterplot(ax=ax[0,1],x='Expenses',y = 'Purchases',data=df_cust5,hue = 'Income_Category')
sn.histplot(ax=ax[1,0],x='Purchases',hue ='Education',multiple="stack",palette="RdBu",
            edgecolor = ".3",data=df_cust5)
sn.lineplot(ax =ax[1,1],x='Purchases',y ='Expenses',data = df_cust5,hue = 'Family'
            ,palette = "Set2")
ax[0,0].set_title("Regression plot")
ax[0,1].set_title("Purchases - Expenses scatter plot with Income")
ax[1,0].set_title("Purchases - Education Histogram plot")
ax[1,1].set_title("Purchases - Expenses Lineplot with Family Size")


# - Observations
# 
# - In the Regression Plot we are plotting to observe the linear relationship of purchases with expenses and we observed that increase in expenditure results increase in purchases
# 
# - In the scatter plot we are relating to hue as Income Category , we observe
# that the majority of customer segments are comprised by middle and upper middle class with highest rate of purchases and expenditures, when compared to upper and lower income class
# 
# - In the histogram plot we observe that people with higher education have higher purchases compared to basic education.
# 
# -  In the Lineplot people with family have higher expenditures and purchases compared to people who are single, this  also indicates that the average purchases is an upward trend and for future we can see average rate of expenditures and purchases rising for families compared to bachelors
# 
# - in the lineplot we also observe that there is a sudden spike with family that comprise atleast 2 who have purchases interval between 0 - 5 products atleast.
# 
# 

#  - Binning for Customer Age Feature

# In[ ]:


#since Our target is to find customer segments that  are likely to buy the product dividing our Customer Age into Segments by binning
#binning for Income column
min_value = df_cust5['Customer_Age'].min()
max_value = df_cust5['Customer_Age'].max()
age_range = (np.round(max_value - min_value))
bins = 4
bin_width = int(np.round(age_range/bins))


# In[ ]:



#printing our values
print("the max range is ,",max_value)
print("the min range is ,",min_value)
print("the age range is ,",age_range)
print("the bin width is ,",bin_width)


# In[ ]:


#plotting for histogram before binning  and afte binning
values = df_cust5.Customer_Age.values
fig,ax = plt.subplots(ncols = 2,figsize = (15,7))
fig.suptitle("Age Binning plots",fontsize = 30)
ax[0].set_title('Before Bining Distribution',fontsize = 15)
ax[1].set_title('After binning Distribution',fontsize = 15)
plt.hist( values,edgecolor="red", bins= 5)
sn.histplot(ax=ax[0],x = 'Customer_Age',data=df_cust5)


# - Observation
#  - Before binnning we observe that we have peaks with irregular intervals of data points.
# 
#  - after binning our intervals have been generalised and even peaks have been reduced  with a normal distribution

# In[ ]:


#locating the range for binning using numpy library
np.linspace(min_value,max_value,5)


# ## dividing Age  in 4 catogories as customer segment
# 
# * The Silent Generation: Born 1928-1945 (78-84 years old)
# * Boomers: Born 1946-1964 (59-70 years old)
# * Gen X: Born 1965-1980 (43-56 years old)
# * Millennials: Born 1981-1996 (27-42 years old)

# In[ ]:


#creating bins for Age columns

df_cust5['Customer_Segment'] = pd.cut(df_cust5['Customer_Age'],bins = [26,42,56,70,84],labels = ['Millennial','Gen X','Boomer','Silent'],right=False,include_lowest=True)
df_cust5.head()


# In[ ]:


#dropping the Customer Age feature as we have got age type
df_cust5.drop(['Customer_Age'],axis = 1,inplace = True)


# In[ ]:


df_cust5.head()


# In[ ]:


#checking with the counts of our feature customer segment
df_cust5.Customer_Segment.value_counts()


# In[ ]:


#plotting with Age Distribution With Expenses and purchases
fig,ax = plt.subplots(1,2,figsize = (15,10))
fig.suptitle("Customer Type Purchases - Expenses Distribution",fontsize = 20)
ax[0].set_title('Customer Age Expenses - Purchases Distribution',fontsize = 15)
ax[1].set_title('Customer Distribution',fontsize = 15)
sn.scatterplot(ax=ax[0],data = df_cust5,x='Expenses',y = 'Purchases', hue ='Customer_Segment',markers = True,
               palette="bright")
plt.pie(df_cust5.Customer_Segment.value_counts(),
      labels=['Gen X','Boomer','Millennial','Silent'], autopct= '%.2f%%',shadow=True,
      colors = ['#FFCB9A','#D2E8E3'],explode= [0.1,0.0,0.0,0.0],
      textprops={'size':'large',
                 'fontweight':'bold'})


# - Observations
# 
# - In the customer segments scatterplot we observe that majority of expenses and purchases are done by GenX and Boomers, that is customer in the range of 42 - 56 yrs and 56 - 70yrs, when compared to customers who are in millenials and Silent  .
# 
# - In the pie chart we can relate to the scatterplot that Gen X and Boomer have higher expenses and purchases as they are in higher percentage when compared to millenials and silent customers with least percentage distribution

# In[ ]:


df_cust5.Acceptance.value_counts()


# In[ ]:


#checking with customers who have accepted campaign programs
fig,ax = plt.subplots(1,2,figsize =(15,8))
fig.suptitle("Customer Campaign Analytics",fontsize = 30)
ax[0].set_title('Customer Campaign Distribution',fontsize = 15)
ax[1].set_title('Campaign Donut Distribution',fontsize = 15)
sn.histplot(ax=ax[0],data= df_cust5,x = 'Acceptance',hue='Customer_Segment',multiple = 'stack',
            palette = 'rocket')
labels = '0','1','2','3','4'
colors =['#93BFB7', '#13678A', '#45C4B0',
          '#9AEBA3', '#DAFDBA']
explode = (0.02, 0.02, 0.02, 0.02, 0.02)
plt.pie(df_cust5.Acceptance.value_counts(),autopct='%1.1f%%', pctdistance=0.85,
       labels=labels,colors = colors,explode = explode,shadow = True,
          textprops={'size':'large',
                 'fontweight':'bold'})
# draw circle
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
plot = plt.gcf()
# Adding Circle in Pie chart
plot.gca().add_artist(centre_circle)

# Displaying Chart
plt.show()
plt.tight_layout()


# - Observations
# - In the campaign Analytics, we observe that 79 percent of have accepted the  offer in the first campaign, that includes majority of Gen X and Boomers compared to while the millenials and silent are in minority
# 
# - In the second campaign we observe that only 14.5 percent have accepted the offer and in majority are Gen X and Boomers compared to silent and millennials
# 
# - In the third campaign the percentage is around 3.5 percent comparatively to the first two campaigns, offer accepted is lesser and almost all the age groups share the same distribution
# 
# - In the fourth campaign the percentage is around 2.0 percent, we observe that the people are not interested in the offers of any age group, even distribution of acceptance to offers is lower than other campaigns
# 
# - In the fifth campaign the percentage of customers accepted offer is just 0.5 percent,this also tells us that people might be interested in better    offers and this could attract customers of all age groups

# In[ ]:


df_cust5.head()


# In[ ]:


#plotting Income segment and Age segment with day and  year engagement with the company
fig,ax = plt.subplots(nrows = 2,ncols = 2,figsize = (15,10))
fig.suptitle(" Day Engaged - Year Enrollment Distribution",fontsize = 20)
ax[0,0].set_title('Income Class - Day Engaged Distribution',fontsize = 15)
ax[0,1].set_title('Customer Segment - Day Engaged Distribution',fontsize = 15)
ax[1,0].set_title('Income Class - Years Enrolled Distribution',fontsize = 15)
ax[1,1].set_title('Customer Segment - Years Enrolled Distribution',fontsize = 15)
sn.histplot(ax=ax[0,0],data = df_cust5,x ='day_engaged', hue ='Income_Category',
               palette="Paired",multiple = 'dodge')
sn.histplot(ax=ax[0,1],data = df_cust5,x ='day_engaged', hue ='Customer_Segment',
               palette="Paired",multiple = 'dodge')
sn.histplot(ax=ax[1,0],data = df_cust5,x ='Enr_Freq', hue ='Income_Category',
               palette="Paired",multiple = 'dodge')
sn.histplot(ax=ax[1,1],data = df_cust5,x ='Enr_Freq', hue ='Customer_Segment',
               palette="Paired",multiple = 'dodge')
plt.tight_layout()



# - Observations
# 
# -  the first plot shows us that according to income categories for how many days the customers have been enrolled, and we observe that, customers from middle class and upper middle class have been enrolled for quite a long time and are in majority, followed by other classes with least enrolment from upper class
# 
# - the second plot shows us that according to customer type for how any days the customers have enrolled and we observe that,customers of age group GenX and Boomers have been with the company and are in majority followed by other classes with least from upper class.
# 
# - the third plot shows us according income classes for how many years customers have enrolled with the company, and we observe that, customers enrolled from 9 years are from middle and upper middle class and respectively same with customers enrolled from 10 and 11 years,we also observe that the frequency of customers from lower and upper class is least.
# 
# - the fourth plot shows us according customer segment classes for how many years customers have enrolled with the company, and we observe that, customers enrolled from are from age groups GenX and Boomer with high frequency of customers comparatively to other age groups with customers from silent age group being least
# 
# - these plots also gives us information that most of our customers are from middle and upper middle class within age groups GenX to Boomers,as they share highest frequency comparatively to other classes in Income or Age groups

# In[ ]:


#checking with value counts with income category and family
df_cust5.groupby('Income_Category')['Family'].sum().sort_values(ascending = False)


# In[ ]:


df_cust5.groupby('Customer_Segment')['Family'].sum().sort_values(ascending = False)


# In[ ]:


#plotting with Income category and Customer Segment With Family
fig,ax = plt.subplots(nrows = 2,ncols = 3,figsize = (15,10))
fig.suptitle(" Income Class - Customer Segment- Family Distribution",fontsize = 20)
ax[0,0].set_title('Income Class-Family',fontsize = 15)
ax[0,1].set_title('Customer Segment-Family',fontsize = 15)
ax[1,0].set_title('Income Class-Purchases-Family',fontsize = 15)
ax[1,1].set_title('Customer Segment-Purchases-Family',fontsize = 15)
ax[0,2].set_title('Income Segment-Family Division',fontsize = 15)
ax[1,2].set_title('Customer Age Segment-Family Division',fontsize = 15)
sn.histplot(ax=ax[0,0],data = df_cust5,x ='Income_Category', hue ='Family',
               palette="Paired",multiple = 'dodge')
sn.histplot(ax=ax[0,1],data = df_cust5,x ='Customer_Segment', hue ='Family',
               palette="Paired",multiple = 'dodge')
sn.stripplot(ax=ax[1,0],data = df_cust5,x ='Family',y = 'Purchases', hue ='Income_Category',
               palette="Paired")
sn.stripplot(ax=ax[1,1],data = df_cust5,x ='Family',y = 'Purchases', hue ='Customer_Segment',
               palette="Paired",)
colors =['#93BFB7', '#13678A', '#45C4B0',
          '#9AEBA3']
explode = (0.1, 0.00, 0.00, 0.00,)

labels = 'Middle','Upper-Middle','lower','upper' #adding labels

grp_family = df_cust5.groupby('Income_Category')['Family'].sum().sort_values(ascending = False) #grouping by income category and family
ax[0,2].pie(grp_family,colors = colors,explode = explode,
            autopct='%1.1f%%', textprops={'size':'large',
                 'fontweight':'bold'},shadow = True,labels = labels,startangle = 180,
            pctdistance = 0.80)
labels2 = 'GenX','Boomer','Millennial','Silent'
grp_segment = df_cust5.groupby('Customer_Segment')['Family'].sum().sort_values(ascending = False) #grouping by customer Age segment and family
ax[1,2].pie(grp_segment,colors = colors,explode = explode,
            autopct='%1.1f%%', textprops={'size':'large',
                 'fontweight':'bold'},shadow = True,labels = labels2,startangle = 180,
            pctdistance = 0.80) # plotting a pie chart
plt.tight_layout()


# - Observations
# 
# -  In the first plot we observe that most of the our customers have family members of 2 or 3 and most of them are from Middle and upper middle class followed by lower and upper class
# 
# - In the second plot we observe that most of our customer having family members  of 2 or 3 are from age range from 42-56 years followed by 56-70 years and,with least interaction with silent generation that is 70-85 years.
# 
# - the third plot shows us income division among families and their class and we observe that most  of the families come from middle and upper middle class followed by lower class and with least percentage of upper class
# 
# -  the fourth plot shows us that most of the purchases are made by middle and upper middle class with least purchases  by upper class. we also observe that family with 5 members engage the least when compared to purchases.
# 
# - In the fifth plot we observe that most of the purchases are made by age categories Millennials , GenX and Boomers,with least rate of purchase by silent age categories, we also observe that regardless of age types family members with 5 have least rate of purchase
# 
# - In the sixth plot we observe that that most of the members of family are from age categories GenX followed by Boomers , Millennials and with least percentage silent members in the family.
# 
# - We also observe that 80 percent of our customers are from Lower , Middle and upper Middle class with age ranging from Millennials to baby Boomers, customers from upper class and silent age groups have least purchase rate and come in the 20 percent.

# # MultiVariate Analysis

# In[ ]:


df_cust5.head()


# In[ ]:


#checking with the shape of our dataset
df_cust5.shape


# In[ ]:


#resetting our dataframe
df_cust5.reset_index(inplace = True)


# In[ ]:


#dropping our duplicate index row
df_cust5.drop(['index'],axis = 1)


# In[ ]:


#checking with correlation with varaibles
df_cust5.corr(method = 'pearson')


# In[ ]:


#plotting pearsons coefficient matrix
fig,ax = plt.subplots(figsize = (10,8))
sn.heatmap(df_cust5.corr(),annot=True,fmt='g',cbar=False,linewidths='0.5',linecolor='black',cmap='ocean_r',
           vmin = -1,vmax = 1)


# - observations
# - Day engaged and years enrolled frequency is highly correlated to each other
# 
# - Expenses and purchases are also highly correlated to each other
# 
# 

# In[ ]:


#plotting the pairplot to check the rlationship with numerical features
sn.set_style('darkgrid')
sn.pairplot(df_cust5,
            vars = ['Purchases','day_engaged',
                    'Enr_Freq','Expenses'] )


# - Observations
# - we find Positive relationship of variables with purchases and expenses and positively correlated
# - we also oserve that day engaged and enrollment frequency is having positive relationhip and positively correlated
# 

# In[ ]:


#dropping our one out of correlated features
df_cust6 = df_cust5.copy()
df_cust6.drop(['Expenses','Enr_Freq'],axis = 1 , inplace = True)


# In[ ]:


df_cust6.info()


# In[ ]:


#selecting our features to required index
df_cust6 = df_cust6[['Purchases','day_engaged','Acceptance','Income_Category','Customer_Segment','Family','Education']]


# In[ ]:


df_cust6.head()


# # Feature Scaling

# - Standard Scaler

# In[ ]:


#dividing our features into numerical and categorical
num_df = df_cust6[['Purchases','day_engaged']]
cat_df = df_cust6.iloc[:,2:7]


# In[ ]:



#applying standard scaler on  the dataset
std = StandardScaler()
df_scaled = std.fit_transform(num_df)


# In[ ]:


#our scaled data
df_scaled


# In[ ]:


#storing into our final dataframe
df_final = pd.concat([pd.DataFrame(df_scaled,columns = ['Purchases','day_engaged']),cat_df],axis =1)


# In[ ]:


df_final


# In[ ]:


#converting our categorical columns from string to categorical
df_final['Education'] = df_final['Education'].astype('category')


# In[ ]:


df_final.dtypes


# In[ ]:


df_final.head()


# In[ ]:


# getting dummies for all of our categorical variables
df_final2 = pd.get_dummies(df_final,columns = ['Income_Category','Customer_Segment','Education'])


# In[ ]:


df_final2


# # Feature Selection PCA on Standard Scaler

# In[ ]:


pca = PCA(n_components= 15)
pca_components = pd.DataFrame(pca.fit_transform(df_final2),columns=['pc1','pc2','pc3','pc4','pc5',
                                                                   'pc6','pc7','pc8','pc9','pc10','pc11','pc12','pc13','pc14','pc15'])
pca_components


# In[ ]:


#plotting scatter plot
sn.scatterplot(x=pca_components.pc1,y=pca_components.pc2,palette='dark')


# In[ ]:


#checking with explained variance for our columns
var = (pca.explained_variance_ratio_)
var


# In[ ]:


#checking with cumulative variance
cum_var = pd.DataFrame(np.cumsum(np.round(var,decimals=2)*100))
cum_var


# In[ ]:


#plotting against variance for components
plt.bar(range(1, len(var)+1),var)
plt.xlabel('Number of Components')
plt.ylabel('variance (%)')
plt.title('Explained variance by each component')


# In[ ]:


#plotting scree plot
plt.figure(figsize=(6,6))
plt.plot(cum_var,color = 'blue',marker = 'o')
plt.title('Scree Plot ')
plt.xlabel('Components')
plt.ylabel('cumulative variance')
plt.show()


# - We observe that with 3 components we are getting good variance of above 70% hence we take with components 3

# In[ ]:


#selecting our top 3 features
pca = PCA(n_components=3)
pca_final = pd.DataFrame(pca.fit_transform(df_final2),columns=['pc1','pc2','pc3',])

pca_final


# # Model Building By Clustering Techniques
# 
# 
# 

#  - KMEANS On Standard Scaler
# 
# 

# In[ ]:


#visualizing silhouette score with Kelbow visualizer
model = KMeans(random_state=0,max_iter=500,init = 'k-means++')
visualizer = KElbowVisualizer(model,k =(2,20),metric = 'silhouette',timings = False)
print('Elbow plot for PCA')
visualizer.fit(pca_final)
visualizer.show()
plt.show()


# In[ ]:


#using the sklearn library to view the clusters
range_n_clusters = [2,3,4,5,6,7,8,9]

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(pca_final) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_labels = clusterer.fit_predict(pca_final)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(pca_final, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(pca_final, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values =             sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(pca_final.iloc[:,0], pca_final.iloc[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:,0], centers[:,1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("PCA on Standard Scaled Dataset")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

plt.show()


# - Building Our KMeans with clusters  = 2
# 

# In[ ]:


#fitting our kmeans Algorithm
model_kmeans = KMeans(n_clusters=2,random_state=0,init ='k-means++')
kmeans_pred = model_kmeans.fit_predict(pca_final)


# In[ ]:


#checking with our labels
model_kmeans.labels_


# In[ ]:


#checking with centroids of the clusters
model_kmeans.cluster_centers_


# In[ ]:


#checking with inertia the lower the inertia denser the clusters
model_kmeans.inertia_


# In[ ]:


#labeling our cluster to our original dataset
df_final2['kmeans_label_std'] = model_kmeans.labels_


# In[ ]:


#plotting with with the labels
fig,ax = plt.subplots(ncols = 2 ,figsize = (15,10))
sn.scatterplot(ax = ax[0],x = pca_final.iloc[:,0],y = pca_final.iloc[:,1],data = df_final2,hue = df_final2['Education_Grad'])
sn.scatterplot(ax = ax[1],x = pca_final.iloc[:,0],y = pca_final.iloc[:,1],data = df_final2,hue = df_final2['kmeans_label_std'],palette = 'bright')
ax[0].set_title('Original Classification')
ax[1].set_title('KMeans Classification')


# In[ ]:


# Group data by Clusters (K=2)
df_final2.groupby('kmeans_label_std').agg(['mean'])


# In[ ]:


#saving our average clusters into excel file
kmeans_df = df_final2.groupby('kmeans_label_std').agg(['mean'])
kmeans_df.to_csv('clustering.csv')


#  - building our kmeans with clusters = 3

# In[ ]:


#fitting our kmeans Algorithm
model_kmeans2 = KMeans(n_clusters=3,random_state=0,init ='k-means++')
kmeans_pred2 = model_kmeans2.fit_predict(pca_final)


# In[ ]:


#checking with our labels
model_kmeans2.labels_


# In[ ]:


#checking with centroids of the clusters
model_kmeans2.cluster_centers_


# In[ ]:


#checking with inertia the lower the inertia denser the clusters
model_kmeans2.inertia_


# In[ ]:


#labeling our cluster to our original dataset
df_final2['kmeans_label_std2'] = model_kmeans2.labels_


# In[ ]:


#plotting with with the labels
fig,ax = plt.subplots(ncols = 2 ,figsize = (15,10))
sn.scatterplot(ax = ax[0],x = pca_final.iloc[:,0],y = pca_final.iloc[:,1],data = df_final2,hue = df_final2['Education_Grad'])
sn.scatterplot(ax = ax[1],x = pca_final.iloc[:,0],y = pca_final.iloc[:,1],data = df_final2,hue = df_final2['kmeans_label_std2'],palette = 'bright')
ax[0].set_title('Original Classification')
ax[1].set_title('KMeans Classification')


# In[ ]:


# Group data by Clusters (K=3)
df_final2.groupby('kmeans_label_std2').agg(['mean'])


# - building our kmeans with clusters = 5

# In[ ]:


#fitting our kmeans Algorithm
model_kmeans3 = KMeans(n_clusters=5,random_state=0,init ='k-means++')
kmeans_pred3 = model_kmeans3.fit_predict(pca_final)


# In[ ]:


#checking with our labels
model_kmeans3.labels_


# In[ ]:


#checking with centroids of the clusters
model_kmeans3.cluster_centers_


# In[ ]:


#checking with inertia the lower the inertia denser the clusters
model_kmeans3.inertia_


# In[ ]:


#labeling our cluster to our original dataset
df_final2['kmeans_label_std3'] = model_kmeans3.labels_


# In[ ]:


#plotting with with the labels
fig,ax = plt.subplots(ncols = 2 ,figsize = (15,10))
sn.scatterplot(ax = ax[0],x = pca_final.iloc[:,0],y = pca_final.iloc[:,1],data = df_final2,hue = df_final2['Education_Grad'])
sn.scatterplot(ax = ax[1],x = pca_final.iloc[:,0],y = pca_final.iloc[:,1],data = df_final2,hue = df_final2['kmeans_label_std3'],palette = 'bright')
ax[0].set_title('Original Classification')
ax[1].set_title('KMeans Classification')


# In[ ]:


# Group data by Clusters (K=5)
df_final2.groupby('kmeans_label_std3').agg(['mean'])


# #Hierarchial Clustering

# In[ ]:


for methods in ['single','complete','centroid','average','weighted','ward','median']:
  plt.figure(figsize = (15,10))
  dict = {'fontsize':20,'fontweight':12,'color':'blue'}
  plt.title('Visualizing data clustering,Method- {}'.format(methods),fontdict=dict)
  Dendrogram = sch.dendrogram(sch.linkage(pca_final,method=methods,optimal_ordering=False))


# In[ ]:


# Applying Different Linkages using Euclidean Method for distance Calculation
n_clusters = [2,3,4,5,6,7,8]
for n_clusters in n_clusters:
  for linkages in ["complete","average","single","ward"]:
    for affinities in ["euclidean","l1","l2","manhattan","cosine"]:

      hie_cluster1 = AgglomerativeClustering(n_clusters = n_clusters, linkage = linkages)
      hie_labels1 = hie_cluster1.fit_predict(pca_final)
      silhouette_score1 = silhouette_score(pca_final,hie_labels1)
      print("For n_clusters =", n_clusters,"The average silhouette_score with linkage-",linkages,"and Affinity-",affinities,':',silhouette_score1)
      print()


# - Observation
# 
# With standard scaler we are getting silhouette score of 0.36 with linkage as single and affinity as manhattan

# In[ ]:


#Fitting our agglomerative clustering with optimal Parameters
agg_clus = AgglomerativeClustering(n_clusters = 2,linkage = 'single',affinity='manhattan')
pred_hie = agg_clus.fit_predict(pca_final)
print(pred_hie.shape)
pred_hie


# In[ ]:


#calculating our silhouette score
(silhouette_score(pca_final, agg_clus.labels_)*100).round(3)


# In[ ]:


#fitting our labels on our dataframe
df_final2['hie_labels'] = agg_clus.labels_


# In[ ]:


# Group data by Clusters
df_final2.groupby('hie_labels').agg(['mean'])


# In[ ]:


#plotting with with the labels
fig,ax = plt.subplots(ncols = 2 ,figsize = (15,10))
sn.scatterplot(ax = ax[0],x = pca_final.iloc[:,0],y = pca_final.iloc[:,1],data = df_final2,hue = df_final2['Education_Grad'])
sn.scatterplot(ax = ax[1],x = pca_final.iloc[:,0],y = pca_final.iloc[:,1],data = df_final2,hue = df_final2['hie_labels'],palette = 'bright')
ax[0].set_title('Original Classification')
ax[1].set_title('Agglomerative Classification')


# # Density Based Clustering (DBSCANS)

# In[ ]:


#converting our pca to numpy
X = pca_final.to_numpy()


# In[ ]:


#defining our parameters
epsilons = np.linspace(0.01,1,num = 15)
min_samples = np.arange(2,20,step = 3)
combinations = list(product(epsilons,min_samples))
n = len(combinations)


# In[ ]:


#defining our parameters with manual fuction and gridsearching for best parameters
def get_scores_and_labels(combinations, X):
  scores = []
  all_labels_list = []

  for i, (eps, num_samples) in enumerate(combinations):
    dbscan_cluster_model = DBSCAN(eps=eps, min_samples=num_samples).fit(X)
    labels = dbscan_cluster_model.labels_
    labels_set = set(labels)
    num_clusters = len(labels_set)
    if -1 in labels_set:
      num_clusters -= 1

    if (num_clusters < 2) or (num_clusters > 50):
      scores.append(-10)
      all_labels_list.append('bad')
      c = (eps, num_samples)
      print(f"Combination {c} on iteration {i+1} of {n} has {num_clusters} clusters. Moving on")
      continue

    scores.append(silhouette_score(X, labels))
    all_labels_list.append(labels)
    print(f"Index: {i}, Score: {scores[-1]}, Labels: {all_labels_list[-1]}, NumClusters: {num_clusters}")

  best_index = np.argmax(scores)
  best_parameters = combinations[best_index]
  best_labels = all_labels_list[best_index]
  best_score = scores[best_index]

  return {'best_epsilon': best_parameters[0],
          'best_min_samples': best_parameters[1],
          'best_labels': best_labels,
          'best_score': best_score}

best_dict = get_scores_and_labels(combinations, X)


# In[ ]:


#locating our best parameters for dbscans
best_dict


# In[ ]:


#fitting our dbscan optimal parameters
dbscan = DBSCAN(eps = 0.85,min_samples=2)
dbscan.fit(pca_final)


# In[ ]:


#checking with labels
dbscan.labels_


# In[ ]:


#adding the labels to our dataframe
df_final2['dbscan_labels'] = dbscan.labels_


# In[ ]:


df_final2.groupby('dbscan_labels').agg(['mean'])


# In[ ]:


#plotting with with the labels
fig,ax = plt.subplots(ncols = 2 ,figsize = (15,10))
sn.scatterplot(ax = ax[0],x = pca_final.iloc[:,0],y = pca_final.iloc[:,1],data = df_final2,hue = df_final2['Education_Grad'])
sn.scatterplot(ax = ax[1],x = pca_final.iloc[:,0],y = pca_final.iloc[:,1],data = df_final2,hue = df_final2['dbscan_labels'],palette = 'bright')
ax[0].set_title('Original Classification')
ax[1].set_title('DBSCAN Classification')


# # PCA on MinMax Scaler

# In[ ]:


#dividing our features into numerical and categorical
num_df = df_cust6[['Purchases','day_engaged']]
cat_df = df_cust6.iloc[:,2:7]


# In[ ]:


#applying MinMax scaler on  the dataset
min = MinMaxScaler()
df_minmax = min.fit_transform(num_df)


# In[ ]:


#our scaled data
df_minmax


# In[ ]:


#storing into our final dataframe
df_final_minmax = pd.concat([pd.DataFrame(df_minmax,columns = ['Purchases','day_engaged']),cat_df],axis =1)


# In[ ]:


#converting our categorical columns from string to categorical
df_final_minmax['Education'] = df_final_minmax['Education'].astype('category')


# In[ ]:


df_final_minmax.dtypes


# In[ ]:


# getting dummies for all of our categorical variables
df_final_minmax2 = pd.get_dummies(df_final_minmax,columns = ['Income_Category','Customer_Segment','Education'])


# In[ ]:


pca2 = PCA(n_components= 15)
pca_components2 = pd.DataFrame(pca2.fit_transform(df_final_minmax2),columns=['pc1','pc2','pc3','pc4','pc5',
                                                                   'pc6','pc7','pc8','pc9','pc10','pc11','pc12','pc13','pc14','pc15'])
pca_components2


# In[ ]:


#plotting scatter plot
sn.scatterplot(x=pca_components2.pc1,y=pca_components2.pc2,palette='dark')


# In[ ]:


#checking with explained variance for our columns
var = (pca2.explained_variance_ratio_)
var


# In[ ]:


#checking with cumulative variance
cum_var = pd.DataFrame(np.cumsum(np.round(var,decimals=2)*100))
cum_var


# In[ ]:


#plotting against variance for components
plt.bar(range(1, len(var)+1),var)
plt.xlabel('Number of Components')
plt.ylabel('variance (%)')
plt.title('Explained variance by each component')


# In[ ]:


#plotting scree plot
plt.figure(figsize=(6,6))
plt.plot(cum_var,color = 'blue',marker = 'o')
plt.title('Scree Plot ')
plt.xlabel('Components')
plt.ylabel('cumulative variance')
plt.show()


# In[ ]:


#selecting our top 3 features
pca_2 = PCA(n_components= 3)
pca_final2 = pd.DataFrame(pca_2.fit_transform(df_final_minmax2),columns=['pc1','pc2','pc3'])
pca_final2


# - KMEANS On Min-max Scaler
# 
# 
# 

# In[ ]:


#visualizing silhouette score with Kelbow visualizer
model = KMeans(random_state=0,max_iter=500,init = 'k-means++')
visualizer = KElbowVisualizer(model,k =(2,20),metric = 'silhouette',timings = False)
print('Elbow plot for PCA')
visualizer.fit(pca_final2)
visualizer.show()
plt.show()


# In[ ]:


#using the sklearn library to view the clusters
range_n_clusters = [2,3,4,5,6,7,8,9]

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(pca_final2) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_labels = clusterer.fit_predict(pca_final2)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(pca_final2, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(pca_final2, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values =             sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(pca_final2.iloc[:,0], pca_final2.iloc[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:,0], centers[:,1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("PCA on MinMAx Scaled Dataset")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')


# In[ ]:


#fitting our kmeans Algorithm
model_kmeans = KMeans(n_clusters=5,random_state=0,init ='k-means++')
kmeans_pred = model_kmeans.fit_predict(pca_final2)


# In[ ]:


#checking with our labels
model_kmeans.labels_


# In[ ]:


#checking with centroids of the clusters
model_kmeans.cluster_centers_


# In[ ]:


#checking wiht inertia the lower the inertia denser the clusters
model_kmeans.inertia_


# In[ ]:


#labeling our cluster to our original dataset
df_final_minmax2['kmeans_label_min'] = model_kmeans.labels_


# In[ ]:


#plotting with with the labels
fig,ax = plt.subplots(ncols = 2 ,figsize = (15,10))
sn.scatterplot(ax = ax[0],x = pca_final2.iloc[:,0],y =  pca_final2.iloc[:,1],data =  df_final_minmax2,hue =  df_final_minmax2['Education_Grad'])
sn.scatterplot(ax = ax[1],x = pca_final2.iloc[:,0],y =  pca_final2.iloc[:,1],data =  df_final_minmax2,hue = df_final_minmax2['kmeans_label_min'],
               palette = 'bright')
ax[0].set_title('Original Classification')
ax[1].set_title('KMEANS min max Classification')


# In[ ]:


# Group data by Clusters (K=2)
df_final_minmax2.groupby('kmeans_label_min').agg(['mean'])


# # Hierarchial Clustering

# In[ ]:


for methods in ['single','complete','centroid','average','weighted','ward','median']:
  plt.figure(figsize = (15,10))
  dict = {'fontsize':20,'fontweight':12,'color':'blue'}
  plt.title('Visualizing data clustering,Method- {}'.format(methods),fontdict=dict)
  Dendrogram = sch.dendrogram(sch.linkage(pca_final2,method=methods,optimal_ordering=False))


# In[ ]:


# Applying Different Linkages using Euclidean Method for distance Calculation
n_clusters = [2,3,4,5,6,7,8]
for n_clusters in n_clusters:
  for linkages in ["complete","average","single","ward"]:
    for affinities in ["euclidean","l1","l2","manhattan","cosine"]:

      hie_cluster2 = AgglomerativeClustering(n_clusters = n_clusters, linkage = linkages)
      hie_labels2 = hie_cluster2.fit_predict(pca_final2)
      silhouette_score1 = silhouette_score(pca_final2,hie_labels2)
      print("For n_clusters =", n_clusters,"The average silhouette_score with linkage-",linkages,"and Affinity-",affinities,':',silhouette_score1)
      print()


# - Observation
# 
# With Minmax Scaler For n_clusters = 3 The average silhouette_score with linkage- single and Affinity- l2 : 0.51

# In[ ]:


#Fitting our agglomerative clustering with optimal Parameters
agg_clus = AgglomerativeClustering(n_clusters = 3,linkage = 'single',affinity='manhattan')
pred_hie2 = agg_clus.fit_predict(pca_final2)
print(pred_hie2.shape)
pred_hie2


# In[ ]:


#calculating our silhouette score
(silhouette_score(pca_final2, agg_clus.labels_)*100).round(3)


# In[ ]:


#fitting our labels on our dataframe
df_final_minmax2['hie_labels'] = agg_clus.labels_


# In[ ]:


# Group data by Clusters
df_final_minmax2.groupby('hie_labels').agg(['mean'])


# In[ ]:


#plotting with with the labels
fig,ax = plt.subplots(ncols = 2 ,figsize = (15,10))
sn.scatterplot(ax = ax[0],x = pca_final2.iloc[:,0],y = pca_final2.iloc[:,1],data = df_final_minmax2,hue = df_final_minmax2['Education_Grad'])
sn.scatterplot(ax = ax[1],x = pca_final2.iloc[:,0],y = pca_final2.iloc[:,1],data = df_final_minmax2,hue = df_final_minmax2['hie_labels'],
               palette = 'bright')
ax[0].set_title('Original Classification')
ax[1].set_title('Agglomerative Classification')


# # Density Based Clustering (DBSCANS) on MinMax Scaler

# In[ ]:


#converting our pca to numpy
X = pca_final2.to_numpy()


# In[ ]:


#defining our parameters
epsilons = np.linspace(0.01,1,num = 15)
min_samples = np.arange(2,20,step = 3)
combinations = list(product(epsilons,min_samples))
n = len(combinations)


# In[ ]:


#defining our parameters with manual fuction and gridsearching for best parameters
def get_scores_and_labels(combinations, X):
  scores = []
  all_labels_list = []

  for i, (eps, num_samples) in enumerate(combinations):
    dbscan_cluster_model = DBSCAN(eps=eps, min_samples=num_samples).fit(X)
    labels = dbscan_cluster_model.labels_
    labels_set = set(labels)
    num_clusters = len(labels_set)
    if -1 in labels_set:
      num_clusters -= 1

    if (num_clusters < 2) or (num_clusters > 10):
      scores.append(-10)
      all_labels_list.append('bad')
      c = (eps, num_samples)
      print(f"Combination {c} on iteration {i+1} of {n} has {num_clusters} clusters. Moving on")
      continue

    scores.append(silhouette_score(X, labels))
    all_labels_list.append(labels)
    print(f"Index: {i}, Score: {scores[-1]}, Labels: {all_labels_list[-1]}, NumClusters: {num_clusters}")

  best_index = np.argmax(scores)
  best_parameters = combinations[best_index]
  best_labels = all_labels_list[best_index]
  best_score = scores[best_index]

  return {'best_epsilon': best_parameters[0],
          'best_min_samples': best_parameters[1],
          'best_labels': best_labels,
          'best_score': best_score}

best_dict = get_scores_and_labels(combinations, X)


# In[ ]:


#locating our best parameters for dbscans
best_dict


# In[ ]:


#fitting our dbscan optimal parameters
dbscan = DBSCAN(eps = 0.85,min_samples=2)
dbscan.fit(pca_final2)


# In[ ]:


#checking with labels
dbscan.labels_


# In[ ]:


#adding the labels to our dataframe
df_final_minmax2['dbscan_labels'] = dbscan.labels_


# In[ ]:


df_final_minmax2.groupby('dbscan_labels').agg(['mean'])


# In[ ]:


#plotting with with the labels
fig,ax = plt.subplots(ncols = 2 ,figsize = (15,10))
sn.scatterplot(ax = ax[0],x = pca_final2.iloc[:,0],y = pca_final2.iloc[:,1],data = df_final_minmax2,hue = df_final_minmax2['Education_Grad'])
sn.scatterplot(ax = ax[1],x = pca_final2.iloc[:,0],y = pca_final2.iloc[:,1],data = df_final_minmax2,hue = df_final_minmax2['dbscan_labels'],palette = 'bright')
ax[0].set_title('Original Classification')
ax[1].set_title('DBSCAN Classification')


# # KPrototypes  ALgorithm

# In[ ]:


df_cust6.head()


# In[ ]:


df_cust6.dtypes


# In[ ]:


#importing kmodes library
from kmodes.kprototypes import KPrototypes


# In[ ]:


#converting our string to categorical variable
df_cust6['Education'] = df_cust6['Education'].astype('category')


# In[ ]:


#saving our data into array
array = df_cust6.values


# In[ ]:


#converting our integer types to float
array[:,0] = array[:,0].astype(float)
array[:,1] = array[:,1].astype(float)
array[:,2] = array[:,2].astype(float)
array[:,5] = array[:,5].astype(float)


# In[ ]:


array


# In[ ]:


#locating optimal clusters with optimal k value
cost = []
for cluster in range(1,10):
  try:
    k_pro = KPrototypes(n_clusters=cluster,init = 'huang',random_state=0)
    k_pro.fit_predict(array,categorical = [3,4,6])
    cost.append(k_pro.cost_)
    print('cluster initiation: {}'.format(cluster))
  except:
    break


# In[ ]:


#plotting the elbow plot
plt.plot(cost)
plt.xlabel('K')
plt.ylabel('cost')
plt.show


# In[ ]:


#locating clusters with k = 3
from kneed import KneeLocator
cost_knee_c3 = KneeLocator(x = range(1,10),y = cost,S = 0.1,curve = 'convex',direction='decreasing',online = True)

k_cost_c3 = cost_knee_c3.elbow
print("elbow at k = ",f'{k_cost_c3:0f} clusters')


# In[ ]:


#fitting our clusters with optimal clusters
k_prot = KPrototypes(n_clusters= 3,init='huang',random_state=0)
df_cust6['k_pro_labels'] = k_prot.fit_predict(array,categorical = [3,4,6])


# In[ ]:


#locating centroids
k_prot.cluster_centroids_


# In[ ]:


#computing labels
k_prot.labels_


# In[ ]:


#plotting with with the labels
fig,ax = plt.subplots(ncols = 2 ,figsize = (15,10))
sn.scatterplot(ax = ax[0],x = 'day_engaged',y = 'Purchases',data = df_cust6,hue = 'Education')
sn.scatterplot(ax = ax[1], x = 'day_engaged',y = 'Purchases',data = df_cust6,hue = 'k_pro_labels',palette = 'bright')
ax[0].set_title('Original Classification')
ax[1].set_title('Kprototype Classification')


# In[ ]:


#checking with the average
df_cust6.groupby('k_pro_labels').agg(['mean'])


# In[ ]:


df_cust6['k_pro_labels'].value_counts().plot(kind = 'bar')


# In[ ]:


df_cust6.groupby('k_pro_labels').agg(lambda x: pd.Series.mode(x).iat[0])[['Income_Category','Customer_Segment','Education','Family','Purchases',
                                                                          'day_engaged']]


# # Model Evaluation

#  - After all observation and application different unsupervised on different types of scaling we observe that we are getting good results with kmeans on standard scaler and we will use kmeans with standard scaler for evaluating models

# In[ ]:


#extracting our csv file
df_csv = df_final2.drop(['hie_labels','dbscan_labels'],axis = 1)
df_csv.to_csv('clustering.csv')


# In[ ]:


#building our model for classification
df_cust7 = df_cust6.copy()
df_cust7 = pd.get_dummies(data =df_cust7,columns=['Income_Category','Customer_Segment','Education'])


# In[ ]:


#attaching our kmeans model to our new dataframe
df_cust7.drop(['k_pro_labels'],axis = 1,inplace = True)
df_cust7['clusters'] = model_kmeans2.labels_


# In[ ]:


df_cust7


# In[ ]:


#checking with the distribution of clusters
fig,ax = plt.subplots(1,2,figsize = (15,10))
sn.countplot(ax = ax [0],x = df_cust7['clusters'],palette = 'bright')
colors =['#9EF8EE','#348888']
fig.suptitle('Cluster Analytics',fontsize = 40)
df_cust7.clusters.value_counts().plot(ax = ax [1],kind ='pie', autopct= '%.2f%%',shadow=True,explode = [0.1,0.0,0.0],colors=colors)
plt.tight_layout()


# - Observation
# We observe  that the frequency is highest in cluster 2 while cluster 3 is moderately lower compared to cluster 2 and cluster1 being lowest

# In[ ]:


#checking with our second cluster
df_final2[df_final2['kmeans_label_std2']==0]


# In[ ]:


#checking with our second cluster
df_final2[df_final2['kmeans_label_std2']==1]


# In[ ]:


#checking with our third cluster
df_final2[df_final2['kmeans_label_std2']==2]


# In[ ]:


#attaching our kmeans model to our new dataframe
df_cust7.drop(['k_pro_labels'],axis = 1,inplace = True)
df_cust7['clusters'] = model_kmeans2.labels_


# In[ ]:


#checking with the distribution of clusters according to customer characteristics
for d in df_cust7:
  grid = sn.FacetGrid(df_cust7,col = 'clusters')
  grid = grid.map(plt.hist,d )
  plt.show()


# In[ ]:


#plotting with clusters with customers accepted the offers
sn.countplot(data = df_cust7,x = 'Acceptance',hue = 'clusters',palette = 'bright')


#  - Observation
# 
#  We can observe that the response with the first campaign response has been the best with cluster 2, with moderate response to the cluster 3 with cluster 1 being lowest, but as we go ahead the interaction of customers with every cluster in the campaign offers has been dropping and in the fifth campaign the interaction has been least

# In[ ]:


#plotting with clusters with customers with members in the Family
sn.scatterplot(data = df_cust7,x = 'Family',y = 'Purchases',hue = 'clusters',palette = 'bright')


#  - Observation
#   
#  we observe that a family that have members above 4 are less in numbers and have the least interaction to purchases with the company , the average highest purchases are done by members in the family between 2 - 3 range

# In[ ]:


#evaluating purchases according to income categories
fig,ax = plt.subplots(nrows = 2,ncols = 2,figsize = (15,10))
sn.scatterplot(ax = ax [0,0],data = df_cust7,x =  'Income_Category_lower',y = 'Purchases',hue = 'clusters',palette = 'bright')
sn.scatterplot(ax = ax [0,1],data = df_cust7,x =   'Income_Category_middle',y = 'Purchases',hue = 'clusters',palette = 'bright')
sn.scatterplot(ax = ax [1,0],data = df_cust7,x =  'Income_Category_upper_middle',y = 'Purchases',hue = 'clusters',palette = 'bright')
sn.scatterplot(ax = ax [1,1],data = df_cust7,x =   'Income_Category_upper',y = 'Purchases',hue = 'clusters',palette = 'bright')


#  - Observation
# 
#  -In this we observe that most of our customers are from middle and upper middle class backgrounds with surplus purchases on products and if we observe the clusters they are placed also accordingly
# 
#  - Secondly we observe that purchases of products are least with the customer backgrounds with lower and upper class, this could be the reson for the price of the product for customers with lower income and to upper class it might be they are not happy with the price set for product and feel that products are overpriced
# 
# 

# In[ ]:


#evaluating purchases according to age categories
fig,ax = plt.subplots(nrows = 2,ncols = 2,figsize = (15,10))
sn.scatterplot(ax = ax [0,0],data = df_cust7,x =  'Customer_Segment_Millennial',y = 'Purchases',hue = 'clusters',palette = 'bright')
sn.scatterplot(ax = ax [0,1],data = df_cust7,x =    'Customer_Segment_Gen X',y = 'Purchases',hue = 'clusters',palette = 'bright')
sn.scatterplot(ax = ax [1,0],data = df_cust7,x =   'Customer_Segment_Boomer',y = 'Purchases',hue = 'clusters',palette = 'bright')
sn.scatterplot(ax = ax [1,1],data = df_cust7,x =   'Customer_Segment_Silent',y = 'Purchases',hue = 'clusters',palette = 'bright')


#  - Observation
# 
#  - When it comes to age groups we observe that most of the customers with highest Purchases are from GenX to Boomers category of age, we have a potential that these people mostly be interest in household products.
# 
#  - we also observe that customers who are in age category millennials and silent have moderate purchases to the products so this indicates that these groups might just be very particular and have no association with other products
# 

# In[ ]:


#evaluating purchases according to age categories
fig,ax = plt.subplots(ncols = 3,figsize = (15,10))
sn.scatterplot(ax = ax [0],data = df_cust7,x = 'Education_Grad',y = 'Purchases',hue = 'clusters',palette = 'bright')
sn.scatterplot(ax = ax [1],data = df_cust7,x =     'Education_PostGrad',y = 'Purchases',hue = 'clusters',palette = 'bright')
sn.scatterplot(ax = ax [2],data = df_cust7,x =   'Education_Undergrad',y = 'Purchases',hue = 'clusters',palette = 'bright')


#  - Observation
# 
#   when it comes to educaation categories we observe that most of the purchases are done by customers with higher education compared to customers who are just in their undergraduation

#  - Conclusion With Clusters

# In[ ]:


# Group data by Clusters (K=3)
df_final2.groupby('kmeans_label_std2').agg(['mean'])


# ## Cluster 1
# - In the first cluster we observe that the purchases are moderate with least customers engaged on all days with the company but have highest acceptance to campaign offers and mostly they come from upper  middle  class background,most of them fall in the age groups of Gen X and and Boomers with higher background of education.
# 
# ## Cluster 2
# - In the second cluster we observe that the purchases are lowest and moderately less customers engaged on all days with the company and the acceptance ratio to offers or campaigns is lesser compared to first cluster,most of the customers are from lower and middle class and most of them fall in the age category of Gen X and Boomers having higher education background ,we can also say that most of the families are having members near to 3
# 
# ## Cluster 3
# - In the third cluster we observe that the purchases are highest with most of our customers engaged with the company on daily basis and have a moderate acceptance to any offers or campaigns, in the third cluster we observe that most of the customer come from lower, middle and upper middle class and even in this cluster most of them fall in the age categories of GenX and  Boomers with highest education qualification.
# 
# 
# 
# ## Summary
# 
# We can conclude after observing all clusters are middle and upper middle class and lie in the age groups from Gen X to Boomers and these category of customers comprise 80% of our segmentation of our company and marketing product n this sector would benefit the company

# # Davies-Bouldin index
# 
# The davies_bouldin_score function from the sklearn.metrics module computes the Davies-Bouldin index for a given set of data and labels. The index is calculated as the average similarity between each cluster and its most similar cluster, divided by the average similarity between each cluster and its most dissimilar cluster. The resulting score ranges from 0 to infinity, with lower values indicating better clustering.
# 
# 
# 
# 

# In[ ]:


from sklearn.metrics import davies_bouldin_score

# X is the data matrix
# k is the number of clusters
kmeans = KMeans(n_clusters=3, random_state=0).fit(df_final2)
labels = kmeans.labels_

# Calculate Davies-Bouldin index
db_index = davies_bouldin_score(df_final2, labels)
print("The Davies-Bouldin index is:", db_index)


# # Model Building
# 

# In[ ]:


#building our model for classification
df_cust7 = df_cust6.copy()
df_cust7 = pd.get_dummies(data =df_cust7,columns=['Income_Category','Customer_Segment','Education'])


# In[ ]:


df_cust7.head()


# In[ ]:


#attaching our kmeans model to our new dataframe
df_cust7.drop(['k_pro_labels'],axis = 1,inplace = True)


# In[ ]:


#checking with the shape of the data
df_cust7.shape


# In[ ]:


#checking the data
df_cust7


# In[ ]:


# Group data by Clusters (K=3)
df_cust7.groupby('clusters').agg(['mean'])


# # Model Classification

# In[ ]:


final_df = df_cust7.copy().drop(['clusters'],axis = 1)


# In[ ]:


final_df.shape


# In[ ]:


final_df.head()


# In[ ]:


#splitting our variable into independent and dependent variables
X = pca_final.iloc[:,[0,1]].values
y = model_kmeans2.labels_


# In[ ]:


#plotting our kmeans centroids on pca  with standard scaler
plt.figure(figsize = (8,8))
plt.scatter( X[y==0,0],  X[y==0,1],s = 50 ,c = 'green',label = 'cluster1')
plt.scatter(X[y==1,0],X[y==1,1],s = 50 ,c = 'blue',label = 'cluster2')
plt.scatter(X[y==2,0],X[y==2,1],s = 50 ,c = 'yellow',label = 'cluster3')


#plot the centroids
plt.scatter(model_kmeans2.cluster_centers_[:,0],model_kmeans2.cluster_centers_[:,1],s = 100,c = 'black',label = 'centroids')
plt.title("Centroid Plot ")
plt.xlabel("Purchases")
plt.ylabel("Day Engaged")
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:


final_model = KMeans(n_clusters=3,init ='k-means++',random_state=0)
preds_km = final_model.fit_predict(final_df)
preds_km


# In[ ]:


#splitting our variable into independent and dependent variables
X = final_df.iloc[:,[0,1]].values
y = model_kmeans2.labels_


# In[ ]:


#plotting our centroids
plt.figure(figsize = (8,8))
plt.scatter( X[y==0,0],  X[y==0,1],s = 50 ,c = 'green',label = 'cluster1')
plt.scatter(X[y==1,0],X[y==1,1],s = 50 ,c = 'blue',label = 'cluster2')
plt.scatter(X[y==2,0],X[y==2,1],s = 50 ,c = 'yellow',label = 'cluster3')


#plot the centroids for kmeans with original dataset
plt.scatter(final_model.cluster_centers_[:,0],final_model.cluster_centers_[:,1],s = 100,c = 'black',label = 'centroids')
plt.title("Centroid Plot ")
plt.xlabel("Purchases")
plt.ylabel("Day Engaged")
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:


#attaching our clusters to labels
final_df['clusters'] = final_model.labels_


# In[ ]:


# Group data by Clusters (K=3)
final_df.groupby('clusters').agg(['mean'])


# In[ ]:


final_df.head()


# # Saving the Model for future Predictions

# In[ ]:


import pickle


# In[ ]:


filename = 'final_model.sav'
pickle.dump(final_model,open(filename,'wb'))


# In[ ]:


#load the model from disk
loaded_model = pickle.load(open(filename,'rb'))


# In[ ]:


#saving our excel file
final_df.to_csv("clustering_final.csv")


# # Making a Predictive System

# In[ ]:


input_data = ( 25 ,849,	0	,1,	0,	0,	1,	0,	0,	0, 1,	0,	1,	0,	0)

input_data_array = np.asarray(input_data)

input_data_reshape = input_data_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshape)
print(prediction)

if (prediction[0]==0):
  print(' the purchases are moderate with least customers engaged on all days with the company but have highest acceptance to campaign offers and mostly they come from upper class background,most of them fall in the age groups of Gen X and and Boomers with higher background of education')
elif(prediction[0]==1):
  print( 'the purchases are lowest and moderately less customers engaged on all days with the company and the acceptance ratio to offers or campaigns is lesser compared to first cluster,most of the customers are from lower and middle class and most of them fall in the age category of Gen X and Boomers having higher education background ,we can also say that most of the families are having members near to 3')
else:
  print(' the purchases are highest with most of our customers engaged with the company on daily basis and have a moderate acceptance to any offers or campaigns, in the third cluster we observe that most of the customer come from middle and upper middle class and even in this cluster most of them fall in the age categories of GenX and  Boomers with highest education qualification')


# In[ ]:


#saving our dataframe for bivariate plots
df_cust5.to_csv("plot_dataframe.csv")
df_cust7.to_csv("evaluation_dataframe.csv")


# ## Project Conclusion

# - In this project we have worked over Customer Segmentation for a company and perform a personality analysis of the customers purchases over products
# 
# - we have extracted new features from the original dataset and reduced our dimensions
# 
# - we have performed EDA and outlier analysis and divivded our features like income and age by binning them into segments.
# 
# - we have performed univariate , bivariate and multivariate plots and with different plots we have understood the characteristics of customers according to their purchases and expenses with categories like income and age and where they fall
# 
# - we have also performed feature scaling with Standard and MinMax scaler and observed the clusters formed with both the scalers
# 
# - for feature selection we have used the black box technique,the most often used Principal Component Analysis and we have observed the component with the scree plot for accuracy
# 
# - For model Building we have performed different types of algorithms such as Kmeans , Agglomerative , DBSCANS and also Kprototypes since we had mixed dataframe and observed clusters obtained from the algorithms respectively
# 
# - We have chosen Kmeans on Stnadard Scaler which gave us the best results and with Kmeans we have performed with different K values and later we came to a conclusion that with 3 clusters we are getting the best results
# 
# - for model validation we have performed different types of validation like the silhouette score and david bouldins index and we have got the optimal accuracy with the clusters.
# 
# - for Future predictions we have used the randomforest as the classification alogorithm and we have got the best accuracy for the finalised model with clustering and have validated the results with accuracy score.
# 
# - In this project we have understood that the customer segmentation can be divided into many types of segmentation and also we have observed that which classes have the highest purchases and expenses based on income and age categories

# # Solutions To Company

#  - When it comes to Purchases and expenses we have observed that most of the people come from middle and upper-middle class backgrounds and have the highest purchases and expenses, they almost comprise 80% of the total segmentation.
# 
#  - We can attract this segment with more offers and campaigns with the products and bring the prices to affordable rates with association to another items, so every middle and upper middle class family engage with the company
# 
#  - we can run an advertise campaign for these segments that would improve sales of the products and could attract the consumer with discounts and coupons with two or more products.
# 
#  - We could also look at the age segmentation and we observed that most of our consumers are from GenX to boomers so for these groups we could create awareness by catalogs and run a creative campaigns and keeps offers for consumers who fall in this age group.
# 
#  - When it  comes to families for families we can keep products like hampers and call it as family pack and we have observed that consumers with a family are more in ratio when compared to bachelors or couples
# 
#  - To end with a conclusion the company has to create more awareness of their products, discounts , offers, campaigns, advertisements and digital media marketing  for the income groups that fall in middle class and upper middle class and between age groups that fall in GenX and Boomers that are within a family.This would increase the range of consumers and sales of products.
