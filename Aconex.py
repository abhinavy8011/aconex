
# coding: utf-8

# In[ ]:


import glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# In[ ]:


def create_Summary(project,row):
    
    project['sentDate'] = pd.to_datetime(project['sentDate'],format='%Y-%m-%d %H:%M:%S.%f')
    
    project = project[project['sentDate'].notnull()]
    project = project.sort_values('sentDate')
    
    df = pd.DataFrame()
    df.loc[row,'Project'] = project['projectId'][project.index[0]]

    project.groupby('fromOrganizationId')
    df.loc[row,'Number_of_Organizations'] = len(project.groupby('fromOrganizationId').count())
    
    df.loc[row,'Project_Duration'] = int((project['sentDate'][project.index[-1]] - project['sentDate'][project.index[0]]).days)

    project.groupby('fromUserId')
    df.loc[row,'Number_of_Users'] = len(project.groupby('fromUserId').count())

    project.groupby('correspondenceTypeId')
    df.loc[row,'Kinds_Of_Correspondence'] = len(project.groupby('correspondenceTypeId').count())
    
    df.loc[row,'Number of mails'] = len(project)

    return df


# In[ ]:


# Reading all the .csv files into one
#path =r'C:\Users\212342133\Documents\Python Scripts\1_Exploration\Aconex\Data+science+-+test+(June+2017)\Data science - test (June 2017)\correspondence_data'
allFiles = glob.glob("*.csv")


# In[ ]:


frame = pd.DataFrame()
list_ = []
for index,file_ in enumerate(allFiles):
    df = pd.read_csv(file_, index_col=None, header=0)
    df = create_Summary(df,index)
    list_.append(df)
frame_summary = pd.concat(list_)


# In[ ]:


frame_summary.to_excel('frame_summary.xlsx',index=False)


# In[ ]:


frame_summary = pd.read_excel('frame_summary.xlsx')


# In[ ]:


frame_summary.describe


# In[ ]:


import seaborn as sns
get_ipython().magic('matplotlib inline')


# In[ ]:


sns.heatmap(frame_summary.corr())


# In[ ]:


sns.jointplot(x='Project_Duration',y='Kinds_Of_Correspondence',data=frame_summary)


# In[ ]:


sns.pairplot(frame_summary)


# In[ ]:


frame_summary.head()


# In[ ]:


import sklearn
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()


# In[ ]:


frame_summary[['Number_of_Organizations','Project_Duration','Number_of_Users','Kinds_Of_Correspondence','Number of mails']] = scaler.fit_transform(frame_summary[['Number_of_Organizations','Project_Duration','Number_of_Users','Kinds_Of_Correspondence','Number of mails']]) 


# In[ ]:


frame_summary.head()


# In[ ]:


sns.pairplot(frame_summary)


# In[ ]:


frame_summary_sorted = frame_summary.sort_values('Project_Duration', ascending=False)
frame_summary_top = frame_summary_sorted[2000:]
#sns.pairplot(frame_summary_top)
sns.heatmap(frame_summary_top.corr())


# In[ ]:


frame_model = frame_summary.drop('Project', 1)
frame_model_X = frame_model.drop('Project_Duration',1)
frame_model_y = frame_model['Project_Duration']


# In[ ]:


frame_model.head()


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(frame_model_X, frame_model_y, test_size = 0.3, random_state = 1)


# In[ ]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score


# In[ ]:


cross_val_score(regressor, X_train, y_train, cv=10)


# In[ ]:


# - -------


# In[ ]:


from sklearn.svm import SVR
from sklearn import metrics

regressor = SVR(kernel = 'rbf')
regressor.fit(X_train,y_train)


# In[ ]:


# Training Error
predictions = regressor.predict(X_train)
print('MAE:', metrics.mean_absolute_error(y_train, predictions))
print('MSE:', metrics.mean_squared_error(y_train, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_train, predictions)))


# In[ ]:


# Prediction Error
predictions = regressor.predict(X_test)
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[ ]:


from sklearn.tree import DecisionTreeRegressor

dtree = DecisionTreeRegressor()
dtree.fit(X_train,y_train)


# In[ ]:


# Training Error
predictions = dtree.predict(X_train)
print('MAE:', metrics.mean_absolute_error(y_train, predictions))
print('MSE:', metrics.mean_squared_error(y_train, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_train, predictions)))


# In[ ]:


# Prediction Error
predictions = dtree.predict(X_test)
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(n_estimators=1000)
forest.fit(X_train,y_train)


# In[ ]:


# Training Error
predictions = forest.predict(X_train)
print('MAE:', metrics.mean_absolute_error(y_train, predictions))
print('MSE:', metrics.mean_squared_error(y_train, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_train, predictions)))


# In[ ]:


# Prediction Error
predictions = forest.predict(X_test)
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))



