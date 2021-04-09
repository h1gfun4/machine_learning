#Import libaries and load dataset/ Импортировать библиотеки и загрузить набор данных

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from plotly.validators.scatter.marker import SymbolValidator

import missingno as msno

import plotly.offline as pyo
pyo.init_notebook_mode()

from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("../input/telecom-users-dataset/telecom_users.csv")
pd.options.display.max_columns = 30
df.drop(["Unnamed: 0", "customerID"],axis=True, inplace=True) # these columns not used to analysis

#Visualization / Визуализация

# Visualization object type columns
df['Churn'] = df['Churn'].apply(lambda x: 0 if x == 'No' else 1) # if 1 is churn: yes
df['SeniorCitizen'] = df['SeniorCitizen'].apply(lambda x: 'No' if x == 0 else 'Yes')
df['TotalCharges'] =pd.to_numeric(df['TotalCharges'],errors='coerce')
df = df.dropna()

str_col = []
for i in df.columns:
    if df[i].dtype =='O':
        str_col.append(i)
j=0
k=0
fig, axes = plt.subplots(4,4, figsize=(20,15))
for i in str_col:
    sns.barplot(ax=axes[j,k], data=df,x=list(df[i].unique()),y=list(df.groupby(i)['Churn'].mean()))
    axes[j][k].set_title(i)
    plt.xticks(rotation = 20)
    k+=1
    if k==4:
        k=0
        j+=1

# Object type columns'conclusion Some features like gender, phone service, multipleline are have only some difference. Other features have difference more than 10percents
# Вывод столбцов типа объекта Некоторые функции, такие как пол, телефонная связь, множественная линия, имеют лишь небольшую разницу. Остальные характеристики имеют разницу более 10 процентов.

value = []
for i in str_col:
    value.append(max(df.groupby(i)['Churn'].mean()) - min(df.groupby(i)['Churn'].mean()))

features_value = pd.concat([pd.Series(str_col),pd.Series(value)],axis=1)

features_value.columns = ['feature', 'value']
features_value['value'] = features_value['value'].apply(lambda x : round(x,2))
fig = ff.create_table(features_value, height_constant=30)
fig.add_traces(go.Bar(x=features_value['feature'], y=features_value['value'],
                    marker=dict(color='#0099ff'),
                    xaxis='x2', yaxis='y2',text=features_value['value']))

fig['layout']['xaxis2'] = {}
fig['layout']['yaxis2'] = {}
# Edit layout for subplots
fig.layout.xaxis.update({'domain': [0, .35]})
fig.layout.xaxis2.update({'domain': [0.4, 1.]})
# The graph's yaxis MUST BE anchored to the graph's xaxis
fig.layout.yaxis2.update({'anchor': 'x2'})

# Update the margins to add a title and see graph x-labels.
fig.layout.margin.update({'t':50, 'b':100})
fig.layout.update({'title': 'Features value : Max - Min difference'})

fig.show()

# compare with Max() - Min() value, we can know that Contract have most difference. and others without gender, PhoneService, MultipleLines, they have more than 10percents difference
# сравните со значением Max () - Min (), мы можем знать, что Contract имеет наибольшую разницу. и другие без пола, PhoneService, MultipleLines, разница между ними более 10 процентов

fig = make_subplots(rows=1, cols=3, shared_yaxes=True,subplot_titles=("Tenure", "MonthlyCharges","TotalCharges"))

fig.add_trace(go.Histogram(x = df[df['Churn'] ==0]['tenure'], nbinsx=50),1,1)
fig.add_trace(go.Histogram(x = df[df['Churn'] ==1]['tenure'], nbinsx=50),1,1)

fig.add_trace(go.Histogram(x = df[df['Churn'] ==0]['MonthlyCharges'], nbinsx=50),1,2)
fig.add_trace(go.Histogram(x = df[df['Churn'] ==1]['MonthlyCharges'], nbinsx=50),1,2)

fig.add_trace(go.Histogram(x = df[df['Churn'] ==0]['TotalCharges'], nbinsx=50),1,3)
fig.add_trace(go.Histogram(x = df[df['Churn'] ==1]['TotalCharges'], nbinsx=50),1,3)

fig.update_traces(marker_line_color='red')
fig.update_layout(font_family="Rockwell", showlegend=False)
fig.update_layout(barmode="overlay")
fig.update_layout(paper_bgcolor=px.colors.qualitative.Pastel2[2])
fig.update_layout(paper_bgcolor='white')
fig.show()

# Numeric type columns'conclusion Tenure, TotalCharges have some meaningful values
# Вывод столбцов числового типа Tenure, TotalCharges имеют некоторые значимые значения

df = pd.get_dummies(df, columns =str_col, drop_first = True) # The object type to unit8 type. 
df.head()

#

X = df.drop("Churn",axis=1)
y = df['Churn']

rob = RobustScaler().fit(X)
X = pd.DataFrame(rob.transform(X),columns = X.columns)

#

#Modeling  Моделирование

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.3, random_state=100) # split the train data and test data

accuracy_list= []
for i in range(1,101):
    knn = KNeighborsClassifier( n_neighbors=i,)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    accuracy_list.append(accuracy_score(y_test,pred))

fig = px.scatter(x= range(1,101), y=accuracy_list,text=range(0,100))
fig.update_traces(textposition="top center")
fig.show()

#

knn = KNeighborsClassifier( n_neighbors=accuracy_list.index(max(accuracy_list)))
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
print("accuracy_score : " ,accuracy_score(y_test,pred))

#

confusion_matrix(pred,y_test)