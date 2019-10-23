#!/usr/bin/env python
# coding: utf-8

# # CS210 Spring 2019 - Sample Final Project
# 
# 
# # ACCIDENTS DATA SET EXPLORATION
# 
# 
# # GROUP MEMBERS:
#    -FATIH TALAY
#    -DILARA GIRGIN
#    -SEMIH BALKI

# In[1]:


import pandas as pd
from os.path import join
import seaborn as sns
import matplotlib.pyplot as plt
d_path = "/Users/semihbalki/Documents/CS210/Projects/GROUP PROJECT/DATA SETS"
filename = "Accidents0515.csv"

sns.set(style="darkgrid")
df = pd.read_csv(join(d_path, filename))
df1 = pd.read_csv("/Users/semihbalki/Documents/CS210/Projects/GROUP PROJECT/DATA SETS/Casualties0515.csv", error_bad_lines=False)
df["Age_of_Casualty"] = df1["Age_of_Casualty"]
df["Sex_of_Casualty"] = df1["Sex_of_Casualty"]
df["Casualty_Severity"] = df1["Casualty_Severity"]


# INFORMATION ABOUT THE DATA

# In[2]:


df.head()


# In[4]:


df.info()


# DESCRIBING THE DATA

# In[39]:


df.describe()


# In[36]:


#Number of accidents in terms of gender
#1:Female
#2:Male
accident_number = df.groupby(by="Sex_of_Casualty").count()["Accident_Severity"].sort_values(ascending=False)
print(accident_number)

accident_number.plot(kind="barh", color="steelblue")  # does not match with the official list
                                                  # since we removed some of the veterans

plt.xlabel("# of Accidents")
plt.title("Top 5 Scorers")
plt.show()


# In[38]:


print("Maximum Age column at the data: ", df["Age_of_Casualty"].max())
print("Minimum Age column at the data: ", df["Age_of_Casualty"].min())
print("Mean of the Age column at the data: ", df["Age_of_Casualty"].mean())
print("Standard deviation of the Age column at the data: ", df["Age_of_Casualty"].std())


# In[ ]:


THE HISTOGRAM OF ACCIDENT SEVERITY


# In[40]:


sns.distplot(df["Accident_Severity"].values, norm_hist=True)
plt.show()


# In[ ]:


THE HISTOGRAM OF LIGHT CONDITIONS


# In[41]:


sns.distplot(df["Light_Conditions"].values, norm_hist=True)
plt.show()


# CORRELATION MATRIX OF THE DATA

# In[42]:


sns.set(style="white")
corrmat = df.corr()
f,ax = plt.subplots(figsize = (12,9))
sns.heatmap(corrmat,vmax =.8,square = True)


# INSTEAD OF SCATTER PLOTTING OF THE DATA USED BAR CHART TO VISUALIZE THE DATA. 
# SINCE WE HAVE MOSTLY CATEGORICAL VALUES ON ONUR DATA.

# EFFECT OF AGE ON ACCIDENT SEVERITY
#     -MEASURED AS AN AVERAGE ON ACCIDENT SEVERITY

# In[43]:


#Effect of age on accident severity
#Average age on accident severity
def average(df):
    sum_one = 0
    sum_two = 0
    sum_three = 0
    count_one = 0
    count_two = 0
    count_three = 0
    total = 0
    for _, row in df.iterrows():
        num = int(row["Accident_Severity"])
        hold = int(row["Age_of_Casualty"])
        if num == 1:
            sum_one += hold
            count_one += 1
        elif num == 2:
            sum_two += hold
            count_two += 1
        elif num == 3:
            sum_three += hold
            count_three += 1
    arr = []
    if sum_one == 0:
        arr.append(0)
    if sum_two == 0:
        arr.append(0)
    if sum_three == 0:
        arr.append(0)
    if sum_one != 0:
        arr.append(sum_one/count_one)
    if sum_two != 0:
        arr.append(sum_two/count_two)
    if sum_three != 0:
        arr.append(sum_three/count_three)
    dfd = pd.DataFrame({'Average Age':arr, 'Accident Severity':[1, 2, 3]})
    ax = dfd.plot.barh(x='Average Age', y='Accident Severity')
average(df)


# EFFECT OF LIGHT CONDITION ON ACCIDENT SEVERITY
#     -MEASURE AS AVERAGE ACCIDENT SEVERITY OF LIGHT CONDITIONS

# In[44]:


#Effect of light condition on accident severity
#Average accident severity of light conditions
accident_location = df.groupby(by="Light_Conditions").mean()["Accident_Severity"]
#print(accident_location)

accident_location.plot(kind="barh")


# HYPOTHESIS TESTING

# In[20]:


#Do gender effect the accident severity?
#Low volume: 1 <= Accident severity <= 2
#High volume: Accident severity = 3
#The first bullet point at the Hypothesis testing part evaluation as T-testing
fig, ax = plt.subplots(1, 3, figsize=(30,20))  # a figure with 1 row and 3 columns
                                              # ax variable stores a list with 3 elements
                                              # each element in ax correspons to chart
# Classifying accident severity as in groups
low_df = df[(1<=df["Accident_Severity"]) & (df["Accident_Severity"]<=2)]
high_df = df[df["Accident_Severity"]==3]

# extracting values
low_values = low_df["Sex_of_Casualty"].values
high_era = high_df["Sex_of_Casualty"].values
        
high_df.plot(kind="hist", ax=ax[0], bins=40, color="c")#alternative
ax[0].set_title("Alternative")

low_df.plot(kind="hist", ax=ax[1], bins=40, label="1 <= Accident severity <= 2", color="m")#null
ax[1].set_title("Null")

sns.kdeplot(high_era, shade=True, label="Accident severity=3", ax=ax[2], color="c")
sns.kdeplot(low_values, shade=True, label="1 <= Accident severity <= 2", ax=ax[2], color="m")
ax[2].set_title("Comparison of the Accident Severity")

plt.suptitle("Accident Severity distributions for the Gender")
plt.show()

_, p_value = stats.ttest_ind(low_values, high_era, equal_var=False)

#RESULT
null_hypo2 =  "Gender has no effect on the accident severity."
alt_hypo2 = "gender effects the accident severity"

if(0.05 > p_value):
    print("Since ", p_value, " is smaller than 0.05, we can reject the null hypothesis. Therefore,", alt_hypo2)
else:
    print("Since ", p_value, " is greater than 0.05, we can not reject the null hypothesis. Therefore,", null_hypo2)


# In[22]:


#Do weather conditions effect casualty severity?
#Low volume: 1 <= Weather conditions <= 4
#High volume: 4 <= Weather conditions <= 7
from scipy import stats
fig, ax = plt.subplots(1, 3, figsize=(30,20))  # a figure with 1 row and 3 columns
                                              # ax variable stores a list with 3 elements
                                              # each element in ax correspons to chart

# Classifying accident severity as in groups
lo_df = df[(1<=df["Weather_Conditions"]) & (df["Weather_Conditions"]<=4)]
hig_df = df[(4<=df["Weather_Conditions"]) & (df["Weather_Conditions"]<=7)]

# extracting values
lo_values = lo_df["Casualty_Severity"].values
hig_era = hig_df["Casualty_Severity"].values

hig_df.plot(kind="hist", ax=ax[0], bins=40, color="c")#alternative
ax[0].set_title("Alternative")

low_df.plot(kind="hist", ax=ax[1], bins=40, label="1 <= Accident severity <= 2", color="m")#null
ax[1].set_title("Null")

sns.kdeplot(hig_era, shade=True, label="Accident severity=3", ax=ax[2], color="c")
sns.kdeplot(lo_values, shade=True, label="1 <= Accident severity <= 2", ax=ax[2], color="m")
ax[2].set_title("Comparison of the Casualty Severity")

plt.suptitle("Casualty Severity distributions for the Weather")
plt.show()

_, p_value = stats.ttest_ind(lo_values, hig_era, equal_var=False)

#RESULT
null_hypo2 =  "Weather has no effect on the casualty severity."
alt_hypo2 = "weather effects the casualty severity"

if(0.05 > p_value):
    print("Since ", p_value, " is smaller than 0.05, we can reject the null hypothesis. Therefore,", alt_hypo2)
else:
    print("Since ", p_value, " is greater than 0.05, we can not reject the null hypothesis. Therefore,", null_hypo2)


# LINEAR REGRESSION

# A = ["Number_of_Vehicles", "Light_Conditions", "Weather_Conditions", "Urban_or_Rural_Area", "Age_of_Casualty", "Sex_of_Casualty", "Casualty_Severity"]
# 
# Relationship between A and accident severity

# In[30]:


ind_var = pd.DataFrame(df, columns = ["Number_of_Vehicles", "Light_Conditions", "Weather_Conditions", "Urban_or_Rural_Area"
                                     , "Age_of_Casualty", "Sex_of_Casualty", "Casualty_Severity"])
target_labels = pd.DataFrame(df["Accident_Severity"])

from sklearn import linear_model
lr = linear_model.LinearRegression()

from sklearn.model_selection import train_test_split

X = ind_var
y = target_labels

# Split X and y into X_
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

lr.fit(X_train, y_train)


# In[31]:


lr.coef_


# In[32]:


lr.intercept_


# In[33]:


for index, col_name in enumerate(X_train.columns):
    print("The coefficient for {} is {}".format(col_name, lr.coef_[0][index]))


# In[38]:


lr.score(X_test, y_test)


# In[39]:


from sklearn.metrics import mean_squared_error

y_predict = lr.predict(X_test)

lr_mse = mean_squared_error(y_predict, y_test)

print(lr_mse)


# B = ["Number_of_Vehicles", "Light_Conditions", "Weather_Conditions", "Urban_or_Rural_Area"
#                                      , "Age_of_Casualty", "Sex_of_Casualty", "Accident_Severity"]
# 
# Relationship between B and casualty severity

# In[40]:


indep_var = pd.DataFrame(df, columns = ["Number_of_Vehicles", "Light_Conditions", "Weather_Conditions", "Urban_or_Rural_Area"
                                     , "Age_of_Casualty", "Sex_of_Casualty", "Accident_Severity"])
dep_var = pd.DataFrame(df["Casualty_Severity"])

from sklearn import linear_model
lr = linear_model.LinearRegression()

from sklearn.model_selection import train_test_split

X = indep_var
y = dep_var

# Split X and y into X_
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

lr.fit(X_train, y_train)


# In[41]:


lr.coef_


# In[42]:


lr.intercept_


# In[43]:


for index, col_name in enumerate(X_train.columns):
    print("The coefficient for {} is {}".format(col_name, lr.coef_[0][index]))


# In[44]:


lr.score(X_test, y_test)


# In[45]:


from sklearn.metrics import mean_squared_error

y_predict = lr.predict(X_test)

lr_mse = mean_squared_error(y_predict, y_test)

print(lr_mse)


# TWO ML MODELS

# i) DECISION TREE

# In[4]:


#Extracting some unnecessary columns from the data

new_df = df.drop(['Accident_Index', 'Location_Easting_OSGR', 'Location_Northing_OSGR', 'Longitude', 'Latitude', 'Police_Force', 'Number_of_Vehicles', 'Number_of_Casualties', 'Date', 'Special_Conditions_at_Site', 
                  'Carriageway_Hazards', 'Did_Police_Officer_Attend_Scene_of_Accident', 
                  'LSOA_of_Accident_Location', 'Casualty_Severity', 'Local_Authority_(Highway)', 
                 'Time', 'Junction_Detail', 'Local_Authority_(District)', '1st_Road_Class', '1st_Road_Number',
                 'Road_Type', 'Speed_limit', 'Junction_Control', '2nd_Road_Class', '2nd_Road_Number',
                 'Pedestrian_Crossing-Human_Control', 'Pedestrian_Crossing-Physical_Facilities'], axis='columns')

new_df.head()


# We use regression for decision tree since we want to predict 'Accident_Severity' column which has numerical values.

# --> REGRESSION 

# In[5]:


# Splitting features and the label in the dataset
X = new_df.drop('Accident_Severity', axis='columns')  
y = new_df['Accident_Severity']  

from sklearn.model_selection import train_test_split

# Creating train and test datasets with train_test_split function
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 

# Since we want to perform regression this time, we need to import DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor()  
regressor.fit(X_train, y_train)  

y_pred = regressor.predict(X_test)

# We can create a dataframe to compare real values in the test set and predicted values
df_compare = pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})

df_compare.head()


# ii) RandomForest

# In[6]:


# Creating training and test splits from the original dataframe
from sklearn.model_selection import train_test_split

X = new_df.drop('Accident_Severity', axis='columns')  
y = new_df['Accident_Severity']  

# 75% for training and 25% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25, random_state = 42)

from sklearn.ensemble import RandomForestClassifier
rf  = RandomForestClassifier(random_state = 42)

rf.fit(X_train, y_train)

pred_rf = rf.predict(X_test)

# Checking the performance of the model with accuracy score;
from sklearn import metrics

print("Accuracy of the random forest model: ",metrics.accuracy_score(y_test, pred_rf))

from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()

# We can directly fit the tree, since our data is ready
dtree.fit(X_train, y_train)

pred_dtree = dtree.predict(X_test)
print("Accuracy of the random forest model: ",metrics.accuracy_score(y_test, pred_dtree))

# It seems our model has a function called feature_importances_
# Let's call it and see what it does
rf.feature_importances_


# Creating a bar plot for feature importances

# Firstly creating a Pandas Series to match feature importances values and their indices, also sorting them in decreasing order
feature_importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(12, 12))
sns.barplot(x=feature_importances, y=feature_importances.index)

# Add labels to our graph  
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Feature Importance Rankings")
plt.show()


# VERIFYING OUR MODELS

# In[5]:


# no NaN values in the dataset
new_df.isnull().sum()


# In[6]:


# let's extract the features and the target attribute as numpy arrays
features = new_df.drop(["Accident_Severity"], axis=1).values
target = new_df["Accident_Severity"].values

from sklearn.tree import DecisionTreeRegressor

# let's create a default model and have a look at it's performance
model = DecisionTreeRegressor()
model


# i) Cross validation for Decision Tree

# In[35]:


from sklearn.model_selection import KFold  # the k-fold cross-validator builder
from sklearn.model_selection import cross_val_score  # score computation

kfold = KFold(n_splits=3, random_state=7)  # 3-fold cv

#result = cross_val_score(model, features, target, cv=kfold, scoring='MSE')
result = cross_val_score(model, features, target, cv=kfold, scoring='neg_mean_squared_error')
print(result)
print("%f" % result.mean())


# The cross validation score at above for decision tree regressor is negative but it does not mean that model is not successfull, it is about the library.

# In[12]:


ax = sns.distplot(target, kde=False)
ax.set(ylabel="counts", xlabel="values")
plt.show()


# ii) Cross validation for Random Forest

# In[13]:


model = RandomForestClassifier(random_state = 42)

result = cross_val_score(model, features, target, cv=kfold)
print(result)
print("%f" % result.mean())


# PERFORMANCE MEASUREMENT

# i) Decision Tree

# In[8]:


import numpy as np
from sklearn.model_selection import StratifiedKFold
import time  # to calculate the runtime

skf = StratifiedKFold(n_splits=2)  # again 3-fold cv
# first, we need to define the hyperparameter space
# as a dictionary in which keys are hyperparameter names
param_grid = {
    "max_depth": np.arange(1, 4),
    "min_samples_split": np.arange(2, 4),
    "min_samples_leaf": np.arange(2, 4)
}


# In[9]:


from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(
            estimator=model,  # the model we have
            param_grid=param_grid,  # the search space
            cv = skf  # stratified cross-validation at each step
            #scoring = 'accuracy'
        )

start_time = time.time()
grid_result = grid.fit(features, target)  # fitting the data as if it is a model

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
print("Execution time: " + str((time.time() - start_time)) + ' ms')


# In[10]:


from sklearn.model_selection import RandomizedSearchCV

random = RandomizedSearchCV(
            estimator=model, 
            param_distributions=param_grid, 
            cv = skf
        )

start_time = time.time()
random_result = random.fit(features, target)

print("Best: %f using %s" % (random_result.best_score_, random_result.best_params_))
print("Execution time: " + str((time.time() - start_time)) + ' ms')


# ii) Random Forest

# In[11]:


from sklearn.ensemble import RandomForestClassifier
model  = RandomForestClassifier(random_state = 42)

grid = GridSearchCV(
            estimator=model,  # the model we have
            param_grid=param_grid,  # the search space
            cv = skf  # stratified cross-validation at each step
            #scoring = 'accuracy'
        )

start_time = time.time()
grid_result = grid.fit(features, target)  # fitting the data as if it is a model

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
print("Execution time: " + str((time.time() - start_time)) + ' ms')


# In[12]:


random = RandomizedSearchCV(
            estimator=model, 
            param_distributions=param_grid, 
            cv = skf
        )

start_time = time.time()
random_result = random.fit(features, target)

print("Best: %f using %s" % (random_result.best_score_, random_result.best_params_))
print("Execution time: " + str((time.time() - start_time)) + ' ms')


# We can deduct that, for both decision tree and random forest model are as we have experienced in the recitation "with random search, we obtained a slightly worse accuracy but the search was much more efficient than grid search".

# -DESCRIBE WHICH ONE PERFORMS BETTER WITH A ML TECHNIQUE?
# 
#         Random Forest Model performs better since it's accuracy score is better than decision tree mode which is
#         0.7323346002821433.
#         
# -DESCRIBE WHICH ONE PERFORMS BETTER AND WHY. TRY TO DESCRIBE WHICH FEATURES WORKS BEST FOR EACH ML TECHNIQUE.
# 
#         As we could observe from Feature Importance Rankings; as much as we increase 'Age_of_Casualty' column, 
#         RandomForest accuracy rate increases. Therefore, 'Age_of_Casualty' feature works best. Since, decision tree is
#         a subset of RandomForest whatever we conclude for RandomForest we could directly mention also for decision
#         tree. Thus, 'Age_of_Casualty' also works best for decision tree.
#         
#         -Why Random Forest Model performs better
#                By Random Forest we train more than one decision tree on a random feature subset with random points.At
#                the prediction part, predict from each decision tree and use  techniques such as; majority voting etc.
#                to use their common ideas. Therefore, Random Forest model performs better.
