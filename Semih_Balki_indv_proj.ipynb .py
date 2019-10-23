#!/usr/bin/env python
# coding: utf-8

# Semih Balki 19010

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime
import seaborn as sns  # a visualization library based on matplotlib
from datetime import datetime
from geopy.geocoders import Nominatim


# In[2]:



from os.path import join
d_path = "/Users/semihbalki/Desktop"
filename = "taxi-trips.csv"

df = pd.read_csv(join(d_path, filename))


# DATA EXPLORATION

# In[84]:


#First bullet point
print("Description of the data: ", df.describe())
print("\n")

numpy_matrix = df.as_matrix()
print("Shape of the data: ", numpy_matrix.shape)
print("\n")

print("data type of each column: ")
df.dtypes


# In[64]:


#Second bullet point
get_ipython().system('pip install reverse_geocoder')
import reverse_geocoder as rg
def pickup_district(row):
    #Applying reverse geocoding to associated coordinates
    
    pickup_longitude = row["pickup_longitude"]
    pickup_latitude = row["pickup_latitude"]
    k = (pickup_latitude, pickup_longitude)
    results = rg.search(k, mode=1)
    return results


def dropoff_district(row):
    # Applying reverse geocoding to associated coordinates

    dropoff_longitude = row["dropoff_longitude"]
    dropoff_latitude = row["dropoff_latitude"]
    k = (dropoff_latitude, dropoff_longitude)
    results = rg.search(k, mode=1)
    return results


df["pickup_district"] = df.apply(pickup_district, axis=1)  # with axis=1, we iterate over rows
df["dropoff_district"] = df.apply(dropoff_district, axis=1)  # with axis=1, we iterate over rows


# In[115]:


#Third bullet point
top5_pickup = df['pickup_district'].value_counts()[:5]

print("Top 5 value of the pickup_district: ")
print(df['pickup_district'].value_counts()[:5])
print("\n")

top5_dropoff = df['dropoff_district'].value_counts()[:5]

print("Top 5 value of the dropoff_district: ")
print(df['dropoff_district'].value_counts()[:5])
print("\n")

print("Top 5 value of the dropoff_district and pickup_district as a graph: ")
#top5_dropoff.plot(kind="barh")

top5_pickup.plot.line(label="top5_pickup", color="red")
top5_dropoff.plot.bar(label="top5_dropoff", color="green")
plt.legend()
#top5_dropoff district and top5_pickup places are same, it shows that that places are popular at the data


# In[120]:


#Fourth bullet point
from geopy.distance import geodesic

results = []
for _, row in df.iterrows():
    pickup_longitude = row["pickup_longitude"]
    pickup_latitude = row["pickup_latitude"]
    dropoff_longitude = row["dropoff_longitude"]
    dropoff_latitude = row["dropoff_latitude"]
    pickup = (pickup_latitude, pickup_longitude)
    dropoff = (dropoff_latitude, dropoff_longitude)
    result = float(geodesic(pickup, dropoff).kilometers)
    results.append(result)

df['distance(kilometers)'] = results

print("Maximum distance column at the data: ", df["distance(kilometers)"].max())
print("Minimum distance column at the data: ", df["distance(kilometers)"].min())
print("Mean of the distance column at the data: ", df["distance(kilometers)"].mean())
print("Standard deviation of the distance column at the data: ", df["distance(kilometers)"].std())
sns.distplot(df["distance(kilometers)"].values, norm_hist=True) 
plt.show()


# In[19]:


#Fifth bullet point
import datetime as datetime

def time(row):
    pickup_datetime = row["pickup_datetime"]
    hold = pickup_datetime[-8:]
    x = int(hold[0:2])
    y = int(hold[3:5])
    z = int(hold[6:])
    if x >= 0 and x < 7:
        x += 24
    if x==7 and y==0 and z==0:
        x += 24
    if x == 9 and y == 0 and z == 0:
        return "rush_hour_morning"
    elif x == 16 and y == 0 and z == 0:
        return "afternoon"
    elif x == 18 and y == 0 and z == 0:
        return "rush_hour_evening"
    elif x == 23 and y == 0 and z == 0:
        return "evening"
    elif x == 31 and y == 0 and z == 0:
        return "late_night"
    elif x >= 7 and x < 9:
        return "rush_hour_morning"
    elif x >= 9 and x < 16:
        return "afternoon"
    elif x >= 16 and x < 18:
        return "rush_hour_evening"
    elif x >= 18 and x < 23:
        return "evening"
    elif x >= 23 and x < 31:
        return "late_night"


df["time_of_day"] = df.apply(time, axis=1)  # with axis=1, we iterate over rows


# In[32]:


#Sixth bullet point
rhm = 0.0
aft = 0.0
rhe = 0.0
eve = 0.0
ln = 0.0
count_rhm = 0
count_aft = 0
count_rhe = 0
count_eve = 0
count_ln = 0
for _, row in df.iterrows():
    x = float(row['distance(kilometers)'])
    name = row["time_of_day"]
    if name == "rush_hour_morning":
        rhm += x
        count_rhm += 1
    elif name == "afternoon":
        aft += x
        count_aft += 1
    elif name == "rush_hour_evening":
        rhe += x
        count_rhe += 1
    elif name == "evening":
        eve += x
        count_eve += 1
    elif name == "late_night":
        ln += x
        count_ln += 1

data = {'rush_hour_morning': rhm / count_rhm, 'afternoon': aft / count_aft, 'rush_hour_evening': rhe / count_rhe, 'evening': eve / count_eve, 'late_night': ln / count_ln}
names = list(data.keys())
values = list(data.values())

plt.bar(names, values)
plt.ylabel("Avg. Distance(kilometers)")
plt.xlabel("Time of day")
plt.title("Distance variation as time changes")
plt.show()


# In[39]:


#Seventh bullet point
rhm = 0.0
aft = 0.0
rhe = 0.0
eve = 0.0
ln = 0.0
count_rhm = 0
count_aft = 0
count_rhe = 0
count_eve = 0
count_ln = 0
for _, row in df.iterrows():
    x = int(row['trip_duration'])
    name = row["time_of_day"]
    sum += x
    if name == "rush_hour_morning":
        rhm += x
        count_rhm += 1
    elif name == "afternoon":
        aft += x
        count_aft += 1
    elif name == "rush_hour_evening":
        rhe += x
        count_rhe += 1
    elif name == "evening":
        eve += x
        count_eve += 1
    elif name == "late_night":
        ln += x
        count_ln += 1

data = {'rush_hour_morning': rhm/count_rhm, 'afternoon': aft/count_aft, 'rush_hour_evening': rhe/count_rhe, 'evening': eve/count_eve, 'late_night': ln/count_ln}
name = list(data.keys())
value = list(data.values())

plt.bar(name, value)
#plt.legend()
plt.ylabel("Total Trip duration")
plt.xlabel("Time of day")
plt.title("Trip duration variation as time changes")
plt.show()


# HYPOTHESIS TESTING

# In[128]:


#First bullet point
past_df = df[df["passenger_count"] == 1]
gs_df = df[df["passenger_count"] > 1]

ax = sns.kdeplot(past_df["distance(kilometers)"].rename("1"), shade=True)
sns.kdeplot(gs_df["distance(kilometers)"].rename("2-6"), ax=ax, shade=True)

plt.show()

past_values = past_df["distance(kilometers)"].values
gs_era = gs_df["distance(kilometers)"].values

_, p_value = stats.ttest_ind(a=past_values, b=gs_era, equal_var=False)
print("First result: ", p_value)
print("\n")

null_hypo = "passenger group size has no effect on the distance."
alt_hypo = "passenger group size affect the distance."

if(0.05 > p_value):
    print("Since ", p_value, " is smaller than 0.05, we can reject the null hypothesis. Therefore,", alt_hypo)
else:
    print("Since ", p_value, " is greater than 0.05, we can not reject the null hypothesis. Therefore,", null_hypo)


# In[129]:


#Second bullet point
from datetime import datetime
arr = []
def week_calc(row):
    pickup_datetime = row["pickup_datetime"]
    hold = pickup_datetime[:10]
    x = int(hold[0:4])
    y = int(hold[5:7])
    z = int(hold[8:10])
    d = datetime(x, y, z)
    hold = d.weekday()
    return int(hold)

df["weekor"] = df.apply(week_calc, axis=1)  # with axis=1, we iterate over rows

past = df[df["weekor"] < 5]
alt = df[df["weekor"] >= 5]

ax_2 = sns.kdeplot(past["distance(kilometers)"].rename("weekday"), shade=True)
sns.kdeplot(alt["distance(kilometers)"].rename("weekend"), ax=ax_2, shade=True)

plt.show()

past_val = past["distance(kilometers)"].values
alt_era = alt["distance(kilometers)"].values

_, p_value = stats.ttest_ind(a=past_val, b=alt_era, equal_var=False)
print("Second result: ", p_value)
print("\n")

null_hypo2 =  "the day of the week has no effect on the distance."
alt_hypo2 = "trip distances increase in weekends."

if(0.05 > p_value):
    print("Since ", p_value, " is smaller than 0.05, we can reject the null hypothesis. Therefore,", alt_hypo2)
else:
    print("Since ", p_value, " is greater than 0.05, we can not reject the null hypothesis. Therefore,", null_hypo2)


# T-TESTING

# In[127]:


#The first bullet point at the Hypothesis testing part evaluation as T-testing
fig, ax = plt.subplots(1, 3, figsize=(14,6))  # a figure with 1 row and 3 columns
                                              # ax variable stores a list with 3 elements
                                              # each element in ax correspons to chart
        
comp_studs.plot(kind="hist", ax=ax[0], bins=40, label="completed", color="c")
ax[0].set_title("Alternative")

none_studs.plot(kind="hist", ax=ax[1], bins=40, label="none", color="m")
ax[1].set_title("Null")

sns.kdeplot(gs_era, shade=True, label="passenger size > 1", ax=ax[2], color="c")
sns.kdeplot(past_values, shade=True, label="passenger == 1", ax=ax[2], color="m")
ax[2].set_title("Comparison of the passenger size")

plt.suptitle("Distance distributions for the group size")
plt.show()

stats.ttest_ind(past_values, gs_era, equal_var=False) 


# In[130]:


#The second bullet point at the Hypothesis testing part evaluation as T-testing
fig, ax = plt.subplots(1, 3, figsize=(14,6))  # a figure with 1 row and 3 columns
                                              # ax variable stores a list with 3 elements
                                              # each element in ax correspons to chart
        
comp_studs.plot(kind="hist", ax=ax[0], bins=40, label="completed", color="c")
ax[0].set_title("Alternative")

none_studs.plot(kind="hist", ax=ax[1], bins=40, label="none", color="m")
ax[1].set_title("Null")

sns.kdeplot(alt_era, shade=True, label="weekday", ax=ax[2], color="c")
sns.kdeplot(past_val, shade=True, label="weekend", ax=ax[2], color="m")
ax[2].set_title("Distance distributions for both")

plt.suptitle("Distance distribution for weekend")
plt.show()

stats.ttest_ind(past_val, alt_era, equal_var=False) 


# In[ ]:




