#!/usr/bin/env python
# coding: utf-8

# In[106]:


import pandas as pd 
import numpy as np 
import json 
import requests
import os
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from cycler import cycler


# In[107]:


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# # Gathering data 

# #### In our task, we have three database one of them come from twitter API but unfortunately they decline my application.

# In[108]:


#import tweepy 

#consumer_key = 'xxxx'
#consumer_secret = 'xxxxxx'
#access_token = 'xxxxx'
#access_secret = 'xxxxxx'

#auth = tweepy.OAuthHandler(consumer_key,consumer_secret)
#auth.srt_access_token(access_token,access_secret)
#api = tweepy.API(auth)


# In[109]:


df = pd.read_csv("twitter-archive-enhanced.csv")


# In[110]:


#df_json = pd.read_json('tweet-json.txt', lines=True) other way to read the data 
df_json = pd.read_json(open("C:/Users/www7m/misk_project/project_5/tweet-json", "r", encoding="utf8"),lines=True)


# In[111]:


url = "https://d17h27t6h515a5.cloudfront.net/topher/2017/August/599fd2ad_image-predictions/image-predictions.tsv"
response = requests.get(url)

with open(os.path.join("image-predictions.tsv"),mode = "wb") as file:
    file.write(response.content)

    
image_prediction_df = pd.read_csv("image-predictions.tsv", sep = "\t")


# # Assessing data 

# #### In assessing part we try to find the issue and find solution. 

# In[112]:


df.head(2)


# In[113]:


df.shape[0]


# In[114]:


df.info()


# In[115]:


df.describe()


# In[116]:


df.isnull().sum()


# In[117]:


df.duplicated().sum()


# In[118]:


df.dtypes


# In[119]:


df["doggo"].value_counts()


# In[120]:


df["floofer"].value_counts()


# In[121]:


df["pupper"].value_counts()


# In[122]:


df["puppo"].value_counts()


# In[123]:


df_json.head(2)


# In[124]:


df_json.shape[0]


# In[125]:


df_json.info()


# In[126]:


df_json.describe()


# In[127]:


df_json.isnull().sum()


# In[128]:


image_prediction_df.head(5)


# In[129]:


image_prediction_df.shape[0]


# In[130]:


image_prediction_df.info()


# In[131]:


image_prediction_df.describe()


# In[132]:


image_prediction_df.isnull().sum()


# In[133]:


image_prediction_df.duplicated().sum()


# ## we find too many issues but we take some of them and try to prepare the data for analysis.
# # Quality :
# 
# ### In twitter-archive-enhanced.csv :
# 
# Rating it should not more than 10. 
# 
# I have a lot of missing values in (in_reply_to_status_id , in_reply_to_user_id , retweeted_status_id , retweeted_status_user_id , retweeted_status_timestamp ).
# 
# I don't have any duplicated (This is comfortable ). (:
# 
# Timestamp type it’s should datetime not a object.
# 
# I can merge these four columns ( doggo , floofer , pupper , puppo ) in to one.  
# 
# The null values inside the four columns ( doggo , floofer , pupper , puppo ) it should  come as (null) not (None).
# 
# 
# 
# ### In Tweet_json.txt:
# 
# I have a lot of missing values(Tweet_json.txt)  in (contributors , coordinates, geo, in_reply_to_screen_name , in_reply_to_status_id , in_reply_to_status_id_str , in_reply_to_user_id , in_reply_to_user_id_str , place , quoted_status , quoted_status_id , quoted_status_id_str , retweeted_status ).
# 
# 
# ### In Image_prediction:
# 
# in image_prediction there is no column for most confidence breed of dogs.
# 
# 
# 
# 
# #### there are missing tweets since the tweets in tweet_archeve are 2356 and in image_prediction are 2075.
# #### we need tweet with images together
# 
# 
# # Tidiness :
# 
# 
# 1- All three database it's should be in one dataframe.
# 
# 2-All columns 'doggo','floof', 'pupper' and 'puppo' it should in one column.
# 
# 3-some columns like "in_reply_to_status_id and" they have too many missing value, my opinion is deleting them.

# In[134]:


df.info()


# # Data cleaning

# In[135]:


df = df.drop(['in_reply_to_status_id','in_reply_to_user_id','retweeted_status_id','retweeted_status_user_id' , 'retweeted_status_timestamp'], axis=1)
df.info()


# In[136]:


df["doggo"] = df["doggo"].replace("None", "")
df["floofer"] = df["floofer"].replace("None", "")
df["pupper"] = df["pupper"].replace("None", "")
df["puppo"] = df["puppo"].replace("None", "")


# In[137]:


df["dog_stage"] = df["doggo"] + df["floofer"] + df["pupper"] + df["puppo"]
df["dog_stage"].value_counts()


# In[138]:


df["dog_stage"] = df["dog_stage"].replace('',np.nan)
df["dog_stage"].value_counts()


# In[139]:


df.loc[df["dog_stage"] == "doggofloofer", "dog_stage"] = "dpggp, floofer"
df.loc[df["dog_stage"] == "doggopupper", "dog_stage"] = "doggo, puppo"
df.loc[df["dog_stage"] == "doggopuppo", "dog_stage"] = "dpggp, puppo"
df["dog_stage"].value_counts()


# In[140]:


df = df.drop(['doggo','floofer', 'pupper' , 'puppo'],axis=1)
df.head()


# In[141]:


df_json = df_json.drop(['contributors','coordinates','geo','in_reply_to_screen_name' , 'in_reply_to_screen_name', 'in_reply_to_status_id' , 'in_reply_to_status_id_str' , 'in_reply_to_user_id' , 'in_reply_to_user_id_str' , 'place' , 'quoted_status' , 'quoted_status_id' , 'quoted_status_id_str' , 'retweeted_status' ], axis=1)
df_json.info()


# In[142]:


breed = []
confidence = []

def breed_confidence(row):
    if row['p1_dog'] == True:
        breed.append(row['p1'])
        confidence.append(row['p1_conf'])
    elif row['p2_dog'] == True:
        breed.append(row['p2'])
        confidence.append(row['p2_conf'])
    elif row['p3_dog'] == True:
        breed.append(row['p3'])
        confidence.append(row['p3_conf'])
    else:
        breed.append('Unidentifiable')
        confidence.append(0)
      

image_prediction_df.apply(breed_confidence, axis=1)
image_prediction_df['breed'] = breed
image_prediction_df['confidence'] = confidence
image_prediction_df.head()


#note this code was taken from this source : http://empierce.com/2017/11/14/wrangling-weratedogs/


# In[143]:


merge_df = df.merge(df_json, right_on = "id", left_on = "tweet_id")


# In[144]:


all_data = merge_df.merge(image_prediction_df, right_on = "tweet_id", left_on = "tweet_id")


# In[145]:


all_data.shape[0]


# In[146]:


all_data.info()


# In[147]:


all_data.describe()


# In[148]:


all_data.isnull().sum()


# In[149]:


all_data.head(2)


# In[150]:


all_data = all_data[all_data['jpg_url'].notnull()]


# In[151]:


all_data['timestamp'] = pd.to_datetime(all_data['timestamp'])


# In[152]:


all_data.info()


# In[153]:


all_data.to_csv("final_data.csv")


# # Data Analysing
# #### The basic and easy way to understand the data is plotting, in next section we will find some graph that help us to understand the database.

# In[154]:


#This program code they organizing all the Graph automatics, size or color etc.
plt.rcParams['figure.figsize'] = [15.0, 7.0]
plt.rcParams['figure.dpi'] = 80
plt.rcParams['savefig.dpi'] = 100
plt.rcParams['font.size'] = 12
plt.rcParams['legend.fontsize'] = 'large'
plt.rcParams['figure.titlesize'] = 'medium'
plt.rcParams['figure.facecolor'] = '0.75'
plt.rcParams['lines.markersize'] = np.sqrt(20)
plt.rcParams['patch.force_edgecolor'] = True
plt.rcParams['patch.facecolor'] = 'b'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')


# In[155]:


t_b = all_data.groupby('breed').filter(lambda x: len(x) > 15)
t_b['breed'].value_counts().plot(kind = 'bar',title = 'This graph show which dog thay breeding more than 15 in our tweets')


# In[156]:


plt.plot( all_data['retweet_count'] , color = 'b')
plt.plot(all_data['favorite_count'], color = 'y')
plt.xlabel('Tweet over Time ')
plt.ylabel('Total Count')
plt.title('This graph show the correlation between Retweets and favorites over the time')
plt.show()


# In[157]:


dog_stage.dog_stage.hist();


# In[158]:


dog_stage = all_data[all_data['dog_stage'].notnull()]


# In[159]:


dog_stage[dog_stage['dog_stage'].notnull()]['dog_stage'].value_counts().plot(kind = 'pie')
plt.title('Dog stages')


# In[160]:


all_data.groupby('timestamp')['tweet_id'].mean().plot(title = 'This graph show the correlation between Dates and Tweetes')


# In[ ]:




