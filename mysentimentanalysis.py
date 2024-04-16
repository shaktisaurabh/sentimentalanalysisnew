#!/usr/bin/env python
# coding: utf-8

# In[1]:


#ETL is used for converting raw data to a featurized data. Featurized data is that which can be used to train a model.


# In[2]:


#Data to be extracted can be present in the form of csv or can be present in databases(SQL,nosql).


# In[34]:


#EDA or exploratory data analysis is to make insights from data using bra-charts,wordcloud or any other anlysis.
#video1----ambrish verma,anushka TVF romantic play(best friends)
#video2----TVF play romantic on na tum jano na hum song
#video3----about dhoni
#video4----fittuber product review
#video5----fittuber product review
#video6----fittuber product review 
#video7----fittuber product review 
#video8----goonda movie review


# In[53]:


import pandas as pd 
import numpy as np 
an=pd.read_csv("D://video_comments2.csv") 
an 


# In[54]:


an['comments']=an['comments'].str.lower()
an
#Lower casing


# In[55]:


an['comments']=an['comments'].str.strip() 
an 
#removing leading and trailing spaces


# In[56]:


an[an['comments'].str.contains(r"<.*?>")]


# In[57]:


an[an['comments'].str.contains(r"<.*?>")].iloc[0]


# In[58]:


an[an['comments'].str.contains(r"<.*?>")].iloc[0].values


# In[59]:


an[an['comments'].str.contains(r"https?://\S+|www\.\S+")]


# In[60]:


an[an['comments'].str.contains(r"https?://\S+|www\.\S+")].iloc[0].values


# In[61]:


import re
an['comments']=an['comments'].str.replace(r"https?://\S+|www\.\S+",'') 
an[an['comments'].str.contains(r"https?://\S+|www\.\S+")]
#URLS are removed


# In[62]:


an[an['comments'].str.contains(r"https?://\S+|www\.\S+")]


# In[63]:


an['comments']=an['comments'].str.replace(r"<.*?>","",regex=True)
#html tags too has been removed 


# In[64]:


an[an['comments'].str.contains(r"<.*?>")]


# In[65]:


def remove_abb(data):
    data = re.sub(r"he's", "he is", data)
    data = re.sub(r"there's", "there is", data)
    data = re.sub(r"we're", "we are", data)
    data = re.sub(r"that's", "that is", data)
    data = re.sub(r"won't", "will not", data)
    data = re.sub(r"they're", "they are", data)
    data = re.sub(r"can't", "cannot", data)
    data = re.sub(r"wasn't", "was not", data)
    data = re.sub(r"don't", "do not", data)
    data= re.sub(r"aren't", "are not", data)
    data = re.sub(r"isn't", "is not", data)
    data = re.sub(r"what's", "what is", data)
    data = re.sub(r"haven't", "have not", data)
    data = re.sub(r"hasn't", "has not", data)
    data = re.sub(r"there's", "there is", data)
    data = re.sub(r"he's", "he is", data)
    data = re.sub(r"it's", "it is", data)
    data = re.sub(r"you're", "you are", data)
    data = re.sub(r"i'm", "i am", data)
    data = re.sub(r"shouldn't", "should not", data)
    data = re.sub(r"wouldn't", "would not", data)
    data = re.sub(r"i'm", "i am", data)
    data = re.sub(r"i'm", "i am", data)
    data = re.sub(r"i'm", "i am", data)
    data = re.sub(r"isn't", "is not", data)
    data = re.sub(r"here's", "here is", data)
    data = re.sub(r"you've", "you have", data)
    data = re.sub(r"you've", "you have", data)
    data = re.sub(r"we're", "we are", data)
    data = re.sub(r"what's", "what is", data)
    data = re.sub(r"couldn't", "could not", data)
    data = re.sub(r"we've", "we have", data)
    data = re.sub(r"it's", "it is", data)
    data = re.sub(r"doesn't", "does not", data)
    data = re.sub(r"it's", "it is", data)
    data = re.sub(r"here's", "here is", data)
    data = re.sub(r"who's", "who is", data)
    data = re.sub(r"i've", "i have", data)
    data = re.sub(r"y'all", "you all", data)
    data = re.sub(r"can't", "cannot", data)
    data = re.sub(r"would've", "would have", data)
    data = re.sub(r"it'll", "it will", data)
    data = re.sub(r"we'll", "we will", data)
    data = re.sub(r"wouldn't", "would not", data)
    data = re.sub(r"we've", "we have", data)
    data = re.sub(r"he'll", "he will", data)
    data = re.sub(r"y'all", "you all", data)
    data = re.sub(r"weren't", "were not", data)
    data = re.sub(r"didn't", "did not", data)
    data = re.sub(r"they'll", "they will", data)
    data = re.sub(r"they'd", "they would", data)
    data = re.sub(r"don't", "do not", data)
    data = re.sub(r"that's", "that is", data)
    data = re.sub(r"they've", "they have", data)
    data = re.sub(r"i'd", "i would", data)
    data = re.sub(r"should've", "should have", data)
    data = re.sub(r"you're", "you are", data)
    data = re.sub(r"where's", "where is", data)
    data = re.sub(r"don't", "do not", data)
    data = re.sub(r"we'd", "we would", data)
    data = re.sub(r"i'll", "I will", data)
    data = re.sub(r"weren't", "were not", data)
    data = re.sub(r"they're", "they are", data)
    data = re.sub(r"can't", "cannot", data)
    data = re.sub(r"you'll", "you will", data)
    data = re.sub(r"i'd", "i would", data)
    data = re.sub(r"let's", "let us", data)
    data = re.sub(r"it's", "it is", data)
    data = re.sub(r"can't", "cannot", data)
    data = re.sub(r"don't", "do not", data)
    data = re.sub(r"you're", "you are", data)
    data = re.sub(r"i've", "i have", data)
    data = re.sub(r"that's", "that is", data)
    data = re.sub(r"i'll", "i will", data)
    data = re.sub(r"doesn't", "does not",data)
    data = re.sub(r"i'd", "i would", data)
    data = re.sub(r"didn't", "did not", data)
    data = re.sub(r"ain't", "am not", data)
    data = re.sub(r"you'll", "you will", data)
    data = re.sub(r"i've", "i have", data)
    data = re.sub(r"don't", "do not", data)
    data = re.sub(r"i'll", "i will", data)
    data = re.sub(r"i'd", "i would", data)
    data = re.sub(r"let's", "let us", data)
    data = re.sub(r"you'd", "you would", data)
    data = re.sub(r"it's", "it is", data)
    data = re.sub(r"ain't", "am not", data)
    data = re.sub(r"haven't", "have not", data)
    data = re.sub(r"could've", "could have", data)
    data = re.sub(r"youve", "you have", data)  
    data = re.sub(r"don't", "do not", data)
    
    return data


# In[66]:


an['comments']=an['comments'].apply(remove_abb) 
#expanding abbreviations


# In[67]:


import string 
string.punctuation 


# In[68]:


def remove_punctuation(amn):
    for i in string.punctuation:
        if i in amn:
            amn=amn.replace(i,'') 
    return amn 
#If punctuation is present in amn(each review) then the punctuation is replaced with '' blank.  


# In[69]:


an['comments']=an['comments'].apply(remove_punctuation) 
an


# In[70]:


#to avoid any error while reading a csv file, we use r"" followed by error_bad_lines=False 


# In[72]:


an.isnull().sum() 
# There are 0 rows where no comments are present as can be seen below


# In[1]:


import pandas as pd 
import numpy as np 
a1=pd.read_csv(r"C:\Users\LENOVO\Downloads\UScomments.csv",error_bad_lines=False) 
a1


# In[2]:


a1.isnull().sum() 
#there are no comments in 25 rows as can be seen below


# In[3]:


a1.dropna(inplace=True)


# In[4]:


a1.isnull().sum() 


# In[5]:


from textblob import TextBlob


# In[6]:


amb=a1.head().iloc[0]['comment_text'] 
amb 


# In[7]:


TextBlob(amb).sentiment.polarity 
#polarity is 0.0, hence it is neutral statement


# In[8]:


a1.shape
#As you can see, a1 column has 691375 rows and 4 columns 


# In[9]:


sample=a1[:400000]


# In[10]:


polarity=[] 
for comment in sample['comment_text']:
    try:
        polarity.append(TextBlob(comment).sentiment.polarity)
    except:
        polarity.append(0) 


# In[11]:


len(polarity)


# In[12]:


sample.head() 


# In[13]:


sample['polarity']=polarity 


# In[14]:


sample.head() 


# In[15]:


comments_positive=sample[sample['polarity']==1] 
comments_positive 


# In[16]:


comments_negative=sample[sample['polarity']==-1] 
comments_negative 


# In[17]:


a11=' '.join(comments_positive['comment_text']) 


# In[18]:


a21=' '.join(comments_negative['comment_text'])


# In[19]:


from wordcloud import WordCloud, STOPWORDS 


# In[20]:


import matplotlib.pyplot as plt 
wd=WordCloud(stopwords=set(STOPWORDS)).generate(a11)
plt.imshow(wd) 
plt.axis('off') 


# In[21]:


import matplotlib.pyplot as plt 
wd1=WordCloud(stopwords=set(STOPWORDS)).generate(a21) 
plt.imshow(wd1) 
plt.axis('off')


# In[22]:



def amn(k):
    l={}
    for i in k:
        if i in l:
            l[i]+=1
        else:
            l[i]=1 
    return l 
amn("aaabbbcccdddddd")


# In[23]:


a191=[[1,2,3],[9,10,11],[14,11,6]] 
[i for l in a191 for i in l] 
#This is how you flatten a 2D list 


# In[24]:


[i for l in a191 if i in l]


# In[ ]:


mj=[]
for i in a191:
    for l in i:
        mj.append(l) 
print(mj)


# In[25]:


[i for l in a191 for i in l]


# In[26]:


amb="samb👍 jhyu❤️ hytu❤" 


# In[27]:


import emoji 
[i for i in amb if i in emoji.EMOJI_DATA]


# In[28]:


import emoji 
l1=[]
for i in amb:
    if i in emoji.EMOJI_DATA:
        l1.append(i) 
l1


# In[29]:


set(emoji.EMOJI_DATA)


# In[30]:


[i for l in a191 for i in l]
#list comprehension is a way of replacing the nest loops to a easy code


# In[31]:


import emoji
ad1=[i for cmm in sample['comment_text'].dropna() for i in cmm if i in emoji.EMOJI_DATA] 
ad1


# In[34]:


from collections import Counter 
Counter(ad1)


# In[36]:


ad2=Counter(ad1).most_common(10) 
ad2


# In[39]:


emoji11=[ad2[i][0] for i in range(10)] 
emoji11


# In[43]:


freq11=[ad2[i][1] for i in range(10)] 
freq11


# In[44]:


import plotly.graph_objs as go 
from plotly.offline import iplot


# In[45]:


trace=go.Bar(x=emoji11,y=freq11) 
trace


# In[47]:


iplot([trace]) 
#From the emoji analysis of the comments , we got to know that a lot of people find the video funny and almost equal number of
#people are loving it though.


# In[50]:


import os 
os.listdir(r'D:\additional_data') 
#If you pass a path into it, it will print the list of all the files available in that directory


# In[51]:


import warnings


# In[2]:


import os 
ak1=os.listdir(r'D:\additional_data')
ak1


# In[3]:


files_csv=[file for file in ak1 if ".csv" in file] 
files_csv 
#list comprehension can also be used like this to differentiate csv files from json files


# In[4]:


files_json=[file for file in ak1 if ".json" in file] 
files_json


# In[5]:


import pandas as pd 
full_df=pd.DataFrame() 
path=r'D:\additional_data'
for cm in files_csv:
    new_df=pd.read_csv(path+'/'+cm,encoding='ISO-8859-1')
    full_df=pd.concat([full_df,new_df],ignore_index=False) 
full_df.shape


# In[60]:


full_df.head() 
#As you can see, we are still getting index as index has not been ignored


# In[1]:


import pandas as pd 
full_df=pd.DataFrame() 
path=r'D:\additional_data' 
for cm in files_csv:
    new_df=pd.read_csv(path+'/'+cm,encoding='ISO-8859-1') 
    full_df=pd.concat([full_df,new_df],ignore_index=True) 
full_df


# In[98]:


full_df.shape


# In[78]:


#encoding='ISO-8859-1' will ensure that we can intake a lot of other languages as well as emojis


# In[66]:


full_df.head() 


# In[69]:


full_df.duplicated()


# In[70]:


full_df[full_df.duplicated()].shape


# In[71]:


#As we can see, we have 36417 duplicate rows here and to remove them we use drop_duplicates() function , by default the keep
#parameter is 'first' if we make keep=False then all the duplicates are removed completely 


# In[72]:


full_df=full_df.drop_duplicates()


# In[74]:


full_df[full_df.duplicated()].shape 
#All such duplicate rows have been removed as shown here 
#keep='first' means first row is kept and rest deleted, keep='last' means that last row is kept and rest is deleted


# In[75]:


full_df.head()


# In[77]:


full_df.head(20).to_csv(r"D://banda1234.csv",index=False)


# In[79]:


full_df[:25].to_json(r"D://bandasingh.json")


# In[80]:


from sqlalchemy import create_engine 


# In[81]:


from sqlalchemy import create_engine


# In[82]:


engine=create_engine(r"sqlite:///D:\additional_data/sample.sqlite") 


# In[83]:


full_df[:30].to_sql('user1',con=engine,if_exists='append') 


# In[85]:


a1nn={1:'amit',2:'games',3:'uyi',4:'fyu',5:'uhg'} 
import pandas as pd 
a1nw=pd.DataFrame({
    'cat_id':[1,1,2,3,3,4,1,3,4,5,1,2]
}) 
a1nw['cat_id'].map(a1nn)


# In[86]:


import pandas as pd
import matplotlib.pyplot as plt

# Example dataset
data = [18, 20, 22, 25, 26, 28, 30, 32, 35, 40, 45, 50, 55]

# Create a DataFrame
df = pd.DataFrame(data, columns=['Age'])

# Plot a box plot
plt.figure(figsize=(6, 4))
plt.boxplot(df['Age'])
plt.title('Box Plot of Student Ages')
plt.ylabel('Age')
plt.show()


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Example dataset
data = [18, 20, 22, 25, 26, 28, 30, 32, 35, 40, 45, 50, 55]

# Create a DataFrame
df = pd.DataFrame({
    'Age':[18, 20, 22, 25, 26, 28, 30, 32, 35, 40, 45, 50, 55],
    'type':['music','act','bnd','music','bnd','music','bnd','act','bnd','act','music','act','bnd']
})

# Plot a box plot
plt.figure(figsize=(12, 8))
sns.boxplot(x='type',y='Age',data=df)
plt.title('Box Plot of Student Ages')
plt.ylabel('Age') 
plt.xticks(rotation='vertical')
plt.show()
#20,50,40,32 
#sns boxplot takes sum of Age with respect to each distinct value of type


# # Most liked category

# In[6]:


import pandas as pd 
json_df=pd.read_json(r"D:\additional_data\US_category_id.json") 


# In[8]:


json_df 


# In[7]:


json_df['items'][0]


# In[8]:


json_df['items'][1]


# In[9]:


dckr={} 
for cm in json_df['items']:
    dckr[int(cm['id'])]=cm['snippet']['title'] 
dckr 


# In[10]:


full_df.head()


# In[11]:


full_df['category_name']=full_df['category_id'].map(dckr) 
full_df.head() 


# In[104]:


import matplotlib.pyplot as plt 
import seaborn as sns 
plt.figure(figsize=(12,8)) 
sns.boxplot(x='category_name',y='likes',data=full_df) 
plt.xticks(rotation='vertical')


# In[106]:


#Most of what we see here are the outliers..... These are those datapoints that are either much more than most 75 percentile data
#points or much less than 25 percentile datapoints. The length of the box is the difference between 75th percentile and 25th 
#percentile. The whiskers(lines drawn) are lines drawn across the box.
#whiskers in boxplots represent maximum and minimum datapoint that are not outliers


# In[108]:


full_df.head()


# In[111]:


full_df.columns 


# In[12]:


full_df['likes_rate']=(full_df['likes']/full_df['views'])*100 
full_df['dislikes_rate']=(full_df['dislikes']/full_df['views'])*100 
full_df['comment_count_rate']=(full_df['comment_count']/full_df['views'])*100 


# In[13]:


full_df.columns 


# In[117]:


import matplotlib.pyplot as plt 
import seaborn as sns 
plt.figure(figsize=(8,6)) 
sns.boxplot(x='category_name',y='likes_rate',data=full_df) 
plt.xticks(rotation='vertical') 
plt.show() 


# In[118]:


#As you can see Comedy category_name has the most before 50 percentile, has the most datapoints before 75 percentile.  


# In[119]:


sns.regplot(x='views',y='likes',data=full_df)


# In[120]:


full_df[['views','likes','dislikes']].corr() 


# In[121]:


#As you can see as views increase by 100 likes increase by factor of 0.77 meaning 77...simiarly dislikes too increase
#as views increase although at a lesser co-factor of 0.42 


# In[123]:


import seaborn as sns 
sns.heatmap(full_df[['views','likes','dislikes']].corr(),annot=True)


# In[128]:


import seaborn as sns 
sns.heatmap(full_df[['views','likes',]].corr())
#annot=True would give us  clear picture about the correspondence


# # Analyzing trending videos

# In[132]:


full_df.columns


# In[133]:


full_df['channel_title'].value_counts()


# In[15]:


andesh=full_df['channel_title'].value_counts().reset_index().rename(columns={'index':'channel_name','channel_title':'number'}).set_index('channel_name')


# In[16]:


andesh


# In[17]:


import pandas as pd 
cdf=full_df.groupby('channel_title').size().sort_values(ascending=False).reset_index() 
cdf


# In[18]:


cdf=cdf.rename(columns={0:'countss'})
cdf


# In[21]:


import plotly.express as px 


# In[22]:


px.bar(cdf[:20],x='channel_title',y='countss')


# In[23]:


import plotly.express as px

# Example DataFrame cdf and Plotly Express bar chart
# Assuming cdf contains data and has been defined earlier
bar_chart = px.bar(cdf[:20], x='channel_title', y='countss')

# Save the bar chart as an interactive HTML file
bar_chart.write_html('D:/bar_chart.html')


# In[24]:


import plotly.express as px

# Example DataFrame cdf and Plotly Express bar chart
# Assuming cdf contains data and has been defined earlier
bar_chart = px.bar(cdf[:20], x='channel_title', y='countss', text='countss')
bar_chart
# Update layout to display values on top of bars
# bar_chart.update_traces(textposition='outside')


# In[25]:


import plotly.express as px

# Example DataFrame cdf and Plotly Express bar chart
# Assuming cdf contains data and has been defined earlier
bar_chart = px.bar(cdf[:20], x='channel_title', y='countss', text='countss')

# Update layout to display values on top of bars
bar_chart.update_traces(textposition='outside')


# # Does punctuation in title and tags have impact on the number of likes,comments,views and dislikes?

# In[28]:


full_df.columns


# In[29]:


import string 
string.punctuation


# In[30]:


[char for char in full_df['title'][0] if char in string.punctuation]


# In[34]:


full_df['title'][0]


# In[35]:


full_df['title']


# In[36]:


full_df['title'][0]


# In[37]:


full_df['title'].iloc[0]


# In[39]:


[char for char in full_df['title'].iloc[0] if char in string.punctuation]


# In[40]:


def num_punct(amb):
    return len([char for char in amb if char in string.punctuation])


# In[41]:


full_df['count_punc']=full_df['title'].apply(num_punct)


# In[42]:


import matplotlib.pyplot as plt 
import seaborn as sns 


# In[48]:


plt.figure(figsize=(8,6)) 
sns.boxplot(x='count_punc',y='likes',data=full_df) 
plt.show() 


# In[49]:


#as you can the the title that has 2,3,4,5 punctuations often have many likes as they have a lot of outliers


# In[51]:


plt.figure(figsize=(8,6)) 
sns.boxplot(x='count_punc',y='dislikes',data=full_df) 
plt.show() 


# In[52]:


#as u can see the title that has 1 or 3 punctuation has can have many dislikes as u can see ....many dislike outliers(data points)
#are far apart in this category where punctuations are 1 or 3. 


# In[53]:


plt.figure(figsize=(8,6)) 
sns.boxplot(x='count_punc',y='comment_count',data=full_df) 
plt.show() 


# In[54]:


#AS u can see title with just 1 punctuation mark has many outliers at a far-away distance. so title with 1 punctuation mark
#has many comments 


# In[50]:


full_df.columns


# In[43]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


# In[44]:


# Example DataFrame
data = {
    'Category': ['A', 'A', 'B', 'B', 'B', 'C', 'C', 'C', 'C'],
    'Value': [10, 15, 20, 25, 30, 35, 40, 45, 50]
}
df = pd.DataFrame(data)


# In[46]:


# Create box plot
sns.boxplot(x='Category', y='Value', data=df)
plt.title('Box Plot of Value by Category')
plt.show()
#As you can see boxplot shows all the datapoints of Value w.r.t Category 


# In[47]:


full_df.head()


# In[ ]:




