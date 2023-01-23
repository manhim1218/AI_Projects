#!/usr/bin/env python
# coding: utf-8


import gensim
import pandas as pd
import ujson
import numpy as np



df = pd.read_csv("metadata.csv")
# print(df.head())
print(df.shape)
print(df.abstract[0])


# ### Exploratory Data Anaylsis + Data Preprocessing

# Extract useful columns from the original dataframe
df_info = df[['publish_time','authors','journal','title','pdf_json_files','abstract']].copy()



# Replace any empty cells with nan values
df_info.replace(' ', np.nan, inplace = True)



# Check any null values
print(df_info.isnull().values.any())



# Remove rows containing null values
df_info = df_info.dropna()


df_info


### reset dataframe index
df_info = df_info.reset_index(drop=True)



# Since our task is to make a COVID-19 related Question and Answer Tool, so we are only extracting COVID-10 related research articles from the dataset
# Count occurence of covid in the columns of title and abstract 
sum_covid_title = df_info['title'].str.contains('COVID-19').sum()
print("Occurence of COVID-19 in Title: " + str(sum_covid_title))
print("No Occurence of COVID-19 in Title: " + str(len(df_info) - (sum_covid_title)))
list_occurence_COVID_19_title = [sum_covid_title,(len(df_info) - (sum_covid_title))]
list_occurence_tile = ['Occurence of COVID-19', 'No Occurence of COVID-19']



# plot Occurence of COVID-19 in the title column
import matplotlib.pyplot as plt
plt.figure(figsize=(5,5))
plt.bar(list_occurence_tile,list_occurence_COVID_19_title)
plt.title('Occurence of COVID-19 in the title column')
plt.xlabel('Occurence category')
plt.ylabel('Number of occuerence')
plt.savefig('Occurence of COVID-19 in the title column.png')
plt.show()


### Extract rows basee on the condition of cotaining 'COVID-19'  in the column of Title 
df_search_title_containing_covid = df_info[df_info['title'].str.contains("COVID-19")]


df_search_title_containing_covid


# Visualise the occurence of words 'COVID-19' from the abstract of this dataframe 
df_search_title_containing_covid['abstract'].str.contains('COVID-19').sum()


sum_covid_title_abstract = df_search_title_containing_covid['abstract'].str.contains('COVID-19').sum()
print("Occurence of COVID-19 on both title and abstract: " + str(sum_covid_title_abstract))
print("No Occurence of COVID-19 in Title but not in abstract: " + str(len(df_search_title_containing_covid) - (sum_covid_title_abstract)))
list_occurence_COVID_19_title_abstract = [sum_covid_title_abstract,(len(df_search_title_containing_covid) - (sum_covid_title_abstract))]



# Occurence of COVID-19 in both column of title and abstract
import matplotlib.pyplot as plt
plt.figure(figsize=(5,5))
plt.bar(list_occurence_tile,list_occurence_COVID_19_title_abstract)
plt.title('Occurence of COVID-19 in both column of title and abstract')
plt.xlabel('Occurence category')
plt.ylabel('Number of occuerence')
plt.savefig('Occurence of COVID-19 in both column of title and abstract.png')
plt.show()


### Extract data cotaining word 'COVID-19' from Abstract into a new dataframe  
df_search_title_containing_covid = df_search_title_containing_covid[df_search_title_containing_covid['abstract'].str.contains("COVID-19")]


df_search_title_containing_covid



# Convert publish time column to pandas datetime type
df_search_title_containing_covid['publish_time'] = pd.to_datetime(df_search_title_containing_covid['publish_time'], dayfirst=True)



# Extract COVID-19 related articles after 2021-01-01
df_search_title_containing_covid_filtered = df_search_title_containing_covid.loc[(df_search_title_containing_covid['publish_time'] >= '2021-01-01')]


# Reindex the dataframe
df_search_title_containing_covid_filtered = df_search_title_containing_covid_filtered.reset_index(drop=True)

# top 10 covid-19 cases countries 
top_10_covid_countries = ['US', 'India', 'Brazil', 'UK', 'Russia', 'Turkey', 'France', 'Iran', 'Argentina', 'Spain']


# Sum up the occurence of the nominated countries in the title column
top_10_covid_countries_sum=[]
for name in top_10_covid_countries:
    top_10_covid_countries_sum.append(df_search_title_containing_covid_filtered['title'].str.contains(name).sum())
print(top_10_covid_countries_sum)


# plot Distribution of number of scholar articles related to the countries of having the most COVID-19 cases
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.bar(top_10_covid_countries,top_10_covid_countries_sum)
plt.title('Distribution of number of scholar articles related to the countries of having the most COVID-19 cases')
plt.xlabel('Countries')
plt.ylabel('Number of occurence')
plt.savefig('Distribution of number of scholar articles related to the countries of having the most COVID-19 cases.png')
plt.show()

# Extracting month from the column of publish time 
df_search_title_containing_covid_filtered['month'] = df_search_title_containing_covid_filtered['publish_time'].dt.month


# Sum up numbers of scholar acticles published in each month up to October in 2021
import calendar
df_search_title_containing_covid_filtered['month'] = df_search_title_containing_covid_filtered['month'].apply(lambda x: calendar.month_name[x])
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October']
month_accumulate=[]
for month in months:
    month_accumulate.append(df_search_title_containing_covid_filtered['month'].str.contains(month).sum())
print(month_accumulate)


# plot Distribution of number of COVID-19 related articles published in different months up to October in 2021
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.bar(months,month_accumulate)
plt.title('Distribution of number of COVID-19 related articles published in different months up to October in 2021')
plt.xlabel('Months')
plt.ylabel('Number of COVID-19 related articles')
plt.savefig('Distribution of number of COVID-19 related articles published in different months up to October in 2021.png')
plt.show()


# Due to the limitation of the computer performance, I initially select the first 30000 articles to train my NLP model. 
df_search_title_containing_covid_filtered = df_search_title_containing_covid_filtered.head(30000)


# To extract the json file name from the dataframe
json_file=[]
for i in range(len(df_search_title_containing_covid_filtered)):
    json_file.append((df_search_title_containing_covid_filtered.iloc[i,4].split("json/",1)[1]).split(";",1)[0])



# Read the first 30000 selected json files to assign it to a list
# If there is no locaton data from the pjson file, then we will not use that article
path = r"C:\Users\manhi\Desktop\Archive BIG DATA COVID\document_parses\pdf_json"
sentence = {}
sentence_list = []
paper_id_list = []
country_list = []
title_name_list = []
for j in range(len(json_file)):
    f = open(path+ "\\" + json_file[j],)
    data = ujson.load(f)
    paper_id = data['paper_id']
    try:
        country_name = data['metadata']['authors'][0]['affiliation']['location']['country']
        title_name = data['metadata']['title']
    except KeyError: 
        continue
    except IndexError:
        continue
    for i in range (len(data['body_text'])):
        title_name_list.append(title_name)
        country_list.append(country_name)
        paper_id_list.append(paper_id)
        sentence_list.append(data['body_text'][i]['text'])



# Convert title list to dataframe 
title_name_df = pd.DataFrame(title_name_list)
title_name_df.columns = ['title']

# Convert country list to dataframe 
country_selected_articles = pd.DataFrame(country_list)
country_selected_articles.columns = ['country']

top_10_published_scholar_articles = ['US', 'China', 'Germany','UK','Japan','France','Canada', 'Switzerland', 'South Korea', 'Australia']
top_10_country_selected_articles = []
for name in top_10_published_scholar_articles:
    top_10_country_selected_articles.append(country_selected_articles['country'].str.contains(name).sum())
print(top_10_country_selected_articles)

# plot Distribution of number of COVID-19 related articles published by the top leading countries in science research
plt.figure(figsize=(15,5))
plt.bar(top_10_published_scholar_articles,top_10_country_selected_articles)
plt.title('Distribution of number of COVID-19 related articles published by the top leading countries in science research')
plt.xlabel('Countries')
plt.ylabel('Number of selected articles')
plt.savefig('Distribution of number of COVID-19 related articles published by the top leading countries in science research.png')
plt.show()


# Convert the sentence list to dataframe
df_body_context = pd.DataFrame(sentence_list)

# Assign column name to the dataframe
df_body_context.columns = ['body_content']

# Merge two dataframe together
merge_df_body_context = pd.concat([df_body_context,title_name_df], axis=1)

# Extract abstract column from the original dataframe and assign it to a new dataframe
# Double check any null values
df_abstract = df_search_title_containing_covid_filtered[['abstract','title']]
print(df_abstract.isnull().values.any())

# Reindex dataframe
df_abstract = df_abstract.reset_index(drop=True)

# Convert type to string to all both body context and abstract dataframe
df_abstract = df_abstract.astype(str)
merge_df_body_context = merge_df_body_context.astype(str)


# rename column names
df_abstract = df_abstract.rename(columns={"abstract": "word"})

# rename column names
merge_df_body_context = merge_df_body_context.rename(columns={"body_content": "word"})

# Merge both dataframe to one dataframe
df_merge = pd.concat([df_abstract,merge_df_body_context], axis=0)

# Convert all types to string in dataframe
df_merge = df_merge.astype(str)


faq_keywords_covid = ['coronavirus', 'symptoms', 'transmission', 'incubation period', 'treatment', 'death rate', 'prevent', 'travel', 'face mask', 'social distancing', 'children', 'asymptomatic ' , 'contagious', 'contact tracing']
faq_keywords_covid_accumulate=[]
for keyword in faq_keywords_covid:
    faq_keywords_covid_accumulate.append(df_merge['word'].str.contains(keyword).sum())

# plot Distribution of occurence of some keywords selected from top 10 most commonly asked questions related to COVID-19
plt.figure(figsize=(30,5))
plt.bar(faq_keywords_covid,faq_keywords_covid_accumulate)
plt.title('Distribution of occurence of some keywords selected from top 10 most commonly asked questions related to COVID-19')
plt.xlabel('Keywords')
plt.ylabel('Number of occurence')
plt.savefig('Distribution of occurence of some keywords selected from top 10 most commonly asked questions related to COVID-19.png')
plt.show()


# Create an extra column showing the length of each context 
df_merge['length'] = df_merge.word.str.len()

# Only select the length less than 1000 words for better answer quality
df_modified = df_merge[df_merge.length <= 1000]

df_modified = df_modified.reset_index(drop=True)


faq_keywords_covid_accumulate=[]
for keyword in faq_keywords_covid:
    faq_keywords_covid_accumulate.append(df_modified['word'].str.contains(keyword).sum())


# plot Distribution of COVID-19 related keywords after selecting context with less than 1000 words
plt.figure(figsize=(30,5))
plt.bar(faq_keywords_covid,faq_keywords_covid_accumulate)
plt.title('Distribution of COVID-19 related keywords after selecting context with less than 1000 words')
plt.xlabel('Keywords')
plt.ylabel('Number of occurence')
plt.savefig('Distribution of COVID-19 related keywords after selecting context with less than 1000 words')
plt.show()


# Preprocess data such as tokenisation, lowering cases, punctuation removal
word = df_modified.word.apply(gensim.utils.simple_preprocess)

# Visualise all preprocessed tokenised word for each sentence 
print(word)


# ### Word2Vec Model Training


# Model training, 
# setting window to 10 (conhesion of 10 words before and after a word)
# Min count to 2 (minimum 2 words in a sentence)
# Workers = 4 for quad core CPU running
model = gensim.models.Word2Vec(
    window=5,
    min_count=1,
    workers=4
)

# Build vocabulary from the preprocessed word
model.build_vocab(word, progress_per = 1000)

#Model training 
model.train(word, total_examples=model.corpus_count, epochs=model.epochs)



# Try to test cosine similiarity between a single vocabulary and any word from sentences, for example covid
model.wv.most_similar("covid")

# Try to test another cosine similiarity between a single vocabulary and any word from sentences, for example covid
model.wv.most_similar("vaccine")


def print_answer(mydict):
    for key in mydict.keys():
        val = mydict[key]
    sort_orders = sorted(mydict.items(), key=lambda x: x[1], reverse=True)
    print("Answer 1: ")
    print(df_modified.word[sort_orders[0][0]])
    print("Extracted from (ie.title of scholar article): ")
    print(df_modified.title[sort_orders[0][0]])
    print("--------------------------------")
    print("Answer 2: ")
    print(df_modified.word[sort_orders[1][0]])
    print("Extracted from (ie.title of scholar article): ")
    print(df_modified.title[sort_orders[1][0]])
    print("--------------------------------")
    print("Answer 3: ")
    print(df_modified.word[sort_orders[2][0]])
    print("Extracted from (ie.title of scholar article): ")
    print(df_modified.title[sort_orders[2][0]])
    print("--------------------------------")
    print("Answer 4: ")
    print(df_modified.word[sort_orders[3][0]])
    print("Extracted from (ie.title of scholar article): ")
    print(df_modified.title[sort_orders[3][0]])
    print("--------------------------------")
    print("Answer 5: ")
    print(df_modified.word[sort_orders[4][0]])
    print("Extracted from (ie.title of scholar article): ")
    print(df_modified.title[sort_orders[4][0]])
    print("--------------------------------")
    print("Answer 6: ")
    print(df_modified.word[sort_orders[5][0]])
    print("Extracted from (ie.title of scholar article): ")
    print(df_modified.title[sort_orders[5][0]])
    print("--------------------------------")
    print("Answer 7: ")
    print(df_modified.word[sort_orders[6][0]])
    print("Extracted from (ie.title of scholar article): ")
    print(df_modified.title[sort_orders[6][0]])
    print("--------------------------------")
    print("Answer 8: ")
    print(df_modified.word[sort_orders[7][0]])
    print("Extracted from (ie.title of scholar article): ")
    print(df_modified.title[sort_orders[7][0]])
    print("--------------------------------")
    print("Answer 9: ")
    print(df_modified.word[sort_orders[8][0]])
    print("Extracted from (ie.title of scholar article): ")
    print(df_modified.title[sort_orders[8][0]])
    print("--------------------------------")
    print("Answer 10: ")
    print(df_modified.word[sort_orders[9][0]])
    print("Extracted from (ie.title of scholar article): ")
    print(df_modified.title[sort_orders[9][0]])


def Question_Answering_tool(question):
    for i in range(len(word)):
        try:
            listToStr = ' '.join(map(str, word[i]))
            modified_question = gensim.utils.simple_preprocess(question)
            distance = model.wv.n_similarity(modified_question, listToStr.lower().split())
            mydict[i] = distance
        except KeyError:
            continue
        except ZeroDivisionError:
            continue
    return mydict


mydict = {}
# question = "What is the medications for COVID-19"
question = input("Enter a question: ")
# question = 'Can vaccine protect us from getting COVID-19'
mydict = Question_Answering_tool(question)
print_answer(mydict)

