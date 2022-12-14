#Libary for functions
import os
import numpy as np
import re
import pandas as pd
from bs4 import BeautifulSoup
from transformers import GPT2Model, GPT2Config
import random
# import keyword_extraction_w_parser



def clean_tweet(tweet, allow_new_lines = True):
        bad_start = ['http:', 'https:']
        for w in bad_start:
            tweet = re.sub(f" {w}\\S+", "", tweet)      # removes white space before url
            tweet = re.sub(f"{w}\\S+ ", "", tweet)      # in case a tweet starts with a url
            tweet = re.sub(f"\n{w}\\S+ ", "", tweet)    # in case the url is on a new line
            tweet = re.sub(f"\n{w}\\S+", "", tweet)     # in case the url is alone on a new line
            tweet = re.sub(f"{w}\\S+", "", tweet)       # any other case?
        tweet = re.sub(' +', ' ', tweet)                # replace multiple spaces with one space
        if not allow_new_lines:                         # TODO: predictions seem better without new lines
            tweet = ' '.join(tweet.split())
        return tweet.strip()
        
def boring_tweet(tweet):
    "Check if this is a boring tweet"
    boring_stuff = ['http', '@', '#']
    not_boring_words = len([None for w in tweet.split() if all(bs not in w.lower() for bs in boring_stuff)])
    return not_boring_words < 3

def filter_manual(df, length_min, lang_choice): 
    #filter retweets
    df = df.loc[df['retweetedTweet'].isna()]
    #filter for only lang_choice
    df = df.loc[df['lang'] == lang_choice]
    # print(df)

    # Remove all urls and picture (just remove https, media shows up like "Happy #CowAppreciationDay. https://t.co/cIBYktvsu2")
    data_new = []
    data_old = df[["renderedContent"]].values.tolist()
    for index, value in enumerate(data_old):
        # print("index",index)
        tweet = value[0] #original tweet
        # print(tweet)
        
        filt_tweet = re.sub(r'http\S+', '', tweet) #Removes media and links with https:
        # filt_tweet = re.sub(r'\b(\w+.gov)\S+','',filt_tweet) #Remove .gov
        # filt_tweet = re.sub(r'\b(\w+.com)\S+','',filt_tweet) #Remove .gov
        # filt_tweet = re.sub(r'\b(\w+.org)\S+','',filt_tweet) #Remove .gov
        filt_tweet = re.sub("\S+[^\d\W]+\.[^\d\W]+\S+", "", filt_tweet) #removes any word that contains the pattern: word(notdigit).word(notdigit)
        # filt_tweet = re.sub(r'/\S+', '', filt_tweet) #IDEK where theses residual links are coming from
        


        filt_tweet = re.sub(' +', ' ', filt_tweet) #Removes double spaces
        filt_tweet = BeautifulSoup(filt_tweet, "lxml").get_text(strip=True) #clean it all up:
        #short tweet: could also just set to nan and add to list 
        # filt_tweet = list(filt_tweet.split())
        # print(type(filt_tweet))
        # filt_tweet.reverse()
        # data_new.append(filt_tweet[0:2])
        # print(filt_tweet[0:2])
        if len(filt_tweet.split()) > length_min:
            data_new.append(single_line(filt_tweet))
    return data_new

def single_line(tweet):
    a = tweet
    b = ""
    for letter in a:
        if letter != "\n":
            b += letter
        else:
            b +=". "
    return b

def gen_input(cool_tweets,EPOCHS):
    """
    Input: 
    - cool_tweets: arr of tweets that pass
    Output:
    - total_text: a single string of all tweets spaced out by <|endoftext|> markers"
    """
    seed_data = random.randint(0,2**32-1)
    dataRandom = random.Random(seed_data)
    total_text = '<|endoftext|>'
    all_handle_tweets = []
    epoch_len = max(len(''.join(cool_tweet)) for cool_tweet in cool_tweets)
    for _ in range(EPOCHS):
        for cool_tweet in cool_tweets:
            dataRandom.shuffle(cool_tweet) #Removed as random.Random.shuffle was depreciated
            current_tweet = cool_tweet
            current_len = len(''.join(current_tweet))
            while current_len < epoch_len:
                for t in cool_tweet:
                    current_tweet.append(t)
                    current_len += len(t)
                    if current_len >= epoch_len: break
            dataRandom.shuffle(current_tweet)
            all_handle_tweets.extend(current_tweet)
    total_text += '<|endoftext|>'.join(all_handle_tweets) + '<|endoftext|>'
    return total_text

def gen_input_special_tokens(cool_tweets,EPOCHS, special_tokens):
    """
    Input: 
    - cool_tweets: array of tweets that pass
    - special_tokens: list of special tokens ['<|SenSanders|>','<|polite|>']. 
    Output:
    - total_text: a single string of all tweets spaced out by <|endoftext|> markers"
    """
    special_token_str = ''
    for token in special_tokens:
        special_token_str += token #Should look like <|token1|><|token2|>
    start_token = '<|endoftext|>' + special_token_str
    seed_data = random.randint(0,2**32-1)
    dataRandom = random.Random(seed_data)
    total_text = '<|endoftext|>'
    all_handle_tweets = []
    epoch_len = max(len(''.join(cool_tweet)) for cool_tweet in cool_tweets)
    for _ in range(EPOCHS):
        for cool_tweet in cool_tweets:
            dataRandom.shuffle(cool_tweet) #Removed as random.Random.shuffle was depreciated
            current_tweet = cool_tweet
            current_len = len(''.join(current_tweet))
            while current_len < epoch_len:
                for t in cool_tweet:
                    current_tweet.append(t)
                    current_len += len(t)
                    if current_len >= epoch_len: break
            dataRandom.shuffle(current_tweet)
            all_handle_tweets.extend(current_tweet)
    total_text += start_token.join(all_handle_tweets) + '<|endoftext|>'
    return total_text

def addTag(df,tag):
    """
    adds a tag (str) to every element in a df's elements
    """
    lst_strings = df["Tweets"]

    df_new = pd.DataFrame(columns=["Tweets"])
    df_new["Tweets"] = [tag+string for string in lst_strings]
    return df_new




