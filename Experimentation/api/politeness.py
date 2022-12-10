import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import convokit
from convokit import Corpus, Speaker, Utterance
from convokit import TextParser
from convokit import download
from pandas import DataFrame
from typing import List, Dict, Set
import random
from sklearn import svm
from scipy.sparse import csr_matrix
from sklearn.metrics import classification_report
from convokit import Classifier
from convokit import PolitenessStrategies
import matplotlib
import matplotlib.pyplot as plt



def generate_politeness(df_in, corpus_train = 'wikipedia', percentage_top_tweets=0.25,polite_or_impolite = 'polite'):
    """
    This function uses the stanford politeness corpus to train a logistic regression model to sort polite and impolite tweets

    Inputs:
    df_in, Pandas Dataframe: Containing one column named 'Tweets' with text data
    corpus_train, str: Choice of corpus to use as training. Either 'wikipedia' or 'stack-exchange'
    percentage_top_tweets, float: percentage of highest rated polite or impolite tweets
    polite_or_impolite, str: 'polite' or 'impolite'. Choice to return most polite or most impolite

    returns a pandas df with the top percentage_top_tweets % of polite or impolite tweets
    
    """
    # Downloading the wikipedia portion of annotated data
    if corpus_train == 'wikipedia': #default
        wiki_corpus = Corpus(download("wikipedia-politeness-corpus")) #other options is stack-exchange-politeness-corpus
    if corpus_train =='stack-exchange':
        wiki_corpus = Corpus(download("stack-exchange-politeness-corpus")) 

    df_in = df_in #
    name = 'Tweeter'
    df = pd.DataFrame(columns=['id','speaker','conversation_id','reply_to','timestamp','text'])
    id_col = list(range(len(df_in)))
    speaker_col = [name]*len(df_in)
    conversation_col = list(range(len(df_in)))
    reply_col = ['self']*len(df_in)
    time_col = [0]*len(df_in)
    text_col = df_in["Tweets"]

    df['id'] = id_col
    df['speaker'] = speaker_col
    df['conversation_id'] = conversation_col
    df['reply_to'] = reply_col
    df['timestamp'] = time_col
    df['text'] = text_col
    test_corp = Corpus.from_pandas(df)

    
    parser = TextParser(verbosity=1000)
    #parse train and test. We use wiki_corpus. We can change this by alftering the first line in this function
    wiki_corpus = parser.transform(wiki_corpus)
    test_corp = parser.transform(test_corp)
    ps = PolitenessStrategies()
    wiki_corpus = ps.transform(wiki_corpus, markers=True)
    test_corp = ps.transform(test_corp, markers=True)


    binary_corpus = Corpus(utterances=[utt for utt in wiki_corpus.iter_utterances() if utt.meta["Binary"] != 0])
    #training
    # clf_cv = Classifier(obj_type="utterance", 
    #                     pred_feats=["politeness_strategies"], 
    #                     labeller=lambda utt: utt.meta['Binary'] == 1)

    # clf_cv.evaluate_with_cv(binary_corpus)

    #Now I just use old train test split approach:
    # clf_split = Classifier(obj_type="utterance", 
    #                     pred_feats=["politeness_strategies"], 
    #                     labeller=lambda utt: utt.meta['Binary'] == 1)

    # clf_split.evaluate_with_train_test_split(binary_corpus)
    # test_ids = binary_corpus.get_utterance_ids()[-100:]
    train_corpus = Corpus(utterances=[utt for utt in binary_corpus.iter_utterances()]) #note I just make this all points hahahah - Andrew
    # test_corpus = Corpus(utterances=[utt for utt in binary_corpus.iter_utterances() if utt.id in test_ids])
    clf = Classifier(obj_type="utterance", 
                        pred_feats=["politeness_strategies"], 
                        labeller=lambda utt: utt.meta['Binary'] == 1)
    clf.fit(train_corpus)

    test_pred = clf.transform(test_corp)
    
    scores = clf.summarize(test_pred) # df
    x = scores['pred_score'].tolist()
    plt.ylabel('Politeness')
    plt.xlabel('Point Dds')
    plt.title('Mapping of point IDs to politeness score')
    plt.plot(x)
    #From here dowm, untested! XX TESTED 12/10/2022
    if polite_or_impolite == 'polite':
        choices = scores.loc[scores['prediction']==1] #get all points with polite prediction
        choices = choices.reset_index()
        choice_ids = choices['id']
        
    else:
        choices = scores.loc[scores['prediction']==0] #get all points that are rude :(
        choices = choices.reset_index()
        choice_ids = choices['id']
        choice_ids = choice_ids[::-1] #reverse so the rudes are on top
    print("len", len(choice_ids))
    end_index = len(choice_ids)//(1/percentage_top_tweets) #stop point
    output = [] #list of tweets. Will append this to a df and return that df in the end
    for i in range(0,int(end_index)):
        id = choice_ids[i] #get id for point
        loc = df.loc[df['id']==int(id)]
        output.append(loc['text'].tolist()[0])
    df_out = pd.DataFrame(columns=["Tweets"])
    df_out["Tweets"] = output
    return df_out
    
    
    
    
        
    

