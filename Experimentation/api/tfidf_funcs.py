import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import CountVectorizer
import re
import nltk
from nltk.corpus import stopwords
nltk.download('averaged_perceptron_tagger')

"""
Source:
https://www.nltk.org/_modules/nltk/tag/perceptron.html

"""


def pre_process(text):
    """"
    For cleaning data for tfidf
    """
    
    # lowercase
    text=text.lower()
    
    #remove tags
    text=re.sub("","",text)
    
    # remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ",text)
    return text


def sort_coo(coo_matrix):
    """"
    Helper function for keywords
    """
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items
    Helper function for keywords
    
    """
    
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    for idx, score in sorted_items:
        fname = feature_names[idx]
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results



def tfidf(df_idf):
    """
    returns a list of lists: each internal list is the top 2 keywords in
    """
    from nltk.corpus import stopwords
    nltk.download('stopwords')
    stopwords=stopwords.words('english')
    df_idf['text'] = df_idf['Tweets'].apply(lambda x:pre_process(x))
    docs=df_idf['text'].tolist()
    #create a vocabulary of words, 
    #ignore words that appear in 85% of documents, 
    #eliminate stop words

    cv=CountVectorizer(max_df=0.85,stop_words=stopwords,max_features=10000)

    word_count_vector=cv.fit_transform(docs)
    tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
    tfidf_transformer.fit(word_count_vector)
    return cv, tfidf_transformer


def get_Keywords(cv, tfidf_transformer, tweet):
    """
    cv and tfidf come from tfidf
    tweet (string): string to get keywords for
    
    """
    
    # you only needs to do this once
    feature_names=cv.get_feature_names_out()
    # get the document that we want to extract keywords from
    doc= pre_process(tweet)
    #generate tf-idf for the given document
    tf_idf_vector=tfidf_transformer.transform(cv.transform([doc]))
    #sort the tf-idf vectors by descending order of scores
    sorted_items=sort_coo(tf_idf_vector.tocoo())
    #extract only the top n; n here is 10
    keywords=extract_topn_from_vector(feature_names,sorted_items,2)
    out = []

    for k in keywords:
        val = nltk.pos_tag([k])[0][1]
        if (val == 'NN' or val == 'NNS' or val == 'NNPS' or val == 'NNP'):
            out.append(k)
    return out


def add_Keywords(df,cv,tfidf_transformer):
    keywords = []
    new_tweets = []
    for tweet in df['Tweets'].tolist():
        words = get_Keywords(cv, tfidf_transformer, tweet)
        
        if not words:
            continue
        if len(words) == 1:
            string_new = ''
            word = words[0] 
            word = "<|"+word+"|>"#only one word
            keywords.append(word)
            word += "<|undefined|>"
            string_new += word
            new_tweets.append(string_new+tweet)
        else: 
            string_new = ''
            for word in words:
                word = "<|"+word+"|>"
                keywords.append(word) #add keyword before adding to string of both words
                string_new += word
            new_tweets.append(string_new+tweet)

    df_new = pd.DataFrame(columns=["Tweets"])
    df_new["Tweets"] = new_tweets
    return df_new, keywords