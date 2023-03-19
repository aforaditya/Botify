#import packages
import tweepy
import pandas as pd
import re
# from sklearn.feature_extraction.text import CountVectorizer
# import string
from flask import Flask


app = Flask(__name__)

@app.route('/')
def main():
    #Authentaction
    consumerKey = '6LdA0uZymPQR8bhbJhIHdbdvq'
    consumerSecret = '24xLnZKNJeQdzBgqDTiEeEXC1D7mZq91Jr9LMn3La2zE3m2A2y'
    accessToken = '1581899576734519296-SEcpSxHQvsdiBASOj5QbNnyoeu5nal'
    accessTokenSecret = 'g9v01RDM8cvXc4wnDB4SzWXtd2TEqygA34CFwiO7c4II2'
    auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
    auth.set_access_token(accessToken, accessTokenSecret)
    api = tweepy.API(auth)

    #Searching for keywords
    keyword = "Punjab"
    noOfTweet = 100
    tweets = tweepy.Cursor(api.search_tweets, q=keyword).items(noOfTweet)

    #Create variables to store data
    positive = 0
    negative = 0
    neutral = 0
    polarity = 0
    tweet_list = [] #text in tweet
    neutral_list = [] #
    negative_list = []
    positive_list = []
    location_list = []
    scr_name_list = []
    tweet_id = []
    tweet_link = []
    user_id_list = []
    retweet = []
    date_time_list = []
    followers_count_list = []
    profile_link_list = []
    verified_list = []

    #fetch data

    for tweet in tweets:
            tweet_list.append(tweet.text)
            location_list.append(tweet.user.location)
            scr_name_list.append('@' + tweet.user.screen_name)
            tweet_id.append(tweet.id)
            user_id_list.append(tweet.user.id_str)
            tweet_link.append('https://twitter.com/anyuser/status/{}'.format(tweet.id))
            date_time_list.append(tweet.created_at)
            followers_count_list.append(tweet.user.followers_count)
            profile_link_list.append('https://twitter.com/{}'.format(tweet.user.screen_name))
            if tweet.text.startswith("RT @") == True:
                retweet.append(True)
            else:
                retweet.append(False)
            verified_list.append(tweet.user.verified)

    #Convert data
    data = {'tweet_text': tweet_list, 'location': location_list, 'user_name': scr_name_list, 'tweet_id': tweet_id,
            'user_id': user_id_list,'tweet_link': tweet_link, 'date_time': date_time_list, 'followers': followers_count_list,
            'profile_link': profile_link_list, 'retweet':retweet, 'verified': verified_list}
    data1 = pd.DataFrame(data)

    # Perprocess text in tweet and store in dataframe
    def preprocess_text(text):
        # remove "@RT" string
        text = re.sub("^@RT[\s]+", "", text)
        text = re.sub("^RT[\s]+", "", text)
        # remove @user mentions, URLs, and non-alphanumeric characters
        text = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text)
        # convert to lowercase
        text = text.lower()
        # return the preprocessed text
        return text
    data1["processed_text"] = data1["tweet_text"].apply(preprocess_text)

    #Sentiment analysis
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    #%%
    def predict(text):
        # tokenize the text
        encoded_text = tokenizer(text, return_tensors='pt')
        # perform inference using the model
        logits = model(**encoded_text).logits
        # get the predicted class (0 for negative, 1 for neutral, 2 for positive)
        predicted_class = logits.argmax().item()
        # map the predicted class to a string label
        labels = ['negative', 'neutral', 'positive']
        predicted_label = labels[predicted_class]
        # return the predicted label as a string
        return predicted_label

    data1["sentiment"] = data1["processed_text"].apply(predict)

    # # Find positive and negative impacts
    # data2 = data1.sort_values(by='followers', ascending=False)
    #
    # # create separate dataframes for positive and negative sentiments
    # pos_df = data2[data2['sentiment'] == 'positive'].sort_values(by='followers', ascending=False)
    # neg_df = data2[data2['sentiment'] == 'negative'].sort_values(by='followers', ascending=False)

    # Calculating tweet’s lenght and word count
    # tw_list = pd.DataFrame(data1)
    # tw_list['text_len'] = tw_list['processed_text'].astype(str).apply(len)
    # tw_list['text_word_count'] = tw_list['processed_text'].apply(lambda x: len(str(x).split()))
    # round(pd.DataFrame(tw_list.groupby("sentiment").text_len.mean()), 2)

    # def remove_punct(text):
    #      text = "".join([char for char in text if char not in string.punctuation])
    #      text = re.sub('[0–9]+', '', text)
    #      return text
    #
    # tw_list['punct'] = tw_list['processed_text'].apply(lambda x: remove_punct(x))
    #
    # # Appliyng tokenization
    # def tokenization(text):
    #     text = re.split('\W+', text)
    #     return text
    #
    # tw_list['tokenized'] = tw_list['punct'].apply(lambda x: tokenization(x.lower()))
    # # Removing stopwords
    # import nltk
    # stopword = nltk.corpus.stopwords.words('english')
    #
    # def remove_stopwords(text):
    #     text = [word for word in text if word not in stopword]
    #     return text
    #
    # tw_list['nonstop'] = tw_list['tokenized'].apply(lambda x: remove_stopwords(x))
    # # Appliyng Stemmer
    # import nltk
    # ps = nltk.PorterStemmer()
    #
    # def stemming(text):
    #     text = [ps.stem(word) for word in text]
    #     return text
    #
    # tw_list['stemmed'] = tw_list['nonstop'].apply(lambda x: stemming(x))
    #
    #  # Cleaning Text
    # def clean_text(text):
    #         text_lc = "".join([word.lower() for word in text if word not in string.punctuation])  # remove puntuation
    #         text_rc = re.sub('[0-9]+', '', text_lc)
    #         tokens = re.split('\W+', text_rc)  # tokenization
    #         text = [ps.stem(word) for word in tokens if word not in stopword]  # remove stopwords and stemming
    #         return text
    #
    #  # Appliyng Countvectorizer
    # countVectorizer = CountVectorizer(analyzer=clean_text)
    # countVector = countVectorizer.fit_transform(tw_list['processed_text'])
    #     #print('{} Number of reviews has {} words'.format(countVector.shape[0], countVector.shape[1]))
    #     # print(countVectorizer.get_feature_names())
    # count_vect_df = pd.DataFrame(countVector.toarray(), columns=countVectorizer.get_feature_names_out())
    # count = pd.DataFrame(count_vect_df.sum())
    # countdf = count.sort_values(0, ascending=False).head(20)


    # def positive_impact(pos_df=pos_df):
    #     top_pos = pos_df.head(5)[['followers', 'user_name', 'profile_link', 'tweet_text']]
    #     return top_pos.to_json()
    #
    # def negative_impact(neg_df=neg_df):
    #     top_neg = neg_df.head(5)[['followers', 'user_name', 'profile_link', 'tweet_text']]
    #     return top_neg.to_json()
    #
    # def verified_senti(data1=data1):
    #
    #     verified_sentiments = data1[data1['verified'] == True].groupby('sentiment').size()
    #     verified_sentiments = pd.DataFrame(verified_sentiments)
    #     return verified_sentiments.to_json()
    #
    # def repet_words(countdf=countdf):
    #     repeted_words = countdf[1:12]
    #     return repeted_words.to_json()
    #
    # def sentiments(data1 = data1):
    #     sentiment_counts = data1['sentiment'].value_counts()
    #     sentiment_counts['total'] = len(data1)
    #     return sentiment_counts.to_json()


    # FINAL CODE CELL
    # Metrices
    # neu, pos neg
    total_tweets = len(data1['sentiment'])
    # positive_tweets =
    neutral_tweets = 0
    positive_tweets = 0
    negative_tweets = 0

    sentiment = data1['sentiment'].value_counts()
    try:
        neutral_tweets = sentiment[0]
        positive_tweets = sentiment[1]
        negative_tweets = sentiment[2]
    except:
        pass


    t_count = {
        'total': int(total_tweets),
        'positive_tweets': int(positive_tweets),
        'negative_tweets': int(negative_tweets),
        'neutral_tweets': int(neutral_tweets)
    }
    # Sug2
    sug2 = data1[['date_time', 'followers']]
    sug2 = sug2.groupby(['date_time']).sum()
    t = sug2.index.values
    sug2 = sug2.sort_values(by='followers', ascending= False).reset_index(drop=True)
    sug2 = sug2.set_index(t)
    d_sug2 = sug2.head(10).to_json()

    # SUG3
    sug3 = data1[['retweet', 'followers']]
    sug3 = sug3.sort_values(by='followers', ascending=False).reset_index(drop=True)
    sug3 = sug3.groupby(['retweet']).sum()
    d_sug3 = sug3.to_json()

    # Sug4
    # 'ver_follo_pie'
    sug4 = data1[['verified', 'followers']]
    sug4 = sug4.groupby(['verified']).sum()
    sug4 = sug4.sort_values(by='followers', ascending=False)
    d_sug4 = sug4.to_json()

    # Sug 5
    sun_5 = data1[['tweet_link', 'followers']]
    sun_5 = sun_5.groupby(['tweet_link']).sum()
    sun_5 = sun_5.sort_values(by='followers', ascending=False)
    d_sug5 = sun_5.head(10).to_json()

    # Sug 6
    sug_6 = data1[['sentiment', 'followers']]
    sug_6 = sug_6.groupby(['sentiment']).sum()
    sug_6 = sug_6.sort_values(by='followers', ascending=False)
    d_sug6= sug_6.to_json()

    # SUGGESTION 7 PIE
    sug7 = data1[['user_name', 'followers']]
    sug7 = sug7.groupby(['user_name']).sum()
    sug7 = sug7.sort_values(by='followers', ascending=False)
    d_sug7 = sug7.head(10).to_json()

    # SUGGESTION 8
    su8 = data1.loc[data1['sentiment'] == 'positive']
    su8 = su8[['user_name', 'followers']]
    su8 = su8.groupby(['user_name']).sum()
    su8 = su8.sort_values(by='followers', ascending=False)
    d_sug8 = su8.head(10).to_json()

    # Sug12
    su12 = data1.loc[data1['verified'] == False]
    su12 = su12[['profile_link', 'followers']]
    su12 = su12.groupby(['profile_link']).sum()
    su12 = su12.sort_values(by='followers', ascending=False)
    d_sug12 = su12.head(10).to_json()

    # suggestion 14
    # pie
    su14 = data1.loc[data1['retweet'] == True]
    su14 = su14[['sentiment', 'followers']]
    su14 = su14.groupby(['sentiment']).sum()
    su14 = su14.sort_values(by='followers', ascending=False)
    d_sug14 = su14.to_json()
    # Location
    loc = data1['location'].value_counts()
    loc = loc.head(12).to_json()


    final_dict = {
        "Metrics": t_count,
        "Location": loc,
        "Sug2": d_sug2,
        "Sug3": d_sug3,
        "Sug4": d_sug4,
        "Sug5": d_sug5,
        "Sug6": d_sug6,
        "Sug7": d_sug7,
        "Sug8": d_sug8,
        "Sug12": d_sug12,
        "Sug14": d_sug14
    }

    df = pd.DataFrame(final_dict)
    print(final_dict)
    return final_dict

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')



