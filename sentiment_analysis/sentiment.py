'''
APIKEY: 9CACIn614VYMP2MDIWiuanRol
APIKEY_SECRET: gT1KkIqgPInq9XVi6hj3FlKtB6LmYseczT8gl8NrRXuiXtZ33H
BEARER_TOKEN: AAAAAAAAAAAAAAAAAAAAAI%2FFaAEAAAAASD%2B%2FcZkzLne7htgHfKggoMTEfh4%3DDNPgY3Mx3C6EHHkzBljoqfvrQLz8rnU5Rzxqp1NqGsC9IEVARZ

ACCESS_TOKEN: 1503065591485140993-0fCYxD0vEW6SLCHaz1VJjUF41BCx1Y
ACCESS_TOKEN_SECRET: T1IU9k3jpo5nocMTnNSWwcGrXr8AaAiy9OcNFGB3l7dbr
'''

import os
from numpy import result_type
import tweepy as tw
import pandas as pd
import json 

consumer_key = '9CACIn614VYMP2MDIWiuanRol'
consumer_secret = 'gT1KkIqgPInq9XVi6hj3FlKtB6LmYseczT8gl8NrRXuiXtZ33H'
access_token = '1503065591485140993-0fCYxD0vEW6SLCHaz1VJjUF41BCx1Y'
access_token_secret = 'T1IU9k3jpo5nocMTnNSWwcGrXr8AaAiy9OcNFGB3l7dbr'

auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)

search_words = "Reliance Industries and stock -filter:retweets"

tweets = api.search_tweets(q=search_words, count=10000, lang='en', tweet_mode='extended', result_type='recent')
print(len(tweets))
locations = [] 
times = [] 
count = 0 
df = pd.DataFrame()
tweets_list = [] 
for i, tweet in enumerate(tweets):
    
    tweet = tweet._json
    tweets_list.append(tweet['full_text'])
    if (not tweet["retweeted"]) and ('RT @' not in tweet["full_text"]):
        count += 1

        #df.append([str(tweet["full_text"]), str(tweet["location"]), str(tweet["time"])])

df['Text'] = tweets_list
print(df.head())
print(len(tweets), count)
# for tweet in tweets:
#     print(tweet)
#     print('------------------------------------------')
#     print(tweet.text.strip())
#     print('------------------------------------------')
#     print(tweet.text)
#     times.append(tweet.created_at)
#     locations.append(tweet.user.location)

# print("\n".join([tweet.text for tweet in tweets]))
# print("\n".join(map(str, times)))
# print("\n".join(locations))