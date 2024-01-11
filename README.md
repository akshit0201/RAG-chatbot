# Importing required libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import numpy as np
import random
import string

# Downloading required nltk packages
nltk.download('punkt')
nltk.download('wordnet')

# Defining the ChatBot class
class ChatBot:
    def __init__(self):
        self.text = None
        self.sent_tokens = None
        self.lemmer = nltk.stem.WordNetLemmatizer()

    def load_data(self, data):
        self.text = data
        self.sent_tokens = nltk.sent_tokenize(self.text)

    def LemTokens(self, tokens):
        return [self.lemmer.lemmatize(token) for token in tokens]

    def LemNormalize(self, text):
        remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
        return self.LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

    def response(self, user_response):
        robo_response=''
        self.sent_tokens.append(user_response)
        TfidfVec = TfidfVectorizer(tokenizer=self.LemNormalize, stop_words='english')
        tfidf = TfidfVec.fit_transform(self.sent_tokens)
        vals = cosine_similarity(tfidf[-1], tfidf)
        idx=vals.argsort()[0][-2]
        flat = vals.flatten()
        flat.sort()
        req_tfidf = flat[-2]
        if(req_tfidf==0):
            robo_response=robo_response+"I am sorry! I don't understand you"
            return robo_response
        else:
            robo_response = robo_response+self.sent_tokens[idx]
            return robo_response

    def get_response(self, user_response):
        robo_response = self.response(user_response)
        self.sent_tokens.remove(user_response)
        return robo_response

# Usage
bot = ChatBot()
# Load your text data here
bot.load_data("Your text data here")
print(bot.get_response("Your query here"))
