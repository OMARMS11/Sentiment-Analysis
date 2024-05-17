import nltk
import pandas as pd
import pickle
import joblib
import warnings
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

#Some things we need in the functions
porter_stemmer = PorterStemmer()
sid = SentimentIntensityAnalyzer()
warnings.filterwarnings("ignore")




#functions
def Sentiment_replace(text):
    s_score = sid.polarity_scores(text)["compound"]
    if s_score >= 0.05:
        return "positive"
    elif s_score < -0.05 :
        return "negative"
    else:
        return "neutral"
def remove_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642" 
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)
def remove_articles(text):
    articles = ["a", "an", "the"]
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in articles]
    return ' '.join(filtered_words)
def normalize_text(Featuretext):

    # Remove punctuation
    Featuretext = re.sub(r'[^\w\s]', '', Featuretext)

    # Handle contractions
    contractions = {
        "aren't": "are not",
        "can't": "cannot",
        "couldn't": "could not",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hasn't": "has not",
        "haven't": "have not",
        "he's": "he is",
        "I'm": "I am",
        "isn't": "is not",
        "it's": "it is",
        "let's": "let us",
        "mustn't": "must not",
        "shan't": "shall not",
        "she's": "she is",
        "shouldn't": "should not",
        "that's": "that is",
        "there's": "there is",
        "they're": "they are",
        "wasn't": "was not",
        "we're": "we are",
        "weren't": "were not",
        "what's": "what is",
        "where's": "where is",
        "who's": "who is",
        "won't": "will not",
        "wouldn't": "would not",
        "you'll": "you will",
        "you're": "you are",
        "you've": "you have"
    }
    for contraction, expanded in contractions.items():
        Featuretext = Featuretext.replace(contraction, expanded)

    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', Featuretext).strip()

    return Featuretext
def stem_text(Featuretext):

    words = word_tokenize(Featuretext)
    stemmed_words = [porter_stemmer.stem(word) for word in words]
    stemmed_text = ' '.join(stemmed_words)

    return stemmed_text
def remove_stopwords(Featuretext):
    # Tokenize the text (split into words)
    words = re.findall(r'\w+', Featuretext.lower())

    # Get the English stop words from NLTK
    stop_words = set(stopwords.words('english'))

    # Remove stop words from the text
    filtered_words = [word for word in words if word not in stop_words]

    # Join the filtered words back into a string
    filtered_text = ' '.join(filtered_words)

    return filtered_text


 #load csv file
df = pd.read_csv('sentimentdataset.csv')


# convert text to lower case
editedText = df['Text'].str.lower()
targetText = df['Sentiment (Label)'].str.lower()



# Cleaning the text to be ready to be put to the model
for i in range(0,732):

   editedText[i] = remove_emoji(editedText[i])
   editedText[i] = normalize_text(editedText[i])
   editedText[i] = remove_articles(editedText[i])
   editedText[i] = remove_stopwords(editedText[i])
   editedText[i] = stem_text(editedText[i])




#Cleaning the sentiment column
for i in range(0,732):
    targetText[i] = normalize_text(targetText[i])
    targetText[i] =Sentiment_replace(targetText[i])




#1 Split the data set to training and testing

X_train,X_test,Y_train,Y_test = train_test_split(editedText,targetText,test_size=0.25,random_state=10)


#Feature selection Part !!

vectorizer = CountVectorizer(ngram_range=(1,1))

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

#3 Train a Multinomial Naive Bayes classifier


MnModel =MultinomialNB()
MnModel.fit(X_train_vec, Y_train)
prediction = MnModel.predict(X_test_vec)
print(classification_report(Y_test,prediction))



pickle.dump(MnModel, open("sentiment model.pkl", "wb"))

pickle.dump(vectorizer, open("vectors1.pickle", "wb"))

"""

for i in range(0,10):
    text= input(": ")
    # text =remove_emoji(text)
    # text = normalize_text(text)
    # text = remove_articles(text)
    # text = remove_stopwords(text)
    # text = stem_text(text)

    pred = classifier.predict(vectorizer.transform([text]))

    if pred == "positive":
        print("positive")
    elif pred == "negative":
        print("negative")
    else:
        print("neutral")

"""



