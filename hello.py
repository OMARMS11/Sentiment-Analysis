
import pandas as pd
import pickle
import joblib
import warnings
import string
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize



porter_stemmer = PorterStemmer()
#functions
def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text
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
def normalize_text(text):
    # Convert text to lowercase
    #text = text.lower()

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Handle contractions (optional)
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
        text = text.replace(contraction, expanded)

    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text
def stem_text(text):
    # Tokenize the text into words
    words = word_tokenize(text)

    # Stem each word using the PorterStemmer
    stemmed_words = [porter_stemmer.stem(word) for word in words]

    # Join the stemmed words back into a string
    stemmed_text = ' '.join(stemmed_words)

    return stemmed_text
def remove_stopwords(text):
    # Tokenize the text (split into words)
    words = re.findall(r'\w+', text.lower())

    # Get the English stop words from NLTK
    stop_words = set(stopwords.words('english'))

    # Remove stop words from the text
    filtered_words = [word for word in words if word not in stop_words]

    # Join the filtered words back into a string
    filtered_text = ' '.join(filtered_words)

    return filtered_text
def replace_words(text, replacements):
    for word, replacement in replacements.items():
        text = text.replace(word, replacement)
    return text

warnings.filterwarnings("ignore")
 #load csv file
df = pd.read_csv('sentimentdataset.csv')

#variables
words_to_replace = {
    "acceptance" : "positive",
    "accomplishment" : "positive",
    "admiration" : "positive",
    "affection" : "positive",
    "amazement" : "positive",
    "amusement" : "positive",
    "arousal" : "positive",
    "charm" : "positive",
    "colorful" : "positive",
    "contemplation" : "positive",
    "dazzle" : "positive",
    "determination" : "positive",
    "ecstasy" : "positive",
    "elation" : "positive",
    "euphoria" : "positive",
    "elegance" : "positive",
    "empathetic" : "positive",
    "enjoyment" : "positive",
    "enthusiasm" : "positive",
    "excitement" : "positive",
    "freedom" : "positive",
    "radiance" : "positive",
    "fulfillment" : "positive",
    "grateful" : "positive",
    "gratitude" : "positive",
    "happiness" : "positive",
    "happy" : "positive",
    "harmony" : "positive",
    "contentment" : "positive",
    "hope" : "positive",
    "hopeful" : "positive",
    "inspiration" : "positive",
    "joy" : "positive",
    "kind" : "positive",
    "joyfulreunion" : "positive",
    "love" : "positive",
    "playful" : "positive",
    "mesmerizing" : "positive",
    "innerjourney" : "positive",
    "pride" : "positive",
    "proud" : "positive",
    "relief" : "positive",
    "satisfaction" : "positive",
    "success" : "positive",
    "serenity" : "positive",
    "tenderness" : "positive",
    "compassionate" : "positive",
    "compassion" : "positive",
    "positive in baking" : "positive",
    "spark" : "positive",
    "exploration" : "positive",
    "artisticburst" : "positive",
    "winter magic" : "positive",
    "emotion" : "positive",
    "thrill" : "positive",
    "vibrancy" : "positive",
    "adventure" : "positive",
    "challenge" : "positive",
    "creative positive" : "positive",
    "yearning" : "positive",
    "enchantment" : "positive",
    "resilience" : "positive",
    "positive" : "positive",
    "pensive" : "positive",
    "positiveful" : "positive",
    "freespirited" : "positive",
    "zest" : "positive",
    "positivefulreunion" : "positive",
    "inspired" : "positive",
    "positivereunion" : "positive",
    "oceans positive" : "positive",
    "sorrow" : "negative",
    "suffering" : "negative",
    "resentment" : "negative",
    "solitude" : "negative",
    "shame" : "negative",
    "sadness" : "negative",
    "obstacle" : "negative",
    "whimsy" : "negative",
    "sad" : "negative",
    "ruins" : "negative",
    "regret" : "negative",
    "energy" : "positive",
    "anxiety" : "negative",
    "overwhelmed" : "negative",
    "exhaustion" : "negative",
    "numbness" : "negative",
    "loneliness" : "negative",
    "jealousy" : "negative",
    "isolation" : "negative",
    "heartbreak" : "negative",
    "grief" : "negative",
    "frustrated" : "negative",
    "frustration" : "negative",
    "fearful" : "negative",
    "fear" : "negative",
    "embarrassed" : "negative",
    "disgust" : "negative",
    "disappointed" : "negative",
    "heartache" : "negative",
    "disappointment" : "negative",
    "Disgust" : "negative",
    "Confusion" : "negative",
    "bad" : "negative",
    "anger" : "negative",
    "awe" : "positive",
    "betrayal" : "negative",
    "bitter" : "negative",
    "bitterness" : "negative",
    "boredom" : "negative",
    "confusion" : "negative",
    "devastated" : "negative",
    "desolation" : "negative",
    "intimid" : "negative",
    "ambivalence" : "negative",
    "apprehensive" : "negative",
    "captivation" : "negative",
    "dismissive" : "negative",
    "envious" : "negative",
    "hate" : "negative",
    "intrigue" : "negative",
    "melancholy" : "negative",
    "mischievous" : "negative",
    "miscalculation" : "negative",
    "negativeness" : "negative",
    "despair" : "negative",
    "negativeation" : "negative",
    "whispers of the past" : "negative",
    "touched" : "positive",
    "reverence" : "positive",
    "reflection" : "positive",
    "nostalgia" : "negative",
    "negativesweet" : "negative",
    "lostpositive" : "positive",
    "indifference" : "negative",
    "envisioning history" : "positive",
    "engagement" : "positive",
    "curiosity" : "positive",
    "natures beauty" : "positive",
    "renewed effort" : "positive",
    "creativity" : "positive",
    "suspense" : "positive",
    "coziness" : "positive",
    "connection" : "positive",
    "calmness" : "positive",
    "empowerment" : "positive",
    "surprise" : "positive",
    "adrenaline" : "negative",
    "anticipation" : "positive",
    "positivealstorm" : "positive",
    "tranquility" : "positive",
    "emotionalstorm" : "positive",
    "wonder" : "positive",
    "celestial wonder" : "positive",
    "celestial positive" : "positive",
    "celestial neutral" : "positive",
    ##Amr words

    "helplessness":"negative",
    "neutral":"negative",
    "darkness":"negative",
"loss":"negative",
"envy": "negative",
"jealous": "negative",
"hypnotic": "negative",
"desperation": "negative",
"dreamchaser": "negative",
"sympathy": "positive",
"culinary positive": "positive",
"positiveing journey": "positive",
"runway positive" : "positive",
"positivement": "positive",
"adoration" : "positive",
"overpositiveed": "positive",
"marvel": "positive",
"festivepositive": "positive",
"breakthrough": "positive",
"heartwarming": "positive",
"positivepositive": "positive",
"confidence": "positive",
"positiveness": "positive",
"culinaryodyssey": "positive",
"grandeur": "positive",
"iconic": "positive",
"motivation": "positive",
"confident": "positive",
"blessed": "positive",
"appreciation": "positive",
"positivity": "positive",
"rejuvenation": "positive",
"optimism": "positive",
"celebration": "positive",
"mindfulness": "positive",
"melodic": "positive",
"journey": "positive",
"triumph": "positive",
"solace": "negative",
"imagination":"positive",
"pressure": "negative",
"friendship": "positive",
"immersion": "negative",
"romance": "positive"



}



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
    targetText[i] =replace_words(targetText[i],words_to_replace)




#Feature selection Part !!

target = targetText
feature = editedText


#Writing the model !!

#1 Split the data set to training and testing

X_train,X_test,Y_train,Y_test= train_test_split(feature,target,test_size=0.45,random_state=2024)

#2 convert the text to numirical valaues

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

#3 Train a Multinomial Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_vec, Y_train)
y_pred = classifier.predict(X_test_vec)
print(classification_report(Y_test,y_pred))

#pickle.dump(classifier, open("sentiment model.pkl", "wb"))
pickle.dump(vectorizer, open("vectors1.pickle", "wb"))
"""""
for i in range(0,10):
    text= input(": ")

    #newText = remove_emoji(text)
    newText = normalize_text(text)
    newText = remove_articles(newText)
    newText = remove_stopwords(newText)
    newText =stem_text(newText)


    t = [newText]

    text_vec = vectorizer.transform(t)
    prediction = classifier.predict(text_vec)
    s = ''.join(prediction)
    d_prob = classifier.predict_proba(text_vec)
    print('you are feeling',s)
    nump = int(d_prob[:,1]*100)
    numn = int(d_prob[:,0]*100)

    if prediction == 'positive':
        print( f'I am {nump}% Sure')
    else :
        print(f'I am {numn}% Sure')
"""""






