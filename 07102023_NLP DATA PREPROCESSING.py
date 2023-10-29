## NLP INTRODUCTION


#NLP - Natural Language processing:
# sentiments: Positive, Neutral, Negative
#
'''
we will use nltk library for NLP:
pip install nltk
'''
import nltk
#1. Convert into lowercase
text = "Product is great but I amn't liking the colors as they are worst"
text = text.lower()

'''
2. Tokenize the content: break it into words or sentences
'''
text1 = text.split()
#using nltk
from nltk.tokenize import sent_tokenize,word_tokenize
text = word_tokenize(text)
#print("Text =\n",text)
#print("Text =\n",text1)

'''
3. Removing Stop words: Words which are not significant
for your analysis. E.g. an, a, the, is, are
'''
my_stopwords = ['is','i','the']
text1 = text
for w in text1:
    if w in my_stopwords:
text.remove(w)
print("Text after my stopwords:",text1)

nltk.download("stopwords")
from nltk.corpus import stopwords
nltk_eng_stopwords = set(stopwords.words("english"))
#print("NLTK list of stop words in English: ",nltk_eng_stopwords)
'''
Just for example: we see the word but in the STOP WORDS but
we want to include it, then we need to remove the word from the set
'''
# removing but from the NLTK stop words
nltk_eng_stopwords.remove('but')

for w in text:
    if w in nltk_eng_stopwords:
text.remove(w)
print("Text after NLTK stopwords:",text)

'''
4. Stemming: changing the word to its root
eg: {help: [help, helped, helping, helper]}

One of the method is Porter stemmer
'''
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
text = [stemmer.stem(w) for w in text]
''' above line is like below:
t_list=[]
for w in text:
    a = stemmer.stem(w)
t_list.append(a)
'''
print("Text after Stemming:",text)
'''
5. Part of Speech Tagging (POS Tagging)
grammatical word which deals with the roles they place
like - 8 parts of speeches - noun, verb, ...

Reference: https://www.educba.com/nltk-pos-tag/
POS Tagging will give Tags like

CC: It is the conjunction of coordinating
CD: It is a digit of cardinal
DT: It is the determiner
EX: Existential
FW: It is a foreign word
IN: Preposition and conjunction
JJ: Adjective
JJR and JJS: Adjective and superlative
LS: List marker
MD: Modal
NN: Singular noun
NNS, NNP, NNPS: Proper and plural noun
PDT: Predeterminer
WRB: Adverb of wh
WP$: Possessive wh
WP: Pronoun of wh
WDT: Determiner of wp
VBZ: Verb
VBP, VBN, VBG, VBD, VB: Forms of verbs
UH: Interjection
TO: To go
RP: Particle
RBS, RB, RBR: Adverb
PRP, PRP$: Pronoun personal and professional

But to perform this, we need to download any one tagger:
e.g. averaged_perceptron_tagger
nltk.download('averaged_perceptron_tagger')
'''
nltk.download('averaged_perceptron_tagger')

import nltk
from nltk.tag import DefaultTagger
py_tag = DefaultTagger ('NN')
tag_eg1 = py_tag.tag (['Example', 'tag'])
print(tag_eg1)

#txt = "Example of nltkpos tag list"
#txt = ['product', 'great', 'but', "not", 'like', 'color']
#txt = word_tokenize(txt)
#txt = ['Example','of','nltk','pos','tag','list']
pos_txt = nltk.pos_tag(text)
print("POS Tagging:", pos_txt)

'''
6. Lemmetising
takes a word to its core meaning
We need to download:  wordnet
'''
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
print("Very good = ",lemmatizer.lemmatize("very good"))
print("Halves = ",lemmatizer.lemmatize("halves"))

text = "Product is great but I amn't liking the colors as they are worst"
text = word_tokenize(text)
text = [lemmatizer.lemmatize(w) for w in text]
print("Text after Lemmatizer: ",text)

'''
Full fledged Project work as assignment:
#Practice analyzing review comment of an ecommerce company and understand
why they like or dislike certain products. This data is in Portuguese

Book: Data Visualization using Python by Swapnil Saurav

Entire Project content is available online here:
https://designrr.page/?id=151900&token=3955289376&type=FP&h=7075

Dataset is available from here:
https://github.com/swapnilsaurav/OnlineRetail

Start from page number: 6 from Pareto Analysis
'''


## WORD CLOUD PROGRAM


'''
This program does web scrapping and gets the data and creates a Word Cloud
- pictorial format of showing the number of appearences of each word
'''
link="http://kahanistore.com/blog/shantanu-rakhta-charitra-book-i/"
import nltk
from urllib.request import urlopen
html_content = urlopen(link).read()
#print(html_content)
from bs4 import BeautifulSoup
soup = BeautifulSoup(html_content)
#print(soup)
for script in soup(["script","style"]):
    script.extract()  #remove it
#print(soup)
text = soup.get_text()
print(text)
#break them into lines and remove trailing spaces
lines = (line.strip() for line in text.splitlines())
# breaking the multiple lines into a line each
chunks = (phrase.strip() for line in lines for phrase in line.split(" "))
#print(chunks)
text = 'n'.join(chunk for chunk in chunks if chunk)
#print(text)

from nltk.corpus import stopwords
stopwords = set(stopwords.words("english"))

#tokenise
from nltk.tokenize import word_tokenize
words = word_tokenize(text)
#remove punctuation
words_txt = [word.lower() for word in words if word.isalpha()]

#now remove stopwords
words_txt = [word for word in words_txt if word not in stopwords]

#Now plot the wordcloud
# pip install wordcloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt
wc = WordCloud(max_words=200, background_color="white").generate(' '.join(words_txt))
plt.imshow(wc)
plt.show()


## NLP SENTIMENT INTENSITY ANALYZER


# Sentiment analysis - read the sentiments of each sentence
'''
If you need more data for your analysis, this is a good source:
https://github.com/pycaret/pycaret/tree/master/datasets

We will use Amazon.csv for this program

'''
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem  import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

link = "https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/amazon.csv"
df = pd.read_csv(link)
print(df)

#Let's create a function to perform all the preprocessing steps
# of a nlp analysis
def preprocess_nlp(text):
    #tokenise
    #print("0")
text = text.lower() #lowercase
    #print("1")
text = word_tokenize(text)  #tokenize
    #print("2")
text = [w for w in text if w not in stopwords.words("english")]
#lemmatize
    #print("3")
lemm = WordNetLemmatizer()
#print("4")
text = [lemm.lemmatize(w) for w in text]
#print("5")
    # now join all the words as we are predicting on each line of text
text_out = ' '.join(text)
#print("6")
return text_out

# import Resource vader_lexicon
import nltk
nltk.download('vader_lexicon')


df['reviewText'] = df['reviewText'].apply(preprocess_nlp)
print(df)

# NLTK Sentiment Analyzer
# we will now define a function get_sentiment() which will return
# 1 for positive and 0 for non-positive
analyzer = SentimentIntensityAnalyzer()
def get_sentiment(text):
    score = analyzer.polarity_scores(text)
    sentiment = 1 if score['pos'] >0 else 0
returnsentiment

df['sentiment'] = df['reviewText'].apply(get_sentiment)

print("Dataframe after analyzing the sentiments: \n",df)

#confusion matrix
from sklearn.metrics import confusion_matrix
print("Confusion matrix:\n",confusion_matrix(df['Positive'],df['sentiment']))

''' RESULT

Confusion matrix:
 [[ 1131  3636]
 [  576 14657]]
 Accuracy: (1131 + 14657) / (1131 + 14657 + 576 + 3636) = 15788/20000 = 78.94%
'''


