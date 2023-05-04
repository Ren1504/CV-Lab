with open('sample.txt', 'r') as f:
    line = f.readline()
    print(line)


#A Change into LowerCase

lowerCase = line.lower()
print(lowerCase) 

#B Remove Punctuation

import string

translator = str.maketrans("","",string.punctuation)
print(line.translate(translator))

#C Remove Digits

digit_translator = str.maketrans("","",string.digits)
print(line.translate(digit_translator))

#D Remove StopWords

import nltk
nltk.download('stopwords') #include this when the stopwords are not downlaoded
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

words = word_tokenize(line)

filtered = [word for word in words if word.lower() not in stopwords.words('english')]
filtered_sentence = ' '.join(filtered)

print(filtered_sentence)

#E Text Stemming and Lemmatization

from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('wordnet')
nltk.download('omw-1.4')

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

stemmed_words = [stemmer.stem(word) for word in words] #words from previous
lemmatized_words = [lemmatizer.lemmatize(word) for word in words] #words from previous

print(stemmed_words)
print(lemmatized_words)