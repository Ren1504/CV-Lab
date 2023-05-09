import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("sentiment.csv")
y = data["sentiment"]
xraw = data["review"]


vec = TfidfVectorizer()
x = vec.fit_transform(xraw)
x = x.toarray()


x_train, x_test, y_train, y_test = train_test_split(x, y)

model = LogisticRegression()
model.fit(x_train, y_train)
score = model.score(x_test, y_test)

print(score)

test = [
    "This is terrible and I am disappointed",
    "I am quite happy with this",
    "Never going to buy from this store ever again",
    "Works pretty well and functions as it should",
    "I hate this",]

x_test = vec.transform(test)
predictions = model.predict(x_test.toarray())
print(predictions)