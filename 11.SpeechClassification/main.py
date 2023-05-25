import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

data_dir = '11.SpeechClassification\dataset'

labels = ['music_wav','speech_wav']

features = []
target_labels = []

for label in labels:
    class_dir = f'{data_dir}/{label}'
    
    for filename in os.listdir(class_dir):
        file_path = f'{class_dir}/{filename}'
        
        audio, _ = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y = audio)
        
        mfcc_flat = mfcc.flatten()
        
        features.append(mfcc_flat)
        target_labels.append(label)

X = np.array(features)
y = np.array(target_labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifier = RandomForestClassifier(n_estimators=100)
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)