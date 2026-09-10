# import for csv files
import pandas as pd

# splits data for learning and testing
from sklearn.model_selection import train_test_split

# converts sms text to numbers
from sklearn.feature_extraction.text import TfidfVectorizer

# model Multinomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB

# accuracy_score counts for whole accuracy, classification_report counts precision, recall and F1 score
from sklearn.metrics import accuracy_score, classification_report

# imports for graph
import matplotlib.pyplot as plt

# confusion_matrix makes table with right and wrong predictions, ConfusionMatrixDisplay shows that table like graph
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# 1. Load dataset
df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 'ham', 'spam': 'spam'})

# 2. Share data
# x is for entry data and y is for right answers 0 or 1
# train is what model use for studying and test in this step model does not use
# 20 % is for test and 80% for training
# 42 is just whole number and it's for repeated results to see same numbers, and model will always start from 42.
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)

# 3. Transfer text to numbers
# TF = Term Frequency, mesures how many times one word is repeated in one message
# IDF = Inverse Document Frequency, lowers rating of words that are common in all messages
# Word that repeat more in one message and not in other messages are higher rated
# Words like 'win' or 'iphone' repeats many times and that helps for spam messages but also common english words, that's why stop_words='english' is added
vectorizer = TfidfVectorizer(stop_words='english')
# fit is telling TF-IDF to learn words and rates but only from training
# transform is converting from text messages to numeric vectors
X_train_vec = vectorizer.fit_transform(X_train)
# test data converts with already learned dictionary without fit because model would than access to test data and evaluation would be compromised
X_test_vec = vectorizer.transform(X_test)

# 4. Train model
# Multinomial Naive Bayes model
# this model is for classification based on statistics
# how does it work? checks what are common words in spam and ham messages and what class in more dominant than it chooses the class with more estimated probability
# why naive ? because it simplified estimates that words don't have any connections or relationship, this is not true but model still works good
model = MultinomialNB()
# sms messages that are converted in numbers
# y_train are their real values; 'ham' or 'spam'
# this is learning time
model.fit(X_train_vec, y_train)

# 5. Test
# prediction, this is where model gets messages that weren't in training and gives for everyone prediction
# y_pred are predictions and y_test are real values from dataset
y_pred = model.predict(X_test_vec)
# accuracy is calculated from all right predictions divided by all predictions
print(f"Tačnost: {accuracy_score(y_test, y_pred):.2%}")
print(classification_report(y_test, y_pred))

# 6. Example of prediction
# this is for test two new messages
test_sms = ["Win a free iPhone!", "Hey, are we still on for lunch?"]
# again convert to numeric vectors
test_vec = vectorizer.transform(test_sms)
# now model for every message gives estimation
predictions = model.predict(test_vec)
# just print results with prediction, 1 or 0
for sms, pred in zip(test_sms, predictions):
    print(f"'{sms}' → {'SPAM' if pred == 1 else 'HAM'}")

# 7. Confusion matrix graph
# makes new graph with cm var
cm = confusion_matrix(y_test, y_pred)

# use spam instead of 1 and ham instead of 0
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Ham", "Spam"]
)

# this 'prints' graph
display.plot(cmap="Blues")
plt.title("Matrica konfuzije - klasifikacija SMS poruka")
plt.show()


# precision is how really is accurate model
# recall is from all spam messages how many model did find
# f1-score (precision and recall)
# support is how many real messages from that class does exist (spam or ham)

# in dataset real ham messages are 965 and spam are 113 + 37 = 150


