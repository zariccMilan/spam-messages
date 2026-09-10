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
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# 2. Share data
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)

# 3. Transfer text to numbers
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 4. Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 5. Test
y_pred = model.predict(X_test_vec)
print(f"Tačnost: {accuracy_score(y_test, y_pred):.2%}")
print(classification_report(y_test, y_pred))

# 6. Example of prediction
test_sms = ["Win a free iPhone!", "Hey, are we still on for lunch?"]
test_vec = vectorizer.transform(test_sms)
predictions = model.predict(test_vec)
for sms, pred in zip(test_sms, predictions):
    print(f"'{sms}' → {'SPAM' if pred == 1 else 'HAM'}")

# 7. Confusion matrix graph
cm = confusion_matrix(y_test, y_pred)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Ham", "Spam"]
)

display.plot(cmap="Blues")
plt.title("Matrica konfuzije - klasifikacija SMS poruka")
plt.show()
