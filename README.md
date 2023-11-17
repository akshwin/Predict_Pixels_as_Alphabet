# PREDICT PIXELS AS ALPHABET

**Introduction:**

In this project, we implemented a Support Vector Machine (SVM) to classify black-and-white rectangular pixel displays into one of the 26 capital letters in the English alphabet. The dataset used for this task consists of 20,000 unique stimuli, each representing a distorted version of the 26 letters based on 20 different fonts. Each stimulus is characterized by 16 primitive numerical attributes, such as statistical moments and edge counts, which were scaled to integer values ranging from 0 to 15.

The primary objective is to train the SVM on the first 16,000 items and then utilize the trained model to predict the letter category for the remaining 4,000 items. The SVM algorithm, provided by the scikit-learn library, is employed for this classification task.

**Steps:**

1. **Import Libraries:**
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm, metrics
import seaborn as sns
%matplotlib inline
```

2. **Load and Explore Data:**
```python
df = pd.read_csv("letterdata.csv")
df.head(10)
```

3. **Data Splitting:**
```python
x = np.array(df)[:, 1:16]
y = np.array(df.letter)[:]

x_train = x[:16000, :]
x_test = x[16001:, :]
y_train = y[:16000] 
y_test = y[16001:]
```

4. **Build SVM Model:**
```python
model = svm.SVC(C=3)  # C is the penalty for wrong classification
model.fit(x_train, y_train)
```

5. **Predictions and Evaluation:**
```python
y_predict = model.predict(x_test)
accuracy = model.score(x_test, y_test)
print(f"Model Accuracy: {accuracy}")
```

6. **Confusion Matrix:**
```python
cm = metrics.confusion_matrix(y_test, y_predict)
df_cm = pd.DataFrame(cm, index=[i for i in lab], columns=[i for i in plab])

plt.figure(figsize=(20, 13))
sns.heatmap(df_cm, annot=True, fmt='g', cmap='PiYG')
plt.show()
```

**Source:**

The dataset and details about the experiment can be found in the article "Letter Recognition Using Holland-style Adaptive Classifiers" by P. W. Frey and D. J. Slate, published in Machine Learning Vol. 6, No. 2, March 1991.

**Conclusion:**

In conclusion, the SVM model achieved an accuracy of approximately 92.7% on the test set, as indicated by the confusion matrix. This demonstrates the effectiveness of SVM in classifying distorted alphabet characters based on the given numerical attributes. The implementation involved loading the data, splitting it into training and testing sets, building the SVM model, making predictions, and evaluating the model's performance.

