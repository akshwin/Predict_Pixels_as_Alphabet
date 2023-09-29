# Support Vector Machine for Letter Recognition
# Introduction
Support Vector Machines (SVM) are powerful supervised learning algorithms used for classification and regression tasks. In this project, we employ an SVM to recognize black-and-white rectangular pixel displays representing the 26 capital letters of the English alphabet. The dataset comprises 20,000 stimuli derived from 20 different fonts, with each letter distorted to create unique images. The stimuli are transformed into 16 numerical attributes, including statistical moments and edge counts, which are then scaled to integer values between 0 and 15. The SVM is trained on 16,000 items, and the resulting model is applied to predict the letter category for the remaining 4,000 stimuli.

# Summary of Steps
Importing Libraries: We begin by importing essential libraries, including NumPy, pandas, and Matplotlib, to handle data and visualize results.

Data Loading and Exploration: The dataset, named "letterdata.csv," is loaded into a pandas DataFrame (df). The first 10 rows of the dataset are displayed for initial exploration.

Data Preprocessing: The dataset is split into features (x) and labels (y). The features consist of the numerical attributes, and the labels represent the corresponding letters.

Data Splitting: The data is divided into training and testing sets. The first 16,000 items are used for training (x_train and y_train), while the remaining 4,000 items are reserved for testing (x_test and y_test).

Building SVM Model: An SVM model is constructed using the Support Vector Classification (SVC) from the scikit-learn library. The model is trained on the training data with a specified penalty parameter (C).

Model Evaluation: The trained model is used to predict letter categories for the test set (y_predict). The accuracy of the model is evaluated using the score method.

Confusion Matrix and Visualization: A confusion matrix is generated to assess the model's performance. The matrix is visualized using a heatmap with annotations to highlight correct and incorrect predictions.

# Source
The details of the letter recognition dataset and the SVM model implementation can be found in the article titled "Letter Recognition Using Holland-style Adaptive Classifiers" by P. W. Frey and D. J. Slate (Machine Learning Vol. 6 No.2, March 91).

# Conclusion
The SVM model demonstrates a commendable accuracy of approximately 92.7% in predicting the letter categories for the given dataset. The confusion matrix and heatmap provide insights into the model's performance for each letter. This project serves as a practical example of applying SVM for character recognition, showcasing its effectiveness in handling complex classification tasks. Further fine-tuning and optimization can be explored to enhance the model's accuracy and robustness.
