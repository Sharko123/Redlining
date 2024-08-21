# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, accuracy_score, f1_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

import tensorflow as tf

np.random.seed(1)

# %%
# df_large = pd.read_csv('2023 Data/year_2023.csv')

# sample_size = int(0.001*len(df_large))
# print(df_large.shape)
# # Create a representative sample
# df = df_large.sample(n=sample_size, random_state=42)
# print(df.shape)
# df = df.drop(
#     columns=["activity_year", "lei", "derived_msa-md", "county_code", "census_tract", "conforming_loan_limit",
#              "loan_to_value_ratio", "rate_spread", "total_points_and_fees",
#              "prepayment_penalty_term", "intro_rate_period", "multifamily_affordable_units", "applicant_ethnicity-2",
#              "applicant_ethnicity-3", "applicant_ethnicity-4", "applicant_ethnicity-5",
#              "co-applicant_ethnicity-2", "co-applicant_ethnicity-3", "co-applicant_ethnicity-4",
#              "co-applicant_ethnicity-5", "applicant_race-2", "applicant_race-3", "applicant_race-4", "applicant_race-5",
#              "co-applicant_race-2", "co-applicant_race-3", "co-applicant_race-4", "co-applicant_race-5",
#              "applicant_ethnicity_observed", "co-applicant_ethnicity_observed",
#              "applicant_age_above_62", "co-applicant_age_above_62", "applicant_race-1", "co-applicant_race-1",
#              "applicant_sex", "co-applicant_sex", "applicant_ethnicity-1", "co-applicant_ethnicity-1"
#         , "aus-1", "aus-2", "aus-3", "aus-4", "aus-5", "denial_reason-1", "denial_reason-2", "denial_reason-3",
#              "denial_reason-4", "applicant_age", "co-applicant_age", "debt_to_income_ratio", "interest_rate", "total_loan_costs", "origination_charges", "discount_points"
#              , "lender_credits"])
# df = df.dropna(axis=1, how="all")
# print(df.shape)

# #encode action_taken
# df["action_taken"] = df["action_taken"].apply(lambda x: np.nan if (x == 4 or x == 5) else x)
# df["action_taken"] = df["action_taken"].apply(
#     lambda x: 1 if (x == 1 or x == 2 or x == 6 or x == 8) else 0)

# # Male (1) Female (0)
# df["derived_sex"] = df["derived_sex"].apply(lambda x: np.nan if (x == "Sex Not Available") or (x == "Joint") else x)
# df["derived_sex"] = df["derived_sex"].apply(lambda x: 1 if (x == "Male") else 0)

# # Non Latino/Hispanic Ethnicity (1) Latino/Hispanic Ethnicity (0)
# df["derived_ethnicity"] = df["derived_ethnicity"].apply(
#     lambda x: np.nan if (x == "Ethnicity Not Available") or (x == "Free Form Text Only") or (x == "Joint") else x)
# df["derived_ethnicity"] = df["derived_ethnicity"].apply(lambda x: 1 if (x == "Not Hispanic or Latino") else 0)



# df["derived_race"] = df["derived_race"].apply(
#     lambda x: np.nan if (x == "Race Not Available") or (x == "Free Form Text Only") else x)
# df["derived_race"] = df["derived_race"].apply(
#     lambda x: "American Indian" if (x == "American Indian or Alaska Native") else x)
# df["derived_race"] = df["derived_race"].apply(
#     lambda x: "Native Hawaiian" if (x == "Native Hawaiian or Other Pacific Islander") else x)
# # Non-Black (1) African American (0)

# df["derived_race"] = df["derived_race"].apply(lambda x: 0 if (x == "Black or African American") else 1)

# df.replace("Exempt", np.nan, inplace=True)

# df = df.dropna()
# print(df.shape)

# # one hot encode helpful columns
# categoricalFeatures = ["state_code", "derived_loan_product_type", "derived_dwelling_category", "purchaser_type",
#                        "preapproval", "loan_type", "loan_purpose", "lien_status", "reverse_mortgage",
#                        "open-end_line_of_credit",
#                        "business_or_commercial_purpose", "hoepa_status", "submission_of_application",
#                        "initially_payable_to_institution", "total_units"]

# for feature in categoricalFeatures:
#     onehot = pd.get_dummies(df[feature], prefix=feature)
#     df = df.drop(feature, axis=1)
#     df = df.join(onehot)

# df.to_csv("data23.csv")

df = pd.read_csv("data23.csv")

# %%
# Labels are the values we want to predict
labels = np.array(df['action_taken']) # Y
df = df.drop('action_taken', axis = 1)
x_list = list(df.columns) # X
# Convert to numpy array
df = np.array(df)

# %%
X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size = 0.25, random_state = 42)

#####################################################################################################
###                                Logistic Regression                                            ###
#####################################################################################################
print("Logistic Regression")

# %%
# instantiate the model (using the default parameters)
logreg = LogisticRegression(random_state=16)

# fit the model with data
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

# %%
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

# %%
target_names = ['LOW SSL SCORE', 'HIGH SSL SCORE']
print(classification_report(y_test, y_pred, target_names=target_names))

# %%
print('Accuracy: %.4f' % accuracy_score(y_test, y_pred))
print('Precision: %.4f' % precision_score(y_test, y_pred))
print('Recall: %.4f' % recall_score(y_test, y_pred))
print('F1 Score: %.4f' % f1_score(y_test, y_pred))


#####################################################################################################
###                                     Random Forest                                             ###
#####################################################################################################
print("Random Forest")

rf = RandomForestClassifier(n_estimators = 1000, random_state = 42)
rf.fit(X_train, y_train);
y_pred = rf.predict(X_test)

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

target_names = ['LOW SSL SCORE', 'HIGH SSL SCORE']
print(classification_report(y_test, y_pred, target_names=target_names))

print('Accuracy: %.4f' % accuracy_score(y_test, y_pred))
print('Precision: %.4f' % precision_score(y_test, y_pred))
print('Recall: %.4f' % recall_score(y_test, y_pred))
print('F1 Score: %.4f' % f1_score(y_test, y_pred))

#####################################################################################################
###                                Support Vector Machine                                         ###
#####################################################################################################
print("Linear SVM")

clf = svm.SVC(kernel='linear', max_iter=10000) # Linear Kernel
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

target_names = ['LOW SSL SCORE', 'HIGH SSL SCORE']
print(classification_report(y_test, y_pred, target_names=target_names))

print('Accuracy: %.4f' % accuracy_score(y_test, y_pred))
print('Precision: %.4f' % precision_score(y_test, y_pred))
print('Recall: %.4f' % recall_score(y_test, y_pred))
print('F1 Score: %.4f' % f1_score(y_test, y_pred))


#####################################################################################################
###                                RBF SVM                                         ###
#####################################################################################################
print("RBF SVM")

clf = svm.SVC(kernel='rbf', max_iter=10000) # Linear Kernel
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

target_names = ['LOW SSL SCORE', 'HIGH SSL SCORE']
print(classification_report(y_test, y_pred, target_names=target_names))

print('Accuracy: %.4f' % accuracy_score(y_test, y_pred))
print('Precision: %.4f' % precision_score(y_test, y_pred))
print('Recall: %.4f' % recall_score(y_test, y_pred))
print('F1 Score: %.4f' % f1_score(y_test, y_pred))



#####################################################################################################
###                                Neural Network                                       ###
#####################################################################################################
X_train = np.array(X_train).astype(np.float32)
X_test = np.array(X_test).astype(np.float32)
y_train = np.array(y_train).astype(np.int32)
y_test = np.array(y_test).astype(np.int32)


print("Neural Network")

# Defining the model
model = Sequential()

# Adding the input layer and the first hidden layer
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.5))  # Dropout to prevent overfitting

# Adding the second hidden layer
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))

# Adding the output layer
model.add(Dense(1, activation='sigmoid'))

# Compiling the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Training the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

# Predicting on the test set
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

print('Accuracy: %.4f' % accuracy_score(y_test, y_pred))
print('Precision: %.4f' % precision_score(y_test, y_pred))
print('Recall: %.4f' % recall_score(y_test, y_pred))
print('F1 Score: %.4f' % f1_score(y_test, y_pred))