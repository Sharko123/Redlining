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

import tensorflow as tf

np.random.seed(1)

# # %%
# # read data
# df_large = pd.read_csv('2017 Data/all_loan_data.csv')



# sample_size = int(0.001 * len(df_large))
# print(df_large.shape)
# # Create a representative sample
# df = df_large.sample(n=sample_size, random_state=42)
# print(df.shape)
# df = df.drop(
#     columns=["as_of_year", "respondent_id", "agency_name", "agency_code", "loan_type", "property_type", "loan_purpose",
#              "owner_occupancy", "preapproval",
#              "action_taken_name", "msamd", "state_name", "state_code", "county_code", "applicant_ethnicity_name",
#              "co_applicant_ethnicity",
#              "applicant_race_name_1", "applicant_sex_name", "co_applicant_sex", "purchaser_type", "hoepa_status",
#              "hoepa_status_name", "lien_status", "lien_status_name",
#              "rate_spread", 'applicant_race_name_2', 'applicant_race_2', 'applicant_race_name_3',
#              'applicant_race_3', 'applicant_race_name_4', 'applicant_race_4',
#              'applicant_race_name_5', 'applicant_race_5', 'co_applicant_race_name_2',
#              'co_applicant_race_2', 'co_applicant_race_name_3',
#              'co_applicant_race_3', 'co_applicant_race_name_4',
#              'co_applicant_race_4', 'co_applicant_race_name_5',
#              'co_applicant_race_5', 'denial_reason_name_1', 'denial_reason_1', 'denial_reason_name_2',
#              'denial_reason_2', 'denial_reason_name_3', "preapproval_name", 'denial_reason_3', "owner_occupancy_name", "msamd_name", "county_name", "co_applicant_ethnicity_name", "co_applicant_race_name_1"])
# df = df.dropna(axis=1, how="all")
# #encode action_taken
# df["action_taken"] = df["action_taken"].apply(lambda x: np.nan if (x == 4 or x == 5) else x)
# df["action_taken"] = df["action_taken"].apply(lambda x: 1 if (x == 1 or x == 2 or x == 6 or x == 8) else 0)

# # White (1) African American (0)
# df["applicant_race_1"] = df["applicant_race_1"].apply(lambda x: np.nan if (x > 5) else x)
# df["applicant_race_1"] = df["applicant_race_1"].apply(lambda x: 1 if (x == 5) else (0 if (x==3) else np.nan))

# # Male (1) Female (0)
# df["applicant_sex"] = df["applicant_sex"].apply(lambda x: np.nan if (x > 2) else x)
# df["applicant_sex"] = df["applicant_sex"].apply(lambda x: 1 if (x == 1) else 0)

# # Non Latino/Hispanic Ethnicity (1) Latino/Hispanic Ethnicity (0)
# df["applicant_ethnicity"] = df["applicant_ethnicity"].apply(lambda x: np.nan if (x > 2) else x)
# df["applicant_ethnicity"] = df["applicant_ethnicity"].apply(lambda x: 1 if (x == 2) else 0)


# df = df.dropna()

# print(df.head())

# # one hot encode helpful columns
# categoricalFeatures = ["agency_abbr", "loan_type_name", "property_type_name", "loan_purpose_name", "state_abbr",
#                        "co_applicant_sex_name", "purchaser_type_name"]

# for feature in categoricalFeatures:
#     onehot = pd.get_dummies(df[feature], prefix=feature)
#     df = df.drop(feature, axis=1)
#     df = df.join(onehot)

# df.to_csv("data17.csv")
df = pd.read_csv("data17.csv")


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
