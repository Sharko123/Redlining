
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

df_large = pd.read_csv('all_loan_data.csv')


sample_size = int(0.001 * len(df_large))
print(df_large.shape)
# Create a representative sample
df = df_large.sample(n=sample_size, random_state=42)
print(df.shape)
df = df.drop(
    columns=["as_of_year", "respondent_id", "agency_name", "agency_code", "loan_type", "property_type", "loan_purpose",
             "owner_occupancy", "preapproval",
             "action_taken_name", "msamd", "state_name", "state_code", "county_code", "applicant_ethnicity",
             "co_applicant_ethnicity",
             "applicant_race_1", "applicant_sex_name", "co_applicant_sex", "purchaser_type", "hoepa_status",
             "hoepa_status_name", "lien_status", "lien_status_name",
             "rate_spread", 'applicant_race_name_2', 'applicant_race_2', 'applicant_race_name_3',
             'applicant_race_3', 'applicant_race_name_4', 'applicant_race_4',
             'applicant_race_name_5', 'applicant_race_5', 'co_applicant_race_name_2',
             'co_applicant_race_2', 'co_applicant_race_name_3',
             'co_applicant_race_3', 'co_applicant_race_name_4',
             'co_applicant_race_4', 'co_applicant_race_name_5',
             'co_applicant_race_5', 'denial_reason_name_1', 'denial_reason_1', 'denial_reason_name_2',
             'denial_reason_2', 'denial_reason_name_3', 'denial_reason_3'])
df = df.dropna(axis=1, how="all")
#encode action_taken
df["action_taken"] = df["action_taken"].apply(lambda x: np.nan if (x == 4 or x == 5) else x)
df["action_taken"] = df["action_taken"].apply(lambda x: "Accepted" if (x == 1 or x == 2 or x == 6 or x == 8) else "Rejected")

# White (1) African American (0)

# Male (1) Female (0)
df["applicant_sex"] = df["applicant_sex"].apply(lambda x: np.nan if (x > 2) else x)
df["applicant_sex"] = df["applicant_sex"].apply(lambda x: "Male" if (x == 1) else "Female")

# Non Latino/Hispanic Ethnicity (1) Latino/Hispanic Ethnicity (0)
# df["applicant_ethnicity"] = df["applicant_ethnicity"].apply(lambda x: np.nan if (x > 2) else x)
df["applicant_ethnicity_name"] = df["applicant_ethnicity_name"].apply(lambda x: "Not Provided" if (x == "Information not provided by applicant in mail, Internet, or telephone application") else x)
df["applicant_sex"] = df["applicant_sex"].apply(lambda x: "Not Provided" if (x == "Information not provided by applicant in mail, Internet, or telephone application") else x)
df["applicant_race_name_1"] = df["applicant_race_name_1"].apply(lambda x: np.nan if (x == "Information not provided by applicant in mail, Internet, or telephone application" or x=="Not applicable") else x)
df["applicant_race_name_1"] = df["applicant_race_name_1"].apply(lambda x: "American Indian" if (x == "American Indian or Alaska Native") else x)
df["applicant_race_name_1"] = df["applicant_race_name_1"].apply(lambda x: "Native Hawaiian" if (x == "Native Hawaiian or Other Pacific Islander") else x)

df = df.dropna()


# one hot encode helpful columns
categoricalFeatures = ["agency_abbr", "loan_type_name", "property_type_name", "loan_purpose_name",
                       "owner_occupancy_name", "preapproval_name", "state_abbr", "msamd_name",
                       "county_name", "census_tract_number", "co_applicant_ethnicity_name", "co_applicant_race_name_1",
                       "co_applicant_sex_name", "purchaser_type_name"]

for feature in categoricalFeatures:
    onehot = pd.get_dummies(df[feature], prefix=feature)
    df = df.drop(feature, axis=1)
    df = df.join(onehot)

plt.rcParams.update({'font.size': 12})

# Calculate the density of each group
density_table = df.groupby(['applicant_sex', 'action_taken']).size().unstack().fillna(0)
density_table = density_table.div(density_table.sum(axis=1), axis=0)

# Plotting
ax = density_table.plot(kind='bar', stacked=False, color=['blue', 'orange'])
plt.title('Action Taken by Sex')
plt.xlabel('Sex')
plt.ylabel('Density')

# Adding annotations
for p in ax.patches:
    ax.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points', bbox=dict(facecolor='white', alpha=0.8))

plt.legend(title='Action Taken')
plt.xticks(rotation=0, ha='right')
plt.tight_layout()


# Calculate the density of each group
density_table = df.groupby(['applicant_ethnicity_name', 'action_taken']).size().unstack().fillna(0)
density_table = density_table.div(density_table.sum(axis=1), axis=0)

# Plotting
ax = density_table.plot(kind='bar', stacked=False, color=['blue', 'orange'], figsize=(10, 6))
plt.title('Action Taken by Ethnicity')
plt.xlabel('Ethnicity')
plt.ylabel('Density')

# Adding annotations
for p in ax.patches:
    ax.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points'  , bbox=dict(facecolor='white', alpha=0.8))

plt.legend(title='Action Taken')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Calculate the density of each group
density_table = df.groupby(['applicant_race_name_1', 'action_taken']).size().unstack().fillna(0)
density_table = density_table.div(density_table.sum(axis=1), axis=0)

# Plotting
ax = density_table.plot(kind='bar', stacked=False, color=['blue', 'orange'], figsize=(10, 6))
plt.title('Action Taken by Race')
plt.xlabel('Race')
plt.ylabel('Density')

# Adding annotations
for p in ax.patches:
    ax.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points' , bbox=dict(facecolor='white', alpha=0.8))

plt.legend(title='Action Taken')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

plt.show()