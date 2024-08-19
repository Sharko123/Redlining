
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

df_large = pd.read_csv('2017 Data/all_loan_data.csv')


sample_size = int(0.001 * len(df_large))
print(df_large.shape)
# Create a representative sample
df = df_large.sample(n=sample_size, random_state=42)
print(df.shape)
df = df.drop(
    columns=["activity_year", "lei", "derived_msa", "county_code", "census_tract", "conforming", "loan_to_value_ratio", "rate_spread", "total_points_and_fees",
             "prepayment_penalty_term", "intro_rate_period", "multifamily_affordable_units", "applicant_ethnicity-2", "applicant_ethnicity-3", "applicant_ethnicity-4", "applicant_ethnicity-5",
             "co-applicant_ethnicity-2", "co-applicant_ethnicity-3", "co-applicant_ethnicity-4", "co-applicant_ethnicity-5", "applicant_race-2", "applicant_race-3", "applicant_race-4", "applicant_race-5",
             "co-applicant_race-2", "co-applicant_race-3", "co-applicant_race-4", "co-applicant_race-5", "applicant_ethnicity_observed", "co-applicant_ethnicity_observed",
             "applicant_age_above_62", "co-applicant_age_above_62", "applicant_race_1", "co-applicant_race_1", "applicant_sex_1", "co-applicant_sex_1", "applicant_ethnicity_1", "co-applicant_ethnicity_1"
             , "aus-1", "aus-2", "aus-3", "aus-4", "aus-5", "denial_reason-1", "denial_reason-2", "denial_reason-3", "denial_reason-4"])
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
df["derived_ethnicity"] = df["derived_ethnicity"].apply(lambda x: "Not Provided" if (x == "Ethnicity Not Available") or (x=="Free Form Text Only") else x)
df["derived_sex"] = df["derived_sex"].apply(lambda x: "Not Provided" if (x == "Sex Not Available") else x)
df["derived_race"] = df["derived_race"].apply(lambda x: np.nan if  (x == "Race Not Available") or (x=="Free Form Text Only") else x)
df["derived_race"] = df["derived_race"].apply(lambda x: "American Indian" if (x == "American Indian or Alaska Native") else x)
df["derived_race"] = df["derived_race"].apply(lambda x: "Native Hawaiian" if (x == "Native Hawaiian or Other Pacific Islander") else x)

df = df.dropna()


# one hot encode helpful columns
categoricalFeatures = ["state_code", "derived_loan_product_type", "derived_dwelling_category", "purchaser_type",
                       "preapproval", "loan_type", "loan_purpose", "lien_status", "reverse_mortgage", "open-end_line_of_credit",
                       "business_or_commercial_purpose", "hoepa_status", "submission_of_application", "initially_payable_to_institution"]

for feature in categoricalFeatures:
    onehot = pd.get_dummies(df[feature], prefix=feature)
    df = df.drop(feature, axis=1)
    df = df.join(onehot)

plt.rcParams.update({'font.size': 12})

# Calculate the density of each group
density_table = df.groupby(['derived_sex', 'action_taken']).size().unstack().fillna(0)
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
density_table = df.groupby(['derived_ethnicity', 'action_taken']).size().unstack().fillna(0)
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
density_table = df.groupby(['derived_race', 'action_taken']).size().unstack().fillna(0)
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