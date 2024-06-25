import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import aif360
from aif360.metrics import ClassificationMetric
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing.reweighing import Reweighing
from common_utils import compute_metrics

df = pd.read_csv('all_loan_data.csv')
df = df.drop(
    columns=["as_of_year", "respondent_id", "agency_name", "agency_code", "loan_type", "property_type", "loan_purpose",
             "owner_occupancy", "preapproval",
             "action_taken_name", "msamd", "state_name", "state_code", "county_code", "applicant_ethnicity_name",
             "co_applicant_ethnicity",
             "applicant_race_name_1", "applicant_sex_name", "co_applicant_sex", "purchaser_type", "hoepa_status",
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
df["action_taken"] = df["action_taken"].apply(lambda x: 1 if (x == 1 or x == 2 or x == 6 or x == 8) else 0)

# White (1) Non-White (0)
df["applicant_race_1"] = df["applicant_race_1"].apply(lambda x: np.nan if (x > 5) else x)
df["applicant_race_1"] = df["applicant_race_1"].apply(lambda x: 1 if (x == 5) else 0)

# Male (1) Female (0)
df["applicant_sex"] = df["applicant_sex"].apply(lambda x: np.nan if (x > 2) else x)
df["applicant_sex"] = df["applicant_sex"].apply(lambda x: 1 if (x == 1) else 0)

# Non Latino/Hispanic Ethnicity (1) Latino/Hispanic Ethnicity (0)
df["applicant_ethnicity"] = df["applicant_ethnicity"].apply(lambda x: np.nan if (x > 2) else x)
df["applicant_ethnicity"] = df["applicant_ethnicity"].apply(lambda x: 1 if (x == 2) else 0)


df = df.dropna()

print(df["applicant_race_1"].value_counts())

# one hot encode helpful columns
categoricalFeatures = ["agency_abbr", "loan_type_name", "property_type_name", "loan_purpose_name",
                       "owner_occupancy_name", "preapproval_name", "action_taken", "state_abbr", "msamd_name",
                       "county_name", "census_tract_number", "co_applicant_ethnicity_name", "co_applicant_race_name_1",
                       "co_applicant_sex_name", "purchaser_type_name"]

for feature in categoricalFeatures:
    onehot = pd.get_dummies(df[feature], prefix=feature)
    df = df.drop(feature, axis=1)
    df = df.join(onehot)
