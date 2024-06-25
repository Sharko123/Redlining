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

np.random.seed(1)


def bias_metrics_rf(privileged_groups, unprivileged_groups):
    # train test split
    dataset_orig_train, dataset_orig_vt = binaryLabelDataset.split([0.7], shuffle=True)
    dataset_orig_valid, dataset_orig_test = dataset_orig_vt.split([0.5], shuffle=True)

    # Random Forest classifier and predictions
    scale_orig = StandardScaler()
    X_train = scale_orig.fit_transform(dataset_orig_train.features)
    y_train = dataset_orig_train.labels.ravel()
    w_train = dataset_orig_train.instance_weights.ravel()

    rf = RandomForestClassifier(n_estimators=1000, random_state=42)
    rf.fit(X_train, y_train)
    y_train_pred = rf.predict(X_train)

    # positive class index
    pos_ind = np.where(rf.classes_ == dataset_orig_train.favorable_label)[0][0]

    dataset_orig_train_pred = dataset_orig_train.copy()
    dataset_orig_train_pred.labels = y_train_pred

    dataset_orig_valid_pred = dataset_orig_valid.copy(deepcopy=True)
    X_valid = scale_orig.transform(dataset_orig_valid_pred.features)
    y_valid = dataset_orig_valid_pred.labels
    dataset_orig_valid_pred.scores = rf.predict_proba(X_valid)[:, pos_ind].reshape(-1, 1)

    dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
    X_test = scale_orig.transform(dataset_orig_test_pred.features)
    y_test = dataset_orig_test_pred.labels
    dataset_orig_test_pred.scores = rf.predict_proba(X_test)[:, pos_ind].reshape(-1, 1)

    num_thresh = 100
    ba_arr = np.zeros(num_thresh)
    class_thresh_arr = np.linspace(0.01, 0.99, num_thresh)
    for idx, class_thresh in enumerate(class_thresh_arr):
        fav_inds = dataset_orig_valid_pred.scores > class_thresh
        dataset_orig_valid_pred.labels[fav_inds] = dataset_orig_valid_pred.favorable_label
        dataset_orig_valid_pred.labels[~fav_inds] = dataset_orig_valid_pred.unfavorable_label

        classified_metric_orig_valid = ClassificationMetric(dataset_orig_valid,
                                                            dataset_orig_valid_pred,
                                                            unprivileged_groups=unprivileged_groups,
                                                            privileged_groups=privileged_groups)

        ba_arr[idx] = 0.5 * (classified_metric_orig_valid.true_positive_rate()
                             + classified_metric_orig_valid.true_negative_rate())

    best_ind = np.where(ba_arr == np.max(ba_arr))[0][0]
    best_class_thresh = class_thresh_arr[best_ind]

    print("\nBest balanced accuracy (no reweighing) = %.4f" % np.max(ba_arr))
    print("Optimal classification threshold (no reweighing) = %.4f" % best_class_thresh)

    bal_acc_arr_orig = []
    disp_imp_arr_orig = []
    avg_odds_diff_arr_orig = []

    print("\nClassification threshold used = %.4f" % best_class_thresh)
    for thresh in tqdm(class_thresh_arr):

        if thresh == best_class_thresh:
            disp = True
        else:
            disp = False

        fav_inds = dataset_orig_test_pred.scores > thresh
        dataset_orig_test_pred.labels[fav_inds] = dataset_orig_test_pred.favorable_label
        dataset_orig_test_pred.labels[~fav_inds] = dataset_orig_test_pred.unfavorable_label

        metric_test_bef = compute_metrics(dataset_orig_test, dataset_orig_test_pred,
                                          unprivileged_groups, privileged_groups,
                                          disp=disp)

        bal_acc_arr_orig.append(metric_test_bef["Balanced accuracy"])
        avg_odds_diff_arr_orig.append(metric_test_bef["Average odds difference"])
        disp_imp_arr_orig.append(metric_test_bef["Disparate impact"])

    ###
    # WITH REWEIGHING
    ###

    # Metric for the original dataset
    metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train,
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)
    print(
        "Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_train.mean_difference())

    RW = Reweighing(unprivileged_groups=unprivileged_groups,
                    privileged_groups=privileged_groups)
    RW.fit(dataset_orig_train)
    dataset_transf_train = RW.transform(dataset_orig_train)

    metric_transf_train = BinaryLabelDatasetMetric(dataset_transf_train,
                                                   unprivileged_groups=unprivileged_groups,
                                                   privileged_groups=privileged_groups)
    print(
        "Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_transf_train.mean_difference())

    scale_transf = StandardScaler()
    X_train = scale_transf.fit_transform(dataset_transf_train.features)
    y_train = dataset_transf_train.labels.ravel()

    # logistic regression
    lmod = LogisticRegression()
    lmod.fit(X_train, y_train,
             sample_weight=dataset_transf_train.instance_weights)
    y_train_pred = lmod.predict(X_train)

    # Random Forest classifier and predictions
    rf = RandomForestClassifier(n_estimators=1000, random_state=42)
    rf.fit(X_train, y_train);
    y_train_pred = rf.predict(X_train)

    dataset_transf_test_pred = dataset_orig_test.copy(deepcopy=True)
    X_test = scale_transf.fit_transform(dataset_transf_test_pred.features)
    y_test = dataset_transf_test_pred.labels
    dataset_transf_test_pred.scores = lmod.predict_proba(X_test)[:, pos_ind].reshape(-1, 1)

    bal_acc_arr_transf = []
    disp_imp_arr_transf = []
    avg_odds_diff_arr_transf = []

    print("Classification threshold used = %.4f" % best_class_thresh)
    for thresh in tqdm(class_thresh_arr):

        if thresh == best_class_thresh:
            disp = True
        else:
            disp = False

        fav_inds = dataset_transf_test_pred.scores > thresh

        dataset_transf_test_pred.labels[fav_inds] = dataset_transf_test_pred.favorable_label
        dataset_transf_test_pred.labels[~fav_inds] = dataset_transf_test_pred.unfavorable_label
        metric_test_aft = compute_metrics(dataset_orig_test, dataset_transf_test_pred,
                                          unprivileged_groups, privileged_groups,
                                          disp=disp)

        bal_acc_arr_transf.append(metric_test_aft["Balanced accuracy"])
        avg_odds_diff_arr_transf.append(metric_test_aft["Average odds difference"])
        disp_imp_arr_transf.append(metric_test_aft["Disparate impact"])


# ----------------------------------
#            READ DATA
# ----------------------------------

# read data
df_large = pd.read_csv('all_loan_data.csv')


sample_size = int(0.001 * len(df_large))
print(df_large.shape)
# Create a representative sample
df = df_large.sample(n=sample_size, random_state=42)
print(df.shape)
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

# White (1) African American (0)
df["applicant_race_1"] = df["applicant_race_1"].apply(lambda x: np.nan if (x > 5) else x)
df["applicant_race_1"] = df["applicant_race_1"].apply(lambda x: 1 if (x == 5) else (0 if (x==3) else np.nan))

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
                       "owner_occupancy_name", "preapproval_name", "state_abbr", "msamd_name",
                       "county_name", "census_tract_number", "co_applicant_ethnicity_name", "co_applicant_race_name_1",
                       "co_applicant_sex_name", "purchaser_type_name"]

for feature in categoricalFeatures:
    onehot = pd.get_dummies(df[feature], prefix=feature)
    df = df.drop(feature, axis=1)
    df = df.join(onehot)




# ----------------------------------
#        FAIRNESS FOR RACE
# ----------------------------------

print(
    '\n\n--------------------------------\n RANDOM FOREST RACE vs ACTION TAKEN BIAS METRICS\n--------------------------------')

binaryLabelDataset = aif360.datasets.BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=df,
    label_names=['action_taken'],
    protected_attribute_names=['applicant_race_1'])

# Priviliged group: White (1)
# Unpriviliged group: Non-White (0)
privileged_groups = [{'applicant_race_1': 1}]
unprivileged_groups = [{'applicant_race_1': 0}]

bias_metrics_rf(privileged_groups, unprivileged_groups)

# ----------------------------------
#         FAIRNESS FOR SEX
# ----------------------------------

# print('\n\n--------------------------------\n LOGISTIC REGRESSION SEX vs SSL SCORE BIAS METRICS\n--------------------------------')

binaryLabelDataset = aif360.datasets.BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=df,
    label_names=['action_taken'],
    protected_attribute_names=['applicant_sex'])

# Priviliged group: Male (1)
# Unpriviliged group: Female (0)
privileged_groups = [{'applicant_sex': 1}]
unprivileged_groups = [{'applicant_sex': 0}]


print(
    '\n\n--------------------------------\n RANDOM FOREST SEX vs ACTION_TAKEN BIAS METRICS\n--------------------------------')

bias_metrics_rf(privileged_groups, unprivileged_groups)

# ----------------------------------
#         FAIRNESS FOR ETHNICITY
# ----------------------------------

binaryLabelDataset = aif360.datasets.BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=df,
    label_names=['action_taken'],
    protected_attribute_names=['applicant_ethnicity'])

# Priviliged group: Male (1)
# Unpriviliged group: Female (0)
privileged_groups = [{'applicant_ethnicity': 1}]
unprivileged_groups = [{'applicant_ethnicity': 0}]


print(
    '\n\n--------------------------------\n RANDOM FOREST ETHNICITY vs ACTION_TAKEN SCORE BIAS METRICS\n--------------------------------')

bias_metrics_rf(privileged_groups, unprivileged_groups)
