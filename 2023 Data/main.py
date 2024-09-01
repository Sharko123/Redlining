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

    for thresh in tqdm(class_thresh_arr):

        if thresh == best_class_thresh:
            disp = True
        else:
            disp = False

        fav_inds = dataset_orig_test_pred.scores > thresh
        dataset_orig_test_pred.labels[fav_inds] = dataset_orig_test_pred.favorable_label
        dataset_orig_test_pred.labels[~fav_inds] = dataset_orig_test_pred.unfavorable_label

        metric_test_bef, finbef = compute_metrics(dataset_orig_test, dataset_orig_test_pred,
                                          unprivileged_groups, privileged_groups,
                                          disp=disp)

        bal_acc_arr_orig.append(metric_test_bef["Balanced accuracy"])
        avg_odds_diff_arr_orig.append(metric_test_bef["Average odds difference"])
        disp_imp_arr_orig.append(metric_test_bef["Disparate impact"])
        if disp:
            break

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
    print("POST REWEIGHING!!!!")

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
        metric_test_aft, finaft = compute_metrics(dataset_orig_test, dataset_transf_test_pred,
                                          unprivileged_groups, privileged_groups,
                                          disp=disp)

        bal_acc_arr_transf.append(metric_test_aft["Balanced accuracy"])
        avg_odds_diff_arr_transf.append(metric_test_aft["Average odds difference"])
        disp_imp_arr_transf.append(metric_test_aft["Disparate impact"])
        if disp:
            break
    return finbef, finaft


# ----------------------------------
#            READ DATA
# ----------------------------------

df= pd.read_csv('2023 Data/data23.csv')

# sample_size = int(0.1*len(df_large))
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
# df = df.dropna()
# print(df.shape)
# print(df['action_taken'].value_counts())
# # Male (1) Female (0)
# df["derived_sex"] = df["derived_sex"].apply(lambda x: np.nan if (x == "Sex Not Available") or (x == "Joint") else x)
# df["derived_sex"] = df["derived_sex"].apply(lambda x: 1 if (x == "Male") else 0)
# df = df.dropna()
# print(df.shape)
# # Non Latino/Hispanic Ethnicity (1) Latino/Hispanic Ethnicity (0)
# df["derived_ethnicity"] = df["derived_ethnicity"].apply(
#     lambda x: np.nan if (x == "Ethnicity Not Available") or (x == "Free Form Text Only") or (x == "Joint") else x)
# df["derived_ethnicity"] = df["derived_ethnicity"].apply(lambda x: 1 if (x == "Not Hispanic or Latino") else 0)
# df = df.dropna()
# print(df.shape)
# df["derived_race"] = df["derived_race"].apply(
#     lambda x: np.nan if (x == "Race Not Available") or (x == "Free Form Text Only") else x)

# df = df.dropna()
# print(df.shape)
# df["derived_race"] = df["derived_race"].apply(
#     lambda x: "American Indian" if (x == "American Indian or Alaska Native") else x)
# df["derived_race"] = df["derived_race"].apply(
#     lambda x: "Native Hawaiian" if (x == "Native Hawaiian or Other Pacific Islander") else x)
# # Non-Black (1) African American (0)

# df["derived_race"] = df["derived_race"].apply(lambda x: 0 if (x == "Black or African American") else 1)


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

before_RW = []
after_RW = []

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
    protected_attribute_names=['derived_race'])

# Priviliged group: White (1)
# Unpriviliged group: Non-White (0)
privileged_groups = [{'derived_race': 1}]
unprivileged_groups = [{'derived_race': 0}]

bef, aft = bias_metrics_rf(privileged_groups, unprivileged_groups)
before_RW.append(list(bef.values()))
after_RW.append(list(aft.values()))
print(bef)


# ----------------------------------
#         FAIRNESS FOR SEX
# ----------------------------------
 
# print('\n\n--------------------------------\n LOGISTIC REGRESSION SEX vs SSL SCORE BIAS METRICS\n--------------------------------')

binaryLabelDataset = aif360.datasets.BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=df,
    label_names=['action_taken'],
    protected_attribute_names=['derived_sex'])

# Priviliged group: Male (1)
# Unpriviliged group: Female (0)
privileged_groups = [{'derived_sex': 1}]
unprivileged_groups = [{'derived_sex': 0}]

print(
    '\n\n--------------------------------\n RANDOM FOREST SEX vs ACTION_TAKEN BIAS METRICS\n--------------------------------')

bef, aft = bias_metrics_rf(privileged_groups, unprivileged_groups)
before_RW.append(list(bef.values()))
after_RW.append(list(aft.values()))
# ----------------------------------
#         FAIRNESS FOR ETHNICITY
# ----------------------------------

binaryLabelDataset = aif360.datasets.BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=df,
    label_names=['action_taken'],
    protected_attribute_names=['derived_ethnicity'])

# Priviliged group: Male (1)
# Unpriviliged group: Female (0)
privileged_groups = [{'derived_ethnicity': 1}]
unprivileged_groups = [{'derived_ethnicity': 0}]

print(
    '\n\n--------------------------------\n RANDOM FOREST ETHNICITY vs ACTION_TAKEN SCORE BIAS METRICS\n--------------------------------')

bef, aft = bias_metrics_rf(privileged_groups, unprivileged_groups)
before_RW.append(list(bef.values()))
after_RW.append(list(aft.values()))

before_RW = np.array(before_RW)
after_RW = np.array(after_RW)
print(before_RW)

import matplotlib.pyplot as plt

# Example data: statistical parity difference and disparate impact values before and after reweighing
plt.rcParams.update({'font.size': 14})

groups = ['Race', 'Sex', 'Ethnicity']
metrics = ['Statistical Parity Difference', 'Disparate Impact', 'Theil Index']
# Set width of bars
bar_width = 0.35

# Set position of bars on X axis
r1 = np.arange(len(groups))
r2 = [x + bar_width for x in r1]

# Plotting first metric: Statistical Parity Difference
fig, ax = plt.subplots(figsize=(10, 6))

# Before reweighing bars
bars1 = ax.bar(r1, before_RW[:, 0], color='b', width=bar_width, edgecolor='grey', label='Before Reweighing')
bars2 = ax.bar(r2, after_RW[:, 0], color='r', width=bar_width, edgecolor='grey', hatch='//', label='After Reweighing')

# Adding labels, title, and custom x-axis tick labels
ax.set_xlabel('Groups')
ax.set_ylabel('Statistical Parity Difference')
ax.set_title('Statistical Parity Difference Before and After Reweighing')
ax.set_xticks([r + bar_width / 2 for r in range(len(groups))])
ax.set_xticklabels(groups)
ax.legend()

# Adjust layout
plt.tight_layout()

# Show plot for Statistical Parity Difference
plt.show()

# Plotting second metric: Disparate Impact
fig, ax = plt.subplots(figsize=(10, 6))

# Before reweighing bars
bars1 = ax.bar(r1, before_RW[:, 1], color='b', width=bar_width, edgecolor='grey', label='Before Reweighing')
bars2 = ax.bar(r2, after_RW[:, 1], color='r', width=bar_width, edgecolor='grey', hatch='//', label='After Reweighing')

# Adding labels, title, and custom x-axis tick labels
ax.set_xlabel('Groups')
ax.set_ylabel('Disparate Impact')
ax.set_title('Disparate Impact Before and After Reweighing')
ax.set_xticks([r + bar_width / 2 for r in range(len(groups))])
ax.set_xticklabels(groups)
ax.legend()

# Adjust layout
plt.tight_layout()

# Show plot for Disparate Impact
plt.show()


# Plotting third metric: Theil Index
fig, ax = plt.subplots(figsize=(10, 6))

# Before reweighing bars
bars1 = ax.bar(r1, before_RW[:, 2], color='b', width=bar_width, edgecolor='grey', label='Before Reweighing')
bars2 = ax.bar(r2, after_RW[:, 2], color='r', width=bar_width, edgecolor='grey', hatch='//', label='After Reweighing')

# Adding labels, title, and custom x-axis tick labels
ax.set_xlabel('Groups')
ax.set_ylabel('Theil Index')
ax.set_title('Theil Index Before and After Reweighing')
ax.set_xticks([r + bar_width / 2 for r in range(len(groups))])
ax.set_xticklabels(groups)
ax.legend()

# Adjust layout
plt.tight_layout()

# Show plot for Theil index
plt.show()


print(before_RW)
print(after_RW)