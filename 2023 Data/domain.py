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
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import geopandas as gpd
from census import Census
from us import states

c = Census("03a85c68917f321fde3ab1e9bcd26a952061fc3e")


df = pd.read_csv("original_with_DK.csv")
df1 = pd.read_csv("2023 Data/data23.csv")

df_list = []
print(df.shape)
print(df1.shape)

for i in tqdm(df['census_tract'], desc="Processing Domain Knowledge"):
    i = str(i)
    state = i[:2]
    county = i[2:5]
    census = i[5:-2]
    tract = c.acs5.state_county_tract(fields = ('NAME', 'B23025_001E', 'B23025_002E', 'B23025_007E', 'C17002_001E', 'C17002_002E', 'C17002_003E', 'B15003_001E', 'B15003_002E', 'B15003_003E', 'B15003_004E', 'B15003_005E', 'B15003_006E', 'B15003_007E', 'B15003_008E', 'B15003_009E', 'B15003_010E', 'B15003_011E', 'B15003_012E', 'B15003_013E', 'B15003_014E', 'B15003_015E', 'B15003_016E', 'B15003_017E', 'B15003_018E', 'B15003_019E', 'B15003_020E', 'B15003_021E', 'B15003_022E', 'B15003_023E', 'B15003_024E', 'B15003_025E','B01003_001E'),
                                      state_fips = state,
                                      county_fips = county,
                                      tract = census,
                                      year = 2020,
                                      survey = "acs1",geography = "zcta")
    df_list.extend(tract)

dk = pd.DataFrame(df_list)

dk["GEOID"] = dk["state"].astype(str) + dk["county"].astype(str) + dk["tract"].astype(str)
dk = dk.drop(columns = ["state", "county", "tract"])

dk["less_or_8th_grade"] = (dk['B15003_002E'] + dk['B15003_003E'] + dk['B15003_004E'] + dk[ 'B15003_005E'] + dk['B15003_006E'] + dk[ 'B15003_007E'] + dk['B15003_008E'] + dk[ 'B15003_009E'] + dk['B15003_010E'] + dk[ 'B15003_011E'] + dk['B15003_012E'])
dk["less_or_12th_grade"] = (dk[ 'B15003_013E']+ dk['B15003_014E'] + dk[ 'B15003_015E'] + dk['B15003_016E'])
dk["highschool or equivalent"] = (dk[ 'B15003_017E'] + dk['B15003_018E'])
dk["some college"] = (dk[ 'B15003_019E'] + dk['B15003_020E'])
dk["associate"] = (dk[ 'B15003_021E'])
dk["bachelor"] = (dk['B15003_022E'])
dk["master or higher"] = (dk[ 'B15003_023E'] + dk['B15003_024E'] + dk[ 'B15003_025E'])
dk["Education_Rate"] = (dk["less_or_8th_grade"] + dk["less_or_12th_grade"] + dk["highschool or equivalent"] + dk["some college"] + dk["associate"] + dk["bachelor"] + dk["master or higher"]) / dk["B01003_001E"] * 100
dk["Employment_Rate"] = (dk["B23025_002E"] + dk["B23025_007E"]) / dk["B01003_001E"] * 100
dk["Poverty_Rate"] = (dk["C17002_002E"] + dk["C17002_003E"]) / dk["B01003_001E"] * 100

dk['GEOID'] = dk['GEOID'].astype(np.float64)

main = pd.merge(df, dk, how='inner', left_on = 'census_tract', right_on = 'GEOID')
main = main.drop(['STATEFP', 'COUNTYFP', 'TRACTCE', 'GEOID', 'geometry', 'C17002_001E', 'C17002_002E', 'C17002_003E', 'B15003_001E', 'B15003_002E', 'B15003_003E', 'B15003_004E', 'B15003_005E', 'B15003_006E', 'B15003_007E', 'B15003_008E', 'B15003_009E', 'B15003_010E', 'B15003_011E', 'B15003_012E', 'B15003_013E', 'B15003_014E', 'B15003_015E', 'B15003_016E', 'B15003_017E', 'B15003_018E', 'B15003_019E', 'B15003_020E', 'B15003_021E', 'B15003_022E', 'B15003_023E', 'B15003_024E', 'B15003_025E', 'B01003_001E', 'B23025_001E', 'B23025_002E', 'B23025_007E', 'less_or_8th_grade', 'less_or_12th_grade', 'highschool or equivalent', 'some college', 'associate', 'bachelor', 'master or higher', 'CENSUS TRACT'], axis=1)
main = main.dropna()

main.to_csv('2023 Data/data23DK.csv', index=False)