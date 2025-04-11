import pandas as pd
import numpy as np
import statsmodels as sm
from sklearn.preprocessing import LabelEncoder

df_avn = pd.read_excel("Avian data.xlsx")

# print(df_avn.dtypes)
# print(df_avn.columns, "\n")
# print(df_avn.info)

### Label encoding of Region ###
le = LabelEncoder()
df_avn['Region_Code'] = le.fit_transform(df_avn['Region'])
# Drop original Region column
df_avn.drop('Region', axis=1, inplace=True)

# Check mapping of region codes
region_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Region to Code Mapping:")
for region, code in region_mapping.items():
    print(f"{region}: {code}")

print("\n---------------\n")

### One-Hot Encoding of Age variable ###

df_avn_agehc = pd.get_dummies(df_avn, columns=["Age"], drop_first=True, dtype=int)
# print(df_avn_agehc)

### Push Death column to last ###
# df_avn_agehc = df_avn_agehc[[col for col in df_avn_agehc.columns if col != 'Death'] + ['Death']]
# print(df_avn_agehc)


### Merge year and month to create continuous variable ###
df_avn_age_ym = df_avn_agehc
# print(df_avn_age_ym)

df_avn_age_ym["Year_mon"] = df_avn_age_ym["Year"] + (df_avn_age_ym["Month"] - 1) / 12
df_avn_age_yms = df_avn_age_ym.drop(["Year", "Month"], axis=1, inplace=True)
df_avn_age_ym = df_avn_age_ym[[col for col in df_avn_age_ym.columns if col != "Death"] + ["Death"]]

print(df_avn_age_ym)

