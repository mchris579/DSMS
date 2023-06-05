import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as ps


joblib_file = "models/linear_regression/lr.pkl"
df_full = ps.read_csv('data/preprocessed/train.csv', engine='python', sep=',', on_bad_lines='skip')

df_x_todummy = df_full.drop(['SalePrice','LotShape','LandSlope','ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC','KitchenQual','FireplaceQu','GarageFinish','GarageQual','GarageCond','Fence'], axis=1)
df_x_scale = df_full[['LotShape','LandSlope','ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC','KitchenQual','FireplaceQu','GarageFinish','GarageQual','GarageCond','Fence']].copy()


df_x_scale.loc[df_x_scale['LotShape'] == "Reg", "LotShape"] = "1"
df_x_scale.loc[df_x_scale['LotShape'] == "IR1", "LotShape"] = "2"
df_x_scale.loc[df_x_scale['LotShape'] == "IR2", "LotShape"] = "3"
df_x_scale.loc[df_x_scale['LotShape'] == "IR3", "LotShape"] = "4"

df_x_scale.loc[df_x_scale['LandSlope'] == "Gtl", "LandSlope"] = 1
df_x_scale.loc[df_x_scale['LandSlope'] == "Mod", "LandSlope"] = 2
df_x_scale.loc[df_x_scale['LandSlope'] == "Sev", "LandSlope"] = 3

df_x_scale.loc[df_x_scale['ExterQual'] == "Ex", "ExterQual"] = 1
df_x_scale.loc[df_x_scale['ExterQual'] == "Gd", "ExterQual"] = 2
df_x_scale.loc[df_x_scale['ExterQual'] == "TA", "ExterQual"] = 3
df_x_scale.loc[df_x_scale['ExterQual'] == "Fa", "ExterQual"] = 4
df_x_scale.loc[df_x_scale['ExterQual'] == "Po", "ExterQual"] = 5

df_x_scale.loc[df_x_scale['ExterCond'] == "Ex", "ExterCond"] = 1
df_x_scale.loc[df_x_scale['ExterCond'] == "Gd", "ExterCond"] = 2
df_x_scale.loc[df_x_scale['ExterCond'] == "TA", "ExterCond"] = 3
df_x_scale.loc[df_x_scale['ExterCond'] == "Fa", "ExterCond"] = 4
df_x_scale.loc[df_x_scale['ExterCond'] == "Po", "ExterCond"] = 5

df_x_scale.loc[df_x_scale['BsmtQual'] == "Ex", "BsmtQual"] = 1
df_x_scale.loc[df_x_scale['BsmtQual'] == "Gd", "BsmtQual"] = 2
df_x_scale.loc[df_x_scale['BsmtQual'] == "TA", "BsmtQual"] = 3
df_x_scale.loc[df_x_scale['BsmtQual'] == "Fa", "BsmtQual"] = 4
df_x_scale.loc[df_x_scale['BsmtQual'] == "Po", "BsmtQual"] = 5
df_x_scale.loc[df_x_scale['BsmtQual'] == "XX", "BsmtQual"] = 0

df_x_scale.loc[df_x_scale['BsmtCond'] == "Ex", "BsmtCond"] = 1
df_x_scale.loc[df_x_scale['BsmtCond'] == "Gd", "BsmtCond"] = 2
df_x_scale.loc[df_x_scale['BsmtCond'] == "TA", "BsmtCond"] = 3
df_x_scale.loc[df_x_scale['BsmtCond'] == "Fa", "BsmtCond"] = 4
df_x_scale.loc[df_x_scale['BsmtCond'] == "Po", "BsmtCond"] = 5
df_x_scale.loc[df_x_scale['BsmtCond'] == "XX", "BsmtCond"] = 0

df_x_scale.loc[df_x_scale['HeatingQC'] == "Ex", "HeatingQC"] = 1
df_x_scale.loc[df_x_scale['HeatingQC'] == "Gd", "HeatingQC"] = 2
df_x_scale.loc[df_x_scale['HeatingQC'] == "TA", "HeatingQC"] = 3
df_x_scale.loc[df_x_scale['HeatingQC'] == "Fa", "HeatingQC"] = 4
df_x_scale.loc[df_x_scale['HeatingQC'] == "Po", "HeatingQC"] = 5

df_x_scale.loc[df_x_scale['KitchenQual'] == "Ex", "KitchenQual"] = 1
df_x_scale.loc[df_x_scale['KitchenQual'] == "Gd", "KitchenQual"] = 2
df_x_scale.loc[df_x_scale['KitchenQual'] == "TA", "KitchenQual"] = 3
df_x_scale.loc[df_x_scale['KitchenQual'] == "Fa", "KitchenQual"] = 4
df_x_scale.loc[df_x_scale['KitchenQual'] == "Po", "KitchenQual"] = 5

df_x_scale.loc[df_x_scale['FireplaceQu'] == "Ex", "FireplaceQu"] = 1
df_x_scale.loc[df_x_scale['FireplaceQu'] == "Gd", "FireplaceQu"] = 2
df_x_scale.loc[df_x_scale['FireplaceQu'] == "TA", "FireplaceQu"] = 3
df_x_scale.loc[df_x_scale['FireplaceQu'] == "Fa", "FireplaceQu"] = 4
df_x_scale.loc[df_x_scale['FireplaceQu'] == "Po", "FireplaceQu"] = 5
df_x_scale.loc[df_x_scale['FireplaceQu'] == "XX", "FireplaceQu"] = 0

df_x_scale.loc[df_x_scale['GarageFinish'] == "Fin", "GarageFinish"] = 1
df_x_scale.loc[df_x_scale['GarageFinish'] == "RFn", "GarageFinish"] = 2
df_x_scale.loc[df_x_scale['GarageFinish'] == "Unf", "GarageFinish"] = 3
df_x_scale.loc[df_x_scale['GarageFinish'] == "XX", "GarageFinish"] = 0

df_x_scale.loc[df_x_scale['GarageQual'] == "Ex", "GarageQual"] = 1
df_x_scale.loc[df_x_scale['GarageQual'] == "Gd", "GarageQual"] = 2
df_x_scale.loc[df_x_scale['GarageQual'] == "TA", "GarageQual"] = 3
df_x_scale.loc[df_x_scale['GarageQual'] == "Fa", "GarageQual"] = 4
df_x_scale.loc[df_x_scale['GarageQual'] == "Po", "GarageQual"] = 5
df_x_scale.loc[df_x_scale['GarageQual'] == "XX", "GarageQual"] = 0

df_x_scale.loc[df_x_scale['GarageCond'] == "Ex", "GarageCond"] = 1
df_x_scale.loc[df_x_scale['GarageCond'] == "Gd", "GarageCond"] = 2
df_x_scale.loc[df_x_scale['GarageCond'] == "TA", "GarageCond"] = 3
df_x_scale.loc[df_x_scale['GarageCond'] == "Fa", "GarageCond"] = 4
df_x_scale.loc[df_x_scale['GarageCond'] == "Po", "GarageCond"] = 5
df_x_scale.loc[df_x_scale['GarageCond'] == "XX", "GarageCond"] = 0

df_x_scale.loc[df_x_scale['Fence'] == "GdPrv", "Fence"] = 1
df_x_scale.loc[df_x_scale['Fence'] == "MnPrv", "Fence"] = 2
df_x_scale.loc[df_x_scale['Fence'] == "GdWo", "Fence"] = 3
df_x_scale.loc[df_x_scale['Fence'] == "MnWw", "Fence"] = 4
df_x_scale.loc[df_x_scale['Fence'] == "XX", "Fence"] = 0

df_x_todummy = ps.get_dummies(df_x_todummy)	

df_x = ps.concat([df_x_scale, df_x_todummy], axis=1, join="inner")
df_y = df_full['SalePrice']

df_full_edited = ps.concat([df_x, df_y], axis=1, join="inner")

x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size = 0.25)

x_train.to_csv('data/split/x_train.csv', index=False)
x_test.to_csv('data/split/x_test.csv', index=False)
y_train.to_csv('data/split/y_train.csv', index=False)
y_test.to_csv('data/split/y_test.csv', index=False)
df_full_edited.to_csv('data/split/df_full_edited.csv', index=False)