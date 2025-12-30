import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PowerTransformer, StandardScaler
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np


cat_cols_none_impute = [
    "PoolQC",
    "MiscFeature",
    "Alley",
    "Fence",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "MasVnrType",
    "MSZoning",
]

cols_zero_impute = [
    "BsmtFinSF1",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "GarageCars",
    "GarageArea",
    "MasVnrArea",
    "BsmtFullBath",
    "BsmtHalfBath",
    "GarageYrBlt",
]

outliers = [
    1182,
    691,
    898,
    803,
    1046,
    1169,
    440,
    769,
    178,
    798,
    185,
    1373,
    1298,
    1243,
    1268,
]


def fill_categorical_nulls(df, cols_to_fix, fill_value="Missing"):
    """
    Fills specific categorical columns with a custom string.
    """
    for col in cols_to_fix:
        # We use .astype(str) first to ensure no conflicts with 'Category' dtypes
        df[col] = df[col].fillna(fill_value)
    return df


def fill_zero(df, cols_to_fix):
    """
    Fills specific categorical columns with a custom string.
    """
    for col in cols_to_fix:
        # We use .astype(str) first to ensure no conflicts with 'Category' dtypes
        df[col] = df[col].fillna(0)

    return df


def prep_data_numeric_cols(df, numeric_feats, threshold=0.75):
    """
    Automatically detects and fixes skewness in numerical columns.
    - Positive skew > threshold: Applied log1p
    - Negative skew < -threshold: Applied Yeo-Johnson PowerTransform
    """
    df_transformed = df.copy()

    # 2. Calculate skewness for all columns
    skewness = df_transformed[numeric_feats].skew()

    # 3. Filter columns that exceed the threshold
    skewed_cols = skewness[abs(skewness) > threshold].index

    print(f"Detected {len(skewed_cols)} skewed columns to fix.\n")

    for col in skewed_cols:
        current_skew = skewness[col]

        # Scenario A: Too High (Positive Skew) -> Log Transform
        if current_skew > threshold:
            df_transformed[col] = np.log1p(df_transformed[col])
            print(f"FIXED [{col}]: Positive Skew ({current_skew:.2f}) -> Applied Log1p")

        # Scenario B: Too Low (Negative Skew) -> Power Transform
        elif current_skew < -threshold:
            # PowerTransformer expects a 2D array, so we reshape
            pt = PowerTransformer(method="yeo-johnson")
            df_transformed[col] = pt.fit_transform(df_transformed[[col]])
            print(
                f"FIXED [{col}]: Negative Skew ({current_skew:.2f}) -> Applied Yeo-Johnson"
            )

    scaler = StandardScaler()

    df_transformed[numeric_feats] = scaler.fit_transform(df_transformed[numeric_feats])

    cat_cols = [col for col in df_transformed.columns if col not in numeric_feats]

    df_transformed = df_transformed.drop(columns=cat_cols)

    return df_transformed, scaler, cat_cols


def prep_house_price_data(train_df, test_df):
    # 1. Pre-merge processing (Train only)
    # Remove outliers from training set before merging to avoid affecting test stats
    train_df = train_df.drop(outliers).reset_index(drop=True)

    # Save the target and the length of train_df for splitting later
    y_train = np.log(train_df["SalePrice"])
    train_len = len(train_df)

    # Drop target from train so we can concat
    X_train_temp = train_df.drop(columns=["SalePrice", "Id"])
    X_test_temp = test_df.copy().drop(columns=["Id"])

    # 2. Merge DataFrames
    all_data = pd.concat([X_train_temp, X_test_temp], axis=0).reset_index(drop=True)

    # 3. Apply Transformations to the combined dataset
    all_data = fill_categorical_nulls(all_data, cat_cols_none_impute, fill_value="None")
    all_data = fill_zero(all_data, cols_zero_impute)
    all_data = all_data.drop(columns=["Id"], errors="ignore")

    # Median LotFrontage by Neighborhood across the whole set
    all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median())
    )

    all_data["Electrical"] = all_data["Electrical"].fillna(
        all_data["Electrical"].mode()[0]
    )

    # 4. Numeric Scaling
    numeric_cols = all_data.select_dtypes(include=["int64", "float64"]).columns

    # Process numeric columns (assuming this function handles scaling and returns the scaler)
    # Note: For strict ML, you'd usually fit the scaler on the train portion only.
    all_data_num, _, cat_cols = prep_data_numeric_cols(all_data, numeric_cols)

    # 5. Categorical Encoding (get_dummies handles the whole set at once)
    all_data_cat = pd.get_dummies(all_data[cat_cols], drop_first=True)

    # 6. Combine Features
    X_all_final = pd.concat([all_data_num, all_data_cat], axis=1)

    # 7. Split back into Train and Test
    X_train_final = X_all_final.iloc[:train_len, :]
    X_test_final = X_all_final.iloc[train_len:, :]

    # If test_df had a target column (like in a local validation set), split it here:
    # y_test = test_df["SalePrice"] if "SalePrice" in test_df.columns else None

    id_col = test_df["Id"].copy()

    return (X_train_final, X_test_final, y_train, id_col)


# def train(model, X_train, y_train, X_val, y_val):
#     model_name = type(model).__name__

#     if "XGB" in model_name:
#         model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
#     else:
#         model.fit(X_train, y_train)

#     y_val_preds = model.predict(X_val)

#     # 3. Calculate Performance Metrics
#     r2 = r2_score(y_val, y_val_preds)
#     rmse = np.sqrt(mean_squared_error(y_val, y_val_preds))
#     mae = mean_absolute_error(y_val, y_val_preds)

#     # Relative Absolute Error (RAE)
#     rae = np.sum(np.abs(y_val, y_val_preds)) / np.sum(
#         np.abs(y_val - y_val_preds.mean())
#     )

#     print(f"--- {model_name} Performance ---")
#     print(f"R² Score: {r2:.4f}")
#     print(f"RMSE:     ${rmse:,.2f}")
#     print(f"MAE:      ${mae:,.2f}")
#     print(f"RAE:      {rae:.4f}\n")

#     return model
