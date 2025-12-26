import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PowerTransformer, StandardScaler
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

cat_col_to_fill = [
    "Alley",
    "BsmtCond",
    "BsmtQual",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "PoolQC",
    "Fence",
    "MiscFeature",
]

numeric_cols = [
    "LotFrontage",
    "LotArea",
    "BsmtFinSF1",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "1stFlrSF",
    "2ndFlrSF",
    "GrLivArea",
    "GarageArea",
    "WoodDeckSF",
    "OpenPorchSF",
    "EnclosedPorch",
    "3SsnPorch",
    "ScreenPorch",
    "PoolArea",
]


def fill_categorical_nulls(df, cols_to_fix, fill_value="Missing"):
    """
    Fills specific categorical columns with a custom string.
    """
    for col in cols_to_fix:
        # We use .astype(str) first to ensure no conflicts with 'Category' dtypes
        df[col] = df[col].fillna(fill_value)
    return df


def impute_lot_frontage_train(df):
    features = [
        "LotFrontage",
        "LotArea",
        "MSSubClass",
        "MSZoning",
        "Street",
        "LotShape",
    ]
    subset = df[features].copy()

    # Convert categorical text to numbers
    subset_encoded = pd.get_dummies(
        subset, columns=["MSZoning", "Street", "LotShape"], drop_first=True
    )

    # Store the column names to align with test data later
    impute_cols = subset_encoded.columns

    lr = LinearRegression()
    imp = IterativeImputer(estimator=lr, max_iter=10, random_state=42)

    # Fit and Transform on training data
    imputed_values = imp.fit_transform(subset_encoded)

    df["LotFrontage"] = imputed_values[:, 0]

    return df, imp, impute_cols


def impute_lot_frontage_test(df, fitted_imputer, training_impute_cols):
    features = [
        "LotFrontage",
        "LotArea",
        "MSSubClass",
        "MSZoning",
        "Street",
        "LotShape",
    ]
    subset = df[features].copy()

    # One-Hot Encode test data
    subset_encoded = pd.get_dummies(
        subset, columns=["MSZoning", "Street", "LotShape"], drop_first=True
    )

    # ALIGNMENT: Ensure test columns match training columns exactly
    subset_encoded = subset_encoded.reindex(columns=training_impute_cols, fill_value=0)

    # Use .transform() only (Do NOT fit on test data)
    imputed_values = fitted_imputer.transform(subset_encoded)

    df["LotFrontage"] = imputed_values[:, 0]

    return df


def sync_masonry_data(df, masonry_medians=None):
    # 1. Scenario: Both are Null
    mask_both_null = df["MasVnrType"].isnull() & (
        (df["MasVnrArea"].isnull()) | (df["MasVnrArea"] == 0)
    )
    df.loc[mask_both_null, "MasVnrType"] = "None"
    df.loc[mask_both_null, "MasVnrArea"] = 0

    # 2. Scenario: Type is Null but Area > 0
    mask_type_missing_only = df["MasVnrType"].isnull() & (df["MasVnrArea"] > 0)
    df.loc[mask_type_missing_only, "MasVnrType"] = "Other"

    # 3. Handle Medians for MasVnrArea
    if masonry_medians is None:
        # TRAINING MODE: Calculate medians and store them
        masonry_medians = df.groupby("MasVnrType")["MasVnrArea"].median()

    # Apply the medians (from train) to the current dataframe
    # We map the MasVnrType to its corresponding median value
    for m_type, m_val in masonry_medians.items():
        mask = (df["MasVnrType"] == m_type) & (df["MasVnrArea"].isnull())
        df.loc[mask, "MasVnrArea"] = m_val

    # Final Safety Cleanups
    df.loc[df["MasVnrType"] == "None", "MasVnrArea"] = 0
    df["MasVnrArea"] = df["MasVnrArea"].fillna(0)

    return df, masonry_medians


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


def prep_label_data(y, threshold=0.75):
    """
    Input: y (Series or 1D array)
    Returns: y_transformed (array), info (dict)
    """
    # Ensure y is a DataFrame for sklearn compatibility
    y_df = pd.DataFrame(y)
    target_name = y_df.columns[0]

    info = {"transform_type": None, "scaler": StandardScaler(), "pt": None}

    skew = y_df[target_name].skew()

    # 1. Handle Skewness (Transformation)
    if skew > threshold:
        y_df[target_name] = np.log1p(y_df[target_name])
        info["transform_type"] = "log1p"
    elif skew < -threshold:
        info["pt"] = PowerTransformer(method="yeo-johnson")
        y_df[target_name] = info["pt"].fit_transform(y_df[[target_name]])
        info["transform_type"] = "yeo-johnson"

    # 2. Apply Scaling
    y_scaled = info["scaler"].fit_transform(y_df)

    return y_scaled.flatten(), info


def revert_label_data(y_pred_scaled, info):
    """
    Reverts scale first, then reverts the skewness transformation.
    """
    # Reshape for sklearn
    y_reverted = y_pred_scaled.reshape(-1, 1)

    # 1. Reverse Scaling (Z-score -> Raw Values)
    y_reverted = info["scaler"].inverse_transform(y_reverted)

    # 2. Reverse Skewness Fix
    if info["transform_type"] == "log1p":
        y_reverted = np.expm1(y_reverted)
    elif info["transform_type"] == "yeo-johnson":
        y_reverted = info["pt"].inverse_transform(y_reverted)

    return y_reverted.flatten()


def prep_test_data(df, train_params):
    # 1. ID Handling
    id_col = df["Id"].copy()
    df = df.drop(columns=["Id"])

    # 2. Fill Categorical/Masonry using Training Logic
    df = fill_categorical_nulls(df, cat_col_to_fill, fill_value="None")

    # LotFrontage Regression using fitted imputer
    df = impute_lot_frontage_test(df, train_params["imp"], train_params["impute_cols"])

    # Masonry using training medians
    df, _ = sync_masonry_data(df, masonry_medians=train_params["masonry_medians"])

    for col, mode_val in train_params["train_modes"].items():
        df[col] = df[col].fillna(mode_val)

    # 3. Simple Null Filling from Training Constants
    df["Electrical"] = df["Electrical"].fillna(train_params["electrical_mode"])
    df["GarageYrBlt"] = df["GarageYrBlt"].fillna(train_params["garage_yr_median"])

    # Fill any remaining numeric NaNs (e.g., GarageCars, TotalBsmtSF)
    df[train_params["numeric_cols"]] = df[train_params["numeric_cols"]].fillna(
        train_params["numeric_medians"]
    )

    # 4. Scaling
    X_num_scaled = pd.DataFrame(
        train_params["X_num_col_scaler"].transform(df[train_params["numeric_cols"]]),
        columns=train_params["numeric_cols"],
        index=df.index,
    )

    # 5. Encoding
    X_cat_encoded = pd.get_dummies(df[train_params["cat_cols"]], drop_first=True)

    # 6. Combine and Align
    X_final = pd.concat([X_num_scaled, X_cat_encoded], axis=1)

    # Force test columns to match training columns exactly
    X_final = X_final.reindex(columns=train_params["training_columns"], fill_value=0)

    return id_col, X_final


def prep_house_price_train_data(df):
    df = fill_categorical_nulls(
        df,
        cat_col_to_fill,
        fill_value="None",
    )

    df = df.drop(columns=["Id"])

    df, imp, impute_cols = impute_lot_frontage_train(df)
    df, masonry_medians = sync_masonry_data(df)

    df["Electrical"] = df["Electrical"].fillna(df["Electrical"].mode()[0])

    df["GarageYrBlt"] = df["GarageYrBlt"].fillna(df["GarageYrBlt"].median())

    X = df.drop(columns=["SalePrice"])
    y = df["SalePrice"]

    X_num_col, X_num_col_scaler, cat_cols = prep_data_numeric_cols(X, numeric_cols)

    X_cat_encoded = pd.get_dummies(X[cat_cols], drop_first=True)

    # Step 3: Combine them back together
    X_final = pd.concat([X_num_col, X_cat_encoded], axis=1)
    y_scaled, y_info = prep_label_data(y)

    mode_cols = ["BsmtFullBath", "BsmtHalfBath", "GarageCars"]

    # Capture the modes from training data
    # .mode().iloc[0] gets the most frequent value for each column
    train_modes = {col: df[col].mode()[0] for col in mode_cols}

    train_constants = {
        "electrical_mode": df["Electrical"].mode()[0],
        "garage_yr_median": df["GarageYrBlt"].median(),
        "numeric_medians": df[numeric_cols].median(),
        "imp": imp,
        "impute_cols": impute_cols,
        "numeric_cols": numeric_cols,
        "cat_cols": cat_cols,
        "X_num_col_scaler": X_num_col_scaler,
        "training_columns": X_final.columns.tolist(),
        "masonry_medians": masonry_medians,
        "y_info": y_info,
        "train_modes": train_modes,
        "numeric_medians": df[numeric_cols].median(),
    }

    return X_final, y_scaled, train_constants


def train(model, X_train, y_train, X_val, y_val, y_info):
    model_name = type(model).__name__

    if "XGB" in model_name:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    else:
        model.fit(X_train, y_train)

    y_val_preds_scaled = model.predict(X_val)

    y_val_final = revert_label_data(y_val_preds_scaled, y_info)

    # 2. Revert the actual y_val values to original scale for a fair comparison
    y_val_actual = revert_label_data(y_val, y_info)

    # 3. Calculate Performance Metrics
    r2 = r2_score(y_val_actual, y_val_final)
    rmse = np.sqrt(mean_squared_error(y_val_actual, y_val_final))
    mae = mean_absolute_error(y_val_actual, y_val_final)

    # Relative Absolute Error (RAE)
    rae = np.sum(np.abs(y_val_actual - y_val_final)) / np.sum(
        np.abs(y_val_actual - y_val_actual.mean())
    )

    print(f"--- {model_name} Performance ---")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE:     ${rmse:,.2f}")
    print(f"MAE:      ${mae:,.2f}")
    print(f"RAE:      {rae:.4f}\n")

    return model
