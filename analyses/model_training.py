# %%
from pathlib import Path

import dalex as dx
import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy
from dask_ml.preprocessing import Categorizer
from glum import GeneralizedLinearRegressor, GeneralizedLinearRegressorCV
from lightgbm import LGBMRegressor

# %%
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from core.evaluation.evaluate_predictions import evaluate_predictions
from core.visualisation.plot_utils import plot_test_set_predictions
from data.sample_split import create_sample_split

# %% [markdown]
# ## GLM model
# In this section, I will be predicting the daily total national
# electricity consumptions for UK. Specifically, Ideally,
# I want a model that can capture the many factors that influence consumptions
# – daily min and max temperature, wind speed, rain, humidity seasonality, etc.
#  As a baseline, I will start with a baseline model that only
# uses a few categorical feature. Then, I will fit a model by introducing
# the daily weather features. For both models, I will use GLM regressor.
# I will use a gamma distribution for my model. The target variable, tsd,
# is a positive real number, which matches the support of the gamma distribution.
# Second,from the gamma fit I see that statistically tsd has similar shapes to
# gamma distribution except that tsd exhibits seasonality
# which has more than 1 peak and valley.


df_model = pd.read_parquet(
    Path.cwd().parent / "data" / "model_data" / "model_data.parquet"
)

# Fit daily electricity consumption to gamma distribution
df_model["tsd"].plot.hist(bins=400, density=True, label="Observed")
x = np.linspace(0, 2500000, num=400)
plt.plot(
    x,
    scipy.stats.gamma.pdf(x, *scipy.stats.gamma.fit(df_model["tsd"], floc=0)),
    "g-",
    label="fitted gamma distribution",
)
plt.legend()
plt.title("Daily electricity consumption distribution")
plt.xlim(left=0, right=2.5e6)

# %%
# GLM baseline model
df = create_sample_split(df_model, threshold_date="2021-01-01")
train = np.where(df["sample"] == "train")
test = np.where(df["sample"] == "test")
df_train = df.iloc[train].copy()
df_test = df.iloc[test].copy()
y = df["tsd"]

categorical = [
    "is_holiday",
    "day_of_week",
    "week_of_year",
    "month",
    "quarter",
    "is_weekday",
    "is_weekend",
    "is_summer",
    "is_winter",
]
baseline_categorizer = Categorizer(columns=categorical)
glm_features = categorical
glm_regressor = GeneralizedLinearRegressor(
    family="gamma",
    scale_predictors=True,
    l1_ratio=1,
    alphas=1e-1,
)

X_train_t = baseline_categorizer.fit_transform(df[glm_features].iloc[train])
X_test_t = baseline_categorizer.transform(df[glm_features].iloc[test])
y_train_t, y_test_t = y.iloc[train], y.iloc[test]

glm_regressor.fit(X_train_t, y_train_t)
pd.DataFrame(
    {
        "coefficient": np.concatenate(
            ([glm_regressor.intercept_], glm_regressor.coef_)
        )
    },
    index=["intercept"] + glm_regressor.feature_names_,
).T


# %%
df_test["baseline_glm"] = glm_regressor.predict(X_test_t)
df_train["baseline_glm"] = glm_regressor.predict(X_train_t)

pd.set_option("display.float_format", lambda x: "%.3f" % x)
evaluate_predictions(
    df_test,
    outcome_column="tsd",
    tweedie_power=2.0,
    preds_column="baseline_glm",
)


# %%
evaluate_predictions(
    df_train,
    outcome_column="tsd",
    tweedie_power=2.0,
    preds_column="baseline_glm",
)


# %%
plot_test_set_predictions(
    df_test, "baseline_glm", "GLM baseline model prediction on test set"
)

# %%
# GLM model with weather numeric data
numeric = [
    "min_temp °c",
    "max_temp °c",
    "rain mm",
    "humidity %",
    "cloud_cover %",
    "wind_speed km/h",
]

# Let's put together pipeline for the GLM model
preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline(
                [
                    ("scale", StandardScaler()),
                ]
            ),
            numeric,
        ),
        ("cat", OneHotEncoder(sparse_output=False, drop="first"), categorical),
    ]
)
preprocessor.set_output(transform="pandas")
model_pipeline = Pipeline(
    [
        ("preprocess", preprocessor),
        (
            "estimate",
            glm_regressor,
        ),
    ]
)

# let's have a look at the pipeline
model_pipeline

# let's check that the transforms worked
model_pipeline[:-1].fit_transform(df_train)

model_pipeline.fit(df_train, y_train_t)

pd.DataFrame(
    {
        "coefficient": np.concatenate(
            ([model_pipeline[-1].intercept_], model_pipeline[-1].coef_)
        )
    },
    index=["intercept"] + model_pipeline[-1].feature_names_,
).T

df_test["demand_glm2"] = model_pipeline.predict(df_test)
df_train["demand_glm2"] = model_pipeline.predict(df_train)

evaluate_predictions(
    df_test,
    outcome_column="tsd",
    tweedie_power=2.0,
    preds_column="demand_glm2",
)


# %%
plot_test_set_predictions(
    df_test,
    "demand_glm2",
    "GLM model with weather data prediction on test set",
)

# %%
begin = pd.Timestamp("2022-02-01")
end = pd.Timestamp("2022-06-01")

fig, ax = plt.subplots(figsize=(15, 5))

ax.plot(
    df_test.loc[(df_test.index > begin) & (df_test.index < end)].index,
    df_test.loc[(df_test.index > begin) & (df_test.index < end)]["tsd"],
    "o",
    label="Test set",
)

ax.plot(
    df_test.loc[(df_test.index > begin) & (df_test.index < end)].index,
    df_test.loc[(df_test.index > begin) & (df_test.index < end)][
        "demand_glm2"
    ],
    "o",
    label="Prediction",
)

ax.legend(loc="center", bbox_to_anchor=(1.075, 0.5))

ax.set_title("Prediction on test set - Two weeks")
ax.set_ylabel("Energy Demand (MW)")
ax.set_xlabel("Date")

# %%
# GLM Cross-validation model for hyperparameter tuning
all_features = categorical + numeric
X_train_t = baseline_categorizer.fit_transform(df[all_features].iloc[train])
X_test_t = baseline_categorizer.transform(df[all_features].iloc[test])
y_train_t, y_test_t = y.iloc[train], y.iloc[test]

glmcv = GeneralizedLinearRegressorCV(
    family="gamma",
    alphas=None,  # default
    min_alpha=None,  # default
    min_alpha_ratio=None,  # default
    l1_ratio=[0, 0.1, 0.3, 0.5, 0.7, 0.8, 1.0],
    fit_intercept=True,
    max_iter=150,
)
glmcv.fit(X_train_t, y_train_t)
print(f"Chosen alpha:    {glmcv.alpha_}")
print(f"Chosen l1 ratio: {glmcv.l1_ratio_}")

# %%
df_test["demand_cv_glm"] = glmcv.predict(X_test_t)
df_train["demand_cv_glm"] = glmcv.predict(X_train_t)
evaluate_predictions(
    df_test,
    outcome_column="tsd",
    tweedie_power=2.0,
    preds_column="demand_cv_glm",
)

# %%
plot_test_set_predictions(
    df_test,
    "demand_cv_glm",
    "GLM model with weather data prediction on test set",
)

# %% [markdown]
# ## LGBM model
# The second forecasting method I will use is LGBM.
# The first LGBM model is a simple model for which some of the parameters
# are defined and the data is split into train and test sets.
# This model is fairly simple, but it's a great baseline.
# The second model is a tuned LGBM model. The parameters are tuned using GridSearchCV.

# %%
# Simple LGBM model
model_pipeline = Pipeline(
    [
        (
            "estimate",
            LGBMRegressor(
                objective="gamma",
                n_estimators=500,
                learning_rate=0.01,
                num_leaves=6,
                max_depth=3,
                random_state=43,
                early_stopping_rounds=50,
            ),
        )
    ]
)

model_pipeline.fit(
    X_train_t,
    y_train_t,
    estimate__eval_set=[(X_test_t, y_test_t), (X_train_t, y_train_t)],
)
lgb.plot_metric(model_pipeline[0])

df_test["demand_lgbm"] = model_pipeline.predict(X_test_t)
df_train["demand_lgbm"] = model_pipeline.predict(X_train_t)
evaluate_predictions(
    df_test,
    outcome_column="tsd",
    tweedie_power=2.0,
    preds_column="demand_lgbm",
)

# %%
plot_test_set_predictions(
    df_test, "demand_lgbm", "Simple LGBM predictions on test data"
)

# %%
# Extract the feature importances
feature_importances = model_pipeline.named_steps[
    "estimate"
].feature_importances_

# Get the feature names from the preprocessor
feature_names = (
    model_pipeline.named_steps["estimate"].booster_.feature_name(),
)

# # Create a DataFrame to display the feature importances
feature_importance_df = pd.DataFrame(
    index=feature_names[0],
    data=feature_importances.tolist(),
    columns=["Importance"],
).sort_values(by="Importance", ascending=False)


feature_importance_df.plot(kind="barh")

# %%
# Let's tune the pipeline to reduce overfitting

# Note: Typically we tune many more parameters and larger grids,
# but to save compute time here, we focus on getting the learning rate
# and the number of estimators somewhat aligned

# Define the parameter grid for tuning
param_grid = {
    "estimate__learning_rate": [0.01, 0.02, 0.03, 0.04, 0.05, 0.1],
    "estimate__n_estimators": [1000],
    "estimate__num_leaves": [6, 12, 24],
    "estimate__min_child_weight": [1, 5, 10],
}

# Initialize GridSearchCV with k-fold cross-validation
model_pipeline.named_steps["estimate"].set_params(early_stopping_rounds=25)
cv = GridSearchCV(
    model_pipeline,
    param_grid,
    cv=5,
    verbose=2,
    n_jobs=-1,
)
cv.fit(
    X_train_t,
    y_train_t,
    estimate__eval_set=[(X_test_t, y_test_t), (X_train_t, y_train_t)],
)

lgbm_tuning = cv.best_estimator_
df_test["demand_tuning_lgbm"] = cv.best_estimator_.predict(X_test_t)
df_train["demand_tuning_lgbm"] = cv.best_estimator_.predict(X_train_t)

evaluate_predictions(
    df_test,
    outcome_column="tsd",
    tweedie_power=2.0,
    preds_column="demand_tuning_lgbm",
)


# %%
# Plot learning curve
lgbm_tuning.fit(
    X_train_t,
    y_train_t,
    estimate__eval_set=[(X_test_t, y_test_t), (X_train_t, y_train_t)],
)

lgb.plot_metric(lgbm_tuning[0])

# %%
plot_test_set_predictions(
    df_test, "demand_tuning_lgbm", "Tunned LGBM predictions on test data"
)

# %%
# Extract the top 5 most important features
top_5_features = feature_importance_df.head(5).index.tolist()
top_5 = []
for i in top_5_features:
    if "_%" in i:
        i = i.replace("_%", " %")
    if "_°c" in i:
        i = i.replace("_°c", " °c")
    top_5.append(i)

# Create an explainer for the LGBM model
explainer_lgbm = dx.Explainer(
    lgbm_tuning, X_test_t, y_test_t, label="Tunned LGBM Model"
)

# Plot partial dependence plots for the top 5 features
pdp = explainer_lgbm.model_profile(variables=top_5, type="partial")
pdp.plot()


# %%
top_5

# %%
shap = explainer_lgbm.predict_parts(X_test_t.head(1), type="shap")

shap.plot()
