# --- Objective O4: Identifying Most Significant Features ---

print("\n--- Starting Objective O4: Identifying Most Significant Features ---")

# Ensure 'o3_results' (from Objective O3) is populated
# Ensure 'gs_linear_svc' and 'gs_dt_clf' (or similar GridSearchCV results from Objective O1) are available and fitted.
# If you haven't run O1's GridSearchCV cells yet, please do so before running O4
# to ensure 'gs_linear_svc.best_estimator_' and 'gs_dt_clf.best_estimator_' are populated.

# Helper function to plot feature importance (reused from previous response)
def plot_feature_importance(importances, feature_names, title, filename):
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title(title)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    # Ensure 'figures' directory exists or create it
    import os
    if not os.path.exists('figures'):
        os.makedirs('figures')
    plt.savefig(f"figures/{filename}.png")
    plt.show()

#### 1. Feature Importance for O1 (Death Prediction) Models

print("\n--- Extracting feature importance for O1 (Death Prediction) models ---")

# Accessing best models from O1 GridSearchCV results
# IMPORTANT: These variables (gs_linear_svc, gs_dt_clf) MUST be populated by running your O1 GridSearchCV cells.
# If you defined them differently, adjust the variable names accordingly.

# --- For Best Linear SVC Classifier (O1 Classification) ---
try:
    best_linear_svc_o1 = globals()['gs_linear_svc'].best_estimator_ # Access the best fitted pipeline from O1
    print("\nExtracting feature importance for Best Linear SVC (O1 - Death Prediction):")
    svc_classifier = best_linear_svc_o1.named_steps['classifier'] # Get the LinearSVC model from the pipeline
    coef = svc_classifier.coef_[0] # Get coefficients (for binary classification, it's a 2D array)

    # Get feature names from the preprocessor within the best_linear_svc_o1 pipeline
    # This is the most robust way to get feature names after transformations
    try:
        # This will work if your ColumnTransformer supports get_feature_names_out() and no OHE expands names
        processed_feature_names_o1_svc = best_linear_svc_o1.named_steps['preprocessor'].get_feature_names_out()
    except Exception:
        # Fallback: if get_feature_names_out() doesn't work, we assume features_to_keep_o1 order
        processed_feature_names_o1_svc = initial_features_to_keep # (from O1 setup)

    if len(coef) == len(processed_feature_names_o1_svc):
        plot_feature_importance(np.abs(coef), processed_feature_names_o1_svc,
                                "Feature Importance: Best Linear SVC (O1 - Death Prediction)",
                                "O4_linear_svc_o1_feature_importance")
    else:
        print(f"Warning: Coefficient length ({len(coef)}) does not match feature names length ({len(processed_feature_names_o1_svc)}). "
              "Cannot plot feature importance for Linear SVC accurately without proper feature name mapping. "
              "Check your O1 preprocessor for transformations like OneHotEncoder.")

except (NameError, KeyError):
    print("Warning: gs_linear_svc (best O1 Linear SVC) not found or not fitted. Skipping feature importance for this model.")
except Exception as e:
    print(f"An error occurred while getting Linear SVC feature importance: {e}")


# --- For Best Decision Tree Classifier (O1 Classification) ---
try:
    best_dt_o1 = globals()['gs_dt_clf'].best_estimator_ # Access the best fitted pipeline from O1
    print("\nExtracting feature importance for Best Decision Tree Classifier (O1 - Death Prediction):")
    dt_classifier = best_dt_o1.named_steps['classifier'] # Get the Decision Tree model
    importances = dt_classifier.feature_importances_

    try:
        processed_feature_names_o1_dt = best_dt_o1.named_steps['preprocessor'].get_feature_names_out()
    except Exception:
        processed_feature_names_o1_dt = initial_features_to_keep # Fallback

    if len(importances) == len(processed_feature_names_o1_dt):
        plot_feature_importance(importances, processed_feature_names_o1_dt,
                                "Feature Importance: Best Decision Tree (O1 - Death Prediction)",
                                "O4_decision_tree_o1_feature_importance")
    else:
        print(f"Warning: Feature importances length ({len(importances)}) does not match feature names length ({len(processed_feature_names_o1_dt)}). "
              "Cannot plot feature importance for Decision Tree accurately without proper feature name mapping.")

except (NameError, KeyError):
    print("Warning: gs_dt_clf (best O1 Decision Tree) not found or not fitted. Skipping feature importance for this model.")
except Exception as e:
    print(f"An error occurred while getting Decision Tree Classifier feature importance: {e}")


#### 2. Feature Importance for O3 (Age Prediction) Models

print("\n--- Extracting feature importance for O3 (Age Prediction) models ---")

# Feature names for O3 were defined as features_for_age_prediction
# This list is used for the preprocessor_reg and thus for the models.
processed_feature_names_o3 = features_for_age_prediction

# --- For Best Random Forest Regressor (O3 - COVID-Positive Deceased Age Prediction) ---
if o3_results and o3_results['covid_pos'] and len(o3_results['covid_pos']) > 2 and o3_results['covid_pos'][2] and hasattr(o3_results['covid_pos'][2]['best_model'].named_steps['regressor'], 'feature_importances_'):
    print("\nExtracting feature importance for Best Random Forest Regressor (O3 - COVID-Positive Deceased Age Prediction):")
    rf_regressor_pos = o3_results['covid_pos'][2]['best_model'].named_steps['regressor'] # Index 2 is Random Forest
    importances_pos = rf_regressor_pos.feature_importances_

    if len(importances_pos) == len(processed_feature_names_o3):
        plot_feature_importance(importances_pos, processed_feature_names_o3,
                                "Feature Importance: RF Regressor (O3 - COVID-Positive Deceased Age Prediction)",
                                "O4_rf_regressor_covid_pos_feature_importance")
    else:
        print(f"Warning: Feature importances length ({len(importances_pos)}) does not match feature names length ({len(processed_feature_names_o3)}). "
              "Cannot plot feature importance for RF Regressor (COVID-Positive) accurately.")
else:
    print("Best Random Forest Regressor for COVID-Positive deceased not found or not fitted correctly for feature importance (from O3).")


# --- For Best Random Forest Regressor (O3 - COVID-Negative Deceased Age Prediction) ---
if o3_results and o3_results['covid_neg'] and len(o3_results['covid_neg']) > 2 and o3_results['covid_neg'][2] and hasattr(o3_results['covid_neg'][2]['best_model'].named_steps['regressor'], 'feature_importances_'):
    print("\nExtracting feature importance for Best Random Forest Regressor (O3 - COVID-Negative Deceased Age Prediction):")
    rf_regressor_neg = o3_results['covid_neg'][2]['best_model'].named_steps['regressor'] # Index 2 is Random Forest
    importances_neg = rf_regressor_neg.feature_importances_

    if len(importances_neg) == len(processed_feature_names_o3):
        plot_feature_importance(importances_neg, processed_feature_names_o3,
                                "Feature Importance: RF Regressor (O3 - COVID-Negative Deceased Age Prediction)",
                                "O4_rf_regressor_covid_neg_feature_importance")
    else:
        print(f"Warning: Feature importances length ({len(importances_neg)}) does not match feature names length ({len(processed_feature_names_o3)}). "
              "Cannot plot feature importance for RF Regressor (COVID-Negative) accurately.")
else:
    print("Best Random Forest Regressor for COVID-Negative deceased not found or not fitted correctly for feature importance (from O3).")


# --- For Best Linear Regression (O3 - COVID-Positive Deceased Age Prediction) ---
if o3_results and o3_results['covid_pos'] and len(o3_results['covid_pos']) > 0 and o3_results['covid_pos'][0] and hasattr(o3_results['covid_pos'][0]['best_model'].named_steps['regressor'], 'coef_'):
    print("\nExtracting feature importance for Best Linear Regression (O3 - COVID-Positive Deceased Age Prediction):")
    lin_reg_pos = o3_results['covid_pos'][0]['best_model'].named_steps['regressor'] # Index 0 is Linear Regression
    coef_pos = np.abs(lin_reg_pos.coef_) # Using absolute coefficients for importance

    if len(coef_pos) == len(processed_feature_names_o3):
        plot_feature_importance(coef_pos, processed_feature_names_o3,
                                "Feature Importance: Linear Regressor (O3 - COVID-Positive Deceased Age Prediction)",
                                "O4_lin_reg_covid_pos_feature_importance")
    else:
        print(f"Warning: Coefficient length ({len(coef_pos)}) does not match feature names length ({len(processed_feature_names_o3)}). "
              "Cannot plot feature importance for Linear Regressor (COVID-Positive) accurately.")
else:
    print("Best Linear Regression for COVID-Positive deceased not found or not fitted correctly for feature importance (from O3).")

# --- For Best Linear Regression (O3 - COVID-Negative Deceased Age Prediction) ---
if o3_results and o3_results['covid_neg'] and len(o3_results['covid_neg']) > 0 and o3_results['covid_neg'][0] and hasattr(o3_results['covid_neg'][0]['best_model'].named_steps['regressor'], 'coef_'):
    print("\nExtracting feature importance for Best Linear Regression (O3 - COVID-Negative Deceased Age Prediction):")
    lin_reg_neg = o3_results['covid_neg'][0]['best_model'].named_steps['regressor'] # Index 0 is Linear Regression
    coef_neg = np.abs(lin_reg_neg.coef_) # Using absolute coefficients for importance

    if len(coef_neg) == len(processed_feature_names_o3):
        plot_feature_importance(coef_neg, processed_feature_names_o3,
                                "Feature Importance: Linear Regressor (O3 - COVID-Negative Deceased Age Prediction)",
                                "O4_lin_reg_covid_neg_feature_importance")
    else:
        print(f"Warning: Coefficient length ({len(coef_neg)}) does not match feature names length ({len(processed_feature_names_o3)}). "
              "Cannot plot feature importance for Linear Regressor (COVID-Negative) accurately.")
else:
    print("Best Linear Regression for COVID-Negative deceased not found or not fitted correctly for feature importance (from O3).")


print("\n--- Objective O4 complete. Remember to review the generated plots in the 'figures' directory. ---")