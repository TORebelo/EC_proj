import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# Load the dataset
data = pd.read_csv('data/custom_covid19.csv') 
print("Data loaded with shape:", data.shape)

# 1. Create target variable
data['DIED'] = data['DATE_DIED'].apply(lambda x: 0 if x == '9999-99-99' else 1)

# 2. Mark missing values
data.replace([97, 98, 99], np.nan, inplace=True)

# 3. Convert boolean columns (1=yes, 2=no → 1=yes, 0=no)
bool_cols = ['INTUBED', 'PNEUMONIA', 'PREGNANT', 'DIABETES', 'COPD', 'ASTHMA', 
             'INMSUPR', 'HYPERTENSION', 'OTHER_DISEASE', 'CARDIOVASCULAR', 
             'OBESITY', 'RENAL_CHRONIC', 'TOBACCO', 'ICU']
data[bool_cols] = data[bool_cols].replace(2, 0)

# 4. Create COVID status feature
data['COVID_POSITIVE'] = data['TEST_RESULT'].apply(lambda x: 1 if x in [1,2,3] else 0)

# 5. Define features to keep/drop
features_to_keep = ['USMER', 'MEDICAL_UNIT', 'SEX', 'PATIENT_TYPE', 
                   'INTUBED', 'PNEUMONIA', 'AGE', 'DIABETES', 'COPD',
                   'ASTHMA', 'INMSUPR', 'HYPERTENSION', 'OTHER_DISEASE',
                   'CARDIOVASCULAR', 'OBESITY', 'RENAL_CHRONIC', 'TOBACCO',
                   'ICU', 'COVID_POSITIVE']

# 6. Separate features and target
X = data[features_to_keep]
y = data['DIED']

# 76. Define preprocessing pipeline
numeric_features = ['AGE']
categorical_features = [col for col in features_to_keep if col not in numeric_features + ['DIED']]
print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)


# Preprocessing for numeric features (scaling) and categorical features (imputation)
# Define the preprocessing steps for numeric and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', SimpleImputer(strategy='most_frequent'), categorical_features)
    ])

# 8. Train-test split
# Split the data into training and testing sets
# Stratified split to maintain the same distribution of the target variable in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)



print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
print(f"Class distribution (train): {pd.Series(y_train).value_counts(normalize=True)}")


# 9. Fit Evaluating Models
# === MODEL EVALUATION FRAMEWORK ===
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown

# Configure visualization settings
plt.style.use('seaborn-darkgrid')  
plt.rcParams['figure.figsize'] = (8, 4)
sns.set_palette("husl")

def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """
    Evaluates a classification model and generates report-ready outputs
    
    Parameters:
    - name: str, model name for display
    - model: sklearn classifier object
    - X_train, X_test, y_train, y_test: training/test data
    
    Returns:
    - Dictionary containing metrics and visualization paths
    """
    # Create pipeline and fit model
    clf = make_pipeline(preprocessor, model)
    clf.fit(X_train, y_train)
    
    # Generate predictions
    y_pred = clf.predict(X_test)
    
    # Create classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    # Generate visualizations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Confusion matrix heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['Survived', 'Died'],
                yticklabels=['Survived', 'Died'])
    ax1.set_title(f'{name} Confusion Matrix')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    
    # Metrics bar plot
    metrics = ['precision', 'recall', 'f1-score']
    scores = [report['weighted avg'][m] for m in metrics]
    sns.barplot(x=metrics, y=scores, ax=ax2)
    ax2.set_title(f'{name} Performance Metrics')
    ax2.set_ylim(0, 1)
    
    # Save figures
    fig_path = f'figures/{name.lower().replace(" ", "_")}_performance.png'
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    
    # Display results in notebook
    display(Markdown(f"## {name} Performance"))
    display(Markdown("### Classification Report"))
    print(classification_report(y_test, y_pred))
    
    display(Markdown("### Confusion Matrix"))
    print(cm)
    
    # Return structured results for report
    return {
        'model': name,
        'accuracy': report['accuracy'],
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1': report['weighted avg']['f1-score'],
        'figure_path': fig_path
    }

# Initialize results storage
model_results = []

# Evaluate baseline models
models = [
    ('Naive Bayes', GaussianNB()),
    ('K-Nearest Neighbors (k=5)', KNeighborsClassifier(n_neighbors=5)),
    ('SVM (RBF kernel)', SVC(kernel='rbf', gamma='scale'))
]

for name, model in models:
    result = evaluate_model(name, model, X_train, X_test, y_train, y_test)
    model_results.append(result)

# Convert results to DataFrame for report
results_df = pd.DataFrame(model_results)
results_df.to_markdown('tables/model_comparison.md', index=False)
display(results_df)