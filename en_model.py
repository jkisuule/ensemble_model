
# ENSEMBLE MODEL FOR STUDENT DROPOUT FORECASTING

# -----------------------------
# 1. IMPORTING REQUIRED LIBRARIES
# -----------------------------
import os
import json
import warnings
import subprocess
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')


# ---------------------------------------------------------
# 2. INSTALLING ANY MISSING PACKAGE AUTOMATICALLY IF NEEDED
# ---------------------------------------------------------


def ensure_package(package_name, import_name=None):
    import_name = import_name or package_name
    try:
        __import__(import_name)
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name])


# Install imbalanced-learn for SMOTE if it is not available
ensure_package('imbalanced-learn', 'imblearn')


# --------------------------------------------
# 3. IMPORTING MACHINE LEARNING TOOLS AFTER SETUP
# --------------------------------------------
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# 4. GLOBAL SETTINGS AND FILE PATHS

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)
DATA_PATH = 'dataset.csv'
TARGET_COL = 'Target'

# Checking whether the dataset exists in the working folder
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        "dataset.csv not found.Check path"
    )

# 5. HELPER FUNCTION TO SAVE TEXT FILE
def save_text(path, text):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)


# 6. LOADING THE CSV DATASET

# The attached CSV contains the student features and the final
# target variable used for dropout forecasting.
_df = pd.read_csv(DATA_PATH)


#saving in our output folder in a table format
#save_text(os.path.join(OUTPUT_DIR, 'dataset_preview.txt'), _df.head().to_markdown())

# Cleaning column names to avoid accidental spacing issues
_df.columns = [c.strip() for c in _df.columns]

# Confirming that the expected target column is present
if TARGET_COL not in _df.columns:
    raise ValueError("Target column not found in dataset.csv")

# Show all columns
pd.set_option('display.max_columns', 1000)

# Display first five rows
print(_df.head())

# ---------------------------------------------------
# 7. EXPLORING MISSING VALUES IN THE DATASET
# ---------------------------------------------------
# This section checks how many missing values exist and what
# percentage of the dataset they affect for each feature.
missing_report = pd.DataFrame({
    'feature': _df.columns,
    'missing_count': _df.isna().sum().values,
    'missing_pct': (_df.isna().mean() * 100).round(2).values,
    'dtype': _df.dtypes.astype(str).values
}).sort_values(['missing_pct', 'missing_count'], ascending=False)

# Saving the missing-value summary for reporting
missing_report.to_csv(os.path.join(OUTPUT_DIR, 'missing_values_report.csv'), index=False)


# -------------------------------------------------------------
# 8. CONVERTING TARGET INTO BINARY DROPOUT VS NOT-DROPOUT CLASSES
# -------------------------------------------------------------
# The supplied dataset includes classes such as Dropout,
# Graduate, and Enrolled. For early-warning dropout forecasting,

_df['Target_binary'] = np.where(
    _df[TARGET_COL].astype(str).str.strip().str.lower() == 'dropout',
    'Dropout',
    'Not_Dropout'
)

# Separate predictors and target
X = _df.drop(columns=[TARGET_COL, 'Target_binary']).copy()
y = _df['Target_binary'].copy()


#summary statistics of the model
print(_df.describe())

#summary statistics of the model
print(_df.info())


# ----------------------------------
# 9. FEATURE ENGINEERING
# ----------------------------------
# New features are created to improve prediction performance.
# These are academically meaningful combinations of the
# available institutional variables.
for sem in ['1st', '2nd']:
    approved = f'Curricular units {sem} sem (approved)'
    enrolled = f'Curricular units {sem} sem (enrolled)'
    evaluations = f'Curricular units {sem} sem (evaluations)'
    grade = f'Curricular units {sem} sem (grade)'
    without_eval = f'Curricular units {sem} sem (without evaluations)'

    # Approval rate = approved courses / enrolled courses
    if approved in X.columns and enrolled in X.columns:
        X[f'{sem}_approval_rate'] = np.where(X[enrolled] > 0, X[approved] / X[enrolled], 0)

    # Evaluation intensity = evaluations / enrolled courses
    if evaluations in X.columns and enrolled in X.columns:
        X[f'{sem}_evaluation_intensity'] = np.where(X[enrolled] > 0, X[evaluations] / X[enrolled], 0)

    # Rate of courses without evaluation
    if without_eval in X.columns and enrolled in X.columns:
        X[f'{sem}_without_eval_rate'] = np.where(X[enrolled] > 0, X[without_eval] / X[enrolled], 0)

    # Combined academic performance strength
    if grade in X.columns and approved in X.columns:
        X[f'{sem}_grade_x_approved'] = X[grade] * X[approved]

# Semester progression features
if {'Curricular units 1st sem (grade)', 'Curricular units 2nd sem (grade)'}.issubset(X.columns):
    X['grade_gap_2nd_minus_1st'] = X['Curricular units 2nd sem (grade)'] - X['Curricular units 1st sem (grade)']

if {'Curricular units 1st sem (approved)', 'Curricular units 2nd sem (approved)'}.issubset(X.columns):
    X['approved_gap_2nd_minus_1st'] = X['Curricular units 2nd sem (approved)'] - X['Curricular units 1st sem (approved)']

# Financial risk flag from debtor and tuition payment status
if {'Debtor', 'Tuition fees up to date'}.issubset(X.columns):
    X['financial_risk_flag'] = ((X['Debtor'] == 1) | (X['Tuition fees up to date'] == 0)).astype(int)

# Interaction feature linking age and scholarship support
if {'Age at enrollment', 'Scholarship holder'}.issubset(X.columns):
    X['age_scholarship_interaction'] = X['Age at enrollment'] * (X['Scholarship holder'] + 1)


# ------------------------------------------------------
# 10. IDENTIFING NUMERIC AND CATEGORICAL FEATURES
# ------------------------------------------------------
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = [c for c in X.columns if c not in numeric_features]


# ------------------------------------------------------
# 11. BUILDING PREPROCESSING PIPELINES
# ------------------------------------------------------
# Numeric variables:
# - median imputation for missing values
# - standardization for models like SVM and KNN
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical variables:
# - most frequent imputation
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent'))
])

# Combine transformations
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)


# ---------------------------------
# 12. ENCODING THE TARGET VARIABLE
# ---------------------------------
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
class_names = list(label_encoder.classes_)
positive_class = int(np.where(label_encoder.classes_ == 'Dropout')[0][0])


# ------------------------------------------------
# 13. CHECKING CLASS DISTRIBUTION / IMBALANCE
# ------------------------------------------------
class_dist = y.value_counts().rename_axis('class').reset_index(name='count')
class_dist['percentage'] = (class_dist['count'] / class_dist['count'].sum() * 100).round(2)
class_dist.to_csv(os.path.join(OUTPUT_DIR, 'class_distribution.csv'), index=False)


# -----------------------------------
# 14. TRAIN-TEST SPLIT (70/30)
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
                                    X,
                                    y_encoded,
                                    test_size=0.30,
                                    stratify=y_encoded,
                                    random_state=RANDOM_STATE
                                )


# ---------------------------------------------
# 15. CROSS-VALIDATION STRATEGY
# ---------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


# ---------------------------------------------------------
# 16. DEFINING BASE LEARNERS WITH SMOTE INSIDE THE PIPELINE
# ---------------------------------------------------------
# SMOTE is applied only to training folds, which is the


knn_pipe = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=RANDOM_STATE)),
    ('model', KNeighborsClassifier())
])

nb_pipe = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=RANDOM_STATE)),
    ('model', GaussianNB())
])


# ----------------------------------------------
# 17. HYPERPARAMETER GRIDS FOR OPTIMIZATION
# ----------------------------------------------

knn_grid = {
    'model__n_neighbors': [5, 7, 9, 11],
    'model__weights': ['uniform', 'distance']
}

nb_grid = {
    'model__var_smoothing': [1e-09, 1e-08, 1e-07]
}


# ----------------------------------------------------
# 18. GRID SEARCH FOR BEST MODEL CONFIGURATIONS
# ----------------------------------------------------

knn_search = GridSearchCV(knn_pipe, knn_grid, cv=cv, scoring='f1', n_jobs=-1)
nb_search = GridSearchCV(nb_pipe, nb_grid, cv=cv, scoring='f1', n_jobs=-1)

print('Tuning KNN...')
knn_search.fit(X_train, y_train)

print('Tuning Bayesian classifier...')
nb_search.fit(X_train, y_train)

# Retrieve optimized base learners
best_knn = knn_search.best_estimator_
best_nb = nb_search.best_estimator_


# -------------------------------------------------------
# 19. DEFINING ENSEMBLE MODELS
# -------------------------------------------------------
# Soft voting averages probabilities from all three models.
soft_voting = VotingClassifier(
    estimators=[('knn', best_knn), ('nb', best_nb)],
    voting='soft'
)

# Stacking uses outputs of base models as input for a final learner.
stacking = StackingClassifier(
    estimators=[('knn', best_knn), ('nb', best_nb)],
    final_estimator=GaussianNB(),
    stack_method='predict_proba',
    cv=5,
    n_jobs=-1
)


# ------------------------------------------------------
# 20. STORING ALL MODELS FOR COMPARISON
# ------------------------------------------------------
models = {
    'KNN': best_knn,
    'Bayesian_Network_Proxy': best_nb,
    'Soft_Voting_Ensemble': soft_voting,
    'Stacking_Ensemble': stacking
}
# 21. TRAINING, TESTING, AND EVALUATING EACH MODEL
results = []
reports = {}
conf_mats = {}
for name, model in models.items():
    print(f'Training and evaluating {name}...')

    # Fitting model on training data
    model.fit(X_train, y_train)

    # Predicting on unseen test data
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, positive_class]

    # Computing confusion matrix
    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()

    # Storing evaluation metrics
    row = {
        'model': name,
        'accuracy': accuracy_score(y_test, preds),
        'precision_dropout': precision_score(y_test, preds, pos_label=positive_class),
        'recall_dropout': recall_score(y_test, preds, pos_label=positive_class),
        'f1_dropout': f1_score(y_test, preds, pos_label=positive_class),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        'roc_auc': roc_auc_score(y_test, probs),
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }
    results.append(row)
    reports[name] = classification_report(y_test, preds, target_names=class_names)
    conf_mats[name] = cm

# Converting model results into a dataframe and rank by F1-score
results_df = pd.DataFrame(results).sort_values('f1_dropout', ascending=False)
results_df.to_csv(os.path.join(OUTPUT_DIR, 'model_comparison_results.csv'), index=False)


# ------------------------------------------------------
# 22. SAVING BEST HYPERPARAMETERS
# ------------------------------------------------------
with open(os.path.join(OUTPUT_DIR, 'best_parameters.json'), 'w', encoding='utf-8') as f:
    json.dump({
        'KNN': knn_search.best_params_,
        'Bayesian_Network_Proxy': nb_search.best_params_
    }, f, indent=2)



# 23. CREATING VISUALIZATIONS
# ------------------------------------------------------
sns.set_theme(style='whitegrid')

# Missing value plot
plt.figure(figsize=(12, 6))
sns.barplot(data=missing_report.head(20), x='missing_pct', y='feature', color='#01696f')
plt.title('Top Features by Missing Percentage')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'missing_values.png'), dpi=200)
plt.close()

# Class distribution plot
plt.figure(figsize=(6, 5))
ax = sns.countplot(x=y, palette=['#a13544', '#01696f'])
for p in ax.patches:
    ax.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom')
plt.title('Dropout vs Non-Dropout Distribution')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'class_distribution.png'), dpi=200)
plt.close()

# Correlation heatmap for the strongest numeric variables
numeric_all = X.select_dtypes(include=[np.number]).copy()
numeric_all['Target_encoded'] = y_encoded
corr_features = numeric_all.corr()['Target_encoded'].drop('Target_encoded').abs().sort_values(ascending=False).head(12).index.tolist()

plt.figure(figsize=(11, 8))
sns.heatmap(
    numeric_all[corr_features + ['Target_encoded']].corr(),
    annot=True,
    fmt='.2f',
    cmap='RdBu_r',
    center=0
)
plt.title('Correlation Heatmap of Top Numeric Features')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_heatmap.png'), dpi=200)
plt.close()


# Pairplot for selected variables with clear x-axis labels
key_plot_cols = [
    c for c in [
        'Curricular units 1st sem (approved)',
        'Curricular units 2nd sem (approved)',
        'Age at enrollment'
    ] if c in X.columns
]

if len(key_plot_cols) >= 2:
    pair_df = X[key_plot_cols].copy()
    pair_df['Target'] = y.values

    # Renaming columns to shorter labels for clearer axis display
    pair_df = pair_df.rename(columns={
        'Curricular units 1st sem (approved)': '1st Sem\nApproved Units',
        'Curricular units 2nd sem (approved)': '2nd Sem\nApproved Units',
        'Age at enrollment': 'Age at\nEnrollment'
    })

    g = sns.pairplot(
        pair_df,
        hue='Target',
        corner=True,
        diag_kind='hist',
        height=3.2,
        aspect=1.2,
        plot_kws={'alpha': 0.6, 's': 25}
    )

    # Improving axis readability
    for ax in g.axes.flatten():
        if ax is not None:
            ax.tick_params(axis='x', labelrotation=0, labelsize=10)
            ax.tick_params(axis='y', labelsize=10)
            ax.xaxis.label.set_size(11)
            ax.yaxis.label.set_size(11)

    plt.subplots_adjust(bottom=0.12, left=0.10, right=0.88, top=0.95)
    plt.savefig(os.path.join(OUTPUT_DIR, 'pairplot_key_features.png'), dpi=300, bbox_inches='tight')
    plt.close('all')

# Model performance comparison chart
plt.figure(figsize=(12, 6))
plot_df = results_df[
    ['model', 'accuracy', 'precision_dropout', 'recall_dropout', 'f1_dropout', 'roc_auc']
].melt(id_vars='model', var_name='metric', value_name='score')

sns.barplot(data=plot_df, x='metric', y='score', hue='model')
plt.ylim(0, 1.05)
plt.title('Model Performance Comparison')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'model_performance_comparison.png'), dpi=200)
plt.close()

# Confusion matrices for all models
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()
for ax, (name, cm) in zip(axes, conf_mats.items()):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_title(name)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
for i in range(len(conf_mats), len(axes)):
    axes[i].axis('off')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrices.png'), dpi=200)
plt.close()


# ------------------------------------------------------
# 24. GENERATING A TEXT REPORT
# ------------------------------------------------------
report_lines = []
report_lines.append('ENSEMBLE MODEL REPORT FOR STUDENT DROPOUT FORECASTING')
report_lines.append('=' * 70)
report_lines.append(f'Dataset shape: {_df.shape[0]} rows x {_df.shape[1]} columns')
report_lines.append('')
report_lines.append('Missing values summary:')
report_lines.append(missing_report.head(20).to_string(index=False))
report_lines.append('')
report_lines.append('Class distribution:')
report_lines.append(class_dist.to_string(index=False))
report_lines.append('')
report_lines.append('Best hyperparameters:')
report_lines.append(json.dumps({
    'KNN': knn_search.best_params_,
    'Bayesian_Network_Proxy': nb_search.best_params_
}, indent=2))
report_lines.append('')
report_lines.append('Model comparison:')
report_lines.append(results_df.to_string(index=False))
report_lines.append('')
report_lines.append('Classification reports:')
for name, rep in reports.items():
    report_lines.append(f'\n{name}\n{rep}')
report_lines.append('')
report_lines.append('Actionable insights:')
report_lines.append('- Focus retention interventions on students with low approved units, low grades, poor evaluation participation, and financial risk flags.')
report_lines.append('- Use the best-performing ensemble model for operational screening of at-risk students.')
report_lines.append('- Review confusion matrices and recall for Dropout carefully because missing a true dropout is costly for early intervention.')

save_text(os.path.join(OUTPUT_DIR, 'ensemble_model_report.txt'), '\n'.join(report_lines))


# ------------------------------------------------------
# 25. FINAL COMPLETION MESSAGE
# ------------------------------------------------------
print('\nDone. Main outputs saved in the output/ folder:')
print('- model_comparison_results.csv')
print('- best_parameters.json')
print('- permutation_feature_importance.csv')
print('- ensemble_model_report.txt')
print('- PNG charts for missing values, class distribution, correlation heatmap, pairplot, performance comparison, confusion matrices, and feature importance')
