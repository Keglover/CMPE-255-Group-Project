import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

df = pd.read_csv('diabetes_prediction_dataset.csv')

print("Dataset Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())
print("\nClass Distribution:")
print(df['diabetes'].value_counts(normalize=True))


train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['diabetes'])

val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['diabetes'])

print("\nSplit sizes:")
print(f"Training set: {len(train_df)} samples")
print(f"Validation set: {len(val_df)} samples")
print(f"Test set: {len(test_df)} samples")

def preprocess_data(df):
    """Preprocess the data by converting categorical variables to numeric."""
    df_processed = df.copy()
    
    label_encoders = {}
    categorical_columns = ['gender', 'smoking_history']
    
    for col in categorical_columns:
        label_encoders[col] = LabelEncoder()
        df_processed[col] = label_encoders[col].fit_transform(df_processed[col])
    
    # Separate features and target
    X = df_processed.drop('diabetes', axis=1)
    y = df_processed['diabetes']
    
    return X, y, label_encoders

X_train, y_train, label_encoders = preprocess_data(train_df)
X_val, y_val, _ = preprocess_data(val_df)
X_test, y_test, _ = preprocess_data(test_df)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("\nClass distribution after SMOTE:")
print(pd.Series(y_train_resampled).value_counts(normalize=True))

def evaluate_decision_tree(X_train, y_train, X_val, y_val, X_test, y_test, max_depth=None, min_samples_split=2):

    dt = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    dt.fit(X_train, y_train)
    
    y_val_pred = dt.predict(X_val)
    y_val_prob = dt.predict_proba(X_val)[:, 1]
    
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    val_roc_auc = roc_auc_score(y_val, y_val_prob)
    
    print(f"\nDecision Tree with max_depth={max_depth}, min_samples_split={min_samples_split}")
    print("Validation Set Performance:")
    print(f"Accuracy: {val_accuracy:.4f}")
    print(f"F1 Score: {val_f1:.4f}")
    print(f"ROC-AUC: {val_roc_auc:.4f}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_val, y_val_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix (max_depth={max_depth})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'plots/confusion_matrix_dt_depth_{max_depth}.png')
    plt.close()
    
    # Plot feature importances
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': dt.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title(f'Feature Importances (max_depth={max_depth})')
    plt.tight_layout()
    plt.savefig(f'plots/feature_importances_dt_depth_{max_depth}.png')
    plt.close()
    
    # Plot the decision tree (limited to depth 3 for visualization)
    plt.figure(figsize=(20, 10))
    plot_tree(dt, 
              feature_names=X_train.columns,
              class_names=['No Diabetes', 'Diabetes'],
              filled=True,
              rounded=True,
              max_depth=3)
    plt.title(f'Decision Tree (max_depth={max_depth})')
    plt.savefig(f'plots/decision_tree_depth_{max_depth}.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Evaluate on test set
    y_test_pred = dt.predict(X_test)
    y_test_prob = dt.predict_proba(X_test)[:, 1]
    
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_roc_auc = roc_auc_score(y_test, y_test_prob)
    
    print("\nTest Set Performance:")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"F1 Score: {test_f1:.4f}")
    print(f"ROC-AUC: {test_roc_auc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred))
    
    return dt

# Evaluate different tree depths
depths = [3, 5, 7, 10, None]
best_f1 = 0
best_dt = None

print("\n" + "="*50)
print("Evaluating Different Tree Depths")
print("="*50)

for depth in depths:
    dt = evaluate_decision_tree(
        X_train_resampled, 
        y_train_resampled, 
        X_val, 
        y_val, 
        X_test, 
        y_test,
        max_depth=depth
    )
    
    # Track best model
    y_val_pred = dt.predict(X_val)
    current_f1 = f1_score(y_val, y_val_pred)
    
    if current_f1 > best_f1:
        best_f1 = current_f1
        best_dt = dt

print("\n" + "="*50)
print(f"Best performing model had max_depth={best_dt.max_depth}")
print("="*50) 