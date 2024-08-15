import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load the datasets
file_path = '/Users/adekunle/Downloads/Wheat_traits_27_7_2024(perhaps)full.xlsx'
data = pd.read_excel(file_path)

# 1. Color Model
data['ColorCategory'] = data['Filename'].apply(lambda x: x.split('-')[1].strip().lower() if '-' in x else np.nan)
data = data.dropna(subset=['ColorCategory'])

data['ColorCategory'] = data['ColorCategory'].replace({
    'red 2': 'red', 
    'red': 'red', 
    'purplr': 'purple', 
    'bycolor': 'bicolour',
    'amber': 'amber',
    'white': 'white',
    'brown': 'brown',
})

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(data['ColorCategory'])

X = data[[
    'FormFactor', 'NonConvexArea_advanced', 'FourierShapeDescriptor [1]',
    'ReflectanceTophatMean', 'Compactness Circle', 'Compactness', 
    'CIELabStdevProfile [1]', 'ReflectanceMeanProfile [2]', 
    'CIELabStdevProfile [13]'
]]

X.columns = [col.replace('[', '_').replace(']', '_').replace('<', '_') for col in X.columns]

# Interaction features
X_interactions = X.copy()
for col1 in X.columns:
    for col2 in X.columns:
        if col1 != col2:
            X_interactions[f'{col1}_x_{col2}'] = X[col1] * X[col2]

X_combined = pd.concat([X, X_interactions], axis=1)

# PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_pca, y_encoded, test_size=0.2, random_state=42)

# Define individual models
rf_model = RandomForestClassifier(random_state=42, n_estimators=200, max_depth=20, bootstrap=False)
xgb_model = XGBClassifier(eval_metric='mlogloss', random_state=42, colsample_bytree=0.6, learning_rate=0.2, max_depth=10)
lgb_model = LGBMClassifier(random_state=42)
catboost_model = CatBoostClassifier(silent=True, random_state=42)

# Stacking Classifier
estimators = [
    ('rf', rf_model),
    ('xgb', xgb_model),
    ('lgb', lgb_model),
    ('catboost', catboost_model)
]

stacking_clf = StackingClassifier(
    estimators=estimators, final_estimator=LogisticRegression(), cv=5, n_jobs=-1
)

# Train ensemble model
stacking_clf.fit(X_train, y_train)

# Evaluate ensemble model
y_pred = stacking_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

class_report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", class_report)

# Correlation Matrix for Color Model (only selected features)
correlation_matrix = pd.concat([pd.DataFrame(X), pd.DataFrame(y_encoded, columns=['ColorCategory'])], axis=1).corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix for Selected Features of Color Model')
plt.show()

# Precision-Recall Visualization
categories = label_encoder.classes_
precision = [0.85, 1.00, 0.88, 1.00, 0.83, 0.86]  # Replace with actual precision values
recall = [0.85, 0.81, 0.82, 0.86, 0.89, 0.87]  # Replace with actual recall values

x = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, precision, width, label='Precision')
rects2 = ax.bar(x + width/2, recall, width, label='Recall')

ax.set_xlabel('Color Categories')
ax.set_ylabel('Scores')
ax.set_title('Precision and Recall for Color Classification Categories')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
plt.show()

# Confusion Matrix for Color Model
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=label_encoder.classes_)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix for Color Model')
plt.show()

# Save the Color Model
joblib.dump(stacking_clf, '/Users/adekunle/Downloads/stacking_model_color.pkl')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load the datasets
file_path = '/Users/adekunle/Downloads/Wheat_traits_27_7_2024(perhaps)full.xlsx'
data = pd.read_excel(file_path)

# 1. Color Model
data['ColorCategory'] = data['Filename'].apply(lambda x: x.split('-')[1].strip().lower() if '-' in x else np.nan)
data = data.dropna(subset=['ColorCategory'])

data['ColorCategory'] = data['ColorCategory'].replace({
    'red 2': 'red', 
    'red': 'red', 
    'purplr': 'purple', 
    'bycolor': 'bicolour',
    'amber': 'amber',
    'white': 'white',
    'brown': 'brown',
})

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(data['ColorCategory'])

X = data[[
    'FormFactor', 'NonConvexArea_advanced', 'FourierShapeDescriptor [1]',
    'ReflectanceTophatMean', 'Compactness Circle', 'Compactness', 
    'CIELabStdevProfile [1]', 'ReflectanceMeanProfile [2]', 
    'CIELabStdevProfile [13]'
]]

X.columns = [col.replace('[', '_').replace(']', '_').replace('<', '_') for col in X.columns]

# Interaction features
X_interactions = X.copy()
for col1 in X.columns:
    for col2 in X.columns:
        if col1 != col2:
            X_interactions[f'{col1}_x_{col2}'] = X[col1] * X[col2]

X_combined = pd.concat([X, X_interactions], axis=1)

# PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_pca, y_encoded, test_size=0.2, random_state=42)

# Define individual models
rf_model = RandomForestClassifier(random_state=42, n_estimators=200, max_depth=20, bootstrap=False)
xgb_model = XGBClassifier(eval_metric='mlogloss', random_state=42, colsample_bytree=0.6, learning_rate=0.2, max_depth=10)
lgb_model = LGBMClassifier(random_state=42)
catboost_model = CatBoostClassifier(silent=True, random_state=42)

# Stacking Classifier
estimators = [
    ('rf', rf_model),
    ('xgb', xgb_model),
    ('lgb', lgb_model),
    ('catboost', catboost_model)
]

stacking_clf = StackingClassifier(
    estimators=estimators, final_estimator=LogisticRegression(), cv=5, n_jobs=-1
)

# Train ensemble model
stacking_clf.fit(X_train, y_train)

# Evaluate ensemble model
y_pred = stacking_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

class_report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", class_report)

# Correlation Matrix for Color Model (only selected features)
correlation_matrix = pd.concat([pd.DataFrame(X), pd.DataFrame(y_encoded, columns=['ColorCategory'])], axis=1).corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix for Selected Features of Color Model')
plt.show()

# Precision-Recall Visualization
categories = label_encoder.classes_
precision = [0.85, 1.00, 0.88, 1.00, 0.83, 0.86]  # Replace with actual precision values
recall = [0.85, 0.81, 0.82, 0.86, 0.89, 0.87]  # Replace with actual recall values

x = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, precision, width, label='Precision')
rects2 = ax.bar(x + width/2, recall, width, label='Recall')

ax.set_xlabel('Color Categories')
ax.set_ylabel('Scores')
ax.set_title('Precision and Recall for Color Classification Categories')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
plt.show()

# Confusion Matrix for Color Model
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=label_encoder.classes_)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix for Color Model')
plt.show()

# Save the Color Model
joblib.dump(stacking_clf, '/Users/adekunle/Downloads/stacking_model_color.pkl')

# 2. Original Size Model
data_1 = pd.read_excel(file_path)

small_thresholds_adjusted = {
    'Area': 12,
    'Length': 6,
    'Width': 2.5
}
medium_thresholds_adjusted = {
    'Area': 17,
    'Length': 7,
    'Width': 3.5
}

def classify_seed(row):
    if (row['Area (mm2)'] <= small_thresholds_adjusted['Area'] and 
        row['Length (mm)'] <= small_thresholds_adjusted['Length'] and 
        row['Width (mm)'] <= small_thresholds_adjusted['Width']):
        return 'Small'
    elif (small_thresholds_adjusted['Area'] < row['Area (mm2)'] <= medium_thresholds_adjusted['Area'] and 
          small_thresholds_adjusted['Length'] < row['Length (mm)'] <= medium_thresholds_adjusted['Length'] and 
          small_thresholds_adjusted['Width'] < row['Width (mm)'] <= medium_thresholds_adjusted['Width']):
        return 'Medium'
    elif (medium_thresholds_adjusted['Area'] < row['Area (mm2)'] <= data_1['Area (mm2)'].max() and 
          medium_thresholds_adjusted['Length'] < row['Length (mm)'] <= data_1['Length (mm)'].max() and 
          medium_thresholds_adjusted['Width'] < row['Width (mm)'] <= data_1['Width (mm)'].max()):
        return 'Large'
    else:
        return 'Medium'  # Default to Medium to ensure all are classified

# Apply classification to the first dataset
data_1['Seed Size Category'] = data_1.apply(classify_seed, axis=1)

# Define features and target
features = [
    'Area (mm2)', 'Length (mm)', 'Width (mm)', 
    'Compactness Circle', 'Compactness Ellipse', 
    'BetaShapeParams [0]', 'BetaShapeParams [1]', 'BetaShapeParams [2]', 
    'CIE Colorspace Components [0]'
]
X_size = data_1[features]
y_size = data_1['Seed Size Category']

# Split the data into training and testing sets
X_train_size, X_test_size, y_train_size, y_test_size = train_test_split(X_size, y_size, test_size=0.3, random_state=42)

# Train the model
clf_size = GradientBoostingClassifier(random_state=42)
clf_size.fit(X_train_size, y_train_size)

# Predict on the test set
y_pred_size = clf_size.predict(X_test_size)

# Generate classification report
report_size = classification_report(y_test_size, y_pred_size)
print("Classification Report for Size Model:\n", report_size)

# Correlation Matrix for Size Model (only selected features)
correlation_matrix_size = pd.concat([X_size, y_size.map({'Small': 0, 'Medium': 1, 'Large': 2})], axis=1).corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix_size, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix for Selected Features of Size Model')
plt.show()

# Confusion Matrix for Size Model
conf_matrix_size = confusion_matrix(y_test_size, y_pred_size, labels=clf_size.classes_)
disp_size = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_size, display_labels=clf_size.classes_)
disp_size.plot(cmap='Blues')
plt.title('Confusion Matrix for Size Model')
plt.show()

# Save the model
joblib.dump(clf_size, '/Users/adekunle/Downloads/size_model.pkl')

# Box Plot Visualization for Length
plt.figure(figsize=(10, 6))
sns.boxplot(x='Filename', y='Length (mm)', data=data_1)
plt.title('Distribution of Lengths by Filename')
plt.xticks(rotation=90)
plt.show()

# Save the classified data to an Excel file
classified_data_filename = 'classified_wheat_data.xlsx'
data_1.to_excel(classified_data_filename, index=False)
print(f"Classified data saved to {classified_data_filename}")

# 3. Taxonomy Model
# Filter and map taxonomy categories
filtered_data = data[~data['Reference Classes'].str.contains('g-') & ~data['Filename'].str.contains('g-')]

def map_taxonomy(reference_class, filename):
    if 'dw' in reference_class or 'dw' in filename:
        return 'DW'  # Durum wheat
    elif 'cwi' in reference_class or 'cwi' in filename:
        return 'CWI'  # Common wheat
    elif 'bw' in reference_class or 'bw' in filename:
        return 'BW'  # Bread wheat
    else:
        return None

filtered_data['Taxonomy_Label'] = filtered_data.apply(lambda row: map_taxonomy(row['Reference Classes'], row['Filename']), axis=1)
filtered_data = filtered_data.dropna(subset=['Taxonomy_Label'])

filtered_data['Taxonomy_Encoded'] = filtered_data['Taxonomy_Label'].map({'CWI': 0, 'BW': 0, 'DW': 1})

# Select the top 10 features
selected_features_tax = [
    "BasicImageFeaturesMax [1]", "BasicImageFeaturesMax [0]", 
    "ReflectanceMeanProfile [2]", "ReflectanceTophatMean", 
    "ReflectanceMeanProfile [0]", "FormFactor", 
    "NonConvexArea_basic", "NonConvexArea_advanced", 
    "CIELabStdevProfile [2]", "SpectralMean [5]"
]

X_tax = filtered_data[selected_features_tax]
y_tax = filtered_data['Taxonomy_Encoded']

# Split the data into training and testing sets
X_train_tax, X_test_tax, y_train_tax, y_test_tax = train_test_split(X_tax, y_tax, test_size=0.2, random_state=42)

# Train a Random Forest classifier
model_tax = RandomForestClassifier(random_state=42)
model_tax.fit(X_train_tax, y_train_tax)

# Evaluate the model
y_pred_tax = model_tax.predict(X_test_tax)
test_accuracy_tax = accuracy_score(y_test_tax, y_pred_tax)
print(f"Test Set Accuracy for Taxonomy Model: {test_accuracy_tax:.4f}")

# Correlation Matrix for Taxonomy Model (only selected features)
correlation_matrix_tax = pd.concat([X_tax, y_tax], axis=1).corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix_tax, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix for Selected Features of Taxonomy Model')
plt.show()

# Confusion Matrix for Taxonomy Model
conf_matrix_tax = confusion_matrix(y_test_tax, y_pred_tax)
disp_tax = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_tax, display_labels=['CWI/BW', 'DW'])
disp_tax.plot(cmap='Blues')
plt.title('Confusion Matrix for Taxonomy Model')
plt.show()

# Save the model
joblib.dump(model_tax, '/Users/adekunle/Downloads/taxonomy_model.pkl')

# 4. Length Model (Updated)
file_path_length = '/mnt/data/Wheat_traits_27_7_2024(perhaps)full.xlsx'
excel_data = pd.ExcelFile(file_path_length)
df = excel_data.parse(excel_data.sheet_names[0])

# Define the length categories based on the given criteria
def length_category(length):
    if length <= 4:
        return "Short"
    elif 4 < length <= 7:
        return "Medium"
    else:
        return "Long"

# Apply the function to create a new column for length categories
df['Length_Category'] = df['Length (mm)'].apply(length_category)

# Select relevant features for the model
features = ['Area (mm2)', 'Width (mm)', 'RatioWidthLength', 'Compactness Circle', 'Compactness Ellipse']
X_length = df[features]
y_length = df['Length_Category']

# Split the data into training and testing sets
X_train_length, X_test_length, y_train_length, y_test_length = train_test_split(X_length, y_length, test_size=0.3, random_state=42)

# Initialize and train the RandomForestClassifier
model_length = RandomForestClassifier(random_state=42)
model_length.fit(X_train_length, y_train_length)

# Predict on the test set
y_pred_length = model_length.predict(X_test_length)

# Generate the confusion matrix
conf_matrix_length = confusion_matrix(y_test_length, y_pred_length)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_length, annot=True, fmt='d', cmap='Blues', xticklabels=model_length.classes_, yticklabels=model_length.classes_)
plt.title('Confusion Matrix for Length Model')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Display the classification report
print("Classification Report:")
classification_report_output = classification_report(y_test_length, y_pred_length)
print(classification_report_output)

# Generate a Feature Importance plot to show the contribution of each feature to the model
importances = model_length.feature_importances_
indices = pd.Series(importances, index=features).sort_values(ascending=False)

# Plotting the feature importance
plt.figure(figsize=(10, 6))
indices.plot(kind='bar')
plt.title('Feature Importance for Length Category Prediction')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()

# Plot distribution of Length within each Length Category
plt.figure(figsize=(10, 6))
sns.boxplot(x='Length_Category', y='Length (mm)', data=df)
plt.title('Distribution of Lengths within Each Category')
plt.xlabel('Length Category')
plt.ylabel('Length (mm)')
plt.show()


