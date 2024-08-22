import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np

# Load the dataset
file_path = '/Users/adekunle/Downloads/Liza/wheat_data_cleaned R.xlsx'
data = pd.read_excel(file_path)

# Select relevant columns
features = [
    "DirectionalTexture [1]",
    "FormFactor",
    "MultiAbsorbanceMean [18]",
    "MultiAbsorbanceMean [17]",
    "Compactness Circle",
    "Compactness",
    "FourierShapeDescriptor [1]",
    "RatioWidthLength",
    "CIELabMeanProfile [11]",
    "BlobShapeRegularityStatistics1 [1]",
    # Adding spectral features
    *data.filter(like="SpectralStdev").columns,
    # Adding reflectance features
    *data.filter(like="Reflectance").columns,
    # Adding IHS features
    *data.filter(like="IHS").columns,
    # Adding CIE ColorSpace features
    *data.filter(like="CIE").columns,
    # Adding Eccentricity features
    *data.filter(like="Eccentricity").columns
]

# Extract the relevant features and the target variable
X = data[features]
y = LabelEncoder().fit_transform(data['Reference Classes'] == "VIABLE")  # Binary target variable

# Handle missing values by imputing with the median
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Combine the scaled features with the target variable for model training
final_data = pd.DataFrame(X_scaled, columns=features)
final_data['ReferenceClasses'] = y

# Split the data into training and testing sets
train_data, test_data = train_test_split(final_data, test_size=0.3, random_state=42, stratify=final_data['ReferenceClasses'])

# Set up the Random Forest model
rf_model = RandomForestClassifier(random_state=42)
# Set up the parameter grid for random search
param_distributions = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [10, 20, 30, 40, 50, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Set up control for random search
random_search = RandomizedSearchCV(
    rf_model,
    param_distributions=param_distributions,
    n_iter=10,
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1,
    scoring='accuracy'
)

# Train the model with random search
random_search.fit(train_data.drop(columns=['ReferenceClasses']), train_data['ReferenceClasses'])

# Make predictions on the test set
predictions = random_search.predict(test_data.drop(columns=['ReferenceClasses']))

# Evaluate the model
conf_matrix = confusion_matrix(test_data['ReferenceClasses'], predictions)
accuracy = accuracy_score(test_data['ReferenceClasses'], predictions)

# Calculate the 95% confidence interval for the accuracy
conf_interval = accuracy + np.array([-1, 1]) * 1.96 * np.sqrt((accuracy * (1 - accuracy)) / len(test_data))

# Print the results
print("Best Parameters:", random_search.best_params_)
print("Confusion Matrix:\n", conf_matrix)
print("Accuracy:", accuracy)
print("95% Confidence Interval:", conf_interval)

# Save the trained model to a file
import joblib
joblib.dump(random_search.best_estimator_, '/Users/adekunle/Downloads/Liza/trained_seed_viability_model.pkl')
