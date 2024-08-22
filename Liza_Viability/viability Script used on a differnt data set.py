import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

# Load the new dataset
new_data_file_path = '/Users/adekunle/Downloads/Liza/all_features_blob_data.xlsx'
new_data = pd.read_excel(new_data_file_path)

# Load the trained model
model_file_path = '/Users/adekunle/Downloads/Liza/trained_seed_viability_model.pkl'
model = joblib.load(model_file_path)

# Define the features that were used in the model
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
    *new_data.filter(like="SpectralStdev").columns,
    # Adding reflectance features
    *new_data.filter(like="Reflectance").columns,
    # Adding IHS features
    *new_data.filter(like="IHS").columns,
    # Adding CIE ColorSpace features
    *new_data.filter(like="CIE").columns,
    # Adding Eccentricity features
    *new_data.filter(like="Eccentricity").columns
]

# Preprocess the new dataset in the same way as the training data
X_new = new_data[features]

# Handle missing values by imputing with the median
imputer = SimpleImputer(strategy='median')
X_new_imputed = imputer.fit_transform(X_new)

# Standardize the features
scaler = StandardScaler()
X_new_scaled = scaler.fit_transform(X_new_imputed)

# Predict viability
predictions = model.predict(X_new_scaled)
predicted_viability = ['VIABLE' if pred == 1 else 'NON-VIABLE' for pred in predictions]

# Add the predicted viability to the dataframe
new_data['Predicted Viability'] = predicted_viability

# Prepare the output for Sheet 1
sheet1_output = new_data[['Filename', 'Predicted Viability']]

# Prepare the summary for Sheet 2
summary = sheet1_output.groupby('Filename')['Predicted Viability'].value_counts().unstack(fill_value=0)
summary.columns = ['NON-VIABLE', 'VIABLE']  # Adjusting columns to match the order
summary.reset_index(inplace=True)

# Save the results to an Excel file with two sheets
output_file_path = '/Users/adekunle/Downloads/Liza/viability_predictions.xlsx'
with pd.ExcelWriter(output_file_path) as writer:
    sheet1_output.to_excel(writer, sheet_name='Predicted Viability', index=False)
    summary.to_excel(writer, sheet_name='Summary', index=False)

print(f"Predictions saved to {output_file_path}")
