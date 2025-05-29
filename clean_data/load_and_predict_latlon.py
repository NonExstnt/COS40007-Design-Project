import pandas as pd
import numpy as np
from xgboost import XGBRegressor

def load_xgb_model(model_path):
    model = XGBRegressor()
    model.load_model(model_path)
    return model

# Load test data (assuming same split as before)
df = pd.read_csv('Master5G.csv')
X = df[['hour', 'speed']]
y_long = df['longitude']
y_lat = df['latitude']

from sklearn.model_selection import train_test_split
X_train, X_test, y_long_train, y_long_test = train_test_split(X, y_long, test_size=0.2, random_state=42)
_, _, y_lat_train, y_lat_test = train_test_split(X, y_lat, test_size=0.2, random_state=42)

# Load models
xgb_long = load_xgb_model('xgb_longitude_model.pt')
xgb_lat = load_xgb_model('xgb_latitude_model.pt')

# Predict
y_long_pred = xgb_long.predict(X_test)
y_lat_pred = xgb_lat.predict(X_test)

# Save results
test_results = pd.DataFrame({
    'longitude_true': np.array(y_long_test),
    'longitude_pred': y_long_pred,
    'latitude_true': np.array(y_lat_test),
    'latitude_pred': y_lat_pred
})
test_results.to_csv('latlon_predictions_vs_true_from_loaded.csv', index=False)
print('Saved predictions to latlon_predictions_vs_true_from_loaded.csv')
