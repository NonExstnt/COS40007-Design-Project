from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import torch
import numpy as np
from xgboost import XGBRegressor

app = FastAPI()

# Mount static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load the LSTM model
class LSTMModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class PositionItem(BaseModel):
    hours: float
    speed: float
    mins:  float
    latency_avg: float

class BitrateItem(BaseModel):
    latitude: float
    longitude: float
    speed: float
    hour: float
    latency_avg: float

@app.get("/")
def index():
    return FileResponse("templates/index.html")

@app.get("/map")
def map_page():
    return FileResponse("static/map.html")

@app.post("/predict_position/")
async def predict_position_endpoint(item: PositionItem):
    latitude, longitude = predict_position(item.hours, item.speed, item.mins, item.latency_avg)
    return {
        "hours": item.hours,
        "speed": item.speed,
        "mins": item.mins,
        "latency_avg": item.latency_avg,
        "latitude": float(latitude),
        "longitude": float(longitude)
    }

# Load XGBoost models for latitude and longitude
xgb_lat = XGBRegressor()
xgb_lat.load_model('models/xgb_latitude_model_best.pt')
xgb_long = XGBRegressor()
xgb_long.load_model('models/xgb_longitude_model_best.pt')

def predict_position(hours, speed, mins, latency):
    X = np.array([[hours, speed, mins, latency]], dtype=np.float32)
    latitude = xgb_lat.predict(X)[0]
    longitude = xgb_long.predict(X)[0]
    return round(latitude, 6), round(longitude, 6)

# Load LSTM model for bitrate prediction
BITRATE_INPUT_SIZE = 5
BITRATE_OUTPUT_SIZE = 1
BITRATE_HIDDEN_SIZE = 64
BITRATE_NUM_LAYERS = 2
bitrate_model = LSTMModel(BITRATE_INPUT_SIZE, BITRATE_HIDDEN_SIZE, BITRATE_NUM_LAYERS, BITRATE_OUTPUT_SIZE)
bitrate_model.load_state_dict(torch.load('models/bitrate_predictor.pt', map_location=torch.device('cpu')))
bitrate_model.eval()

def predict_bitrate(latitude, longitude, speed, hour, latency_avg):
    try:
        arr = np.array([[latitude, longitude, speed, hour, latency_avg]], dtype=np.float32).reshape(1, -1, 5)
        tensor = torch.from_numpy(arr)
        with torch.no_grad():
            output = bitrate_model(tensor)
            prediction = output.item()
        return abs(prediction * 100)
    except Exception as e:
        return f"Error: {str(e)}"

@app.post("/predict_bitrate/")
async def predict_bitrate_endpoint(item: BitrateItem):
    prediction = predict_bitrate(item.latitude, item.longitude, item.speed, item.hour, item.latency_avg)
    if isinstance(prediction, str) and prediction.startswith("Error"):
        return {"error": prediction}
    prediction = "{:.3f}".format(prediction)
    return {
        "latitude": item.latitude,
        "longitude": item.longitude,
        "speed": item.speed,
        "hour": item.hour,
        "latency_avg": item.latency_avg,
        "predicted_bitrate": prediction
    }
