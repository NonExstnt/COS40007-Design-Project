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

class Item(BaseModel):
    description: str

@app.post("/predict/")
async def predict(item: Item):
    prediction = predict_latency(item.description)
    prediction = "{:.3f}".format(prediction)
    return {
            "description": item.description, 
            "predicted_label": prediction
            }

@app.get("/")
def index():
    return FileResponse("templates/index.html")

@app.get("/map")
def map_page():
    return FileResponse("static/map.html")

# Load the LSTM model (adjust class definition as needed)
class LSTMModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

INPUT_SIZE = 3
OUTPUT_SIZE = 1
HIDDEN_SIZE = 64
NUM_LAYERS = 2

latency_model = LSTMModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)
latency_model.load_state_dict(torch.load('models/latency_predictor.pt', map_location=torch.device('cpu')))
latency_model.eval()

def predict_latency(text):
    # Expecting comma-separated latency values in text
    try:
        values = [float(x) for x in text.strip().split(',')]
        if (len(values) % 3) != 0:
            return f"Error: Enter Data in multiples of 3"
        arr = np.array(values, dtype=np.float32).reshape(1, -1, 3)  # (batch, seq, input_size)
        tensor = torch.from_numpy(arr)
        with torch.no_grad():
            output = latency_model(tensor)
            prediction = output.item()
        return prediction * 100
    except Exception as e:
        return f"Error: {str(e)}"

# Load XGBoost models for latitude and longitude
xgb_lat = XGBRegressor()
xgb_lat.load_model('models/xgb_latitude_model.pt')
xgb_long = XGBRegressor()
xgb_long.load_model('models/xgb_longitude_model.pt')

class PositionItem(BaseModel):
    hours: float
    speed: float

@app.post("/predict_position/")
async def predict_position(item: PositionItem):
    latitude, longitude = predict_position_from_xgb(item.hours, item.speed)
    return {
        "hours": item.hours,
        "speed": item.speed,
        "latitude": float(latitude),
        "longitude": float(longitude)
    }

def predict_position_from_xgb(hours, speed):
    X = np.array([[hours, speed]], dtype=np.float32)
    latitude = xgb_lat.predict(X)[0]
    longitude = xgb_long.predict(X)[0]
    return round(latitude, 6), round(longitude, 6)


