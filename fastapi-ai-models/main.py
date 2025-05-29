from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse
import torch
import numpy as np

app = FastAPI()

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
    return FileResponse("index.html")

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

POS_INPUT_SIZE = 2
POS_OUTPUT_SIZE = 2

LON_POS_HIDDEN_SIZE = 64
LON_POS_NUM_LAYERS = 1

LAT_POS_HIDDEN_SIZE = 32
LAT_POS_NUM_LAYERS = 2

# Example: adjust these to match your model's architecture
LATENCY_HIDDEN_SIZE = 64
LATENCY_NUM_LAYERS = 2

latency_model = LSTMModel(INPUT_SIZE, LATENCY_HIDDEN_SIZE, LATENCY_NUM_LAYERS, OUTPUT_SIZE)
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

class PositionItem(BaseModel):
    hours: float
    speed: float

@app.post("/predict_position/")
async def predict_position(item: PositionItem):
    latitude, longitude = predict_position_from_models(item.hours, item.speed)
    return {
        "hours": item.hours,
        "speed": item.speed,
        "latitude": latitude,
        "longitude": longitude
    }

# Load the LSTM models for latitude and longitude
lat_model = LSTMModel(POS_INPUT_SIZE, LAT_POS_HIDDEN_SIZE, LAT_POS_NUM_LAYERS, POS_OUTPUT_SIZE)
lat_model.load_state_dict(torch.load('models/latitude_predictor.pt', map_location=torch.device('cpu')))
lat_model.eval()

lon_model = LSTMModel(POS_INPUT_SIZE, LON_POS_HIDDEN_SIZE, LON_POS_NUM_LAYERS, POS_OUTPUT_SIZE)
lon_model.load_state_dict(torch.load('models/longitude_predictor.pt', map_location=torch.device('cpu')))
lon_model.eval()

def predict_position_from_models(hours, speed):
    # For demonstration, we use dummy altitude=0.0 as third input
    arr = np.array([[hours, speed]], dtype=np.float32).reshape(1, -1, 2)
    tensor = torch.from_numpy(arr)
    with torch.no_grad():
        lat_out = lat_model(tensor)
        lon_out = lon_model(tensor)
        latitude = lat_out[0, 0].item()
        longitude = lon_out[0, 1].item()
    return round(latitude, 6), round(longitude, 6)


