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


