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
    prediction = predict_values(item.description)
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

# Example: adjust these to match your model's architecture
INPUT_SIZE = 3
HIDDEN_SIZE = 64
NUM_LAYERS = 2
OUTPUT_SIZE = 1

model = LSTMModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)
model.load_state_dict(torch.load('models/latency_predictor.pt', map_location=torch.device('cpu')))
model.eval()

def predict_values(text):
    # Expecting comma-separated latency values in text
    try:
        values = [float(x) for x in text.strip().split(',')]
        if (len(values) % 3) != 0:
            return f"Error: Enter Data in multiples of 3"
        arr = np.array(values, dtype=np.float32).reshape(1, -1, 3)  # (batch, seq, input_size)
        tensor = torch.from_numpy(arr)
        with torch.no_grad():
            output = model(tensor)
            prediction = output.item()
        return prediction * 100
    except Exception as e:
        return f"Error: {str(e)}"


