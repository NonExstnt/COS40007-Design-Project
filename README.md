# 5G Predictor Website

This web application provides two main features:
- **5G Bitrate Predictor**: Predicts the expected 5G bitrate at a given location and time.
- **5G Position Predictor**: Predicts the likely position (latitude, longitude) based on network conditions and time.

## How It Was Developed

The website was developed using [FastAPI](https://fastapi.tiangolo.com/) for the backend, with machine learning models trained on 5G network data. The backend uses:
- An LSTM (Long Short-Term Memory) neural network (PyTorch) for bitrate prediction.
- XGBoost regression models for latitude and longitude prediction.

The models were trained offline using historical 5G network measurements, including features such as location, speed, time, and average latency. The trained models are loaded at runtime to provide predictions via API endpoints.

## Input Required

### Bitrate Prediction (`/predict_bitrate/`)
- **latitude**: float — Current latitude.
- **longitude**: float — Current longitude.
- **speed**: float — Current speed (e.g., in km/h).
- **hour**: float — Hour of the day (0-23).
- **latency_avg**: float — Average network latency (ms).

Example request:
```json
{
  "latitude": -37.8136,
  "longitude": 144.9631,
  "speed": 50.0,
  "hour": 14,
  "latency_avg": 30.5
}
```

### Position Prediction (`/predict_position/`)
- **hours**: float — Hours since start or a reference point.
- **speed**: float — Current speed.
- **mins**: float — Minutes since start or a reference point.
- **latency_avg**: float — Average network latency (ms).

Example request:
```json
{
  "hours": 2.5,
  "speed": 60.0,
  "mins": 150,
  "latency_avg": 28.0
}
```

## Outcome Produced

- **Bitrate Prediction**:  
  Returns the predicted 5G bitrate (as a string, rounded to three decimal places) for the given location, time, speed, and latency.

  Example response:
  ```json
  {
    "latitude": -37.8136,
    "longitude": 144.9631,
    "speed": 50.0,
    "hour": 14,
    "latency_avg": 30.5,
    "predicted_bitrate": "85.432"
  }
  ```

- **Position Prediction**:  
  Returns the predicted latitude and longitude based on the provided network and time features.

  Example response:
  ```json
  {
    "hours": 2.5,
    "speed": 60.0,
    "mins": 150,
    "latency_avg": 28.0,
    "latitude": -37.812345,
    "longitude": 144.965432
  }
  ```

## Prerequisites

- Python 3.8+
- [pip](https://pip.pypa.io/en/stable/)
- PyTorch
- XGBoost

## Installation

1. **Clone the repository** (if not already done):  
   ```bash
   git clone https://github.com/nonexstnt/COS40007-Design-Project.git
   cd COS40007-Design-Project/website
   ```

2. **Install dependencies**:  
   It is recommended to use a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Ensure model files are present**:  
   Place the following files in the `models/` directory if not already present:
   - `bitrate_predictor.pt`
   - `xgb_latitude_model_best.pt`
   - `xgb_longitude_model_best.pt`

## Running the Website

Start the FastAPI server using Uvicorn:

```bash
uvicorn main:app --reload
```

- The website will be available at [http://127.0.0.1:8000/](http://127.0.0.1:8000/).
- The main page is served from `templates/index.html`.

## File Structure

- `main.py` - FastAPI application entry point.
- `requirements.txt` - Python dependencies.
- `templates/index.html` - Main HTML page.
- `static/` - Static files (CSS, JS, images, etc).
- `models/` - Directory for ML model files.

## API Endpoints

- `POST /predict_bitrate/`  
  Request JSON:  
  ```json
  {
    "latitude": float,
    "longitude": float,
    "speed": float,
    "hour": float,
    "latency_avg": float
  }
  ```
  Response:  
  ```json
  {
    "latitude": ...,
    "longitude": ...,
    "speed": ...,
    "hour": ...,
    "latency_avg": ...,
    "predicted_bitrate": "..."
  }
  ```

- `POST /predict_position/`  
  Request JSON:  
  ```json
  {
    "hours": float,
    "speed": float,
    "mins": float,
    "latency_avg": float
  }
  ```
  Response:  
  ```json
  {
    "hours": ...,
    "speed": ...,
    "mins": ...,
    "latency_avg": ...,
    "latitude": ...,
    "longitude": ...
  }
  ```

## Notes

- If you encounter errors loading models, ensure the model files are present and compatible with your Python and library versions.

---