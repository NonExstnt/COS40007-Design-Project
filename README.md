# 5G Predictor Website

This web application provides two main features:
- **5G Bitrate Predictor**: Predicts the expected 5G bitrate at a given location and time.
- **5G Position Predictor**: Predicts the likely position (latitude, longitude) based on network conditions and time.

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