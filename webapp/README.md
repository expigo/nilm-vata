# NILM VATA Web Application

Modern web dashboard for training NILM models and performing real-time energy disaggregation.

## Features

- **Interactive Model Training**: Train NILM algorithms through a web interface
- **Real-time Disaggregation**: Live WebSocket streaming for real-time power monitoring
- **Multiple Algorithms**: Support for CO, FHMM, kNN, Deep Learning (Seq2Seq, DAE)
- **Multiple Datasets**: REDD, UK-DALE, REFIT, AMPds, ECO, and more
- **Model Management**: Save, load, and manage trained models
- **Live Visualization**: Real-time charts using Chart.js
- **RESTful API**: Complete API for integration with other tools

## Quick Start

### 1. Install Dependencies

```bash
# Install web dependencies
uv pip install fastapi uvicorn websockets

# Or install all extras
uv pip install -e ".[all]"
```

### 2. Run the Server

```bash
# From the project root
cd webapp/backend
python main.py
```

Or using uvicorn directly:

```bash
uvicorn webapp.backend.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Open the Dashboard

Navigate to: http://localhost:8000

- **API Documentation**: http://localhost:8000/docs
- **WebSocket Endpoint**: ws://localhost:8000/ws/disaggregate

## Architecture

```
webapp/
├── backend/
│   └── main.py          # FastAPI application
└── frontend/
    ├── static/
    │   ├── app.js       # Frontend JavaScript
    │   └── style.css    # Dashboard styling
    └── templates/
        └── index.html   # Main dashboard page
```

## API Endpoints

### Core Endpoints

- `GET /` - Web dashboard
- `GET /api/health` - Health check
- `GET /api/algorithms` - List available algorithms
- `GET /api/datasets` - List available datasets
- `GET /api/models` - List saved models

### Training

- `POST /api/train` - Start model training (async)
- `GET /api/train/{task_id}` - Check training status

**Example: Train a Model**

```bash
curl -X POST http://localhost:8000/api/train \
  -H "Content-Type: application/json" \
  -d '{
    "algorithm": "co",
    "dataset": "synthetic",
    "building_id": 1,
    "train_ratio": 0.7
  }'
```

### Prediction

- `POST /api/predict` - Make predictions with a trained model

**Example: Get Predictions**

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "CO_model_20250118_123456.pkl",
    "power_values": [1500, 2000, 1800, 2200, 1900]
  }'
```

### Real-time Streaming

- `WebSocket /ws/disaggregate` - Real-time disaggregation stream

**Example: WebSocket Client**

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/disaggregate');

ws.onopen = () => {
  // Send model configuration
  ws.send(JSON.stringify({ model_id: 'your_model_id' }));

  // Send power readings
  ws.send(JSON.stringify({ power: 1500 }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Disaggregated:', data.appliances);
};
```

## Dashboard Usage

### Training a Model

1. **Select Algorithm**: Choose from CO, FHMM, kNN, or deep learning options
2. **Select Dataset**: Pick from available datasets (synthetic, REDD, UK-DALE, etc.)
3. **Configure Parameters**: Set building ID and train/test split ratio
4. **Start Training**: Click "Start Training" and monitor progress
5. **View Results**: Training shows progress bar and final F1 score

### Real-time Disaggregation

1. **Select Model**: Choose a trained model from the dropdown
2. **Start Stream**: Click "Start Stream" to begin real-time monitoring
3. **View Charts**:
   - Top chart: Total aggregate power consumption
   - Bottom chart: Disaggregated appliance-level power
4. **Monitor Stats**: Real-time power statistics for each appliance
5. **Stop Stream**: Click "Stop Stream" when done

## Configuration

### Environment Variables

```bash
# Server configuration
export NILM_HOST="0.0.0.0"
export NILM_PORT="8000"

# Model storage path
export NILM_MODELS_DIR="./saved_models"

# Dataset path
export NILM_DATA_DIR="./data"
```

### Production Deployment

#### Using Gunicorn

```bash
gunicorn webapp.backend.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

#### Using Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . .

RUN pip install -e ".[all]"

EXPOSE 8000

CMD ["uvicorn", "webapp.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -t nilm-vata .
docker run -p 8000:8000 -v $(pwd)/saved_models:/app/saved_models nilm-vata
```

## Development

### Enable Hot Reload

```bash
uvicorn webapp.backend.main:app --reload
```

### Frontend Development

The frontend is pure JavaScript (no build step required). Just edit the files:

- `webapp/frontend/static/app.js` - Application logic
- `webapp/frontend/static/style.css` - Styling
- `webapp/frontend/templates/index.html` - HTML structure

Changes are immediately reflected on page refresh.

## Troubleshooting

### Port Already in Use

```bash
# Find and kill process on port 8000
lsof -ti:8000 | xargs kill -9
```

### WebSocket Connection Failed

- Check that the server is running
- Ensure firewall allows WebSocket connections
- Verify the WebSocket URL matches your server address

### Model Not Found

- Ensure models are saved in the `saved_models/` directory
- Check that model files have correct permissions
- Verify model ID matches exactly (case-sensitive)

### Dataset Not Available

- Download required datasets to `./data/` directory
- Check dataset structure matches expected format
- Use synthetic dataset for testing without downloads

## API Integration Examples

### Python Client

```python
import requests
import websockets
import asyncio

# Train a model
response = requests.post('http://localhost:8000/api/train', json={
    'algorithm': 'co',
    'dataset': 'synthetic',
    'building_id': 1,
    'train_ratio': 0.7
})
task_id = response.json()['task_id']

# Check training status
status = requests.get(f'http://localhost:8000/api/train/{task_id}').json()
print(f"Status: {status['status']}, Progress: {status['progress']}%")

# Real-time streaming
async def stream_disaggregation():
    uri = "ws://localhost:8000/ws/disaggregate"
    async with websockets.connect(uri) as websocket:
        await websocket.send('{"model_id": "your_model_id"}')

        # Send power readings
        await websocket.send('{"power": 1500}')

        # Receive disaggregated data
        response = await websocket.recv()
        print(f"Received: {response}")

asyncio.run(stream_disaggregation())
```

### JavaScript Client

```javascript
// Train model
fetch('http://localhost:8000/api/train', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    algorithm: 'co',
    dataset: 'synthetic',
    building_id: 1,
    train_ratio: 0.7
  })
})
.then(res => res.json())
.then(data => console.log('Training started:', data.task_id));
```

## Performance

- **Model Training**: Background tasks ensure non-blocking training
- **WebSocket Streaming**: Handles 100+ messages/second
- **Chart Updates**: Optimized for smooth 60 FPS rendering
- **Model Caching**: Loaded models cached in memory for fast predictions

## Security Considerations

For production deployments:

1. **Enable HTTPS**: Use SSL/TLS certificates
2. **Authentication**: Add API key or OAuth2 authentication
3. **Rate Limiting**: Prevent abuse with rate limits
4. **CORS**: Restrict allowed origins (currently set to "*")
5. **Input Validation**: All inputs are validated with Pydantic

## Future Enhancements

- [ ] User authentication and sessions
- [ ] Model comparison dashboard
- [ ] Export predictions to CSV/JSON
- [ ] Scheduled training jobs
- [ ] Multi-model ensemble predictions
- [ ] Mobile-responsive design improvements
- [ ] Database persistence for models and results

## Support

For issues and questions:
- GitHub Issues: https://github.com/your-repo/nilm-vata/issues
- Documentation: See main README.md
- API Docs: http://localhost:8000/docs

## License

Same as main NILM VATA project.
