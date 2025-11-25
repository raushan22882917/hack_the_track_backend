# Telemetry Rush - Backend API Server

A high-performance FastAPI backend server for real-time race telemetry data processing, analysis, and visualization. This unified backend consolidates all services into a single application running on port 8000 (or configured port).

## 🚀 Features

### Core Functionality
- **Real-time Telemetry Processing**: Process and broadcast vehicle telemetry data in real-time
- **Race Lifecycle Management**: Automatic race start, pause, resume, and finish detection
- **Multi-Vehicle Support**: Handle multiple vehicles simultaneously with parallel processing
- **Performance Optimized**: Parallel data loading, caching, and efficient file I/O
- **REST API**: Pure REST API with polling endpoints (no WebSocket complexity)

### Data Services
- **Telemetry Data**: GPS coordinates, speed, throttle, brake, gear, and more
- **Endurance Data**: Lap times, section times, and endurance race metrics
- **Leaderboard**: Real-time race positions, gaps, and lap information
- **Weather Data**: Track conditions, temperature, humidity, wind, and rain

### Analysis & Insights
- **Driver Insights**: Racing line comparison, braking analysis, cornering analysis
- **AI-Powered Analysis**: Gemini AI integration for driver improvement suggestions
- **Predictive Models**: Lap time prediction, performance trajectory, optimal racing line
- **Post-Event Analysis**: Comprehensive race story and performance comparison
- **Sector Analysis**: Detailed sector-by-sector performance breakdown

### Performance Features
- **Parallel Data Loading**: All data sources load simultaneously on startup
- **Intelligent Caching**: Vehicle lists and frequently accessed data cached
- **Optimized File I/O**: Fast CSV reading using parallel processing
- **Combined Initialization**: Single `/api/init` endpoint loads all data in parallel

## 📋 Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
- [API Documentation](#api-documentation)
- [Race Lifecycle Management](#race-lifecycle-management)
- [Performance Optimizations](#performance-optimizations)
- [Usage Examples](#usage-examples)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)

## 🛠️ Installation

### Prerequisites
- Python 3.11 or higher
- pip package manager
- (Optional) Docker for containerized deployment

### Local Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd hack_the_track_backend
   ```

2. **Create virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables (optional)**
   ```bash
   # For AI insights (optional)
   export GEMINI_API_KEY=your_api_key_here
   ```

5. **Prepare data files**
   - Place vehicle CSV files in `logs/vehicles/` directory
   - Place `R1_leaderboard.csv` in `logs/` directory
   - Place `R1_section_endurance.csv` in `logs/` directory
   - Place `R1_weather_data.csv` in `logs/` directory

6. **Start the server**
   ```bash
   # Using the startup script
   ./start.sh
   
   # Or directly with uvicorn
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

The server will start on `http://localhost:8000` by default.

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `PORT` | Server port number | `8000` | No |
| `GEMINI_API_KEY` | Google Gemini API key for AI insights | None | No |

### Data Directory Structure

```
logs/
├── vehicles/
│   ├── GR86-002-000.csv
│   ├── GR86-004-78.csv
│   └── ...
├── R1_leaderboard.csv
├── R1_section_endurance.csv
└── R1_weather_data.csv
```

### Port Configuration

The server uses port 8000 by default, but can be configured:
- **Environment variable**: `PORT=8080`
- **Command line**: `uvicorn main:app --port 8080`
- **Cloud Run**: Automatically uses `$PORT` environment variable

## 📚 API Documentation

### Base URL
```
http://localhost:8000
```

### Interactive API Docs
Once the server is running, visit:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Core Endpoints

#### Health Check
```http
GET /api/health
```
Returns server health status and data loading state.

**Response:**
```json
{
  "status": "healthy",
  "telemetry_loaded": true,
  "endurance_loaded": true,
  "leaderboard_loaded": true
}
```

#### Initialize Race (Combined Endpoint)
```http
GET /api/init?event_name=optional
```
**Optimized endpoint** that loads all race data in parallel. Use this for fast frontend initialization.

**Response:**
```json
{
  "vehicles": {...},
  "telemetry": {...},
  "endurance": {...},
  "leaderboard": {...},
  "event_name": null,
  "initialized_at": "2024-01-01T12:00:00"
}
```

#### Get Vehicles
```http
GET /api/vehicles?event_name=optional
```
Returns list of all available vehicles with driver numbers and metadata.

**Response:**
```json
{
  "vehicles": [
    {
      "id": "GR86-002-000",
      "name": "GR86-002-000",
      "file": "GR86-002-000.csv",
      "vehicle_number": 2,
      "car_number": 2,
      "driver_number": 1,
      "has_endurance_data": true
    }
  ],
  "count": 20,
  "event_name": null
}
```

**Performance**: Cached after first load - returns in <100ms

#### Get Telemetry Data
```http
GET /api/telemetry
```
Returns latest telemetry data frame. Poll this endpoint for real-time updates.

**Response:**
```json
{
  "type": "telemetry_frame",
  "timestamp": "2024-01-01T12:00:00",
  "vehicles": {
    "GR86-002-000": {
      "gps_lat": 33.1234,
      "gps_lon": -86.5678,
      "speed": 120.5,
      "lap": 5,
      ...
    }
  },
  "weather": {
    "air_temp": 25.0,
    "track_temp": 30.0,
    ...
  }
}
```

#### Get Endurance Data
```http
GET /api/endurance
```
Returns endurance/lap event data.

#### Get Leaderboard
```http
GET /api/leaderboard
```
Returns current race leaderboard with positions, gaps, and lap times.

### Race Control Endpoints

#### Control Playback
```http
POST /api/control
Content-Type: application/json

{
  "cmd": "start" | "play" | "pause" | "stop" | "restart" | "speed" | "seek"
}
```

**Commands:**
- `start` / `play`: Start or resume race playback
- `pause`: Pause race playback
- `stop`: Stop race completely
- `restart`: Restart race from beginning
- `speed`: Set playback speed (requires `"value": 1.5`)
- `seek`: Seek to specific timestamp (requires `"timestamp": "2024-01-01T12:00:00"`)

**Response:**
```json
{
  "status": "success",
  "command": "play",
  "race_state": {
    "paused": false,
    "finished": false,
    "playing": true
  }
}
```

### Analysis Endpoints

#### Driver Insights
```http
GET /api/driver/{vehicle_id}/racing-line?lap1=5&lap2=10
GET /api/driver/{vehicle_id}/braking?lap1=5&lap2=10
GET /api/driver/{vehicle_id}/cornering?lap1=5&lap2=10
GET /api/driver/{vehicle_id}/ai-insights
```

#### Predictive Analysis
```http
POST /api/predictive/simulate-stint
POST /api/predictive/predict-new-session
GET /api/driver/{vehicle_id}/performance-prediction?future_laps=5
```

#### Post-Event Analysis
```http
POST /api/analysis/post-event
Content-Type: application/json

{
  "track_name": "Barber",
  "race_session": "R1",
  "min_lap_time": 25.0
}
```

### Data Preprocessing

#### Preprocess Telemetry
```http
POST /api/preprocess
Content-Type: application/json

{
  "input_file": "path/to/input.csv",
  "output_dir": "logs/vehicles"
}
```

## 🏁 Race Lifecycle Management

The backend implements intelligent race lifecycle management with automatic state handling.

### Race States

1. **Ready**: Data loaded, ready to start
2. **Playing**: Race is actively playing
3. **Paused**: Race is paused (can resume)
4. **Finished**: Race completed (auto-stopped)
5. **Stopped**: Race stopped manually

### Race Flow

```
[Ready] → [Start/Play] → [Playing] → [Race Finishes] → [Auto-Stop] → [Finished]
                ↓                              ↓
            [Pause] ← [Resume] ← [Restart] ← [Restart]
```

### Auto-Stop Behavior

When the race reaches the end of telemetry data:
- Playback automatically stops
- `race_finished` flag is set to `true`
- All broadcasts are stopped
- Frontend receives `race_finished` status

### Control Commands

| Command | When Race Finished | Action |
|---------|-------------------|--------|
| `start` / `play` | ✅ Yes | Resets and starts new race |
| `start` / `play` | ❌ No | Starts/resumes race |
| `pause` | ✅ Yes | No effect (already stopped) |
| `pause` | ❌ No | Pauses race |
| `restart` | ✅/❌ | Always resets and allows new start |
| `stop` | ✅/❌ | Stops race completely |

## ⚡ Performance Optimizations

### Parallel Data Loading

All data sources load simultaneously on startup:
- Telemetry data
- Endurance data
- Leaderboard data
- Predictive models

**Before**: Sequential loading (~10-15 seconds)
**After**: Parallel loading (~3-5 seconds)

### Vehicle List Caching

- First request: ~200-500ms (parallel file reading)
- Subsequent requests: <100ms (cached)
- Cache automatically cleared when new data is loaded

### Optimized File I/O

- **Parallel CSV Reading**: Uses `ThreadPoolExecutor` with 10 workers
- **Fast Metadata Extraction**: Uses Python's `csv` module instead of pandas
- **Efficient Binary Search**: O(log n) timestamp lookups

### Combined Initialization Endpoint

Use `/api/init` for fast frontend initialization:
- Loads all data in parallel
- Single request instead of multiple sequential calls
- Returns complete race state in one response

## 💡 Usage Examples

### Frontend Initialization

```javascript
// Fast initialization - loads everything in parallel
const response = await fetch('/api/init');
const data = await response.json();

// Access all data
const vehicles = data.vehicles.vehicles;
const telemetryStatus = data.telemetry;
const enduranceStatus = data.endurance;
const leaderboardStatus = data.leaderboard;
```

### Start Race

```javascript
// Start race
const response = await fetch('/api/control', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ cmd: 'start' })
});

const result = await response.json();
if (result.race_state.playing) {
  console.log('Race started!');
}
```

### Poll for Telemetry Updates

```javascript
// Poll every 100ms for real-time updates
setInterval(async () => {
  const response = await fetch('/api/telemetry');
  const data = await response.json();
  
  if (data.type === 'telemetry_frame') {
    updateMap(data.vehicles);
    updateLeaderboard(data);
  } else if (data.type === 'race_finished') {
    console.log('Race finished!');
    stopPolling();
  }
}, 100);
```

### Pause/Resume Race

```javascript
// Pause
await fetch('/api/control', {
  method: 'POST',
  body: JSON.stringify({ cmd: 'pause' })
});

// Resume
await fetch('/api/control', {
  method: 'POST',
  body: JSON.stringify({ cmd: 'play' })
});
```

### Restart After Race Finishes

```javascript
// Check if race finished
const telemetryResponse = await fetch('/api/telemetry');
const telemetryData = await telemetryResponse.json();

if (telemetryData.race_finished) {
  // Restart race
  await fetch('/api/control', {
    method: 'POST',
    body: JSON.stringify({ cmd: 'restart' })
  });
  
  // Start again
  await fetch('/api/control', {
    method: 'POST',
    body: JSON.stringify({ cmd: 'start' })
  });
}
```

## 🚢 Deployment

### Docker Deployment

1. **Build Docker image**
   ```bash
   docker build -t telemetry-rush-backend .
   ```

2. **Run container**
   ```bash
   docker run -p 8000:8000 \
     -e PORT=8000 \
     -e GEMINI_API_KEY=your_key \
     -v $(pwd)/logs:/app/logs \
     telemetry-rush-backend
   ```

### Cloud Run Deployment

The project includes Cloud Run configuration:

1. **Build and push to Google Container Registry**
   ```bash
   gcloud builds submit --tag gcr.io/PROJECT_ID/telemetry-rush-backend
   ```

2. **Deploy to Cloud Run**
   ```bash
   gcloud run deploy telemetry-rush-backend \
     --image gcr.io/PROJECT_ID/telemetry-rush-backend \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --set-env-vars PORT=8080
   ```

### Environment-Specific Configuration

- **Local Development**: Port 8000
- **Cloud Run**: Uses `$PORT` environment variable (defaults to 8080)
- **Docker**: Configurable via `-e PORT=8000`

## 🔧 Troubleshooting

### Common Issues

#### Data Not Loading
**Problem**: Endpoints return empty data or "loading" status

**Solutions**:
1. Check that CSV files exist in `logs/vehicles/` directory
2. Verify file permissions are correct
3. Check server logs for loading errors
4. Wait for background loading to complete (check `/api/health`)

#### Slow Vehicle List Loading
**Problem**: `/api/vehicles` takes too long

**Solutions**:
1. Ensure you're using the optimized version (should be <500ms)
2. Check that vehicle CSV files aren't corrupted
3. Clear cache: `POST /api/vehicles/clear-cache`
4. Verify parallel processing is working (check logs)

#### Race Not Starting
**Problem**: Control commands don't start playback

**Solutions**:
1. Verify telemetry data is loaded: `GET /api/health`
2. Check if race already finished: `GET /api/telemetry` (check `race_finished`)
3. Use `restart` command if race finished
4. Check server logs for errors

#### Port Already in Use
**Problem**: Server won't start on port 8000

**Solutions**:
1. Change port: `PORT=8080 uvicorn main:app --port 8080`
2. Kill existing process: `lsof -ti:8000 | xargs kill`
3. Use different port in configuration

### Performance Issues

#### Slow Startup
- Normal: First startup loads all data (~3-5 seconds)
- Data loads in background - server starts immediately
- Check `/api/health` to see loading status

#### High Memory Usage
- Normal: Large telemetry datasets require memory
- Consider reducing data size or using data sampling
- Monitor with `htop` or similar tools

### Debug Mode

Enable verbose logging:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --log-level debug
```

## 📊 API Response Times

| Endpoint | First Request | Cached Request |
|----------|--------------|----------------|
| `/api/init` | 200-500ms | 50-100ms |
| `/api/vehicles` | 200-500ms | <100ms |
| `/api/telemetry` | <50ms | <50ms |
| `/api/control` | <50ms | <50ms |
| `/api/leaderboard` | <100ms | <100ms |

## 🔐 Security Considerations

- CORS is configured for frontend access
- No authentication required (add if needed for production)
- File paths are validated to prevent directory traversal
- Input validation on all endpoints

## 📝 License

[Add your license information here]

## 🤝 Contributing

[Add contribution guidelines here]

## 📞 Support

For issues or questions:
- Check the troubleshooting section
- Review server logs
- Check API documentation at `/docs`

## 🎯 Key Features Summary

✅ **Fast Initialization**: Combined `/api/init` endpoint loads everything in parallel  
✅ **Intelligent Caching**: Vehicle lists cached for instant access  
✅ **Race Lifecycle**: Automatic start, pause, resume, and finish detection  
✅ **Performance Optimized**: Parallel processing throughout  
✅ **Real-time Updates**: Poll-based REST API for telemetry data  
✅ **Comprehensive Analysis**: Driver insights, AI analysis, predictive models  
✅ **Production Ready**: Docker and Cloud Run deployment support  

---

**Built with FastAPI, Python, and optimized for performance** 🚀

