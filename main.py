"""
Telemetry Rush - Unified Backend Server
All services consolidated into one FastAPI application running on port 8000

Features:
- REST API endpoints for controls and data access
- Telemetry, Endurance, and Leaderboard services
- Telemetry preprocessing (converts raw telemetry data to per-vehicle CSVs)
- Real-time data via polling REST endpoints
- All running on a single port (8000)
"""

from fastapi import FastAPI, HTTPException, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.encoders import jsonable_encoder
from starlette.middleware.base import BaseHTTPMiddleware
import asyncio
import json
from json import JSONEncoder
from typing import Dict, List, Optional
from datetime import datetime
import uvicorn
import pandas as pd
import numpy as np
from collections import defaultdict
from dateutil import parser as dtparser
import glob
import os
from pathlib import Path
import logging
import sys
import warnings
import bisect
import concurrent.futures
import csv
from threading import Lock

# Analysis endpoints
from analysis_endpoints import get_race_story, get_sector_comparison, get_driver_insights
from post_event_analysis import generate_post_event_analysis
from driver_insights import (
    get_racing_line_comparison,
    get_braking_analysis,
    get_corner_analysis,
    get_driver_improvement_opportunities,
    get_speed_trace_comparison,
    get_best_worst_lap_analysis
)
from ai_insights import (
    get_ai_driver_insights,
    get_sector_ai_analysis
)
from predictive_models import (
    predict_performance_trajectory,
    calculate_optimal_racing_line,
    generate_training_plan,
    predict_sector_performance
)
from realtime_analytics import (
    calculate_real_time_gaps,
    analyze_pit_window,
    simulate_strategy_scenario,
    get_tire_degradation_estimate
)

# Suppress uvicorn warnings
logging.getLogger("uvicorn.error").setLevel(logging.ERROR)
logging.getLogger("fastapi").setLevel(logging.WARNING)

class CustomJSONEncoder(JSONEncoder):
    """Custom JSON encoder that handles NaN and numpy types"""
    def encode(self, obj):
        return super().encode(self._clean(obj))
    
    def _clean(self, obj):
        if isinstance(obj, dict):
            return {k: self._clean(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clean(item) for item in obj]
        elif isinstance(obj, (float, np.floating)):
            if pd.isna(obj) or np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        elif isinstance(obj, (int, np.integer)):
            return int(obj)
        elif isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        elif pd.isna(obj):
            return None
        return obj

def clean_nan_values(obj):
    """Recursively remove NaN values and convert numpy types for JSON serialization"""
    if isinstance(obj, dict):
        cleaned = {}
        for k, v in obj.items():
            cleaned_val = clean_nan_values(v)
            cleaned[k] = cleaned_val
        return cleaned
    elif isinstance(obj, list):
        return [clean_nan_values(item) for item in obj]
    elif isinstance(obj, (float, np.floating)):
        if pd.isna(obj) or np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, (int, np.integer)):
        return int(obj)
    elif isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    elif pd.isna(obj):
        return None
    return obj

# Suppress Windows asyncio ProactorEventLoop socket shutdown warnings
if sys.platform == 'win32':
    # Suppress the specific asyncio ProactorEventLoop socket shutdown error
    warnings.filterwarnings('ignore', category=RuntimeWarning, module='asyncio')
    
    # Also suppress the exception in the event loop callback
    def suppress_proactor_error():
        import asyncio
        original_call_connection_lost = None
        
        try:
            from asyncio.proactor_events import _ProactorBasePipeTransport
            original_call_connection_lost = _ProactorBasePipeTransport._call_connection_lost
            
            def patched_call_connection_lost(self, exc):
                try:
                    return original_call_connection_lost(self, exc)
                except (OSError, AttributeError):
                    # Silently ignore socket shutdown errors on Windows
                    pass
            
            _ProactorBasePipeTransport._call_connection_lost = patched_call_connection_lost
        except (ImportError, AttributeError):
            pass
    
    suppress_proactor_error()

app = FastAPI(title="Telemetry Rush API", version="2.0.0")

# Add custom middleware FIRST to ensure CORS headers are always present
# FastAPI applies middleware in reverse order (last added = first executed)
class CORSHeaderMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        # Handle OPTIONS preflight requests immediately
        if request.method == "OPTIONS":
            origin = request.headers.get("origin", "*")
            response = Response(
                status_code=200,
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, PATCH",
                    "Access-Control-Allow-Headers": "*",
                    "Access-Control-Max-Age": "3600",
                }
            )
            print(f"✅ Handled OPTIONS preflight for {request.url.path} from origin {origin}")
            return response
        
        # Process the request
        try:
            response = await call_next(request)
        except Exception as e:
            # Even on exceptions, ensure CORS headers are present
            print(f"⚠️ Exception in request handler: {e}")
            response = JSONResponse(
                status_code=500,
                content={"detail": str(e)},
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, PATCH",
                    "Access-Control-Allow-Headers": "*",
                }
            )
            return response
        
        # Ensure CORS headers are always present on all responses
        origin = request.headers.get("origin", "*")
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
        response.headers["Access-Control-Allow-Headers"] = "*"
        
        # Log for debugging (only for API endpoints to avoid spam)
        if "/api/" in str(request.url.path):
            print(f"✅ Added CORS headers to {request.method} {request.url.path} (origin: {origin})")
        
        return response

# Add custom middleware first (will be executed last)
app.add_middleware(CORSHeaderMiddleware)

# CORS middleware - must be added after custom middleware
# Configure CORS to allow all origins for development and production
# Note: When allow_credentials=True, we cannot use allow_origins=["*"]
# So we set allow_credentials=False to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=False,  # Must be False when using "*" for origins
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],  # Explicitly list methods
    allow_headers=["*"],  # Allow all headers
    expose_headers=["*"],  # Expose all headers
    max_age=3600,  # Cache preflight requests for 1 hour
)

# Exception handler to ensure CORS headers are added to error responses
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Ensure CORS headers are added to HTTP error responses"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Ensure CORS headers are added to error responses"""
    import traceback
    error_detail = str(exc)
    print(f"⚠️ Unhandled exception: {error_detail}")
    traceback.print_exc()
    
    return JSONResponse(
        status_code=500,
        content={"detail": error_detail, "error": "Internal server error"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )

# Connection tracking removed - using pure REST API with polling

# Data cache for REST API
telemetry_cache: Dict = {}
endurance_cache: List = []
leaderboard_cache: List = []

# Telemetry playback state
telemetry_is_paused = True
telemetry_is_reversed = False
telemetry_has_started = False
telemetry_playback_speed = 1.0
telemetry_master_start_time = None
telemetry_playback_start_timestamp = None
telemetry_rows = []
telemetry_pending_rows = []
telemetry_broadcast_task = None
telemetry_df = None  # Keep DataFrame for efficient lookups
telemetry_data_loaded = False  # Flag to track if data is loaded

# Endurance state
endurance_broadcast_task = None
endurance_df = None
endurance_data_loaded = False

# Leaderboard state
leaderboard_broadcast_task = None
leaderboard_df = None
leaderboard_data_loaded = False

# Data recording state
data_recording_enabled = False
current_event_name = None
recording_event_dir = None
vehicle_csv_writers = {}  # Dict[vehicle_id, csv.DictWriter]
vehicle_csv_files = {}  # Dict[vehicle_id, file handle]
recording_start_time = None
recording_lock = Lock()  # Thread-safe CSV writing


def get_project_root():
    """Get the fastapi-server directory (where this file is located)"""
    current_file = Path(__file__).resolve()
    return current_file.parent


def cast_num(x):
    """Cast value to number or None"""
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return x


# ==================== TELEMETRY PREPROCESSING ====================

# Configuration for preprocessing
KEEP_NAMES = {
    "nmot",
    "aps",
    "gear",
    "VBOX_Lat_Min",
    "VBOX_Long_Minutes",
    "Laptrigger_lapdist_dls",
    "speed",
    "accx_can",
    "accy_can",
    "pbrake_f",
    "pbrake_r",
    "Steering_Angle"
}

CHUNK_SIZE = 200_000


def latlon_to_xy(lat, lon):
    """Convert latitude/longitude to x/y coordinates"""
    R = 6371000
    x = R * np.radians(lon)
    y = R * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))
    return np.array([x, y])


def directional_filter(df):
    """Filter telemetry data based on direction to remove outliers"""
    df = df.sort_values("meta_time").reset_index(drop=True)
    lat_rows = df[df["telemetry_name"] == "VBOX_Lat_Min"].reset_index(drop=True)
    lon_rows = df[df["telemetry_name"] == "VBOX_Long_Minutes"].reset_index(drop=True)

    if len(lat_rows) != len(lon_rows):
        return df
    if len(lat_rows) < 3:
        return df

    filtered_indices = [0, 1, 2]
    for i in range(3, len(lat_rows)):
        p0 = latlon_to_xy(lat_rows.loc[filtered_indices[-3], "telemetry_value"],
                          lon_rows.loc[filtered_indices[-3], "telemetry_value"])
        p1 = latlon_to_xy(lat_rows.loc[filtered_indices[-2], "telemetry_value"],
                          lon_rows.loc[filtered_indices[-2], "telemetry_value"])
        p2 = latlon_to_xy(lat_rows.loc[filtered_indices[-1], "telemetry_value"],
                          lon_rows.loc[filtered_indices[-1], "telemetry_value"])
        p3 = latlon_to_xy(lat_rows.loc[i, "telemetry_value"],
                          lon_rows.loc[i, "telemetry_value"])
        if np.dot(p2 - p1, p3 - p2) > 0:
            filtered_indices.append(i)

    lat_filtered = lat_rows.loc[filtered_indices]
    lon_filtered = lon_rows.loc[filtered_indices]
    others = df[~df["telemetry_name"].isin(["VBOX_Lat_Min", "VBOX_Long_Minutes"])]
    filtered_df = pd.concat([lat_filtered, lon_filtered, others], ignore_index=True)
    return filtered_df.sort_values("meta_time").reset_index(drop=True)


def _preprocess_telemetry_data_sync(input_file: str, output_dir: Path):
    """
    Synchronous preprocessing function (internal)
    Preprocess raw telemetry data into per-vehicle CSV files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    vehicle_data = {}
    
    print(f"\n{'='*60}")
    print(f"Preprocessing Telemetry Data")
    print(f"{'='*60}")
    print(f"Input file: {input_file}")
    print(f"Output directory: {output_dir}")
    print(f"Processing CSV in chunks with filtering and cleanup...")
    
    try:
        # Process file in chunks
        for chunk in pd.read_csv(input_file, chunksize=CHUNK_SIZE):
            chunk["meta_time"] = pd.to_datetime(chunk["meta_time"], utc=True, errors="coerce")
            chunk = chunk.dropna(subset=["meta_time"])

            # Extract lap changes only
            lap_changes = (
                chunk[["meta_time", "vehicle_id", "lap"]]
                .dropna(subset=["lap"])
                .sort_values(["vehicle_id", "meta_time"])
                .drop_duplicates(subset=["vehicle_id", "lap"])
                .copy()
            )
            lap_changes["telemetry_name"] = "lap"
            lap_changes["telemetry_value"] = lap_changes["lap"]
            lap_changes = lap_changes.drop(columns=["lap"])

            # Filter telemetry signals
            chunk = chunk[chunk["telemetry_name"].isin(KEEP_NAMES)]

            # Aggregate duplicates
            chunk = (
                chunk.groupby(["meta_time", "vehicle_id", "telemetry_name"], as_index=False)
                     .agg({"telemetry_value": "median"})
            )

            # Format timestamps
            chunk["meta_time"] = pd.to_datetime(chunk["meta_time"], utc=True, errors="coerce")
            lap_changes["meta_time"] = pd.to_datetime(lap_changes["meta_time"], utc=True, errors="coerce")

            # Combine telemetry and lap rows
            combined = pd.concat([chunk, lap_changes], ignore_index=True)
            combined["meta_time"] = combined["meta_time"].dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ").str[:-3] + "Z"

            for vid, df_vid in combined.groupby("vehicle_id"):
                if vid not in vehicle_data:
                    vehicle_data[vid] = []

                df_vid = directional_filter(df_vid)
                df_vid = df_vid.drop(columns=["vehicle_id"], errors="ignore")
                vehicle_data[vid].append(df_vid)
        
        print("Merging and exporting per-vehicle CSVs...")
        
        # Merge and export per-vehicle CSVs
        results = {}
        for vid, parts in vehicle_data.items():
            df = pd.concat(parts, ignore_index=True)
            df = df.sort_values("meta_time")

            out_path = output_dir / f"{vid}.csv"
            df.to_csv(out_path, index=False)
            results[vid] = {
                "path": str(out_path),
                "rows": len(df)
            }
            print(f"✅ Exported {vid} → {out_path} ({len(df)} rows)")
        
        print(f"All vehicles processed with lap telemetry injected only on change.")
        print(f"{'='*60}\n")
        
        return {
            "status": "success",
            "message": f"Successfully processed {len(results)} vehicles",
            "input_file": input_file,
            "output_dir": str(output_dir),
            "vehicles": results,
            "vehicle_count": len(results),
            "total_rows": sum(r["rows"] for r in results.values())
        }
        
    except Exception as e:
        error_msg = f"Error preprocessing telemetry data: {str(e)}"
        print(f"⚠️ {error_msg}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": error_msg,
            "error": str(e)
        }


async def preprocess_telemetry_data(input_file: str = None, output_dir: str = None):
    """
    Preprocess raw telemetry data into per-vehicle CSV files
    
    Args:
        input_file: Path to raw telemetry CSV file (e.g., R1_barber_telemetry_data.csv)
        output_dir: Directory to write processed vehicle CSV files
    
    Returns:
        dict: Status and results of preprocessing
    """
    project_root = get_project_root()
    
    # Set default paths if not provided
    if input_file is None:
        # Try multiple possible locations
        possible_inputs = [
            project_root / "logs" / "R1_barber_telemetry_data.csv",
            project_root.parent / "telemetry-server" / "logs" / "R1_barber_telemetry_data.csv",
        ]
        input_file = None
        for path in possible_inputs:
            if path.exists():
                input_file = str(path)
                break
        
        if input_file is None:
            # Check if data is already processed
            vehicles_dir = project_root / "logs" / "vehicles"
            if vehicles_dir.exists() and len(list(vehicles_dir.glob("*.csv"))) > 0:
                return {
                    "status": "info",
                    "message": "Data is already processed. Vehicle CSV files found in logs/vehicles/",
                    "vehicles_dir": str(vehicles_dir),
                    "vehicle_count": len(list(vehicles_dir.glob("*.csv")))
                }
            
            return {
                "status": "error",
                "message": f"Input file not found. Searched: {[str(p) for p in possible_inputs]}",
                "note": "If data is already processed, check logs/vehicles/ directory"
            }
    else:
        # Resolve relative paths
        if not os.path.isabs(input_file):
            input_file = str(project_root / input_file)
    
    if not os.path.exists(input_file):
        return {
            "status": "error",
            "message": f"Input file not found: {input_file}"
        }
    
    if output_dir is None:
        output_dir = project_root / "logs" / "vehicles"
    else:
        if not os.path.isabs(output_dir):
            output_dir = project_root / output_dir
    
    output_dir = Path(output_dir)
    
    # Run synchronous preprocessing in executor to avoid blocking event loop
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        _preprocess_telemetry_data_sync,
        input_file,
        output_dir
    )
    
    return result


# ==================== DATA LOADING (Pre-load on startup) ====================

async def load_telemetry_data(event_name: Optional[str] = None):
    """
    Pre-load telemetry data on server startup for fast access
    
    Args:
        event_name: Optional event name to load data from a specific event.
                   If None, loads from default logs/vehicles directory
    """
    global telemetry_rows, telemetry_pending_rows, telemetry_df, telemetry_data_loaded
    global telemetry_playback_start_timestamp
    
    if telemetry_data_loaded and event_name is None:
        return True
    
    print("\n" + "="*60)
    print("Loading Telemetry Data (Pre-loading for fast access)...")
    print("="*60)
    
    project_root = get_project_root()
    
    # Determine which directory to use
    if event_name:
        input_dir = project_root / "logs" / "events" / event_name / "vehicles"
        print(f"Loading data from event: {event_name}")
    else:
        input_dir = project_root / "logs" / "vehicles"
    
    weather_file = project_root / "logs" / "R1_weather_data.csv"
    
    # Load vehicle telemetry
    vehicle_files = glob.glob(str(input_dir / "*.csv"))
    if not vehicle_files:
        print(f"⚠️ WARNING: No vehicle CSVs found in {input_dir}")
        return False
    
    print(f"✅ Found {len(vehicle_files)} vehicle CSV files")
    print(f"Loading vehicle telemetry files (this may take a moment)...")
    
    # Load files in parallel using asyncio
    import concurrent.futures
    loop = asyncio.get_event_loop()
    
    def load_file(f):
        try:
            vehicle_id = os.path.splitext(os.path.basename(f))[0]
            df = pd.read_csv(f, parse_dates=["meta_time"], low_memory=False)
            df["meta_time"] = pd.to_datetime(df["meta_time"], utc=True, errors="coerce")
            df["vehicle_id"] = vehicle_id
            return df
        except Exception as e:
            print(f"⚠️ ERROR: Failed to load {f}: {e}")
            return None
    
    # Load files concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        dfs = await asyncio.gather(*[
            loop.run_in_executor(executor, load_file, f) 
            for f in vehicle_files
        ])
    
    # Filter out None results
    dfs = [df for df in dfs if df is not None]
    
    if not dfs:
        print("⚠️ WARNING: No vehicle telemetry data could be loaded.")
        return False
    
    # Combine and sort
    telemetry_df = pd.concat(dfs, ignore_index=True)
    telemetry_df = telemetry_df.sort_values("meta_time").reset_index(drop=True)
    
    # Convert to list of dicts (only once, after sorting)
    # Use list() to avoid keeping reference to DataFrame
    telemetry_rows = list(telemetry_df.to_dict("records"))
    # Don't copy here - will be set when playback starts
    telemetry_pending_rows = []
    
    if len(telemetry_rows) > 0:
        telemetry_playback_start_timestamp = telemetry_rows[0]["meta_time"]
    
    print(f"✅ Loaded {len(telemetry_df)} telemetry records from {len(dfs)} vehicle files")
    print(f"   Data range: {telemetry_rows[0]['meta_time'] if telemetry_rows else 'N/A'} to {telemetry_rows[-1]['meta_time'] if telemetry_rows else 'N/A'}")
    
    telemetry_data_loaded = True
    print("✅ Telemetry data pre-loaded and ready!")
    print("="*60 + "\n")
    return True


async def load_endurance_data():
    """Pre-load endurance data on server startup"""
    global endurance_df, endurance_data_loaded
    
    if endurance_data_loaded:
        return True
    
    project_root = get_project_root()
    endurance_file = project_root / "logs" / "R1_section_endurance.csv"
    
    if not endurance_file.exists():
        print(f"⚠️ WARNING: Endurance file not found at {endurance_file}")
        return False
    
    try:
        endurance_df = pd.read_csv(endurance_file, sep=";", low_memory=False)
        endurance_df.columns = endurance_df.columns.str.strip()
        endurance_df["CAR_NUMBER"] = endurance_df["NUMBER"].astype(str)
        endurance_df.sort_values(["CAR_NUMBER", "LAP_NUMBER", "ELAPSED"], inplace=True)
        endurance_data_loaded = True
        print(f"✅ Loaded {len(endurance_df)} endurance records")
        return True
    except Exception as e:
        print(f"⚠️ ERROR: Failed to load endurance data: {e}")
        return False


async def load_leaderboard_data():
    """Pre-load leaderboard data on server startup"""
    global leaderboard_df, leaderboard_data_loaded
    
    if leaderboard_data_loaded:
        return True
    
    project_root = get_project_root()
    leaderboard_file = project_root / "logs" / "R1_leaderboard.csv"
    
    if not leaderboard_file.exists():
        print(f"⚠️ WARNING: Leaderboard file not found at {leaderboard_file}")
        print(f"   Leaderboard endpoint will return empty data until file is available")
        return False
    
    try:
        leaderboard_df = pd.read_csv(leaderboard_file, sep=";", low_memory=False)
        leaderboard_df.columns = leaderboard_df.columns.str.strip()
        leaderboard_data_loaded = True
        print(f"✅ Loaded {len(leaderboard_df)} leaderboard records")
        return True
    except Exception as e:
        print(f"⚠️ ERROR: Failed to load leaderboard data: {e}")
        return False


# ==================== DATA RECORDING ====================

def start_data_recording(event_name: str = None):
    """
    Start recording telemetry data to CSV files
    
    Args:
        event_name: Name of the event/session (e.g., "R1", "R2", "Practice_1")
                   If None, uses timestamp-based name
    """
    global data_recording_enabled, current_event_name, recording_event_dir
    global vehicle_csv_writers, vehicle_csv_files, recording_start_time
    
    if data_recording_enabled:
        print("⚠️ Recording already in progress. Stop current recording first.")
        return False
    
    project_root = get_project_root()
    
    # Create event name if not provided
    if event_name is None:
        event_name = f"Event_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    current_event_name = event_name
    
    # Create event directory structure: logs/events/EVENT_NAME/vehicles/
    events_dir = project_root / "logs" / "events"
    recording_event_dir = events_dir / event_name / "vehicles"
    recording_event_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize CSV writers for each vehicle
    vehicle_csv_writers = {}
    vehicle_csv_files = {}
    recording_start_time = datetime.now()
    
    print(f"✅ Started recording telemetry data for event: {event_name}")
    print(f"   Output directory: {recording_event_dir}")
    
    data_recording_enabled = True
    return True


def stop_data_recording():
    """Stop recording telemetry data and close all CSV files"""
    global data_recording_enabled, vehicle_csv_writers, vehicle_csv_files
    
    if not data_recording_enabled:
        print("⚠️ No recording in progress.")
        return False
    
    with recording_lock:
        # Close all CSV files
        for vehicle_id, file_handle in vehicle_csv_files.items():
            try:
                file_handle.close()
            except Exception as e:
                print(f"⚠️ Error closing CSV file for {vehicle_id}: {e}")
        
        vehicle_csv_writers = {}
        vehicle_csv_files = {}
    
    data_recording_enabled = False
    print(f"✅ Stopped recording telemetry data for event: {current_event_name}")
    print(f"   Data saved to: {recording_event_dir}")
    
    return True


def write_telemetry_to_csv(vehicle_id: str, timestamp: str, telemetry_data: dict):
    """
    Write telemetry data to CSV file for a specific vehicle
    
    Args:
        vehicle_id: Vehicle ID (e.g., "GR86-022-13")
        timestamp: ISO format timestamp
        telemetry_data: Dict of telemetry values
    """
    global vehicle_csv_writers, vehicle_csv_files, recording_event_dir
    
    if not data_recording_enabled:
        return
    
    # Get driver number from vehicle mapping
    try:
        from vehicle_mapping import get_cached_mapping
        vehicle_mapping = get_cached_mapping()
        vehicle_info = vehicle_mapping.get(vehicle_id, {})
        driver_number = vehicle_info.get("driver_number")
        car_number = vehicle_info.get("car_number")
    except Exception:
        driver_number = None
        car_number = None
    
    with recording_lock:
        # Initialize CSV writer for this vehicle if not exists
        if vehicle_id not in vehicle_csv_writers:
            csv_file_path = recording_event_dir / f"{vehicle_id}.csv"
            
            # Check if file exists to determine if we need to write header
            file_exists = csv_file_path.exists()
            
            csv_file = open(csv_file_path, 'a', newline='', encoding='utf-8')
            vehicle_csv_files[vehicle_id] = csv_file
            
            # Define CSV columns
            fieldnames = [
                'meta_time',
                'vehicle_id',
                'driver_number',
                'car_number',
                'telemetry_name',
                'telemetry_value'
            ]
            
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            
            # Write header if file is new
            if not file_exists:
                writer.writeheader()
            
            vehicle_csv_writers[vehicle_id] = writer
        
        writer = vehicle_csv_writers[vehicle_id]
        
        # Write each telemetry field as a separate row (matching existing format)
        for telemetry_name, telemetry_value in telemetry_data.items():
            # Skip None values
            if telemetry_value is None:
                continue
            
            # Map frontend field names back to original names
            reverse_field_mapping = {
                "gps_lat": "VBOX_Lat_Min",
                "gps_lon": "VBOX_Long_Minutes",
            }
            original_name = reverse_field_mapping.get(telemetry_name, telemetry_name)
            
            row = {
                'meta_time': timestamp,
                'vehicle_id': vehicle_id,
                'driver_number': driver_number if driver_number is not None else '',
                'car_number': car_number if car_number is not None else '',
                'telemetry_name': original_name,
                'telemetry_value': telemetry_value
            }
            
            try:
                writer.writerow(row)
            except Exception as e:
                print(f"⚠️ Error writing CSV row for {vehicle_id}: {e}")


# ==================== TELEMETRY BROADCAST ====================

async def telemetry_broadcast_loop():
    """Broadcast telemetry data to connected clients (uses pre-loaded data)"""
    global telemetry_master_start_time, telemetry_playback_start_timestamp
    global telemetry_rows, telemetry_pending_rows, telemetry_has_started
    global telemetry_is_paused, telemetry_is_reversed, telemetry_playback_speed
    global telemetry_cache

    # Ensure data is loaded (should already be from startup, but check anyway)
    if not telemetry_data_loaded:
        await load_telemetry_data()
    
    if not telemetry_rows:
        print("⚠️ WARNING: No telemetry data available for broadcast")
        return

    # Load weather data (lightweight, can load on demand)
    project_root = get_project_root()
    weather_file = project_root / "logs" / "R1_weather_data.csv"
    
    weather_rows = []
    pending_weather = []
    latest_weather = None
    
    if weather_file.exists():
        try:
            df_weather = pd.read_csv(weather_file, sep=";", low_memory=False)
            df_weather["meta_time"] = pd.to_datetime(df_weather["TIME_UTC_SECONDS"], utc=True, errors="coerce")
            df_weather = df_weather.sort_values("meta_time")
            weather_rows = df_weather.to_dict("records")
            pending_weather = weather_rows
            print(f"✅ Loaded {len(weather_rows)} weather records")
        except Exception as e:
            print(f"⚠️ ERROR: Failed to load weather data: {e}")
    else:
        print(f"⚠️ WARNING: Weather file not found at {weather_file}")

    # Create sorted list of timestamps for binary search (O(n) once, then O(log n) lookups)
    telemetry_timestamps = [r["meta_time"] for r in telemetry_rows]
    
    # Initialize playback state
    telemetry_master_start_time = None
    telemetry_has_started = True
    current_index = 0  # Track current position in sorted data
    
    target_hz = 60
    send_interval = 1.0 / target_hz
    print("✅ Telemetry broadcast loop started (using pre-loaded data)")

    last_send_time = asyncio.get_event_loop().time()

    while True:
        # Only sleep when paused to reduce CPU usage
        if telemetry_is_paused or telemetry_playback_speed == 0:
            await asyncio.sleep(0.1)  # Longer sleep when paused
        else:
            await asyncio.sleep(0.001)  # Short sleep when playing

        if telemetry_is_paused or telemetry_playback_speed == 0:
            continue

        if telemetry_master_start_time is None:
            telemetry_master_start_time = asyncio.get_event_loop().time()
            # Reset index when starting playback
            if not telemetry_is_reversed:
                current_index = 0
                # Use slice reference instead of copy for better performance
                telemetry_pending_rows = telemetry_rows

        elapsed_real = asyncio.get_event_loop().time() - telemetry_master_start_time
        delta = -elapsed_real * telemetry_playback_speed if telemetry_is_reversed else elapsed_real * telemetry_playback_speed
        sim_time = telemetry_playback_start_timestamp + pd.to_timedelta(delta, unit="s")

        # Use binary search for efficient time-based filtering
        if telemetry_is_reversed:
            # For reverse, find all rows >= sim_time
            idx = bisect.bisect_left(telemetry_timestamps, sim_time)
            to_emit = telemetry_rows[idx:]
            telemetry_pending_rows = telemetry_rows[:idx]
        else:
            # For forward, find all rows <= sim_time starting from current_index
            end_idx = bisect.bisect_right(telemetry_timestamps, sim_time, lo=current_index)
            to_emit = telemetry_rows[current_index:end_idx]
            current_index = end_idx
            telemetry_pending_rows = telemetry_rows[current_index:]

        # Get latest weather sample
        weather_to_emit = [w for w in pending_weather if w["meta_time"] <= sim_time]
        pending_weather = [w for w in pending_weather if w["meta_time"] > sim_time]
        if weather_to_emit:
            latest_weather = weather_to_emit[-1]

        now = asyncio.get_event_loop().time()
        if now - last_send_time < send_interval:
            continue
        last_send_time = now

        if not to_emit:
            continue

        # Group into frames
        frame = defaultdict(lambda: defaultdict(dict))
        seen = set()
        # Field name mapping for frontend compatibility
        field_mapping = {
            "VBOX_Lat_Min": "gps_lat",
            "VBOX_Long_Minutes": "gps_lon",
        }
        
        for r in to_emit:
            ts = r["meta_time"].isoformat()
            vid = r["vehicle_id"]
            key = (ts, vid, r["telemetry_name"])
            if key in seen:
                continue
            seen.add(key)
            name = r["telemetry_name"]
            # Map field names for frontend compatibility
            mapped_name = field_mapping.get(name, name)
            value = cast_num(r["telemetry_value"])
            frame[ts][vid][mapped_name] = int(value) if name == "lap" else value

        # Send frames
        for ts, vehicles in frame.items():
            msg = {
                "type": "telemetry_frame",
                "timestamp": ts,
                "vehicles": vehicles
            }

            if latest_weather:
                msg["weather"] = {
                    "air_temp": cast_num(latest_weather["AIR_TEMP"]),
                    "track_temp": cast_num(latest_weather["TRACK_TEMP"]),
                    "humidity": cast_num(latest_weather["HUMIDITY"]),
                    "pressure": cast_num(latest_weather["PRESSURE"]),
                    "wind_speed": cast_num(latest_weather["WIND_SPEED"]),
                    "wind_direction": cast_num(latest_weather["WIND_DIRECTION"]),
                    "rain": cast_num(latest_weather["RAIN"])
                }

            data = json.dumps(msg)
            telemetry_cache = msg
            
            # Record telemetry data to CSV files if recording is enabled
            if data_recording_enabled:
                for vid, vehicle_data in vehicles.items():
                    write_telemetry_to_csv(vid, ts, vehicle_data)
            
            # Data is cached and served via REST API (clients poll for updates)

        # End condition
        if (not telemetry_pending_rows and not telemetry_is_reversed) or (not to_emit and telemetry_is_reversed):
            print("End of telemetry log reached.")
            end_msg = {
                "type": "telemetry_end",
                "timestamp": sim_time.isoformat()
            }
            # End message is cached and served via REST API
            telemetry_is_paused = True
            telemetry_master_start_time = None


async def process_telemetry_control(msg: dict):
    """Process control commands for telemetry playback"""
    global telemetry_is_paused, telemetry_is_reversed, telemetry_has_started
    global telemetry_playback_speed, telemetry_master_start_time, telemetry_playback_start_timestamp
    global telemetry_pending_rows, telemetry_rows

    cmd = msg.get("cmd")
    if cmd == "play":
        if not telemetry_has_started:
            # First time starting - initialize
            telemetry_has_started = True
            if telemetry_rows:
                telemetry_playback_start_timestamp = telemetry_rows[0]["meta_time"]
                telemetry_pending_rows = telemetry_rows.copy()
            print("Playback started for the first time")
        if telemetry_is_paused:
            telemetry_is_paused = False
            telemetry_is_reversed = False
            telemetry_master_start_time = asyncio.get_event_loop().time()
            print("Playback resumed/started")
    elif cmd == "reverse":
        if telemetry_is_paused and telemetry_has_started:
            telemetry_is_paused = False
            telemetry_is_reversed = True
            telemetry_master_start_time = asyncio.get_event_loop().time()
            print("Reverse playback started")
    elif cmd == "restart":
        telemetry_is_paused = True
        telemetry_is_reversed = False
        telemetry_has_started = True
        telemetry_playback_start_timestamp = telemetry_rows[0]["meta_time"]
        telemetry_pending_rows = telemetry_rows.copy()
        telemetry_master_start_time = None
        print("Playback restarted")
    elif cmd == "pause":
        if not telemetry_is_paused:
            elapsed = asyncio.get_event_loop().time() - telemetry_master_start_time
            delta = -elapsed * telemetry_playback_speed if telemetry_is_reversed else elapsed * telemetry_playback_speed
            telemetry_playback_start_timestamp += pd.to_timedelta(delta, unit="s")
            telemetry_is_paused = True
            telemetry_master_start_time = None
            print("Paused")
    elif cmd == "speed":
        val = float(msg.get("value", 1.0))
        if not telemetry_is_paused and telemetry_master_start_time:
            elapsed = asyncio.get_event_loop().time() - telemetry_master_start_time
            delta = -elapsed * telemetry_playback_speed if telemetry_is_reversed else elapsed * telemetry_playback_speed
            telemetry_playback_start_timestamp += pd.to_timedelta(delta, unit="s")
            telemetry_master_start_time = asyncio.get_event_loop().time()
        telemetry_playback_speed = val
        print(f"Speed set to {telemetry_playback_speed}x")
    elif cmd == "seek":
        telemetry_playback_start_timestamp = dtparser.parse(msg["timestamp"])
        telemetry_master_start_time = asyncio.get_event_loop().time()
        print(f"Seek to {telemetry_playback_start_timestamp}")


# ==================== ENDURANCE BROADCAST ====================

async def endurance_broadcast_loop():
    """Broadcast endurance/lap event data (uses pre-loaded data)"""
    global endurance_cache, endurance_df

    # Ensure data is loaded
    if not endurance_data_loaded:
        await load_endurance_data()
    
    if endurance_df is None or len(endurance_df) == 0:
        print("⚠️ WARNING: No endurance data available")
        return

    wait = 0.01
    print("✅ Endurance stream started (using pre-loaded data)")

    try:
        # Use itertuples() instead of iterrows() for 10-100x speedup
        for row in endurance_df.itertuples():
            # Always process data for REST API cache
            #     await asyncio.sleep(0.1)
            #     continue

            try:
                msg = {
                    "type": "lap_event",
                    "vehicle_id": str(row.CAR_NUMBER),
                    "lap": int(row.LAP_NUMBER),
                    "lap_time": getattr(row, 'LAP_TIME', None),
                    "sector_times": [
                        getattr(row, 'S1_SECONDS', None),
                        getattr(row, 'S2_SECONDS', None),
                        getattr(row, 'S3_SECONDS', None)
                    ],
                    "top_speed": getattr(row, 'TOP_SPEED', None),
                    "flag": getattr(row, 'FLAG_AT_FL', None),
                    "pit": pd.notna(getattr(row, 'CROSSING_FINISH_LINE_IN_PIT', None)),
                    "timestamp": getattr(row, 'HOUR', None),
                }

                data = json.dumps(msg)
                endurance_cache.append(msg)
                
                # Data is cached and served via REST API (clients poll for updates)

                await asyncio.sleep(wait)
            except Exception as e:
                print(f"Error processing endurance row: {e}")
                await asyncio.sleep(wait)
                continue

        print("Endurance event stream finished. Restarting...")
        await asyncio.sleep(1.0)
    except Exception as e:
        print(f"Endurance broadcast loop error: {e}")
        import traceback
        traceback.print_exc()
        await asyncio.sleep(1.0)


# ==================== LEADERBOARD BROADCAST ====================

async def leaderboard_broadcast_loop():
    """Broadcast leaderboard data (uses pre-loaded data)"""
    global leaderboard_cache, leaderboard_df

    # Ensure data is loaded
    if not leaderboard_data_loaded:
        success = await load_leaderboard_data()
        if not success:
            print("⚠️ WARNING: Failed to load leaderboard data")
            return
    
    if leaderboard_df is None or len(leaderboard_df) == 0:
        print("⚠️ WARNING: No leaderboard data available")
        return

    wait = 0.01
    # Use itertuples() for better performance than to_dict("records")
    print("✅ Broadcasting leaderboard data (using pre-loaded data)...")

    try:
        for row in leaderboard_df.itertuples():
            # Always process data for REST API cache
            #     await asyncio.sleep(0.1)
            #     continue

            try:
                # Helper function to safely convert values
                def safe_value(val, default=None, convert_type=None):
                    if pd.isna(val) or val is None:
                        return default
                    try:
                        if convert_type:
                            return convert_type(val)
                        return val
                    except (ValueError, TypeError):
                        return default
                
                msg = {
                    "type": "leaderboard_entry",
                    "class_type": safe_value(getattr(row, 'CLASS_TYPE', None)),
                    "position": safe_value(getattr(row, 'POS', 0), 0, int),
                    "pic": safe_value(getattr(row, 'PIC', 0), 0, int),
                    "vehicle_id": str(safe_value(getattr(row, 'NUMBER', ''), '')),
                    "vehicle": safe_value(getattr(row, 'VEHICLE', None)),
                    "laps": safe_value(getattr(row, 'LAPS', 0), 0, int),
                    "elapsed": safe_value(getattr(row, 'ELAPSED', None)),
                    "gap_first": safe_value(getattr(row, 'GAP_FIRST', None)),
                    "gap_previous": safe_value(getattr(row, 'GAP_PREVIOUS', None)),
                    "best_lap_num": safe_value(getattr(row, 'BEST_LAP_NUM', 0), 0, int),
                    "best_lap_time": safe_value(getattr(row, 'BEST_LAP_TIME', None)),
                    "best_lap_kph": safe_value(getattr(row, 'BEST_LAP_KPH', 0), 0.0, float),
                }

                data = json.dumps(msg)
                # Update cache (replace existing entry for same vehicle_id)
                existing_idx = next((i for i, e in enumerate(leaderboard_cache) if e.get("vehicle_id") == msg["vehicle_id"]), None)
                if existing_idx is not None:
                    leaderboard_cache[existing_idx] = msg
                else:
                    leaderboard_cache.append(msg)
                
                # Data is cached and served via REST API (clients poll for updates)

                await asyncio.sleep(wait)
            except Exception as e:
                print(f"Error processing leaderboard row: {e}")
                await asyncio.sleep(wait)
                continue

        print("Leaderboard event stream finished. Restarting...")
        await asyncio.sleep(1.0)
    except Exception as e:
        print(f"Leaderboard broadcast loop error: {e}")
        import traceback
        traceback.print_exc()
        await asyncio.sleep(1.0)


# ==================== REST API ENDPOINTS ====================

# Handle OPTIONS requests for CORS preflight
@app.options("/{full_path:path}")
async def options_handler(full_path: str):
    """Handle CORS preflight OPTIONS requests"""
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, PATCH",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Max-Age": "3600",
        }
    )

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Telemetry Rush API - Integrated (All on Port 8000)",
        "version": "2.0.0",
        "endpoints": {
            "rest": {
                "telemetry": "/api/telemetry",
                "endurance": "/api/endurance",
                "leaderboard": "/api/leaderboard",
                "health": "/api/health",
                "preprocess": "/api/preprocess",
                "control": "/api/control"
            },
            "note": "Poll REST endpoints for real-time updates. Use /api/control for playback controls."
        }
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    # List all driver-related routes
    driver_routes = [
        r.path for r in app.routes 
        if hasattr(r, 'path') and '/driver/' in r.path
    ]
    
    # List all API routes for debugging
    all_api_routes = [
        r.path for r in app.routes 
        if hasattr(r, 'path') and r.path.startswith('/api/')
    ]
    
    # Check if leaderboard route exists
    leaderboard_exists = '/api/leaderboard' in all_api_routes
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "data_loaded": {
            "telemetry": telemetry_data_loaded,
            "endurance": endurance_data_loaded,
            "leaderboard": leaderboard_data_loaded
        },
        "driver_routes": driver_routes,
        "api_routes": sorted(all_api_routes),
        "leaderboard_route_exists": leaderboard_exists
    }


@app.get("/api/vehicles")
async def get_vehicles(event_name: Optional[str] = None):
    """
    Get list of all available vehicles from telemetry data with driver numbers
    
    Args:
        event_name: Optional event name to get vehicles from a specific event.
                   If None, uses default logs/vehicles directory
    """
    project_root = get_project_root()
    
    # Determine which directory to use
    if event_name:
        vehicles_dir = project_root / "logs" / "events" / event_name / "vehicles"
    else:
        vehicles_dir = project_root / "logs" / "vehicles"
    
    if not vehicles_dir.exists():
        return {"vehicles": [], "count": 0, "event_name": event_name}
    
    # Import vehicle mapping to get driver numbers
    try:
        from vehicle_mapping import get_cached_mapping
        vehicle_mapping = get_cached_mapping()
    except ImportError:
        vehicle_mapping = {}
    
    # Get all CSV files in vehicles directory
    vehicle_files = glob.glob(str(vehicles_dir / "*.csv"))
    vehicles = []
    
    for vehicle_file in vehicle_files:
        vehicle_id = os.path.splitext(os.path.basename(vehicle_file))[0]
        vehicle_info = vehicle_mapping.get(vehicle_id, {})
        
        # Try to get driver number from CSV file if not in mapping
        driver_number = vehicle_info.get("driver_number")
        car_number = vehicle_info.get("car_number")
        
        # Read first row of CSV to get driver_number if available
        try:
            df_sample = pd.read_csv(vehicle_file, nrows=1)
            if 'driver_number' in df_sample.columns and pd.notna(df_sample['driver_number'].iloc[0]):
                driver_number = int(df_sample['driver_number'].iloc[0])
            if 'car_number' in df_sample.columns and pd.notna(df_sample['car_number'].iloc[0]):
                car_number = int(df_sample['car_number'].iloc[0])
        except Exception:
            pass
        
        vehicles.append({
            "id": vehicle_id,
            "name": vehicle_id,
            "file": os.path.basename(vehicle_file),
            "vehicle_number": vehicle_info.get("vehicle_number"),
            "car_number": car_number,
            "driver_number": driver_number,
            "has_endurance_data": vehicle_info.get("has_endurance_data", False),
            "event_name": event_name
        })
    
    # Sort vehicles by ID
    vehicles.sort(key=lambda x: x["id"])
    
    return {
        "vehicles": vehicles,
        "count": len(vehicles),
        "event_name": event_name,
        "source_dir": str(vehicles_dir)
    }


@app.get("/api/telemetry")
async def get_telemetry():
    """Get latest telemetry data - Poll this endpoint for updates"""
    global telemetry_broadcast_task
    
    try:
        # Start broadcast loop if not already running
        if telemetry_broadcast_task is None or telemetry_broadcast_task.done():
            try:
                telemetry_broadcast_task = asyncio.create_task(telemetry_broadcast_loop())
                print("Started telemetry broadcast loop for REST API")
            except Exception as e:
                print(f"⚠️ Error starting telemetry broadcast loop: {e}")
        
        if telemetry_cache:
            return telemetry_cache
        
        # Check if data is loaded but not started
        if telemetry_data_loaded and len(telemetry_rows) > 0:
            return {
                "message": "Telemetry data loaded but playback not started",
                "row_count": len(telemetry_rows),
                "has_data": True,
                "paused": telemetry_is_paused,
                "suggestion": "Poll /api/telemetry for updates. Use /api/control to start playback."
            }
        
        return {
            "message": "No telemetry data available",
            "has_data": False,
            "suggestion": "Ensure CSV files exist in logs/vehicles/ and data has been loaded"
        }
    except Exception as e:
        print(f"⚠️ Error in get_telemetry endpoint: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error retrieving telemetry: {str(e)}")


@app.get("/api/endurance")
async def get_endurance():
    """Get endurance/lap event data - Poll this endpoint for updates"""
    global endurance_broadcast_task
    
    try:
        # Start broadcast loop if not already running
        if endurance_broadcast_task is None or endurance_broadcast_task.done():
            try:
                endurance_broadcast_task = asyncio.create_task(endurance_broadcast_loop())
                print("Started endurance broadcast loop for REST API")
            except Exception as e:
                print(f"⚠️ Error starting endurance broadcast loop: {e}")
        
        return {"events": endurance_cache, "count": len(endurance_cache)}
    except Exception as e:
        print(f"⚠️ Error in get_endurance endpoint: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error retrieving endurance data: {str(e)}")


@app.get("/api/leaderboard")
async def get_leaderboard():
    """Get leaderboard data - Poll this endpoint for updates"""
    global leaderboard_broadcast_task, leaderboard_data_loaded, leaderboard_cache
    
    try:
        print(f"📊 GET /api/leaderboard - Data loaded: {leaderboard_data_loaded}, Cache size: {len(leaderboard_cache)}")
        
        # Start broadcast loop if not already running
        if leaderboard_broadcast_task is None or leaderboard_broadcast_task.done():
            try:
                leaderboard_broadcast_task = asyncio.create_task(leaderboard_broadcast_loop())
                print("✅ Started leaderboard broadcast loop for REST API")
            except Exception as e:
                print(f"⚠️ Error starting leaderboard broadcast loop: {e}")
                import traceback
                traceback.print_exc()
        
        # Always return a valid response, even if data isn't loaded yet
        if not leaderboard_data_loaded:
            print("⚠️ Leaderboard data not loaded yet, returning empty response")
            return {
                "leaderboard": [],
                "count": 0,
                "message": "Leaderboard data not loaded",
                "suggestion": "Ensure R1_leaderboard.csv exists in logs/ directory",
                "status": "loading"
            }
        
        if len(leaderboard_cache) == 0:
            print("⚠️ Leaderboard cache is empty, returning empty response")
            return {
                "leaderboard": [],
                "count": 0,
                "message": "No leaderboard entries in cache yet",
                "has_data": True,
                "suggestion": "Poll /api/leaderboard for updates",
                "status": "empty_cache"
            }
        
        # Clean NaN values before returning
        cleaned_cache = clean_nan_values(leaderboard_cache)
        print(f"✅ Returning {len(cleaned_cache)} leaderboard entries")
        return {
            "leaderboard": cleaned_cache,
            "count": len(cleaned_cache),
            "status": "success"
        }
    except Exception as e:
        error_msg = f"Error retrieving leaderboard: {str(e)}"
        print(f"⚠️ Error in get_leaderboard endpoint: {error_msg}")
        import traceback
        traceback.print_exc()
        # Return 500 instead of 404 to indicate server error
        raise HTTPException(status_code=500, detail=error_msg)


@app.post("/api/control")
async def control_playback(command: dict):
    """Send control command to telemetry playback"""
    await process_telemetry_control(command)
    return {"status": "command_sent", "command": command.get("cmd")}


@app.post("/api/recording/start")
async def start_recording(request: dict = None):
    """Start recording telemetry data to CSV files"""
    event_name = None
    if request:
        event_name = request.get("event_name")
    
    success = start_data_recording(event_name)
    if success:
        return {
            "status": "recording_started",
            "event_name": current_event_name,
            "output_dir": str(recording_event_dir)
        }
    else:
        raise HTTPException(status_code=400, detail="Failed to start recording. Recording may already be in progress.")


@app.post("/api/recording/stop")
async def stop_recording():
    """Stop recording telemetry data"""
    success = stop_data_recording()
    if success:
        return {
            "status": "recording_stopped",
            "event_name": current_event_name,
            "output_dir": str(recording_event_dir) if recording_event_dir else None
        }
    else:
        raise HTTPException(status_code=400, detail="No recording in progress.")


@app.get("/api/recording/status")
async def get_recording_status():
    """Get current recording status"""
    return {
        "recording": data_recording_enabled,
        "event_name": current_event_name,
        "output_dir": str(recording_event_dir) if recording_event_dir else None,
        "start_time": recording_start_time.isoformat() if recording_start_time else None,
        "vehicles_recorded": list(vehicle_csv_writers.keys())
    }


@app.get("/api/events")
async def get_events():
    """Get list of all recorded events/sessions"""
    project_root = get_project_root()
    events_dir = project_root / "logs" / "events"
    
    events = []
    if events_dir.exists():
        for event_dir in events_dir.iterdir():
            if event_dir.is_dir():
                vehicles_dir = event_dir / "vehicles"
                vehicle_count = 0
                if vehicles_dir.exists():
                    vehicle_count = len(list(vehicles_dir.glob("*.csv")))
                
                events.append({
                    "name": event_dir.name,
                    "path": str(event_dir),
                    "vehicle_count": vehicle_count,
                    "created": datetime.fromtimestamp(event_dir.stat().st_mtime).isoformat()
                })
    
    # Sort by creation time (newest first)
    events.sort(key=lambda x: x["created"], reverse=True)
    
    return {"events": events, "count": len(events)}


# ==================== POST-EVENT ANALYSIS ====================

@app.get("/api/analysis/race-story", response_class=Response)
async def race_story_endpoint():
    """Get comprehensive race story with position changes and key moments"""
    result = await get_race_story()
    cleaned = clean_nan_values(result)
    
    # Add Gemini AI insights
    try:
        from gemini_insights import generate_race_story_insights
        gemini_insights = generate_race_story_insights(result)
        if gemini_insights.get("enhanced"):
            cleaned["gemini_insights"] = gemini_insights
    except Exception as e:
        print(f"⚠️ Failed to generate Gemini race story insights: {e}")
    
    # Manually encode JSON with custom encoder to handle all edge cases
    def json_serializer(obj):
        """Custom JSON serializer for numpy types and NaN"""
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        elif pd.isna(obj):
            return None
        raise TypeError(f"Type {type(obj)} not serializable")
    
    try:
        json_str = json.dumps(cleaned, default=json_serializer)
        return Response(content=json_str, media_type="application/json")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error serializing race story: {str(e)}")

@app.get("/api/analysis/sector-comparison")
async def sector_comparison_endpoint():
    """Compare sector times across all drivers"""
    result = await get_sector_comparison()
    return clean_nan_values(result)

@app.get("/api/analysis/driver/{vehicle_id}")
async def driver_insights_endpoint(vehicle_id: str):
    """Get detailed insights for a specific driver"""
    result = await get_driver_insights(vehicle_id)
    return clean_nan_values(result)

@app.post("/api/analysis/post-event")
async def post_event_analysis_endpoint(request: dict = Body(...)):
    """
    Generate comprehensive post-event analysis using predictive models
    
    Compares predicted vs actual performance to tell the story of the race,
    revealing key moments and strategic decisions that led to the outcome.
    
    Request body:
    {
        "track_name": "Barber" | "Sonoma" | "Road America" | "Circuit of the Americas" | "Virginia International Raceway",
        "race_session": "R1" | "R2",
        "min_lap_time": 25.0 (optional)
    }
    """
    track_name = request.get('track_name')
    race_session = request.get('race_session')
    min_lap_time = request.get('min_lap_time', 25.0)
    
    if not track_name or not race_session:
        raise HTTPException(status_code=400, detail="track_name and race_session are required")
    
    if not predictive_models or not predictive_databases:
        raise HTTPException(status_code=503, detail="Predictive models are still loading. Please wait a few seconds and try again.")
    
    try:
        result = await generate_post_event_analysis(
            track_name=track_name,
            race_session=race_session,
            predictive_models=predictive_models,
            predictive_databases=predictive_databases,
            min_lap_time=min_lap_time
        )
        return clean_nan_values(result)
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating post-event analysis: {str(e)}")


# ==================== DRIVER TRAINING & INSIGHTS ====================

@app.get("/api/driver/{vehicle_id}/racing-line")
async def racing_line_endpoint(
    vehicle_id: str,
    lap1: int,
    lap2: int,
    event_name: Optional[str] = None
):
    """Compare racing lines between two laps"""
    result = await get_racing_line_comparison(vehicle_id, lap1, lap2, event_name)
    return clean_nan_values(result)

@app.get("/api/driver/{vehicle_id}/braking")
async def braking_analysis_endpoint(
    vehicle_id: str,
    lap: Optional[int] = None,
    event_name: Optional[str] = None
):
    """Analyze braking points and patterns"""
    result = await get_braking_analysis(vehicle_id, lap, event_name)
    return clean_nan_values(result)

@app.get("/api/driver/{vehicle_id}/cornering")
async def corner_analysis_endpoint(
    vehicle_id: str,
    lap: Optional[int] = None,
    event_name: Optional[str] = None
):
    """Analyze cornering performance"""
    result = await get_corner_analysis(vehicle_id, lap, event_name)
    return clean_nan_values(result)

@app.get("/api/driver/{vehicle_id}/improvements")
async def improvement_opportunities_endpoint(vehicle_id: str, event_name: Optional[str] = None):
    """Identify improvement opportunities for a driver"""
    try:
        result = await get_driver_improvement_opportunities(vehicle_id, event_name)
        return clean_nan_values(result)
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error in improvements analysis: {str(e)}")

@app.get("/api/driver/{vehicle_id}/speed-trace")
async def speed_trace_endpoint(
    vehicle_id: str,
    lap1: int,
    lap2: int,
    event_name: Optional[str] = None
):
    """Compare speed traces between two laps"""
    result = await get_speed_trace_comparison(vehicle_id, lap1, lap2, event_name)
    return clean_nan_values(result)

@app.get("/api/driver/{vehicle_id}/best-worst")
async def best_worst_lap_endpoint(vehicle_id: str, event_name: Optional[str] = None):
    """Compare best and worst laps for a driver"""
    try:
        result = await get_best_worst_lap_analysis(vehicle_id, event_name)
        return clean_nan_values(result)
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"❌ Error in best-worst analysis for {vehicle_id}:")
        print(error_trace)
        raise HTTPException(status_code=500, detail=f"Error in best-worst analysis: {str(e)}")

@app.get("/api/driver/{vehicle_id}/ai-insights")
async def ai_insights_endpoint(vehicle_id: str):
    """Get AI-powered driver insights and recommendations"""
    try:
        result = await get_ai_driver_insights(vehicle_id)
        return clean_nan_values(result)
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"❌ Error in AI insights for {vehicle_id}:")
        print(error_trace)
        raise HTTPException(status_code=500, detail=f"Error in AI insights: {str(e)}")

@app.get("/api/driver/{vehicle_id}/sector/{sector}/ai-analysis")
async def sector_ai_analysis_endpoint(vehicle_id: str, sector: str):
    """Get AI-powered analysis for a specific sector"""
    result = await get_sector_ai_analysis(vehicle_id, sector)
    return clean_nan_values(result)


# ==================== PREDICTIVE MODELS & OPTIMIZATION ====================

@app.get("/api/driver/{vehicle_id}/performance-prediction")
async def performance_prediction_endpoint(
    vehicle_id: str,
    future_laps: int = 5
):
    """
    Predict future performance trajectory using advanced ML models
    
    Uses ensemble of models:
    - Linear Regression
    - Random Forest (if available)
    - Exponential Smoothing
    
    Returns best prediction with confidence intervals
    """
    try:
        result = await predict_performance_trajectory(vehicle_id, future_laps)
        return clean_nan_values(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error in performance prediction: {str(e)}")

@app.get("/api/driver/{vehicle_id}/optimal-racing-line")
async def optimal_racing_line_endpoint(
    vehicle_id: str,
    lap_numbers: Optional[str] = None,
    segment_size: int = 50,
    smooth_window: int = 3,
    include_telemetry: bool = False
):
    """
    Calculate optimal racing line from multiple laps
    
    Uses GPS clustering and speed-weighted path optimization.
    If lap_numbers not provided, uses best laps automatically.
    
    Query params:
    - lap_numbers: Comma-separated list of lap numbers (e.g., "8,12,15")
    - segment_size: Track segment size in meters (default: 50, min: 20, max: 100)
    - smooth_window: Smoothing window size (default: 3, min: 1, max: 10)
    - include_telemetry: Include throttle/brake/gear data (default: false)
    """
    try:
        # Validate parameters
        if segment_size < 20 or segment_size > 100:
            raise HTTPException(status_code=400, detail="segment_size must be between 20 and 100 meters")
        if smooth_window < 1 or smooth_window > 10:
            raise HTTPException(status_code=400, detail="smooth_window must be between 1 and 10")
        
        lap_list = None
        if lap_numbers:
            try:
                lap_list = [int(x.strip()) for x in lap_numbers.split(',')]
                # Filter invalid lap numbers (e.g., 32768)
                lap_list = [lap for lap in lap_list if 1 <= lap <= 1000]
                if not lap_list:
                    raise HTTPException(status_code=400, detail="No valid lap numbers provided (must be 1-1000)")
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid lap_numbers format. Use comma-separated integers.")
        
        result = await calculate_optimal_racing_line(
            vehicle_id, 
            lap_list, 
            segment_size=segment_size,
            smooth_window=smooth_window,
            include_telemetry=include_telemetry
        )
        return clean_nan_values(result)
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error calculating optimal racing line: {str(e)}")

@app.get("/api/driver/{vehicle_id}/training-plan")
async def training_plan_endpoint(vehicle_id: str):
    """
    Generate personalized training plan based on driver performance
    
    Combines:
    - Sector analysis
    - Consistency metrics
    - Improvement opportunities
    - Performance predictions
    
    Returns structured training plan with focus areas, sessions, and goals
    """
    try:
        result = await generate_training_plan(vehicle_id)
        return clean_nan_values(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating training plan: {str(e)}")

@app.get("/api/driver/{vehicle_id}/sector/{sector}/prediction")
async def sector_prediction_endpoint(
    vehicle_id: str,
    sector: str,
    future_laps: int = 5
):
    """
    Predict future sector performance using time series analysis
    
    Provides:
    - Predicted sector times for future laps
    - Confidence intervals
    - Trend analysis
    """
    try:
        result = await predict_sector_performance(vehicle_id, sector, future_laps)
        return clean_nan_values(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error in sector prediction: {str(e)}")


# ==================== REAL-TIME ANALYTICS ====================

@app.get("/api/realtime/gaps")
async def realtime_gaps_endpoint(timestamp: Optional[str] = None):
    """Calculate real-time gaps between drivers"""
    result = await calculate_real_time_gaps(timestamp)
    return clean_nan_values(result)

@app.get("/api/realtime/pit-window/{vehicle_id}")
async def pit_window_endpoint(
    vehicle_id: str,
    current_lap: int,
    total_laps: int = 27
):
    """Analyze optimal pit stop window"""
    try:
        result = await analyze_pit_window(vehicle_id, current_lap, total_laps)
        return clean_nan_values(result)
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error analyzing pit window: {str(e)}")

@app.get("/api/realtime/strategy/{vehicle_id}", response_class=Response)
async def strategy_simulation_endpoint(
    vehicle_id: str,
    pit_lap: int,
    pit_time: float = 25.0,
    total_laps: int = 27
):
    """Simulate pit stop strategy scenario"""
    result = await simulate_strategy_scenario(vehicle_id, pit_lap, pit_time, total_laps)
    cleaned = clean_nan_values(result)
    
    # Manually encode JSON with custom encoder to handle all edge cases
    def json_serializer(obj):
        """Custom JSON serializer for numpy types and NaN"""
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        elif pd.isna(obj):
            return None
        raise TypeError(f"Type {type(obj)} not serializable")
    
    try:
        json_str = json.dumps(cleaned, default=json_serializer)
        return Response(content=json_str, media_type="application/json")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error serializing strategy: {str(e)}")

@app.get("/api/realtime/tire-degradation/{vehicle_id}")
async def tire_degradation_endpoint(vehicle_id: str):
    """Estimate tire degradation based on lap time trends"""
    result = await get_tire_degradation_estimate(vehicle_id)
    return clean_nan_values(result)

@app.get("/api/realtime/strategy-insights/{vehicle_id}")
async def realtime_strategy_insights_endpoint(
    vehicle_id: str,
    current_lap: int = 15,
    total_laps: int = 27,
    pit_lap: int = 15,
    pit_time: float = 25.0
):
    """Get AI-powered real-time strategy insights for race engineer decision-making"""
    try:
        # Gather all real-time data with error handling
        gaps = None
        pit_window = None
        strategy = None
        tire_degradation = None
        
        try:
            gaps = await calculate_real_time_gaps()
        except Exception as e:
            print(f"⚠️ Warning: Failed to get gaps data: {e}")
            gaps = {"gaps": [], "leader": {}}
        
        try:
            pit_window = await analyze_pit_window(vehicle_id, current_lap, total_laps)
        except HTTPException as e:
            print(f"⚠️ Warning: Failed to get pit window for {vehicle_id}: {e.detail}")
            pit_window = {
                "vehicle_id": vehicle_id,
                "current_lap": current_lap,
                "total_laps": total_laps,
                "error": str(e.detail)
            }
        except Exception as e:
            print(f"⚠️ Warning: Failed to get pit window: {e}")
            pit_window = {
                "vehicle_id": vehicle_id,
                "current_lap": current_lap,
                "total_laps": total_laps,
                "error": str(e)
            }
        
        try:
            strategy = await simulate_strategy_scenario(vehicle_id, pit_lap, pit_time, total_laps)
        except HTTPException as e:
            print(f"⚠️ Warning: Failed to simulate strategy for {vehicle_id}: {e.detail}")
            strategy = {
                "vehicle_id": vehicle_id,
                "pit_lap": pit_lap,
                "error": str(e.detail)
            }
        except Exception as e:
            print(f"⚠️ Warning: Failed to simulate strategy: {e}")
            strategy = {
                "vehicle_id": vehicle_id,
                "pit_lap": pit_lap,
                "error": str(e)
            }
        
        try:
            tire_degradation = await get_tire_degradation_estimate(vehicle_id)
        except HTTPException as e:
            print(f"⚠️ Warning: Failed to get tire degradation for {vehicle_id}: {e.detail}")
            tire_degradation = {
                "vehicle_id": vehicle_id,
                "error": str(e.detail)
            }
        except Exception as e:
            print(f"⚠️ Warning: Failed to get tire degradation: {e}")
            tire_degradation = {
                "vehicle_id": vehicle_id,
                "error": str(e)
            }
        
        # Find gap data for this specific vehicle (try multiple matching strategies)
        vehicle_gap = None
        if gaps and gaps.get('gaps'):
            # Try exact vehicle_id match first
            vehicle_gap = next((g for g in gaps['gaps'] if str(g.get('vehicle_id')) == str(vehicle_id)), None)
            
            # If not found, try car_number match
            if not vehicle_gap:
                try:
                    from vehicle_mapping import normalize_vehicle_id
                    car_number, _ = normalize_vehicle_id(vehicle_id)
                    # Try matching by car_number field
                    vehicle_gap = next((g for g in gaps['gaps'] if str(g.get('car_number')) == str(car_number)), None)
                    # Also try matching vehicle_id field with car_number
                    if not vehicle_gap:
                        vehicle_gap = next((g for g in gaps['gaps'] if str(g.get('vehicle_id')) == str(car_number)), None)
                except:
                    pass
            
            # If still not found, try extracting number from vehicle_id format
            if not vehicle_gap and '-' in vehicle_id:
                parts = vehicle_id.split('-')
                # Try chassis number (middle part)
                if len(parts) >= 2:
                    chassis = parts[1].lstrip('0') or '0'
                    vehicle_gap = next((g for g in gaps['gaps'] if str(g.get('vehicle_id')) == chassis or str(g.get('car_number')) == chassis), None)
                # Try car number (last part)
                if not vehicle_gap and len(parts) >= 3:
                    car_num = parts[2].lstrip('0') or '0'
                    if car_num != '0' and car_num != '000':
                        vehicle_gap = next((g for g in gaps['gaps'] if str(g.get('vehicle_id')) == car_num or str(g.get('car_number')) == car_num), None)
            
            # If still not found, use first available gap as fallback (for demo purposes)
            if not vehicle_gap and len(gaps['gaps']) > 0:
                vehicle_gap = gaps['gaps'][0]
                print(f"⚠️ Vehicle {vehicle_id} not found in gaps, using first available gap as fallback")
        
        # Create a simplified gaps structure for Gemini
        gaps_for_gemini = {
            'gaps': [vehicle_gap] if vehicle_gap else [],
            'leader': gaps.get('leader', {}) if gaps else {}
        }
        
        # Generate Gemini insights (even if some data is missing)
        from gemini_insights import generate_realtime_strategy_insights
        gemini_insights = generate_realtime_strategy_insights(
            vehicle_id, gaps_for_gemini, pit_window or {}, strategy or {}, tire_degradation or {}
        )
        
        result = {
            "vehicle_id": vehicle_id,
            "gaps": clean_nan_values(gaps) if gaps else {},
            "pit_window": clean_nan_values(pit_window) if pit_window else {},
            "strategy": clean_nan_values(strategy) if strategy else {},
            "tire_degradation": clean_nan_values(tire_degradation) if tire_degradation else {},
            "gemini_insights": gemini_insights
        }
        
        return clean_nan_values(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating strategy insights: {str(e)}")

# ==================== PREDICTIVE MODEL INTEGRATION FOR REAL-TIME ====================

@app.get("/api/realtime/predict-next-laps/{vehicle_id}")
async def predict_next_laps_realtime(
    vehicle_id: str,
    current_lap: int,
    laps_ahead: int = 5,
    track_name: str = "Barber",
    race_session: str = "R1"
):
    """Use predictive model to predict next N laps in real-time"""
    try:
        if track_name not in predictive_models:
            raise HTTPException(status_code=400, detail=f"Model not loaded for track: {track_name}")
        
        model = predictive_models[track_name]
        db = predictive_databases[track_name]
        
        # Get base features for current lap
        base_features = get_features_for_lap(db, race_session, vehicle_id, current_lap)
        
        if base_features is None or base_features.empty:
            # Try to find any data for this vehicle
            template_row = db[
                (db['meta_session'] == race_session) &
                (db['vehicle_id'] == vehicle_id)
            ].head(1).copy()
            
            if template_row.empty:
                raise HTTPException(status_code=404, detail=f"No data found for vehicle {vehicle_id}")
            
            base_features = template_row
            base_features.loc[:, 'lap'] = current_lap
        
        # Predict next laps
        predicted_times = []
        current_features = base_features.copy()
        row_index = current_features.index[0]
        
        # Get initial values
        default_value = 100.92  # Default lap time
        last_lap_time = current_features.loc[row_index].get('last_normal_lap_time', default_value)
        if pd.isna(last_lap_time) or last_lap_time <= 0:
            last_lap_time = default_value
        
        lap_history = [last_lap_time]
        original_fuel_proxy = current_features.loc[row_index].get('fuel_load_proxy', np.nan)
        original_laps_on_tires = current_features.loc[row_index].get('laps_on_tires', 1)
        
        for i in range(laps_ahead):
            lap_num = current_lap + i + 1
            current_features.loc[row_index, 'lap'] = lap_num
            current_features.loc[row_index, 'is_out_lap'] = 0
            current_features.loc[row_index, 'is_normal_lap'] = 1
            current_features.loc[row_index, 'laps_on_tires'] = original_laps_on_tires + i + 1
            
            if i > 0:
                current_features.loc[row_index, 'last_normal_lap_time'] = predicted_times[-1]
                lap_history.append(predicted_times[-1])
                if len(lap_history) > 3:
                    lap_history.pop(0)
                current_features.loc[row_index, 'rolling_3_normal_lap_avg'] = np.mean(lap_history)
            
            if not pd.isna(original_fuel_proxy):
                current_features.loc[row_index, 'fuel_load_proxy'] = original_fuel_proxy - i - 1
            
            predicted_time = predict_single_lap(model, current_features)
            predicted_times.append(float(predicted_time))
        
        # Calculate confidence based on model performance
        confidence = 85.0  # Base confidence for predictive model
        
        return {
            "vehicle_id": vehicle_id,
            "current_lap": current_lap,
            "predicted_next_laps": predicted_times,
            "laps_ahead": laps_ahead,
            "confidence": confidence,
            "average_predicted_time": float(np.mean(predicted_times)),
            "best_predicted_time": float(np.min(predicted_times)),
            "worst_predicted_time": float(np.max(predicted_times)),
            "recommendations": [
                f"Predicted average lap time: {np.mean(predicted_times):.2f}s",
                f"Best predicted lap: {np.min(predicted_times):.2f}s",
                f"Tire degradation expected: {np.max(predicted_times) - np.min(predicted_times):.2f}s over {laps_ahead} laps"
            ]
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error predicting next laps: {str(e)}")

@app.get("/api/realtime/optimal-pit-timing/{vehicle_id}")
async def optimal_pit_timing_predictive(
    vehicle_id: str,
    current_lap: int,
    total_laps: int = 27,
    track_name: str = "Barber",
    race_session: str = "R1"
):
    """Use predictive model to find optimal pit stop timing"""
    try:
        if track_name not in predictive_models:
            raise HTTPException(status_code=400, detail=f"Model not loaded for track: {track_name}")
        
        scenarios = []
        remaining_laps = total_laps - current_lap
        
        # Test different pit stop scenarios
        for pit_lap in range(current_lap + 1, min(current_lap + remaining_laps - 5, total_laps - 5)):
            try:
                # Predict before pit stop
                before_pit = await simulate_stint({
                    'track_name': track_name,
                    'race_session': race_session,
                    'vehicle_id': vehicle_id,
                    'start_lap': current_lap,
                    'stint_length': pit_lap - current_lap,
                    'is_pit_stop': False
                })
                
                # Predict after pit stop
                after_pit = await simulate_stint({
                    'track_name': track_name,
                    'race_session': race_session,
                    'vehicle_id': vehicle_id,
                    'start_lap': pit_lap,
                    'stint_length': total_laps - pit_lap,
                    'is_pit_stop': True
                })
                
                pit_time = PIT_STOP_TIME.get(track_name, 30.0)
                total_time = (
                    sum(before_pit['predicted_lap_times']) +
                    sum(after_pit['predicted_lap_times']) +
                    pit_time
                )
                
                scenarios.append({
                    'pit_lap': pit_lap,
                    'total_time': float(total_time),
                    'before_pit_laps': len(before_pit['predicted_lap_times']),
                    'after_pit_laps': len(after_pit['predicted_lap_times']),
                    'pit_time': pit_time,
                    'predicted_times': before_pit['predicted_lap_times'] + after_pit['predicted_lap_times']
                })
            except Exception as e:
                print(f"⚠️ Error testing pit lap {pit_lap}: {e}")
                continue
        
        if not scenarios:
            raise HTTPException(status_code=400, detail="Could not generate pit stop scenarios")
        
        # Find optimal scenario
        optimal = min(scenarios, key=lambda x: x['total_time'])
        
        return {
            "vehicle_id": vehicle_id,
            "current_lap": current_lap,
            "total_laps": total_laps,
            "optimal_pit_lap": optimal['pit_lap'],
            "optimal_total_time": optimal['total_time'],
            "time_saved_vs_no_pit": None,  # Can be calculated if needed
            "all_scenarios": scenarios[:10],  # Limit to top 10
            "recommendation": f"Optimal pit stop at lap {optimal['pit_lap']} for predicted total time of {optimal['total_time']:.2f}s",
            "confidence": 80.0
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error calculating optimal pit timing: {str(e)}")

@app.post("/api/realtime/strategy-comparison-predictive")
async def strategy_comparison_predictive(request: dict = Body(...)):
    """Compare multiple strategies using predictive model"""
    try:
        vehicle_id = request.get('vehicle_id')
        current_lap = int(request.get('current_lap', 1))
        total_laps = int(request.get('total_laps', 27))
        track_name = request.get('track_name', 'Barber')
        race_session = request.get('race_session', 'R1')
        pit_lap = int(request.get('pit_lap', current_lap + 10))
        
        if track_name not in predictive_models:
            raise HTTPException(status_code=400, detail=f"Model not loaded for track: {track_name}")
        
        strategies = {}
        
        # Strategy 1: No pit stop
        try:
            no_pit = await simulate_stint({
                'track_name': track_name,
                'race_session': race_session,
                'vehicle_id': vehicle_id,
                'start_lap': current_lap,
                'stint_length': total_laps - current_lap,
                'is_pit_stop': False
            })
            strategies['no_pit'] = {
                'total_time': float(sum(no_pit['predicted_lap_times'])),
                'lap_times': no_pit['predicted_lap_times'],
                'average_lap': float(np.mean(no_pit['predicted_lap_times'])),
                'best_lap': float(np.min(no_pit['predicted_lap_times'])),
                'worst_lap': float(np.max(no_pit['predicted_lap_times']))
            }
        except Exception as e:
            print(f"⚠️ Error in no-pit strategy: {e}")
        
        # Strategy 2: With pit stop
        try:
            before_pit = await simulate_stint({
                'track_name': track_name,
                'race_session': race_session,
                'vehicle_id': vehicle_id,
                'start_lap': current_lap,
                'stint_length': pit_lap - current_lap,
                'is_pit_stop': False
            })
            
            after_pit = await simulate_stint({
                'track_name': track_name,
                'race_session': race_session,
                'vehicle_id': vehicle_id,
                'start_lap': pit_lap,
                'stint_length': total_laps - pit_lap,
                'is_pit_stop': True
            })
            
            pit_time = PIT_STOP_TIME.get(track_name, 30.0)
            all_lap_times = before_pit['predicted_lap_times'] + after_pit['predicted_lap_times']
            
            strategies['with_pit'] = {
                'total_time': float(sum(before_pit['predicted_lap_times']) + sum(after_pit['predicted_lap_times']) + pit_time),
                'pit_lap': pit_lap,
                'pit_time': pit_time,
                'lap_times': all_lap_times,
                'before_pit_laps': before_pit['predicted_lap_times'],
                'after_pit_laps': after_pit['predicted_lap_times'],
                'average_lap': float(np.mean(all_lap_times)),
                'best_lap': float(np.min(all_lap_times)),
                'worst_lap': float(np.max(all_lap_times))
            }
        except Exception as e:
            print(f"⚠️ Error in pit strategy: {e}")
        
        # Determine best strategy
        if strategies.get('no_pit') and strategies.get('with_pit'):
            no_pit_time = strategies['no_pit']['total_time']
            pit_time = strategies['with_pit']['total_time']
            best_strategy = 'no_pit' if no_pit_time < pit_time else 'with_pit'
            time_difference = abs(no_pit_time - pit_time)
        else:
            best_strategy = list(strategies.keys())[0] if strategies else None
            time_difference = 0
        
        return {
            "vehicle_id": vehicle_id,
            "current_lap": current_lap,
            "total_laps": total_laps,
            "strategies": strategies,
            "best_strategy": best_strategy,
            "time_difference": float(time_difference),
            "recommendation": f"Best strategy: {best_strategy.replace('_', ' ').toUpperCase()} ({time_difference:.2f}s difference)" if best_strategy else "Unable to determine best strategy",
            "confidence": 85.0
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error comparing strategies: {str(e)}")


@app.post("/api/preprocess")
async def preprocess_telemetry_endpoint(request: dict = None):
    """
    Preprocess raw telemetry data into per-vehicle CSV files
    
    Request body (optional):
    {
        "input_file": "path/to/input.csv",  # Optional, will search common locations
        "output_dir": "path/to/output"      # Optional, defaults to logs/vehicles
    }
    
    Returns:
        dict: Status and results of preprocessing
    """
    input_file = None
    output_dir = None
    
    if request:
        input_file = request.get("input_file")
        output_dir = request.get("output_dir")
    
    result = await preprocess_telemetry_data(input_file, output_dir)
    
    # If preprocessing was successful, optionally reload telemetry data
    if result.get("status") == "success":
        # Optionally reload telemetry data if it was loaded before
        global telemetry_data_loaded
        if telemetry_data_loaded:
            print("🔄 Reloading telemetry data after preprocessing...")
            telemetry_data_loaded = False
            await load_telemetry_data()
    
    return result


# ==================== PREDICTIVE ANALYSIS ENDPOINTS ====================

# Global variables for predictive models
predictive_models = {}
predictive_databases = {}

PIT_STOP_TIME = {
    'Barber': 34.0,
    'Circuit of the Americas': 36.0,
    'Road America': 52.0,
    'Sonoma': 45.0,
    'Virginia International Raceway': 25.0
}

def get_features_for_lap(db, race_session, vehicle_id, lap):
    """Looks up the feature vector using boolean masking. Returns a 1-row DataFrame."""
    try:
        mask = (db['meta_session'] == race_session) & \
               (db['vehicle_id'] == vehicle_id) & \
               (db['lap'] == lap)
        feature_df = db[mask].copy()
        
        if len(feature_df) == 0:
            print(f"LOOKUP FAILED for: {(race_session, vehicle_id, lap)}")
            return None
        return feature_df
    except Exception as e:
        print(f"Error in get_features_for_lap: {e}")
        return None

def predict_single_lap(model, feature_df):
    """Calls the model with a 1-row feature DataFrame."""
    pred_log = model.predict(feature_df)
    pred_sec = np.expm1(pred_log)
    return pred_sec[0]

def get_model_features(db_dataframe):
    """Gets the X features (all columns except target)"""
    if 'lap_time_seconds' in db_dataframe.columns:
        return db_dataframe.drop(columns=['lap_time_seconds']).columns
    return db_dataframe.columns

@app.post("/api/predictive/simulate-stint")
async def simulate_stint(request: dict = Body(...)):
    """Simulate a race stint with predictions"""
    import joblib
    
    data = request
    track_name = data.get('track_name')
    race_session = data.get('race_session')
    vehicle_id = data.get('vehicle_id')
    start_lap = int(data.get('start_lap'))
    stint_length = int(data.get('stint_length'))
    is_pit_stop = data.get('is_pit_stop', False)

    default_values = {
        'Barber': 100.92,
        'Sonoma': 120.53,
        'Road America': 158.46,
        'Circuit of the Americas': 167.83,
        'Virginia International Raceway': 133.60
    }
    default_value = default_values.get(track_name, 100.92)

    if track_name not in predictive_models:
        raise HTTPException(status_code=400, detail=f"Model not loaded for track: {track_name}")
    
    model = predictive_models[track_name]
    db = predictive_databases[track_name]

    if not all([race_session, vehicle_id, start_lap, stint_length is not None]):
        raise HTTPException(status_code=400, detail="Missing required fields.")

    base_features = get_features_for_lap(db, race_session, vehicle_id, start_lap)
    
    if base_features is None or base_features.empty:
        if start_lap == 1:
            template_row = db[
                (db['meta_session'] == race_session) &
                (db['vehicle_id'] == vehicle_id)
            ].head(1).copy()

            if template_row.empty:
                raise HTTPException(status_code=404, detail=f"No data found AT ALL for vehicle {vehicle_id} in {race_session}.")

            base_features = template_row
            base_features.loc[:, 'lap'] = 1
            base_features.loc[:, 'is_out_lap'] = 1
            base_features.loc[:, 'is_normal_lap'] = 0
            base_features.loc[:, 'laps_on_tires'] = 1
            base_features.loc[:, 'last_normal_lap_time'] = np.nan
            base_features.loc[:, 'rolling_3_normal_lap_avg'] = np.nan
        else:
            raise HTTPException(status_code=404, detail=f"No historical data for {(race_session, vehicle_id, start_lap)}")

    lap_times, true_times = [], []
    current_features = base_features.copy()
    row_index = current_features.index[0] 
    lap_history = [
        current_features.loc[row_index].get('last_normal_lap_time', default_value)
    ]
    lap_history = [t for t in lap_history if pd.notna(t) and t > 0]

    original_fuel_proxy = current_features.loc[row_index].get('fuel_load_proxy', np.nan)
    original_laps_on_tires = current_features.loc[row_index].get('laps_on_tires', 0)

    tire_reset_value = 1
    fuel_reset_adjustment = -8 if is_pit_stop else 0

    for i in range(stint_length):
        current_lap_num_to_predict = start_lap + i
        current_features.loc[row_index, 'lap'] = current_lap_num_to_predict

        if is_pit_stop and i == 0:
            current_features.loc[row_index, 'is_out_lap'] = 1
            current_features.loc[row_index, 'is_normal_lap'] = 0
            current_features.loc[row_index, 'laps_on_tires'] = tire_reset_value
        else:
            current_features.loc[row_index, 'is_out_lap'] = 0
            current_features.loc[row_index, 'is_normal_lap'] = 1
            current_features.loc[row_index, 'laps_on_tires'] = original_laps_on_tires + i + (1 if is_pit_stop else 0)

            if i > 0:
                prev_true = true_times[-1]
                prev_pred = lap_times[-1]
                last_time = prev_true if prev_true > 0 else prev_pred
                current_features.loc[row_index, 'last_normal_lap_time'] = last_time

                lap_history.append(last_time)
                if len(lap_history) > 3:
                    lap_history.pop(0)
                current_features.loc[row_index, 'rolling_3_normal_lap_avg'] = np.mean(lap_history)

        if not pd.isna(original_fuel_proxy):
            current_features.loc[row_index, 'fuel_load_proxy'] = (
                original_fuel_proxy - i + fuel_reset_adjustment
            )

        predicted_time = predict_single_lap(model, current_features)

        if is_pit_stop and i == 0:
            pit_time = PIT_STOP_TIME.get(track_name, 0)
            predicted_time += pit_time

        lap_times.append(float(predicted_time))

        true_row = db[
            (db['meta_session'] == race_session) &
            (db['vehicle_id'] == vehicle_id) &
            (db['lap'] == current_lap_num_to_predict)
        ]
        true_time = float(true_row.iloc[0]['lap_time_seconds']) if len(true_row) > 0 else 0.0
        true_times.append(true_time)

    return {
        "race_session": race_session,
        "vehicle_id": vehicle_id,
        "start_lap": start_lap,
        "predicted_lap_times": lap_times,
        "true_lap_times": true_times
    }

@app.post("/api/predictive/get-vehicles")
async def get_predictive_vehicles(request: dict = Body(...)):
    """Get unique vehicle IDs for a given race session"""
    data = request
    track_name = data.get('track_name')
    race_session = data.get('race_session')

    if not track_name:
        raise HTTPException(status_code=400, detail="Missing 'track_name' in request body.")
    
    if not race_session:
        raise HTTPException(status_code=400, detail="Missing 'race_session' in request body.")

    if not predictive_databases:
        raise HTTPException(status_code=503, detail="Predictive models are still loading. Please wait a few seconds and try again.")

    if track_name not in predictive_databases:
        available_tracks = list(predictive_databases.keys())
        raise HTTPException(
            status_code=400, 
            detail=f"Database not loaded for track: '{track_name}'. Available tracks: {available_tracks if available_tracks else 'None (models still loading)'}"
        )
    
    current_db = predictive_databases[track_name]

    try:
        mask = current_db['meta_session'] == race_session
        unique_vehicles = current_db.loc[mask, 'vehicle_id'].dropna().unique().tolist()

        if not unique_vehicles:
            return {
                "race_session": race_session,
                "vehicle_ids": [],
                "message": f"No vehicles found for race_session '{race_session}'."
            }

        return {
            "race_session": race_session,
            "vehicle_ids": sorted(unique_vehicles),
            "count": len(unique_vehicles)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve vehicles: {e}")

@app.post("/api/predictive/get-laps")
async def get_predictive_laps(request: dict = Body(...)):
    """Get lap numbers for a vehicle"""
    data = request
    track_name = data.get('track_name')
    race_session = data.get('race_session')
    vehicle_id = data.get('vehicle_id')

    if not race_session or not vehicle_id:
        raise HTTPException(status_code=400, detail="Missing 'race_session' or 'vehicle_id' in request body.")

    if track_name not in predictive_databases:
        raise HTTPException(status_code=400, detail=f"Database not loaded for track: {track_name}")
    
    current_db = predictive_databases[track_name]

    try:
        available_laps = current_db[
            (current_db['meta_session'] == race_session) &
            (current_db['vehicle_id'] == vehicle_id)
        ]['lap'].dropna().unique()

        try:
            lap_numbers = sorted([int(x) for x in available_laps])
        except Exception:
            lap_numbers = sorted(list(available_laps))

        return {"lap_numbers": lap_numbers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve laps: {e}")

@app.post("/api/predictive/get-final-results")
async def get_predictive_final_results(request: dict = Body(...)):
    """Get final race results for a track/session"""
    data = request
    track_name = data.get('track_name')
    race_session = data.get('race_session')
    MIN_LAP_TIME = int(data.get('min_lap_time_enforced', 60))

    if not track_name or not race_session:
        raise HTTPException(status_code=400, detail="Missing 'track_name' or 'race_session'")

    if track_name not in predictive_databases:
        raise HTTPException(status_code=400, detail=f"Database not loaded for track: {track_name}")
    
    current_db = predictive_databases[track_name]

    session_df = current_db[current_db['meta_session'] == race_session].copy()

    if session_df.empty:
        raise HTTPException(status_code=404, detail=f"No results found for session '{race_session}'")

    session_df['is_valid_lap'] = session_df['lap_time_seconds'] >= MIN_LAP_TIME
    valid_df = session_df[session_df['is_valid_lap']]
    max_lap = int(valid_df['lap'].max())

    results = []

    for vehicle_id, group in session_df.groupby('vehicle_id'):
        valid_group = group[group['is_valid_lap']]
        completed_laps = int(valid_group['lap'].max()) if not valid_group.empty else 0
        total_time = float(valid_group['lap_time_seconds'].sum()) if completed_laps > 0 else None
        best_lap_time = float(valid_group['lap_time_seconds'].min()) if completed_laps > 0 else None
        did_finish = (completed_laps == max_lap)
        invalid_laps = group[~group['is_valid_lap']]['lap'].tolist()

        results.append({
            "vehicle_id": vehicle_id,
            "completed_laps": completed_laps,
            "total_time": total_time,
            "best_lap_time": best_lap_time,
            "invalid_laps": invalid_laps,
            "status": "Finished" if did_finish else "DNF"
        })

    sorted_results = sorted(
        results,
        key=lambda x: (
            x['status'] == "DNF",
            float('inf') if x['total_time'] is None else x['total_time']
        )
    )

    return {
        "track_name": track_name,
        "race_session": race_session,
        "max_lap": max_lap,
        "min_lap_time_enforced": MIN_LAP_TIME,
        "results": sorted_results
    }

@app.post("/api/predictive/predict-new-session")
async def predict_new_session(request: dict = Body(...)):
    """Predict a new race session based on minimal user input"""
    import joblib
    
    MIN_LAP_TIME_ENFORCE = 25.0
    DEFAULT_SESSION_FOR_TEMPLATE = "R2"
    
    body = request
    track_name = body.get('track_name')
    vehicle_id = body.get('vehicle_id')
    total_laps_to_predict = int(body.get('total_laps_to_predict', 0))
    previous_laps = body.get('previous_laps', [])

    if not track_name or not vehicle_id or total_laps_to_predict <= 0:
        raise HTTPException(status_code=400, detail="track_name, vehicle_id and total_laps_to_predict are required")

    if track_name not in predictive_models:
        raise HTTPException(status_code=400, detail=f"Model not loaded for track: {track_name}")
    
    model = predictive_models[track_name]
    db_used = predictive_databases[track_name]
    
    MODEL_FEATURES = get_model_features(db_used)

    template = db_used[
        (db_used['meta_session'] == DEFAULT_SESSION_FOR_TEMPLATE) & (db_used['vehicle_id'] == vehicle_id)
    ].sort_values('lap').head(1)
    
    if template.empty:
        template = db_used[db_used['meta_session'] == DEFAULT_SESSION_FOR_TEMPLATE].head(1)
    if template.empty:
        base_row = {}
    else:
        base_row = template.iloc[0].to_dict()

    results_rows = []
    lap_history_sec = []
    start_lap = 1
    current_last_lap_time = np.nan
    current_laps_on_tires = 1
    current_fuel_load = None
    current_air_temp = base_row.get('session_air_temp', np.nan)
    current_track_temp = base_row.get('session_track_temp', np.nan)

    if previous_laps:
        prev = sorted(previous_laps, key=lambda x: int(x.get('lap', 0)))
        
        for p in prev:
            lap_time = float(p.get('lap_time_seconds')) if p.get('lap_time_seconds') is not None else None
            results_rows.append({
                'lap': int(p.get('lap')),
                'lap_time_seconds': lap_time,
                'provided': True
            })
            if lap_time is not None and lap_time >= MIN_LAP_TIME_ENFORCE:
                lap_history_sec.append(lap_time)

        last_prev_lap = prev[-1]
        start_lap = int(last_prev_lap.get('lap', 0)) + 1
        current_last_lap_time = float(last_prev_lap.get('lap_time_seconds', np.nan))
        current_laps_on_tires = int(last_prev_lap.get('laps_on_tires', 1)) + 1
        current_fuel_load = float(last_prev_lap.get('fuel_load_proxy', np.nan))
        current_air_temp = float(last_prev_lap.get('session_air_temp', current_air_temp))
        current_track_temp = float(last_prev_lap.get('session_track_temp', current_track_temp))

    for lap_num in range(start_lap, total_laps_to_predict + 1):
        feat = {col: np.nan for col in MODEL_FEATURES}
        
        feat['track'] = track_name
        feat['race_session'] = base_row.get('race_session', DEFAULT_SESSION_FOR_TEMPLATE)
        feat['meta_session'] = base_row.get('meta_session', DEFAULT_SESSION_FOR_TEMPLATE)
        feat['vehicle_id'] = vehicle_id
        feat['original_vehicle_id'] = vehicle_id
        feat['vehicle_number'] = base_row.get('vehicle_number', -1)
        feat['last_lap_time'] = current_last_lap_time
        feat['laps_on_tires'] = current_laps_on_tires
        
        if current_fuel_load is not None and not np.isnan(current_fuel_load):
            current_fuel_load -= 1
        else:
            current_fuel_load = float(max(0, total_laps_to_predict - lap_num))
        feat['fuel_load_proxy'] = current_fuel_load
        
        feat['session_air_temp'] = current_air_temp
        feat['session_track_temp'] = current_track_temp
        feat['CROSSING_FINISH_LINE_IN_PIT'] = 0 
        feat['pit_flag'] = 0
        feat['is_new_stint'] = 0
        
        feature_df = pd.DataFrame([feat], columns=MODEL_FEATURES)

        try:
            pred_log = model.predict(feature_df)
            pred_sec = float(np.expm1(pred_log)[0])
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction failed on lap {lap_num}: {e}")

        if pred_sec < MIN_LAP_TIME_ENFORCE:
            pred_sec = MIN_LAP_TIME_ENFORCE
            
        results_rows.append({
            'lap': lap_num,
            'lap_time_seconds': pred_sec,
            'provided': False
        })
        
        current_last_lap_time = pred_sec
        current_laps_on_tires += 1
        
    lap_times_all = [r['lap_time_seconds'] for r in results_rows if r['lap_time_seconds'] is not None]
    lap_times_predicted = [float(r['lap_time_seconds']) for r in results_rows if not r['provided'] and r['lap_time_seconds'] is not None]
    total_race_time = sum(lap_times_all)
    best_lap = min(lap_times_all) if lap_times_all else 0

    return {
        "track_name": track_name,
        "vehicle_id": vehicle_id,
        "total_laps_to_predict": total_laps_to_predict,
        "start_lap_predicted_from": start_lap,
        "predicted_laps": results_rows,
        "predicted_lap_times": lap_times_predicted,
        "total_predicted_time": total_race_time,
        "best_lap_time": best_lap
    }

async def load_predictive_models():
    """Load predictive models and databases on startup"""
    global predictive_models, predictive_databases
    import joblib
    import warnings
    
    # Compatibility workaround for scikit-learn version mismatch
    # Models were trained with scikit-learn 1.5.1, but newer versions removed _RemainderColsList
    try:
        from sklearn.compose import _column_transformer
        if not hasattr(_column_transformer, '_RemainderColsList'):
            # Create a dummy class for backward compatibility
            class _RemainderColsList:
                pass
            _column_transformer._RemainderColsList = _RemainderColsList
            print("⚠️ Applied scikit-learn compatibility workaround for model loading")
    except Exception as e:
        print(f"⚠️ Could not apply compatibility workaround: {e}")
    
    project_root = get_project_root()
    models_dir = project_root / "models"
    
    if not models_dir.exists():
        print("⚠️ Models directory not found. Predictive analysis endpoints will not be available.")
        return
    
    track_configs = {
        'Barber': {
            'model': 'barber_overall_model.pkl',
            'data': 'barber_features_engineered.parquet'
        },
        'Sonoma': {
            'model': 'sonoma_overall_model.pkl',
            'data': 'sonoma_features_engineered.parquet'
        },
        'Road America': {
            'model': 'road_america_overall_model.pkl',
            'data': 'road_america_features_engineered.parquet'
        },
        'Circuit of the Americas': {
            'model': 'circuit_of_the_americas_overall_model.pkl',
            'data': 'circuit_of_the_americas_features_engineered.parquet'
        },
        'Virginia International Raceway': {
            'model': 'model_virginia_international_raceway_overall.pkl',
            'data': 'virginia_international_raceway_features_engineered.parquet'
        }
    }
    
    print("\n" + "="*60)
    print("Loading Predictive Analysis Models...")
    print("="*60)
    
    for track_name, config in track_configs.items():
        model_path = models_dir / config['model']
        data_path = models_dir / config['data']
        
        if model_path.exists() and data_path.exists():
            try:
                # Load model (compatibility workaround applied at function start)
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=UserWarning)
                    model = joblib.load(str(model_path))
                predictive_models[track_name] = model
                
                # Load database
                db = pd.read_parquet(str(data_path))
                db = db.drop_duplicates(subset=['meta_session', 'vehicle_id', 'lap'], keep='first')
                predictive_databases[track_name] = db
                
                print(f"✅ Loaded {track_name}: {len(db)} rows")
            except Exception as e:
                print(f"⚠️ Failed to load {track_name}: {e}")
        else:
            print(f"⚠️ Files not found for {track_name}")
    
    print(f"✅ Loaded {len(predictive_models)} predictive models")
    print("="*60 + "\n")


# ==================== REST API ONLY - NO SSE/WEBSOCKET ====================
# Broadcast loops run in background to update cache
# Clients poll REST endpoints for updates


# ==================== STARTUP/SHUTDOWN ====================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup - Pre-load all data for fast access (non-blocking)"""
    try:
        print("\n" + "="*60)
        print("Telemetry Rush - FastAPI Server (Integrated)")
        print("="*60)
        
        # Verify critical routes are registered (quick check)
        try:
            registered_routes = [route.path for route in app.routes if hasattr(route, 'path')]
            leaderboard_routes = [r for r in registered_routes if 'leaderboard' in r.lower()]
            print(f"\n📋 Registered routes containing 'leaderboard': {leaderboard_routes}")
            
            if '/api/leaderboard' not in registered_routes:
                print("⚠️ WARNING: /api/leaderboard route not found in registered routes!")
                print(f"   Total routes registered: {len(registered_routes)}")
                print(f"   Sample routes: {registered_routes[:10]}")
            else:
                print(f"✅ Leaderboard route confirmed: {[r for r in registered_routes if 'leaderboard' in r.lower()]}")
        except Exception as e:
            print(f"⚠️ Error checking routes: {e}")
        
        # Start data loading in background (non-blocking) so server can start immediately
        # This is critical for Cloud Run which has startup timeout requirements
        print("\n🚀 Starting data pre-loading in background (non-blocking)...")
        
        # Schedule background tasks - these will run asynchronously without blocking startup
        async def load_data_background():
            """Background task to load data without blocking server startup"""
            try:
                await load_telemetry_data()
            except Exception as e:
                print(f"⚠️ Error loading telemetry data: {e}")
            try:
                await load_endurance_data()
            except Exception as e:
                print(f"⚠️ Error loading endurance data: {e}")
            try:
                await load_leaderboard_data()
            except Exception as e:
                print(f"⚠️ Error loading leaderboard data: {e}")
            try:
                await load_predictive_models()
            except Exception as e:
                print(f"⚠️ Error loading predictive models: {e}")
        
        # Create background task - this returns immediately
        asyncio.create_task(load_data_background())
        
        print("✅ Server is ready and listening (data loading in background)")
        print("="*60 + "\n")
    except Exception as e:
        print(f"⚠️ Error in startup event: {e}")
        import traceback
        traceback.print_exc()
        # Don't fail startup - server should still start even if data loading fails


if __name__ == "__main__":
    # Get port from environment variable (Render provides $PORT) or default to 8000
    port = int(os.environ.get("PORT", 8000))
    
    print("\n" + "="*60)
    print("Telemetry Rush - FastAPI Server (Integrated)")
    print(f"All services running on port {port}")
    print("="*60)
    print(f"\nStarting FastAPI server on http://0.0.0.0:{port}")
    print("Press Ctrl+C to stop")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

