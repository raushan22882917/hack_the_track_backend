"""
Predictive Models Module for Driver Training & Insights
Uses advanced ML algorithms for performance prediction and optimization
"""

from fastapi import HTTPException
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Try to import ML libraries, fallback to basic models if not available
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️ Warning: scikit-learn not available. Using basic statistical models.")


def get_project_root():
    """Get the project root directory"""
    return Path(__file__).parent


def normalize_vehicle_id(vehicle_id: str) -> tuple:
    """
    Normalize vehicle_id to find matching car number in endurance data
    
    Returns:
        (car_number, original_vehicle_id) tuple
    """
    original_id = vehicle_id
    
    # Try vehicle mapping first
    try:
        from vehicle_mapping import get_vehicle_number_as_string
        car_number = get_vehicle_number_as_string(vehicle_id)
        if car_number and car_number != '0' and car_number != '000':
            return (car_number, original_id)
    except:
        pass
    
    # Extract from GR86-XXX-YYY format
    if '-' in vehicle_id:
        parts = vehicle_id.split('-')
        if len(parts) >= 3:
            # Try car number (last part) first
            car_num = parts[2].lstrip('0') or '0'
            if car_num != '000' and car_num != '0':
                return (car_num, original_id)
            
            # If car number is 000/0, try chassis number (middle part)
            chassis_num = parts[1].lstrip('0') or '0'
            if chassis_num != '000' and chassis_num != '0':
                return (chassis_num, original_id)
            
            # If both are 000, return chassis as fallback
            return (chassis_num, original_id)
    
    # Return as-is if no conversion possible
    return (vehicle_id, original_id)


def find_vehicle_in_endurance_data(endurance_df: pd.DataFrame, vehicle_id: str) -> pd.DataFrame:
    """
    Find vehicle data in endurance DataFrame using multiple matching strategies
    
    Returns:
        DataFrame with vehicle data, or empty DataFrame if not found
    """
    car_number, original_id = normalize_vehicle_id(vehicle_id)
    
    # Strategy 1: Match by normalized car number (try both string and int)
    try:
        vehicle_data = endurance_df[endurance_df['NUMBER'].astype(str) == str(car_number)]
        if len(vehicle_data) > 0:
            return vehicle_data
        
        # Also try as integer
        try:
            car_num_int = int(car_number)
            vehicle_data = endurance_df[endurance_df['NUMBER'] == car_num_int]
            if len(vehicle_data) > 0:
                return vehicle_data
        except ValueError:
            pass
    except:
        pass
    
    # Strategy 2: Match by original vehicle_id
    try:
        vehicle_data = endurance_df[endurance_df['NUMBER'].astype(str) == str(original_id)]
        if len(vehicle_data) > 0:
            return vehicle_data
    except:
        pass
    
    # Strategy 3: Try extracting from GR86 format
    if '-' in vehicle_id:
        parts = vehicle_id.split('-')
        # Try middle part (chassis) - this is often the NUMBER in endurance data
        if len(parts) >= 2:
            chassis = parts[1].lstrip('0') or '0'
            try:
                vehicle_data = endurance_df[endurance_df['NUMBER'].astype(str) == chassis]
                if len(vehicle_data) > 0:
                    return vehicle_data
                # Also try as integer
                try:
                    chassis_int = int(chassis)
                    vehicle_data = endurance_df[endurance_df['NUMBER'] == chassis_int]
                    if len(vehicle_data) > 0:
                        return vehicle_data
                except ValueError:
                    pass
            except:
                pass
        
        # Try last part (car number)
        if len(parts) >= 3:
            car_num = parts[2].lstrip('0') or '0'
            if car_num != '0' and car_num != '000':
                try:
                    vehicle_data = endurance_df[endurance_df['NUMBER'].astype(str) == car_num]
                    if len(vehicle_data) > 0:
                        return vehicle_data
                    # Also try as integer
                    try:
                        car_num_int = int(car_num)
                        vehicle_data = endurance_df[endurance_df['NUMBER'] == car_num_int]
                        if len(vehicle_data) > 0:
                            return vehicle_data
                    except ValueError:
                        pass
                except:
                    pass
    
    # Return empty DataFrame if not found
    return pd.DataFrame()


def parse_lap_time(lap_time_str: str) -> float:
    """Convert lap time string (e.g., '1:37.428') to seconds"""
    if pd.isna(lap_time_str) or not lap_time_str:
        return None
    try:
        parts = str(lap_time_str).split(':')
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(parts[0])
    except:
        return None


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two GPS points in meters (Haversine formula)"""
    from math import radians, sin, cos, sqrt, atan2
    
    R = 6371000  # Earth radius in meters
    lat1_rad = radians(lat1)
    lat2_rad = radians(lat2)
    delta_lat = radians(lat2 - lat1)
    delta_lon = radians(lon2 - lon1)
    
    a = sin(delta_lat/2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(delta_lon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    return R * c


async def predict_performance_trajectory(vehicle_id: str, future_laps: int = 5) -> Dict:
    """
    Predict future performance trajectory using time series forecasting
    
    Uses multiple models:
    1. Linear Regression (baseline)
    2. Random Forest (if available)
    3. Exponential Smoothing (statistical)
    
    Returns best prediction with confidence intervals
    """
    project_root = get_project_root()
    endurance_file = project_root / "logs" / "R1_section_endurance.csv"
    
    if not endurance_file.exists():
        raise HTTPException(status_code=404, detail="Endurance data not found")
    
    endurance_df = pd.read_csv(endurance_file, sep=";", low_memory=False)
    endurance_df.columns = endurance_df.columns.str.strip()
    
    # Use helper function to find vehicle data
    vehicle_data = find_vehicle_in_endurance_data(endurance_df, vehicle_id)
    
    if len(vehicle_data) == 0:
        available_numbers = sorted(endurance_df['NUMBER'].dropna().unique().astype(str).tolist())
        car_number, _ = normalize_vehicle_id(vehicle_id)
        raise HTTPException(
            status_code=404, 
            detail=f"Vehicle {vehicle_id} (tried car_number: {car_number}) not found in endurance data. Available vehicles: {available_numbers[:20]}"
        )
    
    vehicle_data = vehicle_data.sort_values('LAP_NUMBER')
    vehicle_data['lap_time_seconds'] = vehicle_data['LAP_TIME'].apply(parse_lap_time)
    valid_laps = vehicle_data[vehicle_data['lap_time_seconds'].notna()]
    
    if len(valid_laps) < 3:
        raise HTTPException(status_code=400, detail="Insufficient data for prediction (need at least 3 laps)")
    
    lap_times = valid_laps['lap_time_seconds'].values
    lap_numbers = valid_laps['LAP_NUMBER'].values
    
    predictions = []
    models_used = []
    
    # Model 1: Linear Regression (always available)
    X = lap_numbers.reshape(-1, 1)
    y = lap_times
    
    lr_model = LinearRegression()
    lr_model.fit(X, y)
    
    # Predict future laps
    future_lap_numbers = np.arange(lap_numbers.max() + 1, lap_numbers.max() + 1 + future_laps)
    lr_predictions = lr_model.predict(future_lap_numbers.reshape(-1, 1))
    
    # Calculate confidence (using residual standard error)
    residuals = y - lr_model.predict(X)
    std_error = np.std(residuals)
    
    predictions.append({
        'model': 'Linear Regression',
        'predictions': lr_predictions.tolist(),
        'confidence': max(0, min(100, 100 - (std_error / np.mean(lap_times) * 100))),
        'mae': float(np.mean(np.abs(residuals)))
    })
    models_used.append('Linear Regression')
    
    # Model 2: Random Forest (if available)
    if ML_AVAILABLE and len(lap_times) >= 5:
        try:
            # Create features: lap number, previous lap time, rolling averages
            features = []
            targets = []
            
            for i in range(2, len(lap_times)):
                feat = [
                    lap_numbers[i],
                    lap_times[i-1] if i > 0 else lap_times[i],
                    np.mean(lap_times[max(0, i-3):i]) if i >= 3 else lap_times[i],
                    np.std(lap_times[max(0, i-3):i]) if i >= 3 else 0
                ]
                features.append(feat)
                targets.append(lap_times[i])
            
            if len(features) >= 3:
                X_rf = np.array(features)
                y_rf = np.array(targets)
                
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
                rf_model.fit(X_rf, y_rf)
                
                # Predict future laps
                rf_predictions = []
                last_lap_time = lap_times[-1]
                last_lap_num = lap_numbers[-1]
                recent_times = lap_times[-3:].tolist()
                
                for future_lap in future_lap_numbers:
                    feat = [
                        future_lap,
                        last_lap_time,
                        np.mean(recent_times),
                        np.std(recent_times) if len(recent_times) > 1 else 0
                    ]
                    pred = rf_model.predict([feat])[0]
                    rf_predictions.append(pred)
                    last_lap_time = pred
                    recent_times.append(pred)
                    if len(recent_times) > 3:
                        recent_times.pop(0)
                
                # Calculate confidence
                rf_residuals = y_rf - rf_model.predict(X_rf)
                rf_std_error = np.std(rf_residuals)
                
                predictions.append({
                    'model': 'Random Forest',
                    'predictions': rf_predictions,
                    'confidence': max(0, min(100, 100 - (rf_std_error / np.mean(lap_times) * 100))),
                    'mae': float(np.mean(np.abs(rf_residuals)))
                })
                models_used.append('Random Forest')
        except Exception as e:
            print(f"⚠️ Random Forest model failed: {e}")
    
    # Model 3: Exponential Smoothing (statistical)
    try:
        # Simple exponential smoothing
        alpha = 0.3  # Smoothing parameter
        smoothed = [lap_times[0]]
        
        for i in range(1, len(lap_times)):
            smoothed.append(alpha * lap_times[i] + (1 - alpha) * smoothed[-1])
        
        # Predict future (using trend if available)
        trend = (lap_times[-1] - lap_times[0]) / len(lap_times) if len(lap_times) > 1 else 0
        es_predictions = []
        last_smoothed = smoothed[-1]
        
        for _ in range(future_laps):
            last_smoothed = alpha * (last_smoothed + trend) + (1 - alpha) * last_smoothed
            es_predictions.append(last_smoothed)
        
        # Calculate confidence
        es_residuals = lap_times - np.array(smoothed)
        es_std_error = np.std(es_residuals)
        
        predictions.append({
            'model': 'Exponential Smoothing',
            'predictions': es_predictions,
            'confidence': max(0, min(100, 100 - (es_std_error / np.mean(lap_times) * 100))),
            'mae': float(np.mean(np.abs(es_residuals)))
        })
        models_used.append('Exponential Smoothing')
    except Exception as e:
        print(f"⚠️ Exponential Smoothing failed: {e}")
    
    # Select best model (lowest MAE)
    if predictions:
        best_model = min(predictions, key=lambda x: x['mae'])
        ensemble_pred = np.mean([p['predictions'] for p in predictions], axis=0)
        
        return {
            'vehicle_id': vehicle_id,
            'current_laps': len(valid_laps),
            'current_best_lap': float(np.min(lap_times)),
            'current_average_lap': float(np.mean(lap_times)),
            'prediction_horizon': future_laps,
            'best_model': best_model['model'],
            'models_used': models_used,
            'predictions': {
                'lap_numbers': future_lap_numbers.tolist(),
                'predicted_times': ensemble_pred.tolist(),
                'best_model_predictions': best_model['predictions'],
                'confidence_interval': {
                    'lower': (ensemble_pred - std_error).tolist(),
                    'upper': (ensemble_pred + std_error).tolist()
                }
            },
            'model_comparison': predictions,
            'trend': {
                'direction': 'improving' if lr_model.coef_[0] < 0 else 'declining' if lr_model.coef_[0] > 0 else 'stable',
                'rate_per_lap': float(lr_model.coef_[0]),
                'expected_improvement': float(abs(lr_model.coef_[0]) * future_laps)
            },
            'insights': [
                f"Based on {len(valid_laps)} laps, predicted performance over next {future_laps} laps",
                f"Best model: {best_model['model']} (MAE: {best_model['mae']:.3f}s)",
                f"Trend: {('improving' if lr_model.coef_[0] < 0 else 'declining' if lr_model.coef_[0] > 0 else 'stable')} at {abs(lr_model.coef_[0]):.3f}s per lap"
            ]
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to generate predictions")


def parse_vehicle_id(vehicle_id: str) -> Dict[str, str]:
    """
    Parse vehicle ID to extract chassis and car number
    
    Format: GR86-{chassis}-{car_number}
    Example: GR86-004-78 -> chassis: 004, car_number: 78
    Special: car_number 000 means not assigned
    """
    parts = vehicle_id.split('-')
    if len(parts) >= 3:
        return {
            'chassis': parts[1],
            'car_number': parts[2],
            'is_assigned': parts[2] != '000'
        }
    return {
        'chassis': vehicle_id,
        'car_number': '000',
        'is_assigned': False
    }


def filter_invalid_laps(lap_values: pd.Series) -> pd.Series:
    """
    Filter out invalid lap numbers (e.g., 32768 which indicates corrupted data)
    
    Valid laps are typically 1-1000 for racing scenarios
    """
    return lap_values[(lap_values >= 1) & (lap_values <= 1000)]


async def calculate_optimal_racing_line(
    vehicle_id: str, 
    lap_numbers: Optional[List[int]] = None,
    segment_size: int = 50,
    smooth_window: int = 3,
    include_telemetry: bool = False
) -> Dict:
    """
    Calculate optimal racing line from multiple laps using GPS clustering
    
    Algorithm:
    1. Load GPS coordinates for specified laps (or best laps)
    2. Cluster GPS points by track position (segment_size meters)
    3. For each cluster, select point with highest speed
    4. Smooth the resulting path (smooth_window)
    5. Calculate speed profile along optimal line
    
    Parameters:
    - vehicle_id: Vehicle identifier (e.g., "GR86-004-78")
    - lap_numbers: List of lap numbers to analyze (None = use best laps)
    - segment_size: Track segment size in meters (default: 50)
    - smooth_window: Smoothing window size (default: 3)
    - include_telemetry: Include throttle/brake/gear data (default: False)
    """
    project_root = get_project_root()
    vehicle_file = project_root / "logs" / "vehicles" / f"{vehicle_id}.csv"
    endurance_file = project_root / "logs" / "R1_section_endurance.csv"
    
    if not vehicle_file.exists():
        raise HTTPException(status_code=404, detail=f"Vehicle {vehicle_id} telemetry not found")
    
    # Parse vehicle ID
    vehicle_info = parse_vehicle_id(vehicle_id)
    
    # Load endurance data to find best laps if not specified
    if lap_numbers is None or len(lap_numbers) == 0:
        if endurance_file.exists():
            endurance_df = pd.read_csv(endurance_file, sep=";", low_memory=False)
            endurance_df.columns = endurance_df.columns.str.strip()
            
            # Use helper function to find vehicle data
            vehicle_data = find_vehicle_in_endurance_data(endurance_df, vehicle_id)
            
            if len(vehicle_data) > 0:
                vehicle_data['lap_time_seconds'] = vehicle_data['LAP_TIME'].apply(parse_lap_time)
                valid_laps = vehicle_data[vehicle_data['lap_time_seconds'].notna()]
                
                # Filter invalid lap numbers
                valid_laps = valid_laps[valid_laps['LAP_NUMBER'].apply(lambda x: 1 <= x <= 1000)]
                
                if len(valid_laps) > 0:
                    # Get top 3 best laps
                    best_laps = valid_laps.nsmallest(3, 'lap_time_seconds')
                    lap_numbers = best_laps['LAP_NUMBER'].tolist()
        
        if lap_numbers is None or len(lap_numbers) == 0:
            lap_numbers = [1, 2, 3]  # Default to first 3 laps
    
    # Load telemetry data
    df = pd.read_csv(vehicle_file, parse_dates=["meta_time"], low_memory=False)
    df["meta_time"] = pd.to_datetime(df["meta_time"], utc=True, errors="coerce")
    
    all_coords = []
    
    # Extract GPS and speed data for each lap
    for lap_num in lap_numbers:
        # Filter invalid lap numbers
        if lap_num < 1 or lap_num > 1000:
            continue
            
        lap_rows = df[(df["telemetry_name"] == "lap") & (df["telemetry_value"] == lap_num)]
        
        if len(lap_rows) == 0:
            continue
        
        lap_start = lap_rows.iloc[0]["meta_time"]
        lap_end = df[df["meta_time"] > lap_start]["meta_time"].min()
        
        if pd.isna(lap_end):
            lap_end = df["meta_time"].max()
        
        lap_df = df[(df["meta_time"] >= lap_start) & (df["meta_time"] < lap_end)]
        
        # Get GPS coordinates (using correct field names from dataset guide)
        lat_data = lap_df[lap_df["telemetry_name"] == "VBOX_Lat_Min"].sort_values("meta_time")
        lon_data = lap_df[lap_df["telemetry_name"] == "VBOX_Long_Minutes"].sort_values("meta_time")
        speed_data = lap_df[lap_df["telemetry_name"] == "speed"].sort_values("meta_time")
        
        # Get telemetry data if requested
        throttle_data = lap_df[lap_df["telemetry_name"] == "ath"].sort_values("meta_time") if include_telemetry else pd.DataFrame()
        brake_data = lap_df[lap_df["telemetry_name"] == "pbrake_f"].sort_values("meta_time") if include_telemetry else pd.DataFrame()
        gear_data = lap_df[lap_df["telemetry_name"] == "gear"].sort_values("meta_time") if include_telemetry else pd.DataFrame()
        
        # Pair coordinates
        for idx in lat_data.index:
            lat = lat_data.loc[idx, "telemetry_value"]
            time = lat_data.loc[idx, "meta_time"]
            
            lon_row = lon_data[lon_data["meta_time"] == time]
            speed_row = speed_data[speed_data["meta_time"] == time]
            
            if len(lon_row) > 0:
                lon = lon_row.iloc[0]["telemetry_value"]
                speed = speed_row.iloc[0]["telemetry_value"] if len(speed_row) > 0 else 0
                
                coord_data = {
                    'lat': float(lat),
                    'lon': float(lon),
                    'speed': float(speed),
                    'lap': lap_num,
                    'time': time.isoformat()
                }
                
                # Add telemetry data if requested
                if include_telemetry:
                    throttle_row = throttle_data[throttle_data["meta_time"] == time]
                    brake_row = brake_data[brake_data["meta_time"] == time]
                    gear_row = gear_data[gear_data["meta_time"] == time]
                    
                    coord_data['throttle'] = float(throttle_row.iloc[0]["telemetry_value"]) if len(throttle_row) > 0 else None
                    coord_data['brake_pressure'] = float(brake_row.iloc[0]["telemetry_value"]) if len(brake_row) > 0 else None
                    coord_data['gear'] = int(gear_row.iloc[0]["telemetry_value"]) if len(gear_row) > 0 else None
                
                all_coords.append(coord_data)
    
    if len(all_coords) == 0:
        raise HTTPException(status_code=404, detail="No GPS data found for specified laps")
    
    # Convert to DataFrame for easier processing
    coords_df = pd.DataFrame(all_coords)
    
    # Simple clustering: divide track into segments and find fastest point in each segment
    # Sort by time (using meta_time) to maintain track order
    coords_df = coords_df.sort_values('time')
    
    # Calculate cumulative distance
    distances = [0]
    for i in range(1, len(coords_df)):
        prev = coords_df.iloc[i-1]
        curr = coords_df.iloc[i]
        dist = calculate_distance(prev['lat'], prev['lon'], curr['lat'], curr['lon'])
        distances.append(distances[-1] + dist)
    
    coords_df['cumulative_distance'] = distances
    
    # Divide into segments (configurable segment_size meters)
    max_distance = coords_df['cumulative_distance'].max()
    num_segments = int(max_distance / segment_size) + 1
    
    optimal_line = []
    
    for seg_num in range(num_segments):
        seg_start = seg_num * segment_size
        seg_end = (seg_num + 1) * segment_size
        
        segment = coords_df[
            (coords_df['cumulative_distance'] >= seg_start) &
            (coords_df['cumulative_distance'] < seg_end)
        ]
        
        if len(segment) > 0:
            # Find point with highest speed in this segment
            fastest_idx = segment['speed'].idxmax()
            fastest_point = segment.loc[fastest_idx]
            
            point_data = {
                'lat': fastest_point['lat'],
                'lon': fastest_point['lon'],
                'speed': fastest_point['speed'],
                'distance': fastest_point['cumulative_distance'],
                'segment': seg_num
            }
            
            # Add telemetry data if available
            if include_telemetry and 'throttle' in fastest_point:
                point_data['throttle'] = fastest_point.get('throttle')
                point_data['brake_pressure'] = fastest_point.get('brake_pressure')
                point_data['gear'] = fastest_point.get('gear')
            
            optimal_line.append(point_data)
    
    # Smooth the line (configurable smooth_window)
    if len(optimal_line) > smooth_window:
        smoothed_line = []
        
        for i in range(len(optimal_line)):
            start_idx = max(0, i - smooth_window // 2)
            end_idx = min(len(optimal_line), i + smooth_window // 2 + 1)
            
            window_points = optimal_line[start_idx:end_idx]
            avg_lat = np.mean([p['lat'] for p in window_points])
            avg_lon = np.mean([p['lon'] for p in window_points])
            avg_speed = np.mean([p['speed'] for p in window_points])
            
            smoothed_point = {
                'lat': float(avg_lat),
                'lon': float(avg_lon),
                'speed': float(avg_speed),
                'distance': optimal_line[i]['distance'],
                'segment': optimal_line[i]['segment']
            }
            
            # Average telemetry data if available
            if include_telemetry and 'throttle' in optimal_line[i]:
                throttles = [p.get('throttle', 0) for p in window_points if p.get('throttle') is not None]
                brakes = [p.get('brake_pressure', 0) for p in window_points if p.get('brake_pressure') is not None]
                gears = [p.get('gear', 0) for p in window_points if p.get('gear') is not None]
                
                smoothed_point['throttle'] = float(np.mean(throttles)) if throttles else None
                smoothed_point['brake_pressure'] = float(np.mean(brakes)) if brakes else None
                smoothed_point['gear'] = int(np.round(np.mean(gears))) if gears else None
            
            smoothed_line.append(smoothed_point)
        
        optimal_line = smoothed_line
    
    # Calculate statistics
    speeds = [p['speed'] for p in optimal_line]
    avg_speed = np.mean(speeds)
    max_speed = np.max(speeds)
    min_speed = np.min(speeds)
    
    stats = {
        'total_segments': len(optimal_line),
        'average_speed': float(avg_speed),
        'max_speed': float(max_speed),
        'min_speed': float(min_speed),
        'speed_variation': float(max_speed - min_speed)
    }
    
    # Add telemetry statistics if available
    if include_telemetry and optimal_line and 'throttle' in optimal_line[0]:
        throttles = [p.get('throttle', 0) for p in optimal_line if p.get('throttle') is not None]
        brakes = [p.get('brake_pressure', 0) for p in optimal_line if p.get('brake_pressure') is not None]
        
        if throttles:
            stats['average_throttle'] = float(np.mean(throttles))
        if brakes:
            stats['max_brake_pressure'] = float(np.max(brakes))
    
    return {
        'vehicle_id': vehicle_id,
        'chassis_number': vehicle_info['chassis'],
        'car_number': vehicle_info['car_number'],
        'laps_analyzed': lap_numbers,
        'optimal_line': optimal_line[:500],  # Limit for performance
        'statistics': stats,
        'insights': [
            f"Optimal racing line calculated from {len(lap_numbers)} lap(s)",
            f"Average speed along optimal line: {avg_speed:.1f} km/h",
            f"Speed range: {min_speed:.1f} - {max_speed:.1f} km/h"
        ]
    }


async def generate_training_plan(vehicle_id: str) -> Dict:
    """
    Generate personalized training plan based on driver performance analysis
    
    Combines:
    - Sector analysis
    - Consistency metrics
    - Improvement opportunities
    - Performance predictions
    """
    from driver_insights import get_driver_improvement_opportunities, get_best_worst_lap_analysis
    from ai_insights import get_ai_driver_insights
    
    # Get existing analyses
    improvements = await get_driver_improvement_opportunities(vehicle_id)
    best_worst = await get_best_worst_lap_analysis(vehicle_id)
    ai_insights = await get_ai_driver_insights(vehicle_id)
    
    # Generate training plan
    training_plan = {
        'vehicle_id': vehicle_id,
        'generated_at': pd.Timestamp.now().isoformat(),
        'focus_areas': [],
        'training_sessions': [],
        'recommendations': [],
        'goals': []
    }
    
    # Identify focus areas from improvements
    if improvements and 'improvement_opportunities' in improvements:
        for opp in improvements['improvement_opportunities'][:3]:  # Top 3
            improvement_potential = opp.get('improvement_potential_seconds', 0)
            training_plan['focus_areas'].append({
                'area': opp.get('sector', 'Unknown'),
                'current_average': opp.get('average', 0),
                'best_time': opp.get('best_time', 0),
                'improvement_potential': improvement_potential,
                'priority': 'high' if improvement_potential > 1.0 else 'medium'
            })
    
    # Add AI recommendations
    if ai_insights and 'recommendations' in ai_insights:
        for rec in ai_insights['recommendations']:
            if isinstance(rec, dict):
                training_plan['recommendations'].append({
                    'title': rec.get('title', 'Recommendation'),
                    'description': rec.get('description', ''),
                    'priority': rec.get('priority', 'medium'),
                    'action_items': rec.get('action_items', []),
                    'expected_improvement': rec.get('expected_improvement', 'N/A')
                })
            elif isinstance(rec, str):
                # Handle string recommendations
                training_plan['recommendations'].append({
                    'title': 'AI Recommendation',
                    'description': rec,
                    'priority': 'medium',
                    'action_items': [],
                    'expected_improvement': 'N/A'
                })
    
    # Generate training sessions
    session_num = 1
    for focus_area in training_plan['focus_areas'][:3]:
        improvement_potential = focus_area.get('improvement_potential', 0)
        best_time = focus_area.get('best_time', 0)
        area_name = focus_area.get('area', 'Unknown Sector')
        
        training_plan['training_sessions'].append({
            'session_number': session_num,
            'focus': area_name,
            'duration_minutes': 30,
            'objectives': [
                f"Improve consistency in {area_name}",
                f"Match best time of {best_time:.2f}s" if best_time > 0 else f"Improve performance in {area_name}",
                f"Reduce variation in {area_name}"
            ],
            'exercises': [
                f"Review telemetry from best {area_name} lap",
                f"Practice maintaining consistent racing line",
                f"Focus on braking points and corner entry speeds"
            ],
            'success_metrics': {
                'target_improvement': f"{improvement_potential * 0.5:.2f}s" if improvement_potential > 0 else "0.00s",
                'consistency_target': "Reduce variation by 20%"
            }
        })
        session_num += 1
    
    # Set goals
    if best_worst and 'total_difference_seconds' in best_worst:
        improvement_goal = best_worst['total_difference_seconds'] * 0.3  # 30% of gap
        training_plan['goals'].append({
            'goal': 'Lap Time Improvement',
            'current_best': best_worst['best_lap']['lap_time_seconds'],
            'target': best_worst['best_lap']['lap_time_seconds'] - improvement_goal,
            'improvement_needed': improvement_goal,
            'timeline_weeks': 4
        })
    
    if ai_insights and 'summary' in ai_insights:
        consistency_goal = ai_insights['summary']['consistency_score'] * 0.8  # 20% improvement
        training_plan['goals'].append({
            'goal': 'Consistency Improvement',
            'current_score': ai_insights['summary']['consistency_score'],
            'target_score': consistency_goal,
            'improvement_needed': ai_insights['summary']['consistency_score'] - consistency_goal,
            'timeline_weeks': 6
        })
    
    return training_plan


async def predict_sector_performance(vehicle_id: str, sector: str, future_laps: int = 5) -> Dict:
    """
    Predict future sector performance using time series analysis
    """
    project_root = get_project_root()
    endurance_file = project_root / "logs" / "R1_section_endurance.csv"
    
    if not endurance_file.exists():
        raise HTTPException(status_code=404, detail="Endurance data not found")
    
    endurance_df = pd.read_csv(endurance_file, sep=";", low_memory=False)
    endurance_df.columns = endurance_df.columns.str.strip()
    
    # Use helper function to find vehicle data
    vehicle_data = find_vehicle_in_endurance_data(endurance_df, vehicle_id)
    
    if len(vehicle_data) == 0:
        available_numbers = sorted(endurance_df['NUMBER'].dropna().unique().astype(str).tolist())
        car_number, _ = normalize_vehicle_id(vehicle_id)
        raise HTTPException(
            status_code=404, 
            detail=f"Vehicle {vehicle_id} (tried car_number: {car_number}) not found in endurance data. Available vehicles: {available_numbers[:20]}"
        )
    
    sector_col = f'{sector.upper()}_SECONDS'
    if sector_col not in vehicle_data.columns:
        raise HTTPException(status_code=400, detail=f"Invalid sector: {sector}")
    
    vehicle_data = vehicle_data.sort_values('LAP_NUMBER')
    sector_times = vehicle_data[sector_col].dropna()
    
    if len(sector_times) < 3:
        raise HTTPException(status_code=400, detail="Insufficient data for prediction")
    
    # Use linear regression for sector prediction
    lap_numbers = vehicle_data[vehicle_data[sector_col].notna()]['LAP_NUMBER'].values
    sector_values = sector_times.values
    
    X = lap_numbers.reshape(-1, 1)
    y = sector_values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict future
    future_lap_numbers = np.arange(lap_numbers.max() + 1, lap_numbers.max() + 1 + future_laps)
    predictions = model.predict(future_lap_numbers.reshape(-1, 1))
    
    residuals = y - model.predict(X)
    std_error = np.std(residuals)
    
    return {
        'vehicle_id': vehicle_id,
        'sector': sector,
        'current_laps': len(sector_times),
        'current_best': float(sector_times.min()),
        'current_average': float(sector_times.mean()),
        'prediction_horizon': future_laps,
        'predictions': {
            'lap_numbers': future_lap_numbers.tolist(),
            'predicted_times': predictions.tolist(),
            'confidence_interval': {
                'lower': (predictions - std_error).tolist(),
                'upper': (predictions + std_error).tolist()
            }
        },
        'trend': {
            'direction': 'improving' if model.coef_[0] < 0 else 'declining' if model.coef_[0] > 0 else 'stable',
            'rate_per_lap': float(model.coef_[0])
        }
    }

