"""
Driver Training & Insights Module
Analyzes driver performance, racing lines, and improvement opportunities
"""

from fastapi import HTTPException
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from collections import defaultdict
import json


def get_project_root():
    """Get the project root directory"""
    return Path(__file__).parent


def find_vehicle_file(vehicle_id: str, event_name: Optional[str] = None) -> Path:
    """
    Find vehicle CSV file in logs directory or event directory
    
    Args:
        vehicle_id: Vehicle ID (e.g., "GR86-022-13")
        event_name: Optional event name to search in event directory
    
    Returns:
        Path to vehicle CSV file
    
    Raises:
        HTTPException if file not found
    """
    project_root = get_project_root()
    
    # Try event directory first if specified
    if event_name:
        event_file = project_root / "logs" / "events" / event_name / "vehicles" / f"{vehicle_id}.csv"
        if event_file.exists():
            return event_file
    
    # Try default vehicles directory
    default_file = project_root / "logs" / "vehicles" / f"{vehicle_id}.csv"
    if default_file.exists():
        return default_file
    
    # Try searching all event directories
    events_dir = project_root / "logs" / "events"
    if events_dir.exists():
        for event_dir in events_dir.iterdir():
            if event_dir.is_dir():
                event_file = event_dir / "vehicles" / f"{vehicle_id}.csv"
                if event_file.exists():
                    return event_file
    
    raise HTTPException(status_code=404, detail=f"Vehicle {vehicle_id} telemetry not found")


def extract_chassis_number(vehicle_id: str) -> str:
    """
    Extract chassis number from vehicle ID format: GR86-{chassis}-{car_number}
    Example: GR86-002-000 -> 002
    """
    if '-' in vehicle_id:
        parts = vehicle_id.split('-')
        if len(parts) >= 2:
            return parts[1]  # Return chassis number
    return vehicle_id  # Return as-is if format doesn't match


def extract_car_number(vehicle_id: str) -> str:
    """
    Extract car number from vehicle ID format: GR86-{chassis}-{car_number}
    Example: GR86-010-16 -> 16
    """
    if '-' in vehicle_id:
        parts = vehicle_id.split('-')
        if len(parts) >= 3:
            car_num = parts[2]  # Return car number
            # Remove leading zeros (16, not 016)
            try:
                return str(int(car_num)) if car_num.isdigit() else car_num
            except ValueError:
                return car_num
    return vehicle_id  # Return as-is if format doesn't match


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


async def get_racing_line_comparison(vehicle_id: str, compare_lap1: int, compare_lap2: int, event_name: Optional[str] = None) -> Dict:
    """
    Compare racing lines between two laps
    Returns GPS coordinates and deviations
    
    Args:
        vehicle_id: Vehicle ID
        compare_lap1: First lap number to compare
        compare_lap2: Second lap number to compare
        event_name: Optional event name to load data from specific event
    """
    vehicle_file = find_vehicle_file(vehicle_id, event_name)
    
    df = pd.read_csv(vehicle_file, parse_dates=["meta_time"], low_memory=False)
    df["meta_time"] = pd.to_datetime(df["meta_time"], utc=True, errors="coerce")
    
    # Extract GPS coordinates for each lap
    lap1_data = df[df["telemetry_name"] == "lap"][df["telemetry_value"] == compare_lap1]
    lap2_data = df[df["telemetry_name"] == "lap"][df["telemetry_value"] == compare_lap2]
    
    if len(lap1_data) == 0 or len(lap2_data) == 0:
        raise HTTPException(status_code=404, detail="Lap data not found")
    
    # Get time ranges for each lap
    lap1_start = lap1_data.iloc[0]["meta_time"]
    lap2_start = lap2_data.iloc[0]["meta_time"]
    
    # Get next lap start or end of data
    lap1_end = df[df["meta_time"] > lap1_start]["meta_time"].min()
    lap2_end = df[df["meta_time"] > lap2_start]["meta_time"].min()
    
    if pd.isna(lap1_end):
        lap1_end = df["meta_time"].max()
    if pd.isna(lap2_end):
        lap2_end = df["meta_time"].max()
    
    # Extract GPS coordinates
    lap1_gps = df[
        (df["meta_time"] >= lap1_start) & 
        (df["meta_time"] < lap1_end) &
        (df["telemetry_name"].isin(["VBOX_Lat_Min", "VBOX_Long_Minutes"]))
    ].sort_values("meta_time")
    
    lap2_gps = df[
        (df["meta_time"] >= lap2_start) & 
        (df["meta_time"] < lap2_end) &
        (df["telemetry_name"].isin(["VBOX_Lat_Min", "VBOX_Long_Minutes"]))
    ].sort_values("meta_time")
    
    # Pair lat/lon coordinates
    lap1_coords = []
    lap2_coords = []
    
    # Process lap 1
    lat_rows = lap1_gps[lap1_gps["telemetry_name"] == "VBOX_Lat_Min"]
    lon_rows = lap1_gps[lap1_gps["telemetry_name"] == "VBOX_Long_Minutes"]
    
    for idx in lat_rows.index:
        lat = lat_rows.loc[idx, "telemetry_value"]
        # Find corresponding lon
        time = lat_rows.loc[idx, "meta_time"]
        lon_row = lon_rows[lon_rows["meta_time"] == time]
        if len(lon_row) > 0:
            lon = lon_row.iloc[0]["telemetry_value"]
            lap1_coords.append({"lat": float(lat), "lon": float(lon), "time": time.isoformat()})
    
    # Process lap 2
    lat_rows = lap2_gps[lap2_gps["telemetry_name"] == "VBOX_Lat_Min"]
    lon_rows = lap2_gps[lap2_gps["telemetry_name"] == "VBOX_Long_Minutes"]
    
    for idx in lat_rows.index:
        lat = lat_rows.loc[idx, "telemetry_value"]
        time = lat_rows.loc[idx, "meta_time"]
        lon_row = lon_rows[lon_rows["meta_time"] == time]
        if len(lon_row) > 0:
            lon = lon_row.iloc[0]["telemetry_value"]
            lap2_coords.append({"lat": float(lat), "lon": float(lon), "time": time.isoformat()})
    
    # Calculate deviations (simplified - compare closest points)
    deviations = []
    if len(lap1_coords) > 0 and len(lap2_coords) > 0:
        # Sample points for comparison (every 10th point)
        sample1 = lap1_coords[::max(1, len(lap1_coords)//50)]
        sample2 = lap2_coords[::max(1, len(lap2_coords)//50)]
        
        for p1 in sample1:
            min_dist = float('inf')
            closest_p2 = None
            for p2 in sample2:
                dist = calculate_distance(p1["lat"], p1["lon"], p2["lat"], p2["lon"])
                if dist < min_dist:
                    min_dist = dist
                    closest_p2 = p2
            
            if closest_p2:
                deviations.append({
                    "lat": p1["lat"],
                    "lon": p1["lon"],
                    "deviation_meters": float(min_dist),
                    "lap1_time": p1["time"],
                    "lap2_time": closest_p2["time"]
                })
    
    return {
        "vehicle_id": vehicle_id,
        "lap1": compare_lap1,
        "lap2": compare_lap2,
        "lap1_coordinates": lap1_coords[:1000],  # Limit for performance
        "lap2_coordinates": lap2_coords[:1000],
        "deviations": deviations[:100],  # Limit deviations
        "average_deviation": float(np.mean([d["deviation_meters"] for d in deviations])) if deviations else 0
    }


async def get_braking_analysis(vehicle_id: str, lap_number: Optional[int] = None, event_name: Optional[str] = None) -> Dict:
    """
    Analyze braking points and patterns
    Returns braking zones, brake pressure, and speed analysis
    """
    vehicle_file = find_vehicle_file(vehicle_id, event_name)
    
    df = pd.read_csv(vehicle_file, parse_dates=["meta_time"], low_memory=False)
    df["meta_time"] = pd.to_datetime(df["meta_time"], utc=True, errors="coerce")
    
    # Filter by lap if specified
    if lap_number:
        lap_rows = df[df["telemetry_name"] == "lap"][df["telemetry_value"] == lap_number]
        if len(lap_rows) > 0:
            lap_start = lap_rows.iloc[0]["meta_time"]
            lap_end = df[df["meta_time"] > lap_start]["meta_time"].min()
            if pd.isna(lap_end):
                lap_end = df["meta_time"].max()
            df = df[(df["meta_time"] >= lap_start) & (df["meta_time"] < lap_end)]
    
    # Extract braking data
    brake_data = df[df["telemetry_name"] == "pbrake_f"].copy()
    speed_data = df[df["telemetry_name"] == "speed"].copy()
    gps_data = df[df["telemetry_name"].isin(["VBOX_Lat_Min", "VBOX_Long_Minutes"])].copy()
    
    # Identify braking zones (brake pressure > threshold)
    BRAKE_THRESHOLD = 10.0  # bar
    braking_zones = []
    
    brake_data = brake_data.sort_values("meta_time")
    speed_data = speed_data.sort_values("meta_time")
    
    in_braking = False
    brake_start = None
    max_brake_pressure = 0
    brake_start_speed = None
    
    for idx, row in brake_data.iterrows():
        brake_pressure = row["telemetry_value"]
        time = row["meta_time"]
        
        # Get speed at this time
        speed_row = speed_data[speed_data["meta_time"] == time]
        current_speed = speed_row.iloc[0]["telemetry_value"] if len(speed_row) > 0 else None
        
        if brake_pressure > BRAKE_THRESHOLD:
            if not in_braking:
                in_braking = True
                brake_start = time
                brake_start_speed = current_speed
                max_brake_pressure = brake_pressure
            else:
                max_brake_pressure = max(max_brake_pressure, brake_pressure)
        else:
            if in_braking:
                # End of braking zone
                brake_end = time
                brake_end_speed = current_speed
                
                # Get GPS coordinates
                gps_at_start = gps_data[gps_data["meta_time"] == brake_start]
                lat = None
                lon = None
                if len(gps_at_start) > 0:
                    lat_row = gps_at_start[gps_at_start["telemetry_name"] == "VBOX_Lat_Min"]
                    lon_row = gps_at_start[gps_at_start["telemetry_name"] == "VBOX_Long_Minutes"]
                    if len(lat_row) > 0:
                        lat = float(lat_row.iloc[0]["telemetry_value"])
                    if len(lon_row) > 0:
                        lon = float(lon_row.iloc[0]["telemetry_value"])
                
                braking_zones.append({
                    "start_time": brake_start.isoformat(),
                    "end_time": brake_end.isoformat(),
                    "duration_seconds": float((brake_end - brake_start).total_seconds()),
                    "max_brake_pressure": float(max_brake_pressure),
                    "start_speed": float(brake_start_speed) if brake_start_speed else None,
                    "end_speed": float(brake_end_speed) if brake_end_speed else None,
                    "speed_reduction": float(brake_start_speed - brake_end_speed) if (brake_start_speed and brake_end_speed) else None,
                    "gps_lat": lat,
                    "gps_lon": lon
                })
                in_braking = False
    
    # Calculate statistics
    if braking_zones:
        avg_brake_pressure = np.mean([z["max_brake_pressure"] for z in braking_zones])
        avg_duration = np.mean([z["duration_seconds"] for z in braking_zones])
        total_braking_time = sum([z["duration_seconds"] for z in braking_zones])
    else:
        avg_brake_pressure = 0
        avg_duration = 0
        total_braking_time = 0
    
    return {
        "vehicle_id": vehicle_id,
        "lap_number": lap_number,
        "braking_zones": braking_zones[:50],  # Limit for performance
        "statistics": {
            "total_braking_zones": len(braking_zones),
            "average_brake_pressure": float(avg_brake_pressure),
            "average_duration_seconds": float(avg_duration),
            "total_braking_time_seconds": float(total_braking_time)
        }
    }


async def get_corner_analysis(vehicle_id: str, lap_number: Optional[int] = None, event_name: Optional[str] = None) -> Dict:
    """
    Analyze cornering performance
    Returns corner speeds, lateral G-forces, and steering angles
    """
    vehicle_file = find_vehicle_file(vehicle_id, event_name)
    
    df = pd.read_csv(vehicle_file, parse_dates=["meta_time"], low_memory=False)
    df["meta_time"] = pd.to_datetime(df["meta_time"], utc=True, errors="coerce")
    
    # Filter by lap if specified
    if lap_number:
        lap_rows = df[df["telemetry_name"] == "lap"][df["telemetry_value"] == lap_number]
        if len(lap_rows) > 0:
            lap_start = lap_rows.iloc[0]["meta_time"]
            lap_end = df[df["meta_time"] > lap_start]["meta_time"].min()
            if pd.isna(lap_end):
                lap_end = df["meta_time"].max()
            df = df[(df["meta_time"] >= lap_start) & (df["meta_time"] < lap_end)]
    
    # Extract cornering data
    lateral_g = df[df["telemetry_name"] == "accy_can"].copy()
    steering = df[df["telemetry_name"] == "Steering_Angle"].copy()
    speed = df[df["telemetry_name"] == "speed"].copy()
    gps = df[df["telemetry_name"].isin(["VBOX_Lat_Min", "VBOX_Long_Minutes"])].copy()
    
    # Identify corners (lateral G > threshold)
    CORNER_THRESHOLD = 0.3  # G's
    corners = []
    
    lateral_g = lateral_g.sort_values("meta_time")
    steering = steering.sort_values("meta_time")
    speed = speed.sort_values("meta_time")
    
    in_corner = False
    corner_start = None
    max_lateral_g = 0
    min_speed = None
    max_steering = 0
    
    for idx, row in lateral_g.iterrows():
        lateral_accel = abs(row["telemetry_value"])
        time = row["meta_time"]
        
        # Get steering and speed at this time
        steering_row = steering[steering["meta_time"] == time]
        speed_row = speed[speed["meta_time"] == time]
        current_steering = abs(steering_row.iloc[0]["telemetry_value"]) if len(steering_row) > 0 else 0
        current_speed = speed_row.iloc[0]["telemetry_value"] if len(speed_row) > 0 else None
        
        if lateral_accel > CORNER_THRESHOLD:
            if not in_corner:
                in_corner = True
                corner_start = time
                max_lateral_g = lateral_accel
                max_steering = current_steering
                min_speed = current_speed
            else:
                max_lateral_g = max(max_lateral_g, lateral_accel)
                max_steering = max(max_steering, current_steering)
                if current_speed and min_speed:
                    min_speed = min(min_speed, current_speed)
        else:
            if in_corner:
                # End of corner
                corner_end = time
                
                # Get GPS coordinates
                gps_at_start = gps[gps["meta_time"] == corner_start]
                lat = None
                lon = None
                if len(gps_at_start) > 0:
                    lat_row = gps_at_start[gps_at_start["telemetry_name"] == "VBOX_Lat_Min"]
                    lon_row = gps_at_start[gps_at_start["telemetry_name"] == "VBOX_Long_Minutes"]
                    if len(lat_row) > 0:
                        lat = float(lat_row.iloc[0]["telemetry_value"])
                    if len(lon_row) > 0:
                        lon = float(lon_row.iloc[0]["telemetry_value"])
                
                corners.append({
                    "start_time": corner_start.isoformat(),
                    "end_time": corner_end.isoformat(),
                    "duration_seconds": float((corner_end - corner_start).total_seconds()),
                    "max_lateral_g": float(max_lateral_g),
                    "max_steering_angle": float(max_steering),
                    "min_speed": float(min_speed) if min_speed else None,
                    "gps_lat": lat,
                    "gps_lon": lon
                })
                in_corner = False
    
    # Calculate statistics
    if corners:
        avg_lateral_g = np.mean([c["max_lateral_g"] for c in corners])
        avg_min_speed = np.mean([c["min_speed"] for c in corners if c["min_speed"]])
        avg_duration = np.mean([c["duration_seconds"] for c in corners])
    else:
        avg_lateral_g = 0
        avg_min_speed = 0
        avg_duration = 0
    
    return {
        "vehicle_id": vehicle_id,
        "lap_number": lap_number,
        "corners": corners[:50],  # Limit for performance
        "statistics": {
            "total_corners": len(corners),
            "average_lateral_g": float(avg_lateral_g),
            "average_min_speed": float(avg_min_speed),
            "average_duration_seconds": float(avg_duration)
        }
    }


async def get_driver_improvement_opportunities(vehicle_id: str, event_name: Optional[str] = None) -> Dict:
    """
    Identify specific improvement opportunities for a driver
    Compares best vs worst sectors and identifies areas for improvement
    """
    project_root = get_project_root()
    endurance_file = project_root / "logs" / "R1_section_endurance.csv"
    
    if not endurance_file.exists():
        raise HTTPException(status_code=404, detail="Endurance data not found")
    
    endurance_df = pd.read_csv(endurance_file, sep=";", low_memory=False)
    endurance_df.columns = endurance_df.columns.str.strip()
    
    # Extract both chassis and car number from vehicle ID
    # Format: GR86-{chassis}-{car_number}
    # Example: GR86-010-16 -> chassis: 010, car: 16
    # The NUMBER column in endurance data contains car numbers (the last part of vehicle ID)
    chassis_number = extract_chassis_number(vehicle_id)
    car_number = extract_car_number(vehicle_id)
    
    # Filter for specific vehicle - try multiple matching strategies
    # Priority: car_number (most likely match) > chassis_number > full vehicle_id
    vehicle_data = endurance_df[
        (endurance_df['NUMBER'].astype(str) == str(car_number)) |
        (endurance_df['NUMBER'].astype(str) == str(chassis_number)) |
        (endurance_df['NUMBER'].astype(str) == str(vehicle_id))
    ]
    
    if len(vehicle_data) == 0:
        # Try with leading zeros removed
        try:
            chassis_no_leading_zeros = str(int(chassis_number)) if chassis_number.isdigit() else chassis_number
            # Also try car number with original format (in case it has leading zeros)
            car_with_zeros = vehicle_id.split('-')[2] if '-' in vehicle_id and len(vehicle_id.split('-')) >= 3 else car_number
            
            vehicle_data = endurance_df[
                (endurance_df['NUMBER'].astype(str) == chassis_no_leading_zeros) |
                (endurance_df['NUMBER'].astype(str) == str(car_with_zeros))
            ]
        except (ValueError, TypeError):
            pass
        
        if len(vehicle_data) == 0:
            # Debug: show available vehicle numbers (convert to clean Python list)
            available_numbers = [str(n) for n in sorted(endurance_df['NUMBER'].dropna().unique())]
            raise HTTPException(
                status_code=404, 
                detail=f"Vehicle {vehicle_id} (chassis: {chassis_number}, car: {car_number}) not found in endurance data. Available car numbers: {available_numbers[:15]}. This vehicle may not have completed any laps in the race."
            )
    
    vehicle_data = vehicle_data.sort_values('LAP_NUMBER')
    
    # Calculate sector statistics
    s1_times = vehicle_data['S1_SECONDS'].dropna()
    s2_times = vehicle_data['S2_SECONDS'].dropna()
    s3_times = vehicle_data['S3_SECONDS'].dropna()
    
    if len(s1_times) == 0 or len(s2_times) == 0 or len(s3_times) == 0:
        return {"error": "Insufficient sector data"}
    
    # Find best and worst sectors
    best_s1_lap = vehicle_data.loc[s1_times.idxmin()]
    worst_s1_lap = vehicle_data.loc[s1_times.idxmax()]
    
    best_s2_lap = vehicle_data.loc[s2_times.idxmin()]
    worst_s2_lap = vehicle_data.loc[s2_times.idxmax()]
    
    best_s3_lap = vehicle_data.loc[s3_times.idxmin()]
    worst_s3_lap = vehicle_data.loc[s3_times.idxmax()]
    
    # Calculate improvement potential
    improvements = []
    
    s1_improvement = float(worst_s1_lap['S1_SECONDS'] - best_s1_lap['S1_SECONDS'])
    s2_improvement = float(worst_s2_lap['S2_SECONDS'] - best_s2_lap['S2_SECONDS'])
    s3_improvement = float(worst_s3_lap['S3_SECONDS'] - best_s3_lap['S3_SECONDS'])
    
    improvements.append({
        "sector": "Sector 1",
        "improvement_potential_seconds": s1_improvement,
        "best_time": float(best_s1_lap['S1_SECONDS']),
        "worst_time": float(worst_s1_lap['S1_SECONDS']),
        "best_lap": int(best_s1_lap['LAP_NUMBER']),
        "worst_lap": int(worst_s1_lap['LAP_NUMBER']),
        "consistency": float(s1_times.std()),
        "average": float(s1_times.mean())
    })
    
    improvements.append({
        "sector": "Sector 2",
        "improvement_potential_seconds": s2_improvement,
        "best_time": float(best_s2_lap['S2_SECONDS']),
        "worst_time": float(worst_s2_lap['S2_SECONDS']),
        "best_lap": int(best_s2_lap['LAP_NUMBER']),
        "worst_lap": int(worst_s2_lap['LAP_NUMBER']),
        "consistency": float(s2_times.std()),
        "average": float(s2_times.mean())
    })
    
    improvements.append({
        "sector": "Sector 3",
        "improvement_potential_seconds": s3_improvement,
        "best_time": float(best_s3_lap['S3_SECONDS']),
        "worst_time": float(worst_s3_lap['S3_SECONDS']),
        "best_lap": int(best_s3_lap['LAP_NUMBER']),
        "worst_lap": int(worst_s3_lap['LAP_NUMBER']),
        "consistency": float(s3_times.std()),
        "average": float(s3_times.mean())
    })
    
    # Sort by improvement potential
    improvements.sort(key=lambda x: x["improvement_potential_seconds"], reverse=True)
    
    return {
        "vehicle_id": vehicle_id,
        "improvement_opportunities": improvements,
        "recommendations": [
            f"Focus on {imp['sector']} - potential improvement of {imp['improvement_potential_seconds']:.2f}s"
            for imp in improvements[:2]  # Top 2 recommendations
        ]
    }


async def get_speed_trace_comparison(vehicle_id: str, lap1: int, lap2: int, event_name: Optional[str] = None) -> Dict:
    """
    Compare speed traces between two laps
    Returns speed data points along the track for visualization
    """
    vehicle_file = find_vehicle_file(vehicle_id, event_name)
    
    df = pd.read_csv(vehicle_file, parse_dates=["meta_time"], low_memory=False)
    df["meta_time"] = pd.to_datetime(df["meta_time"], utc=True, errors="coerce")
    
    def get_lap_data(lap_num):
        lap_rows = df[df["telemetry_name"] == "lap"][df["telemetry_value"] == lap_num]
        if len(lap_rows) == 0:
            return None
        
        lap_start = lap_rows.iloc[0]["meta_time"]
        lap_end = df[df["meta_time"] > lap_start]["meta_time"].min()
        if pd.isna(lap_end):
            lap_end = df["meta_time"].max()
        
        lap_df = df[(df["meta_time"] >= lap_start) & (df["meta_time"] < lap_end)]
        
        # Get speed and GPS data
        speed_data = lap_df[lap_df["telemetry_name"] == "speed"].sort_values("meta_time")
        lat_data = lap_df[lap_df["telemetry_name"] == "VBOX_Lat_Min"].sort_values("meta_time")
        lon_data = lap_df[lap_df["telemetry_name"] == "VBOX_Long_Minutes"].sort_values("meta_time")
        
        # Combine speed with GPS coordinates
        trace_points = []
        for idx, speed_row in speed_data.iterrows():
            time = speed_row["meta_time"]
            speed = speed_row["telemetry_value"]
            
            # Find closest GPS coordinates
            lat_row = lat_data.iloc[(lat_data["meta_time"] - time).abs().argsort()[:1]]
            lon_row = lon_data.iloc[(lon_data["meta_time"] - time).abs().argsort()[:1]]
            
            if len(lat_row) > 0 and len(lon_row) > 0:
                trace_points.append({
                    "time": time.isoformat(),
                    "speed": float(speed),
                    "lat": float(lat_row.iloc[0]["telemetry_value"]),
                    "lon": float(lon_row.iloc[0]["telemetry_value"])
                })
        
        return trace_points
    
    lap1_data = get_lap_data(lap1)
    lap2_data = get_lap_data(lap2)
    
    if lap1_data is None or lap2_data is None:
        raise HTTPException(status_code=404, detail="Lap data not found")
    
    # Calculate statistics
    lap1_speeds = [p["speed"] for p in lap1_data]
    lap2_speeds = [p["speed"] for p in lap2_data]
    
    return {
        "vehicle_id": vehicle_id,
        "lap1": lap1,
        "lap2": lap2,
        "lap1_trace": lap1_data[::max(1, len(lap1_data)//200)],  # Sample for performance
        "lap2_trace": lap2_data[::max(1, len(lap2_data)//200)],
        "lap1_stats": {
            "max_speed": float(max(lap1_speeds)) if lap1_speeds else None,
            "min_speed": float(min(lap1_speeds)) if lap1_speeds else None,
            "avg_speed": float(np.mean(lap1_speeds)) if lap1_speeds else None
        },
        "lap2_stats": {
            "max_speed": float(max(lap2_speeds)) if lap2_speeds else None,
            "min_speed": float(min(lap2_speeds)) if lap2_speeds else None,
            "avg_speed": float(np.mean(lap2_speeds)) if lap2_speeds else None
        }
    }


async def get_best_worst_lap_analysis(vehicle_id: str, event_name: Optional[str] = None) -> Dict:
    """
    Compare best and worst laps to identify key differences
    """
    project_root = get_project_root()
    endurance_file = project_root / "logs" / "R1_section_endurance.csv"
    vehicle_file = find_vehicle_file(vehicle_id, event_name)
    
    if not endurance_file.exists():
        raise HTTPException(status_code=404, detail="Endurance data not found")
    
    endurance_df = pd.read_csv(endurance_file, sep=";", low_memory=False)
    endurance_df.columns = endurance_df.columns.str.strip()
    
    # Extract both chassis and car number from vehicle ID
    # Format: GR86-{chassis}-{car_number}
    # Example: GR86-010-16 -> chassis: 010, car: 16
    # The NUMBER column in endurance data contains car numbers (the last part of vehicle ID)
    chassis_number = extract_chassis_number(vehicle_id)
    car_number = extract_car_number(vehicle_id)
    
    # Filter for specific vehicle - try multiple matching strategies
    # Priority: car_number (most likely match) > chassis_number > full vehicle_id
    vehicle_data = endurance_df[
        (endurance_df['NUMBER'].astype(str) == str(car_number)) |
        (endurance_df['NUMBER'].astype(str) == str(chassis_number)) |
        (endurance_df['NUMBER'].astype(str) == str(vehicle_id))
    ]
    
    if len(vehicle_data) == 0:
        # Try with leading zeros removed
        try:
            chassis_no_leading_zeros = str(int(chassis_number)) if chassis_number.isdigit() else chassis_number
            # Also try car number with original format (in case it has leading zeros)
            car_with_zeros = vehicle_id.split('-')[2] if '-' in vehicle_id and len(vehicle_id.split('-')) >= 3 else car_number
            
            vehicle_data = endurance_df[
                (endurance_df['NUMBER'].astype(str) == chassis_no_leading_zeros) |
                (endurance_df['NUMBER'].astype(str) == str(car_with_zeros))
            ]
        except (ValueError, TypeError):
            pass
        
        if len(vehicle_data) == 0:
            # Debug: show available vehicle numbers (convert to clean Python list)
            available_numbers = [str(n) for n in sorted(endurance_df['NUMBER'].dropna().unique())]
            raise HTTPException(
                status_code=404, 
                detail=f"Vehicle {vehicle_id} (chassis: {chassis_number}, car: {car_number}) not found in endurance data. Available car numbers: {available_numbers[:15]}. This vehicle may not have completed any laps in the race."
            )
    
    # Convert lap times to seconds
    try:
        vehicle_data['lap_time_seconds'] = vehicle_data['LAP_TIME'].apply(parse_lap_time)
        valid_laps = vehicle_data[vehicle_data['lap_time_seconds'].notna()]
        
        if len(valid_laps) == 0:
            return {
                "error": "No valid lap times found",
                "vehicle_id": vehicle_id,
                "total_rows": len(vehicle_data),
                "sample_lap_times": vehicle_data['LAP_TIME'].head(5).tolist() if len(vehicle_data) > 0 else []
            }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing lap times: {str(e)}")
    
    # Find best and worst laps
    try:
        best_lap_idx = valid_laps['lap_time_seconds'].idxmin()
        worst_lap_idx = valid_laps['lap_time_seconds'].idxmax()
        
        best_lap = valid_laps.loc[best_lap_idx]
        worst_lap = valid_laps.loc[worst_lap_idx]
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error finding best/worst laps: {str(e)}")
    
    # Calculate differences with error handling
    try:
        sector_differences = {
            "sector_1": {
                "best": float(best_lap['S1_SECONDS']) if pd.notna(best_lap.get('S1_SECONDS')) else None,
                "worst": float(worst_lap['S1_SECONDS']) if pd.notna(worst_lap.get('S1_SECONDS')) else None,
                "difference": float(worst_lap['S1_SECONDS'] - best_lap['S1_SECONDS']) if pd.notna(best_lap.get('S1_SECONDS')) and pd.notna(worst_lap.get('S1_SECONDS')) else None
            },
            "sector_2": {
                "best": float(best_lap['S2_SECONDS']) if pd.notna(best_lap.get('S2_SECONDS')) else None,
                "worst": float(worst_lap['S2_SECONDS']) if pd.notna(worst_lap.get('S2_SECONDS')) else None,
                "difference": float(worst_lap['S2_SECONDS'] - best_lap['S2_SECONDS']) if pd.notna(best_lap.get('S2_SECONDS')) and pd.notna(worst_lap.get('S2_SECONDS')) else None
            },
            "sector_3": {
                "best": float(best_lap['S3_SECONDS']) if pd.notna(best_lap.get('S3_SECONDS')) else None,
                "worst": float(worst_lap['S3_SECONDS']) if pd.notna(worst_lap.get('S3_SECONDS')) else None,
                "difference": float(worst_lap['S3_SECONDS'] - best_lap['S3_SECONDS']) if pd.notna(best_lap.get('S3_SECONDS')) and pd.notna(worst_lap.get('S3_SECONDS')) else None
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error calculating sector differences: {str(e)}")
    
    try:
        return {
            "vehicle_id": vehicle_id,
            "best_lap": {
                "lap_number": int(best_lap.get('LAP_NUMBER', 0)),
                "lap_time": str(best_lap.get('LAP_TIME', '')) if pd.notna(best_lap.get('LAP_TIME')) else None,
                "lap_time_seconds": float(best_lap.get('lap_time_seconds', 0)),
                "sector_1": float(best_lap.get('S1_SECONDS')) if pd.notna(best_lap.get('S1_SECONDS')) else None,
                "sector_2": float(best_lap.get('S2_SECONDS')) if pd.notna(best_lap.get('S2_SECONDS')) else None,
                "sector_3": float(best_lap.get('S3_SECONDS')) if pd.notna(best_lap.get('S3_SECONDS')) else None,
                "top_speed": float(best_lap.get('TOP_SPEED')) if pd.notna(best_lap.get('TOP_SPEED')) else None
            },
            "worst_lap": {
                "lap_number": int(worst_lap.get('LAP_NUMBER', 0)),
                "lap_time": str(worst_lap.get('LAP_TIME', '')) if pd.notna(worst_lap.get('LAP_TIME')) else None,
                "lap_time_seconds": float(worst_lap.get('lap_time_seconds', 0)),
                "sector_1": float(worst_lap.get('S1_SECONDS')) if pd.notna(worst_lap.get('S1_SECONDS')) else None,
                "sector_2": float(worst_lap.get('S2_SECONDS')) if pd.notna(worst_lap.get('S2_SECONDS')) else None,
                "sector_3": float(worst_lap.get('S3_SECONDS')) if pd.notna(worst_lap.get('S3_SECONDS')) else None,
                "top_speed": float(worst_lap.get('TOP_SPEED')) if pd.notna(worst_lap.get('TOP_SPEED')) else None
            },
            "sector_differences": sector_differences,
            "total_difference_seconds": float(worst_lap.get('lap_time_seconds', 0) - best_lap.get('lap_time_seconds', 0)),
            "insights": [
                f"Best lap was {best_lap.get('lap_time_seconds', 0):.2f}s faster than worst lap",
                f"Largest sector difference: {max([s['difference'] for s in sector_differences.values() if s['difference'] is not None], default=0):.2f}s",
                f"Focus on consistency - {len(valid_laps)} laps completed"
            ]
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error building response: {str(e)}")

