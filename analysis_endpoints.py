"""
Post-Event Analysis Endpoints
Fast implementation for race story and driver insights
Add these endpoints to main.py for quick race analysis
"""

from fastapi import HTTPException
import pandas as pd
from typing import Dict, List, Optional
from collections import defaultdict
import numpy as np
from pathlib import Path


def get_project_root():
    """Get the project root directory"""
    return Path(__file__).parent


def safe_float(value):
    """Safely convert to float, handling NaN and None"""
    if pd.isna(value) or value is None:
        return None
    try:
        result = float(value)
        if pd.isna(result) or np.isnan(result):
            return None
        return result
    except:
        return None

def safe_int(value):
    """Safely convert to int, handling NaN and None"""
    if pd.isna(value) or value is None:
        return None
    try:
        result = int(value)
        return result
    except:
        return None

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


def parse_time_gap(gap_str: str) -> float:
    """Convert gap string (e.g., '+2.740' or '+1'14.985') to seconds"""
    if pd.isna(gap_str) or not gap_str or gap_str == '':
        return None
    try:
        gap_str = str(gap_str).strip()
        if gap_str.startswith('+'):
            gap_str = gap_str[1:]
        
        # Handle format like "1'14.985"
        if "'" in gap_str:
            parts = gap_str.split("'")
            minutes = int(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
        
        return float(gap_str)
    except:
        return None


async def get_race_story() -> Dict:
    """
    Generate race story with position changes and key moments
    Returns comprehensive race analysis
    """
    project_root = get_project_root()
    endurance_file = project_root / "logs" / "R1_section_endurance.csv"
    leaderboard_file = project_root / "logs" / "R1_leaderboard.csv"
    
    if not endurance_file.exists():
        raise HTTPException(status_code=404, detail="Endurance data not found")
    
    # Load data
    endurance_df = pd.read_csv(endurance_file, sep=";", low_memory=False)
    endurance_df.columns = endurance_df.columns.str.strip()
    
    # Ensure all numeric columns handle NaN properly
    for col in endurance_df.columns:
        if endurance_df[col].dtype == 'float64':
            endurance_df[col] = endurance_df[col].replace([np.inf, -np.inf], np.nan)
    
    leaderboard_df = None
    if leaderboard_file.exists():
        leaderboard_df = pd.read_csv(leaderboard_file, sep=";", low_memory=False)
        leaderboard_df.columns = leaderboard_df.columns.str.strip()
    
    # Convert lap times to seconds
    endurance_df['lap_time_seconds'] = endurance_df['LAP_TIME'].apply(parse_lap_time)
    
    # Get position changes over time
    position_changes = []
    for vehicle_id in endurance_df['NUMBER'].unique():
        vehicle_laps = endurance_df[endurance_df['NUMBER'] == vehicle_id].sort_values('LAP_NUMBER')
        
        # Estimate position per lap (simplified - use lap times relative to others)
        positions = []
        for lap_num in vehicle_laps['LAP_NUMBER'].unique():
            lap_data = vehicle_laps[vehicle_laps['LAP_NUMBER'] == lap_num]
            if len(lap_data) > 0:
                lap_time = lap_data.iloc[0]['lap_time_seconds']
                if lap_time:
                    # Compare with all other drivers on same lap
                    same_lap = endurance_df[
                        (endurance_df['LAP_NUMBER'] == lap_num) & 
                        (endurance_df['lap_time_seconds'].notna())
                    ]
                    if len(same_lap) > 0:
                        position = (same_lap['lap_time_seconds'] <= lap_time).sum()
                        positions.append({
                            "lap": safe_int(lap_num),
                            "position": safe_int(position),
                            "lap_time": str(lap_data.iloc[0]['LAP_TIME']) if pd.notna(lap_data.iloc[0]['LAP_TIME']) else None
                        })
        
        if positions:
            position_changes.append({
                "vehicle_id": str(vehicle_id),
                "position_history": positions
            })
    
    # Identify key moments
    key_moments = []
    
    # Fastest laps
    fastest_laps = endurance_df[
        endurance_df['lap_time_seconds'].notna()
    ].nsmallest(10, 'lap_time_seconds')
    
    for _, row in fastest_laps.iterrows():
        key_moments.append({
            "type": "fastest_lap",
            "lap": safe_int(row['LAP_NUMBER']),
            "vehicle_id": str(row['NUMBER']),
            "lap_time": str(row['LAP_TIME']) if pd.notna(row['LAP_TIME']) else None,
            "lap_time_seconds": safe_float(row['lap_time_seconds'])
        })
    
    # Pit stops - only include if PIT_TIME is valid
    pit_stops = endurance_df[endurance_df['CROSSING_FINISH_LINE_IN_PIT'].notna()]
    for _, row in pit_stops.iterrows():
        pit_time_val = row.get('PIT_TIME', None)
        # Only add if pit_time is actually a valid number
        if pit_time_val is not None and pd.notna(pit_time_val):
            try:
                pit_time_float = float(pit_time_val)
                # Double check for NaN/Inf
                if not (np.isnan(pit_time_float) or np.isinf(pit_time_float)):
                    # Ensure it's a Python float, not numpy float
                    pit_time_clean = float(pit_time_float)
                    key_moments.append({
                        "type": "pit_stop",
                        "lap": safe_int(row['LAP_NUMBER']),
                        "vehicle_id": str(row['NUMBER']),
                        "pit_time": pit_time_clean
                    })
            except (ValueError, TypeError):
                pass  # Skip invalid pit_time values
        # Also add pit stops without pit_time (just mark as pit stop)
        elif pd.notna(row.get('CROSSING_FINISH_LINE_IN_PIT', None)):
            key_moments.append({
                "type": "pit_stop",
                "lap": safe_int(row['LAP_NUMBER']),
                "vehicle_id": str(row['NUMBER']),
                "pit_time": None  # Explicitly None instead of NaN
            })
    
    # Sector improvements
    sector_improvements = []
    for vehicle_id in endurance_df['NUMBER'].unique():
        vehicle_laps = endurance_df[endurance_df['NUMBER'] == vehicle_id].sort_values('LAP_NUMBER')
        
        best_s1 = vehicle_laps['S1_SECONDS'].min()
        best_s2 = vehicle_laps['S2_SECONDS'].min()
        best_s3 = vehicle_laps['S3_SECONDS'].min()
        
        if pd.notna(best_s1) and pd.notna(best_s2) and pd.notna(best_s3):
            sector_improvements.append({
                "vehicle_id": str(vehicle_id),
                "best_sector_1": safe_float(best_s1),
                "best_sector_2": safe_float(best_s2),
                "best_sector_3": safe_float(best_s3),
                "best_lap_time": safe_float(best_s1 + best_s2 + best_s3) if all(pd.notna([best_s1, best_s2, best_s3])) else None
            })
    
    # Race statistics
    statistics = {}
    if leaderboard_df is not None:
        statistics = {
            "total_drivers": len(leaderboard_df),
            "total_laps": int(leaderboard_df['LAPS'].max()) if 'LAPS' in leaderboard_df.columns else None,
            "fastest_lap": leaderboard_df['BEST_LAP_TIME'].iloc[0] if len(leaderboard_df) > 0 else None,
            "fastest_driver": str(leaderboard_df['NUMBER'].iloc[0]) if len(leaderboard_df) > 0 else None,
        }
    
    return {
        "position_changes": position_changes,
        "key_moments": sorted(key_moments, key=lambda x: x.get('lap', 0)),
        "sector_improvements": sector_improvements,
        "statistics": statistics
    }


async def get_sector_comparison() -> Dict:
    """
    Compare sector times across all drivers
    Returns sector analysis and comparisons
    """
    project_root = get_project_root()
    endurance_file = project_root / "logs" / "R1_section_endurance.csv"
    
    if not endurance_file.exists():
        raise HTTPException(status_code=404, detail="Endurance data not found")
    
    endurance_df = pd.read_csv(endurance_file, sep=";", low_memory=False)
    endurance_df.columns = endurance_df.columns.str.strip()
    
    # Sector analysis
    sector_1_data = []
    sector_2_data = []
    sector_3_data = []
    
    for vehicle_id in endurance_df['NUMBER'].unique():
        vehicle_laps = endurance_df[endurance_df['NUMBER'] == vehicle_id]
        
        s1_times = vehicle_laps['S1_SECONDS'].dropna()
        s2_times = vehicle_laps['S2_SECONDS'].dropna()
        s3_times = vehicle_laps['S3_SECONDS'].dropna()
        
        if len(s1_times) > 0:
            sector_1_data.append({
                "vehicle_id": str(vehicle_id),
                "best": safe_float(s1_times.min()),
                "average": safe_float(s1_times.mean()),
                "worst": safe_float(s1_times.max()),
                "consistency": safe_float(s1_times.std()) if len(s1_times) > 1 else 0
            })
        
        if len(s2_times) > 0:
            sector_2_data.append({
                "vehicle_id": str(vehicle_id),
                "best": safe_float(s2_times.min()),
                "average": safe_float(s2_times.mean()),
                "worst": safe_float(s2_times.max()),
                "consistency": safe_float(s2_times.std()) if len(s2_times) > 1 else 0
            })
        
        if len(s3_times) > 0:
            sector_3_data.append({
                "vehicle_id": str(vehicle_id),
                "best": safe_float(s3_times.min()),
                "average": safe_float(s3_times.mean()),
                "worst": safe_float(s3_times.max()),
                "consistency": safe_float(s3_times.std()) if len(s3_times) > 1 else 0
            })
    
    # Find best sector times overall
    all_s1 = endurance_df['S1_SECONDS'].dropna()
    all_s2 = endurance_df['S2_SECONDS'].dropna()
    all_s3 = endurance_df['S3_SECONDS'].dropna()
    
    best_s1_row = endurance_df.loc[all_s1.idxmin()] if len(all_s1) > 0 else None
    best_s2_row = endurance_df.loc[all_s2.idxmin()] if len(all_s2) > 0 else None
    best_s3_row = endurance_df.loc[all_s3.idxmin()] if len(all_s3) > 0 else None
    
    return {
        "sector_1": {
            "best_overall": {
                "vehicle_id": str(best_s1_row['NUMBER']) if best_s1_row is not None else None,
                "time": safe_float(best_s1_row['S1_SECONDS']) if best_s1_row is not None else None,
                "lap": safe_int(best_s1_row['LAP_NUMBER']) if best_s1_row is not None else None
            },
            "drivers": sorted(sector_1_data, key=lambda x: x['best'])
        },
        "sector_2": {
            "best_overall": {
                "vehicle_id": str(best_s2_row['NUMBER']) if best_s2_row is not None else None,
                "time": float(best_s2_row['S2_SECONDS']) if best_s2_row is not None else None,
                "lap": int(best_s2_row['LAP_NUMBER']) if best_s2_row is not None else None
            },
            "drivers": sorted(sector_2_data, key=lambda x: x['best'])
        },
        "sector_3": {
            "best_overall": {
                "vehicle_id": str(best_s3_row['NUMBER']) if best_s3_row is not None else None,
                "time": float(best_s3_row['S3_SECONDS']) if best_s3_row is not None else None,
                "lap": int(best_s3_row['LAP_NUMBER']) if best_s3_row is not None else None
            },
            "drivers": sorted(sector_3_data, key=lambda x: x['best'])
        }
    }


async def get_driver_insights(vehicle_id: str) -> Dict:
    """
    Get detailed insights for a specific driver
    Returns sector analysis, lap time trends, and improvement opportunities
    """
    project_root = get_project_root()
    endurance_file = project_root / "logs" / "R1_section_endurance.csv"
    
    if not endurance_file.exists():
        raise HTTPException(status_code=404, detail="Endurance data not found")
    
    endurance_df = pd.read_csv(endurance_file, sep=";", low_memory=False)
    endurance_df.columns = endurance_df.columns.str.strip()
    
    # Filter for specific vehicle
    vehicle_data = endurance_df[endurance_df['NUMBER'].astype(str) == str(vehicle_id)]
    
    if len(vehicle_data) == 0:
        raise HTTPException(status_code=404, detail=f"Vehicle {vehicle_id} not found")
    
    vehicle_data = vehicle_data.sort_values('LAP_NUMBER')
    
    # Convert lap times
    vehicle_data['lap_time_seconds'] = vehicle_data['LAP_TIME'].apply(parse_lap_time)
    
    # Best and worst laps
    valid_laps = vehicle_data[vehicle_data['lap_time_seconds'].notna()]
    
    if len(valid_laps) == 0:
        return {"error": "No valid lap times found"}
    
    best_lap_idx = valid_laps['lap_time_seconds'].idxmin()
    worst_lap_idx = valid_laps['lap_time_seconds'].idxmax()
    
    best_lap = valid_laps.loc[best_lap_idx]
    worst_lap = valid_laps.loc[worst_lap_idx]
    
    # Sector analysis
    s1_times = vehicle_data['S1_SECONDS'].dropna()
    s2_times = vehicle_data['S2_SECONDS'].dropna()
    s3_times = vehicle_data['S3_SECONDS'].dropna()
    
    # Improvement opportunities
    improvements = []
    
    # Compare best vs worst sectors
    if pd.notna(best_lap['S1_SECONDS']) and pd.notna(worst_lap['S1_SECONDS']):
        s1_diff = worst_lap['S1_SECONDS'] - best_lap['S1_SECONDS']
        if s1_diff > 0.5:  # More than 0.5 seconds difference
            improvements.append({
                "sector": "Sector 1",
                "improvement_potential": safe_float(s1_diff),
                "best_time": safe_float(best_lap['S1_SECONDS']),
                "worst_time": safe_float(worst_lap['S1_SECONDS']),
                "best_lap": safe_int(best_lap['LAP_NUMBER']),
                "worst_lap": safe_int(worst_lap['LAP_NUMBER'])
            })
    
    if pd.notna(best_lap['S2_SECONDS']) and pd.notna(worst_lap['S2_SECONDS']):
        s2_diff = worst_lap['S2_SECONDS'] - best_lap['S2_SECONDS']
        if s2_diff > 0.5:
            improvements.append({
                "sector": "Sector 2",
                "improvement_potential": safe_float(s2_diff),
                "best_time": safe_float(best_lap['S2_SECONDS']),
                "worst_time": safe_float(worst_lap['S2_SECONDS']),
                "best_lap": safe_int(best_lap['LAP_NUMBER']),
                "worst_lap": safe_int(worst_lap['LAP_NUMBER'])
            })
    
    if pd.notna(best_lap['S3_SECONDS']) and pd.notna(worst_lap['S3_SECONDS']):
        s3_diff = worst_lap['S3_SECONDS'] - best_lap['S3_SECONDS']
        if s3_diff > 0.5:
            improvements.append({
                "sector": "Sector 3",
                "improvement_potential": safe_float(s3_diff),
                "best_time": safe_float(best_lap['S3_SECONDS']),
                "worst_time": safe_float(worst_lap['S3_SECONDS']),
                "best_lap": safe_int(best_lap['LAP_NUMBER']),
                "worst_lap": safe_int(worst_lap['LAP_NUMBER'])
            })
    
    # Lap time trend
    lap_times = []
    for _, row in vehicle_data.iterrows():
        if pd.notna(row['lap_time_seconds']):
            lap_times.append({
                "lap": int(row['LAP_NUMBER']),
                "lap_time": row['LAP_TIME'],
                "lap_time_seconds": safe_float(row['lap_time_seconds']),
                "sector_1": safe_float(row['S1_SECONDS']),
                "sector_2": safe_float(row['S2_SECONDS']),
                "sector_3": safe_float(row['S3_SECONDS'])
            })
    
    return {
        "vehicle_id": vehicle_id,
        "best_lap": {
            "lap_number": safe_int(best_lap['LAP_NUMBER']),
            "lap_time": best_lap['LAP_TIME'],
            "lap_time_seconds": safe_float(best_lap['lap_time_seconds']),
            "sector_1": safe_float(best_lap['S1_SECONDS']),
            "sector_2": safe_float(best_lap['S2_SECONDS']),
            "sector_3": safe_float(best_lap['S3_SECONDS'])
        },
        "worst_lap": {
            "lap_number": safe_int(worst_lap['LAP_NUMBER']),
            "lap_time": worst_lap['LAP_TIME'],
            "lap_time_seconds": safe_float(worst_lap['lap_time_seconds']),
            "sector_1": safe_float(worst_lap['S1_SECONDS']),
            "sector_2": safe_float(worst_lap['S2_SECONDS']),
            "sector_3": safe_float(worst_lap['S3_SECONDS'])
        },
        "sector_statistics": {
            "sector_1": {
                "best": safe_float(s1_times.min()) if len(s1_times) > 0 else None,
                "average": safe_float(s1_times.mean()) if len(s1_times) > 0 else None,
                "consistency": safe_float(s1_times.std()) if len(s1_times) > 1 else 0
            },
            "sector_2": {
                "best": safe_float(s2_times.min()) if len(s2_times) > 0 else None,
                "average": safe_float(s2_times.mean()) if len(s2_times) > 0 else None,
                "consistency": safe_float(s2_times.std()) if len(s2_times) > 1 else 0
            },
            "sector_3": {
                "best": safe_float(s3_times.min()) if len(s3_times) > 0 else None,
                "average": safe_float(s3_times.mean()) if len(s3_times) > 0 else None,
                "consistency": safe_float(s3_times.std()) if len(s3_times) > 1 else 0
            }
        },
        "improvement_opportunities": improvements,
        "lap_times": lap_times
    }

