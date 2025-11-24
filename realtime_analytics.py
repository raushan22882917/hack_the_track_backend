"""
Real-Time Analytics Module
Provides real-time gap calculations, pit window analysis, and strategy simulation
"""

from fastapi import HTTPException
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime, timedelta
import json

# Import vehicle mapping utility
try:
    from vehicle_mapping import get_vehicle_number_as_string, get_vehicle_info
except ImportError:
    # Fallback if vehicle_mapping not available
    def get_vehicle_number_as_string(vehicle_id: str) -> Optional[str]:
        """Extract car number from vehicle_id format GR86-XXX-YYY"""
        if '-' in vehicle_id:
            parts = vehicle_id.split('-')
            if len(parts) >= 3:
                try:
                    return str(int(parts[2]))  # Remove leading zeros
                except ValueError:
                    return parts[2]
        return vehicle_id
    
    def get_vehicle_info(vehicle_id: str) -> Optional[Dict]:
        return None


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


def parse_time_gap(gap_str: str) -> float:
    """Convert gap string (e.g., '+2.740' or '+1'14.985') to seconds"""
    if pd.isna(gap_str) or not gap_str or gap_str == '':
        return None
    try:
        gap_str = str(gap_str).strip()
        if gap_str.startswith('+'):
            gap_str = gap_str[1:]
        
        if "'" in gap_str:
            parts = gap_str.split("'")
            minutes = int(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
        
        return float(gap_str)
    except:
        return None


async def calculate_real_time_gaps(timestamp: Optional[str] = None) -> Dict:
    """
    Calculate real-time gaps between drivers
    Uses current leaderboard position and lap times
    """
    project_root = get_project_root()
    leaderboard_file = project_root / "logs" / "R1_leaderboard.csv"
    endurance_file = project_root / "logs" / "R1_section_endurance.csv"
    
    if not leaderboard_file.exists():
        raise HTTPException(status_code=404, detail="Leaderboard data not found")
    
    leaderboard_df = pd.read_csv(leaderboard_file, sep=";", low_memory=False)
    leaderboard_df.columns = leaderboard_df.columns.str.strip()
    
    # Get leader's elapsed time
    leader = leaderboard_df.iloc[0]
    leader_elapsed = parse_lap_time(leader.get('ELAPSED', None))
    
    if leader_elapsed is None:
        # Try to calculate from endurance data
        if endurance_file.exists():
            endurance_df = pd.read_csv(endurance_file, sep=";", low_memory=False)
            endurance_df.columns = endurance_df.columns.str.strip()
            leader_laps = endurance_df[endurance_df['NUMBER'].astype(str) == str(leader['NUMBER'])]
            if len(leader_laps) > 0:
                leader_elapsed = parse_lap_time(leader_laps.iloc[-1]['ELAPSED'])
    
    gaps = []
    
    for idx, row in leaderboard_df.iterrows():
        car_number = str(row['NUMBER'])
        position = int(row.get('POS', idx + 1))
        
        # Try to find vehicle_id from mapping (reverse lookup)
        vehicle_id_str = car_number  # Default to car number
        try:
            from vehicle_mapping import get_cached_mapping
            mapping = get_cached_mapping()
            # Find vehicle_id that maps to this car_number
            for vid, info in mapping.items():
                if info.get('car_number') and str(info['car_number']) == car_number:
                    vehicle_id_str = vid
                    break
        except:
            pass
        
        # Get gap to leader
        gap_to_leader = parse_time_gap(row.get('GAP_FIRST', None))
        gap_to_previous = parse_time_gap(row.get('GAP_PREVIOUS', None))
        
        # Calculate estimated time behind leader
        if gap_to_leader is None and leader_elapsed:
            # Estimate based on position and average lap time
            estimated_gap = (position - 1) * 2.0  # Rough estimate: 2 seconds per position
        else:
            estimated_gap = gap_to_leader
        
        gaps.append({
            "vehicle_id": vehicle_id_str,  # Use mapped vehicle_id if available
            "car_number": car_number,  # Also include car_number for reference
            "position": position,
            "gap_to_leader_seconds": float(gap_to_leader) if gap_to_leader else None,
            "gap_to_previous_seconds": float(gap_to_previous) if gap_to_previous else None,
            "estimated_gap_seconds": float(estimated_gap) if estimated_gap else None,
            "laps": int(row.get('LAPS', 0)),
            "best_lap_time": row.get('BEST_LAP_TIME', None)
        })
    
    return {
        "timestamp": timestamp or datetime.now().isoformat(),
        "leader": {
            "vehicle_id": str(leader['NUMBER']),
            "position": 1,
            "elapsed_time": leader.get('ELAPSED', None)
        },
        "gaps": gaps
    }


async def analyze_pit_window(vehicle_id: str, current_lap: int, total_laps: int = 27) -> Dict:
    """
    Analyze optimal pit stop window
    Considers tire degradation, position, and race strategy
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
        # List available vehicles for debugging
        available_numbers = sorted(endurance_df['NUMBER'].dropna().unique().astype(str).tolist())
        car_number, original_id = normalize_vehicle_id(vehicle_id)
        
        # Try to provide helpful suggestions
        suggestions = []
        if '-' in vehicle_id:
            parts = vehicle_id.split('-')
            if len(parts) >= 2:
                suggestions.append(f"chassis: {parts[1]}")
            if len(parts) >= 3:
                suggestions.append(f"car_number: {parts[2]}")
        
        error_msg = f"Vehicle '{vehicle_id}' not found. "
        error_msg += f"Tried: car_number={car_number}, original_id={original_id}"
        if suggestions:
            error_msg += f", {', '.join(suggestions)}"
        error_msg += f". Available vehicles in endurance data: {available_numbers[:20]}"
        
        raise HTTPException(status_code=404, detail=error_msg)
    
    vehicle_data = vehicle_data.sort_values('LAP_NUMBER')
    
    # Calculate lap times
    vehicle_data['lap_time_seconds'] = vehicle_data['LAP_TIME'].apply(parse_lap_time)
    valid_laps = vehicle_data[vehicle_data['lap_time_seconds'].notna()]
    
    if len(valid_laps) < 5:
        return {"error": "Insufficient lap data for pit window analysis"}
    
    # Estimate tire degradation (simplified: compare early vs late laps)
    early_laps = valid_laps[valid_laps['LAP_NUMBER'] <= 10]
    late_laps = valid_laps[valid_laps['LAP_NUMBER'] > 10]
    
    if len(early_laps) > 0 and len(late_laps) > 0:
        early_avg = early_laps['lap_time_seconds'].mean()
        late_avg = late_laps['lap_time_seconds'].mean()
        degradation_per_lap = (late_avg - early_avg) / (late_laps['LAP_NUMBER'].max() - early_laps['LAP_NUMBER'].min())
    else:
        degradation_per_lap = 0.1  # Default: 0.1 seconds per lap
    
    # Estimate pit stop time (from data if available, otherwise default)
    pit_stops = vehicle_data[vehicle_data['CROSSING_FINISH_LINE_IN_PIT'].notna()]
    if len(pit_stops) > 0 and 'PIT_TIME' in pit_stops.columns:
        avg_pit_time = float(pit_stops['PIT_TIME'].mean()) if pd.notna(pit_stops['PIT_TIME'].mean()) else 25.0
    else:
        avg_pit_time = 25.0  # Default: 25 seconds
    
    # Calculate optimal pit window
    remaining_laps = total_laps - current_lap
    
    # Strategy options
    strategies = []
    
    # Option 1: Pit now
    if current_lap < total_laps - 5:  # Don't pit if too close to end
        laps_after_pit = remaining_laps
        time_lost_in_pit = avg_pit_time
        time_gained_from_fresh_tires = -degradation_per_lap * laps_after_pit * 0.5  # Estimate
        net_effect = time_lost_in_pit + time_gained_from_fresh_tires
        
        strategies.append({
            "strategy": "Pit Now",
            "pit_lap": current_lap,
            "laps_on_fresh_tires": laps_after_pit,
            "estimated_pit_time": float(avg_pit_time),
            "estimated_time_gain": float(time_gained_from_fresh_tires),
            "net_effect_seconds": float(net_effect),
            "recommendation": "Good" if net_effect < 5 else "Consider waiting"
        })
    
    # Option 2: Pit in 5 laps
    if current_lap + 5 < total_laps:
        laps_after_pit = remaining_laps - 5
        time_lost_in_pit = avg_pit_time
        time_gained_from_fresh_tires = -degradation_per_lap * laps_after_pit * 0.5
        net_effect = time_lost_in_pit + time_gained_from_fresh_tires
        
        strategies.append({
            "strategy": "Pit in 5 Laps",
            "pit_lap": current_lap + 5,
            "laps_on_fresh_tires": laps_after_pit,
            "estimated_pit_time": float(avg_pit_time),
            "estimated_time_gain": float(time_gained_from_fresh_tires),
            "net_effect_seconds": float(net_effect),
            "recommendation": "Good" if net_effect < 5 else "Consider waiting"
        })
    
    # Option 3: No pit (if already pitted or near end)
    if current_lap > total_laps - 10:
        strategies.append({
            "strategy": "No Pit Stop",
            "pit_lap": None,
            "laps_on_fresh_tires": 0,
            "estimated_pit_time": 0,
            "estimated_time_gain": 0,
            "net_effect_seconds": 0,
            "recommendation": "Finish race on current tires"
        })
    
    # Find best strategy
    best_strategy = min(strategies, key=lambda x: x['net_effect_seconds']) if strategies else None
    
    # Convert strategies to optimal_windows format for frontend compatibility
    optimal_windows = []
    for strategy in strategies:
        if strategy.get('pit_lap'):
            optimal_windows.append({
                "start_lap": max(1, strategy['pit_lap'] - 2),
                "end_lap": min(total_laps, strategy['pit_lap'] + 2),
                "optimal_lap": strategy['pit_lap'],
                "recommendation": strategy.get('recommendation', 'Consider'),
                "net_effect_seconds": strategy.get('net_effect_seconds', 0),
                "estimated_time_gain": strategy.get('estimated_time_gain', 0)
            })
    
    # If no strategies, create a default window
    if not optimal_windows and remaining_laps > 5:
        optimal_windows.append({
            "start_lap": current_lap + 1,
            "end_lap": min(total_laps, current_lap + 10),
            "optimal_lap": current_lap + 5,
            "recommendation": "Good window for pit stop",
            "net_effect_seconds": 0,
            "estimated_time_gain": 0
        })
    
    return {
        "vehicle_id": vehicle_id,
        "current_lap": current_lap,
        "total_laps": total_laps,
        "remaining_laps": remaining_laps,
        "tire_degradation_per_lap_seconds": float(degradation_per_lap),
        "estimated_pit_time_seconds": float(avg_pit_time),
        "strategies": strategies,
        "recommended_strategy": best_strategy,
        "optimal_windows": optimal_windows  # Frontend expects this
    }


async def simulate_strategy_scenario(
    vehicle_id: str,
    pit_lap: int,
    pit_time: float = 25.0,
    total_laps: int = 27
) -> Dict:
    """
    Simulate "what if" scenario for pit stop strategy
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
            detail=f"Vehicle {vehicle_id} (tried car_number: {car_number}) not found. Available: {available_numbers[:20]}"
        )
    
    vehicle_data = vehicle_data.sort_values('LAP_NUMBER')
    vehicle_data['lap_time_seconds'] = vehicle_data['LAP_TIME'].apply(parse_lap_time)
    valid_laps = vehicle_data[vehicle_data['lap_time_seconds'].notna()]
    
    if len(valid_laps) < 3:
        return {"error": "Insufficient lap data"}
    
    # Calculate baseline (no pit)
    baseline_total_time = valid_laps['lap_time_seconds'].sum()
    
    # Estimate tire degradation
    early_laps = valid_laps[valid_laps['LAP_NUMBER'] <= 10]
    late_laps = valid_laps[valid_laps['LAP_NUMBER'] > 10]
    
    if len(early_laps) > 0 and len(late_laps) > 0:
        degradation_per_lap = (late_laps['lap_time_seconds'].mean() - early_laps['lap_time_seconds'].mean()) / max(1, late_laps['LAP_NUMBER'].max() - early_laps['LAP_NUMBER'].min())
    else:
        degradation_per_lap = 0.1
    
    # Simulate with pit stop
    laps_before_pit = valid_laps[valid_laps['LAP_NUMBER'] < pit_lap]
    laps_after_pit = valid_laps[valid_laps['LAP_NUMBER'] >= pit_lap]
    
    # Before pit: normal degradation
    time_before_pit = laps_before_pit['lap_time_seconds'].sum() if len(laps_before_pit) > 0 else 0
    
    # After pit: fresh tires (faster)
    if len(laps_after_pit) > 0:
        avg_lap_before_pit = laps_before_pit['lap_time_seconds'].mean() if len(laps_before_pit) > 0 else valid_laps['lap_time_seconds'].mean()
        fresh_tire_advantage = -0.5  # Fresh tires are ~0.5s faster per lap initially
        laps_after_pit_count = len(laps_after_pit)
        time_after_pit = (avg_lap_before_pit + fresh_tire_advantage) * laps_after_pit_count
    else:
        time_after_pit = 0
    
    # Total time with pit
    total_time_with_pit = time_before_pit + pit_time + time_after_pit
    
    # Calculate difference
    time_difference = total_time_with_pit - baseline_total_time
    
    # Ensure all values are Python native types, not numpy types
    baseline_avg = valid_laps['lap_time_seconds'].mean()
    baseline_avg_clean = float(baseline_avg) if pd.notna(baseline_avg) else None
    
    return {
        "vehicle_id": vehicle_id,
        "scenario": {
            "pit_lap": int(pit_lap),
            "pit_time_seconds": float(pit_time),
            "total_laps": int(total_laps)
        },
        "baseline": {
            "total_time_seconds": float(baseline_total_time) if pd.notna(baseline_total_time) else None,
            "average_lap_time": baseline_avg_clean
        },
        "with_pit_stop": {
            "time_before_pit": float(time_before_pit) if pd.notna(time_before_pit) else None,
            "pit_time": float(pit_time),
            "time_after_pit": float(time_after_pit) if pd.notna(time_after_pit) else None,
            "total_time_seconds": float(total_time_with_pit) if pd.notna(total_time_with_pit) else None
        },
        "comparison": {
            "time_difference_seconds": float(time_difference) if pd.notna(time_difference) else None,
            "faster": bool(time_difference < 0) if pd.notna(time_difference) else None,
            "recommendation": "Pit stop beneficial" if (pd.notna(time_difference) and time_difference < -5) else "Pit stop not beneficial"
        }
    }


async def get_tire_degradation_estimate(vehicle_id: str) -> Dict:
    """
    Estimate tire degradation based on lap time trends
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
            detail=f"Vehicle {vehicle_id} (tried car_number: {car_number}) not found. Available: {available_numbers[:20]}"
        )
    
    vehicle_data = vehicle_data.sort_values('LAP_NUMBER')
    vehicle_data['lap_time_seconds'] = vehicle_data['LAP_TIME'].apply(parse_lap_time)
    valid_laps = vehicle_data[vehicle_data['lap_time_seconds'].notna()]
    
    if len(valid_laps) < 5:
        return {"error": "Insufficient lap data"}
    
    # Calculate degradation trend
    lap_numbers = valid_laps['LAP_NUMBER'].values
    lap_times = valid_laps['lap_time_seconds'].values
    
    # Simple linear regression for degradation
    if len(lap_numbers) > 1:
        degradation_rate = np.polyfit(lap_numbers, lap_times, 1)[0]  # Slope
    else:
        degradation_rate = 0
    
    # Calculate degradation per lap
    degradation_per_lap = degradation_rate
    
    # Estimate remaining tire life
    best_lap = valid_laps['lap_time_seconds'].min()
    worst_lap = valid_laps['lap_time_seconds'].max()
    degradation_so_far = worst_lap - best_lap
    
    # Estimate when tires will be significantly degraded (2 seconds slower than best)
    if degradation_per_lap > 0:
        laps_until_degraded = (2.0 - degradation_so_far) / degradation_per_lap
    else:
        laps_until_degraded = float('inf')
    
    # Current tire condition
    current_lap = valid_laps['LAP_NUMBER'].max()
    current_lap_time = valid_laps[valid_laps['LAP_NUMBER'] == current_lap]['lap_time_seconds'].iloc[0] if len(valid_laps[valid_laps['LAP_NUMBER'] == current_lap]) > 0 else None
    
    if current_lap_time:
        time_vs_best = current_lap_time - best_lap
        tire_condition = "Fresh" if time_vs_best < 0.5 else "Good" if time_vs_best < 1.0 else "Degraded" if time_vs_best < 2.0 else "Very Degraded"
    else:
        tire_condition = "Unknown"
        time_vs_best = None
    
    return {
        "vehicle_id": vehicle_id,
        "degradation_analysis": {
            "degradation_per_lap_seconds": float(degradation_per_lap),
            "total_degradation_seconds": float(degradation_so_far),
            "best_lap_time": float(best_lap),
            "worst_lap_time": float(worst_lap),
            "estimated_laps_until_degraded": float(laps_until_degraded) if laps_until_degraded != float('inf') else None
        },
        "current_condition": {
            "current_lap": int(current_lap),
            "current_lap_time": float(current_lap_time) if current_lap_time else None,
            "time_vs_best_seconds": float(time_vs_best) if time_vs_best else None,
            "tire_condition": tire_condition
        },
        "recommendation": f"Tires are {tire_condition.lower()}. Consider pit stop in {int(laps_until_degraded)} laps" if laps_until_degraded != float('inf') and laps_until_degraded < 10 else f"Tires are {tire_condition.lower()}"
    }

