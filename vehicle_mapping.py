"""
Vehicle Mapping Utility
Maps vehicle_id (GR86-XXX-YYY) to vehicle_number, car_number (NUMBER), and DRIVER_NUMBER

Data Sources:
1. barber/R1_barber_lap_start.csv - Contains vehicle_id -> vehicle_number mapping
2. logs/R1_section_endurance.csv - Contains NUMBER (car_number) -> DRIVER_NUMBER mapping

The relationship is:
- vehicle_id (e.g., GR86-010-16) -> vehicle_number (16) from barber files
- vehicle_number matches NUMBER in endurance file
- NUMBER -> DRIVER_NUMBER from endurance file
"""

import pandas as pd
import json
from pathlib import Path
from typing import Optional, Dict


def get_project_root():
    """Get project root directory"""
    return Path(__file__).parent


def load_vehicle_mapping() -> Dict:
    """
    Load vehicle mapping from barber files and endurance data
    
    Returns:
        Dict mapping vehicle_id to {
            'vehicle_number': int,  # From barber files (car number)
            'car_number': int,      # NUMBER from endurance file (same as vehicle_number if exists)
            'driver_number': int,   # DRIVER_NUMBER from endurance file
            'has_endurance_data': bool
        }
    """
    project_root = get_project_root()
    
    # Try to load cached mapping first
    mapping_file = project_root / "vehicle_driver_mapping.json"
    if mapping_file.exists():
        try:
            with open(mapping_file, 'r') as f:
                return json.load(f)
        except:
            pass
    
    # Build mapping from source files
    lap_start_file = project_root / "barber" / "R1_barber_lap_start.csv"
    endurance_file = project_root / "logs" / "R1_section_endurance.csv"
    
    mapping = {}
    
    if lap_start_file.exists():
        # Get vehicle_id -> vehicle_number from barber file
        lap_df = pd.read_csv(lap_start_file)
        vehicle_mapping = lap_df[['vehicle_id', 'vehicle_number']].drop_duplicates()
        
        # Get NUMBER -> DRIVER_NUMBER from endurance file
        if endurance_file.exists():
            endurance_df = pd.read_csv(endurance_file, sep=';')
            endurance_df.columns = endurance_df.columns.str.strip()
            number_driver_mapping = endurance_df[['NUMBER', 'DRIVER_NUMBER']].drop_duplicates()
            
            # Merge
            merged = vehicle_mapping.merge(
                number_driver_mapping,
                left_on='vehicle_number',
                right_on='NUMBER',
                how='left'
            )
            
            for _, row in merged.iterrows():
                vehicle_id = str(row['vehicle_id'])
                vehicle_number = int(row['vehicle_number']) if pd.notna(row['vehicle_number']) else None
                car_number = int(row['NUMBER']) if pd.notna(row['NUMBER']) else None
                driver_number = int(row['DRIVER_NUMBER']) if pd.notna(row['DRIVER_NUMBER']) else None
                
                mapping[vehicle_id] = {
                    'vehicle_number': vehicle_number,
                    'car_number': car_number,
                    'driver_number': driver_number,
                    'has_endurance_data': pd.notna(row['NUMBER'])
                }
        else:
            # Just vehicle_id -> vehicle_number if no endurance file
            for _, row in vehicle_mapping.iterrows():
                vehicle_id = str(row['vehicle_id'])
                vehicle_number = int(row['vehicle_number']) if pd.notna(row['vehicle_number']) else None
                mapping[vehicle_id] = {
                    'vehicle_number': vehicle_number,
                    'car_number': vehicle_number,
                    'driver_number': None,
                    'has_endurance_data': False
                }
    
    # Cache the mapping
    try:
        with open(mapping_file, 'w') as f:
            json.dump(mapping, f, indent=2)
    except:
        pass
    
    return mapping


def get_vehicle_number_as_string(vehicle_id: str) -> Optional[str]:
    """
    Get vehicle_number (car number) as string for a given vehicle_id
    
    Args:
        vehicle_id: Vehicle ID in format GR86-XXX-YYY
        
    Returns:
        Vehicle number as string, or None if not found
    """
    mapping = load_vehicle_mapping()
    vehicle_info = mapping.get(vehicle_id)
    
    if vehicle_info and vehicle_info.get('car_number') is not None:
        return str(vehicle_info['car_number'])
    
    # Fallback: extract car number from vehicle_id
    if '-' in vehicle_id:
        parts = vehicle_id.split('-')
        if len(parts) >= 3:
            car_num = parts[2]
            try:
                return str(int(car_num))  # Remove leading zeros
            except ValueError:
                return car_num
    
    return None


def get_driver_number(vehicle_id: str) -> Optional[int]:
    """
    Get DRIVER_NUMBER for a given vehicle_id
    
    Args:
        vehicle_id: Vehicle ID in format GR86-XXX-YYY
        
    Returns:
        Driver number as int, or None if not found
    """
    mapping = load_vehicle_mapping()
    vehicle_info = mapping.get(vehicle_id)
    
    if vehicle_info:
        return vehicle_info.get('driver_number')
    
    return None


def get_vehicle_info(vehicle_id: str) -> Optional[Dict]:
    """
    Get complete vehicle information
    
    Args:
        vehicle_id: Vehicle ID in format GR86-XXX-YYY
        
    Returns:
        Dict with vehicle_number, car_number, driver_number, has_endurance_data
        or None if not found
    """
    mapping = load_vehicle_mapping()
    return mapping.get(vehicle_id)


# Pre-load mapping on import
_vehicle_mapping_cache = None

def get_cached_mapping():
    """Get cached vehicle mapping"""
    global _vehicle_mapping_cache
    if _vehicle_mapping_cache is None:
        _vehicle_mapping_cache = load_vehicle_mapping()
    return _vehicle_mapping_cache

