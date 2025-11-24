"""
Post-Event Analysis Module
Uses predictive models to tell the story of the race, revealing key moments
and strategic decisions that led to the outcome.

This module compares predicted vs actual performance to identify:
- Strategic moments (pit stops, tire changes, etc.)
- Performance deviations (over/under-performance vs predictions)
- Critical decisions that affected race outcomes
- Race narrative with key turning points
"""

from fastapi import HTTPException
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Try to import ML libraries
try:
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️ Warning: joblib not available. Post-event analysis will be limited.")


def get_project_root():
    """Get the project root directory"""
    return Path(__file__).parent


def safe_float(value):
    """Safely convert to float, handling NaN and None"""
    if pd.isna(value) or value is None:
        return None
    try:
        result = float(value)
        if pd.isna(result) or np.isnan(result) or np.isinf(result):
            return None
        return result
    except:
        return None


def safe_int(value):
    """Safely convert to int, handling NaN and None"""
    if pd.isna(value) or value is None:
        return None
    try:
        return int(value)
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


def get_model_features(db_dataframe: pd.DataFrame) -> List[str]:
    """Get feature columns expected by the model"""
    if 'lap_time_seconds' in db_dataframe.columns:
        return db_dataframe.drop(columns=['lap_time_seconds']).columns.tolist()
    return db_dataframe.columns.tolist()


def predict_single_lap(model, features_df: pd.DataFrame) -> float:
    """Predict a single lap time using the model"""
    try:
        pred_log = model.predict(features_df)
        pred_sec = float(np.expm1(pred_log)[0])
        return max(25.0, pred_sec)  # Enforce minimum lap time
    except Exception as e:
        print(f"⚠️ Prediction error: {e}")
        return 100.0  # Default fallback


def get_features_for_lap(db: pd.DataFrame, race_session: str, vehicle_id: str, lap: int) -> Optional[pd.DataFrame]:
    """Get feature row for a specific lap"""
    row = db[
        (db['meta_session'] == race_session) &
        (db['vehicle_id'] == vehicle_id) &
        (db['lap'] == lap)
    ]
    return row if not row.empty else None


async def generate_post_event_analysis(
    track_name: str,
    race_session: str,
    predictive_models: Dict,
    predictive_databases: Dict,
    min_lap_time: float = 25.0
) -> Dict:
    """
    Generate comprehensive post-event analysis comparing predicted vs actual performance
    
    This creates a narrative story of the race by:
    1. Comparing predicted vs actual lap times for all drivers
    2. Identifying strategic moments (pit stops, tire degradation, etc.)
    3. Finding performance deviations (over/under-performance)
    4. Highlighting critical decisions that affected outcomes
    5. Creating a race narrative with key turning points
    """
    
    if not ML_AVAILABLE:
        raise HTTPException(status_code=503, detail="ML libraries not available")
    
    if track_name not in predictive_models:
        raise HTTPException(status_code=400, detail=f"Model not loaded for track: {track_name}")
    
    if track_name not in predictive_databases:
        raise HTTPException(status_code=400, detail=f"Database not loaded for track: {track_name}")
    
    model = predictive_models[track_name]
    db = predictive_databases[track_name]
    
    # Filter to race session
    session_df = db[db['meta_session'] == race_session].copy()
    
    if session_df.empty:
        raise HTTPException(status_code=404, detail=f"No data found for session: {race_session}")
    
    # Get model features
    MODEL_FEATURES = get_model_features(db)
    
    # Track statistics
    all_vehicles = session_df['vehicle_id'].unique()
    max_lap = int(session_df['lap'].max())
    
    # Default values for tracks
    default_values = {
        'Barber': 100.92,
        'Sonoma': 120.53,
        'Road America': 158.46,
        'Circuit of the Americas': 167.83,
        'Virginia International Raceway': 133.60
    }
    default_lap_time = default_values.get(track_name, 100.0)
    
    # Analyze each vehicle
    vehicle_analyses = []
    strategic_moments = []
    performance_deviations = []
    
    for vehicle_id in all_vehicles:
        vehicle_data = session_df[session_df['vehicle_id'] == vehicle_id].sort_values('lap')
        
        if vehicle_data.empty:
            continue
        
        # Get actual lap times
        actual_laps = []
        predicted_laps = []
        lap_analysis = []
        
        lap_history = []
        current_features = None
        
        for idx, row in vehicle_data.iterrows():
            lap_num = int(row['lap'])
            actual_time = safe_float(row.get('lap_time_seconds'))
            
            if actual_time is None or actual_time < min_lap_time:
                continue
            
            # Get features for this lap
            features_row = get_features_for_lap(db, race_session, vehicle_id, lap_num)
            
            if features_row is None or features_row.empty:
                # Create synthetic features based on previous lap
                if current_features is not None:
                    features_row = current_features.copy()
                    features_row.loc[features_row.index[0], 'lap'] = lap_num
                    # Update rolling features
                    if lap_history:
                        features_row.loc[features_row.index[0], 'last_normal_lap_time'] = lap_history[-1]
                        if len(lap_history) >= 3:
                            features_row.loc[features_row.index[0], 'rolling_3_normal_lap_avg'] = np.mean(lap_history[-3:])
                else:
                    # Use template row
                    template = db[
                        (db['meta_session'] == race_session) &
                        (db['vehicle_id'] == vehicle_id)
                    ].head(1)
                    
                    if template.empty:
                        template = db[db['meta_session'] == race_session].head(1)
                    
                    if not template.empty:
                        features_row = template.copy()
                        features_row.loc[features_row.index[0], 'lap'] = lap_num
                        features_row.loc[features_row.index[0], 'is_out_lap'] = 0
                        features_row.loc[features_row.index[0], 'is_normal_lap'] = 1
                        features_row.loc[features_row.index[0], 'last_normal_lap_time'] = default_lap_time if not lap_history else lap_history[-1]
                        if lap_history and len(lap_history) >= 3:
                            features_row.loc[features_row.index[0], 'rolling_3_normal_lap_avg'] = np.mean(lap_history[-3:])
            
            if features_row is not None and not features_row.empty:
                # Predict lap time
                try:
                    predicted_time = predict_single_lap(model, features_row[MODEL_FEATURES])
                except Exception as e:
                    predicted_time = default_lap_time
                
                predicted_laps.append(predicted_time)
                actual_laps.append(actual_time)
                
                deviation = actual_time - predicted_time
                deviation_pct = (deviation / predicted_time * 100) if predicted_time > 0 else 0
                
                lap_analysis.append({
                    'lap': lap_num,
                    'actual_time': actual_time,
                    'predicted_time': predicted_time,
                    'deviation_seconds': deviation,
                    'deviation_percent': deviation_pct,
                    'is_out_lap': safe_int(features_row.iloc[0].get('is_out_lap', 0)) == 1,
                    'laps_on_tires': safe_int(features_row.iloc[0].get('laps_on_tires', 1)),
                    'fuel_load_proxy': safe_float(features_row.iloc[0].get('fuel_load_proxy')),
                })
                
                # Track significant deviations (strategic moments)
                if abs(deviation) > 2.0:  # More than 2 seconds difference
                    strategic_moments.append({
                        'type': 'performance_deviation',
                        'lap': lap_num,
                        'vehicle_id': vehicle_id,
                        'actual_time': actual_time,
                        'predicted_time': predicted_time,
                        'deviation_seconds': deviation,
                        'reason': 'out_lap' if safe_int(features_row.iloc[0].get('is_out_lap', 0)) == 1 else 'performance_change'
                    })
                
                # Detect pit stops (out laps)
                if safe_int(features_row.iloc[0].get('is_out_lap', 0)) == 1:
                    strategic_moments.append({
                        'type': 'pit_stop',
                        'lap': lap_num,
                        'vehicle_id': vehicle_id,
                        'lap_time': actual_time,
                        'laps_on_tires': safe_int(features_row.iloc[0].get('laps_on_tires', 1))
                    })
                
                # Update for next iteration
                current_features = features_row.copy()
                lap_history.append(actual_time)
                if len(lap_history) > 10:
                    lap_history.pop(0)
        
        if not lap_analysis:
            continue
        
        # Calculate overall statistics
        actual_times = [l['actual_time'] for l in lap_analysis]
        predicted_times = [l['predicted_time'] for l in lap_analysis]
        deviations = [l['deviation_seconds'] for l in lap_analysis]
        
        avg_deviation = np.mean(deviations)
        total_deviation = sum(deviations)
        consistency_score = 1.0 / (1.0 + np.std(deviations))  # Higher is better
        
        # Performance classification
        if avg_deviation < -1.0:
            performance_class = 'over_performed'  # Faster than predicted
        elif avg_deviation > 1.0:
            performance_class = 'under_performed'  # Slower than predicted
        else:
            performance_class = 'as_expected'
        
        vehicle_analyses.append({
            'vehicle_id': vehicle_id,
            'total_laps': len(lap_analysis),
            'actual_total_time': sum(actual_times),
            'predicted_total_time': sum(predicted_times),
            'total_time_deviation': total_deviation,
            'average_deviation': avg_deviation,
            'consistency_score': consistency_score,
            'best_lap_actual': min(actual_times),
            'best_lap_predicted': min(predicted_times),
            'average_lap_actual': np.mean(actual_times),
            'average_lap_predicted': np.mean(predicted_times),
            'performance_class': performance_class,
            'lap_by_lap': lap_analysis[:50]  # Limit for performance
        })
        
        # Track significant performance deviations
        if abs(avg_deviation) > 1.5:
            performance_deviations.append({
                'vehicle_id': vehicle_id,
                'deviation_seconds': avg_deviation,
                'total_time_impact': total_deviation,
                'performance_class': performance_class,
                'consistency_score': consistency_score
            })
    
    # Sort strategic moments by lap
    strategic_moments.sort(key=lambda x: x.get('lap', 0))
    
    # Sort performance deviations by impact
    performance_deviations.sort(key=lambda x: abs(x['total_time_impact']), reverse=True)
    
    # Generate race narrative
    narrative = generate_race_narrative(
        vehicle_analyses,
        strategic_moments,
        performance_deviations,
        max_lap
    )
    
    # Calculate race statistics
    race_stats = {
        'total_vehicles': len(vehicle_analyses),
        'total_laps': max_lap,
        'vehicles_finished': len([v for v in vehicle_analyses if v['total_laps'] >= max_lap * 0.9]),
        'total_pit_stops': len([m for m in strategic_moments if m['type'] == 'pit_stop']),
        'significant_deviations': len([m for m in strategic_moments if m['type'] == 'performance_deviation'])
    }
    
    return {
        'track_name': track_name,
        'race_session': race_session,
        'race_statistics': race_stats,
        'vehicle_analyses': vehicle_analyses,
        'strategic_moments': strategic_moments[:100],  # Limit for performance
        'performance_deviations': performance_deviations,
        'narrative': narrative,
        'key_insights': generate_key_insights(vehicle_analyses, strategic_moments, performance_deviations)
    }


def generate_race_narrative(
    vehicle_analyses: List[Dict],
    strategic_moments: List[Dict],
    performance_deviations: List[Dict],
    max_lap: int
) -> Dict:
    """Generate a narrative story of the race"""
    
    narrative_sections = []
    
    # Introduction
    narrative_sections.append({
        'section': 'race_overview',
        'title': 'Race Overview',
        'content': f"The race featured {len(vehicle_analyses)} vehicles competing over {max_lap} laps. "
                   f"Throughout the race, {len([m for m in strategic_moments if m['type'] == 'pit_stop'])} pit stops were made, "
                   f"and {len(performance_deviations)} drivers showed significant performance deviations from predictions."
    })
    
    # Early race (first 25% of laps)
    early_lap_threshold = max_lap * 0.25
    early_moments = [m for m in strategic_moments if m.get('lap', 0) <= early_lap_threshold]
    
    if early_moments:
        narrative_sections.append({
            'section': 'early_race',
            'title': 'Early Race Dynamics',
            'content': f"In the opening stages (laps 1-{int(early_lap_threshold)}), the field began to spread out. "
                       f"{len([m for m in early_moments if m['type'] == 'pit_stop'])} early pit stops occurred, "
                       f"setting the stage for strategic variations throughout the race."
        })
    
    # Mid race (25-75% of laps)
    mid_start = max_lap * 0.25
    mid_end = max_lap * 0.75
    mid_moments = [m for m in strategic_moments if mid_start < m.get('lap', 0) <= mid_end]
    
    if mid_moments:
        narrative_sections.append({
            'section': 'mid_race',
            'title': 'Mid-Race Strategy',
            'content': f"During the middle phase (laps {int(mid_start)}-{int(mid_end)}), strategic decisions came to the fore. "
                       f"Tire degradation and fuel management became critical factors as drivers balanced pace with longevity."
        })
    
    # Performance highlights
    if performance_deviations:
        top_overperformer = next((d for d in performance_deviations if d['performance_class'] == 'over_performed'), None)
        top_underperformer = next((d for d in performance_deviations if d['performance_class'] == 'under_performed'), None)
        
        if top_overperformer:
            narrative_sections.append({
                'section': 'performance_highlights',
                'title': 'Performance Highlights',
                'content': f"Vehicle {top_overperformer['vehicle_id']} exceeded expectations, finishing "
                           f"{abs(top_overperformer['total_time_impact']):.1f} seconds faster than predicted. "
                           f"This represents one of the standout performances of the race."
            })
    
    # Final phase
    late_lap_threshold = max_lap * 0.75
    late_moments = [m for m in strategic_moments if m.get('lap', 0) > late_lap_threshold]
    
    if late_moments:
        narrative_sections.append({
            'section': 'final_phase',
            'title': 'Final Phase',
            'content': f"In the closing stages (laps {int(late_lap_threshold)}-{max_lap}), drivers pushed to the limit. "
                       f"Final pit stop strategies and tire management decisions would ultimately determine the final standings."
        })
    
    return {
        'sections': narrative_sections,
        'total_sections': len(narrative_sections)
    }


def generate_key_insights(
    vehicle_analyses: List[Dict],
    strategic_moments: List[Dict],
    performance_deviations: List[Dict]
) -> List[Dict]:
    """Generate key insights from the analysis"""
    
    insights = []
    
    # Find most consistent driver
    if vehicle_analyses:
        most_consistent = max(vehicle_analyses, key=lambda x: x.get('consistency_score', 0))
        insights.append({
            'type': 'consistency',
            'title': 'Most Consistent Performance',
            'description': f"Vehicle {most_consistent['vehicle_id']} showed the most consistent performance "
                         f"relative to predictions, with a consistency score of {most_consistent['consistency_score']:.3f}.",
            'vehicle_id': most_consistent['vehicle_id'],
            'metric': most_consistent['consistency_score']
        })
    
    # Find biggest overperformer
    overperformers = [d for d in performance_deviations if d['performance_class'] == 'over_performed']
    if overperformers:
        top_over = max(overperformers, key=lambda x: abs(x['total_time_impact']))
        insights.append({
            'type': 'overperformance',
            'title': 'Biggest Overperformance',
            'description': f"Vehicle {top_over['vehicle_id']} exceeded predictions by {abs(top_over['total_time_impact']):.1f} seconds total, "
                         f"averaging {abs(top_over['deviation_seconds']):.2f} seconds faster per lap than expected.",
            'vehicle_id': top_over['vehicle_id'],
            'metric': abs(top_over['total_time_impact'])
        })
    
    # Find biggest underperformer
    underperformers = [d for d in performance_deviations if d['performance_class'] == 'under_performed']
    if underperformers:
        top_under = max(underperformers, key=lambda x: abs(x['total_time_impact']))
        insights.append({
            'type': 'underperformance',
            'title': 'Biggest Underperformance',
            'description': f"Vehicle {top_under['vehicle_id']} fell short of predictions by {abs(top_under['total_time_impact']):.1f} seconds total, "
                         f"averaging {abs(top_under['deviation_seconds']):.2f} seconds slower per lap than expected.",
            'vehicle_id': top_under['vehicle_id'],
            'metric': abs(top_under['total_time_impact'])
        })
    
    # Pit stop strategy insight
    pit_stops = [m for m in strategic_moments if m['type'] == 'pit_stop']
    if pit_stops:
        avg_pit_lap = np.mean([m['lap'] for m in pit_stops])
        insights.append({
            'type': 'strategy',
            'title': 'Pit Stop Strategy',
            'description': f"An average of {len(pit_stops) / len(vehicle_analyses) if vehicle_analyses else 0:.1f} pit stops per vehicle, "
                         f"with the average pit stop occurring around lap {int(avg_pit_lap)}.",
            'metric': len(pit_stops)
        })
    
    return insights

