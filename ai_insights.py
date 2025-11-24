"""
AI-Powered Driver Insights Module
Uses pattern analysis and intelligent recommendations to provide actionable insights
"""

from fastapi import HTTPException
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import json


def get_project_root():
    """Get the project root directory"""
    return Path(__file__).parent


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


def analyze_driver_patterns(vehicle_id: str) -> Dict:
    """
    AI-powered analysis of driver patterns and behaviors
    Identifies trends, anomalies, and improvement areas
    """
    try:
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
        
        # Convert lap times to seconds
        def parse_lap_time(lap_time_str: str) -> float:
            if pd.isna(lap_time_str) or not lap_time_str:
                return None
            try:
                parts = str(lap_time_str).split(':')
                if len(parts) == 2:
                    return int(parts[0]) * 60 + float(parts[1])
                return float(parts[0])
            except:
                return None
        
        vehicle_data['lap_time_seconds'] = vehicle_data['LAP_TIME'].apply(parse_lap_time)
        valid_laps = vehicle_data[vehicle_data['lap_time_seconds'].notna()]
        
        if len(valid_laps) < 3:
            return {
                "error": "Insufficient data for analysis",
                "vehicle_id": vehicle_id,
                "valid_laps_count": len(valid_laps),
                "total_rows": len(vehicle_data)
            }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"❌ Error in analyze_driver_patterns for {vehicle_id} (data loading):")
        print(error_trace)
        raise HTTPException(status_code=500, detail=f"Error loading data: {str(e)}")
    
    insights = []
    recommendations = []
    patterns = []
    
    # 1. Performance Trend Analysis
    lap_times = valid_laps['lap_time_seconds'].values
    if len(lap_times) > 5:
        # Calculate trend (improving, declining, or stable)
        first_half = np.mean(lap_times[:len(lap_times)//2])
        second_half = np.mean(lap_times[len(lap_times)//2:])
        trend_diff = second_half - first_half
        
        if trend_diff < -0.5:
            insights.append({
                "type": "positive_trend",
                "category": "Performance Trend",
                "title": "Improving Performance",
                "description": f"Lap times improved by {abs(trend_diff):.2f}s in the second half of the race",
                "severity": "low",
                "action": "Maintain current driving style and focus on consistency"
            })
        elif trend_diff > 0.5:
            insights.append({
                "type": "negative_trend",
                "category": "Performance Trend",
                "title": "Performance Decline",
                "description": f"Lap times slowed by {trend_diff:.2f}s in the second half - possible tire degradation or fatigue",
                "severity": "medium",
                "action": "Review tire management strategy and physical conditioning"
            })
    
    # 2. Consistency Analysis
    lap_time_std = np.std(lap_times)
    lap_time_mean = np.mean(lap_times)
    consistency_score = (lap_time_std / lap_time_mean) * 100
    
    if consistency_score < 1.0:
        insights.append({
            "type": "excellent_consistency",
            "category": "Consistency",
            "title": "Excellent Consistency",
            "description": f"Very consistent lap times with only {consistency_score:.2f}% variation",
            "severity": "low",
            "action": "Maintain consistency while pushing for faster lap times"
        })
    elif consistency_score > 3.0:
        insights.append({
            "type": "poor_consistency",
            "category": "Consistency",
            "title": "Inconsistent Performance",
            "description": f"High variation in lap times ({consistency_score:.2f}%) - focus on consistency",
            "severity": "high",
            "action": "Practice maintaining consistent racing lines and braking points"
        })
    
    # 3. Sector Analysis
    try:
        s1_times = valid_laps['S1_SECONDS'].dropna() if 'S1_SECONDS' in valid_laps.columns else pd.Series(dtype=float)
        s2_times = valid_laps['S2_SECONDS'].dropna() if 'S2_SECONDS' in valid_laps.columns else pd.Series(dtype=float)
        s3_times = valid_laps['S3_SECONDS'].dropna() if 'S3_SECONDS' in valid_laps.columns else pd.Series(dtype=float)
    except Exception as e:
        print(f"⚠️ Error accessing sector times: {e}")
        s1_times = pd.Series(dtype=float)
        s2_times = pd.Series(dtype=float)
        s3_times = pd.Series(dtype=float)
    
    if len(s1_times) > 0 and len(s2_times) > 0 and len(s3_times) > 0:
        s1_std = s1_times.std()
        s2_std = s2_times.std()
        s3_std = s3_times.std()
        
        # Find most inconsistent sector
        sector_stds = {
            "Sector 1": s1_std,
            "Sector 2": s2_std,
            "Sector 3": s3_std
        }
        most_inconsistent = max(sector_stds.items(), key=lambda x: x[1])
        
        if most_inconsistent[1] > 0.5:
            insights.append({
                "type": "sector_inconsistency",
                "category": "Sector Performance",
                "title": f"Inconsistent {most_inconsistent[0]}",
                "description": f"{most_inconsistent[0]} shows {most_inconsistent[1]:.2f}s variation - largest inconsistency",
                "severity": "medium",
                "action": f"Focus on consistent racing line and braking points in {most_inconsistent[0]}"
            })
        
        # Find weakest sector (highest average time relative to best)
        sector_avgs = {
            "Sector 1": (s1_times.mean(), s1_times.min()),
            "Sector 2": (s2_times.mean(), s2_times.min()),
            "Sector 3": (s3_times.mean(), s3_times.min())
        }
        
        sector_gaps = {
            sector: avg - best for sector, (avg, best) in sector_avgs.items()
        }
        weakest_sector = max(sector_gaps.items(), key=lambda x: x[1])
        
        if weakest_sector[1] > 0.3:
            insights.append({
                "type": "sector_weakness",
                "category": "Sector Performance",
                "title": f"{weakest_sector[0]} Needs Improvement",
                "description": f"{weakest_sector[0]} is {weakest_sector[1]:.2f}s slower than your best on average",
                "severity": "medium",
                "action": f"Analyze your best lap in {weakest_sector[0]} and replicate that approach"
            })
    
    # 4. Best Lap Analysis
    try:
        best_lap_idx = valid_laps['lap_time_seconds'].idxmin()
        best_lap = valid_laps.loc[best_lap_idx]
    except Exception as e:
        print(f"⚠️ Error finding best lap: {e}")
        best_lap = None
    
    # Compare best lap sectors to average
    if best_lap is not None and 'S1_SECONDS' in best_lap.index and 'S2_SECONDS' in best_lap.index and 'S3_SECONDS' in best_lap.index:
        if pd.notna(best_lap.get('S1_SECONDS')) and pd.notna(best_lap.get('S2_SECONDS')) and pd.notna(best_lap.get('S3_SECONDS')):
            avg_s1 = s1_times.mean() if len(s1_times) > 0 else 0
            avg_s2 = s2_times.mean() if len(s2_times) > 0 else 0
            avg_s3 = s3_times.mean() if len(s3_times) > 0 else 0
            
            s1_gap = avg_s1 - best_lap.get('S1_SECONDS', 0)
            s2_gap = avg_s2 - best_lap.get('S2_SECONDS', 0)
            s3_gap = avg_s3 - best_lap.get('S3_SECONDS', 0)
            
            if s1_gap > 0.5 or s2_gap > 0.5 or s3_gap > 0.5:
                largest_gap_sector = max([
                    ("Sector 1", s1_gap),
                    ("Sector 2", s2_gap),
                    ("Sector 3", s3_gap)
                ], key=lambda x: x[1])
                
                insights.append({
                    "type": "potential_improvement",
                    "category": "Performance Potential",
                    "title": f"Replicate {largest_gap_sector[0]} Performance",
                    "description": f"Your best {largest_gap_sector[0]} was {largest_gap_sector[1]:.2f}s faster than average - study that lap",
                    "severity": "low",
                    "action": f"Review telemetry from lap {int(best_lap.get('LAP_NUMBER', 0))} to understand what made {largest_gap_sector[0]} so fast"
                })
    
    # 5. Speed Analysis
    try:
        top_speeds = valid_laps['TOP_SPEED'].dropna() if 'TOP_SPEED' in valid_laps.columns else pd.Series(dtype=float)
    except Exception as e:
        print(f"⚠️ Error accessing top speeds: {e}")
        top_speeds = pd.Series(dtype=float)
    if len(top_speeds) > 0:
        avg_top_speed = top_speeds.mean()
        max_top_speed = top_speeds.max()
        
        if max_top_speed - avg_top_speed > 5:
            insights.append({
                "type": "speed_potential",
                "category": "Speed Analysis",
                "title": "Top Speed Potential",
                "description": f"Maximum speed ({max_top_speed:.1f} km/h) is {max_top_speed - avg_top_speed:.1f} km/h higher than average",
                "severity": "low",
                "action": "Identify conditions that allowed higher top speed and replicate them"
            })
    
    # 6. Generate AI Recommendations
    try:
        recommendations = generate_ai_recommendations(insights, vehicle_data, valid_laps)
    except Exception as e:
        print(f"⚠️ Error generating recommendations: {e}")
        recommendations = []
    
    # 7. Identify Patterns
    try:
        patterns = identify_driver_patterns(valid_laps)
    except Exception as e:
        print(f"⚠️ Error identifying patterns: {e}")
        patterns = []
    
    # 8. Enhance with Gemini AI (if available)
    try:
        from gemini_insights import enhance_insights_with_gemini
        enhanced_result = enhance_insights_with_gemini(
            vehicle_id,
            vehicle_data,
            valid_laps,
            insights,
            recommendations,
            patterns
        )
        
        # Use enhanced insights if Gemini was successful
        if enhanced_result.get("gemini_enhanced"):
            insights = enhanced_result["insights"]
            recommendations = enhanced_result["recommendations"]
            patterns = enhanced_result["patterns"]
    except ImportError:
        # Gemini module not available, use basic insights
        pass
    except Exception as e:
        # Gemini failed, use basic insights
        print(f"⚠️ Gemini enhancement failed, using basic insights: {e}")
    
    try:
        return {
            "vehicle_id": vehicle_id,
            "insights": insights,
            "recommendations": recommendations,
            "patterns": patterns,
            "summary": {
                "total_laps": len(valid_laps),
                "best_lap_time": float(valid_laps['lap_time_seconds'].min()) if len(valid_laps) > 0 else None,
                "average_lap_time": float(valid_laps['lap_time_seconds'].mean()) if len(valid_laps) > 0 else None,
                "consistency_score": float(consistency_score) if consistency_score is not None else None,
                "total_insights": len(insights),
                "priority_insights": [i for i in insights if i.get("severity") == "high"]
            }
        }
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"❌ Error building response for {vehicle_id}:")
        print(error_trace)
        raise HTTPException(status_code=500, detail=f"Error building AI insights response: {str(e)}")


def generate_ai_recommendations(insights: List[Dict], vehicle_data: pd.DataFrame, valid_laps: pd.DataFrame) -> List[Dict]:
    """
    Generate AI-powered recommendations based on insights
    """
    recommendations = []
    
    # Analyze insights to generate recommendations
    high_priority_insights = [i for i in insights if i.get("severity") == "high"]
    medium_priority_insights = [i for i in insights if i.get("severity") == "medium"]
    
    # Sector-specific recommendations
    s1_times = valid_laps['S1_SECONDS'].dropna()
    s2_times = valid_laps['S2_SECONDS'].dropna()
    s3_times = valid_laps['S3_SECONDS'].dropna()
    
    if len(s1_times) > 0 and len(s2_times) > 0 and len(s3_times) > 0:
        sector_performance = {
            "Sector 1": {
                "avg": s1_times.mean(),
                "best": s1_times.min(),
                "std": s1_times.std(),
                "improvement": s1_times.max() - s1_times.min()
            },
            "Sector 2": {
                "avg": s2_times.mean(),
                "best": s2_times.min(),
                "std": s2_times.std(),
                "improvement": s2_times.max() - s2_times.min()
            },
            "Sector 3": {
                "avg": s3_times.mean(),
                "best": s3_times.min(),
                "std": s3_times.std(),
                "improvement": s3_times.max() - s3_times.min()
            }
        }
        
        # Find sector with highest improvement potential
        max_improvement_sector = max(sector_performance.items(), key=lambda x: x[1]["improvement"])
        
        if max_improvement_sector[1]["improvement"] > 1.0:
            recommendations.append({
                "priority": "high",
                "category": "Sector Focus",
                "title": f"Focus Training on {max_improvement_sector[0]}",
                "description": f"{max_improvement_sector[0]} shows {max_improvement_sector[1]['improvement']:.2f}s variation between best and worst",
                "action_items": [
                    f"Review telemetry from your best {max_improvement_sector[0]} lap",
                    f"Compare braking points between best and worst {max_improvement_sector[0]} laps",
                    f"Practice consistent racing line in {max_improvement_sector[0]}",
                    f"Aim to match your best {max_improvement_sector[0]} time consistently"
                ],
                "expected_improvement": f"{max_improvement_sector[1]['improvement'] * 0.5:.2f}s potential gain"
            })
        
        # Find most inconsistent sector
        max_std_sector = max(sector_performance.items(), key=lambda x: x[1]["std"])
        
        if max_std_sector[1]["std"] > 0.5:
            recommendations.append({
                "priority": "medium",
                "category": "Consistency",
                "title": f"Improve Consistency in {max_std_sector[0]}",
                "description": f"{max_std_sector[0]} has {max_std_sector[1]['std']:.2f}s standard deviation - focus on consistency",
                "action_items": [
                    f"Identify reference points in {max_std_sector[0]}",
                    f"Practice maintaining same braking points",
                    f"Work on smooth throttle application",
                    f"Focus on consistent racing line"
                ],
                "expected_improvement": "More consistent lap times"
            })
    
    # Performance trend recommendations
    lap_times = valid_laps['lap_time_seconds'].values
    if len(lap_times) > 5:
        first_third = np.mean(lap_times[:len(lap_times)//3])
        last_third = np.mean(lap_times[-len(lap_times)//3:])
        
        if last_third > first_third + 1.0:
            recommendations.append({
                "priority": "high",
                "category": "Endurance",
                "title": "Address Performance Degradation",
                "description": "Lap times slow significantly toward the end of the race",
                "action_items": [
                    "Review tire management strategy",
                    "Consider physical conditioning",
                    "Analyze fuel load impact",
                    "Check for mechanical issues",
                    "Practice maintaining pace throughout race"
                ],
                "expected_improvement": "Better race pace consistency"
            })
    
    # Consistency recommendations
    consistency_insights = [i for i in insights if "consistency" in i.get("category", "").lower()]
    if consistency_insights:
        recommendations.append({
            "priority": "medium",
            "category": "Consistency",
            "title": "Improve Overall Consistency",
            "description": "Focus on reducing lap time variation",
            "action_items": [
                "Practice same racing line repeatedly",
                "Use consistent braking markers",
                "Maintain consistent throttle application",
                "Work on smooth steering inputs",
                "Practice in various track conditions"
            ],
            "expected_improvement": "More predictable race pace"
        })
    
    return recommendations


def identify_driver_patterns(valid_laps: pd.DataFrame) -> List[Dict]:
    """
    Identify patterns in driver behavior
    """
    patterns = []
    
    # Pattern 1: Lap time progression
    lap_times = valid_laps['lap_time_seconds'].values
    if len(lap_times) > 10:
        # Check for warm-up pattern
        first_5_avg = np.mean(lap_times[:5])
        next_5_avg = np.mean(lap_times[5:10])
        
        if first_5_avg > next_5_avg + 1.0:
            patterns.append({
                "type": "warm_up_pattern",
                "description": "Driver shows warm-up pattern - first 5 laps slower than next 5",
                "implication": "May benefit from better warm-up routine or more aggressive early pace"
            })
    
    # Pattern 2: Sector balance
    s1_times = valid_laps['S1_SECONDS'].dropna()
    s2_times = valid_laps['S2_SECONDS'].dropna()
    s3_times = valid_laps['S3_SECONDS'].dropna()
    
    if len(s1_times) > 0 and len(s2_times) > 0 and len(s3_times) > 0:
        s1_avg = s1_times.mean()
        s2_avg = s2_times.mean()
        s3_avg = s3_times.mean()
        
        # Check for sector balance
        sector_avgs = [s1_avg, s2_avg, s3_avg]
        max_sector = max(sector_avgs)
        min_sector = min(sector_avgs)
        
        if max_sector - min_sector > 2.0:
            patterns.append({
                "type": "sector_imbalance",
                "description": f"Significant imbalance between sectors ({max_sector - min_sector:.2f}s difference)",
                "implication": "Driver may be stronger in some sectors than others - focus on weaker sectors"
            })
    
    # Pattern 3: Consistency pattern
    lap_time_std = np.std(lap_times)
    if lap_time_std < 0.5:
        patterns.append({
            "type": "high_consistency",
            "description": "Very consistent lap times throughout race",
            "implication": "Driver has good consistency - can focus on pushing for faster times"
        })
    elif lap_time_std > 2.0:
        patterns.append({
            "type": "low_consistency",
            "description": "High variation in lap times",
            "implication": "Driver needs to focus on consistency before pushing for speed"
        })
    
    return patterns


async def get_ai_driver_insights(vehicle_id: str) -> Dict:
    """
    Main function to get AI-powered driver insights
    """
    try:
        return analyze_driver_patterns(vehicle_id)
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"❌ Error in get_ai_driver_insights for {vehicle_id}:")
        print(error_trace)
        raise HTTPException(status_code=500, detail=f"Error generating AI insights: {str(e)}")


async def get_sector_ai_analysis(vehicle_id: str, sector: str) -> Dict:
    """
    AI-powered analysis for a specific sector
    """
    project_root = get_project_root()
    endurance_file = project_root / "logs" / "R1_section_endurance.csv"
    
    if not endurance_file.exists():
        raise HTTPException(status_code=404, detail="Endurance data not found")
    
    endurance_df = pd.read_csv(endurance_file, sep=";", low_memory=False)
    endurance_df.columns = endurance_df.columns.str.strip()
    
    vehicle_data = endurance_df[endurance_df['NUMBER'].astype(str) == str(vehicle_id)]
    
    if len(vehicle_data) == 0:
        raise HTTPException(status_code=404, detail=f"Vehicle {vehicle_id} not found")
    
    sector_col = f'{sector.upper()}_SECONDS'
    if sector_col not in vehicle_data.columns:
        raise HTTPException(status_code=400, detail=f"Invalid sector: {sector}")
    
    sector_times = vehicle_data[sector_col].dropna()
    
    if len(sector_times) == 0:
        return {"error": "No sector data available"}
    
    best_time = sector_times.min()
    worst_time = sector_times.max()
    avg_time = sector_times.mean()
    std_time = sector_times.std()
    
    # Find best and worst laps
    best_lap = vehicle_data.loc[sector_times.idxmin()]
    worst_lap = vehicle_data.loc[sector_times.idxmax()]
    
    analysis = {
        "vehicle_id": vehicle_id,
        "sector": sector,
        "statistics": {
            "best_time": float(best_time),
            "worst_time": float(worst_time),
            "average_time": float(avg_time),
            "std_deviation": float(std_time),
            "improvement_potential": float(worst_time - best_time),
            "consistency_score": float(std_time / avg_time * 100)
        },
        "best_lap": {
            "lap_number": int(best_lap['LAP_NUMBER']),
            "sector_time": float(best_lap[sector_col]),
            "lap_time": str(best_lap['LAP_TIME']) if pd.notna(best_lap['LAP_TIME']) else None
        },
        "worst_lap": {
            "lap_number": int(worst_lap['LAP_NUMBER']),
            "sector_time": float(worst_lap[sector_col]),
            "lap_time": str(worst_lap['LAP_TIME']) if pd.notna(worst_lap['LAP_TIME']) else None
        },
        "ai_insights": []
    }
    
    # Generate AI insights
    if std_time > 0.5:
        analysis["ai_insights"].append({
            "type": "consistency_issue",
            "message": f"{sector} shows {std_time:.2f}s standard deviation - focus on consistency",
            "recommendation": "Practice maintaining same racing line and braking points"
        })
    
    if worst_time - best_time > 1.0:
        analysis["ai_insights"].append({
            "type": "improvement_opportunity",
            "message": f"Potential {worst_time - best_time:.2f}s improvement by matching best {sector} time",
            "recommendation": f"Review telemetry from lap {int(best_lap['LAP_NUMBER'])} to understand what made it fast"
        })
    
    return analysis


