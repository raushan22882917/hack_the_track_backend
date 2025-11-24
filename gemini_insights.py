"""
Gemini AI Integration for Enhanced Driver Insights
Uses Google's Gemini API to generate intelligent, contextual recommendations
"""

import json
from typing import Dict, List, Optional
from pathlib import Path
import pandas as pd
from fastapi import HTTPException

try:
    import google.generativeai as genai  # type: ignore
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None  # type: ignore
    print("⚠️ google-generativeai not installed. Install with: pip install google-generativeai")


def get_gemini_api_key() -> Optional[str]:
    """Get Gemini API key directly from .env file"""
    # Try to find .env file in project root
    env_path = Path(__file__).parent / '.env'
    
    # If not found, try current directory
    if not env_path.exists():
        env_path = Path('.env')
    
    api_key = None
    
    # Read directly from .env file
    if env_path.exists():
        try:
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    # Remove leading/trailing whitespace
                    line = line.strip()
                    # Skip comments and empty lines
                    if not line or line.startswith('#'):
                        continue
                    
                    # Try multiple possible variable names
                    # Handle both KEY=value and KEY = value formats
                    for var_name in ['GEMINI_API_KEY', 'GOOGLE_API_KEY', 'VITE_GEMINI_API_KEY']:
                        if line.startswith(var_name):
                            # Split on = and take everything after it
                            parts = line.split('=', 1)
                            if len(parts) == 2:
                                api_key = parts[1].strip()
                                # Remove quotes if present
                                if api_key.startswith('"') and api_key.endswith('"'):
                                    api_key = api_key[1:-1]
                                elif api_key.startswith("'") and api_key.endswith("'"):
                                    api_key = api_key[1:-1]
                                # Only break if we found a non-empty value
                                if api_key:
                                    break
                    if api_key:
                        break
        except Exception as e:
            print(f"⚠️ Error reading .env file: {e}")
            import traceback
            traceback.print_exc()
    
    if not api_key:
        print("⚠️ GEMINI_API_KEY not found in .env file")
        print(f"   Looking for .env file at: {env_path}")
        print("   Set it in .env file: GEMINI_API_KEY=your-api-key")
    else:
        print(f"✅ Found Gemini API key from .env file (length: {len(api_key)})")
    
    return api_key


def initialize_gemini():
    """Initialize Gemini AI client"""
    if not GEMINI_AVAILABLE:
        print("⚠️ Gemini not available: google-generativeai package not installed")
        return None
    
    api_key = get_gemini_api_key()
    if not api_key:
        print("⚠️ Gemini not available: API key not found")
        return None
    
    try:
        genai.configure(api_key=api_key)
        # Use gemini-2.5-flash for faster responses (updated model name)
        # Fallback to gemini-flash-latest if 2.5-flash not available
        try:
            model = genai.GenerativeModel('gemini-2.5-flash')
        except:
            model = genai.GenerativeModel('gemini-flash-latest')
        print("✅ Gemini AI model initialized successfully")
        return model
    except Exception as e:
        print(f"⚠️ Failed to initialize Gemini: {e}")
        import traceback
        traceback.print_exc()
        return None


def format_driver_data_for_gemini(vehicle_data: pd.DataFrame, valid_laps: pd.DataFrame) -> str:
    """Format driver performance data for Gemini prompt"""
    if len(valid_laps) == 0:
        return "No valid lap data available"
    
    lap_times = valid_laps['lap_time_seconds'].values
    best_lap = valid_laps.loc[valid_laps['lap_time_seconds'].idxmin()]
    worst_lap = valid_laps.loc[valid_laps['lap_time_seconds'].idxmax()]
    
    # Sector analysis
    s1_times = valid_laps['S1_SECONDS'].dropna()
    s2_times = valid_laps['S2_SECONDS'].dropna()
    s3_times = valid_laps['S3_SECONDS'].dropna()
    
    data_summary = f"""
Driver Performance Data:
- Total Laps: {len(valid_laps)}
- Best Lap Time: {best_lap['lap_time_seconds']:.2f}s (Lap {int(best_lap['LAP_NUMBER'])})
- Worst Lap Time: {worst_lap['lap_time_seconds']:.2f}s (Lap {int(worst_lap['LAP_NUMBER'])})
- Average Lap Time: {lap_times.mean():.2f}s
- Consistency (std dev): {lap_times.std():.2f}s
- Time Spread: {lap_times.max() - lap_times.min():.2f}s

Sector Performance:
- Sector 1: Best {s1_times.min():.2f}s, Avg {s1_times.mean():.2f}s, Std {s1_times.std():.2f}s
- Sector 2: Best {s2_times.min():.2f}s, Avg {s2_times.mean():.2f}s, Std {s2_times.std():.2f}s
- Sector 3: Best {s3_times.min():.2f}s, Avg {s3_times.mean():.2f}s, Std {s3_times.std():.2f}s

Best Lap Details:
- Lap {int(best_lap['LAP_NUMBER'])}: {best_lap['lap_time_seconds']:.2f}s
  S1: {best_lap['S1_SECONDS']:.2f}s, S2: {best_lap['S2_SECONDS']:.2f}s, S3: {best_lap['S3_SECONDS']:.2f}s
  Top Speed: {best_lap['TOP_SPEED']:.1f} km/h

Worst Lap Details:
- Lap {int(worst_lap['LAP_NUMBER'])}: {worst_lap['lap_time_seconds']:.2f}s
  S1: {worst_lap['S1_SECONDS']:.2f}s, S2: {worst_lap['S2_SECONDS']:.2f}s, S3: {worst_lap['S3_SECONDS']:.2f}s
  Top Speed: {worst_lap['TOP_SPEED']:.1f} km/h

Improvement Potential:
- Sector 1: {s1_times.max() - s1_times.min():.2f}s variation
- Sector 2: {s2_times.max() - s2_times.min():.2f}s variation
- Sector 3: {s3_times.max() - s3_times.min():.2f}s variation
- Total potential: {worst_lap['lap_time_seconds'] - best_lap['lap_time_seconds']:.2f}s
"""
    
    return data_summary


def generate_gemini_insights(
    vehicle_id: str,
    vehicle_data: pd.DataFrame,
    valid_laps: pd.DataFrame,
    existing_insights: List[Dict]
) -> Dict:
    """
    Use Gemini AI to generate enhanced insights and recommendations
    """
    model = initialize_gemini()
    
    if not model:
        # Fallback to basic insights if Gemini is not available
        return {
            "enhanced": False,
            "message": "Gemini AI not available. Using basic insights."
        }
    
    try:
        # Format data for Gemini
        data_summary = format_driver_data_for_gemini(vehicle_data, valid_laps)
        
        # Create prompt for Gemini
        prompt = f"""You are an expert racing coach analyzing telemetry data for a professional driver.

{data_summary}

Based on this performance data, provide:
1. 3-5 key insights about the driver's performance (focus on patterns, consistency, and improvement areas)
2. Specific, actionable recommendations prioritized by impact
3. Identify any concerning patterns or trends

Format your response as JSON with this structure:
{{
  "insights": [
    {{
      "title": "Brief insight title",
      "description": "Detailed explanation",
      "severity": "high|medium|low",
      "category": "Performance Trend|Consistency|Sector Performance|etc",
      "action": "Specific actionable recommendation"
    }}
  ],
  "recommendations": [
    {{
      "priority": "high|medium|low",
      "title": "Recommendation title",
      "description": "Detailed description",
      "action_items": ["Action 1", "Action 2"],
      "expected_improvement": "Expected time gain or improvement"
    }}
  ],
  "patterns": [
    {{
      "description": "Pattern description",
      "implication": "What this means for performance",
      "type": "trend|degradation|consistency|etc"
    }}
  ]
}}

Be specific with numbers, focus on actionable advice, and prioritize by impact on lap times.
Return ONLY valid JSON, no additional text."""

        # Generate response
        response = model.generate_content(prompt)
        response_text = ""
        
        # Safely get response text
        if hasattr(response, 'text'):
            response_text = response.text.strip()
        else:
            response_text = str(response).strip()
        
        # Try to extract JSON from response (sometimes Gemini adds markdown formatting)
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        # Parse JSON response
        gemini_data = json.loads(response_text)
        
        return {
            "enhanced": True,
            "gemini_insights": gemini_data.get("insights", []),
            "gemini_recommendations": gemini_data.get("recommendations", []),
            "gemini_patterns": gemini_data.get("patterns", [])
        }
        
    except json.JSONDecodeError as e:
        response_preview = response_text[:500] if response_text else "No response text available"
        print(f"⚠️ Failed to parse Gemini JSON response: {e}")
        print(f"Response was: {response_preview}")
        return {
            "enhanced": False,
            "error": "Failed to parse Gemini response",
            "raw_response": response_preview
        }
    except Exception as e:
        error_str = str(e)
        print(f"⚠️ Gemini API error: {error_str}")
        
        # Check for specific API errors
        if "API_KEY_SERVICE_BLOCKED" in error_str or "403" in error_str:
            return {
                "enhanced": False,
                "error": "API key blocked or Generative Language API not enabled",
                "message": "Gemini API key is blocked. Please enable 'Generative Language API' in Google Cloud Console.",
                "help": "Visit https://console.cloud.google.com/apis/library/generativelanguage.googleapis.com"
            }
        elif "API_KEY_INVALID" in error_str or "401" in error_str:
            return {
                "enhanced": False,
                "error": "Invalid API key",
                "message": "Gemini API key is invalid. Please check your API key.",
                "help": "Get a valid API key from https://makersuite.google.com/app/apikey"
            }
        
        return {
            "enhanced": False,
            "error": error_str,
            "message": f"Gemini API error: {error_str[:200]}"
        }


def enhance_insights_with_gemini(
    vehicle_id: str,
    vehicle_data: pd.DataFrame,
    valid_laps: pd.DataFrame,
    existing_insights: List[Dict],
    existing_recommendations: List[Dict],
    existing_patterns: List[Dict]
) -> Dict:
    """
    Enhance existing insights with Gemini AI analysis
    Merges AI-generated insights with existing rule-based insights
    """
    gemini_result = generate_gemini_insights(vehicle_id, vehicle_data, valid_laps, existing_insights)
    
    if not gemini_result.get("enhanced"):
        # Return existing insights if Gemini is not available
        return {
            "insights": existing_insights,
            "recommendations": existing_recommendations,
            "patterns": existing_patterns,
            "gemini_enhanced": False
        }
    
    # Merge Gemini insights with existing ones
    # Prioritize Gemini insights but keep existing ones for completeness
    merged_insights = existing_insights.copy()
    gemini_insights = gemini_result.get("gemini_insights", [])
    
    # Add Gemini insights, avoiding duplicates
    for gemini_insight in gemini_insights:
        # Check if similar insight already exists
        is_duplicate = any(
            existing.get("title", "").lower() == gemini_insight.get("title", "").lower()
            for existing in merged_insights
        )
        if not is_duplicate:
            merged_insights.append(gemini_insight)
    
    # Merge recommendations (prioritize Gemini)
    merged_recommendations = gemini_result.get("gemini_recommendations", []) + existing_recommendations
    
    # Merge patterns
    merged_patterns = existing_patterns.copy()
    gemini_patterns = gemini_result.get("gemini_patterns", [])
    for gemini_pattern in gemini_patterns:
        is_duplicate = any(
            existing.get("description", "").lower() == gemini_pattern.get("description", "").lower()
            for existing in merged_patterns
        )
        if not is_duplicate:
            merged_patterns.append(gemini_pattern)
    
    return {
        "insights": merged_insights,
        "recommendations": merged_recommendations[:10],  # Limit to top 10
        "patterns": merged_patterns,
        "gemini_enhanced": True
    }


def generate_race_story_insights(race_story_data: Dict) -> Dict:
    """
    Generate AI-powered race story insights using Gemini
    Analyzes position changes, key moments, and strategic decisions
    """
    model = initialize_gemini()
    
    if not model:
        return {
            "enhanced": False,
            "message": "Gemini AI not available. Using basic race story."
        }
    
    try:
        # Format race story data for Gemini
        story_summary = f"""
Race Story Data:
- Total Position Changes: {len(race_story_data.get('position_changes', []))}
- Key Moments: {len(race_story_data.get('key_moments', []))}
- Sector Improvements: {len(race_story_data.get('sector_improvements', []))}

Statistics:
{json.dumps(race_story_data.get('statistics', {}), indent=2)}

Key Moments Summary:
"""
        for moment in race_story_data.get('key_moments', [])[:10]:
            story_summary += f"- {moment.get('type', 'Unknown')} on Lap {moment.get('lap', 'N/A')} by Vehicle {moment.get('vehicle_id', 'N/A')}\n"
        
        prompt = f"""You are an expert race analyst. Analyze this race data and provide:

1. A compelling narrative of the race story (2-3 paragraphs)
2. Key strategic decisions that impacted the outcome
3. Critical moments that changed the race
4. Performance trends and patterns
5. What-if scenarios and missed opportunities

Race Data:
{story_summary}

Format your response as JSON:
{{
  "narrative": "A compelling 2-3 paragraph story of the race",
  "strategic_decisions": [
    {{
      "moment": "Description of the strategic moment",
      "impact": "How it affected the race",
      "vehicle_id": "Vehicle involved",
      "lap": 5
    }}
  ],
  "critical_moments": [
    {{
      "description": "What happened",
      "significance": "Why it mattered",
      "lap": 10,
      "type": "pit_stop|fastest_lap|position_change|etc"
    }}
  ],
  "trends": [
    {{
      "pattern": "Pattern description",
      "implication": "What it means",
      "vehicles_affected": ["vehicle_id1", "vehicle_id2"]
    }}
  ],
  "missed_opportunities": [
    {{
      "scenario": "What could have happened",
      "potential_impact": "How it would have changed things",
      "vehicle_id": "vehicle_id"
    }}
  ]
}}

Return ONLY valid JSON, no additional text."""

        response = model.generate_content(prompt)
        response_text = ""
        
        # Safely get response text
        if hasattr(response, 'text'):
            response_text = response.text.strip()
        else:
            response_text = str(response).strip()
        
        # Extract JSON
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        gemini_data = json.loads(response_text)
        
        return {
            "enhanced": True,
            "narrative": gemini_data.get("narrative", ""),
            "strategic_decisions": gemini_data.get("strategic_decisions", []),
            "critical_moments": gemini_data.get("critical_moments", []),
            "trends": gemini_data.get("trends", []),
            "missed_opportunities": gemini_data.get("missed_opportunities", [])
        }
        
    except json.JSONDecodeError as e:
        response_preview = response_text[:500] if response_text else "No response text available"
        print(f"⚠️ Failed to parse Gemini JSON response: {e}")
        print(f"Response was: {response_preview}")
        return {
            "enhanced": False,
            "error": "Failed to parse Gemini response",
            "raw_response": response_preview
        }
    except Exception as e:
        error_str = str(e)
        print(f"⚠️ Gemini race story error: {error_str}")
        
        # Check for specific API errors
        if "API_KEY_SERVICE_BLOCKED" in error_str or "403" in error_str:
            return {
                "enhanced": False,
                "error": "API key blocked or Generative Language API not enabled",
                "message": "Gemini API key is blocked. Please enable 'Generative Language API' in Google Cloud Console.",
                "help": "Visit https://console.cloud.google.com/apis/library/generativelanguage.googleapis.com"
            }
        elif "API_KEY_INVALID" in error_str or "401" in error_str:
            return {
                "enhanced": False,
                "error": "Invalid API key",
                "message": "Gemini API key is invalid. Please check your API key.",
                "help": "Get a valid API key from https://makersuite.google.com/app/apikey"
            }
        
        return {
            "enhanced": False,
            "error": error_str,
            "message": f"Gemini API error: {error_str[:200]}"
        }


def generate_realtime_strategy_insights(
    vehicle_id: str,
    gaps_data: Dict,
    pit_window_data: Dict,
    strategy_data: Dict,
    tire_degradation_data: Dict
) -> Dict:
    """
    Generate AI-powered real-time strategy insights using Gemini
    Provides race engineer-style decision support
    """
    model = initialize_gemini()
    
    if not model:
        return {
            "enhanced": False,
            "message": "Gemini AI not available."
        }
    
    try:
        # Format real-time data for Gemini
        # Safely get gap data
        gap_info = gaps_data.get('gaps', [])
        vehicle_gap = gap_info[0] if gap_info else {}
        
        strategy_summary = f"""
Real-Time Race Data for Vehicle {vehicle_id}:

Gap Analysis:
- Position: {vehicle_gap.get('position', 'N/A')}
- Gap to Leader: {vehicle_gap.get('gap_to_leader_seconds', 'N/A')}s

Pit Window Analysis:
- Current Lap: {pit_window_data.get('current_lap', 'N/A')}
- Remaining Laps: {pit_window_data.get('remaining_laps', 'N/A')}
- Tire Degradation: {pit_window_data.get('tire_degradation_per_lap_seconds', 'N/A')}s per lap
- Recommended Strategy: {pit_window_data.get('recommended_strategy', {}).get('strategy', 'N/A') if pit_window_data.get('recommended_strategy') else 'N/A'}

Strategy Simulation:
- Time Difference: {strategy_data.get('comparison', {}).get('time_difference_seconds', 'N/A')}s
- Recommendation: {strategy_data.get('comparison', {}).get('recommendation', 'N/A')}

Tire Condition:
- Condition: {tire_degradation_data.get('current_condition', {}).get('tire_condition', 'N/A')}
- Degradation Rate: {tire_degradation_data.get('degradation_analysis', {}).get('degradation_per_lap_seconds', 'N/A')}s per lap
"""

        prompt = f"""You are a race engineer making real-time strategic decisions. Analyze this data and provide:

1. Immediate action recommendations (what to do right now)
2. Pit stop timing analysis (when is the perfect window)
3. Risk assessment (what could go wrong)
4. Alternative strategies (what-if scenarios)
5. Key factors to monitor

Real-Time Data:
{strategy_summary}

Format your response as JSON:
{{
  "immediate_actions": [
    {{
      "action": "What to do now",
      "priority": "high|medium|low",
      "reasoning": "Why this action",
      "expected_impact": "What will happen"
    }}
  ],
  "pit_stop_analysis": {{
    "optimal_window": "Lap range for pit stop",
    "reasoning": "Why this window",
    "risks": ["Risk 1", "Risk 2"],
    "benefits": ["Benefit 1", "Benefit 2"]
  }},
  "risk_assessment": [
    {{
      "risk": "Risk description",
      "probability": "high|medium|low",
      "impact": "high|medium|low",
      "mitigation": "How to mitigate"
    }}
  ],
  "alternative_strategies": [
    {{
      "strategy": "Alternative approach",
      "scenario": "When to use",
      "pros": ["Pro 1", "Pro 2"],
      "cons": ["Con 1", "Con 2"]
    }}
  ],
  "monitoring_points": [
    {{
      "metric": "What to watch",
      "threshold": "When to act",
      "action": "What to do if threshold reached"
    }}
  ]
}}

Return ONLY valid JSON, no additional text."""

        response = model.generate_content(prompt)
        response_text = ""
        
        # Safely get response text
        if hasattr(response, 'text'):
            response_text = response.text.strip()
        else:
            response_text = str(response).strip()
        
        # Extract JSON
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        gemini_data = json.loads(response_text)
        
        return {
            "enhanced": True,
            "immediate_actions": gemini_data.get("immediate_actions", []),
            "pit_stop_analysis": gemini_data.get("pit_stop_analysis", {}),
            "risk_assessment": gemini_data.get("risk_assessment", []),
            "alternative_strategies": gemini_data.get("alternative_strategies", []),
            "monitoring_points": gemini_data.get("monitoring_points", [])
        }
        
    except json.JSONDecodeError as e:
        response_preview = response_text[:500] if response_text else "No response text available"
        print(f"⚠️ Failed to parse Gemini JSON response: {e}")
        print(f"Response was: {response_preview}")
        return {
            "enhanced": False,
            "error": "Failed to parse Gemini response",
            "raw_response": response_preview
        }
    except Exception as e:
        error_str = str(e)
        print(f"⚠️ Gemini real-time strategy error: {error_str}")
        
        # Check for specific API errors
        if "API_KEY_SERVICE_BLOCKED" in error_str or "403" in error_str:
            return {
                "enhanced": False,
                "error": "API key blocked or Generative Language API not enabled",
                "message": "Gemini API key is blocked. Please enable 'Generative Language API' in Google Cloud Console and ensure your API key has proper permissions.",
                "help": "Visit https://console.cloud.google.com/apis/library/generativelanguage.googleapis.com to enable the API"
            }
        elif "API_KEY_INVALID" in error_str or "401" in error_str:
            return {
                "enhanced": False,
                "error": "Invalid API key",
                "message": "Gemini API key is invalid. Please check your API key in the .env file.",
                "help": "Get a valid API key from https://makersuite.google.com/app/apikey"
            }
        
        return {
            "enhanced": False,
            "error": error_str,
            "message": f"Gemini API error: {error_str[:200]}"
        }

