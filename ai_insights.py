#!/usr/bin/env python3
"""
AI Insights Module for Margin Impact Simulator
Provides trend analysis, forecasting, and predictive analytics using AI
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import requests
import json
import re
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def _get_google_ai_api_key() -> str:
    """
    Get Google AI API key from environment or Streamlit secrets.
    """
    # Check environment variable first
    api_key = os.getenv('GOOGLE_AI_API_KEY')
    if api_key:
        return api_key
    
    # Try Streamlit secrets (only in cloud environments to avoid local warnings)
    try:
        import streamlit as st
        # Check if we're in a cloud environment
        is_cloud_env = (
            'STREAMLIT_CLOUD' in os.environ or 
            'STREAMLIT_SHARING' in os.environ or
            hasattr(st, 'secrets') and hasattr(st.secrets, '_secrets')
        )
        
        if is_cloud_env:
            # Try multiple access patterns for the API key
            try:
                # Try direct access first
                api_key = st.secrets.get("GOOGLE_AI_API_KEY")
                if api_key:
                    return api_key
            except:
                pass
            
            try:
                # Try nested access (secrets section format)
                api_key = st.secrets.get("secrets", {}).get("GOOGLE_AI_API_KEY")
                if api_key:
                    return api_key
            except:
                pass
            
            try:
                # Try direct dictionary access
                api_key = st.secrets["GOOGLE_AI_API_KEY"]
                if api_key:
                    return api_key
            except:
                pass
            
            try:
                # Try nested dictionary access
                api_key = st.secrets["secrets"]["GOOGLE_AI_API_KEY"]
                if api_key:
                    return api_key
            except:
                pass
    except:
        pass
    
    return None

def _clean_ai_text(text: str) -> str:
    """Clean up AI-generated text for better formatting."""
    # Remove excessive line breaks and normalize whitespace
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Fix common formatting issues
    text = re.sub(r'(\d+)\s*,\s*(\d+)', r'\1,\2', text)  # Fix comma spacing in numbers
    text = re.sub(r'(\d+)\s*\(\s*(\d+)', r'\1 (\2', text)  # Fix parentheses spacing
    text = re.sub(r'(\d+)\s*\.\s*(\d+)', r'\1.\2', text)  # Fix decimal spacing
    
    # Fix spacing around % and $ symbols
    text = re.sub(r'(\d+)\s*%', r'\1%', text)  # Remove space before % in numbers
    text = re.sub(r'%\s*([a-zA-Z])', r'% \1', text)  # Add space after % before letters
    text = re.sub(r'([a-zA-Z])\s*\$', r'\1 $', text)  # Add space before $ after letters
    text = re.sub(r'\$\s*(\d+)', r'$\1', text)  # Remove space after $ before numbers
    
    # Remove bullet points and list formatting
    text = re.sub(r'^\s*[-*•]\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s*', '', text, flags=re.MULTILINE)
    
    # Clean up any remaining formatting artifacts
    text = re.sub(r'\s*:\s*', ': ', text)
    
    # Ensure proper sentence structure
    text = text.strip()
    
    # Remove trailing periods and add just one
    text = re.sub(r'\.+$', '', text)
    if text and not text.endswith(('.', '!', '?')):
        text += '.'
    
    return text

def generate_trend_analysis(historical_data: pd.DataFrame, current_metrics: Dict[str, Any]) -> str:
    """
    Generate AI-powered trend analysis using Ollama, Google AI, or intelligent fallback.
    
    Args:
        historical_data: DataFrame with historical scenarios
        current_metrics: Current scenario metrics
    
    Returns:
        AI-generated trend analysis narrative
    """
    # Try Ollama first (local AI)
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=2)
        if response.status_code == 200:
            print("🔍 DEBUG: Using Ollama for trend analysis")
            ai_narrative = _generate_ollama_trend_analysis(historical_data, current_metrics)
            if ai_narrative:
                return ai_narrative
    except Exception as e:
        print(f"🔍 DEBUG: Ollama not available for trend analysis ({e})")

    # Try Google AI (cloud AI)
    ai_narrative = _generate_google_ai_trend_analysis(historical_data, current_metrics)
    if ai_narrative:
        return ai_narrative

    # Fallback to intelligent templates
    return _generate_template_trend_analysis(historical_data, current_metrics)

def _generate_ollama_trend_analysis(historical_data: pd.DataFrame, current_metrics: Dict[str, Any]) -> Optional[str]:
    """
    Generate trend analysis using Ollama (local LLM).
    """

    # Calculate trend statistics
    avg_margin = historical_data['margin'].mean()
    avg_coverage = historical_data['coverage'].mean()
    margin_trend = "increasing" if historical_data['margin'].iloc[-1] > historical_data['margin'].iloc[0] else "decreasing"
    coverage_trend = "increasing" if historical_data['coverage'].iloc[-1] > historical_data['coverage'].iloc[0] else "decreasing"
    
    # Prepare the prompt
    prompt = f"""As a senior business analyst, provide a comprehensive trend analysis of this connectivity service performance:

Historical Performance:
- Average margin: {avg_margin:.1%} across {len(historical_data)} scenarios
- Average coverage: {avg_coverage:.1f}%
- Margin trend: {margin_trend}
- Coverage trend: {coverage_trend}
- Current margin: {current_metrics.get('margin', 0):.1%}
- Current coverage: {current_metrics.get('coverage', 0):.1f}%

Analyze the business implications, identify key patterns, and provide strategic recommendations. Focus on operational insights and actionable business intelligence.
Keep it under 150 words. Use business terminology - refer to 'lines', 'customers', 'service coverage', NOT 'students' or 'educational'.
IMPORTANT: Write as a single paragraph without line breaks or bullet points."""

    try:
        # Use Ollama API - try common model names
        models_to_try = ['llama3.2', 'llama3.1', 'llama2', 'mistral', 'codellama']

        for model_name in models_to_try:
            data = {
                'model': model_name,
                'prompt': f"You are an expert business analyst providing strategic insights. Be analytical and professional. Write in a single paragraph without line breaks or bullet points.\n\n{prompt}",
                'stream': False,
                'options': {
                    'temperature': 0.7,
                    'num_predict': 200,
                    'stop': ['\n\n', '**', '##', '###']  # Stop at formatting markers
                }
            }

            try:
                response = requests.post(
                    'http://localhost:11434/api/generate',
                    json=data,
                    timeout=30
                )

                if response.status_code == 200:
                    result = response.json()
                    narrative = result['response'].strip()

                    # Clean up the narrative formatting
                    narrative = _clean_ai_text(narrative)

                    print(f"🔍 DEBUG: Ollama trend analysis success with {model_name} - generated {len(narrative)} character narrative")
                    return narrative
                elif response.status_code == 404:
                    print(f"🔍 DEBUG: Model {model_name} not found for trend analysis, trying next...")
                    continue
                else:
                    print(f"🔍 DEBUG: Ollama API error with {model_name} for trend analysis: {response.status_code}")
                    continue
            except Exception as e:
                print(f"🔍 DEBUG: Ollama API error with {model_name} for trend analysis: {e}")
                continue

        print("🔍 DEBUG: No available models found in Ollama for trend analysis")
        return None

    except Exception as e:
        print(f"🔍 DEBUG: Ollama API error for trend analysis: {e}")
        return None

def _generate_google_ai_trend_analysis(historical_data: pd.DataFrame, current_metrics: Dict[str, Any]) -> Optional[str]:
    """
    Generate trend analysis using Google AI (Gemini API).
    """
    
    # Get API key using the helper function
    api_key = _get_google_ai_api_key()
    if not api_key:
        print("🔍 DEBUG: Google AI API key not found for trend analysis - using fallback templates")
        return None
    
    print("🔍 DEBUG: Using Google AI for trend analysis")
    
    # Calculate trend statistics
    avg_margin = historical_data['margin'].mean()
    avg_coverage = historical_data['coverage'].mean()
    margin_trend = "increasing" if historical_data['margin'].iloc[-1] > historical_data['margin'].iloc[0] else "decreasing"
    coverage_trend = "increasing" if historical_data['coverage'].iloc[-1] > historical_data['coverage'].iloc[0] else "decreasing"
    
    # Prepare the prompt
    prompt = f"""As a senior business analyst, provide a comprehensive trend analysis of this connectivity service performance:

Historical Performance:
- Average margin: {avg_margin:.1%} across {len(historical_data)} scenarios
- Average coverage: {avg_coverage:.1f}%
- Margin trend: {margin_trend}
- Coverage trend: {coverage_trend}
- Current margin: {current_metrics.get('margin', 0):.1%}
- Current coverage: {current_metrics.get('coverage', 0):.1f}%

Analyze the business implications, identify key patterns, and provide strategic recommendations. Focus on operational insights and actionable business intelligence.
Keep it under 150 words. Use business terminology - refer to 'lines', 'customers', 'service coverage', NOT 'students' or 'educational'.
IMPORTANT: Write as a single paragraph without line breaks or bullet points."""

    try:
        # Use Google AI Gemini API
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
        
        headers = {
            'Content-Type': 'application/json',
        }
        
        data = {
            "contents": [{
                "parts": [{
                    "text": f"You are an expert business analyst providing strategic insights. Be analytical and professional. Write in a single paragraph without line breaks or bullet points.\n\n{prompt}"
                }]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 250,
                "stopSequences": ["\n\n", "**", "##", "###"]
            }
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            
            # Extract the generated text from Google AI response
            if 'candidates' in result and len(result['candidates']) > 0:
                candidate = result['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content']:
                    narrative = candidate['content']['parts'][0]['text'].strip()
                    
                    # Clean up the narrative formatting
                    narrative = _clean_ai_text(narrative)
                    
                    print(f"🔍 DEBUG: Google AI trend analysis success - generated {len(narrative)} character narrative")
                    return narrative
            
            print("🔍 DEBUG: Google AI trend analysis response format unexpected")
            return None
        else:
            print(f"🔍 DEBUG: Google AI trend analysis API error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"🔍 DEBUG: Google AI trend analysis API error: {e}")
        return None

def _generate_template_trend_analysis(historical_data: pd.DataFrame, current_metrics: Dict[str, Any]) -> str:
    """Generate template-based trend analysis when AI is not available."""
    
    avg_margin = historical_data['margin'].mean()
    avg_coverage = historical_data['coverage'].mean()
    current_margin = current_metrics.get('margin', 0)
    current_coverage = current_metrics.get('coverage', 0)
    
    # Determine performance level
    if current_margin > avg_margin * 1.1:
        margin_performance = "above average"
        margin_emoji = "📈"
    elif current_margin < avg_margin * 0.9:
        margin_performance = "below average"
        margin_emoji = "📉"
    else:
        margin_performance = "average"
        margin_emoji = "📊"
    
    if current_coverage > avg_coverage * 1.1:
        coverage_performance = "above average"
        coverage_emoji = "🎯"
    elif current_coverage < avg_coverage * 0.9:
        coverage_performance = "below average"
        coverage_emoji = "⚠️"
    else:
        coverage_performance = "average"
        coverage_emoji = "📊"
    
    narrative = f"{margin_emoji} **Trend Analysis:** Current margin performance is {margin_performance} at {current_margin:.1%} compared to historical average of {avg_margin:.1%}. {coverage_emoji} Coverage performance is {coverage_performance} at {current_coverage:.1f}% versus historical average of {avg_coverage:.1f}%. Based on {len(historical_data)} historical scenarios, the service shows consistent patterns in operational efficiency and customer reach."
    
    return _clean_ai_text(narrative)

def generate_forecasting_insights(historical_data: pd.DataFrame, forecast_periods: int = 6) -> Dict[str, Any]:
    """
    Generate forecasting insights using machine learning and AI.
    
    Args:
        historical_data: DataFrame with historical scenarios
        forecast_periods: Number of periods to forecast ahead
    
    Returns:
        Dictionary containing forecasts and AI insights
    """
    try:
        # Prepare data for forecasting
        data = historical_data.copy()
        
        # Create time series (assuming scenarios are chronological)
        data['period'] = range(len(data))
        
        forecasts = {}
        
        # Forecast margin
        margin_forecast = _forecast_metric(data, 'margin', forecast_periods)
        forecasts['margin'] = margin_forecast
        
        # Forecast coverage
        coverage_forecast = _forecast_metric(data, 'coverage', forecast_periods)
        forecasts['coverage'] = coverage_forecast
        
        # Forecast revenue
        revenue_forecast = _forecast_metric(data, 'revenue', forecast_periods)
        forecasts['revenue'] = revenue_forecast
        
        # Generate AI insights for forecasts
        ai_insights = _generate_forecast_ai_insights(forecasts, historical_data)
        
        return {
            'forecasts': forecasts,
            'ai_insights': ai_insights,
            'forecast_periods': forecast_periods,
            'model_accuracy': {
                'margin_r2': margin_forecast.get('r2_score', 0),
                'coverage_r2': coverage_forecast.get('r2_score', 0),
                'revenue_r2': revenue_forecast.get('r2_score', 0)
            }
        }
        
    except Exception as e:
        print(f"🔍 DEBUG: Error in forecasting: {e}")
        return {
            'forecasts': {},
            'ai_insights': "Forecasting analysis unavailable due to insufficient data.",
            'forecast_periods': forecast_periods,
            'model_accuracy': {}
        }

def _forecast_metric(data: pd.DataFrame, metric: str, periods: int) -> Dict[str, Any]:
    """Forecast a specific metric using polynomial regression."""
    try:
        X = data[['period']].values
        y = data[metric].values
        
        if len(X) < 3:
            # Not enough data for meaningful forecasting
            return {
                'values': [data[metric].iloc[-1]] * periods,
                'r2_score': 0,
                'mae': 0,
                'trend': 'insufficient_data'
            }
        
        # Use polynomial features for better curve fitting
        poly_features = PolynomialFeatures(degree=min(2, len(X)-1))
        X_poly = poly_features.fit_transform(X)
        
        # Fit model
        model = LinearRegression()
        model.fit(X_poly, y)
        
        # Calculate accuracy
        y_pred = model.predict(X_poly)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        
        # Generate forecasts
        future_periods = np.array(range(len(X), len(X) + periods)).reshape(-1, 1)
        future_poly = poly_features.transform(future_periods)
        future_values = model.predict(future_poly)
        
        # Ensure we have the expected number of forecast values
        if len(future_values) < periods:
            # Pad with the last value if needed
            last_value = future_values[-1] if len(future_values) > 0 else y[-1]
            while len(future_values) < periods:
                future_values = np.append(future_values, last_value)
        
        # Determine trend
        if len(future_values) > 1:
            trend = 'increasing' if future_values[-1] > future_values[0] else 'decreasing'
        else:
            trend = 'stable'
        
        return {
            'values': future_values.tolist(),
            'r2_score': r2,
            'mae': mae,
            'trend': trend
        }
        
    except Exception as e:
        print(f"🔍 DEBUG: Error forecasting {metric}: {e}")
        return {
            'values': [data[metric].iloc[-1]] * periods,
            'r2_score': 0,
            'mae': 0,
            'trend': 'error'
        }

def _generate_forecast_ai_insights(forecasts: Dict[str, Any], historical_data: pd.DataFrame) -> str:
    """Generate AI insights for forecasting results using Ollama, Google AI, or intelligent fallback."""
    
    # Try Ollama first (local AI)
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=2)
        if response.status_code == 200:
            print("🔍 DEBUG: Using Ollama for forecast insights")
            ai_narrative = _generate_ollama_forecast_insights(forecasts, historical_data)
            if ai_narrative:
                return ai_narrative
    except Exception as e:
        print(f"🔍 DEBUG: Ollama not available for forecast insights ({e})")

    # Try Google AI (cloud AI)
    ai_narrative = _generate_google_ai_forecast_insights(forecasts, historical_data)
    if ai_narrative:
        return ai_narrative

    # Fallback to intelligent templates
    return _generate_template_forecast_insights(forecasts, historical_data)

def _generate_ollama_forecast_insights(forecasts: Dict[str, Any], historical_data: pd.DataFrame) -> Optional[str]:
    """Generate forecast insights using Ollama (local LLM)."""

    # Prepare forecast summary
    margin_trend = forecasts.get('margin', {}).get('trend', 'unknown')
    coverage_trend = forecasts.get('coverage', {}).get('trend', 'unknown')
    revenue_trend = forecasts.get('revenue', {}).get('trend', 'unknown')
    
    avg_historical_margin = historical_data['margin'].mean()
    avg_historical_coverage = historical_data['coverage'].mean()
    
    prompt = f"""As a senior business analyst, provide strategic forecasting insights for this connectivity service:

Forecast Trends (next 6 periods):
- Margin trend: {margin_trend}
- Coverage trend: {coverage_trend}
- Revenue trend: {revenue_trend}

Historical Context:
- Average historical margin: {avg_historical_margin:.1%}
- Average historical coverage: {avg_historical_coverage:.1f}%
- Data points analyzed: {len(historical_data)}

Provide strategic recommendations, risk assessments, and business implications based on these forecasts. Focus on actionable insights for business planning.
Keep it under 150 words. Use business terminology - refer to 'lines', 'customers', 'service coverage', NOT 'students' or 'educational'.
IMPORTANT: Write as a single paragraph without line breaks or bullet points."""

    try:
        # Use Ollama API
        models_to_try = ['llama3.2', 'llama3.1', 'llama2', 'mistral', 'codellama']

        for model_name in models_to_try:
            data = {
                'model': model_name,
                'prompt': f"You are an expert business analyst providing strategic forecasting insights. Be analytical and professional. Write in a single paragraph without line breaks or bullet points.\n\n{prompt}",
                'stream': False,
                'options': {
                    'temperature': 0.7,
                    'num_predict': 200,
                    'stop': ['\n\n', '**', '##', '###']
                }
            }

            try:
                response = requests.post(
                    'http://localhost:11434/api/generate',
                    json=data,
                    timeout=30
                )

                if response.status_code == 200:
                    result = response.json()
                    narrative = result['response'].strip()

                    # Clean up the narrative formatting
                    narrative = _clean_ai_text(narrative)

                    print(f"🔍 DEBUG: Ollama forecast insights success with {model_name} - generated {len(narrative)} character narrative")
                    return narrative
                elif response.status_code == 404:
                    print(f"🔍 DEBUG: Model {model_name} not found for forecast insights, trying next...")
                    continue
                else:
                    print(f"🔍 DEBUG: Ollama API error with {model_name} for forecast insights: {response.status_code}")
                    continue
            except Exception as e:
                print(f"🔍 DEBUG: Ollama API error with {model_name} for forecast insights: {e}")
                continue

        print("🔍 DEBUG: No available models found in Ollama for forecast insights")
        return None

    except Exception as e:
        print(f"🔍 DEBUG: Ollama API error for forecast insights: {e}")
        return None

def _generate_google_ai_forecast_insights(forecasts: Dict[str, Any], historical_data: pd.DataFrame) -> Optional[str]:
    """Generate forecast insights using Google AI (Gemini API)."""
    
    # Get API key using the helper function
    api_key = _get_google_ai_api_key()
    if not api_key:
        print("🔍 DEBUG: Google AI API key not found for forecast insights - using fallback templates")
        return None
    
    print("🔍 DEBUG: Using Google AI for forecast insights")
    
    # Prepare forecast summary
    margin_trend = forecasts.get('margin', {}).get('trend', 'unknown')
    coverage_trend = forecasts.get('coverage', {}).get('trend', 'unknown')
    revenue_trend = forecasts.get('revenue', {}).get('trend', 'unknown')
    
    avg_historical_margin = historical_data['margin'].mean()
    avg_historical_coverage = historical_data['coverage'].mean()
    
    prompt = f"""As a senior business analyst, provide strategic forecasting insights for this connectivity service:

Forecast Trends (next 6 periods):
- Margin trend: {margin_trend}
- Coverage trend: {coverage_trend}
- Revenue trend: {revenue_trend}

Historical Context:
- Average historical margin: {avg_historical_margin:.1%}
- Average historical coverage: {avg_historical_coverage:.1f}%
- Data points analyzed: {len(historical_data)}

Provide strategic recommendations, risk assessments, and business implications based on these forecasts. Focus on actionable insights for business planning.
Keep it under 150 words. Use business terminology - refer to 'lines', 'customers', 'service coverage', NOT 'students' or 'educational'.
IMPORTANT: Write as a single paragraph without line breaks or bullet points."""

    try:
        # Use Google AI Gemini API
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
        
        headers = {
            'Content-Type': 'application/json',
        }
        
        data = {
            "contents": [{
                "parts": [{
                    "text": f"You are an expert business analyst providing strategic forecasting insights. Be analytical and professional. Write in a single paragraph without line breaks or bullet points.\n\n{prompt}"
                }]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 250,
                "stopSequences": ["\n\n", "**", "##", "###"]
            }
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            
            # Extract the generated text from Google AI response
            if 'candidates' in result and len(result['candidates']) > 0:
                candidate = result['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content']:
                    narrative = candidate['content']['parts'][0]['text'].strip()
                    
                    # Clean up the narrative formatting
                    narrative = _clean_ai_text(narrative)
                    
                    print(f"🔍 DEBUG: Google AI forecast insights success - generated {len(narrative)} character narrative")
                    return narrative
            
            print("🔍 DEBUG: Google AI forecast insights response format unexpected")
            return None
        else:
            print(f"🔍 DEBUG: Google AI forecast insights API error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"🔍 DEBUG: Google AI forecast insights API error: {e}")
        return None

def _generate_template_forecast_insights(forecasts: Dict[str, Any], historical_data: pd.DataFrame) -> str:
    """Generate template-based forecast insights when AI is not available."""
    
    margin_trend = forecasts.get('margin', {}).get('trend', 'unknown')
    coverage_trend = forecasts.get('coverage', {}).get('trend', 'unknown')
    
    # Determine overall outlook
    if margin_trend == 'increasing' and coverage_trend == 'increasing':
        outlook = "very positive"
        emoji = "🚀"
    elif margin_trend == 'increasing' or coverage_trend == 'increasing':
        outlook = "positive"
        emoji = "📈"
    elif margin_trend == 'decreasing' and coverage_trend == 'decreasing':
        outlook = "challenging"
        emoji = "⚠️"
    else:
        outlook = "mixed"
        emoji = "📊"
    
    narrative = f"{emoji} **Forecast Outlook:** The {outlook} forecast indicates {margin_trend} margin trends and {coverage_trend} coverage patterns over the next 6 periods. Based on {len(historical_data)} historical data points, strategic adjustments may be needed to optimize performance and maintain competitive positioning in the connectivity market."
    
    return _clean_ai_text(narrative)

def create_forecast_chart(forecasts: Dict[str, Any], historical_data: pd.DataFrame) -> plt.Figure:
    """Create an interactive forecast chart."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('AI-Powered Forecasting Analysis', fontsize=16, fontweight='bold')
    
    # Historical periods
    hist_periods = range(len(historical_data))
    # Future periods
    future_periods = range(len(historical_data), len(historical_data) + forecasts.get('forecast_periods', 6))
    
    # Plot 1: Margin Forecast
    ax1 = axes[0, 0]
    ax1.plot(hist_periods, historical_data['margin'] * 100, 'b-o', label='Historical', linewidth=2)
    if 'margin' in forecasts and forecasts['margin']['values'] and len(forecasts['margin']['values']) > 0:
        # Ensure we don't exceed the forecast periods
        forecast_values = forecasts['margin']['values'][:len(future_periods)]
        ax1.plot(future_periods[:len(forecast_values)], [v * 100 for v in forecast_values], 'r--o', label='Forecast', linewidth=2)
    ax1.set_title('Margin Forecast (%)', fontweight='bold')
    ax1.set_xlabel('Period')
    ax1.set_ylabel('Margin (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Coverage Forecast
    ax2 = axes[0, 1]
    ax2.plot(hist_periods, historical_data['coverage'], 'g-o', label='Historical', linewidth=2)
    if 'coverage' in forecasts and forecasts['coverage']['values'] and len(forecasts['coverage']['values']) > 0:
        # Ensure we don't exceed the forecast periods
        forecast_values = forecasts['coverage']['values'][:len(future_periods)]
        ax2.plot(future_periods[:len(forecast_values)], forecast_values, 'r--o', label='Forecast', linewidth=2)
    ax2.set_title('Coverage Forecast (%)', fontweight='bold')
    ax2.set_xlabel('Period')
    ax2.set_ylabel('Coverage (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Revenue Forecast
    ax3 = axes[1, 0]
    ax3.plot(hist_periods, historical_data['revenue'], 'purple', marker='o', label='Historical', linewidth=2)
    if 'revenue' in forecasts and forecasts['revenue']['values'] and len(forecasts['revenue']['values']) > 0:
        # Ensure we don't exceed the forecast periods
        forecast_values = forecasts['revenue']['values'][:len(future_periods)]
        ax3.plot(future_periods[:len(forecast_values)], forecast_values, 'r--o', label='Forecast', linewidth=2)
    ax3.set_title('Revenue Forecast ($)', fontweight='bold')
    ax3.set_xlabel('Period')
    ax3.set_ylabel('Revenue ($)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Model Accuracy
    ax4 = axes[1, 1]
    accuracy_data = forecasts.get('model_accuracy', {})
    metrics = ['Margin R²', 'Coverage R²', 'Revenue R²']
    values = [
        accuracy_data.get('margin_r2', 0),
        accuracy_data.get('coverage_r2', 0),
        accuracy_data.get('revenue_r2', 0)
    ]
    
    bars = ax4.bar(metrics, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax4.set_title('Model Accuracy (R² Score)', fontweight='bold')
    ax4.set_ylabel('R² Score')
    ax4.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    return fig

def get_ai_insights_status() -> Dict[str, Any]:
    """Get the status of AI insights capabilities."""
    # Check if Ollama is available
    ollama_available = False
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=2)
        ollama_available = response.status_code == 200
    except:
        pass
    
    # Check if Google AI API key is available
    google_ai_available = bool(_get_google_ai_api_key())
    
    # Determine status message
    if ollama_available and google_ai_available:
        status = 'Ollama (Local) + Google AI (Cloud) available'
    elif ollama_available:
        status = 'Ollama (Local AI) available'
    elif google_ai_available:
        status = 'Google AI (Cloud) available'
    else:
        status = 'Using intelligent templates'

    return {
        'ollama_available': ollama_available,
        'google_ai_available': google_ai_available,
        'fallback_available': True,
        'status': status
    }
