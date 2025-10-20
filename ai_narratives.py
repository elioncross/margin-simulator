"""
AI Narratives Module for Margin Impact Simulator
Provides AI-powered contextual narratives using Ollama (local), Google AI (cloud), or intelligent fallbacks.
"""

import requests
import json
import os
import re
from typing import Dict, Any, Optional

def _clean_narrative(text: str) -> str:
    """
    Clean up AI-generated narrative text to fix formatting issues.
    """
    # Remove excessive line breaks and normalize whitespace first
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Fix specific problematic patterns that cause character splitting
    # Fix the specific pattern: "10.00/GBwith3.5GBcap" -> "10.00/GB with 3.5GB cap"
    text = re.sub(r'(\d+\.\d+)/GBwith(\d+\.\d+)GBcap', r'\1/GB with \2GB cap', text)
    
    # Fix "versus" spacing issues: "7,500versus8,000" -> "7,500 versus 8,000"
    text = re.sub(r'(\d+,\d+)versus(\d+,\d+)', r'\1 versus \2', text)
    text = re.sub(r'(\d+)versus(\d+)', r'\1 versus \2', text)
    
    # Fix comma spacing in numbers: "7 , 500" -> "7,500"
    text = re.sub(r'(\d+)\s*,\s*(\d+)', r'\1,\2', text)
    
    # Fix decimal spacing: "7 . 5" -> "7.5"
    text = re.sub(r'(\d+)\s*\.\s*(\d+)', r'\1.\2', text)
    
    # Fix any remaining "G B" spacing issues
    text = re.sub(r'G\s+B', 'GB', text)
    
    # Remove bullet points and list formatting
    text = re.sub(r'^\s*[-*•]\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s*', '', text, flags=re.MULTILINE)
    
    # Clean up formatting artifacts
    text = re.sub(r'\s*:\s*', ': ', text)
    
    # Fix spacing around % and $ symbols
    text = re.sub(r'(\d+)\s*%', r'\1%', text)  # Remove space before % in numbers
    text = re.sub(r'%\s*([a-zA-Z])', r'% \1', text)  # Add space after % before letters
    text = re.sub(r'([a-zA-Z])\s*\$', r'\1 $', text)  # Add space before $ after letters
    text = re.sub(r'\$\s*(\d+)', r'$\1', text)  # Remove space after $ before numbers
    
    # Ensure proper sentence structure
    text = text.strip()
    
    # Remove trailing periods and add just one
    text = re.sub(r'\.+$', '', text)
    if text and not text.endswith(('.', '!', '?')):
        text += '.'
    
    return text

def generate_ai_narrative(metrics: Dict[str, Any], context: Dict[str, Any], view_type: str = "financial") -> str:
    """
    Generate AI-powered narrative using Ollama (local), Google AI (cloud), or intelligent fallback.
    
    Args:
        metrics: Current calculated metrics (usage, cost, revenue, margin, coverage)
        context: Scenario context (students, cap, budget, policy, etc.)
        view_type: "customer" or "financial" for different narrative styles
    
    Returns:
        Generated narrative string
    """
    
    # Try Ollama first (local AI)
    print(f"🔍 DEBUG: Trying Ollama for {view_type} narrative")
    ai_narrative = _generate_ollama_narrative(metrics, context, view_type)
    if ai_narrative:
        print(f"🔍 DEBUG: Ollama success for {view_type} narrative")
        return ai_narrative
    
    # Try Google AI (cloud AI)
    print(f"🔍 DEBUG: Trying Google AI for {view_type} narrative")
    ai_narrative = _generate_google_ai_narrative(metrics, context, view_type)
    if ai_narrative:
        print(f"🔍 DEBUG: Google AI success for {view_type} narrative")
        return ai_narrative
    
    # Fallback to intelligent templates
    return _generate_template_narrative(metrics, context, view_type)

def _generate_ollama_narrative(metrics: Dict[str, Any], context: Dict[str, Any], view_type: str) -> Optional[str]:
    """
    Generate narrative using Ollama (local LLM).
    
    Note: This requires Ollama to be running locally. Default model is 'llama3.2'.
    """
    
    # Check if Ollama is available
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=2)
        if response.status_code != 200:
            print("🔍 DEBUG: Ollama not responding - using fallback templates")
            return None
    except Exception as e:
        print(f"🔍 DEBUG: Ollama not available ({e}) - using fallback templates")
        return None
    
    print(f"🔍 DEBUG: Using Ollama for {view_type} narrative")
    
    # Prepare the prompt based on view type
    if view_type == "customer":
        prompt = f"""As a business technology consultant, provide an encouraging, customer-focused narrative about this connectivity scenario:

- {context['students']} lines with {context['cap']}GB data cap each
- {metrics['coverage']:.1f}% coverage achieved
- Vertical: {context['policy']}
- Throttling: {'Enabled' if context['throttling'] else 'Disabled'}

Focus on service coverage, customer satisfaction, and business outcomes. Use business terminology - refer to 'lines', 'customers', 'service coverage', NOT 'students' or 'educational'. Be encouraging and solution-oriented.
Keep it under 100 words and use emojis appropriately. Write in a professional, accessible tone.
IMPORTANT: Write as a single paragraph without line breaks or bullet points."""
    else:  # financial
        prompt = f"""As a business analyst, provide a concise analysis of this connectivity service performance:

- Revenue: ${metrics['revenue']:,.0f} (Budget: ${context['budget']:,.0f})
- Margin: {metrics['margin']:.1%}
- Coverage: {metrics['coverage']:.1f}%
- Vertical: {context['policy']}
- Throttling: {'On' if context['throttling'] else 'Off'}
- Usage: {metrics['usage']:,.1f} GB

Highlight key business insights, risks, and opportunities. Use business terminology - refer to 'lines', 'customers', 'service coverage', NOT 'students' or 'educational'. Be analytical but accessible.
Keep it under 120 words. Focus on operational performance and actionable business insights.
IMPORTANT: Write as a single paragraph without line breaks or bullet points."""
    
    try:
        # Use Ollama API - try common model names
        models_to_try = ['llama3.2', 'llama3.1', 'llama2', 'mistral', 'codellama']
        
        for model_name in models_to_try:
            data = {
                'model': model_name,
                'prompt': f"You are an expert analyst providing clear, actionable insights. Be concise and professional. Write in a single paragraph without line breaks or bullet points.\n\n{prompt}",
                'stream': False,
                'options': {
                    'temperature': 0.7,
                    'num_predict': 150,
                    'stop': ['\n\n', '**', '##', '###']  # Stop at formatting markers
                }
            }
            
            try:
                response = requests.post(
                    'http://localhost:11434/api/generate',
                    json=data,
                    timeout=30  # Increased timeout for first request
                )
                
                if response.status_code == 200:
                    result = response.json()
                    narrative = result['response'].strip()
                    
                    # Clean up the narrative formatting
                    narrative = _clean_narrative(narrative)
                    
                    print(f"🔍 DEBUG: Ollama success with {model_name} - generated {len(narrative)} character narrative")
                    return narrative
                elif response.status_code == 404:
                    print(f"🔍 DEBUG: Model {model_name} not found, trying next...")
                    continue
                else:
                    print(f"🔍 DEBUG: Ollama API error with {model_name}: {response.status_code}")
                    continue
            except Exception as e:
                print(f"🔍 DEBUG: Ollama API error with {model_name}: {e}")
                continue
        
        print("🔍 DEBUG: No available models found in Ollama")
        return None
            
    except Exception as e:
        print(f"🔍 DEBUG: Ollama API error: {e}")
        return None

def _generate_google_ai_narrative(metrics: Dict[str, Any], context: Dict[str, Any], view_type: str) -> Optional[str]:
    """
    Generate narrative using Google AI (Gemini API).
    
    Note: This requires a Google AI API key to be set in the environment variable GOOGLE_AI_API_KEY.
    """
    
    # Get API key using the helper function
    api_key = _get_google_ai_api_key()
    
    if not api_key:
        return None
    
    # Prepare the prompt based on view type
    if view_type == "customer":
        prompt = f"""As a business technology consultant, provide an encouraging, customer-focused narrative about this connectivity scenario:

- {context['students']} lines with {context['cap']}GB data cap each
- {metrics['coverage']:.1f}% coverage achieved
- Vertical: {context['policy']}
- Throttling: {'Enabled' if context['throttling'] else 'Disabled'}

Focus on service coverage, customer satisfaction, and business outcomes. Use business terminology - refer to 'lines', 'customers', 'service coverage', NOT 'students' or 'educational'. Be encouraging and solution-oriented.
Keep it under 100 words and use emojis appropriately. Write in a professional, accessible tone.
IMPORTANT: Write as a single paragraph without line breaks or bullet points."""
    else:  # financial
        prompt = f"""As a business analyst, provide a concise analysis of this connectivity service performance:

- Revenue: ${metrics['revenue']:,.0f} (Budget: ${context['budget']:,.0f})
- Margin: {metrics['margin']:.1%}
- Coverage: {metrics['coverage']:.1f}%
- Vertical: {context['policy']}
- Throttling: {'On' if context['throttling'] else 'Off'}
- Usage: {metrics['usage']:,.1f} GB

Highlight key business insights, risks, and opportunities. Use business terminology - refer to 'lines', 'customers', 'service coverage', NOT 'students' or 'educational'. Be analytical but accessible.
Keep it under 120 words. Focus on operational performance and actionable business insights.
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
                    "text": f"You are an expert analyst providing clear, actionable insights. Be concise and professional. Write in a single paragraph without line breaks or bullet points.\n\n{prompt}"
                }]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 200,
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
                    narrative = _clean_narrative(narrative)
                    
                    print(f"🔍 DEBUG: Google AI success - generated {len(narrative)} character narrative")
                    return narrative
            
            print("🔍 DEBUG: Google AI response format unexpected")
            return None
        else:
            print(f"🔍 DEBUG: Google AI API error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"🔍 DEBUG: Google AI API error: {e}")
        return None

def _generate_template_narrative(metrics: Dict[str, Any], context: Dict[str, Any], view_type: str) -> str:
    """
    Generate intelligent template-based narrative as fallback.
    """
    
    print(f"🔍 DEBUG: Using template-based narrative for {view_type} view")
    
    if view_type == "customer":
        return _generate_customer_narrative(metrics, context)
    else:
        return _generate_financial_narrative(metrics, context)

def _generate_customer_narrative(metrics: Dict[str, Any], context: Dict[str, Any]) -> str:
    """Generate customer-focused narrative."""
    
    coverage = metrics['coverage']
    lines = context['students']
    connected_lines = int(lines * coverage / 100)
    
    if coverage > 85:
        return f"🎉 **Excellent connectivity!** Your program is successfully reaching {coverage:.1f}% of lines ({connected_lines} out of {lines}), ensuring comprehensive service coverage. This high coverage rate demonstrates strong operational effectiveness and customer satisfaction."
    
    elif coverage > 70:
        return f"📈 **Strong progress!** You're connecting {coverage:.1f}% of lines ({connected_lines} out of {lines}), providing substantial service coverage. Consider expanding your program to reach even more customers and maximize market penetration."
    
    elif coverage > 50:
        return f"🎯 **Growing opportunity!** Currently {coverage:.1f}% coverage ({connected_lines} out of {lines} lines). Strategic investments in connectivity infrastructure could significantly improve service coverage and customer access."
    
    else:
        return f"💡 **Significant potential!** With {coverage:.1f}% coverage ({connected_lines} out of {lines} lines), there's substantial opportunity to expand connectivity and ensure comprehensive service coverage for all customers."

def _generate_financial_narrative(metrics: Dict[str, Any], context: Dict[str, Any]) -> str:
    """Generate financial business-focused narrative."""
    
    margin = metrics['margin']
    revenue = metrics['revenue']
    coverage = metrics['coverage']
    budget = context['budget']
    
    # Determine profitability status
    if margin > 0.15:
        profitability = "highly profitable"
        emoji = "💰"
    elif margin > 0.05:
        profitability = "profitable"
        emoji = "✅"
    elif margin > 0:
        profitability = "marginally profitable"
        emoji = "⚖️"
    else:
        profitability = "unprofitable"
        emoji = "⚠️"
    
    # Coverage analysis
    if coverage > 80:
        coverage_status = "excellent"
    elif coverage > 60:
        coverage_status = "good"
    else:
        coverage_status = "needs improvement"
    
    # Budget utilization
    budget_utilization = (revenue / budget) * 100 if budget > 0 else 0
    
    return f"{emoji} **{profitability.title()} scenario** with {margin:.1%} margin. Revenue of ${revenue:,.0f} against ${metrics['carrier_cost']:,.0f} costs shows {profitability} unit economics. {coverage_status.title()} coverage at {coverage:.1f}% with {budget_utilization:.1f}% budget utilization. {'Consider optimization to improve profitability' if margin < 0.1 else 'Strong operational performance'}."

def generate_optimization_narrative(optimization_result: Dict[str, Any], current_metrics: Dict[str, Any]) -> str:
    """
    Generate AI narrative for pricing and data cap optimization results.
    Tries Ollama first, then Google AI, then falls back to intelligent templates.
    """

    # Try Ollama first (local AI)
    ai_narrative = _generate_ollama_optimization_narrative(optimization_result, current_metrics)
    if ai_narrative:
        return ai_narrative

    # Try Google AI (cloud AI)
    ai_narrative = _generate_google_ai_optimization_narrative(optimization_result, current_metrics)
    if ai_narrative:
        return ai_narrative

    # Fallback to intelligent templates
    return _generate_template_optimization_narrative(optimization_result, current_metrics)

def _generate_ollama_optimization_narrative(optimization_result: Dict[str, Any], current_metrics: Dict[str, Any]) -> Optional[str]:
    """
    Generate optimization narrative using Ollama (local LLM).
    """

    # Check if Ollama is available
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=2)
        if response.status_code != 200:
            print("🔍 DEBUG: Ollama not responding for optimization - using fallback templates")
            return None
    except Exception as e:
        print(f"🔍 DEBUG: Ollama not available for optimization ({e}) - using fallback templates")
        return None

    print("🔍 DEBUG: Using Ollama for optimization narrative")

    if not optimization_result.get('feasible', False):
        prompt = f"""As a business analyst, provide a concise analysis of why no feasible optimization solution was found:

Current constraints:
- Budget: ${current_metrics.get('budget', 0):,.0f}
- Current margin: {current_metrics.get('margin', 0):.1%}
- Current coverage: {current_metrics.get('coverage', 0):.1f}%

Focus on practical recommendations for adjusting constraints to find viable solutions.
Keep it under 100 words and be solution-oriented.
IMPORTANT: Write as a single paragraph without line breaks or bullet points."""
    else:
        recommended_price = optimization_result['recommended_customer_price']
        recommended_cap = optimization_result['recommended_cap']
        opt_metrics = optimization_result['metrics']
        improvement = optimization_result['improvement']
        margin_improvement = improvement['margin_improvement']
        revenue_improvement = improvement['revenue_improvement']

        prompt = f"""As a business analyst, provide a comprehensive analysis of this pricing optimization opportunity:

Current vs Optimal:
- Current: ${current_metrics.get('customer_price', 0):.2f}/month per line, {current_metrics.get('cap', 0):.1f}GB cap
- Optimal: ${recommended_price:.2f}/month per line, {recommended_cap:.1f}GB cap
- Margin improvement: {margin_improvement:.1%}
- Revenue improvement: ${revenue_improvement:,.0f}
- New coverage: {opt_metrics['coverage']:.1f}%

Analyze the business implications, risks, and strategic opportunities. Be analytical but accessible.
Keep it under 150 words. Focus on actionable insights and business impact.
IMPORTANT: Write as a single paragraph without line breaks or bullet points."""

    try:
        # Use Ollama API - try common model names
        models_to_try = ['llama3.2', 'llama3.1', 'llama2', 'mistral', 'codellama']

        for model_name in models_to_try:
            data = {
                'model': model_name,
                'prompt': f"You are an expert business analyst providing clear, actionable insights. Be concise and professional. Write in a single paragraph without line breaks or bullet points.\n\n{prompt}",
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
                    timeout=30  # Increased timeout for first request
                )

                if response.status_code == 200:
                    result = response.json()
                    narrative = result['response'].strip()

                    # Clean up the narrative formatting
                    narrative = _clean_narrative(narrative)

                    print(f"🔍 DEBUG: Ollama optimization success with {model_name} - generated {len(narrative)} character narrative")
                    return narrative
                elif response.status_code == 404:
                    print(f"🔍 DEBUG: Model {model_name} not found for optimization, trying next...")
                    continue
                else:
                    print(f"🔍 DEBUG: Ollama API error with {model_name} for optimization: {response.status_code}")
                    continue
            except Exception as e:
                print(f"🔍 DEBUG: Ollama API error with {model_name} for optimization: {e}")
                continue

        print("🔍 DEBUG: No available models found in Ollama for optimization")
        return None

    except Exception as e:
        print(f"🔍 DEBUG: Ollama API error for optimization: {e}")
        return None

def _generate_google_ai_optimization_narrative(optimization_result: Dict[str, Any], current_metrics: Dict[str, Any]) -> Optional[str]:
    """
    Generate optimization narrative using Google AI (Gemini API).
    """
    
    # Get API key using the helper function
    api_key = _get_google_ai_api_key()
    
    if not api_key:
        return None
    
    if not optimization_result.get('feasible', False):
        prompt = f"""As a business analyst, provide a concise analysis of why no feasible optimization solution was found:

Current constraints:
- Budget: ${current_metrics.get('budget', 0):,.0f}
- Current margin: {current_metrics.get('margin', 0):.1%}
- Current coverage: {current_metrics.get('coverage', 0):.1f}%

Focus on practical recommendations for adjusting constraints to find viable solutions.
Keep it under 100 words and be solution-oriented.
IMPORTANT: Write as a single paragraph without line breaks or bullet points."""
    else:
        recommended_price = optimization_result['recommended_customer_price']
        recommended_cap = optimization_result['recommended_cap']
        opt_metrics = optimization_result['metrics']
        improvement = optimization_result['improvement']
        margin_improvement = improvement['margin_improvement']
        revenue_improvement = improvement['revenue_improvement']

        prompt = f"""As a business analyst, provide a comprehensive analysis of this pricing optimization opportunity:

Current vs Optimal:
- Current: ${current_metrics.get('customer_price', 0):.2f}/month per line, {current_metrics.get('cap', 0):.1f}GB cap
- Optimal: ${recommended_price:.2f}/month per line, {recommended_cap:.1f}GB cap
- Margin improvement: {margin_improvement:.1%}
- Revenue improvement: ${revenue_improvement:,.0f}
- New coverage: {opt_metrics['coverage']:.1f}%

Analyze the business implications, risks, and strategic opportunities. Be analytical but accessible.
Keep it under 150 words. Focus on actionable insights and business impact.
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
                    "text": f"You are an expert business analyst providing clear, actionable insights. Be concise and professional. Write in a single paragraph without line breaks or bullet points.\n\n{prompt}"
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
                    narrative = _clean_narrative(narrative)
                    
                    print(f"🔍 DEBUG: Google AI optimization success - generated {len(narrative)} character narrative")
                    return narrative
            
            print("🔍 DEBUG: Google AI optimization response format unexpected")
            return None
        else:
            print(f"🔍 DEBUG: Google AI optimization API error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"🔍 DEBUG: Google AI optimization API error: {e}")
        return None

def _generate_template_optimization_narrative(optimization_result: Dict[str, Any], current_metrics: Dict[str, Any]) -> str:
    """
    Generate a template-based optimization narrative when AI is not available.
    """

    if not optimization_result.get('feasible', False):
        return "❌ **No feasible solution found** with current constraints. Consider adjusting budget limits, minimum coverage, or margin requirements to find viable optimization opportunities."

    recommended_price = optimization_result['recommended_customer_price']
    recommended_cap = optimization_result['recommended_cap']
    opt_metrics = optimization_result['metrics']
    improvement = optimization_result['improvement']

    margin_improvement = improvement['margin_improvement']
    revenue_improvement = improvement['revenue_improvement']

    if margin_improvement > 0.05:
        improvement_level = "significant"
        emoji = "🚀"
    elif margin_improvement > 0.01:
        improvement_level = "moderate"
        emoji = "📈"
    else:
        improvement_level = "marginal"
        emoji = "📊"

    # Determine pricing strategy
    if recommended_price > current_metrics.get('customer_price', 0):
        pricing_strategy = "premium pricing"
    elif recommended_price < current_metrics.get('customer_price', 0):
        pricing_strategy = "competitive pricing"
    else:
        pricing_strategy = "current pricing"

    # Determine cap strategy
    if recommended_cap > current_metrics.get('cap', 0):
        cap_strategy = "increased data allowance"
    elif recommended_cap < current_metrics.get('cap', 0):
        cap_strategy = "optimized data cap"
    else:
        cap_strategy = "current data cap"

    # Generate the narrative text
    narrative = f"{emoji} {improvement_level.title()} optimization opportunity identified! Optimal configuration: ${recommended_price:.2f}/GB with {recommended_cap:.1f}GB cap. This {pricing_strategy} approach with {cap_strategy} could improve margin by {margin_improvement:.1%} and revenue by ${revenue_improvement:,.0f}. The strategy balances profitability with line coverage at {opt_metrics['coverage']:.1f}%."

    # Clean up the narrative formatting
    return _clean_narrative(narrative)

def _get_google_ai_api_key() -> str:
    """
    Get Google AI API key from environment or Streamlit secrets.
    """
    # Check environment variable first
    api_key = os.getenv('GOOGLE_AI_API_KEY')
    if api_key:
        return api_key
    
    # Try Streamlit secrets (handle errors gracefully to avoid local warnings)
    try:
        import streamlit as st
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

def get_ai_narrative_status() -> Dict[str, Any]:
    """
    Get the status of AI narrative capabilities.
    """
    # Check if Ollama is available
    ollama_available = False
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=2)
        ollama_available = response.status_code == 200
    except:
        pass
    
    # Check if Google AI API key is available
    api_key = _get_google_ai_api_key()
    
    google_ai_available = bool(api_key)
    
    # Debug information for troubleshooting
    debug_info = {
        'api_key_present': bool(api_key),
        'api_key_length': len(api_key) if api_key else 0,
        'api_key_prefix': api_key[:10] + "..." if api_key and len(api_key) > 10 else "N/A",
        'all_env_vars': [k for k in os.environ.keys() if 'GOOGLE' in k or 'AI' in k]
    }
    
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
        'status': status,
        'debug_info': debug_info
    }
