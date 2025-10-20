#!/usr/bin/env python3
"""
Test script to verify Ollama integration
"""

import os
from ai_narratives import generate_ai_narrative, get_ai_narrative_status

def test_ollama_integration():
    """Test Ollama integration"""
    
    print("🔍 Testing Ollama Integration")
    print("=" * 50)
    
    # Get AI status
    ai_status = get_ai_narrative_status()
    print(f"📊 AI Status: {ai_status['status']}")
    print()
    
    # Test data
    metrics = {
        'usage': 1500.0,
        'carrier_cost': 7500.0,
        'revenue': 10000.0,
        'margin': 0.25,
        'coverage': 85.0
    }
    
    context = {
        'students': 200,  # Note: 'students' is the variable name, but represents lines
        'cap': 10.0,
        'budget': 12000.0,
        'carrier_rate': 5.0,
        'customer_price': 6.67,
        'policy': 'Public Sector (Schools)',
        'throttling': False
    }
    
    print("🧪 Testing Customer Narrative:")
    print("-" * 30)
    customer_narrative = generate_ai_narrative(metrics, context, "customer")
    print(f"Result: {customer_narrative}")
    print()
    
    print("🧪 Testing Internal Narrative:")
    print("-" * 30)
    internal_narrative = generate_ai_narrative(metrics, context, "internal")
    print(f"Result: {internal_narrative}")
    print()
    
    # Check if we can detect which mode was used
    if "🔍 DEBUG: Using Ollama" in str(customer_narrative):
        print("✅ Ollama was used!")
    elif "🔍 DEBUG: Using template-based narrative" in str(customer_narrative):
        print("📝 Template-based narrative was used")
    else:
        print("🤔 Could not determine which mode was used")
    
    print()
    print("🎯 To use Ollama:")
    print("1. Install Ollama: https://ollama.ai/")
    print("2. Pull a model: ollama pull llama3.2")
    print("3. Start Ollama: ollama serve")
    print("4. Run this test again")

if __name__ == "__main__":
    test_ollama_integration()
