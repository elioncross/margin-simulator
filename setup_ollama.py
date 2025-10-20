#!/usr/bin/env python3
"""
Ollama Setup Helper Script
Helps you install and configure Ollama for the Margin Impact Simulator
"""

import subprocess
import requests
import time

def check_ollama_running():
    """Check if Ollama is running"""
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=2)
        return response.status_code == 200
    except:
        return False

def get_available_models():
    """Get list of available models"""
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=2)
        if response.status_code == 200:
            data = response.json()
            return [model['name'] for model in data.get('models', [])]
    except:
        pass
    return []

def install_model(model_name):
    """Install a model using ollama pull"""
    print(f"📥 Installing {model_name}...")
    try:
        result = subprocess.run(['ollama', 'pull', model_name], 
                              capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"✅ Successfully installed {model_name}")
            return True
        else:
            print(f"❌ Failed to install {model_name}: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout installing {model_name} (this can take several minutes)")
        return False
    except FileNotFoundError:
        print("❌ Ollama not found. Please install Ollama first: https://ollama.ai/")
        return False

def main():
    print("🔧 Ollama Setup for Margin Impact Simulator")
    print("=" * 50)
    
    # Check if Ollama is running
    if not check_ollama_running():
        print("❌ Ollama is not running!")
        print("Please start Ollama first:")
        print("  ollama serve")
        print()
        return
    
    print("✅ Ollama is running")
    
    # Check available models
    available_models = get_available_models()
    print(f"📋 Available models: {available_models}")
    
    if not available_models:
        print("\n🎯 No models installed. Let's install one!")
        
        # Recommended models (lightweight to heavyweight)
        recommended_models = [
            ("llama3.2", "Lightweight, fast, good quality"),
            ("llama3.1", "Better quality, slower"),
            ("mistral", "Alternative lightweight option"),
            ("llama2", "Older but stable")
        ]
        
        print("\nRecommended models:")
        for i, (model, description) in enumerate(recommended_models, 1):
            print(f"  {i}. {model} - {description}")
        
        print("\nInstalling llama3.2 (recommended for speed)...")
        if install_model("llama3.2"):
            print("\n🎉 Setup complete! You can now use AI narratives in the app.")
        else:
            print("\n⚠️  Installation failed. You can still use the app with smart templates.")
    else:
        print(f"\n🎉 You already have models installed: {available_models}")
        print("You can now use AI narratives in the app!")
    
    print("\n📖 Next steps:")
    print("1. Run the app: streamlit run app.py")
    print("2. Enable 'AI Narratives' in the sidebar")
    print("3. Check the status indicator shows 'Ollama (Local AI) Available'")

if __name__ == "__main__":
    main()
