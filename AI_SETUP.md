# AI Narratives Setup Guide

## Overview
The Margin Impact Simulator now includes AI-powered contextual narratives that provide intelligent insights based on your data. The system uses Ollama (local AI) for dynamic AI generation with intelligent fallback templates.

## Setup Options

### Option 1: Ollama (Local AI) - Recommended
For the most sophisticated AI narratives running locally on your machine:

1. **Install Ollama**
   - Visit [Ollama.ai](https://ollama.ai/)
   - Download and install Ollama for your platform

2. **Pull a Model**
   ```bash
   # Pull a lightweight model (recommended)
   ollama pull llama3.2
   
   # Or pull a larger model for better quality
   ollama pull llama3.1
   ```

3. **Start Ollama**
   ```bash
   # Start Ollama server
   ollama serve
   ```

### Option 2: Smart Templates (No AI Required)
If you don't want to use AI, the app will automatically use intelligent template-based narratives that are still contextual and insightful.

## Features

### 🤖 AI-Powered Narratives
- **Customer View**: Encouraging, student-focused narratives about connectivity and equity
- **Internal View**: Business-focused analysis with insights on profitability and operations
- **Optimization Results**: AI explanation of optimization opportunities and recommendations

### 📝 Smart Fallbacks
- Intelligent template-based narratives when API is unavailable
- Contextual insights based on metrics and scenarios
- Professional, actionable language

### ⚙️ User Controls
- Toggle AI narratives on/off in the sidebar
- Real-time status indicator showing Ollama availability
- Seamless fallback to templates

## Usage

1. **Enable AI Narratives**: Check the "Enable AI Narratives" checkbox in the sidebar
2. **View Status**: The sidebar shows whether Ollama is available or using templates
3. **Generate Insights**: AI narratives appear automatically in Customer and Internal views
4. **Optimization Analysis**: Get AI-powered explanations of optimization results

## Example Narratives

### Customer View (AI-Generated)
> "🎉 Excellent connectivity! Your program is successfully reaching 87.3% of lines (174 out of 200), ensuring comprehensive service coverage. This high coverage rate demonstrates strong operational effectiveness and customer satisfaction."

### Internal View (AI-Generated)
> "💰 Highly profitable scenario with 23.4% margin. Revenue of $12,500 against $9,580 costs shows highly profitable unit economics. Excellent coverage at 87.3% with 100.0% budget utilization. Strong operational performance."

### Optimization Results (AI-Generated)
> "🚀 Significant optimization opportunity identified! Switching to Per Household policy with throttling disabled could improve margin by 8.2% (from 15.2% to 23.4%). Coverage would increase by 2.1 percentage points."

## Technical Details

### AI Integration
- Uses Ollama with `llama3.2` model (configurable)
- Optimized prompts for different view types
- Error handling with graceful fallbacks
- Local processing - no external API calls

### Template System
- Context-aware narrative generation
- Metric-based insights and recommendations
- Professional business language
- Emoji-enhanced readability

## Troubleshooting

### Ollama Not Working
- Check Ollama is running: `ollama serve`
- Verify model is installed: `ollama list`
- Check Ollama is accessible at `http://localhost:11434`
- App will automatically fallback to templates

### Slow Response
- AI generation takes 3-10 seconds (depending on model size)
- Templates are instant
- Consider using smaller models for faster response

### Customization
- Modify prompts in `ai_narratives.py`
- Change model in the `_generate_ollama_narrative` function
- Adjust template logic for different insights
- Add new narrative types as needed

## Cost Considerations

### Ollama (Local AI)
- Completely free - runs on your machine
- No API costs or usage limits
- Uses your computer's resources
- Templates are also free

### Recommendations
- Use Ollama for important presentations/demos
- Use templates for regular analysis
- Toggle based on your needs

## Support

For issues or questions:
1. Check the sidebar status indicator
2. Verify Ollama is running
3. Test with templates first
4. Review error messages in browser console
