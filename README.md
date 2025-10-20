# 📊 Margin Impact Simulator

A comprehensive Streamlit application for analyzing the financial impact of different policy and pricing scenarios in connectivity services.

## 🔐 Access

**Password:** `hackathon2025`

## 🚀 Features

- **Dual-Audience Views**: Customer vs Financial perspectives
- **Smart Cost Optimization (SCO)**: Advanced cost optimization analysis
- **AI-Powered Insights**: Forecasting and trend analysis
- **Optimization Engine**: Automated pricing and data cap optimization
- **Interactive Visualizations**: Professional charts and analytics
- **Export & Scenario Management**: Data portability and scenario comparison

## 🛠️ Local Development

### Prerequisites
- Python 3.7+
- pip

### Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd margin-simulator

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### Optional: AI Features
The app supports AI-powered narratives using multiple AI providers:

#### Local AI (Ollama)
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull the model
ollama pull llama3.2

# Start Ollama server
ollama serve
```

#### Cloud AI (Google AI)
```bash
# Get API Key from Google AI Studio
# https://aistudio.google.com

# Set environment variable
export GOOGLE_AI_API_KEY='your_api_key_here'
```

#### AI Priority
- **Local Development**: Ollama (if available) → Google AI (if API key set) → Templates
- **Cloud Deployment**: Google AI (if API key set) → Templates

**Note**: AI features are optional and the app works perfectly without them.

## 🌐 Deployment Options

### 1. Streamlit Community Cloud (Recommended for Hackathons)
- **Free tier available**
- Direct GitHub integration
- Automatic deployments
- **Google AI Support**: Set `GOOGLE_AI_API_KEY` in Streamlit secrets

**Steps:**
1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set Google AI API key in app secrets (optional)
5. Deploy with one click

### 2. AWS Container Deployment
For production use:
- **AWS App Runner**: Easiest container deployment
- **AWS ECS**: Full container orchestration
- **AWS EC2**: Direct server deployment

## 📱 Usage

1. **Enter Password**: Use `hackathon2025` to access the app
2. **Load Scenarios**: Choose from predefined scenarios or create custom ones
3. **Adjust Parameters**: Modify lines, pricing, data caps, and business types
4. **View Results**: Switch between Customer View, Financial View, and SCO Analysis
5. **Get Insights**: Use AI features for advanced analysis and forecasting
6. **Export Data**: Download scenarios and analysis results

## 🎯 Key Scenarios

- **✅ Profitable Example**: High margin, good coverage
- **❌ Unprofitable Example**: Low margin, budget-constrained
- **🎯 SCO-Enabled Enterprise**: Enterprise with intelligent cost optimization
- **🎯 SCO-Enabled Retail**: Retail households with SCO optimization
- **📊 Traditional Static Plans**: Fixed plans with overage charges

## 🔧 Configuration

### Sidebar Controls
- **Number of Lines**: Total lines in your service
- **Data Cap per Line**: GB allowance per line
- **Budget**: Total budget available
- **Carrier Rate**: Cost per GB from carrier
- **Monthly Price per Line**: Fixed monthly fee per line
- **Vertical**: Business type (Public Sector, Retail, Enterprise)
- **SCO Settings**: Smart Cost Optimization parameters

### AI Features
- **AI Narratives**: Contextual insights for different views
- **Trend Analysis**: Historical performance comparison
- **Forecasting**: Machine learning predictions
- **Optimization**: Automated pricing recommendations

## 📊 Understanding Results

### Coverage
- **100%**: Full budget utilization
- **85%+**: Excellent coverage
- **50-85%**: Good coverage
- **<50%**: Limited coverage

### Margin
- **Positive**: Profitable scenario
- **Zero**: Break-even
- **Negative**: Loss-making scenario

## 🏆 Hackathon Demo Flow

1. **Start with Excel View**: Show basic spreadsheet model
2. **Switch to Customer View**: Demonstrate unlimited experience
3. **Show Financial View**: Highlight profitability analysis
4. **Enable SCO**: Demonstrate cost optimization
5. **Run Optimization**: Show AI-powered recommendations
6. **AI Insights**: Display forecasting capabilities
7. **Compare Scenarios**: Show traditional vs advanced approaches

## 🔒 Security

- **Password Protection**: App-level authentication
- **Session Management**: Secure logout functionality
- **Client-Side Security**: Suitable for hackathon demonstrations

## 📞 Support

For questions or issues during the hackathon:
- Check the Help & User Guide in the app
- Review the calculation formulas
- Use predefined scenarios as starting points
- Enable AI Narratives for contextual insights

## 🎉 Ready to Demo!

The Margin Impact Simulator is now ready for your hackathon presentation. The password protection ensures only authorized users can access the application, while the comprehensive feature set demonstrates advanced analytics capabilities beyond traditional Excel models.

**Happy Hacking! 🚀**