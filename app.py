"""
Margin Impact Simulator - Enhanced Main Streamlit Application
A comprehensive tool for analyzing the financial impact of different policy and pricing scenarios.
Enhanced with dual-audience design, formula transparency, and advanced visualizations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import io
from datetime import datetime
from model import calculate_metrics, optimize_pricing_and_caps, calculate_sco_metrics, compare_sco_vs_traditional
from scenarios import get_scenario, get_all_scenarios
from ai_narratives import generate_ai_narrative, generate_optimization_narrative, get_ai_narrative_status
from ai_insights import generate_trend_analysis, generate_forecasting_insights, create_forecast_chart, get_ai_insights_status

# Page configuration
st.set_page_config(
    page_title="Margin Impact Simulator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .scenario-card {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .customer-view {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2ca02c;
    }
    .internal-view {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

def check_password():
    """Returns `True` if the user had the correct password."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == "hackathon2025":
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated initially.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Enter password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False

def load_synthetic_data():
    """Load synthetic data from CSV file."""
    try:
        return pd.read_csv('data/synthetic.csv')
    except FileNotFoundError:
        st.error("Synthetic data file not found. Please ensure data/synthetic.csv exists.")
        return None

def create_bar_chart(carrier_cost, revenue):
    """Create a bar chart comparing carrier cost vs revenue."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    categories = ['Carrier Cost', 'Revenue']
    values = [carrier_cost, revenue]
    colors = ['#ff7f0e', '#2ca02c']
    
    bars = ax.bar(categories, values, color=colors, alpha=0.7)
    ax.set_ylabel('Amount ($)')
    ax.set_title('Carrier Cost vs Revenue')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                f'${value:,.0f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_pie_chart(coverage):
    """Create a pie chart showing coverage percentage."""
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Handle negative coverage values
    covered = max(0, coverage)  # Ensure covered is not negative
    uncovered = max(0, 100 - coverage)  # Ensure uncovered is not negative
    
    # If both values are 0, show a single slice indicating no coverage
    if covered == 0 and uncovered == 0:
        sizes = [100]
        labels = ['No Coverage Data']
        colors = ['#ff7f0e']
        explode = (0,)
    else:
        sizes = [covered, uncovered]
        labels = [f'Covered ({covered:.1f}%)', f'Uncovered ({uncovered:.1f}%)']
        colors = ['#2ca02c', '#d62728']
        explode = (0.05, 0)  # Explode the covered slice slightly
    
    wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                                     autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
    
    ax.set_title('Line Coverage', fontsize=14, fontweight='bold')
    
    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)
    
    plt.tight_layout()
    return fig

def create_scatter_plot(data):
    """Create a scatter plot of Margin % vs Coverage % with sweet spot highlighting."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create scatter plot
    scatter = ax.scatter(data['coverage'], data['margin'] * 100, 
                        c=data['margin'], cmap='RdYlGn', alpha=0.7, s=100)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Margin %', rotation=270, labelpad=20)
    
    # Highlight sweet spot (high coverage, high margin)
    sweet_spot = data[(data['coverage'] > 60) & (data['margin'] > 0.1)]
    if not sweet_spot.empty:
        ax.scatter(sweet_spot['coverage'], sweet_spot['margin'] * 100, 
                  c='gold', s=150, marker='*', edgecolors='black', linewidth=2, 
                  label='Sweet Spot (High Coverage + High Margin)')
        ax.legend()
    
    # Add quadrant lines
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.axvline(x=50, color='black', linestyle='--', alpha=0.5)
    
    # Add quadrant labels
    ax.text(75, 15, 'Sweet Spot\n(High Coverage\nHigh Margin)', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
            ha='center', va='center', fontsize=10)
    ax.text(25, 15, 'High Margin\nLow Coverage', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7),
            ha='center', va='center', fontsize=10)
    ax.text(75, -15, 'High Coverage\nLow Margin', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7),
            ha='center', va='center', fontsize=10)
    ax.text(25, -15, 'Low Coverage\nLow Margin', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7),
            ha='center', va='center', fontsize=10)
    
    ax.set_xlabel('Coverage %')
    ax.set_ylabel('Margin %')
    ax.set_title('Margin vs Coverage Analysis\n(Historical Data)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_optimization_comparison_chart(current_metrics, optimized_metrics):
    """Create a comparison chart specifically for optimization results."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = ['Carrier Cost', 'Revenue', 'Margin %', 'Coverage %']
    current_values = [
        current_metrics['carrier_cost'],
        current_metrics['revenue'],
        current_metrics['margin'] * 100,
        current_metrics['coverage']
    ]
    optimized_values = [
        optimized_metrics['carrier_cost'],
        optimized_metrics['revenue'],
        optimized_metrics['margin'] * 100,
        optimized_metrics['coverage']
    ]
    
    x = np.arange(len(categories))
    width = 0.35
    
    # Color bars based on margin (green if positive, red if negative)
    current_color = '#2ca02c' if current_metrics['margin'] >= 0 else '#d62728'
    optimized_color = '#2ca02c' if optimized_metrics['margin'] >= 0 else '#d62728'
    
    bars1 = ax.bar(x - width/2, current_values, width, label='Current', color=current_color, alpha=0.7)
    bars2 = ax.bar(x + width/2, optimized_values, width, label='Optimized', color=optimized_color, alpha=0.7)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if bar.get_x() + bar.get_width()/2 < 2:  # For cost and revenue
                ax.text(bar.get_x() + bar.get_width()/2., height + max(max(current_values[:2]), max(optimized_values[:2]))*0.01,
                        f'${height:,.0f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
            else:  # For percentage values
                ax.text(bar.get_x() + bar.get_width()/2., height + max(max(current_values[2:]), max(optimized_values[2:]))*0.01,
                        f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    ax.set_ylabel('Value')
    ax.set_title('🎯 Optimization Results: Current vs Recommended')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add improvement indicators
    margin_improvement = optimized_metrics['margin'] - current_metrics['margin']
    coverage_improvement = optimized_metrics['coverage'] - current_metrics['coverage']
    
    if margin_improvement > 0:
        ax.text(0.02, 0.98, f'📈 Margin improvement: {margin_improvement:.1%}', 
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    if coverage_improvement > 0:
        ax.text(0.02, 0.88, f'📈 Coverage improvement: {coverage_improvement:.1f}%', 
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    plt.tight_layout()
    return fig

def display_help_guide():
    """Display comprehensive help guide with user guide and calculation formulas."""
    with st.expander("📚 Help & User Guide", expanded=False):
        st.markdown("""
        **The Margin Impact Simulator is a decision-support tool that quantifies how service policies affect both customer reach and business margins.**
        
        ## 🚀 Getting Started
        
        Analyze connectivity service performance across different scenarios and stakeholder perspectives.
        
        ### Quick Start:
        1. **Load a Scenario**: Use the sidebar to select "Profitable Example" or "Unprofitable Example"
        2. **Adjust Parameters**: Modify lines, data caps, pricing, or business types
        3. **View Results**: Switch between tabs to see different perspectives
        4. **Get Insights**: Use AI features for advanced analysis
        
        ## 📱 Understanding the Tabs
        
        ### Customer View - "What customers see: Coverage and reach"
        - **Purpose**: Show service coverage and reach distribution
        - **Key Metric**: Coverage % (e.g., "87.3% - 174 out of 200 lines")
        - **Visualization**: Pie chart of connected vs. unconnected lines
        - **AI Narrative**: Customer-oriented narratives
        - **Best For**: Customer presentations, reach discussions
        
        ### Financial View - "What stakeholders see: Financial performance"
        - **Purpose**: Analyze profitability and operational efficiency
        - **Key Metrics**: Revenue, Carrier Cost, Margin %, Budget utilization
        - **Visualization**: Bar charts comparing costs and revenue
        - **AI Narrative**: Financial performance insights
        - **Best For**: Internal reporting, financial planning
        
        ### AI Insights - "AI-powered forecasting and analysis" (NEW!)
        - **Purpose**: Get strategic insights and predict future performance
        - **Features**: 
          - **Trend Analysis**: AI compares current vs. historical performance
          - **Forecasting**: Machine learning predictions (3-12 periods ahead)
          - **Advanced Analytics**: Correlation analysis, performance distributions
        - **Requirements**: Ollama for full AI features, templates as fallback
        - **Best For**: Strategic planning, risk assessment, competitive analysis
        
        ### SCO Analysis - "Smart Cost Optimization comparison" (NEW!)
        - **Purpose**: Compare SCO-enabled vs non-SCO-enabled plans
        - **Features**: 
          - **Cost Analysis**: Detailed breakdown of SCO vs non-SCO costs
          - **ROI Calculator**: Return on investment for SCO implementation
          - **Customer Experience**: What customers see vs internal reality
          - **Visualizations**: Cost comparisons and usage breakdowns
        - **Best For**: Pre-sales demonstrations, cost optimization, competitive differentiation
        
        ### Excel View - "Traditional spreadsheet model" (NEW!)
        - **Purpose**: Show basic Excel-style analysis for comparison
        - **Features**: 
          - **Basic Financial Model**: Simple parameter table with formulas
          - **Basic Charts**: Traditional bar and pie charts
          - **Scenario Comparison**: Simple side-by-side comparison
          - **Limitations Highlight**: Shows what traditional models miss
        - **Best For**: Demonstrating the difference between basic and advanced solutions
        
        ### Historical Analysis - "Advanced data visualization"
        - **Purpose**: Analyze trends and export data
        - **Features**: Scatter plots, sweet spot analysis, data export, custom scenario management
        - **Best For**: Trend analysis, data export, research, scenario management
        
        ## ⚙️ Configuration Guide
        
        ### Sidebar Controls:
        - **Number of Lines**: Total lines in your service
        - **Data Cap per Line**: GB allowance per line (what customer sees)
        - **Budget**: Total budget available
        - **Carrier Rate**: Cost per GB from carrier
        - **Monthly Price per Line**: Fixed monthly fee charged per line
        - **Vertical**: Business type (Public Sector, Retail, Enterprise)
        - **Throttling**: Whether speed limitations are applied
        - **Enable SCO**: Toggle Smart Cost Optimization
        - **Base Plan Size**: Internal base plan (smaller than customer cap)
        - **SCO Efficiency**: How well SCO prevents overages (50-95%)
        - **Overage Rate**: Cost per GB for overage charges
        - **Plan Switching Cost**: Cost per line when switching plans
        
        ### Predefined Scenarios:
        - **✅ Profitable Example**: High margin, good coverage scenario
        - **❌ Unprofitable Example**: Low margin, budget-constrained scenario
        - **High Volume**: Large number of lines
        - **Premium**: High-end pricing scenario
        - **Budget**: Cost-optimized scenario
        - **🎯 SCO-Enabled Enterprise**: Enterprise with intelligent cost optimization
        - **🎯 SCO-Enabled Retail**: Retail households with SCO optimization
        - **📊 Traditional Static Plans**: Fixed plans with overage charges (no SCO)
        
        ## 🤖 AI Features Guide
        
        ### AI Narratives:
        - **Enable/Disable**: Use sidebar checkbox
        - **Customer View**: Customer-oriented narratives
        - **Financial View**: Financial performance insights
        - **Optimization**: AI explanation of recommended changes
        
        ### AI Options (Priority Order):
        1. **Ollama (Local)**: Free, private, runs on your machine
        2. **Google AI (Cloud)**: Enhanced cloud-based AI via Gemini API
        3. **Smart Templates**: Fallback when AI unavailable
        
        ### Ollama Setup (Optional):
        1. Install Ollama: https://ollama.ai/
        2. Pull model: `ollama pull llama3.2`
        3. Start server: `ollama serve`
        4. Enable AI Narratives in sidebar
        
        ### Google AI Setup (Cloud Only):
        - Set `GOOGLE_AI_API_KEY` in Streamlit secrets
        - Automatically used when Ollama unavailable
        - Enhanced cloud-based AI narratives
        
        ### Fallback Templates:
        - Smart templates when AI unavailable
        - Still contextual and insightful
        - No setup required
        
        ## 📊 Understanding Results
        
        ### Coverage:
        - **100%**: Full budget utilization, serving all lines at capacity
        - **85%+**: Excellent coverage, most lines connected
        - **50-85%**: Good coverage, room for improvement
        - **<50%**: Limited coverage, significant expansion opportunity
        
        ### Margin:
        - **Positive**: Profitable scenario
        - **Zero**: Break-even
        - **Negative**: Loss-making scenario
        
        ### Optimization Results:
        - **Feasible**: Solution found within constraints
        - **Infeasible**: No solution meets all constraints
        - **Alternatives**: Top 5 alternative solutions
        
        ### Optimization Features:
        - **Pricing & Data Cap Optimization** (Financial View): Uses AI to find optimal pricing and data caps with comprehensive comparison charts and analysis
        
        ## 📐 Calculation Formulas
        
        **Usage Calculation:**
        ```
        Usage = Lines × Data Cap × Vertical Factor × Efficiency Factor
        ```
        - Vertical Factor: 1.0 (Public Sector), 0.8 (Retail), 0.9 (Enterprise)
        - Efficiency Factor: 0.9 (Throttling On), 1.0 (Throttling Off)
        
        **Carrier Cost:**
        ```
        Carrier Cost = Usage × Carrier Rate
        ```
        
        **Revenue:**
        ```
        Revenue = min(Budget, Lines × Monthly Price per Line)
        ```
        
        **Margin:**
        ```
        Margin = (Revenue - Carrier Cost) / Revenue
        ```
        
        **Coverage:**
        ```
        Coverage = (Revenue / (Lines × Monthly Price per Line)) × 100
        ```
        
        *Coverage % indicates how much of the maximum potential revenue can be realized within budget. 100% = full capacity; lower values reflect budget constraints limiting service reach.*
        
        ### Vertical Impact
        - **Public Sector (Schools)**: Each line gets full data allowance (100% usage)
        - **Retail (Households)**: Data is shared among household members (20% reduction in usage)
        - **Enterprise**: Corporate usage patterns with moderate efficiency (10% reduction in usage)
        - **Throttling**: Reduces actual usage by 10% due to speed limitations
        
        ### SCO (Smart Cost Optimization) Formulas:
        
        **SCO Usage Calculation:**
        ```
        Base Usage = Lines × Base Plan Size × Vertical Factor × Efficiency Factor
        Potential Overage = Lines × (Customer Cap - Base Plan Size) × Vertical Factor × Efficiency Factor
        Actual Overage = Potential Overage × (1 - SCO Efficiency)
        Total Usage = Base Usage + Actual Overage
        ```
        
        **SCO Cost Calculation:**
        ```
        Base Carrier Cost = Base Usage × Carrier Rate
        Overage Cost = Actual Overage × Overage Rate
        Switching Cost = Lines × Plan Switching Cost × Switching Frequency
        SCO Operational Cost = Lines × $0.50 (monthly SCO service)
        Total Carrier Cost = Base Carrier Cost + Overage Cost + Switching Cost + SCO Operational Cost
        ```
        
        **SCO Benefits:**
        ```
        Overage Savings = Potential Overage - Actual Overage
        Cost Savings = Overage Savings × Overage Rate
        SCO ROI = (Cost Savings - SCO Operational Cost - Switching Cost) / (SCO Operational Cost + Switching Cost)
        ```
        
        ## 💡 Best Practices
        
        ### Key Differentiators (vs. Excel models):
        - **Dual-Audience Views**: Customer vs. Financial perspectives
        - **SCO Analysis**: Smart Cost Optimization comparison and ROI analysis
        - **Optimization Engine**: AI-powered recommendations under constraints
        - **AI Insights**: Forecasting, correlations, advanced analytics
        - **Export & Scenario Management**: Professional polish and data portability
        
        ### For Demos:
        1. Start with "SCO-Enabled Enterprise" scenario
        2. Show Excel View first (basic spreadsheet model)
        3. Switch to Customer View (unlimited experience)
        4. Switch to Financial View (profitability focus)
        5. Demonstrate SCO Analysis tab (cost savings and ROI)
        6. Run "Find Optimal Pricing" to get optimization results
        7. Show AI Insights (trend analysis)
        8. Compare with "Traditional Static Plans" scenario
        
        ### For Analysis:
        1. Load multiple scenarios for comparison
        2. Use Historical Analysis for trend identification
        3. Apply AI Insights for strategic recommendations
        4. Export data for further analysis
        
        ### For Different Audiences:
        - **Customers**: Focus on Customer View (unlimited experience, no overages)
        - **Executives**: Emphasize Financial View (margins, ROI) and SCO Analysis
        - **Sales Teams**: Use SCO Analysis for competitive differentiation
        - **Analysts**: Use Historical Analysis and AI Insights
        - **Operations**: Show optimization and scenario management
        - **Decision Makers**: Start with Excel View, then show advanced features
        
        ## 🔧 Troubleshooting
        
        ### Common Issues:
        - **No AI Narratives**: Check if Ollama is running (`ollama serve`) or Google AI API key is set
        - **Slow Performance**: Try smaller models or disable AI features
        - **Missing Data**: Use Historical Analysis tab to generate synthetic data
        - **Optimization Fails**: Relax constraints (lower minimum coverage/margin)
        - **Google AI Not Working**: Verify API key in Streamlit secrets
        
        ### Getting Help:
        - Check the sidebar status indicators
        - Review the calculation formulas above
        - Use predefined scenarios as starting points
        - Enable AI Narratives for contextual insights
        """)

def save_custom_scenario(scenario_data, name):
    """Save a custom scenario to session state."""
    if 'custom_scenarios' not in st.session_state:
        st.session_state.custom_scenarios = {}
    
    st.session_state.custom_scenarios[name] = {
        'name': name,
        'description': f'Custom scenario saved on {datetime.now().strftime("%Y-%m-%d %H:%M")}',
        **scenario_data
    }

def export_scenario_summary(metrics, students, cap, budget, carrier_rate, customer_price, policy, throttling):
    """Export current scenario as CSV."""
    summary_data = {
        'Parameter': [
            'Students', 'Data Cap (GB)', 'Budget ($)', 'Carrier Rate ($/GB)', 
            'Monthly Price per Line ($)', 'Policy', 'Throttling', 'Usage (GB)',
            'Carrier Cost ($)', 'Revenue ($)', 'Margin (%)', 'Coverage (%)'
        ],
        'Value': [
            students, cap, budget, carrier_rate, customer_price, policy, 
            'On' if throttling else 'Off', metrics['usage'], metrics['carrier_cost'],
            metrics['revenue'], f"{metrics['margin']:.1%}", f"{metrics['coverage']:.1f}%"
        ]
    }
    
    df = pd.DataFrame(summary_data)
    return df.to_csv(index=False)

def main():
    # Password protection
    if not check_password():
        st.stop()  # Do not continue if check_password is not True.
    
    # Header
    st.markdown('<h1 class="main-header">📊 Margin Impact Simulator</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Help & User Guide
    display_help_guide()
    
    # Sidebar for inputs
    st.sidebar.header("🎛️ Configuration")
    
    
    # Load scenarios
    scenarios = get_all_scenarios()
    
    # Scenario selection with improved labeling
    st.sidebar.subheader("📋 Load Predefined Scenarios")
    
    # Quick scenario buttons
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("✅ Profitable Example", help="Load profitable scenario", key="sidebar_profitable_example"):
            st.session_state.current_scenario = scenarios['profitable']
            # Update SCO toggle to match scenario setting
            if 'sco_enabled' in scenarios['profitable']:
                st.session_state.sco_enabled = scenarios['profitable']['sco_enabled']
            # Update policy to match scenario setting
            if 'policy' in scenarios['profitable']:
                st.session_state.policy = scenarios['profitable']['policy']
            # Update SCO parameters to match scenario settings
            if 'base_plan_gb' in scenarios['profitable']:
                st.session_state.base_plan_gb = scenarios['profitable']['base_plan_gb']
            if 'sco_efficiency' in scenarios['profitable']:
                st.session_state.sco_efficiency = scenarios['profitable']['sco_efficiency']
            if 'overage_rate' in scenarios['profitable']:
                st.session_state.overage_rate = scenarios['profitable']['overage_rate']
            if 'plan_switching_cost' in scenarios['profitable']:
                st.session_state.plan_switching_cost = scenarios['profitable']['plan_switching_cost']
            if 'monthly_usage_per_line' in scenarios['profitable']:
                st.session_state.monthly_usage_per_line = scenarios['profitable']['monthly_usage_per_line']
            # Trigger rerun to update the UI
            st.experimental_rerun()
    
    with col2:
        if st.button("❌ Unprofitable Example", help="Load unprofitable scenario", key="sidebar_unprofitable_example"):
            st.session_state.current_scenario = scenarios['unprofitable']
            # Update SCO toggle to match scenario setting
            if 'sco_enabled' in scenarios['unprofitable']:
                st.session_state.sco_enabled = scenarios['unprofitable']['sco_enabled']
            # Update policy to match scenario setting
            if 'policy' in scenarios['unprofitable']:
                st.session_state.policy = scenarios['unprofitable']['policy']
            # Update SCO parameters to match scenario settings
            if 'base_plan_gb' in scenarios['unprofitable']:
                st.session_state.base_plan_gb = scenarios['unprofitable']['base_plan_gb']
            if 'sco_efficiency' in scenarios['unprofitable']:
                st.session_state.sco_efficiency = scenarios['unprofitable']['sco_efficiency']
            if 'overage_rate' in scenarios['unprofitable']:
                st.session_state.overage_rate = scenarios['unprofitable']['overage_rate']
            if 'plan_switching_cost' in scenarios['unprofitable']:
                st.session_state.plan_switching_cost = scenarios['unprofitable']['plan_switching_cost']
            if 'monthly_usage_per_line' in scenarios['unprofitable']:
                st.session_state.monthly_usage_per_line = scenarios['unprofitable']['monthly_usage_per_line']
            # Trigger rerun to update the UI
            st.experimental_rerun()
    
    # Dropdown for all scenarios
    scenario_options = {f"{k}: {v['name']}": k for k, v in scenarios.items()}
    selected_scenario_key = st.sidebar.selectbox(
        "Choose a scenario:",
        options=list(scenario_options.keys()),
        index=0,
        key="scenario_selectbox"
    )
    
    # Auto-load scenario when dropdown selection changes
    if 'last_selected_scenario' not in st.session_state:
        st.session_state.last_selected_scenario = selected_scenario_key
    
    if st.session_state.last_selected_scenario != selected_scenario_key:
        scenario_key = scenario_options[selected_scenario_key]
        scenario = get_scenario(scenario_key)
        st.session_state.current_scenario = scenario
        st.session_state.last_selected_scenario = selected_scenario_key
        
        # Update SCO toggle to match scenario setting
        if 'sco_enabled' in scenario:
            st.session_state.sco_enabled = scenario['sco_enabled']
        
        # Update policy to match scenario setting
        if 'policy' in scenario:
            st.session_state.policy = scenario['policy']
        
        # Update SCO parameters to match scenario settings
        if 'base_plan_gb' in scenario:
            st.session_state.base_plan_gb = scenario['base_plan_gb']
        if 'sco_efficiency' in scenario:
            st.session_state.sco_efficiency = scenario['sco_efficiency']
        if 'overage_rate' in scenario:
            st.session_state.overage_rate = scenario['overage_rate']
        if 'plan_switching_cost' in scenario:
            st.session_state.plan_switching_cost = scenario['plan_switching_cost']
        if 'monthly_usage_per_line' in scenario:
            st.session_state.monthly_usage_per_line = scenario['monthly_usage_per_line']
        
        # Trigger rerun to update the UI
        st.experimental_rerun()
    
    # Manual input controls
    st.sidebar.subheader("⚙️ Manual Configuration")
    
    # Initialize session state if not exists
    if 'current_scenario' not in st.session_state:
        st.session_state.current_scenario = scenarios['profitable']
    
    current = st.session_state.current_scenario
    
    # Input controls
    students = st.sidebar.slider(
        "Number of Lines",
        min_value=100,
        max_value=1000,
        value=current['students'],
        step=50,
        key="students_slider"
    )
    
    cap = st.sidebar.slider(
        "Data Cap per Line (GB)",
        min_value=1.0,
        max_value=10.0,
        value=current['cap'],
        step=0.5,
        key="cap_slider"
    )
    
    budget = st.sidebar.slider(
        "Budget ($)",
        min_value=5000.0,
        max_value=20000.0,
        value=current['budget'],
        step=500.0,
        key="budget_slider"
    )
    
    carrier_rate = st.sidebar.number_input(
        "Carrier Rate ($/GB)",
        min_value=1.0,
        max_value=20.0,
        value=current['carrier_rate'],
        step=0.5,
        key="carrier_rate_input"
    )
    
    customer_price = st.sidebar.number_input(
        "Monthly Price per Line ($)",
        min_value=10.0,
        max_value=100.0,
        value=current['customer_price'],
        step=5.0,
        key="customer_price_input"
    )
    
    # Initialize policy session state if not exists
    if 'policy' not in st.session_state:
        st.session_state.policy = current['policy']
    
    # Get current policy index
    policy_options = ["Public Sector (Schools)", "Retail (Households)", "Enterprise"]
    current_policy_index = policy_options.index(st.session_state.policy) if st.session_state.policy in policy_options else 0
    
    policy = st.sidebar.selectbox(
        "Vertical",
        options=policy_options,
        index=current_policy_index,
        key="policy_selectbox"
    )
    
    # Update session state when policy changes
    st.session_state.policy = policy
    
    throttling = st.sidebar.checkbox(
        "Throttling",
        value=current['throttling'],
        key="throttling_checkbox"
    )
    
    # SCO (Smart Cost Optimization) section
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Smart Cost Optimization (SCO)")
    
    # SCO toggle (persistent in session state)
    if 'sco_enabled' not in st.session_state:
        st.session_state.sco_enabled = current.get('sco_enabled', False)
    
    sco_enabled = st.sidebar.checkbox(
        "Enable SCO",
        value=st.session_state.sco_enabled,
        help="Enable intelligent cost optimization to reduce overage costs",
        key="sco_toggle"
    )
    
    # Update session state when toggle changes
    st.session_state.sco_enabled = sco_enabled
    
    if sco_enabled:
        # Initialize SCO parameters in session state
        if 'base_plan_gb' not in st.session_state:
            st.session_state.base_plan_gb = current.get('base_plan_gb', 3.0)
        if 'sco_efficiency' not in st.session_state:
            st.session_state.sco_efficiency = current.get('sco_efficiency', 0.85)
        if 'overage_rate' not in st.session_state:
            st.session_state.overage_rate = current.get('overage_rate', 15.0)
        if 'plan_switching_cost' not in st.session_state:
            st.session_state.plan_switching_cost = current.get('plan_switching_cost', 0.5)
        if 'monthly_usage_per_line' not in st.session_state:
            st.session_state.monthly_usage_per_line = current.get('monthly_usage_per_line', 2.5)
        
        # SCO-specific parameters
        base_plan_gb = st.sidebar.slider(
            "Base Plan Size (GB)",
            min_value=1.0,
            max_value=8.0,
            value=st.session_state.base_plan_gb,
            step=0.5,
            help="Internal base plan size (smaller than customer cap)",
            key="base_plan_gb_slider"
        )
        
        sco_efficiency = st.sidebar.slider(
            "SCO Efficiency (%)",
            min_value=50,
            max_value=95,
            value=int(st.session_state.sco_efficiency * 100),
            step=5,
            help="How well SCO prevents overages (85% = prevents 85% of potential overages)",
            key="sco_efficiency_slider"
        ) / 100.0
        
        overage_rate = st.sidebar.number_input(
            "Overage Rate ($/GB)",
            min_value=5.0,
            max_value=25.0,
            value=st.session_state.overage_rate,
            step=1.0,
            help="Cost per GB for overage charges",
            key="overage_rate_input"
        )
        
        plan_switching_cost = st.sidebar.number_input(
            "Plan Switching Cost ($/line)",
            min_value=0.1,
            max_value=2.0,
            value=st.session_state.plan_switching_cost,
            step=0.1,
            help="Cost per line when switching plans (lower = more efficient SCO)",
            key="plan_switching_cost_input"
        )
        
        monthly_usage_per_line = st.sidebar.number_input(
            "Monthly Usage per Line (GB)",
            min_value=0.5,
            max_value=10.0,
            value=st.session_state.monthly_usage_per_line,
            step=0.1,
            help="Average monthly data consumption per line (realistic usage pattern)",
            key="monthly_usage_per_line_input"
        )
        
        # Update session state when SCO parameters change
        st.session_state.base_plan_gb = base_plan_gb
        st.session_state.sco_efficiency = sco_efficiency
        st.session_state.overage_rate = overage_rate
        st.session_state.plan_switching_cost = plan_switching_cost
        st.session_state.monthly_usage_per_line = monthly_usage_per_line
    else:
        # Default values when SCO is disabled
        base_plan_gb = cap  # Same as customer cap
        sco_efficiency = 0.0
        overage_rate = 15.0
        plan_switching_cost = 0.0
        monthly_usage_per_line = 2.5  # Default realistic usage
    
    # AI Narratives section
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 AI Narratives")
    
    # AI narrative toggle (persistent in session state)
    if 'enable_ai_narratives' not in st.session_state:
        st.session_state.enable_ai_narratives = False
    
    enable_ai_narratives = st.sidebar.checkbox(
        "Enable AI Narratives",
        value=st.session_state.enable_ai_narratives,
        help="Generate AI-powered contextual narratives for better insights",
        key="ai_narrative_toggle"
    )
    
    # Update session state when toggle changes
    st.session_state.enable_ai_narratives = enable_ai_narratives
    
    # Show AI status
    ai_status = get_ai_narrative_status()
    
    # Display status with debug info
    if ai_status['ollama_available'] and ai_status['google_ai_available']:
        st.sidebar.success("✅ Ollama + Google AI Available")
    elif ai_status['ollama_available']:
        st.sidebar.success("✅ Ollama (Local AI) Available")
    elif ai_status['google_ai_available']:
        st.sidebar.success("✅ Google AI (Cloud) Available")
    else:
        st.sidebar.info("📝 Using Smart Templates")
        st.sidebar.caption("No AI services available - using fallback templates")
    
    # AI Status (simplified)
    with st.sidebar.expander("🤖 AI Status", expanded=False):
        st.write("**Status:**", ai_status['status'])
        st.write("**Ollama:**", "✅ Available" if ai_status['ollama_available'] else "❌ Not Available")
        st.write("**Google AI:**", "✅ Available" if ai_status['google_ai_available'] else "❌ Not Available")
    
    # Optimization section
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Pricing & Data Cap Optimization")
    
    st.sidebar.markdown("**Optimize your pricing and data caps for better margins:**")
    
    # Initialize optimization settings in session state
    if 'min_coverage' not in st.session_state:
        st.session_state.min_coverage = 80.0
    if 'min_margin' not in st.session_state:
        st.session_state.min_margin = 0.0
    
    min_coverage = st.sidebar.slider(
        "Minimum Coverage (%)",
        min_value=0.0,
        max_value=100.0,
        value=st.session_state.min_coverage,
        step=5.0,
        help="Minimum student coverage percentage required",
        key="min_coverage_slider"
    )
    
    min_margin = st.sidebar.slider(
        "Minimum Margin (%)",
        min_value=0.0,
        max_value=50.0,
        value=st.session_state.min_margin,
        step=1.0,
        help="Minimum profit margin required (0% = break-even)",
        key="min_margin_slider"
    )
    
    # Update session state when sliders change
    st.session_state.min_coverage = min_coverage
    st.session_state.min_margin = min_margin
    
    if st.sidebar.button("🚀 Find Optimal Pricing", type="primary"):
        with st.spinner("🔍 Analyzing pricing and data cap combinations..."):
            optimization_result = optimize_pricing_and_caps(
                students, cap, budget, carrier_rate, 
                customer_price, policy, throttling, min_coverage, min_margin/100,
                sco_enabled=sco_enabled, base_plan_gb=base_plan_gb, 
                sco_efficiency=sco_efficiency, overage_rate=overage_rate, 
                plan_switching_cost=plan_switching_cost, monthly_usage_per_line=monthly_usage_per_line
            )
            st.session_state.optimization_result = optimization_result
            # Set flag to show completion notification
            st.session_state.optimization_completed = True
            if optimization_result['feasible']:
                st.sidebar.success("✅ Optimal pricing found! Check the Financial View tab for results.")
            else:
                st.sidebar.error("❌ No feasible solution found. Try relaxing constraints.")
            # Force rerun to show notification
            # Trigger rerun to update the UI
            st.experimental_rerun()
    
    # Calculate metrics (use SCO if enabled)
    if sco_enabled:
        metrics = calculate_sco_metrics(
            students, base_plan_gb, cap, budget, carrier_rate, 
            customer_price, policy, throttling, sco_enabled=True,
            sco_efficiency=sco_efficiency, overage_rate=overage_rate, 
            plan_switching_cost=plan_switching_cost, monthly_usage_per_line=monthly_usage_per_line
        )
    else:
        metrics = calculate_metrics(
            students, cap, budget, carrier_rate, 
            customer_price, policy, throttling, monthly_usage_per_line,
            base_plan_gb, overage_rate
        )
    
    # Help guide is now displayed at the top of the page
    
    # Show notification if optimization was just completed
    if st.session_state.get('optimization_completed', False):
        st.session_state.optimization_completed = False
        st.success("🎯 **Optimization Complete!** Check the **Financial View** tab below to see the results and comparison charts.")
    
    # Main content area with dual-audience design using native tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📱 Customer View", "💰 Financial View", "🎯 SCO Analysis", "🤖 AI Insights", "📊 Excel View", "📊 Historical Analysis"])
    
    with tab1:
        st.header("📱 Customer View")
        st.markdown("### What customers see: Service coverage and connectivity experience")
        
        # Customer-focused metrics and visualizations
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="customer-view">', unsafe_allow_html=True)
            
            # Key customer metrics
            st.subheader("🎯 Service Coverage Overview")
            
            # Main coverage metric with enhanced display
            coverage_col1, coverage_col2, coverage_col3 = st.columns(3)
            
            with coverage_col1:
                st.metric(
                    label="📶 Lines Connected",
                    value=f"{int(students * metrics['coverage'] / 100)}",
                    delta=f"out of {students} total lines"
                )
            
            with coverage_col2:
                st.metric(
                    label="📊 Coverage Rate",
                    value=f"{metrics['coverage']:.1f}%",
                    delta="service availability"
                )
            
            with coverage_col3:
                # Calculate data allowance per line
                data_per_line = cap if not sco_enabled else cap  # Customer sees the full cap
                st.metric(
                    label="💾 Data Allowance",
                    value=f"{data_per_line:.1f} GB",
                    delta="per line per month"
                )
            
            # Service quality indicators
            st.subheader("⭐ Service Quality Indicators")
            
            quality_col1, quality_col2, quality_col3 = st.columns(3)
            
            with quality_col1:
                # Coverage quality
                if metrics['coverage'] >= 90:
                    st.success("🟢 **Excellent Coverage**\n\nPremium service level")
                elif metrics['coverage'] >= 70:
                    st.warning("🟡 **Good Coverage**\n\nStandard service level")
                else:
                    st.error("🔴 **Limited Coverage**\n\nNeeds improvement")
            
            with quality_col2:
                # Data allowance quality
                if cap >= 5.0:
                    st.success("🟢 **Generous Data**\n\n5GB+ per line")
                elif cap >= 3.0:
                    st.warning("🟡 **Moderate Data**\n\n3-4.9GB per line")
                else:
                    st.error("🔴 **Limited Data**\n\n<3GB per line")
            
            with quality_col3:
                # Service type quality
                if policy == "Public Sector (Schools)":
                    st.info("🏫 **Educational Focus**\n\nFull data allowance")
                elif policy == "Enterprise":
                    st.info("🏢 **Business Grade**\n\nOptimized for productivity")
                else:
                    st.info("🏠 **Household Service**\n\nShared data plans")
            
            # AI-powered customer narrative
            st.subheader("💬 Service Summary")
            if enable_ai_narratives:
                with st.spinner("🤖 Generating AI narrative..."):
                    context = {
                        'students': students,
                        'cap': cap,
                        'budget': budget,
                        'carrier_rate': carrier_rate,
                        'customer_price': customer_price,
                        'policy': policy,
                        'throttling': throttling
                    }
                    ai_narrative = generate_ai_narrative(metrics, context, "customer")
                    
                    # Show which AI mode is being used
                    if ai_status['ollama_available']:
                        ai_label = "🤖 **AI-Generated** (Ollama)"
                    elif ai_status['google_ai_available']:
                        ai_label = "🤖 **AI-Generated** (Google AI)"
                    else:
                        ai_label = "📝 **Smart Template**"
                    
                    # Display using HTML to avoid browser rendering issues
                    st.markdown(f"""
                    <div style="
                        background-color: #e3f2fd;
                        border: 1px solid #2196f3;
                        border-radius: 0.5rem;
                        padding: 1rem;
                        margin: 0.5rem 0;
                        font-family: 'Source Sans Pro', sans-serif;
                        font-size: 14px;
                        line-height: 1.5;
                    ">
                        <strong>{ai_label}:</strong><br>
                        {ai_narrative}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                # Enhanced fallback narrative
                connected_lines = int(students * metrics['coverage'] / 100)
                
                if metrics['coverage'] > 90:
                    st.success(f"✅ **Excellent service coverage** - {connected_lines} out of {students} lines connected with reliable connectivity!")
                elif metrics['coverage'] > 70:
                    st.warning(f"⚠️ **Good service coverage** - {connected_lines} out of {students} lines connected, with room for improvement")
                else:
                    st.error(f"❌ **Limited service coverage** - Only {connected_lines} out of {students} lines connected, needs connectivity support")
            
            # Customer benefits section
            st.subheader("🎁 Customer Benefits")
            
            # Calculate connected lines for benefits section
            connected_lines = int(students * metrics['coverage'] / 100)
            
            benefits_col1, benefits_col2 = st.columns(2)
            
            with benefits_col1:
                st.markdown("**📱 What You Get:**")
                st.write(f"• **{data_per_line:.1f}GB** data allowance per line")
                st.write(f"• **{connected_lines} connected lines** out of {students}")
                st.write(f"• **{policy}** service tier")
                if not throttling:
                    st.write("• **Full-speed** connectivity")
                else:
                    st.write("• **Optimized** connectivity with speed management")
            
            with benefits_col2:
                st.markdown("**💡 Service Features:**")
                if sco_enabled:
                    st.write("• **Smart optimization** for cost efficiency")
                    st.write("• **Unlimited experience** for customers")
                    st.write("• **No overage charges**")
                else:
                    st.write("• **Predictable** monthly costs")
                    st.write("• **Reliable** service delivery")
                st.write("• **Professional** support and monitoring")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            # Coverage pie chart
            st.subheader("📊 Service Distribution")
            st.pyplot(create_pie_chart(metrics['coverage']))
            
            # Additional customer insights
            st.subheader("📈 Service Insights")
            
            # Coverage trend indicator
            if metrics['coverage'] >= 80:
                st.success("🎯 **Target Achieved**\n\nCoverage meets or exceeds 80% target")
            elif metrics['coverage'] >= 60:
                st.warning("📈 **Approaching Target**\n\nCoverage is close to 80% target")
            else:
                st.error("📉 **Below Target**\n\nCoverage needs improvement")
            
            # Data efficiency indicator
            if cap >= 4.0:
                st.info("💾 **High Data Allowance**\n\nSufficient for most use cases")
            elif cap >= 2.0:
                st.info("💾 **Moderate Data Allowance**\n\nGood for standard usage")
            else:
                st.info("💾 **Basic Data Allowance**\n\nSuitable for light usage")
            
            # Service reliability indicator
            if not throttling:
                st.success("⚡ **Full Performance**\n\nNo speed limitations")
            else:
                st.info("⚡ **Optimized Performance**\n\nSpeed management enabled")
    
    with tab2:
        st.header("💰 Financial View")
        st.markdown("### What stakeholders see: Financial performance")
        
        # Show optimization results indicator if available
        if 'optimization_result' in st.session_state and st.session_state.optimization_result['feasible']:
            st.success("🎯 **Fresh Optimization Results Available!** Scroll down to see the detailed analysis and comparison charts.")
        
        # Internal-focused metrics and visualizations
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="internal-view">', unsafe_allow_html=True)
            st.metric(
                label="💰 Revenue",
                value=f"${metrics['revenue']:,.0f}",
                delta=f"${metrics['revenue'] - metrics['carrier_cost']:,.0f}"
            )
            
            st.metric(
                label="📊 Margin",
                value=f"{metrics['margin']:.1%}",
                delta=f"{'Profit' if metrics['margin'] > 0 else 'Loss'}"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            # AI-powered financial narrative
            if enable_ai_narratives:
                with st.spinner("🤖 Generating AI analysis..."):
                    context = {
                        'students': students,
                        'cap': cap,
                        'budget': budget,
                        'carrier_rate': carrier_rate,
                        'customer_price': customer_price,
                        'policy': policy,
                        'throttling': throttling
                    }
                    ai_narrative = generate_ai_narrative(metrics, context, "financial")
                    
                    # Show which AI mode is being used
                    if ai_status['ollama_available']:
                        ai_label = "🤖 **AI-Generated** (Ollama)"
                    elif ai_status['google_ai_available']:
                        ai_label = "🤖 **AI-Generated** (Google AI)"
                    else:
                        ai_label = "📝 **Smart Template**"
                    
                    # Display text using HTML to avoid browser rendering issues
                    if metrics['margin'] > 0:
                        st.markdown(f"""
                        <div style="
                            background-color: #e8f5e8;
                            border: 1px solid #4caf50;
                            border-radius: 0.5rem;
                            padding: 1rem;
                            margin: 0.5rem 0;
                            font-family: 'Source Sans Pro', sans-serif;
                            font-size: 14px;
                            line-height: 1.5;
                        ">
                            <strong>{ai_label}:</strong><br>
                            {ai_narrative}
                        </div>
                        """, unsafe_allow_html=True)
                    elif metrics['margin'] == 0:
                        st.markdown(f"""
                        <div style="
                            background-color: #fff3e0;
                            border: 1px solid #ff9800;
                            border-radius: 0.5rem;
                            padding: 1rem;
                            margin: 0.5rem 0;
                            font-family: 'Source Sans Pro', sans-serif;
                            font-size: 14px;
                            line-height: 1.5;
                        ">
                            <strong>{ai_label}:</strong><br>
                            {ai_narrative}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="
                            background-color: #ffebee;
                            border: 1px solid #f44336;
                            border-radius: 0.5rem;
                            padding: 1rem;
                            margin: 0.5rem 0;
                            font-family: 'Source Sans Pro', sans-serif;
                            font-size: 14px;
                            line-height: 1.5;
                        ">
                            <strong>{ai_label}:</strong><br>
                            {ai_narrative}
                        </div>
                        """, unsafe_allow_html=True)
            else:
                # Fallback to static narrative
                if metrics['margin'] > 0:
                    st.success(f"**Margin is {metrics['margin']:.1%}** - Profitable scenario ✅")
                elif metrics['margin'] == 0:
                    st.warning(f"**Margin is {metrics['margin']:.1%}** - Break-even scenario ⚠️")
                else:
                    st.error(f"**Margin is {metrics['margin']:.1%}** - Loss scenario ❌")
        
        with col2:
            # Financial bar chart
            st.pyplot(create_bar_chart(metrics['carrier_cost'], metrics['revenue']))
        
        # Detailed breakdown for internal view
        st.subheader("📋 Detailed Financial Breakdown")
        
        if sco_enabled:
            # SCO-enabled breakdown with additional SCO metrics
            breakdown_data = {
                'Metric': [
                    'Total Usage (GB)', 'Base Plan Usage (GB)', 'Potential Overage (GB)', 'Actual Overage (GB)', 'Overage Savings (GB)',
                    'Total Carrier Cost ($)', 'Base Plan Cost ($)', 'Overage Cost ($)', 'Plan Switching Cost ($)', 'SCO Operational Cost ($)',
                    'Revenue ($)', 'Margin (%)', 'Coverage (%)', 'Budget Utilization (%)', 'Budget Remaining ($)',
                    'SCO Efficiency (%)', 'SCO ROI'
                ],
                'Value': [
                    f"{metrics['usage']:,.1f}",
                    f"{metrics.get('base_usage', 0):,.1f}",
                    f"{metrics.get('potential_overage', 0):,.1f}",
                    f"{metrics.get('actual_overage', 0):,.1f}",
                    f"{metrics.get('overage_savings', 0):,.1f}",
                    f"${metrics['carrier_cost']:,.2f}",
                    f"${metrics.get('base_carrier_cost', 0):,.2f}",
                    f"${metrics.get('overage_carrier_cost', 0):,.2f}",
                    f"${metrics.get('switching_cost', 0):,.2f}",
                    f"${metrics.get('sco_operational_cost', 0):,.2f}",
                    f"${metrics['revenue']:,.2f}",
                    f"{metrics['margin']:.1%}",
                    f"{metrics['coverage']:.1f}%",
                    f"{metrics.get('budget_utilization', 0):.1f}%",
                    f"${metrics.get('budget_remaining', 0):,.2f}",
                    f"{metrics.get('sco_efficiency', 0):.1f}%",
                    f"{metrics.get('sco_savings', {}).get('roi', 0):.1f}x"
                ],
                'Formula': [
                    f"Lines × Monthly Usage per Line × Policy Factor × Efficiency Factor\n= {students} × {monthly_usage_per_line} × {metrics.get('policy_factor', 1.0):.1f} × {metrics.get('efficiency_factor', 1.0):.1f}",
                    f"Lines × Base Plan GB\n= {students} × {base_plan_gb}",
                    f"Total Usage - Base Usage\n= {metrics['usage']:,.1f} - {metrics.get('base_usage', 0):,.1f}",
                    f"Potential Overage × (1 - SCO Efficiency)\n= {metrics.get('potential_overage', 0):,.1f} × (1 - {metrics.get('sco_efficiency', 0.85):.2f})",
                    f"Potential Overage - Actual Overage\n= {metrics.get('potential_overage', 0):,.1f} - {metrics.get('actual_overage', 0):,.1f}",
                    f"Base Cost + Overage Cost + Switching Cost + SCO Operational Cost\n= ${metrics.get('base_carrier_cost', 0):,.2f} + ${metrics.get('overage_carrier_cost', 0):,.2f} + ${metrics.get('switching_cost', 0):,.2f} + ${metrics.get('sco_operational_cost', 0):,.2f}",
                    f"Base Usage × Carrier Rate\n= {metrics.get('base_usage', 0):,.1f} × ${carrier_rate:.2f}",
                    f"Actual Overage × Overage Rate\n= {metrics.get('actual_overage', 0):,.1f} × ${overage_rate:.2f}",
                    f"Lines × Plan Switching Cost\n= {students} × ${plan_switching_cost:.2f}",
                    f"Lines × SCO Operational Cost per Line\n= {students} × $0.20",
                    f"min(Budget, Lines × Monthly Price per Line)\n= min(${budget:,.0f}, {students} × ${customer_price:.2f})",
                    f"(Revenue - Total Carrier Cost) / Revenue\n= (${metrics['revenue']:,.2f} - ${metrics['carrier_cost']:,.2f}) / ${metrics['revenue']:,.2f}",
                    f"Revenue / (Lines × Monthly Price per Line) × 100\n= ${metrics['revenue']:,.2f} / ({students} × ${customer_price:.2f}) × 100",
                    f"Revenue / Budget × 100\n= ${metrics['revenue']:,.2f} / ${budget:,.2f} × 100",
                    f"Budget - Revenue\n= ${budget:,.2f} - ${metrics['revenue']:,.2f}",
                    f"SCO Efficiency Setting\n= {sco_efficiency:.1%}",
                    f"SCO Savings / SCO Costs\n= ${metrics.get('sco_savings', {}).get('total_savings', 0):,.2f} / ${metrics.get('sco_savings', {}).get('total_costs', 1):,.2f}"
                ]
            }
        else:
            # Standard breakdown without SCO metrics
            breakdown_data = {
                'Metric': ['Usage (GB)', 'Carrier Cost ($)', 'Revenue ($)', 'Margin (%)', 'Coverage (%)', 'Budget Utilization (%)', 'Budget Remaining ($)'],
                'Value': [
                    f"{metrics['usage']:,.1f}",
                    f"${metrics['carrier_cost']:,.2f}",
                    f"${metrics['revenue']:,.2f}",
                    f"{metrics['margin']:.1%}",
                    f"{metrics['coverage']:.1f}%",
                    f"{metrics.get('budget_utilization', 0):.1f}%",
                    f"${metrics.get('budget_remaining', 0):,.2f}"
                ],
                'Formula': [
                    f"Lines × Monthly Usage per Line × Policy Factor × Efficiency Factor\n= {students} × {monthly_usage_per_line} × {metrics.get('policy_factor', 1.0):.1f} × {metrics.get('efficiency_factor', 1.0):.1f}",
                    f"Usage × Carrier Rate\n= {metrics['usage']:,.1f} × ${carrier_rate:.2f}",
                    f"min(Budget, Lines × Monthly Price per Line)\n= min(${budget:,.0f}, {students} × ${customer_price:.2f})",
                    f"(Revenue - Carrier Cost) / Revenue\n= (${metrics['revenue']:,.2f} - ${metrics['carrier_cost']:,.2f}) / ${metrics['revenue']:,.2f}",
                    f"Revenue / (Lines × Monthly Price per Line) × 100\n= ${metrics['revenue']:,.2f} / ({students} × ${customer_price:.2f}) × 100",
                    f"Revenue / Budget × 100\n= ${metrics['revenue']:,.2f} / ${budget:,.2f} × 100",
                    f"Budget - Revenue\n= ${budget:,.2f} - ${metrics['revenue']:,.2f}"
                ]
            }
        
        breakdown_df = pd.DataFrame(breakdown_data)
        st.dataframe(breakdown_df, use_container_width=True, hide_index=True)
        
        # Optimization section within Financial View
        st.markdown("---")
        st.subheader("🎯 Pricing & Data Cap Optimization")
        
        st.markdown("**Find optimal pricing and data caps for maximum profitability:**")
        
        if 'optimization_result' in st.session_state:
            result = st.session_state.optimization_result
            
            if result['feasible']:
                st.success("✅ " + result['message'])
                
                # AI-powered optimization narrative (full width)
                if enable_ai_narratives:
                    with st.spinner("🤖 Generating optimization analysis..."):
                        opt_narrative = generate_optimization_narrative(result, metrics)
                        
                        # Show which AI mode is being used
                        if ai_status['ollama_available']:
                            ai_label = "🤖 **AI-Generated** (Ollama)"
                        elif ai_status['google_ai_available']:
                            ai_label = "🤖 **AI-Generated** (Google AI)"
                        else:
                            ai_label = "📝 **Smart Template**"
                        
                        if result['feasible']:
                            st.markdown(f"""
                            <div style="
                                background-color: #e8f5e8;
                                border: 1px solid #4caf50;
                                border-radius: 0.5rem;
                                padding: 1rem;
                                margin: 0.5rem 0;
                                font-family: 'Source Sans Pro', sans-serif;
                                font-size: 14px;
                                line-height: 1.5;
                            ">
                                <strong>{ai_label}:</strong><br>
                                {opt_narrative}
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div style="
                                background-color: #ffebee;
                                border: 1px solid #f44336;
                                border-radius: 0.5rem;
                                padding: 1rem;
                                margin: 0.5rem 0;
                                font-family: 'Source Sans Pro', sans-serif;
                                font-size: 14px;
                                line-height: 1.5;
                            ">
                                <strong>{ai_label}:</strong><br>
                                {opt_narrative}
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    # Fallback to static message
                    st.info("🎯 **Optimization Complete:** Found optimal pricing and data cap configuration for maximum profitability.")
                
                # Show optimization results
                opt_metrics = result['metrics']
                improvement = result['improvement']
                
                # Create columns for the configuration comparison
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**Current Configuration:**")
                    st.write(f"Monthly Price per Line: ${customer_price:.2f}")
                    st.write(f"Data Cap: {cap:.1f} GB")
                    st.write(f"Margin: {metrics['margin']:.1%}")
                    st.write(f"Coverage: {metrics['coverage']:.1f}%")
                    st.write(f"Revenue: ${metrics['revenue']:,.0f}")
                
                with col_b:
                    st.markdown("**Optimized Configuration:**")
                    st.write(f"Monthly Price per Line: ${result['recommended_customer_price']:.2f}")
                    st.write(f"Data Cap: {result['recommended_cap']:.1f} GB")
                    st.write(f"Margin: {opt_metrics['margin']:.1%}")
                    st.write(f"Coverage: {opt_metrics['coverage']:.1f}%")
                    st.write(f"Revenue: ${opt_metrics['revenue']:,.0f}")
                
                # Show improvement metrics
                st.markdown("**📈 Improvement Analysis:**")
                col_imp1, col_imp2 = st.columns(2)
                with col_imp1:
                    if improvement['margin_improvement'] > 0:
                        st.success(f"Margin: +{improvement['margin_improvement']:.1%}")
                    else:
                        st.info(f"Margin: {improvement['margin_improvement']:.1%}")
                
                with col_imp2:
                    if improvement['revenue_improvement'] > 0:
                        st.success(f"Revenue: +${improvement['revenue_improvement']:,.0f}")
                    else:
                        st.info(f"Revenue: ${improvement['revenue_improvement']:,.0f}")
                
                # Show comparison chart
                st.pyplot(create_optimization_comparison_chart(metrics, opt_metrics))
                
                # Apply button
                if st.button("✅ Apply Optimal Pricing", type="primary", key="apply_optimal_pricing"):
                    # Update session state with optimized values
                    current_scenario = st.session_state.current_scenario.copy()
                    current_scenario['customer_price'] = result['recommended_customer_price']
                    current_scenario['cap'] = result['recommended_cap']
                    st.session_state.current_scenario = current_scenario
                    # Trigger rerun to update the UI
                    st.experimental_rerun()
            else:
                st.error("❌ " + result['message'])
                st.info("💡 Try relaxing your minimum coverage or margin requirements")
        else:
            st.info("👆 **Click 'Find Optimal Pricing' in the sidebar to get pricing recommendations**")
        
        # Optimization Settings (moved to bottom for better layout)
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Optimization Settings:**")
            st.write(f"Min Coverage: {min_coverage}%")
            st.write(f"Min Margin: {min_margin}%")
            st.write(f"Budget: ${budget:,.0f}")
        with col2:
            st.markdown("**Current Parameters:**")
            st.write(f"Lines: {students}")
            st.write(f"Carrier Rate: ${carrier_rate}/GB")
            st.write(f"Vertical: {policy}")
    
    with tab3:
        st.header("🎯 Smart Cost Optimization (SCO) Analysis")
        st.markdown("### Compare SCO-enabled vs non-SCO-enabled plans")
        
        if sco_enabled:
            st.success("✅ SCO is currently enabled")
            
            # Calculate SCO comparison
            sco_comparison = compare_sco_vs_traditional(
                students, base_plan_gb, cap, budget, carrier_rate, 
                customer_price, policy, throttling, sco_efficiency, 
                overage_rate, plan_switching_cost, monthly_usage_per_line
            )
            
            # Display SCO benefits
            st.subheader("📊 SCO Benefits Analysis")
            
            # Check if SCO is actually beneficial
            cost_improvement = sco_comparison['comparison']['cost_improvement']
            margin_improvement = sco_comparison['comparison']['margin_improvement']
            
            if cost_improvement < 0 or margin_improvement < 0:
                st.warning("⚠️ **SCO Analysis:** In this scenario, SCO is not providing cost benefits. This could be due to high operational costs, low overage rates, or insufficient data usage patterns. Consider adjusting SCO parameters or using non-SCO plans.")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                cost_savings = sco_comparison['comparison']['cost_improvement']
                if cost_savings > 0:
                    st.metric(
                        "Cost Savings", 
                        f"${cost_savings:,.0f}",
                        delta=f"{sco_comparison['comparison']['cost_reduction_percent']:.1f}%"
                    )
                else:
                    st.metric(
                        "Additional Cost", 
                        f"${abs(cost_savings):,.0f}",
                        delta=f"{abs(sco_comparison['comparison']['cost_reduction_percent']):.1f}%",
                        delta_color="inverse"
                    )
            
            with col2:
                margin_improvement = sco_comparison['comparison']['margin_improvement']
                if margin_improvement > 0:
                    st.metric(
                        "Margin Improvement", 
                        f"{margin_improvement:.1%}",
                        delta=f"{sco_comparison['comparison']['margin_improvement_percent']:.1f}%"
                    )
                else:
                    st.metric(
                        "Margin Reduction", 
                        f"{abs(margin_improvement):.1%}",
                        delta=f"{abs(sco_comparison['comparison']['margin_improvement_percent']):.1f}%",
                        delta_color="inverse"
                    )
            
            with col3:
                roi = sco_comparison['comparison']['sco_roi']
                st.metric(
                    "SCO ROI", 
                    f"{roi:.1f}x",
                    delta="Return on SCO investment"
                )
            
            # Detailed SCO breakdown
            st.subheader("🔍 SCO Cost Breakdown")
            
            sco_metrics = sco_comparison['sco_metrics']
            non_sco_metrics = sco_comparison['non_sco_metrics']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**SCO-Enabled Plan:**")
                st.write(f"Monthly Usage per Line: {monthly_usage_per_line:.1f} GB")
                st.write(f"Total Monthly Usage: {sco_metrics['usage']:,.1f} GB")
                st.write(f"Base Plan Coverage: {sco_metrics['base_usage']:,.1f} GB")
                st.write(f"Potential Overage: {sco_metrics['potential_overage']:,.1f} GB")
                st.write(f"Actual Overage: {sco_metrics['actual_overage']:,.1f} GB")
                st.write(f"Overage Savings: {sco_metrics['overage_savings']:,.1f} GB")
                st.write(f"Base Carrier Cost: ${sco_metrics['base_carrier_cost']:,.0f}")
                st.write(f"Overage Cost: ${sco_metrics['overage_carrier_cost']:,.0f}")
                st.write(f"Switching Cost: ${sco_metrics['switching_cost']:,.0f}")
                st.write(f"SCO Operational Cost: ${sco_metrics['sco_operational_cost']:,.0f}")
                st.write(f"**Total Carrier Cost: ${sco_metrics['carrier_cost']:,.0f}**")
            
            with col2:
                st.markdown("**Non-SCO Plan:**")
                st.write(f"Total Usage: {non_sco_metrics['usage']:,.1f} GB")
                st.write(f"Carrier Cost: ${non_sco_metrics['carrier_cost']:,.0f}")
                st.write(f"Revenue: ${non_sco_metrics['revenue']:,.0f}")
                st.write(f"Margin: {non_sco_metrics['margin']:.1%}")
                st.write(f"Coverage: {non_sco_metrics['coverage']:.1f}%")
            
            # SCO efficiency visualization
            st.subheader("📈 SCO Efficiency Analysis")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Cost comparison
            categories = ['Base Cost', 'Overage Cost', 'Switching Cost', 'SCO Cost']
            sco_costs = [
                sco_metrics['base_carrier_cost'],
                sco_metrics['overage_carrier_cost'],
                sco_metrics['switching_cost'],
                sco_metrics['sco_operational_cost']
            ]
            non_sco_costs = [
                non_sco_metrics['carrier_cost'],
                0, 0, 0
            ]
            
            x = np.arange(len(categories))
            width = 0.35
            
            ax1.bar(x - width/2, sco_costs, width, label='SCO-Enabled', color='lightgreen', alpha=0.7)
            ax1.bar(x + width/2, non_sco_costs, width, label='Non-SCO', color='lightcoral', alpha=0.7)
            
            ax1.set_xlabel('Cost Categories')
            ax1.set_ylabel('Cost ($)')
            ax1.set_title('Cost Comparison: SCO vs Non-SCO')
            ax1.set_xticks(x)
            ax1.set_xticklabels(categories, rotation=45)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Usage breakdown - show actual data usage patterns
            usage_categories = ['Base Plan Usage', 'Overage Usage']
            usage_values = [
                max(0, sco_metrics['base_usage']),
                max(0, sco_metrics['actual_overage'])
            ]
            colors = ['lightblue', 'lightcoral']
            
            # Only create pie chart if we have positive values
            if sum(usage_values) > 0:
                ax2.pie(usage_values, labels=usage_categories, colors=colors, autopct='%1.1f%%', startangle=90)
            else:
                ax2.text(0.5, 0.5, 'No usage data available', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_xlim(0, 1)
                ax2.set_ylim(0, 1)
            ax2.set_title('Data Usage Breakdown (GB)')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Customer experience
            st.subheader("👥 Customer Experience")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**What Customers See:**")
                st.success(f"✅ Unlimited data experience ({cap}GB per line)")
                st.success("✅ No overage charges")
                st.success("✅ Seamless connectivity")
                st.success("✅ Predictable monthly costs")
            
            with col2:
                st.markdown("**Internal Reality:**")
                st.info(f"📊 Base plan: {base_plan_gb}GB per line")
                st.info(f"🔄 SCO efficiency: {sco_efficiency:.0%}")
                st.info(f"💰 Overage rate: ${overage_rate}/GB")
                st.info(f"⚡ Plan switching: ${plan_switching_cost}/line")
            
        else:
            st.info("💡 Enable SCO in the sidebar to see detailed analysis")
            
            # Show what SCO could provide
            st.subheader("🚀 What SCO Could Provide")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Customer Benefits:**")
                st.write("• Unlimited data experience")
                st.write("• No overage surprises")
                st.write("• Seamless connectivity")
                st.write("• Predictable costs")
            
            with col2:
                st.markdown("**Business Benefits:**")
                st.write("• Reduced overage costs")
                st.write("• Better margins")
                st.write("• Competitive advantage")
                st.write("• Operational efficiency")
            
            with col3:
                st.markdown("**Technical Benefits:**")
                st.write("• Intelligent cost optimization")
                st.write("• Usage pattern analysis")
                st.write("• Automated optimization")
                st.write("• Real-time adjustments")
    
    with tab4:
        st.header("🤖 AI Insights & Forecasting")
        st.markdown("### Advanced AI-powered trend analysis and predictive analytics")
        
        # AI Insights status
        ai_insights_status = get_ai_insights_status()
        if ai_insights_status['ollama_available'] and ai_insights_status['google_ai_available']:
            st.success("✅ Ollama + Google AI Available for Advanced Insights")
        elif ai_insights_status['ollama_available']:
            st.success("✅ Ollama (Local AI) Available for Advanced Insights")
        elif ai_insights_status['google_ai_available']:
            st.success("✅ Google AI (Cloud) Available for Advanced Insights")
        else:
            st.info("📝 Using Intelligent Templates - No AI services available")
            st.caption("AI insights require Ollama (local) or Google AI (cloud) to be configured")
        
        # Load historical data for analysis
        synthetic_data = load_synthetic_data()
        if synthetic_data is not None and len(synthetic_data) >= 3:
            
            # Current metrics for comparison
            current_metrics = {
                'margin': metrics['margin'],
                'coverage': metrics['coverage'],
                'revenue': metrics['revenue'],
                'carrier_cost': metrics['carrier_cost']
            }
            
            # AI Trend Analysis
            st.subheader("📊 AI Trend Analysis")
            st.markdown("Compare current performance against historical patterns")
            
            if st.button("🔍 Generate AI Trend Analysis", type="primary", key="ai_trend_analysis_1"):
                with st.spinner("🤖 AI is analyzing trends..."):
                    trend_analysis = generate_trend_analysis(synthetic_data, current_metrics)
                    st.session_state.trend_analysis_result = trend_analysis
            
            # Display trend analysis results if available
            if 'trend_analysis_result' in st.session_state:
                st.markdown("#### 🧠 AI Insights:")
                # Show which AI mode is being used
                if ai_insights_status['ollama_available']:
                    st.info(f"🤖 **AI-Generated** (Ollama): {st.session_state.trend_analysis_result}")
                else:
                    st.info(f"📝 **Smart Template**: {st.session_state.trend_analysis_result}")
            
            # AI Forecasting
            st.subheader("🔮 AI Forecasting & Predictions")
            st.markdown("Predict future performance using machine learning")
            
            forecast_periods = st.slider(
                "Forecast Periods (months)", 
                min_value=3, 
                max_value=12, 
                value=6,
                help="Number of future periods to predict"
            )
            
            if st.button("🔮 Generate AI Forecasts", type="primary", key="ai_forecasts_1"):
                with st.spinner("🤖 AI is generating forecasts..."):
                    forecast_results = generate_forecasting_insights(synthetic_data, forecast_periods)
                    st.session_state.forecast_results = forecast_results
            
            # Display forecast results if available
            if 'forecast_results' in st.session_state:
                forecast_results = st.session_state.forecast_results
                
                st.markdown("#### 🧠 AI Forecast Insights:")
                # Show which AI mode is being used
                if ai_insights_status['ollama_available']:
                    st.success(f"🤖 **AI-Generated** (Ollama): {forecast_results['ai_insights']}")
                else:
                    st.success(f"📝 **Smart Template**: {forecast_results['ai_insights']}")
                
                # Display forecast chart
                st.markdown("#### 📈 Forecast Visualization:")
                forecast_chart = create_forecast_chart(forecast_results, synthetic_data)
                st.pyplot(forecast_chart)
                
                # Model accuracy
                st.markdown("#### 📊 Model Accuracy:")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Margin R²", f"{forecast_results['model_accuracy']['margin_r2']:.3f}")
                with col2:
                    st.metric("Coverage R²", f"{forecast_results['model_accuracy']['coverage_r2']:.3f}")
                with col3:
                    st.metric("Revenue R²", f"{forecast_results['model_accuracy']['revenue_r2']:.3f}")
                
                # Forecast summary table
                st.markdown("#### 📋 Forecast Summary:")
                forecast_data = []
                for period in range(1, forecast_periods + 1):
                    max_periods = len(forecast_results['forecasts']['margin']['values'])
                    if period <= max_periods:
                        row = {
                            'Period': f"Month {period}",
                            'Margin (%)': f"{forecast_results['forecasts']['margin']['values'][period-1]*100:.1f}",
                            'Coverage (%)': f"{forecast_results['forecasts']['coverage']['values'][period-1]:.1f}",
                            'Revenue ($)': f"{forecast_results['forecasts']['revenue']['values'][period-1]:,.0f}"
                        }
                        forecast_data.append(row)
                
                if forecast_data:
                    forecast_df = pd.DataFrame(forecast_data)
                    st.dataframe(forecast_df, use_container_width=True)
            
            # Advanced Analytics
            st.subheader("📊 Advanced Analytics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Correlation Matrix:**")
                # Calculate correlation matrix
                numeric_cols = ['margin', 'coverage', 'revenue', 'carrier_cost']
                corr_matrix = synthetic_data[numeric_cols].corr()
                
                # Create correlation heatmap
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                           square=True, ax=ax, cbar_kws={'shrink': 0.8})
                ax.set_title('Metric Correlations')
                st.pyplot(fig)
            
            with col2:
                st.markdown("**Margin Distribution:**")
                # Create margin distribution histogram
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.hist(synthetic_data['margin'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                ax.axvline(metrics['margin'], color='red', linestyle='--', linewidth=2, label='Current')
                ax.set_xlabel('Margin')
                ax.set_ylabel('Frequency')
                ax.set_title('Historical Margin Distribution')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
        
        else:
            st.warning("⚠️ Insufficient historical data for AI analysis. Need at least 3 data points.")
            st.info("💡 Load more scenarios or use the Historical Analysis tab to generate synthetic data.")
    with tab5:
        st.header("📊 Traditional Excel Model View")
        st.markdown("### Basic spreadsheet-style analysis (for comparison)")
        
        # Excel-style layout with basic calculations
        st.markdown("---")
        st.subheader("📋 Basic Financial Model")
        
        # Create Excel-like table structure
        if sco_enabled:
            # SCO-enabled Excel view with additional SCO metrics
            excel_data = {
                'Parameter': [
                    'Number of Lines',
                    'Customer Data Cap per Line (GB)',
                    'Base Plan per Line (GB)',
                    'Total Data Capacity (GB)',
                    'Budget ($)',
                    'Carrier Rate ($/GB)',
                    'Overage Rate ($/GB)',
                    'Monthly Price per Line ($)',
                    'Vertical Factor',
                    'Efficiency Factor',
                    'SCO Efficiency (%)',
                    'Total Usage (GB)',
                    'Base Plan Usage (GB)',
                    'Potential Overage (GB)',
                    'Actual Overage (GB)',
                    'Overage Savings (GB)',
                    'Base Plan Cost ($)',
                    'Overage Cost ($)',
                    'Plan Switching Cost ($)',
                    'SCO Operational Cost ($)',
                    'Total Carrier Cost ($)',
                    'Revenue ($)',
                    'Margin ($)',
                    'Margin (%)',
                    'Coverage (%)',
                    'Budget Utilization (%)',
                    'Budget Remaining ($)',
                    'SCO ROI'
                ],
                'Value': [
                    f"{students:,}",
                    f"{cap:.1f}",
                    f"{base_plan_gb:.1f}",
                    f"{students * cap:,.1f}",
                    f"${budget:,.2f}",
                    f"${carrier_rate:.2f}",
                    f"${overage_rate:.2f}",
                    f"${customer_price:.2f}",
                    f"{metrics.get('policy_factor', 1.0):.1f}",
                    f"{metrics.get('efficiency_factor', 1.0):.1f}",
                    f"{metrics.get('sco_efficiency', 0.85):.1f}%",
                    f"{metrics['usage']:,.1f}",
                    f"{metrics.get('base_usage', 0):,.1f}",
                    f"{metrics.get('potential_overage', 0):,.1f}",
                    f"{metrics.get('actual_overage', 0):,.1f}",
                    f"{metrics.get('overage_savings', 0):,.1f}",
                    f"${metrics.get('base_carrier_cost', 0):,.2f}",
                    f"${metrics.get('overage_carrier_cost', 0):,.2f}",
                    f"${metrics.get('switching_cost', 0):,.2f}",
                    f"${metrics.get('sco_operational_cost', 0):,.2f}",
                    f"${metrics['carrier_cost']:,.2f}",
                    f"${metrics['revenue']:,.2f}",
                    f"${metrics['revenue'] - metrics['carrier_cost']:,.2f}",
                    f"{metrics['margin']:.1%}",
                    f"{metrics['coverage']:.1f}%",
                    f"{metrics.get('budget_utilization', 0):.1f}%",
                    f"${metrics.get('budget_remaining', 0):,.2f}",
                    f"{metrics.get('sco_savings', {}).get('roi', 0):.1f}x"
                ],
                'Formula': [
                    "Input (Total lines in service)",
                    "Input (What customer sees)", 
                    "Input (Internal base plan size)",
                    "Lines × Customer Data Cap per Line",
                    "Input (Total budget available)",
                    "Input (Cost per GB from carrier)",
                    "Input (Overage charge rate)",
                    "Input (Fixed monthly fee per line)",
                    "Policy Factor (Public Sector: 1.2, Retail: 1.0, Enterprise: 1.1)",
                    "Efficiency Factor (Throttling ON: 0.8, OFF: 1.0)",
                    "Input (SCO efficiency setting)",
                    "Lines × Monthly Usage per Line × Policy Factor × Efficiency Factor",
                    "Lines × Base Plan GB",
                    "Total Usage - Base Usage",
                    "Potential Overage × (1 - SCO Efficiency)",
                    "Potential Overage - Actual Overage",
                    "Base Usage × Carrier Rate",
                    "Actual Overage × Overage Rate",
                    "Lines × Plan Switching Cost",
                    "Lines × SCO Operational Cost per Line",
                    "Base Cost + Overage Cost + Switching Cost + SCO Operational Cost",
                    "min(Budget, Lines × Monthly Price per Line)",
                    "Revenue - Total Carrier Cost",
                    "(Revenue - Total Carrier Cost) / Revenue",
                    "Revenue / (Lines × Monthly Price per Line) × 100",
                    "Revenue / Budget × 100",
                    "Budget - Revenue",
                    "SCO Savings / SCO Costs"
                ]
            }
        else:
            # Standard Excel view without SCO metrics
            excel_data = {
                'Parameter': [
                    'Number of Lines',
                    'Data Cap per Line (GB)',
                    'Total Data Capacity (GB)',
                    'Budget ($)',
                    'Carrier Rate ($/GB)',
                    'Monthly Price per Line ($)',
                    'Vertical Factor',
                    'Efficiency Factor',
                    'Actual Usage (GB)',
                    'Carrier Cost ($)',
                    'Revenue ($)',
                    'Margin ($)',
                    'Margin (%)',
                    'Coverage (%)',
                    'Budget Utilization (%)',
                    'Budget Remaining ($)'
                ],
                'Value': [
                    f"{students:,}",
                    f"{cap:.1f}",
                    f"{students * cap:,.1f}",
                    f"${budget:,.2f}",
                    f"${carrier_rate:.2f}",
                    f"${customer_price:.2f}",
                    f"{metrics.get('policy_factor', 1.0):.1f}",
                    f"{metrics.get('efficiency_factor', 1.0):.1f}",
                    f"{metrics['usage']:,.1f}",
                    f"${metrics['carrier_cost']:,.2f}",
                    f"${metrics['revenue']:,.2f}",
                    f"${metrics['revenue'] - metrics['carrier_cost']:,.2f}",
                    f"{metrics['margin']:.1%}",
                    f"{metrics['coverage']:.1f}%",
                    f"{metrics.get('budget_utilization', 0):.1f}%",
                    f"${metrics.get('budget_remaining', 0):,.2f}"
                ],
                'Formula': [
                    "Input (Total lines in service)",
                    "Input (GB allowance per line)", 
                    "Lines × Data Cap per Line",
                    "Input (Total budget available)",
                    "Input (Cost per GB from carrier)",
                    "Input (Fixed monthly fee per line)",
                    "Policy Factor (Public Sector: 1.2, Retail: 1.0, Enterprise: 1.1)",
                    "Efficiency Factor (Throttling ON: 0.8, OFF: 1.0)",
                    "Lines × Monthly Usage per Line × Policy Factor × Efficiency Factor",
                    "Actual Usage × Carrier Rate",
                    "min(Budget, Lines × Monthly Price per Line)",
                    "Revenue - Carrier Cost",
                    "(Revenue - Carrier Cost) / Revenue",
                    "Revenue / (Lines × Monthly Price per Line) × 100",
                    "Revenue / Budget × 100",
                    "Budget - Revenue"
                ]
            }
        
        excel_df = pd.DataFrame(excel_data)
        st.dataframe(excel_df, use_container_width=True, hide_index=True)
        
        # Detailed formula explanations
        st.markdown("---")
        st.subheader("🧮 Formula Explanations")
        
        with st.expander("📖 Click to view detailed formula explanations", expanded=False):
            st.markdown("""
            **Key Calculation Formulas:**
            
            **1. Actual Usage Calculation:**
            ```
            Actual Usage = Lines × Monthly Usage per Line × Policy Factor × Efficiency Factor
            ```
            - **Monthly Usage per Line**: Typically 2.5 GB (realistic consumption)
            - **Policy Factor**: Public Sector (1.2), Retail (1.0), Enterprise (1.1)
            - **Efficiency Factor**: Throttling ON (0.8), OFF (1.0)
            
            **2. Carrier Cost:**
            ```
            Carrier Cost = Actual Usage × Carrier Rate
            ```
            - This represents what you pay the carrier for data consumption
            
            **3. Revenue (New Per-Line Model):**
            ```
        Revenue = min(Budget, Lines × Monthly Price per Line)
        ```
        - **Customer's budget limits** how much they can pay
        - If total monthly fees exceed budget, revenue is capped at budget
            
            **4. Margin Calculation:**
            ```
            Margin = (Revenue - Carrier Cost) / Revenue
            ```
            - Positive margin = profitable, Negative margin = loss
            
            **5. Coverage Percentage:**
            ```
            Coverage = Revenue / (Lines × Monthly Price per Line) × 100
            ```
            - 100% = full capacity, <100% = budget constraints limiting service reach
            
            **6. Budget Utilization:**
            ```
            Budget Utilization = (Revenue / Budget) × 100
            ```
            - Shows how much of the customer's budget is being used
            - 100% = customer's budget fully utilized, >100% = over budget
            
            **7. Budget Remaining:**
            ```
            Budget Remaining = Budget - Revenue
            ```
            - Shows how much of the customer's budget is left unused
            
            **Business Model:** Customer has a budget limit that constrains how much they can pay monthly.
            """)
        
        # Basic charts (Excel-style)
        st.markdown("---")
        st.subheader("📈 Basic Charts")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Simple bar chart
            st.markdown("**Cost vs Revenue**")
            fig, ax = plt.subplots(figsize=(8, 5))
            categories = ['Carrier Cost', 'Revenue']
            values = [metrics['carrier_cost'], metrics['revenue']]
            colors = ['#ff6b6b', '#4ecdc4']
            
            bars = ax.bar(categories, values, color=colors, alpha=0.7)
            ax.set_ylabel('Amount ($)')
            ax.set_title('Basic Financial Overview')
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'${value:,.0f}', ha='center', va='bottom')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            # Simple pie chart
            st.markdown("**Coverage Distribution**")
            fig, ax = plt.subplots(figsize=(8, 5))
            covered = metrics['coverage']
            uncovered = 100 - covered
            
            sizes = [covered, uncovered]
            labels = ['Covered Lines', 'Uncovered Lines']
            colors = ['#4ecdc4', '#ff6b6b']
            
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax.set_title('Line Coverage')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Limitations section
        st.markdown("---")
        st.subheader("⚠️ Traditional Excel Model Limitations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**What's Missing:**")
            st.write("❌ No SCO analysis")
            st.write("❌ No AI-powered insights")
            st.write("❌ No optimization engine")
            st.write("❌ No dual-audience views")
            st.write("❌ No forecasting capabilities")
            st.write("❌ No interactive scenario management")
            st.write("❌ No real-time calculations")
            st.write("❌ No advanced visualizations")
        
        with col2:
            st.markdown("**What You Get with Our Solution:**")
            st.write("✅ SCO cost optimization")
            st.write("✅ AI narratives and forecasting")
            st.write("✅ Automated optimization")
            st.write("✅ Customer vs Financial views")
            st.write("✅ Predictive analytics")
            st.write("✅ Interactive scenario comparison")
            st.write("✅ Real-time parameter adjustment")
            st.write("✅ Professional visualizations")
        
        # Call to action
        st.markdown("---")
        st.success("🚀 **Ready to see the difference?** Switch to other tabs to experience advanced features like SCO analysis, AI insights, and optimization!")
    
    with tab6:
        st.header("📊 Historical Data Analysis")
        st.markdown("### Advanced data visualization and trend analysis")
        
        synthetic_data = load_synthetic_data()
        if synthetic_data is not None:
            # Summary statistics
            st.subheader("📈 Dataset Overview")
            summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
            
            with summary_col1:
                st.metric("Total Scenarios", len(synthetic_data))
            
            with summary_col2:
                avg_margin = synthetic_data['margin'].mean()
                st.metric("Average Margin", f"{avg_margin:.1%}")
            
            with summary_col3:
                avg_coverage = synthetic_data['coverage'].mean()
                st.metric("Average Coverage", f"{avg_coverage:.1f}%")
            
            with summary_col4:
                profitable_count = len(synthetic_data[synthetic_data['margin'] > 0])
                st.metric("Profitable Scenarios", f"{profitable_count}/{len(synthetic_data)}")
            
            # Scatter plot analysis
            st.subheader("🎯 Margin vs Coverage Analysis")
            st.pyplot(create_scatter_plot(synthetic_data))
            
            # Data table
            st.subheader("📋 Historical Dataset")
            st.dataframe(synthetic_data, use_container_width=True)
            
            # Download buttons
            col_download1, col_download2 = st.columns(2)
            
            with col_download1:
                csv = synthetic_data.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name="synthetic_data.csv",
                    mime="text/csv"
                )
            
            with col_download2:
                json_data = synthetic_data.to_json(orient='records', indent=2)
                st.download_button(
                    label="📥 Download JSON",
                    data=json_data,
                    file_name="synthetic_data.json",
                    mime="application/json"
                )
        else:
            st.warning("⚠️ No historical data available. Generate some scenarios first.")
    
    # Export functionality
    st.markdown("---")
    st.subheader("📤 Export & Scenario Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📤 Export Current Scenario", key="export_scenario"):
            csv_data = export_scenario_summary(metrics, students, cap, budget, carrier_rate, customer_price, policy, throttling)
            st.download_button(
                label="📥 Download Scenario Summary",
                data=csv_data,
                file_name=f"scenario_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("💾 Save Current Scenario", key="save_scenario"):
            scenario_name = st.text_input("Scenario Name", value=f"Custom_{datetime.now().strftime('%Y%m%d_%H%M')}")
            if scenario_name:
                save_custom_scenario(scenario_name, students, cap, budget, carrier_rate, customer_price, policy, throttling)
                st.success(f"✅ Scenario '{scenario_name}' saved!")

if __name__ == "__main__":
    main()
