"""
Margin Impact Simulator - Scenarios
Contains preloaded profitable and unprofitable scenarios for testing.
"""

# Profitable Scenario
PROFITABLE_SCENARIO = {
    'name': 'Profitable Example',
    'description': 'Optimized settings for maximum profitability',
    'students': 500,
    'cap': 5.0,  # GB per student
    'budget': 12000.0,  # $12k budget
    'carrier_rate': 8.0,  # $8/GB
    'customer_price': 25.0,  # $25/month per line
    'policy': 'Retail (Households)',
    'throttling': True,
    'monthly_usage_per_line': 2.0  # Lower usage for better profitability
}

# Unprofitable Scenario
UNPROFITABLE_SCENARIO = {
    'name': 'Unprofitable Example',
    'description': 'Suboptimal settings leading to losses',
    'students': 500,
    'cap': 8.0,  # High data cap
    'budget': 8000.0,  # Low budget
    'carrier_rate': 12.0,  # High carrier cost
    'customer_price': 15.0,  # Low customer price
    'policy': 'Public Sector (Schools)',
    'throttling': False,
    'monthly_usage_per_line': 3.5  # Higher usage to ensure unprofitability
}

# Additional test scenarios
SCENARIOS = {
    'profitable': PROFITABLE_SCENARIO,
    'unprofitable': UNPROFITABLE_SCENARIO,
    'high_volume': {
        'name': 'High Volume Scenario',
        'description': 'Large student base with volume discounts',
        'students': 1000,
        'cap': 4.0,
        'budget': 20000.0,
        'carrier_rate': 6.0,  # Volume discount
        'customer_price': 18.0,  # $18/month per line
        'policy': 'Retail (Households)',
        'throttling': True,
        'monthly_usage_per_line': 2.5  # Moderate usage for volume scenario
    },
    'premium': {
        'name': 'Premium Service Scenario',
        'description': 'High-end service with premium pricing',
        'students': 200,
        'cap': 10.0,
        'budget': 12000.0,
        'carrier_rate': 8.0,
        'customer_price': 45.0,  # $45/month per line
        'policy': 'Enterprise',
        'throttling': False,
        'monthly_usage_per_line': 6.0  # High usage for premium service
    },
    'budget': {
        'name': 'Budget Scenario',
        'description': 'Cost-conscious approach with limited budget',
        'students': 400,
        'cap': 2.5,
        'budget': 6000.0,
        'carrier_rate': 6.5,
        'customer_price': 12.0,  # $12/month per line
        'policy': 'Public Sector (Schools)',
        'throttling': True,
        'monthly_usage_per_line': 1.5  # Low usage for budget scenario
    },
    'sco_enterprise': {
        'name': 'SCO-Enabled Enterprise',
        'description': 'Enterprise with intelligent cost optimization',
        'students': 500,
        'cap': 8.0,  # What customer sees
        'base_plan_gb': 3.0,  # Internal base plan
        'budget': 15000.0,
        'carrier_rate': 8.0,
        'customer_price': 25.0,  # Realistic enterprise pricing
        'policy': 'Enterprise',
        'throttling': True,
        'sco_enabled': True,
        'sco_efficiency': 0.85,
        'overage_rate': 15.0,
        'plan_switching_cost': 0.5,  # Lower cost for better SCO benefits
        'monthly_usage_per_line': 4.5  # Higher usage to generate overages and show SCO benefits
    },
    'sco_retail': {
        'name': 'SCO-Enabled Retail',
        'description': 'Retail households with SCO optimization',
        'students': 800,
        'cap': 5.0,  # What customer sees
        'base_plan_gb': 2.0,  # Internal base plan
        'budget': 12000.0,
        'carrier_rate': 7.0,
        'customer_price': 20.0,  # $20/month per line
        'policy': 'Retail (Households)',
        'throttling': True,
        'sco_enabled': True,
        'sco_efficiency': 0.80,
        'overage_rate': 12.0,
        'plan_switching_cost': 0.3,  # Lower cost for better SCO benefits
        'monthly_usage_per_line': 3.2  # Higher usage to generate overages and show SCO benefits
    },
    'enterprise_baseline': {
        'name': 'Enterprise Baseline (No SCO)',
        'description': 'Enterprise scenario without SCO for comparison',
        'students': 500,
        'cap': 8.0,
        'budget': 15000.0,
        'carrier_rate': 8.0,
        'customer_price': 25.0,  # Same pricing as SCO enterprise
        'policy': 'Enterprise',
        'throttling': True,
        'sco_enabled': False,
        'monthly_usage_per_line': 4.5  # Same usage as SCO enterprise for fair comparison
    },
    'high_usage': {
        'name': 'High Usage Scenario',
        'description': 'Heavy data usage scenario with potential overages',
        'students': 300,
        'cap': 6.0,
        'budget': 10000.0,
        'carrier_rate': 9.0,
        'customer_price': 30.0,  # $30/month per line
        'policy': 'Retail (Households)',
        'throttling': False,
        'monthly_usage_per_line': 7.0  # Very high usage to test overage scenarios
    },
    'low_usage': {
        'name': 'Low Usage Scenario',
        'description': 'Light data usage scenario with minimal overages',
        'students': 600,
        'cap': 3.0,
        'budget': 8000.0,
        'carrier_rate': 7.0,
        'customer_price': 15.0,  # $15/month per line
        'policy': 'Public Sector (Schools)',
        'throttling': True,
        'monthly_usage_per_line': 1.0  # Very low usage for efficiency testing
    },
}

def get_scenario(scenario_key):
    """Get a specific scenario by key."""
    return SCENARIOS.get(scenario_key, PROFITABLE_SCENARIO)

def get_all_scenarios():
    """Get all available scenarios."""
    return SCENARIOS
