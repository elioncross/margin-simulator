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
    'throttling': True
}

# Unprofitable Scenario
UNPROFITABLE_SCENARIO = {
    'name': 'Unprofitable Example',
    'description': 'Suboptimal settings leading to losses',
    'students': 500,
    'cap': 5.0,  # GB per student
    'budget': 10000.0,  # $10k budget
    'carrier_rate': 8.0,  # $8/GB
    'customer_price': 25.0,  # $25/month per line
    'policy': 'Public Sector (Schools)',
    'throttling': False
}

# Additional test scenarios
SCENARIOS = {
    'profitable': PROFITABLE_SCENARIO,
    'unprofitable': UNPROFITABLE_SCENARIO,
    'high_volume': {
        'name': 'High Volume Scenario',
        'description': 'Large student base with moderate settings',
        'students': 800,
        'cap': 3.0,
        'budget': 15000.0,
        'carrier_rate': 7.0,
        'customer_price': 20.0,  # $20/month per line
        'policy': 'Retail (Households)',
        'throttling': True
    },
    'premium': {
        'name': 'Premium Service Scenario',
        'description': 'High-end service with premium pricing',
        'students': 300,
        'cap': 8.0,
        'budget': 18000.0,
        'carrier_rate': 9.0,
        'customer_price': 35.0,  # $35/month per line
        'policy': 'Public Sector (Schools)',
        'throttling': False
    },
    'budget': {
        'name': 'Budget Scenario',
        'description': 'Cost-conscious approach with limited budget',
        'students': 600,
        'cap': 2.0,
        'budget': 8000.0,
        'carrier_rate': 6.0,
        'customer_price': 15.0,  # $15/month per line
        'policy': 'Retail (Households)',
        'throttling': True
    },
    'sco_enterprise': {
        'name': 'SCO-Enabled Enterprise',
        'description': 'Enterprise with intelligent cost optimization',
        'students': 500,
        'cap': 8.0,  # What customer sees
        'base_plan_gb': 3.0,  # Internal base plan
        'budget': 15000.0,
        'carrier_rate': 8.0,
        'customer_price': 12.0,
        'policy': 'Enterprise',
        'throttling': True,
        'sco_enabled': True,
        'sco_efficiency': 0.85,
        'overage_rate': 15.0,
        'plan_switching_cost': 2.0
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
        'plan_switching_cost': 1.5
    },
    'traditional_static': {
        'name': 'Non-SCO Plans',
        'description': 'Fixed plans with overage charges (no SCO)',
        'students': 500,
        'cap': 5.0,
        'base_plan_gb': 5.0,  # Same as customer cap
        'budget': 12000.0,
        'carrier_rate': 8.0,
        'customer_price': 12.0,
        'policy': 'Public Sector (Schools)',
        'throttling': False,
        'sco_enabled': False,
        'sco_efficiency': 0.0,
        'overage_rate': 15.0,
        'plan_switching_cost': 0.0
    }
}

def get_scenario(scenario_key):
    """Get a specific scenario by key."""
    return SCENARIOS.get(scenario_key, PROFITABLE_SCENARIO)

def get_all_scenarios():
    """Get all available scenarios."""
    return SCENARIOS
