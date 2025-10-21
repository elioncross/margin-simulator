"""
Margin Impact Simulator - Model Layer
Contains formulas for usage, cost, revenue, margin, and coverage calculations.
Enhanced with Smart Cost Optimization (SCO) capabilities.
"""

def calculate_metrics(students, cap, budget, carrier_rate, customer_price, policy, throttling, monthly_usage_per_line=2.5, base_plan_gb=None, overage_rate=15.0):
    """
    Calculate key metrics for the margin impact simulator.
    
    Args:
        students (int): Number of lines
        cap (float): Data cap per line in GB
        budget (float): Budget in dollars
        carrier_rate (float): Carrier rate in $/GB
        customer_price (float): Customer pricing in $/GB
        policy (str): Vertical - "Public Sector (Schools)", "Retail (Households)", or "Enterprise"
        throttling (bool): Whether throttling is enabled
        monthly_usage_per_line (float): Average monthly data usage per line (GB)
        base_plan_gb (float): Base plan GB per line (carrier's actual plan)
        overage_rate (float): Overage rate per GB when exceeding base plan
    
    Returns:
        dict: Dictionary containing usage, cost, revenue, margin, coverage
    """
    
    # Policy factor mapping to business verticals:
    # Public Sector (Schools) = 1.0, Retail (Households) = 0.8, Enterprise = 0.9
    if policy == "Public Sector (Schools)":
        policy_factor = 1.0
    elif policy == "Retail (Households)":
        policy_factor = 0.8
    elif policy == "Enterprise":
        policy_factor = 0.9
    else:
        # Fallback for legacy values
        policy_factor = 1.0 if policy == "Per Student" else 0.8
    
    # Efficiency factor: Throttling On = 0.9, Off = 1.0
    efficiency_factor = 0.9 if throttling else 1.0
    
    # Calculate actual usage based on real consumption patterns
    # This represents the actual data consumed, not the cap offered
    total_monthly_usage = students * monthly_usage_per_line * policy_factor * efficiency_factor
    
    # If base_plan_gb is provided, use the same logic as SCO
    if base_plan_gb is not None:
        # Calculate base plan coverage
        base_plan_coverage = students * base_plan_gb
        
        # Calculate overage (actual usage - base plan coverage)
        if total_monthly_usage > base_plan_coverage:
            overage_gb = total_monthly_usage - base_plan_coverage
            base_usage = base_plan_coverage
        else:
            overage_gb = 0
            base_usage = total_monthly_usage
        
        # Calculate carrier cost with overage charges
        base_carrier_cost = base_usage * carrier_rate
        overage_carrier_cost = overage_gb * overage_rate
        carrier_cost = base_carrier_cost + overage_carrier_cost
        
        # Usage is the total actual consumption
        usage = total_monthly_usage
    else:
        # Legacy behavior: usage limited by customer cap
        usage = min(total_monthly_usage, students * cap)
        carrier_cost = usage * carrier_rate
    
    # Calculate revenue based on customer's budget constraint
    # Customer has a budget limit - they can't pay more than their budget
    monthly_revenue_per_line = customer_price  # Customer pays fixed monthly fee per line
    total_monthly_revenue = students * monthly_revenue_per_line
    # Revenue is limited by customer's budget
    revenue = min(budget, total_monthly_revenue)
    
    # Calculate margin (avoid division by zero)
    if revenue > 0:
        margin = (revenue - carrier_cost) / revenue
    else:
        margin = 0.0
    
    # Calculate coverage percentage (based on plan capacity)
    max_potential_revenue = students * customer_price
    if max_potential_revenue > 0:
        coverage = (revenue / max_potential_revenue) * 100
    else:
        coverage = 0.0
    
    # Calculate budget utilization (how much of customer's budget is being used)
    budget_utilization = (revenue / budget) * 100 if budget > 0 else 0
    budget_remaining = budget - revenue
    
    return {
        'usage': usage,
        'carrier_cost': carrier_cost,
        'revenue': revenue,
        'margin': margin,
        'coverage': coverage,
        'policy_factor': policy_factor,
        'efficiency_factor': efficiency_factor,
        'budget_utilization': budget_utilization,
        'budget_remaining': budget_remaining
    }

def optimize_sco_parameters(students, current_cap, budget, carrier_rate, current_customer_price, 
                           policy, throttling, min_coverage=80.0, min_margin=0.0, 
                           sco_enabled=False, current_base_plan_gb=None, current_sco_efficiency=0.85, 
                           current_overage_rate=15.0, current_plan_switching_cost=2.0, 
                           current_monthly_usage_per_line=2.5):
    """
    Optimize SCO parameters and service settings for maximum profitability.
    
    When SCO is enabled: Optimizes SCO parameters (base plan, efficiency, overage rates)
    When SCO is disabled: Optimizes service parameters (usage patterns, throttling)
    
    Objective: Maximize Margin = (Revenue - CarrierCost) / Revenue
    Constraints:
    - Revenue ≤ Budget (customer's budget limit)
    - Coverage ≥ Minimum Coverage
    - Margin ≥ Minimum Margin (to ensure profitability)
    
    Args:
        students (int): Number of lines
        current_cap (float): Current data cap per line in GB (fixed - service feature)
        budget (float): Budget in dollars
        carrier_rate (float): Carrier rate in $/GB (fixed)
        current_customer_price (float): Current customer pricing in $/month per line (fixed)
        policy (str): Vertical - "Public Sector (Schools)", "Retail (Households)", or "Enterprise" (fixed)
        throttling (bool): Current throttling setting
        min_coverage (float): Minimum coverage percentage required
        min_margin (float): Minimum margin required (default 0%)
        sco_enabled (bool): Whether SCO is enabled
        current_base_plan_gb (float): Current base plan size for SCO
        current_sco_efficiency (float): Current SCO efficiency (0.0-1.0)
        current_overage_rate (float): Current overage rate in $/GB
        current_plan_switching_cost (float): Current plan switching cost per line
        current_monthly_usage_per_line (float): Current monthly usage per line
    
    Returns:
        dict: Optimization results containing:
            - recommended_base_plan_gb (float): Optimal base plan size (SCO only)
            - recommended_sco_efficiency (float): Optimal SCO efficiency (SCO only)
            - recommended_overage_rate (float): Optimal overage rate (SCO only)
            - recommended_plan_switching_cost (float): Optimal switching cost (SCO only)
            - recommended_monthly_usage_per_line (float): Optimal usage pattern
            - recommended_throttling (bool): Optimal throttling setting
            - metrics (dict): Calculated metrics for optimal solution
            - feasible (bool): Whether a feasible solution was found
            - message (str): Status message
            - alternatives (list): Top 5 alternative solutions
            - improvement (dict): Improvement over current settings
    """
    
    # Set default base plan if not provided
    if current_base_plan_gb is None:
        current_base_plan_gb = current_cap
    
    best_solution = None
    best_margin = -float('inf')
    all_solutions = []
    
    if sco_enabled:
        # SCO-enabled optimization: Focus on SCO parameters
        # Base plan sizes: ±50% of current, in 0.5GB increments
        base_plan_min = max(1.0, current_base_plan_gb * 0.5)
        base_plan_max = min(current_cap * 0.9, current_base_plan_gb * 1.5)  # Can't exceed customer cap
        base_plan_step = 0.5
        
        # SCO efficiency: 0.6 to 0.95, in 0.05 increments
        sco_efficiency_values = [round(0.6 + i * 0.05, 2) for i in range(8)]  # 0.6, 0.65, ..., 0.95
        
        # Overage rates: ±30% of current, in $2 increments
        overage_min = max(5.0, current_overage_rate * 0.7)
        overage_max = min(25.0, current_overage_rate * 1.3)
        overage_step = 2.0
        
        # Plan switching costs: ±50% of current, in $0.25 increments
        switching_min = max(0.25, current_plan_switching_cost * 0.5)
        switching_max = min(5.0, current_plan_switching_cost * 1.5)
        switching_step = 0.25
        
        # Usage patterns: ±20% of current, in 0.25GB increments
        usage_min = max(1.0, current_monthly_usage_per_line * 0.8)
        usage_max = min(5.0, current_monthly_usage_per_line * 1.2)
        usage_step = 0.25
        
        # Generate search space
        base_plans = [round(base_plan_min + i * base_plan_step, 1) for i in range(int((base_plan_max - base_plan_min) / base_plan_step) + 1)]
        overage_rates = [round(overage_min + i * overage_step, 1) for i in range(int((overage_max - overage_min) / overage_step) + 1)]
        switching_costs = [round(switching_min + i * switching_step, 2) for i in range(int((switching_max - switching_min) / switching_step) + 1)]
        usage_patterns = [round(usage_min + i * usage_step, 2) for i in range(int((usage_max - usage_min) / usage_step) + 1)]
        
        # Limit search space to reasonable size (max 200 combinations)
        if len(base_plans) * len(sco_efficiency_values) * len(overage_rates) * len(switching_costs) * len(usage_patterns) > 200:
            # Sample more intelligently
            base_plans = [round(base_plan_min + i * (base_plan_max - base_plan_min) / 4, 1) for i in range(5)]
            overage_rates = [round(overage_min + i * (overage_max - overage_min) / 4, 1) for i in range(5)]
            switching_costs = [round(switching_min + i * (switching_max - switching_min) / 4, 2) for i in range(5)]
            usage_patterns = [round(usage_min + i * (usage_max - usage_min) / 4, 2) for i in range(5)]
        
        # Search over SCO parameter combinations
        for base_plan_gb in base_plans:
            for sco_efficiency in sco_efficiency_values:
                for overage_rate in overage_rates:
                    for plan_switching_cost in switching_costs:
                        for monthly_usage_per_line in usage_patterns:
                            # Calculate metrics for this SCO combination
                            metrics = calculate_sco_metrics(
                                students, base_plan_gb, current_cap, budget, carrier_rate, 
                                current_customer_price, policy, throttling, sco_enabled=True,
                                sco_efficiency=sco_efficiency, overage_rate=overage_rate, 
                                plan_switching_cost=plan_switching_cost, monthly_usage_per_line=monthly_usage_per_line
                            )
                            
                            # Check constraints
                            budget_constraint = metrics['revenue'] <= budget
                            coverage_constraint = metrics['coverage'] >= min_coverage
                            margin_constraint = metrics['margin'] >= min_margin
                            
                            # Store solution for analysis
                            solution = {
                                'base_plan_gb': base_plan_gb,
                                'sco_efficiency': sco_efficiency,
                                'overage_rate': overage_rate,
                                'plan_switching_cost': plan_switching_cost,
                                'monthly_usage_per_line': monthly_usage_per_line,
                                'throttling': throttling,  # Keep current throttling
                                'metrics': metrics,
                                'feasible': budget_constraint and coverage_constraint and margin_constraint
                            }
                            all_solutions.append(solution)
                            
                            # Update best solution if constraints are met and margin is better
                            if solution['feasible'] and metrics['margin'] > best_margin:
                                best_margin = metrics['margin']
                                best_solution = solution
    else:
        # Non-SCO optimization: Focus on service parameters
        # Usage patterns: ±30% of current, in 0.25GB increments
        usage_min = max(1.0, current_monthly_usage_per_line * 0.7)
        usage_max = min(5.0, current_monthly_usage_per_line * 1.3)
        usage_step = 0.25
        
        # Throttling options: True, False
        throttling_options = [True, False]
        
        # Generate search space
        usage_patterns = [round(usage_min + i * usage_step, 2) for i in range(int((usage_max - usage_min) / usage_step) + 1)]
        
        # Search over service parameter combinations
        for monthly_usage_per_line in usage_patterns:
            for throttling in throttling_options:
                # Calculate metrics for this service combination
                metrics = calculate_metrics(
                    students, current_cap, budget, carrier_rate, 
                    current_customer_price, policy, throttling, monthly_usage_per_line
                )
                
                # Check constraints
                budget_constraint = metrics['revenue'] <= budget
                coverage_constraint = metrics['coverage'] >= min_coverage
                margin_constraint = metrics['margin'] >= min_margin
                
                # Store solution for analysis
                solution = {
                    'base_plan_gb': None,  # Not applicable for non-SCO
                    'sco_efficiency': None,  # Not applicable for non-SCO
                    'overage_rate': None,  # Not applicable for non-SCO
                    'plan_switching_cost': None,  # Not applicable for non-SCO
                    'monthly_usage_per_line': monthly_usage_per_line,
                    'throttling': throttling,
                    'metrics': metrics,
                    'feasible': budget_constraint and coverage_constraint and margin_constraint
                }
                all_solutions.append(solution)
                
                # Update best solution if constraints are met and margin is better
                if solution['feasible'] and metrics['margin'] > best_margin:
                    best_margin = metrics['margin']
                    best_solution = solution
    
    # Calculate current metrics for comparison
    if sco_enabled:
        current_metrics = calculate_sco_metrics(
            students, current_base_plan_gb, current_cap, budget, carrier_rate, 
            current_customer_price, policy, throttling, sco_enabled=True,
            sco_efficiency=current_sco_efficiency, overage_rate=current_overage_rate, 
            plan_switching_cost=current_plan_switching_cost, monthly_usage_per_line=current_monthly_usage_per_line
        )
    else:
        current_metrics = calculate_metrics(
            students, current_cap, budget, carrier_rate, 
            current_customer_price, policy, throttling, current_monthly_usage_per_line
        )
    
    # Prepare results
    if best_solution is not None:
        # Sort all feasible solutions by margin (descending)
        feasible_solutions = [s for s in all_solutions if s['feasible']]
        feasible_solutions.sort(key=lambda x: x['metrics']['margin'], reverse=True)
        
        # Calculate improvement
        margin_improvement = best_solution['metrics']['margin'] - current_metrics['margin']
        revenue_improvement = best_solution['metrics']['revenue'] - current_metrics['revenue']
        
        sco_status = " (SCO-enabled)" if sco_enabled else ""
        
        if sco_enabled:
            message = f"Optimal SCO parameters found{sco_status}: Base plan {best_solution['base_plan_gb']:.1f}GB, Efficiency {best_solution['sco_efficiency']:.1%}, Usage {best_solution['monthly_usage_per_line']:.1f}GB/line → {best_solution['metrics']['margin']:.1%} margin"
        else:
            message = f"Optimal service parameters found{sco_status}: Usage {best_solution['monthly_usage_per_line']:.1f}GB/line, Throttling {best_solution['throttling']} → {best_solution['metrics']['margin']:.1%} margin"
        
        return {
            'recommended_base_plan_gb': best_solution['base_plan_gb'],
            'recommended_sco_efficiency': best_solution['sco_efficiency'],
            'recommended_overage_rate': best_solution['overage_rate'],
            'recommended_plan_switching_cost': best_solution['plan_switching_cost'],
            'recommended_monthly_usage_per_line': best_solution['monthly_usage_per_line'],
            'recommended_throttling': best_solution['throttling'],
            'metrics': best_solution['metrics'],
            'feasible': True,
            'message': message,
            'alternatives': feasible_solutions[:5],  # Top 5 alternatives
            'total_combinations_tested': len(all_solutions),
            'feasible_combinations': len(feasible_solutions),
            'improvement': {
                'margin_improvement': margin_improvement,
                'revenue_improvement': revenue_improvement,
                'current_margin': current_metrics['margin'],
                'current_revenue': current_metrics['revenue']
            }
        }
    else:
        # No feasible solution found
        sco_status = " (SCO-enabled)" if sco_enabled else ""
        return {
            'recommended_base_plan_gb': None,
            'recommended_sco_efficiency': None,
            'recommended_overage_rate': None,
            'recommended_plan_switching_cost': None,
            'recommended_monthly_usage_per_line': None,
            'recommended_throttling': None,
            'metrics': None,
            'feasible': False,
            'message': f"No feasible solution found{sco_status}. Try relaxing constraints (min coverage: {min_coverage}%, min margin: {min_margin:.1%})",
            'alternatives': [],
            'total_combinations_tested': len(all_solutions),
            'feasible_combinations': 0,
            'improvement': {
                'margin_improvement': 0,
                'revenue_improvement': 0,
                'current_margin': current_metrics['margin'],
                'current_revenue': current_metrics['revenue']
            }
        }

def calculate_sco_metrics(students, base_plan_gb, customer_cap, budget, carrier_rate, 
                         customer_price, policy, throttling, sco_enabled=True, 
                         sco_efficiency=0.85, overage_rate=15.0, plan_switching_cost=2.0,
                         monthly_usage_per_line=2.5):
    """
    Calculate metrics with Smart Cost Optimization (SCO) capabilities.
    
    SCO Model:
    - Customer sees "unlimited" experience (customer_cap)
    - Internal base plan is smaller (base_plan_gb)
    - SCO prevents overages by switching plans dynamically
    - Overage charges apply when SCO can't prevent them
    
    Args:
        students (int): Number of lines
        base_plan_gb (float): Base plan size per line in GB (internal)
        customer_cap (float): What customer sees as their data allowance
        budget (float): Budget in dollars
        carrier_rate (float): Base carrier rate in $/GB
        customer_price (float): Customer pricing in $/month per line
        policy (str): Vertical - "Public Sector (Schools)", "Retail (Households)", or "Enterprise"
        throttling (bool): Whether throttling is enabled
        sco_enabled (bool): Whether SCO is enabled
        sco_efficiency (float): How well SCO prevents overages (0.0-1.0)
        overage_rate (float): Rate for overage charges in $/GB
        plan_switching_cost (float): Cost per line per plan switch
    
    Returns:
        dict: Dictionary containing SCO metrics and analysis
    """
    
    # Policy factor mapping to business verticals
    if policy == "Public Sector (Schools)":
        policy_factor = 1.0
    elif policy == "Retail (Households)":
        policy_factor = 0.8
    elif policy == "Enterprise":
        policy_factor = 0.9
    else:
        policy_factor = 1.0 if policy == "Per Student" else 0.8
    
    # Efficiency factor: Throttling On = 0.9, Off = 1.0
    efficiency_factor = 0.9 if throttling else 1.0
    
    # Calculate actual monthly usage (realistic consumption pattern)
    total_monthly_usage = students * monthly_usage_per_line * policy_factor * efficiency_factor
    
    # Calculate base plan coverage
    base_plan_coverage = students * base_plan_gb
    
    # Calculate overage (actual usage - base plan coverage)
    if total_monthly_usage > base_plan_coverage:
        potential_overage = total_monthly_usage - base_plan_coverage
        base_usage = base_plan_coverage
    else:
        potential_overage = 0
        base_usage = total_monthly_usage
    
    if sco_enabled:
        # SCO prevents some overage
        actual_overage = potential_overage * (1 - sco_efficiency)
        overage_savings = potential_overage - actual_overage
        
        # Plan switching costs (assume 10% of lines switch plans monthly)
        switching_frequency = 0.1  # 10% of lines switch plans
        switching_cost = students * plan_switching_cost * switching_frequency
        
        # SCO operational cost (assume $0.20 per line per month for SCO service)
        sco_operational_cost = students * 0.2
    else:
        # No SCO - full overage charges
        actual_overage = potential_overage
        overage_savings = 0
        switching_cost = 0
        sco_operational_cost = 0
    
    # Calculate total usage and costs
    total_usage = total_monthly_usage  # This is the actual usage regardless of SCO
    base_carrier_cost = base_usage * carrier_rate
    overage_carrier_cost = actual_overage * overage_rate
    total_carrier_cost = base_carrier_cost + overage_carrier_cost + switching_cost + sco_operational_cost
    
    # Customer revenue (based on customer's budget constraint)
    # Customer has a budget limit - they can't pay more than their budget
    monthly_revenue_per_line = customer_price  # Customer pays fixed monthly fee per line
    total_monthly_revenue = students * monthly_revenue_per_line
    # Revenue is limited by customer's budget
    revenue = min(budget, total_monthly_revenue)
    
    # Calculate margin
    if revenue > 0:
        margin = (revenue - total_carrier_cost) / revenue
    else:
        margin = 0.0
    
    # Calculate coverage percentage (based on plan capacity)
    max_potential_revenue = students * customer_price
    if max_potential_revenue > 0:
        coverage = (revenue / max_potential_revenue) * 100
    else:
        coverage = 0.0
    
    # Calculate traditional metrics for comparison
    traditional_metrics = calculate_metrics(
        students, customer_cap, budget, carrier_rate, customer_price, policy, throttling, monthly_usage_per_line,
        base_plan_gb, overage_rate
    )
    
    # Calculate budget utilization (how much of customer's budget is being used)
    budget_utilization = (revenue / budget) * 100 if budget > 0 else 0
    budget_remaining = budget - revenue
    
    return {
        'usage': total_usage,
        'base_usage': base_usage,
        'potential_overage': potential_overage,
        'actual_overage': actual_overage,
        'overage_savings': overage_savings,
        'carrier_cost': total_carrier_cost,
        'base_carrier_cost': base_carrier_cost,
        'overage_carrier_cost': overage_carrier_cost,
        'switching_cost': switching_cost,
        'sco_operational_cost': sco_operational_cost,
        'revenue': revenue,
        'margin': margin,
        'coverage': coverage,
        'policy_factor': policy_factor,
        'efficiency_factor': efficiency_factor,
        'sco_enabled': sco_enabled,
        'sco_efficiency': sco_efficiency,
        'traditional_metrics': traditional_metrics,
        'budget_utilization': budget_utilization,
        'budget_remaining': budget_remaining,
        'sco_savings': {
            'overage_cost_savings': overage_savings * overage_rate,
            'total_savings': overage_savings * overage_rate - sco_operational_cost - switching_cost,
            'roi': (overage_savings * overage_rate - sco_operational_cost - switching_cost) / max(sco_operational_cost + switching_cost, 1)
        }
    }

def compare_sco_vs_traditional(students, base_plan_gb, customer_cap, budget, carrier_rate, 
                              customer_price, policy, throttling, sco_efficiency=0.85, 
                              overage_rate=15.0, plan_switching_cost=2.0, monthly_usage_per_line=2.5):
    """
    Compare SCO-enabled vs non-SCO-enabled plans using the same calculation method.
    
    Returns:
        dict: Comparison analysis between SCO-enabled and non-SCO-enabled approaches
    """
    
    # Calculate SCO metrics (SCO enabled)
    sco_metrics = calculate_sco_metrics(
        students, base_plan_gb, customer_cap, budget, carrier_rate, 
        customer_price, policy, throttling, sco_enabled=True,
        sco_efficiency=sco_efficiency, overage_rate=overage_rate, 
        plan_switching_cost=plan_switching_cost, monthly_usage_per_line=monthly_usage_per_line
    )
    
    # Calculate non-SCO metrics using the same calculation method but with SCO disabled
    # This ensures we're comparing apples to apples - same method, just SCO on/off
    non_sco_metrics = calculate_sco_metrics(
        students, base_plan_gb, customer_cap, budget, carrier_rate, 
        customer_price, policy, throttling, sco_enabled=False,
        sco_efficiency=0.0, overage_rate=overage_rate, 
        plan_switching_cost=0.0, monthly_usage_per_line=monthly_usage_per_line
    )
    
    # Calculate improvements
    cost_improvement = non_sco_metrics['carrier_cost'] - sco_metrics['carrier_cost']
    margin_improvement = sco_metrics['margin'] - non_sco_metrics['margin']
    revenue_improvement = sco_metrics['revenue'] - non_sco_metrics['revenue']
    
    return {
        'sco_metrics': sco_metrics,
        'non_sco_metrics': non_sco_metrics,
        'comparison': {
            'cost_improvement': cost_improvement,
            'margin_improvement': margin_improvement,
            'revenue_improvement': revenue_improvement,
            'cost_reduction_percent': (cost_improvement / non_sco_metrics['carrier_cost']) * 100 if non_sco_metrics['carrier_cost'] > 0 else 0,
            'margin_improvement_percent': margin_improvement * 100,
            'sco_roi': sco_metrics['sco_savings']['roi']
        }
    }
