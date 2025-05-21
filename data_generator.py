import pandas as pd
import numpy as np
import random

def generate_mock_data():
    """
    Generate mock data for the business dashboard.
    In a real application, this would be replaced with actual data loading from the Excel file.
    """
    np.random.seed(42)  # For reproducibility
    
    # Overall budget and expense data
    overall_budget = {
        'BU1': 1_200_000,
        'BU2': 950_000,
        'BU3': 1_500_000
    }
    
    overall_expense = {
        'BU1': 980_000,
        'BU2': 920_000,
        'BU3': 1_350_000
    }
    
    # Profit and revenue data
    overall_profit = 1_400_000
    overall_profit_change = 12.5
    overall_revenue = 3_800_000
    overall_revenue_change = 8.2
    
    # Profit and revenue breakdown
    bu_profit_revenue = pd.DataFrame([
        {'category': 'BU1 Profit', 'value': 420_000},
        {'category': 'BU2 Profit', 'value': 380_000},
        {'category': 'BU3 Profit', 'value': 600_000},
        {'category': 'BU1 Revenue', 'value': 1_150_000},
        {'category': 'BU2 Revenue', 'value': 950_000},
        {'category': 'BU3 Revenue', 'value': 1_700_000}
    ])
    
    # Customer distribution data
    customer_by_bu = pd.DataFrame([
        {'bu': 'BU1', 'customers': 1200},
        {'bu': 'BU2', 'customers': 950},
        {'bu': 'BU3', 'customers': 1500}
    ])
    
    # Customer satisfaction trend
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    satisfaction_trend = pd.DataFrame()
    
    for bu in ['BU1', 'BU2', 'BU3']:
        base = 75 + np.random.randint(0, 10)
        trend = base + np.cumsum(np.random.normal(0.5, 1, len(months)))
        trend = np.clip(trend, 65, 95)
        
        for i, month in enumerate(months):
            satisfaction_trend = pd.concat([
                satisfaction_trend, 
                pd.DataFrame({'month': [month], 'bu': [bu], 'satisfaction': [trend[i]]})
            ])
    
    # Target vs realization data
    target_realization = {
        'BU1': {'target': 95, 'realization': 87},
        'BU2': {'target': 90, 'realization': 85},
        'BU3': {'target': 92, 'realization': 90}
    }
    
    # Quality and velocity data
    overall_velocity = 22.5
    overall_quality = 88.3
    
    # Manpower data
    manpower = {
        'current': {'BU1': 48, 'BU2': 52, 'BU3': 67},
        'required': {'BU1': 50, 'BU2': 55, 'BU3': 70}
    }
    
    # Competency data
    competency = {'BU1': 3.8, 'BU2': 4.1, 'BU3': 3.9}
    
    # Turnover ratio
    turnover_ratio = 14.5
    
    # BU-specific data
    bu_data = []
    
    # BU1 data
    bu1 = {
        'subdivs': ['Subdiv A', 'Subdiv B', 'Subdiv C', 'Subdiv D'],
        'products': ['Product X', 'Product Y', 'Product Z'],
        'subdiv_budget': [350_000, 280_000, 320_000, 250_000],
        'subdiv_expense': [320_000, 260_000, 280_000, 220_000],
        'subdiv_profit': [110_000, 95_000, 125_000, 90_000],
        'subdiv_profit_change': [11.2, 8.5, 15.4, 7.8],
        'subdiv_revenue': [320_000, 280_000, 350_000, 200_000],
        'subdiv_revenue_change': [6.8, 5.5, 12.3, 4.5],
        'subdiv_financial_health': [85, 90, 78, 82],
        'subdiv_customers': [
            [150, 120, 80],  # Subdiv A customers by product
            [110, 90, 70],   # Subdiv B customers by product
            [180, 140, 60],  # Subdiv C customers by product
            [100, 50, 50]    # Subdiv D customers by product
        ],
        'subdiv_satisfaction': [
            [75, 76, 77, 79, 81, 82],  # Subdiv A satisfaction trend
            [78, 80, 79, 81, 80, 83],  # Subdiv B satisfaction trend
            [72, 73, 75, 78, 80, 81],  # Subdiv C satisfaction trend
            [76, 77, 79, 80, 81, 83]   # Subdiv D satisfaction trend
        ],
        'subdiv_target': [92, 90, 94, 88],
        'subdiv_realization': [85, 86, 88, 80],
        'subdiv_velocity': [21.5, 23.2, 22.8, 20.5],
        'subdiv_quality': [86, 88, 90, 84],
        'subdiv_current_emp': [12, 10, 15, 11],
        'subdiv_required_emp': [13, 12, 15, 10],
        'subdiv_competency': [3.7, 3.9, 4.0, 3.6],
        'turnover_ratio': 13.8
    }
    bu_data.append(bu1)
    
    # BU2 data
    bu2 = {
        'subdivs': ['Proker A', 'Proker B', 'Proker C'],
        'products': ['Service X', 'Service Y', 'Service Z', 'Service W'],
        'subdiv_budget': [320_000, 280_000, 350_000],
        'subdiv_expense': [310_000, 260_000, 350_000],
        'subdiv_profit': [125_000, 120_000, 135_000],
        'subdiv_profit_change': [10.8, 12.5, 9.4],
        'subdiv_revenue': [350_000, 320_000, 380_000],
        'subdiv_revenue_change': [7.8, 8.5, 6.3],
        'subdiv_financial_health': [80, 88, 75],
        'subdiv_customers': [
            [120, 140, 90, 70],  # Proker A customers by product
            [100, 120, 80, 60],  # Proker B customers by product
            [90, 80, 70, 50]     # Proker C customers by product
        ],
        'subdiv_satisfaction': [
            [76, 78, 80, 82, 83, 85],  # Proker A satisfaction trend
            [74, 75, 77, 80, 81, 83],  # Proker B satisfaction trend
            [77, 78, 80, 82, 84, 86]   # Proker C satisfaction trend
        ],
        'subdiv_target': [88, 92, 90],
        'subdiv_realization': [83, 88, 84],
        'subdiv_velocity': [20.8, 24.5, 22.0],
        'subdiv_quality': [85, 90, 88],
        'subdiv_current_emp': [18, 16, 18],
        'subdiv_required_emp': [20, 18, 17],
        'subdiv_competency': [4.0, 4.3, 3.9],
        'turnover_ratio': 15.2
    }
    bu_data.append(bu2)
    
    # BU3 data
    bu3 = {
        'subdivs': ['Proker X', 'Proker Y', 'Proker Z', 'Proker W', 'Proker V'],
        'products': ['Product A', 'Product B', 'Product C'],
        'subdiv_budget': [320_000, 280_000, 350_000, 300_000, 250_000],
        'subdiv_expense': [290_000, 250_000, 320_000, 280_000, 210_000],
        'subdiv_profit': [140_000, 120_000, 150_000, 130_000, 110_000],
        'subdiv_profit_change': [14.2, 10.5, 16.8, 12.3, 9.5],
        'subdiv_revenue': [380_000, 320_000, 410_000, 350_000, 240_000],
        'subdiv_revenue_change': [9.8, 7.5, 12.3, 8.6, 6.2],
        'subdiv_financial_health': [92, 85, 88, 90, 87],
        'subdiv_customers': [
            [180, 150, 120],  # Proker X customers by product
            [160, 140, 100],  # Proker Y customers by product
            [200, 180, 140],  # Proker Z customers by product
            [140, 120, 90],   # Proker W customers by product
            [120, 100, 80]    # Proker V customers by product
        ],
        'subdiv_satisfaction': [
            [78, 80, 82, 84, 85, 86],  # Proker X satisfaction trend
            [77, 79, 80, 82, 83, 85],  # Proker Y satisfaction trend
            [80, 82, 83, 85, 87, 88],  # Proker Z satisfaction trend
            [76, 77, 79, 81, 83, 84],  # Proker W satisfaction trend
            [75, 76, 78, 80, 82, 84]   # Proker V satisfaction trend
        ],
        'subdiv_target': [94, 90, 92, 93, 91],
        'subdiv_realization': [90, 85, 89, 91, 88],
        'subdiv_velocity': [25.6, 23.4, 26.0, 24.2, 22.8],
        'subdiv_quality': [92, 89, 91, 93, 90],
        'subdiv_current_emp': [14, 12, 15, 13, 13],
        'subdiv_required_emp': [15, 14, 15, 14, 12],
        'subdiv_competency': [3.8, 3.7, 4.2, 4.0, 3.9],
        'turnover_ratio': 12.8
    }
    bu_data.append(bu3)
    
    # Return all data in a dictionary
    return {
        'overall_budget': overall_budget,
        'overall_expense': overall_expense,
        'overall_profit': overall_profit,
        'overall_profit_change': overall_profit_change,
        'overall_revenue': overall_revenue,
        'overall_revenue_change': overall_revenue_change,
        'bu_profit_revenue': bu_profit_revenue,
        'customer_by_bu': customer_by_bu,
        'satisfaction_trend': satisfaction_trend,
        'target_realization': target_realization,
        'overall_velocity': overall_velocity,
        'overall_quality': overall_quality,
        'manpower': manpower,
        'competency': competency,
        'turnover_ratio': turnover_ratio,
        'bu_data': bu_data
    }
