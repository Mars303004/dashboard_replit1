import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
from PIL import Image
from data_generator import generate_mock_data
from dashboard_utils import (
    create_donut_chart, 
    create_line_chart, 
    create_pie_chart, 
    create_scorecard
)

# Set page configuration
st.set_page_config(
    page_title="Business Performance Dashboard",
    page_icon="📊",
    layout="wide"
)

def load_excel_data():
    """
    Load data from the Excel file, specifically from Sheet1 and Sheet2BU1.
    Sheet1 contains the overall BU data, and Sheet2BU1 contains the BU1-specific data.
    """
    try:
        if os.path.exists("attached_assets/Test replit.xlsx"):
            # Read Sheet1 for Overall BU data (Table "BU")
            df_overall = pd.read_excel("attached_assets/Test replit.xlsx", sheet_name="Sheet1")
            
            # Read Sheet2BU1 for BU1 detailed data (Table "53")
            df_bu1 = pd.read_excel("attached_assets/Test replit.xlsx", sheet_name="Sheet2BU1")
            
            # Process both sheets' data
            return process_excel_data(df_overall, df_bu1)
        else:
            st.error("Excel file not found")
            return generate_mock_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return generate_mock_data()
        
def process_excel_data(df_overall, df_bu1):
    """
    Process the loaded Excel data into the format needed for the dashboard.
    df_overall: Data from Sheet1 (Overall BU)
    df_bu1: Data from Sheet2BU1 (BU1 detailed)
    """
    # Initialize the data structure
    data = {}
    
    # Process Overall BU data from Sheet1
    # Extract rows for February and January
    feb_data_overall = df_overall[df_overall['Unnamed: 19'] == '28/02/2025']
    jan_data_overall = df_overall[df_overall['Unnamed: 19'] == '31/01/2025']
    
    # Extract budget and expense data for each BU from February data
    # The structure of Sheet1 has fixed columns for each BU
    budget_bu1 = feb_data_overall['Unnamed: 1'].sum()  # Budget BU1
    expense_bu1 = feb_data_overall['Unnamed: 2'].sum()  # Expense BU1
    
    budget_bu2 = feb_data_overall['Unnamed: 7'].sum()  # Budget BU2
    expense_bu2 = feb_data_overall['Unnamed: 8'].sum()  # Expense BU2
    
    budget_bu3 = feb_data_overall['Unnamed: 13'].sum()  # Budget BU3
    expense_bu3 = feb_data_overall['Unnamed: 14'].sum()  # Expense BU3
    
    overall_budget = {
        'BU1': budget_bu1,
        'BU2': budget_bu2,
        'BU3': budget_bu3
    }
    
    overall_expense = {
        'BU1': expense_bu1,
        'BU2': expense_bu2,
        'BU3': expense_bu3
    }
    
    # Extract revenue and profit for each BU
    revenue_bu1 = feb_data_overall['Unnamed: 3'].sum()  # Revenue BU1
    profit_bu1 = feb_data_overall['Unnamed: 4'].sum()   # Profit BU1
    
    revenue_bu2 = feb_data_overall['Unnamed: 9'].sum()  # Revenue BU2
    profit_bu2 = feb_data_overall['Unnamed: 10'].sum()  # Profit BU2
    
    revenue_bu3 = feb_data_overall['Unnamed: 15'].sum() # Revenue BU3
    profit_bu3 = feb_data_overall['Unnamed: 16'].sum()  # Profit BU3
    
    # Get January data for comparison
    revenue_bu1_jan = jan_data_overall['Unnamed: 3'].sum()
    profit_bu1_jan = jan_data_overall['Unnamed: 4'].sum()
    
    revenue_bu2_jan = jan_data_overall['Unnamed: 9'].sum()
    profit_bu2_jan = jan_data_overall['Unnamed: 10'].sum()
    
    revenue_bu3_jan = jan_data_overall['Unnamed: 15'].sum()
    profit_bu3_jan = jan_data_overall['Unnamed: 16'].sum()
    
    # Calculate changes
    revenue_bu1_change = ((revenue_bu1 / revenue_bu1_jan) - 1) * 100 if revenue_bu1_jan > 0 else 0
    profit_bu1_change = ((profit_bu1 / profit_bu1_jan) - 1) * 100 if profit_bu1_jan > 0 else 0
    
    revenue_bu2_change = ((revenue_bu2 / revenue_bu2_jan) - 1) * 100 if revenue_bu2_jan > 0 else 0
    profit_bu2_change = ((profit_bu2 / profit_bu2_jan) - 1) * 100 if profit_bu2_jan > 0 else 0
    
    revenue_bu3_change = ((revenue_bu3 / revenue_bu3_jan) - 1) * 100 if revenue_bu3_jan > 0 else 0
    profit_bu3_change = ((profit_bu3 / profit_bu3_jan) - 1) * 100 if profit_bu3_jan > 0 else 0
    
    # Calculate overall metrics
    overall_revenue = revenue_bu1 + revenue_bu2 + revenue_bu3
    overall_revenue_jan = revenue_bu1_jan + revenue_bu2_jan + revenue_bu3_jan
    
    overall_profit = profit_bu1 + profit_bu2 + profit_bu3
    overall_profit_jan = profit_bu1_jan + profit_bu2_jan + profit_bu3_jan
    
    # Calculate overall changes
    overall_revenue_change = ((overall_revenue / overall_revenue_jan) - 1) * 100 if overall_revenue_jan > 0 else 0
    overall_profit_change = ((overall_profit / overall_profit_jan) - 1) * 100 if overall_profit_jan > 0 else 0
    
    # Profit and revenue breakdown
    bu_profit_revenue = pd.DataFrame([
        {'category': 'BU1 Profit', 'value': profit_bu1},
        {'category': 'BU2 Profit', 'value': profit_bu2},
        {'category': 'BU3 Profit', 'value': profit_bu3},
        {'category': 'BU1 Revenue', 'value': revenue_bu1},
        {'category': 'BU2 Revenue', 'value': revenue_bu2},
        {'category': 'BU3 Revenue', 'value': revenue_bu3}
    ])
    
    # Customer data (for Overall BU)
    # Since we don't have customer data in Sheet1, we'll create representative data
    # based on revenue proportions
    total_customers = 3650  # Example total
    bu1_customers = int(total_customers * (revenue_bu1 / overall_revenue))
    bu2_customers = int(total_customers * (revenue_bu2 / overall_revenue))
    bu3_customers = total_customers - bu1_customers - bu2_customers  # Ensure they sum to total
    
    customer_by_bu = pd.DataFrame([
        {'bu': 'BU1', 'customers': bu1_customers},
        {'bu': 'BU2', 'customers': bu2_customers},
        {'bu': 'BU3', 'customers': bu3_customers}
    ])
    
    # Customer satisfaction trends
    # For satisfaction trends, we'll use a consistent dataset
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    satisfaction_trend = pd.DataFrame()
    
    # Set a seed for reproducibility
    np.random.seed(42)  
    
    for bu in ['BU1', 'BU2', 'BU3']:
        base = 75 + np.random.randint(0, 10)
        trend = base + np.cumsum(np.random.normal(0.5, 1, len(months)))
        trend = np.clip(trend, 65, 95)
        
        for i, month in enumerate(months):
            satisfaction_trend = pd.concat([
                satisfaction_trend, 
                pd.DataFrame({'month': [month], 'bu': [bu], 'satisfaction': [trend[i]]})
            ])
    
    # Now process BU1-specific data from Sheet2BU1
    # Process the headers in the first row
    headers = df_bu1.iloc[0].values
    bu1_df = df_bu1.copy()
    bu1_df.columns = headers
    bu1_df = bu1_df.iloc[1:].reset_index(drop=True)
    
    # Extract data for February and January
    feb_data_bu1 = bu1_df[bu1_df['Bulan'] == '28/02/2025']
    jan_data_bu1 = bu1_df[bu1_df['Bulan'] == '31/01/2025']
    
    # Extract quality data
    quality_data = feb_data_bu1[feb_data_bu1['Perspective'] == 'Quality']
    
    # Calculate quality metrics for Overall BU
    target_realization = {
        'BU1': {
            'target': quality_data['Target'].mean(),
            'realization': quality_data['Realization'].mean()
        },
        'BU2': {
            'target': 92,  # Example data
            'realization': 87
        },
        'BU3': {
            'target': 95,
            'realization': 91
        }
    }
    
    overall_velocity = quality_data['Velocity'].mean()
    overall_quality = quality_data['Quality'].mean()
    
    # Extract employee data
    employee_data = feb_data_bu1[feb_data_bu1['Perspective'] == 'Employee']
    
    # Calculate employee metrics
    current_mp_bu1 = employee_data['Current MP'].sum()
    needed_mp_bu1 = employee_data['Needed MP'].sum()
    
    # Create representative data for BU2 and BU3 based on their relative sizes
    current_mp_bu2 = int(current_mp_bu1 * (revenue_bu2 / revenue_bu1))
    needed_mp_bu2 = int(needed_mp_bu1 * (revenue_bu2 / revenue_bu1))
    
    current_mp_bu3 = int(current_mp_bu1 * (revenue_bu3 / revenue_bu1))
    needed_mp_bu3 = int(needed_mp_bu1 * (revenue_bu3 / revenue_bu1))
    
    manpower = {
        'current': {'BU1': current_mp_bu1, 'BU2': current_mp_bu2, 'BU3': current_mp_bu3},
        'required': {'BU1': needed_mp_bu1, 'BU2': needed_mp_bu2, 'BU3': needed_mp_bu3}
    }
    
    competency_bu1 = employee_data['Competency'].mean()
    
    competency = {
        'BU1': competency_bu1,
        'BU2': competency_bu1 * 1.05,  # Slightly higher than BU1
        'BU3': competency_bu1 * 0.98   # Slightly lower than BU1
    }
    
    turnover_ratio = employee_data['Turnover ratio'].mean() * 100
    
    # BU-specific data
    bu_data = []
    
    # Process BU1 detailed data
    # Get financial data for BU1
    bu1_financial = feb_data_bu1[feb_data_bu1['Perspective'] == 'Financial']
    bu1_jan_financial = jan_data_bu1[jan_data_bu1['Perspective'] == 'Financial']
    
    # Get all unique subdivs
    subdivs = bu1_financial['Subdiv'].unique().tolist()
    
    # Initialize BU1 data structure
    bu1 = {
        'subdivs': subdivs,
        'subdiv_budget': [],
        'subdiv_expense': [],
        'subdiv_usage': [],
        'subdiv_profit': [],
        'subdiv_profit_change': [],
        'subdiv_revenue': [],
        'subdiv_revenue_change': [],
        'subdiv_target': [],
        'subdiv_realization': [],
        'subdiv_velocity': [],
        'subdiv_quality': [],
        'subdiv_current_emp': [],
        'subdiv_required_emp': [],
        'subdiv_competency': [],
        'turnover_ratio': turnover_ratio
    }
    
    # Populate BU1 data by subdiv
    for subdiv in subdivs:
        # Financial data
        subdiv_feb = bu1_financial[bu1_financial['Subdiv'] == subdiv]
        subdiv_jan = bu1_jan_financial[bu1_jan_financial['Subdiv'] == subdiv]
        
        bu1['subdiv_budget'].append(subdiv_feb['Budget'].sum())
        bu1['subdiv_expense'].append(subdiv_feb['Expense'].sum())
        bu1['subdiv_usage'].append(subdiv_feb['Usage'].mean())
        bu1['subdiv_profit'].append(subdiv_feb['Profit'].sum())
        
        # Calculate profit change
        profit_feb = subdiv_feb['Profit'].sum()
        profit_jan = subdiv_jan['Profit'].sum() if not subdiv_jan.empty else 0
        profit_change = ((profit_feb / profit_jan) - 1) * 100 if profit_jan > 0 else 0
        bu1['subdiv_profit_change'].append(profit_change)
        
        # Revenue data
        bu1['subdiv_revenue'].append(subdiv_feb['Revenue'].sum())
        
        # Calculate revenue change
        revenue_feb = subdiv_feb['Revenue'].sum()
        revenue_jan = subdiv_jan['Revenue'].sum() if not subdiv_jan.empty else 0
        revenue_change = ((revenue_feb / revenue_jan) - 1) * 100 if revenue_jan > 0 else 0
        bu1['subdiv_revenue_change'].append(revenue_change)
        
        # Quality data
        subdiv_quality = quality_data[quality_data['Subdiv'] == subdiv]
        bu1['subdiv_target'].append(subdiv_quality['Target'].mean() if not subdiv_quality.empty else 0)
        bu1['subdiv_realization'].append(subdiv_quality['Realization'].mean() if not subdiv_quality.empty else 0)
        bu1['subdiv_velocity'].append(subdiv_quality['Velocity'].mean() if not subdiv_quality.empty else 0)
        bu1['subdiv_quality'].append(subdiv_quality['Quality'].mean() if not subdiv_quality.empty else 0)
        
        # Employee data
        subdiv_emp = employee_data[employee_data['Subdiv'] == subdiv]
        bu1['subdiv_current_emp'].append(subdiv_emp['Current MP'].sum() if not subdiv_emp.empty else 0)
        bu1['subdiv_required_emp'].append(subdiv_emp['Needed MP'].sum() if not subdiv_emp.empty else 0)
        bu1['subdiv_competency'].append(subdiv_emp['Competency'].mean() if not subdiv_emp.empty else 0)
    
    bu_data.append(bu1)
    
    # BU2 and BU3 data - use the mock data from the generator
    bu2_mock = generate_mock_data()['bu_data'][1]
    bu3_mock = generate_mock_data()['bu_data'][2]
    
    bu_data.append(bu2_mock)
    bu_data.append(bu3_mock)
    
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

# Custom CSS for layout styling
st.markdown("""
<style>
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px;
        background-color: #F0F2F6;
        padding: 0px 10px;
        border-radius: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        white-space: pre-wrap;
        background-color: #F0F2F6;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding: 10px 20px;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: white;
        color: #4169E1;
        border-bottom: 2px solid #4169E1;
    }
    
    /* Card styling */
    .metric-card {
        background-color: white;
        border-radius: 5px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 20px;
        margin-bottom: 20px;
    }
    
    /* Filter bar styling */
    .filter-container {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04);
        margin-bottom: 20px;
    }
    
    /* Other UI improvements */
    h1, h2, h3, h4, h5, h6 {
        color: #262730;
    }
    
    .main-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #262730;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .section-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #262730;
        margin-bottom: 0.7rem;
    }
    
    /* Hide menu and footer */
    #MainMenu, footer, header {
        visibility: hidden;
    }

    /* Add padding */
    .block-container {
        padding: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Load data from the Excel file
try:
    data = load_excel_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Set up the main dashboard layout
st.markdown("<h1 class='main-title'>Business Performance Dashboard</h1>", unsafe_allow_html=True)

# Add date filter at the top
with st.container():
    st.markdown('<div class="filter-container">', unsafe_allow_html=True)
    filter_cols = st.columns([1, 1, 1, 1])
    
    with filter_cols[0]:
        st.selectbox("Date Range", ["Feb 01-28, 2025", "Jan 01-31, 2025", "Custom Range"], index=0)
    with filter_cols[1]:
        st.selectbox("Business Unit", ["All BUs", "BU1", "BU2", "BU3"], index=0)
    with filter_cols[2]:
        st.selectbox("Report Type", ["Detailed", "Summary", "Comparison"], index=0)
    with filter_cols[3]:
        st.selectbox("Export", ["PDF", "Excel", "CSV", "Print"], index=0)
        
    st.markdown('</div>', unsafe_allow_html=True)

# Main tabs at the top
main_tabs = st.tabs(["Overall BU Performance", "BU1", "BU2", "BU3"])

# ====== Tab 1: Overall BU Performance ======
with main_tabs[0]:
        # Create expandable sections
            with st.expander("Financial", expanded=False):
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                
                # Create two columns layout for this card
                fin_cols = st.columns(2)
                
                # First column - Budget vs Expense chart with gauge
                with fin_cols[0]:
                    st.markdown("<h4>Total Budget and Expense (February)</h4>", unsafe_allow_html=True)
                    fig = create_donut_chart(data['overall_budget'], data['overall_expense'])
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add gauge for average usage
                    usage_bu1 = data['overall_expense']['BU1'] / data['overall_budget']['BU1'] * 100 if data['overall_budget']['BU1'] else 0
                    usage_bu2 = data['overall_expense']['BU2'] / data['overall_budget']['BU2'] * 100 if data['overall_budget']['BU2'] else 0
                    usage_bu3 = data['overall_expense']['BU3'] / data['overall_budget']['BU3'] * 100 if data['overall_budget']['BU3'] else 0
                    avg_usage = (usage_bu1 + usage_bu2 + usage_bu3) / 3
                    
                    # Create gauge chart for average usage
                    gauge_fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=avg_usage,
                        title={'text': "Average Usage (%)"},
                        gauge={
                            'axis': {'range': [0, 150]},
                            'steps': [
                                {'range': [0, 85], 'color': "lightgreen"},
                                {'range': [85, 100], 'color': "lightyellow"},
                                {'range': [100, 150], 'color': "lightcoral"},
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 100}
                        }
                    ))
                    st.plotly_chart(gauge_fig, use_container_width=True)
                
                # Second column - Profit and Revenue scorecards
                with fin_cols[1]:
                    st.markdown("<h4>Profit & Revenue (February)</h4>", unsafe_allow_html=True)
                    # Display metrics directly
                    st.metric("Profit", f"${data['overall_profit']:,.0f}", f"{data['overall_profit_change']:+.1f}% from January")
                    st.metric("Revenue", f"${data['overall_revenue']:,.0f}", f"{data['overall_revenue_change']:+.1f}% from January")
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            with st.expander("Customer & Service", expanded=False):
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                
                # Create two columns layout
                cust_cols = st.columns(2)
                
                # First column - Customer Distribution
                with cust_cols[0]:
                    st.markdown("<h4>Total Number of Customers (February)</h4>", unsafe_allow_html=True)
                    
                    # Customer total from all BUs
                    total_customers = data['customer_by_bu']['customers'].sum()
                    st.metric("Total Customers", f"{total_customers:,}")
                    
                    # Create pie chart for customer distribution
                    fig = create_pie_chart(
                        data['customer_by_bu'],
                        values='customers',
                        names='bu',
                        title="Customer Distribution by Business Unit"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Second column - Customer Satisfaction
                with cust_cols[1]:
                    st.markdown("<h4>Average Customer Satisfaction</h4>", unsafe_allow_html=True)
                    
                    # Filter February data
                    feb_satisfaction = data['satisfaction_trend'][data['satisfaction_trend']['month'] == 'Feb']
                    jan_satisfaction = data['satisfaction_trend'][data['satisfaction_trend']['month'] == 'Jan']
                    
                    # Calculate average satisfaction for February
                    avg_satisfaction_feb = feb_satisfaction['satisfaction'].mean()
                    avg_satisfaction_jan = jan_satisfaction['satisfaction'].mean()
                    satisfaction_change = avg_satisfaction_feb - avg_satisfaction_jan
                    
                    # Display scorecard with comparison
                    st.metric(
                        "Average Satisfaction", 
                        f"{avg_satisfaction_feb:.1f}%", 
                        f"{satisfaction_change:+.1f}% from January"
                    )
                    
                    # Create line chart for satisfaction trend
                    fig = create_line_chart(
                        data['satisfaction_trend'],
                        x='month',
                        y='satisfaction',
                        color='bu',
                        title="Customer Satisfaction Trend by Business Unit"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            with st.expander("Quality", expanded=False):
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                
                # Target vs Realization table
                st.markdown("<h4>Average Target vs Realization (February)</h4>", unsafe_allow_html=True)
                
                # Create a DataFrame for the table
                target_real_df = pd.DataFrame({
                    'Business Unit': list(data['target_realization'].keys()),
                    'Target': [data['target_realization'][bu]['target'] for bu in data['target_realization']],
                    'Realization': [data['target_realization'][bu]['realization'] for bu in data['target_realization']],
                    'Achievement (%)': [data['target_realization'][bu]['realization'] / data['target_realization'][bu]['target'] * 100 
                                      if data['target_realization'][bu]['target'] else 0 
                                      for bu in data['target_realization']]
                })
                
                # Format the Achievement column
                target_real_df['Achievement (%)'] = target_real_df['Achievement (%)'].map('{:.1f}%'.format)
                
                # Display the table
                st.table(target_real_df)
                
                # Create two columns for velocity and quality metrics
                qual_cols = st.columns(2)
                
                with qual_cols[0]:
                    st.metric("Average Velocity", f"{data['overall_velocity']:.2f}")
                
                with qual_cols[1]:
                    st.metric("Average Quality", f"{data['overall_quality']:.2f}")
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            with st.expander("Employee", expanded=False):
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                
                # Manpower section
                st.markdown("<h4>Manpower</h4>", unsafe_allow_html=True)
                
                # Calculate manpower metrics
                total_current = sum(data['manpower']['current'].values())
                total_needed = sum(data['manpower']['required'].values())
                mp_gap = total_current - total_needed
                
                # Create three columns for manpower metrics
                mp_cols = st.columns(3)
                
                with mp_cols[0]:
                    st.metric("Current MP", f"{total_current}")
                
                with mp_cols[1]:
                    st.metric("MP Needed", f"{total_needed}")
                
                with mp_cols[2]:
                    st.metric("Gap", f"{mp_gap}", delta=f"{mp_gap}")
                
                # Competency section
                st.markdown("<h4>Competency (February)</h4>", unsafe_allow_html=True)
                
                # Create a pie chart for competency
                competency_df = pd.DataFrame({
                    'Business Unit': list(data['competency'].keys()),
                    'Competency': list(data['competency'].values())
                })
                
                # Calculate average competency
                avg_competency = competency_df['Competency'].mean()
                
                # Create pie chart with text in the center
                fig = px.pie(
                    competency_df,
                    values='Competency',
                    names='Business Unit',
                    hole=0.6,
                    color_discrete_sequence=px.colors.sequential.Blues_r
                )
                
                # Add the average competency in the center
                fig.add_annotation(
                    text=f"Avg: {avg_competency:.2f}",
                    x=0.5, y=0.5,
                    font_size=20,
                    showarrow=False
                )
                
                fig.update_layout(title="Competency by Business Unit")
                st.plotly_chart(fig, use_container_width=True)
                
                # Turnover Ratio section
                st.markdown("<h4>Turnover Ratio</h4>", unsafe_allow_html=True)
                
                # Display turnover ratio as a gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=data['turnover_ratio'],
                    title={'text': "Overall Turnover Ratio (%)"},
                    number={'suffix': "%"},
                    gauge={
                        'axis': {'range': [0, 30]},
                        'steps': [
                            {'range': [0, 5], 'color': "lightgreen"},
                            {'range': [5, 15], 'color': "lightyellow"},
                            {'range': [15, 30], 'color': "lightcoral"},
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': data['turnover_ratio']}
                    }
                ))
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
        
        # ----- BU1 Tab -----
        with tabs[1]:
            st.info("Business Unit 1 - See in the BU1 tab from the main navigation")
        
        # ----- BU2 Tab -----
        with tabs[2]:
            st.info("Business Unit 2 - See in the BU2 tab from the main navigation")
        
        # ----- BU3 Tab -----
        with tabs[3]:
            st.info("Business Unit 3 - See in the BU3 tab from the main navigation")
    
    # ====== Business Unit Specific Dashboards ======
    # BU specific dashboards in main tabs
    with main_tabs[1]:  # BU1
        # Get the BU1 data
        bu_data = data['bu_data'][0]
        
        if selected_tab == "BU1":
            # Create tabs for different metrics categories
            bu1_tabs = st.tabs(["Financial", "Customer", "Quality", "Employee"])
            
            # ----- Financial Tab -----
            with bu1_tabs[0]:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.markdown("<h3 class='section-title'>Financial Performance</h3>", unsafe_allow_html=True)
                
                # Create selectbox for subdiv selection
                fin_subdiv_selected = st.selectbox(
                    "Select Subdivision",
                    bu_data['subdivs'],
                    key="financial_subdiv"
                )
                
                # Get the index of the selected subdiv
                fin_subdiv_idx = bu_data['subdivs'].index(fin_subdiv_selected)
                
                # Display financial data for selected subdiv
                fin_cols = st.columns(2)
                
                with fin_cols[0]:
                    # Show Budget vs Expense with gauge
                    st.markdown("<h4>Budget vs Expense (February)</h4>", unsafe_allow_html=True)
                    
                    budget = bu_data['subdiv_budget'][fin_subdiv_idx]
                    expense = bu_data['subdiv_expense'][fin_subdiv_idx]
                    usage = bu_data['subdiv_usage'][fin_subdiv_idx] * 100  # Convert to percentage
                    
                    # Create bar chart for budget vs expense
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        x=['Budget', 'Expense'],
                        y=[budget, expense],
                        marker_color=['lightblue', 'royalblue']
                    ))
                    
                    fig.update_layout(
                        title=f"Budget vs Expense for {fin_subdiv_selected}",
                        yaxis_title="Amount"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Create gauge for usage
                    gauge_fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=usage,
                        title={'text': "Usage (%)"},
                        number={'suffix': "%"},
                        gauge={
                            'axis': {'range': [0, 150]},
                            'steps': [
                                {'range': [0, 85], 'color': "lightgreen"},
                                {'range': [85, 100], 'color': "lightyellow"},
                                {'range': [100, 150], 'color': "lightcoral"},
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 100}
                        }
                    ))
                    
                    st.plotly_chart(gauge_fig, use_container_width=True)
                
                with fin_cols[1]:
                    # Show Profit and Revenue with comparison
                    st.markdown("<h4>Profit & Revenue</h4>", unsafe_allow_html=True)
                    
                    profit = bu_data['subdiv_profit'][fin_subdiv_idx]
                    profit_change = bu_data['subdiv_profit_change'][fin_subdiv_idx]
                    revenue = bu_data['subdiv_revenue'][fin_subdiv_idx]
                    revenue_change = bu_data['subdiv_revenue_change'][fin_subdiv_idx]
                    
                    # Display metrics
                    st.metric("Profit", f"${profit:,.0f}", f"{profit_change:+.1f}% from January")
                    st.metric("Revenue", f"${revenue:,.0f}", f"{revenue_change:+.1f}% from January")
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            # ----- Customer Tab -----
            with bu1_tabs[1]:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.markdown("<h3 class='section-title'>Customer Metrics</h3>", unsafe_allow_html=True)
                
                # For BU1, we don't have actual product data in the Excel
                # So we'll use representative products for demonstration
                products = ["Product A", "Product B", "Product C"]
                product_selected = st.selectbox(
                    "Select Product",
                    products,
                    key="customer_product"
                )
                
                # Since we don't have actual product data, we'll use the subdiv data instead
                # And create a representative visualization
                st.markdown("<h4>Total Customers by Subdivision (February)</h4>", unsafe_allow_html=True)
                
                # Create customer data by subdivision
                # We'll use the subdivision index as a basis for distribution
                customers_by_subdiv = {}
                for idx, subdiv in enumerate(bu_data['subdivs']):
                    # Use revenue data to create representative customer numbers
                    base_customers = bu_data['subdiv_revenue'][idx] / 10  # Revenue-based estimate
                    customers_by_subdiv[subdiv] = int(base_customers)
                
                # Create DataFrame for the donut chart
                customer_df = pd.DataFrame({
                    'Subdivision': list(customers_by_subdiv.keys()),
                    'Customers': list(customers_by_subdiv.values())
                })
                
                # Create two columns layout
                cust_cols = st.columns(2)
                
                with cust_cols[0]:
                    # Create donut chart
                    fig = px.pie(
                        customer_df,
                        values='Customers',
                        names='Subdivision',
                        hole=0.6,
                        color_discrete_sequence=px.colors.sequential.Blues_r
                    )
                    
                    # Add total customers in the center
                    total_customers = customer_df['Customers'].sum()
                    fig.add_annotation(
                        text=f"Total: {total_customers}",
                        x=0.5, y=0.5,
                        font_size=20,
                        showarrow=False
                    )
                    
                    fig.update_layout(title="Customer Distribution by Subdivision")
                    st.plotly_chart(fig, use_container_width=True)
                
                with cust_cols[1]:
                    # Show customer satisfaction with comparison to previous month
                    st.markdown("<h4>Customer Satisfaction</h4>", unsafe_allow_html=True)
                    
                    # Calculate customer satisfaction from the satisfaction trend data
                    bu1_satisfaction = data['satisfaction_trend'][data['satisfaction_trend']['bu'] == 'BU1']
                    feb_satisfaction = bu1_satisfaction[bu1_satisfaction['month'] == 'Feb']['satisfaction'].values[0]
                    jan_satisfaction = bu1_satisfaction[bu1_satisfaction['month'] == 'Jan']['satisfaction'].values[0]
                    satisfaction_change = feb_satisfaction - jan_satisfaction
                    
                    # Display scorecard
                    st.metric(
                        "Average Customer Satisfaction", 
                        f"{feb_satisfaction:.1f}%", 
                        f"{satisfaction_change:+.1f}% from January"
                    )
                    
                    # Create line chart for satisfaction trend
                    fig = px.line(
                        bu1_satisfaction,
                        x='month',
                        y='satisfaction',
                        markers=True,
                        title="Customer Satisfaction Trend"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            # ----- Quality Tab -----
            with bu1_tabs[2]:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.markdown("<h3 class='section-title'>Quality Metrics</h3>", unsafe_allow_html=True)
                
                # Create selectbox for subdiv selection
                quality_subdiv_selected = st.selectbox(
                    "Select Subdivision",
                    bu_data['subdivs'],
                    key="quality_subdiv"
                )
                
                # Get the index of the selected subdiv
                quality_subdiv_idx = bu_data['subdivs'].index(quality_subdiv_selected)
                
                # Target vs Realization table
                st.markdown("<h4>Target vs Realization</h4>", unsafe_allow_html=True)
                
                target = bu_data['subdiv_target'][quality_subdiv_idx]
                realization = bu_data['subdiv_realization'][quality_subdiv_idx]
                achievement = (realization / target * 100) if target > 0 else 0
                
                # Create a table
                quality_table = pd.DataFrame({
                    'Metric': ['Target', 'Realization', 'Achievement (%)'],
                    'Value': [f"{target:.1f}", f"{realization:.1f}", f"{achievement:.1f}%"]
                })
                
                st.table(quality_table)
                
                # Quality metrics
                st.markdown("<h4>Quality Metrics</h4>", unsafe_allow_html=True)
                
                # Create two columns for velocity and quality
                qual_cols = st.columns(2)
                
                with qual_cols[0]:
                    velocity = bu_data['subdiv_velocity'][quality_subdiv_idx]
                    st.metric("Velocity", f"{velocity:.2f}")
                
                with qual_cols[1]:
                    quality = bu_data['subdiv_quality'][quality_subdiv_idx]
                    st.metric("Quality", f"{quality:.2f}")
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            # ----- Employee Tab -----
            with bu1_tabs[3]:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.markdown("<h3 class='section-title'>Employee Metrics</h3>", unsafe_allow_html=True)
                
                # Create selectbox for subdiv selection
                emp_subdiv_selected = st.selectbox(
                    "Select Subdivision",
                    bu_data['subdivs'],
                    key="employee_subdiv"
                )
                
                # Get the index of the selected subdiv
                emp_subdiv_idx = bu_data['subdivs'].index(emp_subdiv_selected)
                
                # Manpower section
                st.markdown("<h4>Manpower</h4>", unsafe_allow_html=True)
                
                current_mp = bu_data['subdiv_current_emp'][emp_subdiv_idx]
                needed_mp = bu_data['subdiv_required_emp'][emp_subdiv_idx]
                mp_gap = current_mp - needed_mp
                
                # Create three columns for manpower metrics
                mp_cols = st.columns(3)
                
                with mp_cols[0]:
                    st.metric("Current MP", f"{current_mp}")
                
                with mp_cols[1]:
                    st.metric("MP Needed", f"{needed_mp}")
                
                with mp_cols[2]:
                    st.metric("Gap", f"{mp_gap}", delta=f"{mp_gap}")
                
                # Competency section
                st.markdown("<h4>Competency</h4>", unsafe_allow_html=True)
                
                competency_score = bu_data['subdiv_competency'][emp_subdiv_idx]
                
                # Create gauge for competency
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=competency_score,
                    title={'text': "Competency Score"},
                    gauge={
                        'axis': {'range': [0, 5]},
                        'steps': [
                            {'range': [0, 2], 'color': "lightcoral"},
                            {'range': [2, 3.5], 'color': "lightyellow"},
                            {'range': [3.5, 5], 'color': "lightgreen"},
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 3.5}
                    }
                ))
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Turnover Ratio section
                st.markdown("<h4>Turnover Ratio</h4>", unsafe_allow_html=True)
                
                # We're using the overall BU1 turnover ratio
                turnover_ratio = bu_data['turnover_ratio']
                
                st.metric("Turnover Ratio", f"{turnover_ratio:.1f}%")
                
                st.markdown("</div>", unsafe_allow_html=True)
        
        elif selected_tab == "BU2":
            # BU2 tab - empty for now as per requirements
            bu2_tabs = st.tabs(["Financial", "Customer", "Quality", "Employee"])
            
            # Show info message in each tab
            with bu2_tabs[0]:
                st.info("Business Unit 2 Financial data will be implemented in the future.")
            with bu2_tabs[1]:
                st.info("Business Unit 2 Customer data will be implemented in the future.")
            with bu2_tabs[2]:
                st.info("Business Unit 2 Quality data will be implemented in the future.")
            with bu2_tabs[3]:
                st.info("Business Unit 2 Employee data will be implemented in the future.")
            
        elif selected_tab == "BU3":
            # BU3 tab - empty for now as per requirements
            bu3_tabs = st.tabs(["Financial", "Customer", "Quality", "Employee"])
            
            # Show info message in each tab
            with bu3_tabs[0]:
                st.info("Business Unit 3 Financial data will be implemented in the future.")
            with bu3_tabs[1]:
                st.info("Business Unit 3 Customer data will be implemented in the future.")
            with bu3_tabs[2]:
                st.info("Business Unit 3 Quality data will be implemented in the future.")
            with bu3_tabs[3]:
                st.info("Business Unit 3 Employee data will be implemented in the future.")
            
            # This section is intentionally left empty - info messages already displayed in tabs
            
        else:
            st.info("Please select one of the Business Units from the navigation panel.")
                
                # Competency gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=competency,
                    title={'text': "Competency Score"},
                    gauge={
                        'axis': {'range': [0, 5]},
                        'steps': [
                            {'range': [0, 2], 'color': "pink"},
                            {'range': [2, 3.5], 'color': "lightyellow"},
                            {'range': [3.5, 5], 'color': "lightgreen"},
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': competency}
                    }
                ))
                
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Second card - Turnover ratio
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown("<h3 class='section-title'>Turnover Ratio</h3>", unsafe_allow_html=True)
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=bu_data['turnover_ratio'],
                title={'text': f"Annual Turnover Ratio for {selected_tab}"},
                gauge={
                    'axis': {'range': [0, 30]},
                    'steps': [
                        {'range': [0, 10], 'color': "lightgreen"},
                        {'range': [10, 20], 'color': "lightyellow"},
                        {'range': [20, 30], 'color': "pink"},
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': bu_data['turnover_ratio']}
                }
            ))
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)