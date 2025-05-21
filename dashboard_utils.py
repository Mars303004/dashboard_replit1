import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd

def create_donut_chart(budget_data, expense_data):
    """
    Create a donut chart showing budget vs expense.
    
    Args:
        budget_data (dict): Dictionary with budget values by category
        expense_data (dict): Dictionary with expense values by category
        
    Returns:
        plotly.graph_objects.Figure: The donut chart figure
    """
    # Calculate total usage percentage
    total_budget = sum(budget_data.values())
    total_expense = sum(expense_data.values())
    usage_percentage = (total_expense / total_budget) * 100 if total_budget > 0 else 0
    
    # Create a DataFrame for the chart
    data = []
    
    for category in budget_data:
        data.append({
            'Category': f"{category} Budget",
            'Value': budget_data[category]
        })
        data.append({
            'Category': f"{category} Expense",
            'Value': expense_data[category]
        })
    
    df = pd.DataFrame(data)
    
    # Create the donut chart
    fig = px.pie(
        df, 
        values='Value', 
        names='Category', 
        hole=0.6,
        color_discrete_sequence=px.colors.sequential.Blues_r
    )
    
    # Add the usage percentage in the center
    fig.add_annotation(
        text=f"{usage_percentage:.1f}%<br>Usage",
        x=0.5, y=0.5,
        font_size=20,
        showarrow=False
    )
    
    fig.update_layout(
        title="Budget vs Expense",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    
    return fig

def create_pie_chart(data, values, names, title):
    """
    Create a pie chart.
    
    Args:
        data (pd.DataFrame): DataFrame with data for the chart
        values (str): Column name for values
        names (str): Column name for names/categories
        title (str): Chart title
        
    Returns:
        plotly.graph_objects.Figure: The pie chart figure
    """
    fig = px.pie(
        data, 
        values=values, 
        names=names, 
        title=title,
        color_discrete_sequence=px.colors.sequential.Blues_r
    )
    
    return fig

def create_line_chart(data, x, y, color, title):
    """
    Create a line chart.
    
    Args:
        data (pd.DataFrame): DataFrame with data for the chart
        x (str): Column name for x-axis
        y (str): Column name for y-axis
        color (str): Column name for color
        title (str): Chart title
        
    Returns:
        plotly.graph_objects.Figure: The line chart figure
    """
    fig = px.line(
        data, 
        x=x, 
        y=y, 
        color=color, 
        title=title,
        markers=True,
        line_shape='spline'
    )
    
    fig.update_layout(
        xaxis_title=x.capitalize(),
        yaxis_title=y.capitalize()
    )
    
    return fig

def create_scorecard(title, value, change, prefix="", suffix=""):
    """
    Create a scorecard with a metric and delta indicator.
    
    Args:
        title (str): Scorecard title
        value (float): Metric value
        change (float): Percentage change
        prefix (str): Prefix for the value (e.g., "$")
        suffix (str): Suffix for the value (e.g., "%")
    """
    formatted_value = f"{prefix}{value:,.0f}{suffix}"
    
    if change != 0:
        delta = f"{change:+.1f}% from last month"
    else:
        delta = None
    
    st.metric(
        label=title,
        value=formatted_value,
        delta=delta
    )
