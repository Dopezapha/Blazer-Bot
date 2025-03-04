import logging

logger = logging.getLogger(__name__)

# Any utility functions can go here
def format_currency(value):
    """Format currency values for display"""
    return f"${value:.2f}"

def calculate_risk(capital, risk_percentage):
    """Calculate position size based on risk percentage"""
    return capital * (risk_percentage / 100)