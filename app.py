import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Set up paths safely
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from modeling import load_data

st.set_page_config(page_title="EduPro Dashboard", layout="wide")
st.title("EduPro | Course Demand & Revenue Forecasting")
st.caption("Decision-support prototype using the supplied 2025 course and transaction data")

# Load data safely
summary_df, model = load_data()

if summary_df.empty:
    st.error("Data files could not be loaded. Please check your data directory.")
else:
    # Sidebar filters with safe dynamic column picking
    st.sidebar.header("Dashboard Filters")
    
    exp_col = next((c for c in summary_df.columns if 'expertise' in c.lower()), None)
    expertise_list = sorted(summary_df[exp_col].dropna().unique()) if exp_col else []
    expertise_list = ["All"] + list(expertise_list)
    selected_exp = st.sidebar.selectbox("Instructor expertise", expertise_list)
    
    lvl_col = next((c for c in summary_df.columns if 'level' in c.lower()), None)
    level_list = sorted(summary_df[lvl_col].dropna().unique()) if lvl_col else []
    level_list = ["All"] + list(level_list)
    selected_lvl = st.sidebar.selectbox("Course level", level_list)
    
    # Filter data
    filtered_df = summary_df.copy()
    if selected_exp != "All" and exp_col:
        filtered_df = filtered_df[filtered_df[exp_col] == selected_exp]
    if selected_lvl != "All" and lvl_col:
        filtered_df = filtered_df[filtered_df[lvl_col] == selected_lvl]
        
    # Main metrics layout
    st.header("Overview Metrics")
    m1, m2, m3 = st.columns(3)
    
    m1.metric("Total Courses", len(filtered_df))
    
    enroll_col = next((c for c in filtered_df.columns if 'enrollment' in c.lower() or 'count' in c.lower()), None)
    total_enroll = filtered_df[enroll_col].sum() if enroll_col else 0
    m2.metric("Total Enrollments", f"{int(total_enroll):,}")
    
    rev_col = next((c for c in filtered_df.columns if 'revenue' in c.lower() or 'fee' in c.lower()), None)
    total_rev = filtered_df[rev_col].sum() if rev_col else 0
    m3.metric("Observed Revenue", f"₹{total_rev:,.0f}")
    
    # Data display
    st.subheader("Course Details Data")
    st.dataframe(filtered_df, use_container_width=True)
