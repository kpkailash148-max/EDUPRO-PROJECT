from pathlib import Path
import pandas as pd
import joblib

def load_data():
    """
    Loads data and features for the Streamlit dashboard app.
    """
    ROOT = Path(__file__).resolve().parent.parent
    
    # Load the processed summary data
    try:
        summary_df = pd.read_csv(ROOT / 'data' / 'course_level_summary.csv')
    except Exception:
        summary_df = pd.DataFrame()
        
    # Load your trained model if it exists
    try:
        model = joblib.load(ROOT / 'outputs' / 'best_models.json') # or your active model format
    except Exception:
        model = None
        
    return summary_df, model

