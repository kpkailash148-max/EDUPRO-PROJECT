from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import joblib
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from src.modeling import load_data
st.set_page_config(page_title="EduPro Forecasting",page_icon="📊",layout="wide")
st.title("EduPro | Course Demand & Revenue Forecasting")
st.caption("Decision-support prototype using the supplied 2025 course and transaction data")
data,features=load_data()
summary=pd.read_csv(ROOT/'data'/'course_level_summary.csv')
cat=pd.read_csv(ROOT/'data'/'category_summary.csv')
monthly=pd.read_csv(ROOT/'data'/'monthly_summary.csv',parse_dates=['Month'])
metrics_path=ROOT/'outputs'/'model_comparison.csv'
if not metrics_path.exists():
    st.warning("Models have not been trained yet. Run `python train_models.py` in the project folder, then refresh this page.")
    st.stop()
metrics=pd.read_csv(metrics_path)
with st.sidebar:
    st.header("Course assumptions")
    category=st.selectbox("Course category",sorted(data.CourseCategory.dropna().unique()))
    course_type=st.selectbox("Course type",sorted(data.CourseType.dropna().unique()))
    level=st.selectbox("Course level",sorted(data.CourseLevel.dropna().unique()))
    price=st.number_input("Price",min_value=0.0,max_value=10000.0,value=199.0,step=10.0)
    duration=st.number_input("Duration (hours)",min_value=0.5,max_value=300.0,value=10.0,step=0.5)
    rating=st.slider("Expected course rating",min_value=1.0,max_value=5.0,value=4.0,step=0.1)
    exp=st.number_input("Instructor experience (years)",min_value=0,max_value=60,value=5)
    teacher_rating=st.slider("Instructor rating",min_value=1.0,max_value=5.0,value=4.0,step=0.1)
    teacher_age=st.number_input("Instructor age",min_value=18,max_value=90,value=35)
    expertise=st.selectbox("Instructor expertise",sorted(data.Expertise.dropna().unique()))

def derive_features(row):
    row['PriceBand']=pd.cut(pd.Series([row['CoursePrice']]),bins=[-np.inf,0,50,150,300,np.inf],labels=['Free','Low','Mid','High','Premium']).astype(str).iloc[0]
    row['DurationBucket']=pd.cut(pd.Series([row['CourseDuration']]),bins=[-np.inf,3,8,15,np.inf],labels=['Short','Medium','Long','Extended']).astype(str).iloc[0]
    row['RatingTier']=pd.cut(pd.Series([row['CourseRating']]),bins=[-np.inf,2.5,3.5,4.25,np.inf],labels=['Low','Fair','Good','Excellent']).astype(str).iloc[0]
    row['ExperienceBucket']=pd.cut(pd.Series([row['YearsOfExperience']]),bins=[-np.inf,2,5,10,np.inf],labels=['Junior','Intermediate','Experienced','Senior']).astype(str).iloc[0]
    row['ExpertiseCategoryMatch']=int(str(row['Expertise']).lower() in str(row['CourseCategory']).lower() or str(row['CourseCategory']).lower() in str(row['Expertise']).lower())
    return row

def predict(target,row):
    model_info=json.loads((ROOT/'outputs'/'best_models.json').read_text())[target]
    model=joblib.load(ROOT/'outputs'/model_info['pipeline'])
    return max(0,float(model.predict(pd.DataFrame([row])[features])[0]))
# KPI values from the cleaned supplied dataset
k1,k2,k3,k4=st.columns(4)
k1.metric("Courses",f"{len(summary):,}")
k2.metric("Transactions",f"{sum(pd.read_csv(ROOT/'data'/'transactions_cleaned.csv',usecols=['TransactionID']).shape[0] for _ in [0]):,}")
k3.metric("Observed revenue",f"₹{summary.Course_Revenue.sum():,.0f}")
k4.metric("Observed enrollments",f"{summary.Enrollment_Count.sum():,}")
tab1,tab2,tab3,tab4=st.tabs(["Predict a course","Market overview","Model evaluation","Recommendations"])
with tab1:
    row={'CourseCategory':category,'CourseType':course_type,'CourseLevel':level,'CoursePrice':price,'CourseDuration':duration,'CourseRating':rating,'YearsOfExperience':exp,'TeacherRating':teacher_rating,'Age':teacher_age,'Expertise':expertise}
    row=derive_features(row)
    e,r=st.columns(2)
    e.metric("Estimated enrollments",f"{predict('Enrollment_Count',row):,.0f}")
    r.metric("Estimated course revenue",f"₹{predict('Course_Revenue',row):,.0f}")
    st.caption("These are model estimates from a small historical dataset, not guarantees. Estimates assume course attributes are comparable to courses in the supplied data.")
with tab2:
    a,b=st.columns(2)
    with a: st.plotly_chart(px.bar(cat.sort_values('Total_Revenue',ascending=False),x='CourseCategory',y='Total_Revenue',title='Revenue by category',labels={'Total_Revenue':'Revenue (₹)','CourseCategory':'Category'}),use_container_width=True)
    with b: st.plotly_chart(px.bar(cat.sort_values('Total_Enrollments',ascending=False),x='CourseCategory',y='Total_Enrollments',title='Enrollments by category',labels={'Total_Enrollments':'Enrollments','CourseCategory':'Category'}),use_container_width=True)
    st.plotly_chart(px.line(monthly,x='Month',y='Revenue',markers=True,title='Monthly revenue in 2025'),use_container_width=True)
    st.plotly_chart(px.scatter(summary,x='CoursePrice',y='Enrollment_Count',color='CourseCategory',hover_name='CourseName',title='Course price vs enrollments'),use_container_width=True)
with tab3:
    target=st.selectbox("Target",['Enrollment_Count','Course_Revenue'],format_func=lambda x:'Enrollment demand' if x=='Enrollment_Count' else 'Course revenue')
    view=metrics[metrics.Target==target].sort_values('RMSE')
    st.dataframe(view.style.format({'MAE':'{:,.2f}','RMSE':'{:,.2f}','R2':'{:.3f}'}),use_container_width=True,hide_index=True)
    best=view.iloc[0]
    st.write(f"Lowest cross-validated RMSE: **{best['Model']}** (RMSE {best['RMSE']:,.2f}).")
    st.caption("Metrics use five-fold shuffled cross-validation across 60 courses. With a small sample, results are exploratory and should not be interpreted as production-grade validation.")
    fpath=ROOT/'outputs'/f'feature_importance_{target}.csv'
    fi=pd.read_csv(fpath).head(12)
    st.plotly_chart(px.bar(fi.sort_values('Importance'),x='Importance',y='Feature',orientation='h',title='Random Forest feature importance'),use_container_width=True)
with tab4:
    st.subheader("Evidence-led planning checklist")
    top_rev=cat.sort_values('Total_Revenue',ascending=False).iloc[0]
    top_enr=cat.sort_values('Total_Enrollments',ascending=False).iloc[0]
    st.write(f"- Review **{top_rev.CourseCategory}** for revenue patterns; it has the highest observed category revenue in the supplied data (₹{top_rev.Total_Revenue:,.0f}).")
    st.write(f"- Review **{top_enr.CourseCategory}** for reach; it has the highest observed category enrollments ({top_enr.Total_Enrollments:,}).")
    st.write("- Pilot price and duration changes on a small set of courses before broad rollout; compare forecast ranges with actual outcomes.")
    st.write("- Re-train after adding more course cohorts and a future holdout period. This dataset covers only 2025 and has 60 course-level observations.")
    st.write("- Use model predictions alongside instructor capacity, learner feedback, completion, refunds, and acquisition costs before allocating budget.")
