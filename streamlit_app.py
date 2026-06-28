import streamlit as st
import pandas as pd
import numpy as np
from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTEENN
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, PowerTransformer
from category_encoders import CatBoostEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from catboost import CatBoostClassifier

# Set page configuration
st.set_page_config(page_title="HR Attrition Predictor", layout="wide", page_icon="👔")

# 1. Cached Pipeline Training function to ensure optimal performance
@st.cache_resource
def train_and_get_pipeline():
    # Load original dataset
    df = pd.read_csv("data/HR-Employee-Attrition.csv")
    
    # Feature group declarations matching your pipeline logic
    df_cat_onehot = ['BusinessTravel', 'Department']
    df_cat_ordinal = ['Gender', 'OverTime']
    df_cat_catencode = ['EducationField', 'JobRole', 'MaritalStatus']
    df_num = df.select_dtypes(include='number')
    
    # Define categorical and numerical transformations
    transform_cat_onehot = Pipeline(steps=[('cat_onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))])
    tranform_cat_ordinal = Pipeline(steps=[('cat_ordinal', OrdinalEncoder(handle_unknown='error'))])
    tranform_cat_catencode = Pipeline(steps=[('cat_catencode', CatBoostEncoder(handle_unknown='value', random_state=42, verbose=False))])
    tranform_num = Pipeline(steps=[('num_powertranform', PowerTransformer(method='yeo-johnson'))])

    # Combine transformers
    preprocess = ColumnTransformer(transformers=[
        ('num', tranform_num, df_num.columns.to_list()),
        ('cat_onehot_col', transform_cat_onehot, df_cat_onehot),
        ('cat_ordinal_col', tranform_cat_ordinal, df_cat_ordinal),
        ('cat_catencode_col', tranform_cat_catencode, df_cat_catencode)
    ], remainder='drop')

    # Feature selection matching your setup
    select_feature = SelectKBest(score_func=mutual_info_classif, k=23)

    # Base estimator and final meta-classifier setup
    estimators = [('catboost', CatBoostClassifier(iterations=1000, auto_class_weights='Balanced', learning_rate=0.0001, depth=6, loss_function='Logloss', eval_metric='Accuracy', early_stopping_rounds=50, od_type='Iter', verbose=False))]
    final_estimator_RF = RandomForestClassifier(bootstrap=True, class_weight='balanced', criterion='gini', ccp_alpha=0.0001, n_estimators=1000, n_jobs=-1, verbose=False, random_state=42, oob_score=True, max_depth=10)

    # Build the full Stacking architecture
    model_stack = StackingClassifier(estimators=estimators, final_estimator=final_estimator_RF, cv=10, n_jobs=-1, verbose=False)

    # Final execution pipeline assembly
    model = Pipeline(steps=[
        ('preprocess', preprocess),
        ('SMOTEENN', SMOTEENN(random_state=42)),
        ('select_feature', select_feature),
        ('model_stack', model_stack)                       
    ])
    
    # Explicit alignment of independent columns
    X = df[df_cat_onehot + df_cat_ordinal + df_cat_catencode + df_num.columns.tolist()]
    y = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # Fit model on baseline split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    model.fit(X_train, y_train)
    
    return model, X

# Initialize model training inside Streamlit loader spinner
with st.spinner("Initializing HR Stacking Pipeline Model... Please wait..."):
    model, X_reference = train_and_get_pipeline()

# Title banner layout
st.title("👔 Employee Flight Risk Analysis Dashboard")
st.markdown("Predict the likelihood of employee attrition using your trained CatBoost-RandomForest Stacking Classifier.")

# Create layout tabs for separate processing modes
tab1, tab2 = st.tabs(["Individual Employee Analysis", "Batch CSV Processing"])

with tab1:
    st.subheader("Enter Employee Attributes")
    
    # Split UI inputs evenly across three clean presentation columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📊 Demographics & Status")
        age = st.slider("Age", int(X_reference['Age'].min()), int(X_reference['Age'].max()), 35)
        gender = st.selectbox("Gender", X_reference['Gender'].unique())
        marital_status = st.selectbox("Marital Status", X_reference['MaritalStatus'].unique())
        education = st.slider("Education Level (1-5)", int(X_reference['Education'].min()), int(X_reference['Education'].max()), 3)
        education_field = st.selectbox("Education Field", X_reference['EducationField'].unique())
        
    with col2:
        st.markdown("### 💼 Job & Workplace Context")
        department = st.selectbox("Department", X_reference['Department'].unique())
        job_role = st.selectbox("Job Role", X_reference['JobRole'].unique())
        job_level = st.slider("Job Level", int(X_reference['JobLevel'].min()), int(X_reference['JobLevel'].max()), 2)
        job_involvement = st.slider("Job Involvement (1-4)", int(X_reference['JobInvolvement'].min()), int(X_reference['JobInvolvement'].max()), 3)
        overtime = st.selectbox("Overtime Workspace Requirement", X_reference['OverTime'].unique())
        business_travel = st.selectbox("Business Travel Frequency", X_reference['BusinessTravel'].unique())
        distance_from_home = st.slider("Distance From Home (km)", int(X_reference['DistanceFromHome'].min()), int(X_reference['DistanceFromHome'].max()), 5)

    with col3:
        st.markdown("### 💰 Compensation & History")
        monthly_income = st.number_input("Monthly Income ($)", int(X_reference['MonthlyIncome'].min()), int(X_reference['MonthlyIncome'].max()), 5000)
        percent_salary_hike = st.slider("Percent Salary Hike", int(X_reference['PercentSalaryHike'].min()), int(X_reference['PercentSalaryHike'].max()), 12)
        stock_option_level = st.slider("Stock Option Level", int(X_reference['StockOptionLevel'].min()), int(X_reference['StockOptionLevel'].max()), 0)
        total_working_years = st.slider("Total Working Years", int(X_reference['TotalWorkingYears'].min()), int(X_reference['TotalWorkingYears'].max()), 10)
        years_at_company = st.slider("Years At Company", int(X_reference['YearsAtCompany'].min()), int(X_reference['YearsAtCompany'].max()), 5)
        years_in_current_role = st.slider("Years In Current Role", int(X_reference['YearsInCurrentRole'].min()), int(X_reference['YearsInCurrentRole'].max()), 2)
        years_since_last_promotion = st.slider("Years Since Last Promotion", int(X_reference['YearsSinceLastPromotion'].min()), int(X_reference['YearsSinceLastPromotion'].max()), 1)
        years_with_curr_manager = st.slider("Years With Current Manager", int(X_reference['YearsWithCurrManager'].min()), int(X_reference['YearsWithCurrManager'].max()), 2)

    # Type-Safe Fallback: Pre-populate ALL remaining background features 
    input_data = {}
    for col in X_reference.columns:
        if pd.api.types.is_numeric_dtype(X_reference[col]):
            input_data[col] = float(X_reference[col].median())
        else:
            input_data[col] = str(X_reference[col].mode().iloc[0]) # Safely extract the raw scalar string

    # Direct UI explicit map dictionary override
    ui_mappings = {
        'Age': age, 'Gender': gender, 'MaritalStatus': marital_status, 'Education': education,
        'EducationField': education_field, 'Department': department, 'JobRole': job_role,
        'JobLevel': job_level, 'JobInvolvement': job_involvement, 'OverTime': overtime,
        'BusinessTravel': business_travel, 'DistanceFromHome': distance_from_home,
        'MonthlyIncome': monthly_income, 'PercentSalaryHike': percent_salary_hike,
        'StockOptionLevel': stock_option_level, 'TotalWorkingYears': total_working_years,
        'YearsAtCompany': years_at_company, 'YearsInCurrentRole': years_in_current_role,
        'YearsSinceLastPromotion': years_since_last_promotion, 'YearsWithCurrManager': years_with_curr_manager
    }
    
    # Merge values and assemble the prediction row DataFrame
    input_data.update(ui_mappings)
    input_df = pd.DataFrame([input_data])[X_reference.columns]

    st.markdown("---")
    if st.button("📊 Evaluate Attrition Risk", type="primary"):
        # Predict flight probability using the stacking model pipeline
        risk_probability = model.predict_proba(input_df)[0, 1]
        
        col_res1, col_res2 = st.columns([1, 2])
        with col_res1:
            st.metric(label="Flight Risk Probability", value=f"{risk_probability:.1%}")
        
        with col_res2:
            # Custom proactive threshold matching an aggressive recall metric strategy
            if risk_probability >= 0.35:
                st.error("⚠️ **High Risk Flight Candidate:** This employee crosses the risk threshold. Immediate retention planning recommended.")
            else:
                st.success("✅ **Low Risk Retention Profile:** Employee shows stable metrics within acceptable non-attrition limits.")

with tab2:
    st.subheader("Batch Process Employee List")
    st.markdown("Upload a raw CSV matching the feature format of your HR system to evaluate multiple employees instantly.")
    
    # 1. GENERATE DUMMY TEMPLATE FOR USER DOWNLOAD
    # This automatically samples 3 rows from your dataset features to create a clean template
    dummy_template_df = X_reference.head(3).copy()
    
    # Add a mock EmployeeNumber column if it wasn't part of features, as HR users usually want it
    if 'EmployeeNumber' not in dummy_template_df.columns:
        dummy_template_df.insert(0, 'EmployeeNumber', [1001, 1002, 1003])
        
    # Convert dataframe to CSV bytes
    template_csv_bytes = dummy_template_df.to_csv(index=False).encode('utf-8')
    
    # Render the template download button
    st.download_button(
        label="📥 Download Sample Template CSV",
        data=template_csv_bytes,
        file_name="hr_attrition_upload_template.csv",
        mime="text/csv",
        help="Click here to download a pre-formatted CSV template file to populate with your employee details."
    )
    
    st.markdown("---")
    
    # 2. FILE UPLOADER FOR TESTING
    uploaded_file = st.file_uploader("Upload your completed employee data sheet here", type="csv")
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        
        try:
            # Reorder columns to ensure an exact pipeline match
            processed_batch = batch_df[X_reference.columns]
            
            # Generate predictions using the stacking pipeline
            batch_probs = model.predict_proba(processed_batch)[:, 1]
            batch_df['Flight_Risk_Probability'] = batch_probs
            batch_df['Risk_Flag'] = np.where(batch_probs >= 0.35, "⚠️ High Risk", "✅ Stable")
            
            st.markdown("### 📋 Prediction Previews")
            display_cols = ['Department', 'JobRole', 'Flight_Risk_Probability', 'Risk_Flag']
            if 'EmployeeNumber' in batch_df.columns:
                display_cols.insert(0, 'EmployeeNumber')
                
            st.dataframe(batch_df[display_cols], use_container_width=True)
            
            # Offer download for the complete dataset including model predictions
            csv_output = batch_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📊 Download Complete Risk Analysis Reports",
                data=csv_output,
                file_name="hr_attrition_predictions_output.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Error processing batch columns: {str(e)}")
            st.info("💡 Hint: Please ensure your uploaded file has the exact columns provided in our download template above.")

            
