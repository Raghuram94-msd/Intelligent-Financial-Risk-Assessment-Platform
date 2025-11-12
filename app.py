#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="EMI Amount Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #f0f8ff;
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .feature-importance {
        background-color: #fffaf0;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .stProgress .st-bo {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

class EMIPredictorApp:
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.scaler = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model and feature information"""
        try:
            # Load the best model (replace with your actual model file)
            self.model = joblib.load('best_emi_amount_model_xgboost_regressor.pkl')
            
            # Feature names used in the model (replace with your actual feature names)
            self.feature_names = [
                'monthly_salary', 'credit_score', 'disposable_income', 
                'financial_stability_score', 'debt_to_income_ratio',
                'bank_balance', 'emergency_fund', 'years_of_employment',
                'employment_quality_score', 'savings_ratio',
                'expense_to_income_ratio', 'loan_purpose_risk_score',
                'composite_risk_score', 'requested_amount',
                'current_emi_amount', 'family_size', 'dependents',
                'age', 'monthly_rent', 'school_fees', 'college_fees',
                'travel_expenses', 'groceries_utilities', 'other_monthly_expenses'
            ]
            
            # Initialize scaler
            self.scaler = StandardScaler()
            
            st.sidebar.success("✅ Model loaded successfully!")
            
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            st.info("Please ensure the model file is in the correct directory.")
    
    def user_input_features(self):
        """Create user input form for feature values"""
        st.sidebar.markdown("## 📊 Customer Information")
        
        # Personal Information
        st.sidebar.markdown("### Personal Details")
        age = st.sidebar.slider("Age", 18, 70, 35)
        family_size = st.sidebar.slider("Family Size", 1, 10, 3)
        dependents = st.sidebar.slider("Number of Dependents", 0, 8, 1)
        
        # Employment Information
        st.sidebar.markdown("### Employment Details")
        employment_type = st.sidebar.selectbox(
            "Employment Type",
            ["Private", "Government", "Self-employed", "Other"]
        )
        
        company_type = st.sidebar.selectbox(
            "Company Type",
            ["MNC", "Large Indian", "Mid-size", "Startup", "Small"]
        )
        
        years_of_employment = st.sidebar.slider(
            "Years of Employment", 0.0, 40.0, 5.0, 0.5
        )
        
        # Financial Information
        st.sidebar.markdown("### Financial Details")
        monthly_salary = st.sidebar.number_input(
            "Monthly Salary (₹)", 
            min_value=0, 
            max_value=1000000, 
            value=50000, 
            step=1000
        )
        
        credit_score = st.sidebar.slider(
            "Credit Score", 300, 900, 700
        )
        
        bank_balance = st.sidebar.number_input(
            "Bank Balance (₹)", 
            min_value=0, 
            max_value=5000000, 
            value=100000, 
            step=10000
        )
        
        emergency_fund = st.sidebar.number_input(
            "Emergency Fund (₹)", 
            min_value=0, 
            max_value=2000000, 
            value=50000, 
            step=10000
        )
        
        # Expenses
        st.sidebar.markdown("### Monthly Expenses")
        monthly_rent = st.sidebar.number_input(
            "Monthly Rent (₹)", 
            min_value=0, 
            max_value=100000, 
            value=15000, 
            step=1000
        )
        
        school_fees = st.sidebar.number_input(
            "School Fees (₹)", 
            min_value=0, 
            max_value=50000, 
            value=0, 
            step=1000
        )
        
        college_fees = st.sidebar.number_input(
            "College Fees (₹)", 
            min_value=0, 
            max_value=100000, 
            value=0, 
            step=1000
        )
        
        travel_expenses = st.sidebar.number_input(
            "Travel Expenses (₹)", 
            min_value=0, 
            max_value=50000, 
            value=5000, 
            step=500
        )
        
        groceries_utilities = st.sidebar.number_input(
            "Groceries & Utilities (₹)", 
            min_value=0, 
            max_value=100000, 
            value=10000, 
            step=1000
        )
        
        other_monthly_expenses = st.sidebar.number_input(
            "Other Monthly Expenses (₹)", 
            min_value=0, 
            max_value=100000, 
            value=5000, 
            step=1000
        )
        
        # Loan Information
        st.sidebar.markdown("### Loan Details")
        current_emi_amount = st.sidebar.number_input(
            "Current EMI Amount (₹)", 
            min_value=0, 
            max_value=100000, 
            value=0, 
            step=1000
        )
        
        requested_amount = st.sidebar.number_input(
            "Requested Loan Amount (₹)", 
            min_value=0, 
            max_value=5000000, 
            value=500000, 
            step=10000
        )
        
        loan_purpose = st.sidebar.selectbox(
            "Loan Purpose",
            ["Education EMI", "Home Appliances EMI", "Vehicle EMI", 
             "Personal Loan EMI", "E-commerce Shopping EMI"]
        )
        
        # Calculate derived features
        disposable_income = monthly_salary - (
            monthly_rent + school_fees + college_fees + 
            travel_expenses + groceries_utilities + other_monthly_expenses + 
            current_emi_amount
        )
        
        debt_to_income_ratio = current_emi_amount / (monthly_salary + 1)
        savings_ratio = bank_balance / (monthly_salary + 1)
        expense_to_income_ratio = (
            monthly_rent + school_fees + college_fees + 
            travel_expenses + groceries_utilities + other_monthly_expenses
        ) / (monthly_salary + 1)
        
        # Employment quality score
        employment_score_map = {
            "Government": 10, "Private": 8, "Self-employed": 6, "Other": 5
        }
        company_size_map = {
            "MNC": 10, "Large Indian": 9, "Mid-size": 7, "Startup": 5, "Small": 4
        }
        employment_quality_score = (
            employment_score_map.get(employment_type, 5) +
            company_size_map.get(company_type, 5) +
            min(years_of_employment * 2, 20)
        )
        
        # Loan purpose risk score
        loan_purpose_risk_map = {
            "Education EMI": 3, "Home Appliances EMI": 5, 
            "Vehicle EMI": 6, "Personal Loan EMI": 8, 
            "E-commerce Shopping EMI": 9
        }
        loan_purpose_risk_score = loan_purpose_risk_map.get(loan_purpose, 5)
        
        # Financial stability score
        financial_stability_score = (
            (credit_score / 10) +
            min(savings_ratio * 20, 30) +
            min((1 - debt_to_income_ratio) * 30, 30) +
            min(years_of_employment * 2, 20)
        )
        
        # Composite risk score
        composite_risk_score = (
            (debt_to_income_ratio * 25) +
            ((1 - (credit_score / 1000)) * 30) +
            (loan_purpose_risk_score * 15) +
            (expense_to_income_ratio * 20) +
            ((1 - (disposable_income / monthly_salary)) * 10)
        )
        
        # Create feature dictionary
        features = {
            'monthly_salary': monthly_salary,
            'credit_score': credit_score,
            'disposable_income': max(disposable_income, 0),
            'financial_stability_score': financial_stability_score,
            'debt_to_income_ratio': debt_to_income_ratio,
            'bank_balance': bank_balance,
            'emergency_fund': emergency_fund,
            'years_of_employment': years_of_employment,
            'employment_quality_score': employment_quality_score,
            'savings_ratio': savings_ratio,
            'expense_to_income_ratio': expense_to_income_ratio,
            'loan_purpose_risk_score': loan_purpose_risk_score,
            'composite_risk_score': composite_risk_score,
            'requested_amount': requested_amount,
            'current_emi_amount': current_emi_amount,
            'family_size': family_size,
            'dependents': dependents,
            'age': age,
            'monthly_rent': monthly_rent,
            'school_fees': school_fees,
            'college_fees': college_fees,
            'travel_expenses': travel_expenses,
            'groceries_utilities': groceries_utilities,
            'other_monthly_expenses': other_monthly_expenses
        }
        
        return features
    
    def predict_emi(self, features):
        """Make prediction using the trained model"""
        try:
            # Convert features to dataframe
            feature_df = pd.DataFrame([features])
            
            # Ensure all required features are present
            for feature in self.feature_names:
                if feature not in feature_df.columns:
                    feature_df[feature] = 0
            
            # Reorder columns to match training
            feature_df = feature_df[self.feature_names]
            
            # Make prediction
            prediction = self.model.predict(feature_df)[0]
            
            return max(prediction, 0)  # Ensure non-negative prediction
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None
    
    def display_feature_importance(self):
        """Display feature importance visualization"""
        try:
            if hasattr(self.model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False).head(10)
                
                st.markdown("### 🔍 Top 10 Feature Importance")
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(data=importance_df, y='feature', x='importance', ax=ax)
                ax.set_title('Feature Importance for EMI Prediction')
                ax.set_xlabel('Importance Score')
                plt.tight_layout()
                st.pyplot(fig)
                
        except Exception as e:
            st.warning("Feature importance visualization not available for this model.")
    
    def display_financial_health(self, features):
        """Display financial health assessment"""
        st.markdown("### 💰 Financial Health Assessment")
        
        # Create columns for metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            debt_ratio = features['debt_to_income_ratio']
            if debt_ratio < 0.3:
                status = "✅ Excellent"
                color = "green"
            elif debt_ratio < 0.5:
                status = "⚠️ Moderate"
                color = "orange"
            else:
                status = "❌ High Risk"
                color = "red"
            st.metric("Debt-to-Income Ratio", f"{debt_ratio:.2f}", status)
        
        with col2:
            expense_ratio = features['expense_to_income_ratio']
            if expense_ratio < 0.6:
                status = "✅ Good"
                color = "green"
            elif expense_ratio < 0.8:
                status = "⚠️ Moderate"
                color = "orange"
            else:
                status = "❌ High"
                color = "red"
            st.metric("Expense-to-Income Ratio", f"{expense_ratio:.2f}", status)
        
        with col3:
            credit_score = features['credit_score']
            if credit_score >= 750:
                status = "✅ Excellent"
            elif credit_score >= 700:
                status = "✅ Good"
            elif credit_score >= 650:
                status = "⚠️ Fair"
            else:
                status = "❌ Poor"
            st.metric("Credit Score", credit_score, status)
        
        with col4:
            stability_score = features['financial_stability_score']
            if stability_score >= 150:
                status = "✅ Excellent"
            elif stability_score >= 120:
                status = "✅ Good"
            elif stability_score >= 90:
                status = "⚠️ Fair"
            else:
                status = "❌ Poor"
            st.metric("Financial Stability", f"{stability_score:.0f}", status)
    
    def run(self):
        """Main application runner"""
        # Header
        st.markdown('<h1 class="main-header">💰 EMI Amount Predictor</h1>', 
                   unsafe_allow_html=True)
        st.markdown("### Predict Maximum Affordable EMI Based on Customer Profile")
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Get user inputs
            features = self.user_input_features()
            
            # Prediction button
            if st.button("🚀 Predict Maximum EMI", use_container_width=True):
                with st.spinner("Calculating maximum affordable EMI..."):
                    prediction = self.predict_emi(features)
                    
                    if prediction is not None:
                        # Display prediction
                        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                        st.markdown("### 📈 EMI Prediction Result")
                        
                        # Animated progress bar for prediction confidence
                        progress_bar = st.progress(0)
                        for i in range(100):
                            progress_bar.progress(i + 1)
                        
                        # Display prediction with styling
                        col_pred1, col_pred2, col_pred3 = st.columns([1, 2, 1])
                        with col_pred2:
                            st.markdown(f"""
                            <div style="text-align: center;">
                                <h2 style="color: #1f77b4; margin-bottom: 0;">₹ {prediction:,.0f}</h2>
                                <p style="color: #666; font-size: 1.1rem;">Maximum Monthly EMI</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Additional insights
                        st.markdown("---")
                        self.display_financial_health(features)
                        
                        # Loan affordability analysis
                        st.markdown("#### 💡 Loan Affordability Insights")
                        monthly_salary = features['monthly_salary']
                        emi_affordability_ratio = prediction / monthly_salary
                        
                        if emi_affordability_ratio < 0.4:
                            insight = "✅ **Comfortable**: EMI is well within affordable limits"
                            color = "green"
                        elif emi_affordability_ratio < 0.6:
                            insight = "⚠️ **Moderate**: EMI is at reasonable levels"
                            color = "orange"
                        else:
                            insight = "❌ **Strained**: EMI may cause financial stress"
                            color = "red"
                        
                        st.info(insight)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
            
            # Feature importance section
            st.markdown("---")
            self.display_feature_importance()
        
        with col2:
            # Quick insights sidebar
            st.markdown("### 📋 Quick Insights")
            
            # Financial ratios
            if 'features' in locals():
                features = features
                
                st.markdown("#### Financial Ratios")
                
                # Debt ratio gauge
                debt_ratio = features['debt_to_income_ratio']
                st.metric("Current Debt Burden", f"{debt_ratio:.1%}")
                
                # Savings adequacy
                monthly_expenses = (
                    features['monthly_rent'] + features['school_fees'] + 
                    features['college_fees'] + features['travel_expenses'] + 
                    features['groceries_utilities'] + features['other_monthly_expenses']
                )
                emergency_fund_months = features['emergency_fund'] / monthly_expenses if monthly_expenses > 0 else 0
                st.metric("Emergency Fund (Months)", f"{emergency_fund_months:.1f}")
                
                # Disposable income
                disposable_income = features['disposable_income']
                st.metric("Monthly Disposable Income", f"₹ {disposable_income:,.0f}")
            
            # Tips section
            st.markdown("---")
            st.markdown("#### 💡 Tips to Improve EMI Eligibility")
            st.markdown("""
            - Maintain credit score above 750
            - Keep debt-to-income ratio below 40%
            - Build 6+ months of emergency fund
            - Demonstrate stable employment history
            - Reduce existing loan obligations
            - Choose lower-risk loan purposes
            """)
        
        # Footer
        st.markdown("---")
        st.markdown(
            """
            <div style="text-align: center; color: #666; font-size: 0.9rem;">
                <p>Built with ❤️ using Streamlit | EMI Prediction Model v1.0</p>
            </div>
            """, 
            unsafe_allow_html=True
        )

# Run the application
if __name__ == "__main__":
    app = EMIPredictorApp()
    app.run()