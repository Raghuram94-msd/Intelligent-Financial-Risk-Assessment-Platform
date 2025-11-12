#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def create_model_files():
    """
    Create the necessary model files for the Streamlit app
    This should be run after training your final model
    """
    
    # Define the exact features expected by the model (24 features)
    expected_features = [
        'age',
        'monthly_salary', 
        'years_of_employment',
        'monthly_rent',
        'family_size',
        'dependents',
        'school_fees',
        'college_fees',
        'travel_expenses',
        'groceries_utilities',
        'other_monthly_expenses',
        'current_emi_amount',
        'credit_score',
        'bank_balance',
        'emergency_fund',
        'requested_amount',
        'requested_tenure',
        'marital_status_Married',
        'marital_status_Single',
        'education_Graduate', 
        'education_Professional',
        'employment_type_Government',
        'employment_type_Private',
        'existing_loans_Yes'
    ]
    
    # Create a simple model for demonstration
    # In practice, you would load your actual trained model here
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Create dummy training data with the exact feature set
    # This is just for demonstration - use your actual training data
    X_dummy = np.random.randn(100, len(expected_features))
    y_dummy = np.random.randn(100) * 10000 + 15000  # EMI amounts between 5k-25k
    
    # Train the model
    model.fit(X_dummy, y_dummy)
    
    # Create model configuration
    model_config = {
        'feature_names': expected_features,
        'expected_features': expected_features,
        'categorical_mappings': {
            'marital_status': ['Single', 'Married', 'Divorced', 'Widowed'],
            'education': ['High School', 'Graduate', 'Post Graduate', 'Professional'],
            'employment_type': ['Private', 'Government', 'Self-employed'],
            'company_type': ['MNC', 'Large Indian', 'Mid-size', 'Startup', 'Small'],
            'emi_scenario': ['Personal Loan EMI', 'Vehicle EMI', 'Home Appliances EMI', 
                           'Education EMI', 'E-commerce Shopping EMI']
        },
        'model_type': 'RandomForestRegressor'
    }
    
    # Save files
    joblib.dump(model, 'best_emi_regression_model.pkl')
    
    with open('model_config.pkl', 'wb') as f:
        pickle.dump(model_config, f)
    
    print("✅ Model files created successfully!")
    print(f"📊 Expected features: {len(expected_features)}")
    print("📁 Files saved:")
    print("   - best_emi_regression_model.pkl")
    print("   - model_config.pkl")
    print("\n🔧 Expected features list:")
    for i, feature in enumerate(expected_features, 1):
        print(f"   {i:2d}. {feature}")

if __name__ == "__main__":
    create_model_files()

