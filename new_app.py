#import neccesary libraries
import pandas as pd
import streamlit as st
import joblib
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


#loading my model
model = joblib.load("new_model.joblib")

def preprocess_data( age, income, loan_amount, credit_score, months_employed, num_credit_lines, interest_rate,loan_term, dti_ratio, 
                    education, employment_type, marital_status, has_mortgage, has_dependents, loan_purpose, has_cosigner):
    #create a dataframe with user imput
    data = pd.DataFrame({
        'Age': [age], 
        'Income': [income], 
        'LoanAmount': [loan_amount], 
        'CreditScore': [credit_score],
        'MonthsEmployed': [months_employed], 
        'NumCreditLines': [num_credit_lines], 
        'InterestRate': [interest_rate],
        'LoanTerm': [loan_term],
        'DTIRatio': [dti_ratio],
        'Education': [education],
        'EmploymentType': [employment_type],
        'MaritalStatus': [marital_status],
        'HasMortgage': [has_mortgage],
        'HasDependents': [has_dependents],
        'LoanPurpose': [loan_purpose ],
        'HasCoSigner':[has_cosigner]
    })
    
    
    #encode my categorical columns
    le= LabelEncoder()
    data['Education']= le.fit_transform(data['Education'])
    data['EmploymentType']= le.fit_transform(data['EmploymentType'])
    data['MaritalStatus']= le.fit_transform(data['MaritalStatus'])
    data['HasMortgage']= le.fit_transform(data['HasMortgage'])
    data['HasDependents']= le.fit_transform(data['HasDependents'])
    data['LoanPurpose']= le.fit_transform(data['LoanPurpose'])
    data['HasCoSigner']= le.fit_transform(data['HasCoSigner'])
    
    #scale numerical data
    scaler = MinMaxScaler()
    numerical_cols = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed', 'NumCreditLines', 'InterestRate','LoanTerm', 'DTIRatio']
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    return data


def main():
    st.title('Loan Default Prediction App')  
    st.subheader('Welcome, please enter the required information to get the prediction.')
    st.image('Loan 1.jpg', use_column_width=True )
    st.subheader("Please enter the required information of customer")
    st.divider()
    
    #get user input
    # Numeric inputs
    age = st.slider('Age', min_value=18, max_value=100, value=35)
    income = st.slider('Income', value = 100000)
    loan_amount = st.slider('Loan Amount', value= 100000)
    credit_score = st.slider('Credit Score', min_value=300, max_value=850, value=500)
    months_employed = st.slider('Months Employed', value=12)
    num_credit_lines = st.slider('Number of Credit Lines', value=2)
    interest_rate = st.slider('Interest Rate', min_value=0, max_value=30, value=15)
    loan_term = st.slider('Loan Term', min_value=12, max_value=60, value=36)
    dti_ratio = st.slider('DTI Ratio', min_value=0.0, max_value=1.0, value=0.5)
    # Categorical inputs
    education = st.selectbox('Education', ['High School', "Bachelor's", "Master's", 'PhD'])
    employment_type = st.selectbox('Employment Type', ['Part-time', 'Full-time', 'Self-employed', 'Unemployed'])
    marital_status = st.selectbox('Marital Status', ['Single', 'Married', 'Divorced'])
    has_mortgage = st.radio('Has Mortgage', ['Yes', 'No'])
    has_dependents = st.radio('Has Dependents', ['Yes', 'No'])
    loan_purpose = st.selectbox('Loan Purpose', ['Personal', 'Education', 'Home', 'Car', 'Other'])
    has_cosigner = st.radio('Has CoSigner', ['Yes', 'No'])
    
    #preprocess user inputs
    user_data = preprocess_data( age, income, loan_amount, credit_score, months_employed, num_credit_lines, interest_rate,loan_term, dti_ratio, 
                    education, employment_type, marital_status, has_mortgage, has_dependents, loan_purpose, has_cosigner)

    # make prediction with loaded model
    prediction = model.predict(user_data)
   
    
  # display the prediction
    st.subheader('Prediction')
    st.write('1: Loan will default, 0: Loan will not default')
    result = 'Loan will default' if prediction[0] == 1 else 'Loan will not default'
    st.write(prediction[0], result)
    
    
    
    
if __name__ == '__main__':
    main()   

