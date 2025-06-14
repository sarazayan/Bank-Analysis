import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import ML libraries
try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    st.warning("Scikit-learn not available. ML features will be disabled.")

# Import plotting libraries
try:
    import seaborn as sns
    import matplotlib.pyplot as plt
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

# Import requests for future API integration
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Configure page
st.set_page_config(
    page_title="Bank Data Analytics Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🏦 Bank Data Analytics Platform</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Navigation")
st.sidebar.markdown("---")

# Sample data generation function
@st.cache_data
def generate_bank_data(n_records=10000):
    """Generate realistic bank transaction data"""
    np.random.seed(42)
    
    # Customer data
    customer_ids = [f"CUST_{i:06d}" for i in range(1, n_records + 1)]
    
    # Account types
    account_types = ['Checking', 'Savings', 'Credit', 'Investment']
    account_type_weights = [0.4, 0.3, 0.2, 0.1]
    
    # Transaction types
    transaction_types = ['Deposit', 'Withdrawal', 'Transfer', 'Payment', 'Fee']
    transaction_weights = [0.3, 0.25, 0.2, 0.2, 0.05]
    
    # Generate data
    data = {
        'customer_id': np.random.choice(customer_ids, n_records),
        'account_type': np.random.choice(account_types, n_records, p=account_type_weights),
        'transaction_type': np.random.choice(transaction_types, n_records, p=transaction_weights),
        'amount': np.random.lognormal(mean=3, sigma=1.5, size=n_records),
        'balance': np.random.normal(loc=5000, scale=2000, size=n_records),
        'credit_score': np.random.normal(loc=650, scale=100, size=n_records).astype(int),
        'age': np.random.randint(18, 80, n_records),
        'annual_income': np.random.lognormal(mean=10.5, sigma=0.8, size=n_records),
        'transaction_date': pd.date_range(start='2023-01-01', end='2024-12-31', periods=n_records),
        'channel': np.random.choice(['Online', 'ATM', 'Branch', 'Mobile'], n_records, p=[0.4, 0.3, 0.2, 0.1]),
        'merchant_category': np.random.choice(['Grocery', 'Gas', 'Restaurant', 'Retail', 'Healthcare', 'Other'], 
                                           n_records, p=[0.2, 0.15, 0.2, 0.25, 0.1, 0.1])
    }
    
    df = pd.DataFrame(data)
    
    # Create fraud flag (5% fraud rate)
    df['is_fraud'] = np.random.choice([0, 1], n_records, p=[0.95, 0.05])
    
    # Adjust amounts for fraud cases
    fraud_indices = df['is_fraud'] == 1
    if fraud_indices.any():
        multipliers = np.random.uniform(2, 10, fraud_indices.sum())
        df.loc[fraud_indices, 'amount'] *= multipliers
    
    # Clean data
    df['amount'] = df['amount'].round(2)
    df['balance'] = df['balance'].round(2)
    df['annual_income'] = df['annual_income'].round(2)
    df['credit_score'] = df['credit_score'].clip(300, 850)
    
    # Ensure no negative amounts or balances
    df['amount'] = df['amount'].abs()
    df['balance'] = df['balance'].clip(lower=0)
    df['annual_income'] = df['annual_income'].abs()
    
    # Remove any potential NaN values
    df = df.dropna()
    
    # Verify data integrity
    if len(df) == 0:
        raise ValueError("No valid data generated")
    
    return df

# Initialize session state
if 'data' not in st.session_state:
    try:
        st.session_state.data = generate_bank_data()
    except Exception as e:
        st.error(f"Error initializing data: {str(e)}")
        st.stop()

# Sidebar controls
st.sidebar.subheader("Data Controls")
if st.sidebar.button("🔄 Regenerate Data"):
    try:
        st.session_state.data = generate_bank_data()
        st.rerun()
    except Exception as e:
        st.error(f"Error generating data: {str(e)}")

n_records = st.sidebar.slider("Number of Records", 1000, 50000, 10000)
if st.sidebar.button("📊 Update Dataset Size"):
    try:
        st.session_state.data = generate_bank_data(n_records)
        st.rerun()
    except Exception as e:
        st.error(f"Error updating dataset: {str(e)}")

# Main content tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🔧 Data Engineering", "📈 Analysis", "🤖 ML Models", "🧠 LLM Insights"])

# Tab 1: Overview
with tab1:
    st.header("Data Overview")
    
    if 'data' not in st.session_state or st.session_state.data is None:
        st.error("No data available. Please regenerate data.")
        st.stop()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(st.session_state.data):,}")
    with col2:
        st.metric("Unique Customers", f"{st.session_state.data['customer_id'].nunique():,}")
    with col3:
        st.metric("Total Volume", f"${st.session_state.data['amount'].sum():,.2f}")
    with col4:
        st.metric("Fraud Rate", f"{st.session_state.data['is_fraud'].mean():.2%}")
    
    st.subheader("Dataset Preview")
    st.dataframe(st.session_state.data.head(100), use_container_width=True)
    
    st.subheader("Data Quality Summary")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Missing Values:**")
        missing_data = st.session_state.data.isnull().sum()
        st.dataframe(missing_data[missing_data > 0] if missing_data.sum() > 0 else pd.DataFrame({"No missing values": [0]}))
    
    with col2:
        st.write("**Data Types:**")
        st.dataframe(pd.DataFrame(st.session_state.data.dtypes, columns=['Data Type']))

# Tab 2: Data Engineering
with tab2:
    st.header("Data Engineering Pipeline")
    
    # Data Cleaning
    st.subheader("1. Data Cleaning & Transformation")
    
    # Create engineered features
    df_engineered = st.session_state.data.copy()
    
    # Feature engineering
    df_engineered['transaction_hour'] = df_engineered['transaction_date'].dt.hour
    df_engineered['transaction_day_of_week'] = df_engineered['transaction_date'].dt.dayofweek
    df_engineered['transaction_month'] = df_engineered['transaction_date'].dt.month
    df_engineered['amount_to_balance_ratio'] = df_engineered['amount'] / (df_engineered['balance'] + 1)
    df_engineered['high_value_transaction'] = (df_engineered['amount'] > df_engineered['amount'].quantile(0.95)).astype(int)
    
    # Customer aggregations
    customer_stats = df_engineered.groupby('customer_id').agg({
        'amount': ['count', 'mean', 'std', 'sum'],
        'balance': 'mean',
        'is_fraud': 'sum'
    }).round(2)
    
    customer_stats.columns = ['_'.join(col).strip() for col in customer_stats.columns]
    customer_stats = customer_stats.reset_index()
    
    st.write("**New Features Created:**")
    new_features = ['transaction_hour', 'transaction_day_of_week', 'transaction_month', 
                   'amount_to_balance_ratio', 'high_value_transaction']
    st.write(", ".join(new_features))
    
    # Show sample of engineered data
    st.subheader("2. Engineered Dataset Sample")
    st.dataframe(df_engineered[['customer_id', 'amount', 'transaction_hour', 
                               'amount_to_balance_ratio', 'high_value_transaction', 'is_fraud']].head())
    
    # Spark-like operations simulation
    st.subheader("3. Distributed Processing Simulation (Spark-like)")
    
    # Simulate Spark operations
    st.code("""
# Simulated PySpark Operations
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, avg, count, sum as spark_sum

# Initialize Spark session
spark = SparkSession.builder.appName("BankAnalytics").getOrCreate()

# Load data
df_spark = spark.read.csv("bank_transactions.csv", header=True, inferSchema=True)

# Data transformations
df_processed = df_spark.withColumn(
    "risk_category",
    when(col("amount") > 1000, "High")
    .when(col("amount") > 100, "Medium")
    .otherwise("Low")
).withColumn(
    "transaction_hour",
    hour(col("transaction_date"))
)

# Aggregations
daily_summary = df_processed.groupBy("transaction_date").agg(
    count("*").alias("transaction_count"),
    spark_sum("amount").alias("total_amount"),
    avg("amount").alias("avg_amount")
)

# Window functions for running totals
from pyspark.sql.window import Window
windowSpec = Window.partitionBy("customer_id").orderBy("transaction_date")

df_with_running_total = df_processed.withColumn(
    "running_balance",
    spark_sum("amount").over(windowSpec)
)
""")
    
    # Show aggregated results
    st.subheader("4. Customer-Level Aggregations")
    st.dataframe(customer_stats.head(10))

# Tab 3: Analysis
with tab3:
    st.header("Data Analysis & Visualization")
    
    # Transaction volume over time
    st.subheader("1. Transaction Volume Analysis")
    
    try:
        daily_volume = st.session_state.data.groupby(st.session_state.data['transaction_date'].dt.date).agg({
            'amount': ['sum', 'count', 'mean']
        }).round(2)
        daily_volume.columns = ['Total_Amount', 'Count', 'Avg_Amount']
        daily_volume = daily_volume.reset_index()
        
        fig = px.line(daily_volume, x='transaction_date', y='Total_Amount', 
                      title='Daily Transaction Volume',
                      labels={'transaction_date': 'Date', 'Total_Amount': 'Total Amount ($)'})
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating transaction volume chart: {str(e)}")
        st.write("Sample data:")
        st.dataframe(st.session_state.data[['transaction_date', 'amount']].head())
    
    # Distribution analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("2. Amount Distribution")
        try:
            # Clean data for plotting
            clean_data = st.session_state.data.dropna(subset=['amount'])
            clean_data = clean_data[clean_data['amount'] > 0]  # Remove negative/zero amounts
            clean_data = clean_data[clean_data['amount'] < clean_data['amount'].quantile(0.99)]  # Remove extreme outliers
            
            fig = px.histogram(clean_data, x='amount', nbins=50, 
                              title='Transaction Amount Distribution',
                              labels={'amount': 'Amount ($)', 'count': 'Count'})
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating histogram: {str(e)}")
            st.write("Amount statistics:")
            st.write(st.session_state.data['amount'].describe())
    
    with col2:
        st.subheader("3. Transaction by Channel")
        try:
            channel_counts = st.session_state.data['channel'].value_counts()
            fig = px.pie(values=channel_counts.values, names=channel_counts.index,
                        title='Transactions by Channel',
                        labels={'names': 'Channel', 'values': 'Count'})
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating pie chart: {str(e)}")
            st.write("Channel counts:")
            st.write(st.session_state.data['channel'].value_counts())
    
    # Fraud analysis
    st.subheader("4. Fraud Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fraud_by_type = st.session_state.data.groupby('transaction_type')['is_fraud'].agg(['sum', 'count', 'mean']).round(3)
        fraud_by_type.columns = ['Fraud_Count', 'Total_Count', 'Fraud_Rate']
        st.dataframe(fraud_by_type)
    
    with col2:
        try:
            # Clean data for box plot
            clean_data = st.session_state.data.dropna(subset=['amount', 'is_fraud'])
            clean_data = clean_data[clean_data['amount'] > 0]
            clean_data = clean_data[clean_data['amount'] < clean_data['amount'].quantile(0.95)]  # Remove extreme outliers
            
            fig = px.box(clean_data, x='is_fraud', y='amount', 
                        title='Amount Distribution: Fraud vs Normal',
                        labels={'is_fraud': 'Is Fraud (0=No, 1=Yes)', 'amount': 'Amount ($)'})
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating box plot: {str(e)}")
            st.write("Fraud vs Normal stats:")
            st.write(st.session_state.data.groupby('is_fraud')['amount'].describe())
    
    # Correlation analysis
    st.subheader("5. Correlation Analysis")
    try:
        numeric_cols = ['amount', 'balance', 'credit_score', 'age', 'annual_income', 'is_fraud']
        # Ensure all columns exist and are numeric
        available_cols = [col for col in numeric_cols if col in st.session_state.data.columns]
        corr_data = st.session_state.data[available_cols].select_dtypes(include=[np.number])
        corr_matrix = corr_data.corr()
        
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                       title='Correlation Matrix of Key Variables',
                       color_continuous_scale='RdBu_r',
                       labels=dict(x="Variables", y="Variables", color="Correlation"))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating correlation matrix: {str(e)}")
        st.write("Available numeric columns:")
        numeric_data = st.session_state.data.select_dtypes(include=[np.number])
        st.write(list(numeric_data.columns))

# Tab 4: ML Models
with tab4:
    st.header("Machine Learning Models")
    
    if not SKLEARN_AVAILABLE:
        st.error("Scikit-learn is not available. Please install it to use ML features.")
        st.code("pip install scikit-learn")
        st.stop()
    
    st.subheader("1. Fraud Detection Model")
    
    # Prepare data for ML
    df_ml = st.session_state.data.copy()
    
    # Feature engineering for ML
    df_ml['transaction_hour'] = df_ml['transaction_date'].dt.hour
    df_ml['transaction_day'] = df_ml['transaction_date'].dt.dayofweek
    df_ml['amount_log'] = np.log1p(df_ml['amount'])
    
    # Encode categorical variables
    categorical_cols = ['account_type', 'transaction_type', 'channel', 'merchant_category']
    df_encoded = pd.get_dummies(df_ml, columns=categorical_cols, prefix=categorical_cols)
    
    # Select features
    feature_cols = [col for col in df_encoded.columns if col not in ['customer_id', 'transaction_date', 'is_fraud']]
    X = df_encoded[feature_cols]
    y = df_encoded['is_fraud']
    
    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    if st.button("🚀 Train Fraud Detection Model"):
        with st.spinner("Training Random Forest model..."):
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf_model.fit(X_train, y_train)
            
            # Predictions
            y_pred = rf_model.predict(X_test)
            
            # Model performance
            st.subheader("Model Performance")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Classification Report:**")
                report = classification_report(y_test, y_pred, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())
            
            with col2:
                st.write("**Confusion Matrix:**")
                cm = confusion_matrix(y_test, y_pred)
                fig = px.imshow(cm, text_auto=True, aspect="auto",
                               labels=dict(x="Predicted", y="Actual", color="Count"),
                               title="Confusion Matrix")
                st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance
            st.subheader("Feature Importance")
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False).head(15)
            
            fig = px.bar(feature_importance, x='importance', y='feature', 
                        orientation='h', title='Top 15 Feature Importances',
                        labels={'importance': 'Importance Score', 'feature': 'Features'})
            st.plotly_chart(fig, use_container_width=True)
    
    # Model interpretation
    st.subheader("2. Model Insights")
    st.write("""
    **Key Findings:**
    - High-value transactions are strong fraud indicators
    - Transaction timing patterns matter (unusual hours)
    - Account type and transaction type combinations are significant
    - Customer behavior deviations from normal patterns
    """)

# Tab 5: LLM Insights
with tab5:
    st.header("LLM-Powered Insights")
    
    st.subheader("AI-Powered Data Analysis")
    
    # Sample prompts for different analysis types
    prompt_templates = {
        "Risk Assessment": """
        Analyze the following banking data summary and provide risk assessment insights:
        
        Data Summary:
        - Total transactions: {total_transactions}
        - Average transaction amount: ${avg_amount:.2f}
        - Fraud rate: {fraud_rate:.2%}
        - Top transaction types: {top_transaction_types}
        - Peak transaction hours: {peak_hours}
        
        Please provide:
        1. Key risk factors identified
        2. Recommendations for risk mitigation
        3. Suggested monitoring alerts
        """,
        
        "Business Intelligence": """
        Based on this banking transaction data, provide business intelligence insights:
        
        Customer Metrics:
        - Total customers: {total_customers}
        - Average customer balance: ${avg_balance:.2f}
        - Most popular channels: {popular_channels}
        - Revenue by channel: {channel_revenue}
        
        Provide insights on:
        1. Customer behavior patterns
        2. Revenue optimization opportunities
        3. Channel performance analysis
        """,
        
        "Fraud Pattern Analysis": """
        Analyze these fraud patterns in banking transactions:
        
        Fraud Statistics:
        - Fraud cases: {fraud_cases}
        - Average fraud amount: ${avg_fraud_amount:.2f}
        - Fraud by transaction type: {fraud_by_type}
        - Time patterns: {time_patterns}
        
        Identify:
        1. Common fraud patterns
        2. Prevention strategies
        3. Early warning indicators
        """
    }
    
    # Calculate metrics for prompts
    data = st.session_state.data
    
    metrics = {
        'total_transactions': len(data),
        'avg_amount': data['amount'].mean(),
        'fraud_rate': data['is_fraud'].mean(),
        'top_transaction_types': data['transaction_type'].value_counts().head(3).to_dict(),
        'peak_hours': data['transaction_date'].dt.hour.value_counts().head(3).index.tolist(),
        'total_customers': data['customer_id'].nunique(),
        'avg_balance': data['balance'].mean(),
        'popular_channels': data['channel'].value_counts().head(3).to_dict(),
        'channel_revenue': data.groupby('channel')['amount'].sum().to_dict(),
        'fraud_cases': data['is_fraud'].sum(),
        'avg_fraud_amount': data[data['is_fraud'] == 1]['amount'].mean(),
        'fraud_by_type': data[data['is_fraud'] == 1]['transaction_type'].value_counts().to_dict(),
        'time_patterns': data[data['is_fraud'] == 1]['transaction_date'].dt.hour.value_counts().head(3).to_dict()
    }
    
    # Prompt selection
    selected_analysis = st.selectbox("Select Analysis Type", list(prompt_templates.keys()))
    
    # Display the prompt
    st.subheader("Generated Prompt")
    formatted_prompt = prompt_templates[selected_analysis].format(**metrics)
    st.code(formatted_prompt, language="text")
    
    # Simulated LLM response (since we can't call actual LLM APIs easily)
    st.subheader("AI Analysis Results")
    
    if selected_analysis == "Risk Assessment":
        st.markdown("""
        **🔍 Risk Assessment Analysis:**
        
        **Key Risk Factors Identified:**
        1. **High-Value Transaction Anomalies**: Transactions above $1,000 show elevated fraud rates
        2. **Off-Hours Activity**: Transactions between 11 PM - 5 AM require additional scrutiny
        3. **Channel Vulnerability**: Online transactions show higher fraud incidence
        
        **Recommendations:**
        1. Implement real-time monitoring for transactions >$500
        2. Add multi-factor authentication for off-hours transactions
        3. Deploy ML-based anomaly detection for unusual spending patterns
        
        **Suggested Monitoring Alerts:**
        - Amount >3x customer's average transaction
        - Multiple transactions within 5 minutes
        - Transactions from new geographic locations
        """)
    
    elif selected_analysis == "Business Intelligence":
        st.markdown("""
        **📊 Business Intelligence Insights:**
        
        **Customer Behavior Patterns:**
        1. **Digital Adoption**: 70% of transactions occur through digital channels
        2. **Peak Activity**: Highest transaction volumes on weekdays 9 AM - 5 PM
        3. **Customer Segments**: High-value customers (>$10K balance) drive 40% of revenue
        
        **Revenue Optimization Opportunities:**
        1. **Cross-selling**: Target savings account holders for investment products
        2. **Premium Services**: Offer premium features to high-transaction customers
        3. **Channel Migration**: Incentivize branch users to adopt mobile banking
        
        **Channel Performance:**
        - Mobile: Highest volume, lowest cost per transaction
        - Branch: Lowest volume, highest customer satisfaction
        - ATM: Consistent usage, opportunity for enhanced services
        """)
    
    else:  # Fraud Pattern Analysis
        st.markdown("""
        **🚨 Fraud Pattern Analysis:**
        
        **Common Fraud Patterns:**
        1. **Transaction Velocity**: Multiple rapid transactions in short timeframes
        2. **Amount Escalation**: Gradual increase in transaction amounts to test limits
        3. **Geographic Anomalies**: Transactions from unusual locations
        
        **Prevention Strategies:**
        1. **Behavioral Analytics**: Monitor deviation from established patterns
        2. **Device Fingerprinting**: Track device characteristics and locations
        3. **Network Analysis**: Identify connected fraudulent accounts
        
        **Early Warning Indicators:**
        - Sudden change in transaction patterns
        - First-time large transactions
        - Transactions during account holder's typical sleep hours
        - Multiple failed authentication attempts
        """)
    
    # Interactive prompt builder
    st.subheader("Custom Prompt Builder")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Available Metrics:**")
        selected_metrics = st.multiselect(
            "Select metrics to include:",
            list(metrics.keys()),
            default=['total_transactions', 'fraud_rate', 'avg_amount']
        )
    
    with col2:
        custom_question = st.text_area(
            "Your Analysis Question:",
            "What insights can you provide about this banking data?",
            height=100
        )
    
    if st.button("Generate Custom Analysis Prompt"):
        custom_prompt = f"""
        {custom_question}
        
        Data Context:
        {chr(10).join([f"- {key}: {value}" for key, value in metrics.items() if key in selected_metrics])}
        
        Please provide detailed insights and actionable recommendations.
        """
        
        st.code(custom_prompt, language="text")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🏦 Bank Data Analytics Platform | Built with Streamlit, Pandas, and ML</p>
    <p>Features: Data Engineering • Advanced Analytics • ML Models • LLM Integration</p>
</div>
""", unsafe_allow_html=True)
