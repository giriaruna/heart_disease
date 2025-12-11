import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from sklearn.feature_selection import chi2, SelectKBest
import plotly.graph_objects as go
import plotly.express as px
import requests
import io

# Page configuration
st.set_page_config(
    page_title="Heart Disease Classification",
    layout="wide"
)

# Title
st.title("Heart Disease Classification")
st.markdown("**Predicting Heart Disease Using Classical Machine Learning Models**")
st.markdown("Team: İlayda Dilek, Aruna Giri")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Data Overview", "Data Preprocessing", "Model Training", "Model Evaluation", "Ensemble Analysis", "Feature Selection"]
)

@st.cache_data
def load_data():
    """Load the UCI Heart Disease dataset"""
    # UCI Heart Disease dataset URL
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    
    # Column names based on UCI dataset documentation
    column_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        # Read data, handling missing values marked as '?'
        df = pd.read_csv(io.StringIO(response.text), names=column_names, na_values='?')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Trying alternative dataset source...")
        # Alternative: use a sample dataset structure
        try:
            url2 = "https://raw.githubusercontent.com/plotly/datasets/master/heart.csv"
            df = pd.read_csv(url2)
            if 'target' not in df.columns:
                st.warning("Using alternative dataset structure")
            return df
        except:
            st.error("Could not load dataset. Please check your internet connection.")
            return None

@st.cache_data
def preprocess_data_basic(df):
    """Basic preprocessing that doesn't require train/test split (no data leakage)"""
    df_clean = df.copy()

    if 'target' in df_clean.columns:
        df_clean['target'] = df_clean['target'].apply(lambda x: 1 if x > 0 else 0)
    
    return df_clean

def preprocess_train_test(X_train, X_test):
    """
    Preprocess training and test data properly to avoid data leakage.
    
    Args:
        X_train: Training features
        X_test: Test features
    
    Returns:
        X_train_clean: Training features with missing values filled
        X_test_clean: Test features with missing values filled using training statistics
        imputation_values: Dictionary of imputation values used
    """
    X_train_clean = X_train.copy()
    X_test_clean = X_test.copy()
    
    # Calculate imputation values ONLY from training data
    imputation_values = {}
    numeric_cols = X_train_clean.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if X_train_clean[col].isnull().sum() > 0:
            # Calculate median from training data only
            imputation_values[col] = X_train_clean[col].median()
            # Apply to both training and test sets
            X_train_clean[col].fillna(imputation_values[col], inplace=True)
            X_test_clean[col].fillna(imputation_values[col], inplace=True)
    
    return X_train_clean, X_test_clean, imputation_values

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    return fig

def plot_roc_curve(y_true, y_pred_proba, model_name):
    """Plot ROC curve"""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc_score = roc_auc_score(y_true, y_pred_proba)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'{model_name} (AUC = {auc_score:.3f})',
        line=dict(width=2)
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(dash='dash', color='gray')
    ))
    fig.update_layout(
        title=f'ROC Curve - {model_name}',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        width=700,
        height=500
    )
    return fig

# Load data
if 'df' not in st.session_state:
    with st.spinner("Loading dataset..."):
        st.session_state.df = load_data()

df = st.session_state.df

if df is not None:
    # Basic preprocessing
    if 'df_clean' not in st.session_state:
        st.session_state.df_clean = preprocess_data_basic(df)
    
    df_clean = st.session_state.df_clean
    
    # Main content based on selected page
    if page == "Data Overview":
        st.header("Data Overview")
        
        st.subheader("Dataset Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", len(df_clean))
        with col2:
            st.metric("Features", len(df_clean.columns) - 1)
        with col3:
            st.metric("Target Classes", df_clean['target'].nunique() if 'target' in df_clean.columns else "N/A")
        
        st.subheader("First Few Rows")
        st.dataframe(df_clean.head(10), use_container_width=True)
        
        st.subheader("Dataset Statistics")
        st.dataframe(df_clean.describe(), use_container_width=True)
        
        st.subheader("Missing Values")
        missing = df_clean.isnull().sum()
        if missing.sum() > 0:
            st.warning("⚠️ Missing values detected in raw data. These will be properly handled during model training using statistics calculated ONLY from the training set (to prevent data leakage).")
            st.dataframe(missing[missing > 0].to_frame('Missing Count'))
        else:
            st.success("No missing values found!")
        
        st.subheader("Target Distribution")
        if 'target' in df_clean.columns:
            target_counts = df_clean['target'].value_counts()
            fig = px.pie(
                values=target_counts.values,
                names=['No Disease', 'Disease'],
                title="Heart Disease Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Feature Visualizations")
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        if 'target' in numeric_cols:
            numeric_cols.remove('target')
        
        selected_feature = st.selectbox("Select a feature to visualize", numeric_cols)
        
        if selected_feature:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # Histogram
            axes[0].hist(df_clean[selected_feature], bins=30, edgecolor='black')
            axes[0].set_title(f'Distribution of {selected_feature}')
            axes[0].set_xlabel(selected_feature)
            axes[0].set_ylabel('Frequency')
            
            # Box plot by target
            if 'target' in df_clean.columns:
                df_clean.boxplot(column=selected_feature, by='target', ax=axes[1])
                axes[1].set_title(f'{selected_feature} by Heart Disease Status')
                axes[1].set_xlabel('Heart Disease')
                axes[1].set_ylabel(selected_feature)
            
            plt.tight_layout()
            st.pyplot(fig)
    
    elif page == "Data Preprocessing":
        st.header("🔧 Data Preprocessing")
        
        st.subheader("Preprocessed Dataset")
        st.dataframe(df_clean.head(10), use_container_width=True)
        
        st.subheader("Data Types")
        st.dataframe(df_clean.dtypes.to_frame('Data Type'), use_container_width=True)
        
        st.subheader("Preprocessing Steps Applied")
        st.markdown("""
        1. Loaded UCI Heart Disease Dataset
        2. Converted target to binary classification (0 = No Disease, 1 = Disease)
        3. **Missing value imputation will be handled AFTER train/test split** (to prevent data leakage)
        4. Ready for model training
        
        **Important:** Missing values are handled during model training using statistics calculated ONLY from the training set.
        """)
        
        st.subheader("Missing Values in Raw Data")
        missing = df_clean.isnull().sum()
        if missing.sum() > 0:
            st.warning("⚠️ Missing values detected. These will be imputed using training set statistics during model training.")
            st.dataframe(missing[missing > 0].to_frame('Missing Count'))
        else:
            st.success("No missing values found!")
        
        st.subheader("Feature Information")
        feature_info = {
            'age': 'Age in years',
            'sex': 'Sex (1 = male; 0 = female)',
            'cp': 'Chest pain type',
            'trestbps': 'Resting blood pressure',
            'chol': 'Serum cholesterol in mg/dl',
            'fbs': 'Fasting blood sugar > 120 mg/dl',
            'restecg': 'Resting electrocardiographic results',
            'thalach': 'Maximum heart rate achieved',
            'exang': 'Exercise induced angina',
            'oldpeak': 'ST depression induced by exercise',
            'slope': 'Slope of the peak exercise ST segment',
            'ca': 'Number of major vessels colored by flourosopy',
            'thal': 'Thalassemia',
            'target': 'Heart disease (0 = no, 1 = yes)'
        }
        
        for feature, description in feature_info.items():
            if feature in df_clean.columns:
                st.markdown(f"**{feature}**: {description}")
    
    elif page == "Model Training":
        st.header("Model Training")
        
        # Prepare data
        if 'target' in df_clean.columns:
            X = df_clean.drop('target', axis=1)
            y = df_clean['target']
            
            # Split data
            test_size = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
            random_state = st.sidebar.number_input("Random State", 0, 100, 42)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            

            X_train_clean, X_test_clean, imputation_values = preprocess_train_test(X_train, X_test)
            
            if imputation_values:
                st.info(f"📊 Missing values imputed using training set statistics: {len(imputation_values)} features")
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_clean)
            X_test_scaled = scaler.transform(X_test_clean)
            
            st.subheader("Training Configuration")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Training Samples", len(X_train))
                st.metric("Test Samples", len(X_test))
            with col2:
                st.metric("Features", X_train.shape[1])
                st.metric("Random State", random_state)
            
            # Train models
            if st.button("Train Models", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                models = {}
                predictions = {}
                probabilities = {}
                
                # Logistic Regression
                status_text.text("Training Logistic Regression...")
                progress_bar.progress(20)
                lr = LogisticRegression(random_state=random_state, max_iter=1000)
                lr.fit(X_train_scaled, y_train)
                models['Logistic Regression'] = lr
                predictions['Logistic Regression'] = lr.predict(X_test_scaled)
                probabilities['Logistic Regression'] = lr.predict_proba(X_test_scaled)[:, 1]
                
                # KNN
                status_text.text("Training K-Nearest Neighbors...")
                progress_bar.progress(50)
                k = st.sidebar.slider("K for KNN", 1, 20, 5)
                knn = KNeighborsClassifier(n_neighbors=k)
                knn.fit(X_train_scaled, y_train)
                models['KNN'] = knn
                predictions['KNN'] = knn.predict(X_test_scaled)
                probabilities['KNN'] = knn.predict_proba(X_test_scaled)[:, 1]
                
                # Random Forest
                status_text.text("Training Random Forest...")
                progress_bar.progress(80)
                n_estimators = st.sidebar.slider("Number of Trees", 10, 200, 100)
                rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
                rf.fit(X_train_clean, y_train)  # RF doesn't need scaling, but uses cleaned data
                models['Random Forest'] = rf
                predictions['Random Forest'] = rf.predict(X_test_clean)
                probabilities['Random Forest'] = rf.predict_proba(X_test_clean)[:, 1]
                
                progress_bar.progress(100)
                status_text.text("Training complete!")
                
                # Store in session state
                st.session_state.models = models
                st.session_state.predictions = predictions
                st.session_state.probabilities = probabilities
                st.session_state.X_train = X_train_clean
                st.session_state.X_train_scaled = X_train_scaled
                st.session_state.X_test = X_test_clean
                st.session_state.X_test_scaled = X_test_scaled
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.scaler = scaler
                st.session_state.imputation_values = imputation_values
                
                st.success("All models trained successfully!")
            
            if 'models' in st.session_state:
                st.subheader("Trained Models")
                for model_name in st.session_state.models.keys():
                    st.success(f"{model_name}")
        else:
            st.error("Target column not found in dataset!")
    
    elif page == "Model Evaluation":
        st.header("Model Evaluation")
        
        if 'models' not in st.session_state:
            st.warning("Please train the models first on the 'Model Training' page.")
        else:
            models = st.session_state.models
            predictions = st.session_state.predictions
            probabilities = st.session_state.probabilities
            y_test = st.session_state.y_test
            
            # Metrics comparison
            st.subheader("Performance Metrics Comparison")
            
            metrics_data = []
            for model_name in models.keys():
                y_pred = predictions[model_name]
                y_proba = probabilities[model_name]
                
                metrics_data.append({
                    'Model': model_name,
                    'Accuracy': accuracy_score(y_test, y_pred),
                    'Precision': precision_score(y_test, y_pred),
                    'Recall': recall_score(y_test, y_pred),
                    'F1-Score': f1_score(y_test, y_pred),
                    'ROC-AUC': roc_auc_score(y_test, y_proba)
                })
            
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df.style.format({
                'Accuracy': '{:.4f}',
                'Precision': '{:.4f}',
                'Recall': '{:.4f}',
                'F1-Score': '{:.4f}',
                'ROC-AUC': '{:.4f}'
            }), use_container_width=True)
            
            # Visualizations
            selected_model = st.selectbox("Select Model for Detailed Analysis", list(models.keys()))
            
            if selected_model:
                y_pred = predictions[selected_model]
                y_proba = probabilities[selected_model]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Confusion Matrix")
                    fig_cm = plot_confusion_matrix(y_test, y_pred, selected_model)
                    st.pyplot(fig_cm)
                
                with col2:
                    st.subheader("ROC Curve")
                    fig_roc = plot_roc_curve(y_test, y_proba, selected_model)
                    st.plotly_chart(fig_roc, use_container_width=True)
                
                st.subheader("Classification Report")
                report = classification_report(y_test, y_pred, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)
            
            # Metrics comparison chart
            st.subheader("Metrics Comparison Chart")
            metrics_melted = metrics_df.melt(
                id_vars='Model',
                value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
                var_name='Metric',
                value_name='Score'
            )
            
            fig = px.bar(
                metrics_melted,
                x='Model',
                y='Score',
                color='Metric',
                barmode='group',
                title="Model Performance Comparison"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Ensemble Analysis":
        st.header("Ensemble Analysis")
        
        if 'models' not in st.session_state:
            st.warning("Please train the models first on the 'Model Training' page.")
        else:
            st.subheader("Voting Ensemble Classifier")
            
            voting_type = st.radio("Voting Type", ["Soft Voting", "Hard Voting"])
            
            if st.button("Train Ensemble Model", type="primary"):
                models = st.session_state.models
                X_test = st.session_state.X_test
                X_test_scaled = st.session_state.X_test_scaled
                y_test = st.session_state.y_test
                X_train = st.session_state.get('X_train')
                X_train_scaled = st.session_state.get('X_train_scaled')
                y_train = st.session_state.get('y_train')
                
                if X_train is None or X_train_scaled is None or y_train is None:
                    st.error("Training data not found. Please retrain models on the 'Model Training' page.")
                else:
                    # Get the already trained models from session state
                    lr = st.session_state.models['Logistic Regression']
                    knn = st.session_state.models['KNN']
                    rf = st.session_state.models['Random Forest']
                    
                    # Create base estimators list
                    estimators = [
                        ('lr', lr),
                        ('knn', knn),
                        ('rf', rf)
                    ]
                    
                    # Create voting classifier
                    voting = 'soft' if voting_type == "Soft Voting" else 'hard'

                    if voting == 'soft':
                        # Get probabilities from each model
                        lr_proba = st.session_state.probabilities['Logistic Regression']
                        knn_proba = st.session_state.probabilities['KNN']
                        rf_proba = st.session_state.probabilities['Random Forest']
                        
                        # Average probabilities for soft voting
                        y_proba_ensemble = (lr_proba + knn_proba + rf_proba) / 3
                        y_pred_ensemble = (y_proba_ensemble >= 0.5).astype(int)
                    else:
                        # Hard voting: majority vote
                        lr_pred = st.session_state.predictions['Logistic Regression']
                        knn_pred = st.session_state.predictions['KNN']
                        rf_pred = st.session_state.predictions['Random Forest']
                        
                        # Majority vote
                        votes = np.array([lr_pred, knn_pred, rf_pred])
                        y_pred_ensemble = (votes.sum(axis=0) >= 2).astype(int)
                        y_proba_ensemble = None
                    
                    # Calculate metrics
                    acc = accuracy_score(y_test, y_pred_ensemble)
                    prec = precision_score(y_test, y_pred_ensemble)
                    rec = recall_score(y_test, y_pred_ensemble)
                    f1 = f1_score(y_test, y_pred_ensemble)
                    
                    st.session_state.ensemble_pred = y_pred_ensemble
                    st.session_state.ensemble_proba = y_proba_ensemble
                    st.session_state.ensemble_voting = voting_type
                    
                    st.success("Ensemble model trained!")
                    
                    # Display metrics
                    st.subheader("Ensemble Model Performance")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Accuracy", f"{acc:.4f}")
                    with col2:
                        st.metric("Precision", f"{prec:.4f}")
                    with col3:
                        st.metric("Recall", f"{rec:.4f}")
                    with col4:
                        st.metric("F1-Score", f"{f1:.4f}")
                    
                    if y_proba_ensemble is not None:
                        auc = roc_auc_score(y_test, y_proba_ensemble)
                        st.metric("ROC-AUC", f"{auc:.4f}")
                        
                        # ROC curve
                        fig_roc = plot_roc_curve(y_test, y_proba_ensemble, f"Ensemble ({voting_type})")
                        st.plotly_chart(fig_roc, use_container_width=True)
                    else:
                        # For hard voting, calculate probabilities from predictions for ROC
                        y_proba_ensemble_approx = y_pred_ensemble.astype(float)
                        auc = roc_auc_score(y_test, y_proba_ensemble_approx)
                        st.metric("ROC-AUC (approximate)", f"{auc:.4f}")
                    
                    # Compare with individual models
                    st.subheader("Comparison with Individual Models")
                    individual_metrics = []
                    for model_name in models.keys():
                        y_pred = st.session_state.predictions[model_name]
                        individual_metrics.append({
                            'Model': model_name,
                            'Accuracy': accuracy_score(y_test, y_pred)
                        })
                    
                    individual_metrics.append({
                        'Model': f'Ensemble ({voting_type})',
                        'Accuracy': acc
                    })
                    
                    comparison_df = pd.DataFrame(individual_metrics)
                    fig = px.bar(
                        comparison_df,
                        x='Model',
                        y='Accuracy',
                        title="Accuracy Comparison: Individual vs Ensemble",
                        color='Model'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            if 'ensemble_pred' in st.session_state:
                st.info(f"Ensemble model ({st.session_state.ensemble_voting}) is ready!")
    
    elif page == "Feature Selection":
        st.header("🔍 Feature Selection Analysis")
        
        if 'target' in df_clean.columns:
            X = df_clean.drop('target', axis=1)
            y = df_clean['target']
            
            st.subheader("Chi-Squared Feature Selection Test")
            
            # Apply chi-squared test
            # Select top k features
            k = st.slider("Number of Top Features to Select", 1, len(X.columns), min(10, len(X.columns)))
            
            if st.button("Run Chi-Squared Test", type="primary"):
                X_chi = X.copy()
                
                # Bin continuous variables
                for col in X_chi.columns:
                    if X_chi[col].dtype in ['float64', 'int64']:
                        n_unique = X_chi[col].nunique()
                        if n_unique > 10:  # Bin if more than 10 unique values
                            X_chi[col] = pd.qcut(X_chi[col], q=5, duplicates='drop', labels=False)
                            X_chi[col] = X_chi[col].fillna(0)
                
                # Ensure all values are non-negative integers
                X_chi = X_chi.astype(int)
                X_chi = X_chi - X_chi.min() + 1  # Shift to positive values
                
                # Apply chi-squared test
                chi_selector = SelectKBest(score_func=chi2, k=k)
                X_chi_selected = chi_selector.fit_transform(X_chi, y)
                
                # Get feature scores
                feature_scores = pd.DataFrame({
                    'Feature': X.columns,
                    'Chi-Squared Score': chi_selector.scores_,
                    'P-Value': chi_selector.pvalues_,
                    'Selected': [col in X.columns[chi_selector.get_support()] for col in X.columns]
                }).sort_values('Chi-Squared Score', ascending=False)
                
                st.session_state.feature_scores = feature_scores
                st.session_state.selected_features = X.columns[chi_selector.get_support()].tolist()
                
                st.success("Chi-Squared test completed!")
            
            if 'feature_scores' in st.session_state:
                st.subheader("Feature Importance Scores")
                st.dataframe(
                    st.session_state.feature_scores.style.format({
                        'Chi-Squared Score': '{:.2f}',
                        'P-Value': '{:.6f}'
                    }).apply(
                        lambda x: ['background-color: lightgreen' if x['Selected'] else '' for _ in x],
                        axis=1
                    ),
                    use_container_width=True
                )
                
                # Visualization
                st.subheader("Chi-Squared Scores Visualization")
                fig = px.bar(
                    st.session_state.feature_scores,
                    x='Feature',
                    y='Chi-Squared Score',
                    color='Selected',
                    title="Chi-Squared Feature Importance Scores",
                    color_discrete_map={True: 'green', False: 'lightblue'}
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Selected Features")
                st.write(f"Top {k} features most associated with heart disease:")
                for i, feature in enumerate(st.session_state.selected_features, 1):
                    score = st.session_state.feature_scores[
                        st.session_state.feature_scores['Feature'] == feature
                    ]['Chi-Squared Score'].values[0]
                    st.write(f"{i}. **{feature}** (Score: {score:.2f})")
        else:
            st.error("Target column not found in dataset!")

else:
    st.error("Failed to load dataset. Please check your internet connection and try again.")
