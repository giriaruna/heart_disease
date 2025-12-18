import streamlit as st
import os
import sys

# Page configuration MUST be first Streamlit command
st.set_page_config(
    page_title="Heart Disease Classification",
    layout="wide"
)

# Debug: Show current directory and check for requirements.txt
if st.sidebar.checkbox("Show Debug Info", value=False):
    st.sidebar.write("**Current directory:**", os.getcwd())
    st.sidebar.write("**Python path:**", sys.executable)
    req_path = os.path.join(os.getcwd(), "requirements.txt")
    st.sidebar.write("**Requirements.txt exists:**", os.path.exists(req_path))
    if os.path.exists(req_path):
        with open(req_path, 'r') as f:
            st.sidebar.write("**Contents:**")
            st.sidebar.code(f.read())

# Check for required dependencies
try:
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
except ImportError as e:
    st.error(f"❌ Missing required dependency: {e}")
    st.error("Please ensure all packages in requirements.txt are installed.")
    
    # Show helpful troubleshooting info
    st.markdown("### Troubleshooting Steps:")
    st.markdown("""
    1. **Check repository structure on GitHub:**
       - `requirements.txt` must be in the **same directory** as `streamlit_app.py`
       - If your app is at `heart_disease/streamlit_app.py`, then `requirements.txt` should be at `heart_disease/requirements.txt`
    
    2. **Verify the file is committed:**
       - Check that `requirements.txt` is committed to your GitHub repository
       - It should appear in the same directory as `streamlit_app.py` on GitHub
    
    3. **On Streamlit Cloud:**
       - Go to "Manage app" → "Settings"
       - Verify "Main file path" is correct
       - Click "Reboot app" to trigger a fresh install
    
    4. **Check logs:**
       - In "Manage app" → "Logs"
       - Look for "Installing requirements from requirements.txt"
       - If you don't see this message, the file isn't being found
    """)
    
    # Try to show where requirements.txt should be
    current_dir = os.getcwd()
    st.info(f"**Current working directory:** `{current_dir}`")
    st.info(f"**Expected requirements.txt location:** `{os.path.join(current_dir, 'requirements.txt')}`")
    
    st.stop()

# Title
st.title("Heart Disease Classification")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    [
        "Project Overview",
        "Data Overview",
        "Data Preprocessing",
        "Model Training",
        "Model Evaluation",
        "Ensemble Analysis",
        "Feature Selection",
        "Conclusion"
    ]
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
        # Calculate median from training data only (even if no missing values in train)
        train_median = X_train_clean[col].median()
        
        # Handle edge case: if all values are NaN in training, use 0 as fallback
        if pd.isna(train_median):
            train_median = 0
        
        imputation_values[col] = train_median
        
        # Apply to both training and test sets (handles NaNs in either set)
        X_train_clean[col].fillna(train_median, inplace=True)
        X_test_clean[col].fillna(train_median, inplace=True)
    
    # Handle non-numeric columns with missing values (use mode for categorical)
    non_numeric_cols = X_train_clean.select_dtypes(exclude=[np.number]).columns
    for col in non_numeric_cols:
        if X_train_clean[col].isnull().sum() > 0 or X_test_clean[col].isnull().sum() > 0:
            # Use mode from training data
            train_mode = X_train_clean[col].mode()
            if len(train_mode) > 0:
                mode_value = train_mode[0]
            else:
                mode_value = 0  # Fallback
            imputation_values[col] = mode_value
            X_train_clean[col].fillna(mode_value, inplace=True)
            X_test_clean[col].fillna(mode_value, inplace=True)
    
    # Final check: ensure no NaNs remain
    if X_train_clean.isnull().sum().sum() > 0 or X_test_clean.isnull().sum().sum() > 0:
        # If still NaNs, fill with 0 as last resort
        X_train_clean = X_train_clean.fillna(0)
        X_test_clean = X_test_clean.fillna(0)
    
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

# ADDITION (START): Helper functions for model explanations and problem formulation
# =========================================================================================================================================================================================================
def get_problem_formulation_table(model_name):
    formulations = {
        "Logistic Regression": {
            "Training Data": r"$X_{\text{train\_scaled}} \in \mathbb{R}^{n \times d}, y_{\text{train}} \in \{0,1\}$",
            "Loss Function": "Binary Cross-Entropy (Log Loss)",
            "Training Procedure": "Gradient-based optimization (LBFGS solver)",
            "Model Output": r"$P(y=1|x) \text{ via sigmoid}$"
        },
        "KNN": {
            "Training Data": r"$X_{\text{train\_scaled}}$ with labels $y_{\text{train}}$", # Corrected LaTeX for KNN
            "Loss Function": "None (instance-based learning)",
            "Training Procedure": "Majority vote among k nearest neighbors",
            "Model Output": "Class label"
        },
        "Random Forest": {
            "Training Data": r"$X_{\text{train\_clean}}$ (unscaled)", # Corrected LaTeX for Random Forest
            "Loss Function": "Gini Impurity",
            "Training Procedure": "Bagging + greedy decision tree splitting",
            "Model Output": "Mean class probability"
        }
    }
    return formulations.get(model_name, {})

def get_model_specific_explanation(model_name, visualization_type, current_metrics=None, y_test=None, y_pred=None, y_proba=None):
    explanations = {
        "Logistic Regression": {
            "confusion_matrix": "Logistic Regression classifies by applying a sigmoid function to a linear combination of features, outputting a probability. If this probability exceeds 0.5, it predicts 'Disease', otherwise 'No Disease'. The confusion matrix reflects these binary predictions.",
            "roc_curve": "Logistic Regression generates a probability score for heart disease. The ROC curve illustrates how well this model distinguishes between positive and negative classes across various probability thresholds.",
            "classification_report": "The classification report for Logistic Regression details its performance (precision, recall, F1-score) for each class based on its binary predictions."
        },
        "KNN": {
            "confusion_matrix": "K-Nearest Neighbors classifies a patient based on the majority class among its 'k' closest neighbors in the training data. This forms the basis for its predictions in the confusion matrix.",
            "roc_curve": "For KNN, the probability of a class is often derived from the proportion of 'k' neighbors belonging to that class. The ROC curve shows its discriminative ability based on these proportions.",
            "classification_report": "The classification report for KNN summarizes its performance metrics based on the class labels assigned by its neighborhood voting mechanism."
        },
        "Random Forest": {
            "confusion_matrix": "Random Forest makes a prediction by averaging the predictions (or probabilities) of multiple decision trees. The final majority vote (or averaged probability threshold) determines the class shown in the confusion matrix.",
            "roc_curve": "Random Forest combines probability estimates from numerous decision trees. The ROC curve demonstrates its ability to differentiate between classes using these aggregated probabilities.",
            "classification_report": "The classification report for Random Forest evaluates its aggregated classification performance, showing precision, recall, and F1-score for each class based on its ensemble predictions."
        }
    }
    
    # Dynamic interpretation for Classification Report (specific to chosen model)
    if visualization_type == "classification_report_detailed" and current_metrics is not None:
        report = classification_report(y_test, y_pred, output_dict=True)
        return f"""
        In this binary classification task for heart disease:
        - **Class 0 represents 'No Disease'.**
        - **Class 1 represents 'Disease'.**

        For the **{model_name}** model:
        - **Precision for Class 0 ({report['0']['precision']:.4f}):** When the model predicts 'No Disease', it is correct {report['0']['precision']:.2%} of the time. This is important for ensuring healthy individuals are not misclassified.
        - **Recall for Class 0 ({report['0']['recall']:.4f}):** Of all actual 'No Disease' cases, the model correctly identified {report['0']['recall']:.2%} of them. This indicates how well the model avoids false negatives for healthy patients.
        - **F1-Score for Class 0 ({report['0']['f1-score']:.4f}):** This balances precision and recall for the 'No Disease' class, indicating overall accuracy for this group.

        - **Precision for Class 1 ({report['1']['precision']:.4f}):** When the model predicts 'Disease', it is correct {report['1']['precision']:.2%} of the time. This tells us how trustworthy a positive diagnosis is.
        - **Recall for Class 1 ({report['1']['recall']:.4f}):** Of all actual 'Disease' cases, the model correctly identified {report['1']['recall']:.2%} of them. This is often the most critical metric in medical diagnosis, as a high recall means fewer actual disease cases are missed (fewer false negatives).
        - **F1-Score for Class 1 ({report['1']['f1-score']:.4f}):** This provides a balanced measure of the model's performance specifically for detecting heart disease.

        The `accuracy`, `macro avg`, and `weighted avg` rows provide overall summary statistics across both classes.
        """
    
    # Dynamic interpretation for ROC Curve (specific to chosen model)
    if visualization_type == "roc_curve_detailed" and y_test is not None and y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_score = roc_auc_score(y_test, y_proba)
        return f"""
        This graph, the **Receiver Operating Characteristic (ROC) curve**, visualizes the trade-off between the True Positive Rate (TPR) and False Positive Rate (FPR) at various threshold settings.
        - The **X-axis (False Positive Rate)** represents the proportion of healthy individuals incorrectly classified as having heart disease. A lower FPR is generally desired.
        - The **Y-axis (True Positive Rate / Recall)** represents the proportion of actual heart disease cases correctly identified. A higher TPR is generally desired.
        - The blue solid line shows the performance of the **{model_name}** model.
        - The dashed gray line represents a **random classifier**, which performs no better than chance (AUC = 0.5).
        - The **Area Under the Curve (AUC)** for {model_name} is **{auc_score:.3f}**. This value, ranging from 0 to 1, quantifies the model's overall ability to distinguish between positive and negative classes. An AUC closer to 1 indicates better discriminative power.
        - If the curve stays high on the Y-axis and close to the left side of the X-axis, it indicates excellent performance. A curve that hugs the top-left corner means the model has a high TPR while maintaining a low FPR across different thresholds.
        """
    
    # Dynamic interpretation for Confusion Matrix (specific to chosen model)
    if visualization_type == "confusion_matrix_detailed" and y_test is not None and y_pred is not None:
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        return f"""
        The **Confusion Matrix** provides a detailed breakdown of the model's predictions versus the actual outcomes for the **{model_name}** model.
        - **True Negatives (Top-Left, {tn}):** The number of patients who *did not* have heart disease and were correctly predicted as 'No Disease'.
        - **False Positives (Top-Right, {fp}):** The number of patients who *did not* have heart disease but were incorrectly predicted as 'Disease' (Type I error).
        - **False Negatives (Bottom-Left, {fn}):** The number of patients who *did* have heart disease but were incorrectly predicted as 'No Disease' (Type II error). This is often the most critical error in medical diagnosis.
        - **True Positives (Bottom-Right, {tp}):** The number of patients who *did* have heart disease and were correctly predicted as 'Disease'.

        A good model aims to maximize True Positives and True Negatives, while minimizing False Positives and False Negatives. The specific numbers here allow us to quantify these outcomes.
        """

    return explanations.get(model_name, {}).get(visualization_type, "")

def get_model_description(model_name):
    descriptions = {
        "Logistic Regression": "A linear model that uses the sigmoid function to output a probability score for binary classification. It's effective for understanding feature impact.",
        "K-Nearest Neighbors (KNN)": "A non-parametric, instance-based learning algorithm that classifies new data points based on the majority class of their 'k' nearest training examples. It's simple but can be computationally intensive for large datasets.",
        "Random Forest": "An ensemble learning method that builds multiple decision trees during training and combines their predictions. It's robust to overfitting and captures complex relationships.",
        "Voting Ensemble": "Combines predictions from multiple individual models (Logistic Regression, KNN, Random Forest) to produce a more robust and often more accurate final prediction by leveraging their diverse strengths."
    }
    return descriptions.get(model_name, "")

def get_overall_model_performance_insight(model_name, metrics):
    """Generates a plain-language insight for a single model's overall performance."""
    accuracy = metrics['Accuracy']
    precision = metrics['Precision']
    recall = metrics['Recall']
    f1 = metrics['F1-Score']
    roc_auc = metrics['ROC-AUC']

    insight = f"The {model_name} model's performance on the unseen test data indicates a general capability to identify heart disease. "

    if accuracy >= 0.85:
        insight += "It shows a strong overall correctness in its predictions, suggesting it's quite reliable. "
    elif accuracy >= 0.75:
        insight += "Its overall correctness is respectable, offering a good balance in identifying both disease and non-disease cases. "
    else:
        insight += "However, its overall correctness suggests there might be room for improvement, as it struggles with a notable portion of the predictions. "

    if recall >= 0.8:
        insight += "Crucially, its ability to detect actual heart disease cases (recall) is very high, which is excellent for minimizing missed diagnoses. "
    elif recall >= 0.7:
        insight += "The model identifies a good proportion of actual heart disease cases, showing a reasonable capacity to avoid false negatives. "
    else:
        insight += "Its recall, or ability to identify actual heart disease cases, is somewhat low, indicating it might be missing a significant number of patients who actually have the disease. "

    if precision >= 0.8:
        insight += "When it predicts heart disease, it's highly trustworthy, with very few incorrect positive diagnoses. "
    elif precision >= 0.7:
        insight += "Its positive predictions for heart disease are fairly reliable, suggesting a good balance in not over-diagnosing. "
    else:
        insight += "The reliability of its positive heart disease predictions is a concern, as it tends to incorrectly flag healthy patients as having the disease more often. "
    
    if roc_auc >= 0.85:
        insight += "The model demonstrates excellent discriminative power, effectively distinguishing between patients with and without heart disease across various thresholds. "
    elif roc_auc >= 0.75:
        insight += "Its ability to differentiate between the two classes is solid, providing a good separation between healthy and diseased patients. "
    else:
        insight += "The model's discriminative power, while present, could be strengthened, indicating some overlap in its ability to separate the two patient groups. "

    return insight.strip()

def get_ensemble_performance_insight(ensemble_metrics, individual_metrics_df, voting_type):
    """Generates a plain-language insight for the ensemble model's overall performance."""
    acc = ensemble_metrics['Accuracy']
    rec = ensemble_metrics['Recall']
    auc = ensemble_metrics['ROC-AUC']

    best_individual_acc = individual_metrics_df['Accuracy'].max()
    best_individual_rec = individual_metrics_df['Recall'].max()
    best_individual_auc = individual_metrics_df['ROC-AUC'].max()

    insight = f"The {voting_type} Ensemble model's performance represents a combined effort from its individual components. "

    if acc > best_individual_acc:
        insight += f"Notably, the ensemble achieved a higher accuracy ({acc:.2%}) than any single model ({best_individual_acc:.2%}), suggesting that combining their predictions successfully leveraged their diverse strengths. "
    elif abs(acc - best_individual_acc) < 0.01: # Within 1%
        insight += f"The ensemble's accuracy ({acc:.2%}) is very competitive with the best individual model ({best_individual_acc:.2%}), indicating a robust and consistent performance without significant degradation. "
    else:
        insight += f"While the ensemble provides a consolidated view, its accuracy ({acc:.2%}) did not surpass the best individual model ({best_individual_acc:.2%}), which might suggest the voting strategy or combination could be further optimized. "

    if rec > best_individual_rec:
        insight += f"More importantly in a medical context, its recall ({rec:.2%}) is higher than any individual model ({best_individual_rec:.2%}), signifying an enhanced capability to detect actual heart disease cases and minimize critical false negatives. "
    elif abs(rec - best_individual_rec) < 0.01:
        insight += f"The ensemble maintains a strong recall ({rec:.2%}), on par with the best individual model ({best_individual_rec:.2%}), which is crucial for not missing actual disease cases. "
    else:
        insight += f"Its recall ({rec:.2%}) is slightly lower than the best individual model's ({best_individual_rec:.2%}), which is an area that might warrant further investigation to ensure no actual disease cases are overlooked. "
    
    if auc > best_individual_auc:
        insight += f"With an improved ROC-AUC ({auc:.3f}) compared to individual models ({best_individual_auc:.3f}), the ensemble demonstrates a superior overall ability to distinguish between healthy and diseased patients across various thresholds. "
    elif abs(auc - best_individual_auc) < 0.01:
        insight += f"The ensemble exhibits robust discriminative power, with an ROC-AUC ({auc:.3f}) closely matching that of the best individual model ({best_individual_auc:.3f}), confirming its strong capability in separating the classes. "
    else:
        insight += f"The ensemble's ROC-AUC ({auc:.3f}) is slightly lower than the best individual model's ({best_individual_auc:.3f}), indicating that while the ensemble is generally strong, there might be a slight compromise in overall separability compared to the top individual performer. "

    insight += f"This {voting_type} approach aims to leverage the 'wisdom of the crowd' to provide a more reliable and generalized prediction, often smoothing out the weaknesses of individual models."

    return insight.strip()

# =========================================================================================================================================================================================================
# ADDITION (END): Helper functions for model explanations and problem formulation


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
    if page == "Project Overview":
        st.markdown("**Predicting Heart Disease Using Classical Machine Learning Models**")
        st.markdown("---")

        st.markdown("""
            This project investigates whether **machine learning models** can accurately predict
            the presence of **heart disease** using common clinical measurements.

            ### Problem Statement
            Heart disease is one of the leading causes of death worldwide.
            Early detection can significantly improve treatment outcomes.
            This project aims to build and evaluate machine learning models that assist in
            identifying patients at risk based on medical data.
            """)

        st.markdown("""
            ### Dataset
            We use the **UCI Heart Disease Dataset**, which contains medical records for over
            **300 patients** with **14 clinical attributes**, including:

            - Age
            - Resting blood pressure
            - Cholesterol level
            - Maximum heart rate
            - Exercise-induced angina
            """)
        st.subheader("Project Goals")

        st.markdown("""
            The primary objectives of this project are:

            - To **compare multiple machine learning models**
            (Logistic Regression, K-Nearest Neighbors, and Random Forest)
            for binary heart disease classification.

            - To evaluate whether **ensemble learning**
            improves predictive performance, stability, and robustness.

            - To apply **Chi-Squared feature selection**
            in order to identify the medical features most strongly associated
            with the presence of heart disease.

            - To analyze how **data preprocessing and feature selection**
            influence model performance and evaluation metrics.
            """)
        
        
        st.markdown("---") 
        st.subheader("Course Information") 
        st.markdown(f"**Syllabus:** [ECE-UY 4563 – Machine Learning (Fall 2025)](https://nikopj.github.io/assets/introml25/syllabus.pdf)") 
        st.markdown("**Instructor:** Prof. Nikola Janjušević") 
        st.markdown("**Contact:** ml25@nyu.edu") 
        st.markdown("---") 
        st.subheader("Team Members") 
        st.markdown("- **Aruna Giri** (ag8876@nyu.edu)") 
        st.markdown("- **İlayda Dilek** (id2275@nyu.edu)") 
        st.markdown("---")

        st.markdown(
            "**Dataset Source:** "
            "[UCI Machine Learning Repository – Heart Disease Dataset]"
            "(https://archive.ics.uci.edu/ml/datasets/Heart+Disease)"
        )

    elif page == "Data Overview":
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
                st.warning("Missing values detected in raw data. These will be properly handled during model training using statistics calculated ONLY from the training set (to prevent data leakage).")
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

            selected_feature = st.selectbox(
                "Select a feature to visualize",
                numeric_cols
            )

            if selected_feature:
                # ====== Plot ======
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))

                # Histogram
                axes[0].hist(df_clean[selected_feature], bins=30, edgecolor='black')
                axes[0].set_title(f'Distribution of {selected_feature}')
                axes[0].set_xlabel(selected_feature)
                axes[0].set_ylabel('Frequency')

                # Box plot by target
                df_clean.boxplot(
                    column=selected_feature,
                    by='target',
                    ax=axes[1]
                )
                axes[1].set_title(f'{selected_feature} by Heart Disease Status')
                axes[1].set_xlabel('Heart Disease (0 = No, 1 = Yes)')
                axes[1].set_ylabel(selected_feature)

                plt.suptitle("")  # Remove auto title
                plt.tight_layout()
                st.pyplot(fig)

                # ====== Statistics ======
                overall_mean = df_clean[selected_feature].mean()
                overall_median = df_clean[selected_feature].median()

                no_disease = df_clean[df_clean['target'] == 0][selected_feature]
                disease = df_clean[df_clean['target'] == 1][selected_feature]

                mean_no = no_disease.mean()
                mean_yes = disease.mean()

                median_no = no_disease.median()
                median_yes = disease.median()

               
                st.markdown("### 📊 Feature Interpretation")

                st.markdown(f"""
                The plots above show the distribution of **{selected_feature}**:

                - The **histogram** displays how values are spread across all patients.
                - The **boxplot** shows how **{selected_feature}** differs between patients **with** and **without heart disease**, highlighting class-specific patterns.

                These visualizations give a quick sense of which features may help the models distinguish between the two classes. 
                The models themselves are trained on individual patient samples, not these summary statistics.
                """)

    
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
            st.warning("Missing values detected. These will be imputed using training set statistics during model training.")
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
            
            # Hyperparameters - define sliders outside button to prevent unnecessary reruns
            test_size = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
            random_state = st.sidebar.number_input("Random State", 0, 100, 42)
            k_neighbors = st.sidebar.slider("K for KNN", 1, 20, 5)
            n_estimators = st.sidebar.slider("Number of Trees", 10, 200, 100)
            
            # Check if we need to resplit data (if test_size or random_state changed)
            split_key = f"split_{test_size}_{random_state}"
            if split_key not in st.session_state or st.session_state.get('test_size') != test_size or st.session_state.get('random_state') != random_state:
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, stratify=y
                )
                
                # Preprocess
                X_train_clean, X_test_clean, imputation_values = preprocess_train_test(X_train, X_test)
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_clean)
                X_test_scaled = scaler.transform(X_test_clean)
                
                # Store in session state
                st.session_state[split_key] = {
                    'X_train': X_train,
                    'X_test': X_test,
                    'X_train_clean': X_train_clean,
                    'X_test_clean': X_test_clean,
                    'X_train_scaled': X_train_scaled,
                    'X_test_scaled': X_test_scaled,
                    'y_train': y_train,
                    'y_test': y_test,
                    'scaler': scaler,
                    'imputation_values': imputation_values
                }
                st.session_state.test_size = test_size
                st.session_state.random_state = random_state
                # Clear models if data split changed
                if 'models' in st.session_state:
                    del st.session_state.models
                    del st.session_state.predictions
                    del st.session_state.probabilities
            
            # Retrieve data from session state
            split_data = st.session_state[split_key]
            X_train = split_data['X_train']
            X_test = split_data['X_test']
            X_train_clean = split_data['X_train_clean']
            X_test_clean = split_data['X_test_clean']
            X_train_scaled = split_data['X_train_scaled']
            X_test_scaled = split_data['X_test_scaled']
            y_train = split_data['y_train']
            y_test = split_data['y_test']
            scaler = split_data['scaler']
            imputation_values = split_data['imputation_values']
            
            # Verify no NaNs remain
            train_nans = X_train_clean.isnull().sum().sum()
            test_nans = X_test_clean.isnull().sum().sum()
            
            if train_nans > 0 or test_nans > 0:
                st.error(f"Warning: {train_nans} NaNs in training set, {test_nans} NaNs in test set after imputation!")
            elif imputation_values:
                st.success(f"=All missing values handled. Imputed {len(imputation_values)} features using training set statistics.")
            else:
                st.info("No missing values detected in the dataset.")
            
            st.subheader("Training Configuration")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Training Samples", len(X_train))
                st.markdown(f"*(The number of data points used to train the models, here {len(X_train)} samples.)*")
            with col2:
                st.metric("Features", X_train.shape[1])
                st.markdown(f"*(The number of input variables used for prediction, here {X_train.shape[1]} features.)*")
            col3, col4 = st.columns(2)
            with col3:
                st.metric("Test Samples", len(X_test))
                st.markdown(f"*(The number of unseen data points used to evaluate the models, here {len(X_test)} samples.)*")
            with col4:
                st.metric("Random State", random_state)
                st.markdown(f"*(A seed used for reproducibility in data splitting and model training, currently set to {random_state}.)*")
            
            # Check if models need retraining (if hyperparameters changed)
            need_retrain = False
            if 'models' not in st.session_state:
                need_retrain = True
            elif st.session_state.get('k_neighbors') != k_neighbors or st.session_state.get('n_estimators') != n_estimators:
                need_retrain = True
            
            if need_retrain:
                st.info("Hyperparameters changed. Click 'Click here to train the models' to retrain with new settings.")
            
            # Train models
            # ADDITION (START): Changed button text
            if st.button("Click here to train the models", type="primary"):
            # ADDITION (END): Changed button text
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
                knn = KNeighborsClassifier(n_neighbors=k_neighbors)
                knn.fit(X_train_scaled, y_train)
                models['KNN'] = knn
                predictions['KNN'] = knn.predict(X_test_scaled)
                probabilities['KNN'] = knn.predict_proba(X_test_scaled)[:, 1]
                
                # Random Forest
                status_text.text("Training Random Forest...")
                progress_bar.progress(80)
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
                st.session_state.k_neighbors = k_neighbors
                st.session_state.n_estimators = n_estimators
                
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
            st.markdown("""
                Here we compare the core evaluation metrics across all trained models.
                - **Accuracy:** Overall proportion of correctly classified patients.
                - **Precision:** How often predicted 'disease' cases are actually true.
                - **Recall (Sensitivity):** How many actual disease cases are correctly identified.
                - **F1-Score:** Balance between precision and recall.
                - **ROC-AUC:** Ability of the model to discriminate between patients with and without heart disease.
                """)
            
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
                # ADDITION (START): Problem formulation section moved here and modified
                # =========================================================================================================================================================================================================
                st.markdown("---")
                st.subheader("Problem Formulation") # Removed "(Automatically Generated)"

                st.markdown(f"**Problem Formulation for {selected_model}:**") # Show formulation for selected model only
                formulation = get_problem_formulation_table(selected_model)
                if formulation:
                    # Convert specific rows to LaTeX if they contain formula-like strings
                    for component, description in formulation.items():
                        st.markdown(f"**{component}:** {description}") # Render directly as description contains the $
                # =========================================================================================================================================================================================================
                # ADDITION (END): Problem formulation section

                y_pred = predictions[selected_model]
                y_proba = probabilities[selected_model]
                st.markdown("""
                    These visualizations help us understand **how the model is performing on the test set**:
                    - **Confusion Matrix:** Shows true positives, true negatives, false positives, and false negatives. 
                    This helps identify whether the model tends to **miss disease cases (false negatives)** or 
                    **overpredict disease (false positives)**.
                    - **ROC Curve:** Plots the tradeoff between true positive rate and false positive rate across thresholds.
                    A higher area under the curve indicates better **discriminative power**.
                    - **Classification Report:** Breaks down precision, recall, and F1-score for each class,
                    giving detailed insight into performance per class.
                    """)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Confusion Matrix")
                    # ADDITION (START): Model-specific explanation for Confusion Matrix
                    # =========================================================================================================================================================================================================
                    st.markdown(get_model_specific_explanation(selected_model, 'confusion_matrix_detailed', y_test=y_test, y_pred=y_pred))
                    # =========================================================================================================================================================================================================
                    # ADDITION (END): Model-specific explanation
                    fig_cm = plot_confusion_matrix(y_test, y_pred, selected_model)
                    st.pyplot(fig_cm)
                
                with col2:
                    st.subheader("ROC Curve")
                    # ADDITION (START): Model-specific explanation for ROC Curve
                    # =========================================================================================================================================================================================================
                    st.markdown(get_model_specific_explanation(selected_model, 'roc_curve_detailed', y_test=y_test, y_proba=y_proba))
                    # =========================================================================================================================================================================================================
                    # ADDITION (END): Model-specific explanation
                    fig_roc = plot_roc_curve(y_test, y_proba, selected_model)
                    st.plotly_chart(fig_roc, use_container_width=True)
                
                st.subheader("Classification Report")
                # ADDITION (START): Model-specific explanation for Classification Report
                # =========================================================================================================================================================================================================
                st.markdown(get_model_specific_explanation(selected_model, 'classification_report', y_test=y_test, y_pred=y_pred))
                # =========================================================================================================================================================================================================
                # ADDITION (END): Model-specific explanation
                report = classification_report(y_test, y_pred, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)
                
                # ADDITION (START): Explanation for 0 and 1 in Classification Report
                # =========================================================================================================================================================================================================
                st.markdown("---")
                st.subheader("Interpreting the Classification Report (Classes 0 and 1)")
                st.markdown(get_model_specific_explanation(selected_model, 'classification_report_detailed', y_test=y_test, y_pred=y_pred))
                # =========================================================================================================================================================================================================
                # ADDITION (END): Explanation for 0 and 1 in Classification Report
            
            # ADDITION (START): Evaluation Interpretation
            # =========================================================================================================================================================================================================
            st.markdown("---")
            st.subheader("Overall Model Performance Insights") # Changed heading

            st.markdown("Here's a quantitative summary of each model's performance on the test set:")
            for model_name in models.keys():
                metrics = metrics_df[metrics_df['Model'] == model_name].iloc[0]
                st.markdown(f"### {model_name}")
                st.markdown(get_overall_model_performance_insight(model_name, metrics))
            
            # Additional summary based on best performing models for key metrics
            best_acc = metrics_df.loc[metrics_df["Accuracy"].idxmax()]
            best_rec = metrics_df.loc[metrics_df["Recall"].idxmax()]
            best_auc = metrics_df.loc[metrics_df["ROC-AUC"].idxmax()]

            st.markdown("---")
            st.subheader("Key Comparative Takeaways")
            st.markdown(f"""
            - **Overall Best Accuracy:** The **{best_acc['Model']}** model achieved the highest accuracy of **{best_acc['Accuracy']:.4f}**. This means it had the highest percentage of correct predictions across both classes.
            - **Best Disease Detection (Recall):** The **{best_rec['Model']}** model showed the highest recall of **{best_rec['Recall']:.4f}**. This is critical in medical diagnosis as it signifies the model's superior ability to correctly identify actual heart disease patients, thus minimizing missed diagnoses.
            - **Strongest Discriminative Power (ROC-AUC):** The **{best_auc['Model']}** model demonstrated the best ROC-AUC of **{best_auc['ROC-AUC']:.4f}**. A higher ROC-AUC suggests that this model is most effective at distinguishing between positive and negative cases across all possible thresholds, making it robust to threshold choices.
            """)
            # =========================================================================================================================================================================================================
            # ADDITION (END): Evaluation Interpretation

            # Metrics comparison chart
            st.subheader("Metrics Comparison Chart")
            st.markdown("""
                This grouped bar chart visualizes **all models across multiple metrics**:
                - It allows us to quickly see which model performs best overall.
                - Highlights tradeoffs: e.g., a model might have higher accuracy but lower recall.
                - Useful for deciding **which model or ensemble might be most appropriate** for clinical prediction.
                """)
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
            st.markdown(f"""
            The **Metrics Comparison Chart** visually compares the performance of Logistic Regression, KNN, and Random Forest across various metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC).
            - The **X-axis** shows the different Machine Learning Models.
            - The **Y-axis** represents the 'Score' for each metric, typically ranging from 0 to 1 (or 0% to 100%).
            - Each group of bars represents a model, and different colored bars within a group represent different metrics.
            - **Interpreting Bar Heights:** A taller bar for a specific metric generally indicates better performance for that model on that particular metric. For example, if the 'Recall' bar for KNN is taller than for Logistic Regression, it means KNN achieved a higher recall.
            - This chart helps quickly identify which models excel in certain aspects (e.g., high recall for medical safety) and where trade-offs exist (e.g., a model might have high accuracy but lower recall, or vice versa).
            """)
    
    elif page == "Ensemble Analysis":
        st.header("Ensemble Analysis")
        
        if 'models' not in st.session_state:
            st.warning("Please train the models first on the 'Model Training' page.")
        else:
            st.subheader("Voting Ensemble Classifier")
            st.markdown(f"*{get_model_description('Voting Ensemble')}*")
            
            voting_type = st.radio("Voting Type", ["Soft Voting", "Hard Voting"])
            st.markdown("""
            - **Soft Voting:** Averages the predicted probabilities from individual models. The class with the highest average probability is chosen. This gives more nuanced decision-making.
            - **Hard Voting:** Uses the predicted class labels and selects the class that receives the majority vote from individual models. This is a simpler, 'winner-takes-all' approach.
            """)
            
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
                        st.markdown(f"*(The ensemble correctly classified {acc:.2%} of all patients.)*")
                    with col2:
                        st.metric("Precision", f"{prec:.4f}")
                        st.markdown(f"*(When the ensemble predicted disease, it was correct {prec:.2%} of the time.)*")
                    with col3:
                        st.metric("Recall", f"{rec:.4f}")
                        st.markdown(f"*(The ensemble identified {rec:.2%} of actual disease cases, crucial for minimizing false negatives.)*")
                    with col4:
                        st.metric("F1-Score", f"{f1:.4f}")
                        st.markdown(f"*(The ensemble's F1-Score of {f1:.2%} balances precision and recall.)*")
                    
                    ensemble_metrics_data = {
                        'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1, 'ROC-AUC': None
                    }

                    if y_proba_ensemble is not None:
                        auc = roc_auc_score(y_test, y_proba_ensemble)
                        st.metric("ROC-AUC", f"{auc:.4f}")
                        st.markdown(f"*(The ensemble's ROC-AUC of {auc:.3f} indicates its strong discriminative power.)*")
                        ensemble_metrics_data['ROC-AUC'] = auc
                        
                        # ROC curve
                        fig_roc = plot_roc_curve(y_test, y_proba_ensemble, f"Ensemble ({voting_type})")
                        st.plotly_chart(fig_roc, use_container_width=True)
                        st.markdown(f"""
                        This ROC curve for the **Ensemble ({voting_type})** model visualizes its performance.
                        - The **X-axis (False Positive Rate)** shows the rate of healthy individuals incorrectly classified as diseased.
                        - The **Y-axis (True Positive Rate / Recall)** shows the rate of actual diseased individuals correctly identified.
                        - The curve's proximity to the top-left corner (high TPR, low FPR) indicates better performance.
                        - The **Area Under the Curve (AUC)** of **{auc:.3f}** quantifies the ensemble's overall ability to distinguish between the two classes across all thresholds. A higher AUC means better separation.
                        """)
                    else:
                        # For hard voting, calculate probabilities from predictions for ROC
                        y_proba_ensemble_approx = y_pred_ensemble.astype(float)
                        auc = roc_auc_score(y_test, y_proba_ensemble_approx)
                        st.metric("ROC-AUC (approximate)", f"{auc:.4f}")
                        st.markdown(f"*(The ensemble's ROC-AUC of {auc:.3f} indicates its strong discriminative power.)*")
                        ensemble_metrics_data['ROC-AUC'] = auc
                        st.markdown(f"""
                        This ROC curve for the **Ensemble ({voting_type})** model visualizes its performance based on approximate probabilities from hard voting.
                        - The **X-axis (False Positive Rate)** shows the rate of healthy individuals incorrectly classified as diseased.
                        - The **Y-axis (True Positive Rate / Recall)** shows the rate of actual diseased individuals correctly identified.
                        - The curve's proximity to the top-left corner (high TPR, low FPR) indicates better performance.
                        - The **Area Under the Curve (AUC)** of **{auc:.3f}** quantifies the ensemble's overall ability to distinguish between the two classes across all thresholds. A higher AUC means better separation.
                        """)
                    
                    # ADDITION (START): Overall Ensemble Performance Insights
                    # =========================================================================================================================================================================================================
                    st.markdown("---")
                    st.subheader("Overall Ensemble Performance Insights")
                    # Need individual metrics to compare against
                    individual_metrics_df = pd.DataFrame([
                        {'Model': name, 
                         'Accuracy': accuracy_score(y_test, st.session_state.predictions[name]),
                         'Precision': precision_score(y_test, st.session_state.predictions[name]),
                         'Recall': recall_score(y_test, st.session_state.predictions[name]),
                         'F1-Score': f1_score(y_test, st.session_state.predictions[name]),
                         'ROC-AUC': roc_auc_score(y_test, st.session_state.probabilities[name])}
                        for name in models.keys()
                    ])
                    st.markdown(get_ensemble_performance_insight(ensemble_metrics_data, individual_metrics_df, voting_type))
                    # =========================================================================================================================================================================================================
                    # ADDITION (END): Overall Ensemble Performance Insights

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
                    st.markdown(f"""
                    This **Accuracy Comparison: Individual vs Ensemble** bar chart visually compares the accuracy of each individual model against the ensemble model.
                    - The **X-axis** lists the different models (Logistic Regression, KNN, Random Forest, and the Ensemble).
                    - The **Y-axis** represents the 'Accuracy' score for each model.
                    - **Interpreting Bar Heights:** A taller bar indicates a higher accuracy. If the ensemble's bar is taller than the individual models, it suggests that combining the models has improved overall accuracy. If an individual model's bar is taller, it might indicate that the ensemble strategy (e.g., hard vs. soft voting) or the specific combination of models was not optimal for accuracy in this particular setup. This visualization helps determine if ensemble learning provided a performance boost.
                    """)
            
            if 'ensemble_pred' in st.session_state:
                st.info(f"Ensemble model ({st.session_state.ensemble_voting}) is ready!")
    
    elif page == "Feature Selection":
        st.header("🔍 Feature Selection Analysis")
        
        if 'target' in df_clean.columns:
            X = df_clean.drop('target', axis=1)
            y = df_clean['target']
            
            st.subheader("Chi-Squared Feature Selection Test")
            # ADDITION (START): Motivation for Feature Selection
            st.markdown("""
            ### Motivation for Feature Selection
            Feature selection is a crucial step in machine learning to:
            - **Reduce overfitting:** By removing irrelevant or redundant features, we can make models less complex and generalize better to new data.
            - **Improve model performance:** Focusing on the most informative features can lead to more accurate and robust models.
            - **Speed up training:** Fewer features mean less computational cost.
            - **Enhance interpretability:** Understanding which features are most important can provide valuable insights into the underlying data and problem.

            The **Chi-Squared test** is used here to assess the independence between each categorical feature and the target variable (heart disease presence). It helps identify features that are most statistically associated with the outcome.
            """)
            # ADDITION (END): Motivation for Feature Selection
            
            # Apply chi-squared test
            # Select top k features
            k = st.slider("Number of Top Features to Select", 1, len(X.columns), min(10, len(X.columns)))
            
            if st.button("Run Chi-Squared Test", type="primary"):
                X_chi = X.copy()
                
                # FIRST: Handle missing values before any other operations
                # Fill numeric columns with median
                numeric_cols = X_chi.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if X_chi[col].isnull().sum() > 0:
                        X_chi[col].fillna(X_chi[col].median(), inplace=True)
                
                # Fill non-numeric columns with mode
                non_numeric_cols = X_chi.select_dtypes(exclude=[np.number]).columns
                for col in non_numeric_cols:
                    if X_chi[col].isnull().sum() > 0:
                        mode_val = X_chi[col].mode()
                        fill_val = mode_val[0] if len(mode_val) > 0 else 0
                        X_chi[col].fillna(fill_val, inplace=True)
                
                # Ensure no NaNs remain before binning
                X_chi = X_chi.fillna(0)
                
                # Bin continuous variables
                for col in X_chi.columns:
                    if X_chi[col].dtype in ['float64', 'int64']:
                        n_unique = X_chi[col].nunique()
                        if n_unique > 10:  # Bin if more than 10 unique values
                            try:
                                X_chi[col] = pd.qcut(X_chi[col], q=5, duplicates='drop', labels=False)
                                # Fill any NaNs that qcut might have created
                                X_chi[col] = X_chi[col].fillna(0)
                            except ValueError:
                                # If qcut fails (e.g., all values are the same), use regular binning
                                X_chi[col] = pd.cut(X_chi[col], bins=5, labels=False, duplicates='drop')
                                X_chi[col] = X_chi[col].fillna(0)
                
                # Final check: ensure no NaNs or infinite values
                X_chi = X_chi.replace([np.inf, -np.inf], 0)
                X_chi = X_chi.fillna(0)
                
                # Ensure all values are non-negative integers
                # Convert to float first to handle any remaining edge cases, then to int
                X_chi = X_chi.astype(float)
                X_chi = X_chi.fillna(0)  # Final safety check
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
                # ADDITION (START): Explanation for Feature Importance Scores table
                st.markdown("""
                This table displays the results of the Chi-Squared feature selection test.
                - **Feature:** The name of the clinical attribute.
                - **Chi-Squared Score:** A higher score indicates a stronger statistical dependency between the feature and the target variable (heart disease presence). Features with higher scores are considered more important for prediction.
                - **P-Value:** The p-value indicates the probability of observing such a strong association by random chance. A very small p-value (typically < 0.05) suggests that the association is statistically significant, meaning the feature is likely a strong predictor.
                - **Selected:** A checkbox indicating whether the feature was selected as one of the top `k` features (based on the slider above).
                """)
                # ADDITION (END): Explanation for Feature Importance Scores table
                
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
                # ADDITION (START): Explanation for Chi-Squared Scores Visualization
                st.markdown("""
                This bar chart visually represents the Chi-Squared score for each feature.
                - The **X-axis** lists all the available clinical features.
                - The **Y-axis** shows the 'Chi-Squared Score'.
                - **Bar Height:** A taller bar signifies a higher Chi-Squared score, indicating that the feature has a stronger statistical relationship with the presence of heart disease. Features with higher bars are more discriminative.
                - **Color Coding:** Bars are colored differently to highlight which features were 'Selected' (green) as the top `k` most important features based on the slider, and which were not (lightblue). This helps quickly identify the most impactful features.
                """)
                # ADDITION (END): Explanation for Chi-Squared Scores Visualization
                
                st.subheader("Selected Features")
                st.write(f"Top {k} features most associated with heart disease:")
                for i, feature in enumerate(st.session_state.selected_features, 1):
                    score = st.session_state.feature_scores[
                        st.session_state.feature_scores['Feature'] == feature
                    ]['Chi-Squared Score'].values[0]
                    st.write(f"{i}. **{feature}** (Score: {score:.2f})")
        else:
            st.error("Target column not found in dataset!")
    
    elif page == "Conclusion":
        st.header("Conclusion")
        st.snow()

        st.markdown("""
        In this project, we explored the use of **classical machine learning models**
        to predict the presence of heart disease based on patient clinical data.

        We implemented and compared the following models:
        - **Logistic Regression**
        - **K-Nearest Neighbors (KNN)**
        - **Random Forest**
        - A **Voting Ensemble** combining all three models
        """)

        # ADDITION (START): Model-specific descriptions in Conclusion
        # =========================================================================================================================================================================================================
        st.subheader("Implemented Models Overview")
        st.markdown(f"**Logistic Regression:** {get_model_description('Logistic Regression')}")
        st.markdown(f"**K-Nearest Neighbors (KNN):** {get_model_description('K-Nearest Neighbors (KNN)')}")
        st.markdown(f"**Random Forest:** {get_model_description('Random Forest')}")
        st.markdown(f"**Voting Ensemble:** {get_model_description('Voting Ensemble')}")
        # =========================================================================================================================================================================================================
        # ADDITION (END): Model-specific descriptions

        st.subheader("Key Takeaways")
        st.markdown("""
        - Different models exhibit different strengths in predicting heart disease.
        - **Random Forest** often performs well due to its ability to capture nonlinear relationships.
        - **Ensemble learning** can improve robustness by combining multiple models.
        - Proper **data preprocessing and leakage prevention** are critical for reliable evaluation.
        """)

        st.subheader("Limitations")
        st.markdown("""
        - The dataset is relatively small and may not represent all populations.
        - Results depend on the chosen train/test split and model hyperparameters.
        - This project focuses on classical ML methods rather than deep learning.
        """)

        st.subheader("Future Work")
        st.markdown("""
        - Tune hyperparameters more extensively using cross-validation
        - Explore additional feature engineering techniques
        - Test advanced ensemble methods or neural networks
        - Evaluate model fairness and interpretability in medical settings
        """)

        st.markdown("---")
        st.markdown("**This project demonstrates how machine learning can assist in medical decision-making, while highlighting the importance of careful evaluation and responsible data handling.**")


else:
    st.error("Failed to load dataset. Please check your internet connection and try again.")

