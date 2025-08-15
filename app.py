import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Course Recommendation System",
    page_icon="🎓",
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
    .prediction-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .confidence-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the course data"""
    try:
        # Try to load the clean data first
        if os.path.exists("course_interests_clean.csv"):
            df = pd.read_csv("course_interests_clean.csv")
        else:
            # If clean data doesn't exist, load and preprocess the raw data
            df = pd.read_csv("course_interests_refined.csv")
            df['course'] = df['course'].str.strip()
            df['interests'] = df['interests'].str.lower().str.strip()
            df['interests'] = df['interests'].str.replace(r'\s+', ' ', regex=True)
            df.to_csv("course_interests_clean.csv", index=False)
        
        return df
    except FileNotFoundError:
        st.error("Dataset file not found. Please make sure 'course_interests_refined.csv' is in your app directory.")
        st.info("You can upload the dataset file to your Streamlit app or check the file path.")
        return None

@st.cache_resource
def train_models(df):
    """Train the machine learning models"""
    X = df["interests"]
    y = df["course"]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Vectorize text
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train models
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
    }
    
    trained_models = {}
    model_scores = {}
    
    for name, model in models.items():
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        
        trained_models[name] = model
        model_scores[name] = accuracy
    
    return trained_models, vectorizer, model_scores, X_test, y_test

def predict_course(interests, model, vectorizer):
    """Predict course based on interests"""
    interests_processed = interests.lower().strip()
    interests_tfidf = vectorizer.transform([interests_processed])
    
    # Get prediction and probabilities
    prediction = model.predict(interests_tfidf)[0]
    
    # Get prediction probabilities if available
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(interests_tfidf)[0]
        classes = model.classes_
        
        # Get top 3 predictions
        top_indices = probabilities.argsort()[-3:][::-1]
        top_predictions = [(classes[i], probabilities[i]) for i in top_indices]
    else:
        top_predictions = [(prediction, 1.0)]
    
    return prediction, top_predictions

def main():
    # Header
    st.markdown('<h1 class="main-header">🎓 Course Recommendation System</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    This system uses machine learning to recommend academic courses based on your interests. 
    Simply describe your interests and get personalized course recommendations!
    """)
    
    # Load data
    df = load_data()
    if df is None:
        st.stop()
    
    # Train models
    with st.spinner("Training models... This may take a moment."):
        trained_models, vectorizer, model_scores, X_test, y_test = train_models(df)
    
    # Sidebar for model selection and information
    st.sidebar.header("Model Configuration")
    
    # Model selection
    selected_model_name = st.sidebar.selectbox(
        "Choose Model:",
        list(trained_models.keys()),
        help="Select which machine learning model to use for predictions"
    )
    
    selected_model = trained_models[selected_model_name]
    
    # Display model performance
    st.sidebar.subheader("Model Performance")
    for name, score in model_scores.items():
        if name == selected_model_name:
            st.sidebar.metric(f"**{name}** (Selected)", f"{score:.2%}")
        else:
            st.sidebar.metric(name, f"{score:.2%}")
    
    # Dataset information
    st.sidebar.subheader("Dataset Info")
    st.sidebar.info(f"""
    - **Total Courses**: {df['course'].nunique()}
    - **Total Records**: {len(df)}
    - **Unique Interests**: {df['interests'].nunique()}
    """)
    
    # Main prediction interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Enter Your Interests")
        
        # Text input for interests
        user_interests = st.text_area(
            "Describe your interests, hobbies, or subjects you're passionate about:",
            placeholder="e.g., programming, artificial intelligence, web development, data analysis, machine learning",
            height=100,
            help="Be specific about your interests for better recommendations"
        )
        
        # Example interests
        st.subheader("Example Interests")
        example_interests = [
            "programming, software development, algorithms",
            "biology, genetics, medical research, healthcare",
            "business, entrepreneurship, marketing, finance",
            "art, design, creativity, visual communication",
            "mathematics, statistics, data analysis, research"
        ]
        
        selected_example = st.selectbox(
            "Or try one of these examples:",
            [""] + example_interests,
            help="Select an example to see how the system works"
        )
        
        if selected_example:
            user_interests = selected_example
        
        # Prediction button
        if st.button("Get Course Recommendations", type="primary"):
            if user_interests.strip():
                with st.spinner("Analyzing your interests..."):
                    prediction, top_predictions = predict_course(
                        user_interests, selected_model, vectorizer
                    )
                
                # Display results
                st.subheader("🎯 Recommended Courses")
                
                # Main prediction
                st.markdown(f"""
                <div class="prediction-box">
                    <h3>Top Recommendation: {prediction}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Top 3 predictions with confidence
                st.subheader("📊 All Recommendations (with confidence)")
                
                for i, (course, confidence) in enumerate(top_predictions, 1):
                    confidence_color = "green" if confidence > 0.7 else "orange" if confidence > 0.4 else "red"
                    st.markdown(f"""
                    <div class="confidence-box">
                        <strong>{i}. {course}</strong><br>
                        <span style="color: {confidence_color};">Confidence: {confidence:.1%}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("Please enter your interests to get recommendations.")
    
    with col2:
        st.subheader("Available Courses")
        
        # Display all available courses
        courses_df = df['course'].value_counts().reset_index()
        courses_df.columns = ['Course', 'Count']
        
        st.dataframe(
            courses_df,
            use_container_width=True,
            height=400
        )
        
        # Download model button
        st.subheader("Export Model")
        if st.button("Download Trained Model"):
            # Save model and vectorizer
            model_data = {
                'model': selected_model,
                'vectorizer': vectorizer,
                'model_name': selected_model_name
            }
            
            import pickle
            model_bytes = pickle.dumps(model_data)
            
            st.download_button(
                label="Download Model File",
                data=model_bytes,
                file_name=f"course_recommendation_model_{selected_model_name.lower().replace(' ', '_')}.pkl",
                mime="application/octet-stream"
            )
    
    # Additional information
    st.markdown("---")
    st.subheader("How It Works")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **1. Text Processing**
        - Your interests are converted to lowercase
        - Extra spaces are removed
        - Text is tokenized for analysis
        """)
    
    with col2:
        st.markdown("""
        **2. Feature Extraction**
        - TF-IDF vectorization converts text to numbers
        - Important keywords are identified
        - Stop words are filtered out
        """)
    
    with col3:
        st.markdown("""
        **3. Machine Learning**
        - Multiple models analyze patterns
        - Best matching courses are identified
        - Confidence scores are calculated
        """)

if __name__ == "__main__":
    main()
