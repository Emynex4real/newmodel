# Course Recommendation System

A machine learning-powered web application that recommends academic courses based on user interests using TF-IDF vectorization and multiple classification algorithms.

## Features

- **Interactive Web Interface**: Built with Streamlit for easy use
- **Multiple ML Models**: Random Forest, Gradient Boosting, and Logistic Regression
- **Real-time Predictions**: Get instant course recommendations
- **Confidence Scores**: See how confident the model is in its predictions
- **Model Comparison**: Compare performance across different algorithms
- **Export Functionality**: Download trained models for later use

## Installation

### Option 1: Using pip

\`\`\`bash
# Clone or download the project files
# Navigate to the project directory

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python setup.py

# Run the application
streamlit run app.py
\`\`\`

### Option 2: Using conda

\`\`\`bash
# Create a new conda environment
conda create -n course-rec python=3.9
conda activate course-rec

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python setup.py

# Run the application
streamlit run app.py
\`\`\`

## Usage

1. **Start the Application**:
   \`\`\`bash
   streamlit run app.py
   \`\`\`

2. **Access the Web Interface**:
   - Open your browser and go to `http://localhost:8501`

3. **Get Recommendations**:
   - Enter your interests in the text area
   - Choose a machine learning model from the sidebar
   - Click "Get Course Recommendations"
   - View your personalized course suggestions with confidence scores

4. **Explore Features**:
   - Try different example interests
   - Compare model performances
   - View available courses in the dataset
   - Download trained models for offline use

## Dataset

The system uses a curated dataset of courses and their associated interest keywords:
- **50+ academic courses** across various disciplines
- **Interest-based matching** using natural language processing
- **Comprehensive coverage** from STEM to humanities

## Models

### Random Forest
- Ensemble method using multiple decision trees
- Good for handling complex patterns
- Provides feature importance insights

### Gradient Boosting
- Sequential learning approach
- Excellent for capturing non-linear relationships
- High accuracy on structured data

### Logistic Regression
- Linear classification method
- Fast training and prediction
- Interpretable results

## Technical Details

- **Text Processing**: TF-IDF vectorization with stop word removal
- **Feature Engineering**: Maximum 1000 features for optimal performance
- **Model Training**: 80/20 train-test split with cross-validation
- **Evaluation**: Accuracy scores and classification reports

## Deployment Options

### Local Development
\`\`\`bash
streamlit run app.py
\`\`\`

### Streamlit Cloud
1. Push code to GitHub repository
2. Connect to Streamlit Cloud
3. Deploy directly from the repository

### Docker Deployment
\`\`\`dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt
RUN python setup.py

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
\`\`\`

### Heroku Deployment
1. Add `Procfile`:
   \`\`\`
   web: sh setup.sh && streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   \`\`\`

2. Add `setup.sh`:
   \`\`\`bash
   mkdir -p ~/.streamlit/
   echo "[server]" > ~/.streamlit/config.toml
   echo "port = $PORT" >> ~/.streamlit/config.toml
   echo "enableCORS = false" >> ~/.streamlit/config.toml
   echo "headless = true" >> ~/.streamlit/config.toml
   \`\`\`

## File Structure

\`\`\`
course-recommendation-system/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── setup.py                       # NLTK data setup
├── course_interests_refined.csv    # Training dataset
├── scripts/
│   └── train_model.py             # Original model training script
├── README.md                      # This file
└── pyproject.toml                 # Project configuration
\`\`\`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Support

For issues or questions:
1. Check the README for common solutions
2. Review the code comments for technical details
3. Open an issue on the project repository
