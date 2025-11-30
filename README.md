---
title: AI Data Storyteller
emoji: 📊
colorFrom: blue
colorTo: purple
sdk: streamlit
app_file: dashboard/app.py
pinned: false
---

# 🤖 AI-Powered Data Storyteller

A comprehensive data analysis platform that automatically performs Exploratory Data Analysis (EDA), generates advanced visualizations, and provides AI-powered technical insights for CSV datasets.

## 🚀 Features

### 📊 Comprehensive EDA Analysis
- **Univariate Analysis**: Distribution analysis, statistical summaries, outlier detection
- **Bivariate Analysis**: Correlation analysis, scatter plots, contingency tables
- **Multivariate Analysis**: Advanced correlation matrices, groupby operations, parallel coordinates
- **Statistical Tests**: Chi-square tests, ANOVA, Pearson/Spearman correlations

### 🧹 Advanced Data Cleaning
- **Missing Value Treatment**: Smart imputation strategies (mode for categorical, median for numeric)
- **Outlier Detection & Treatment**: IQR-based outlier capping
- **Data Type Conversion**: Automatic detection and conversion of data types
- **Duplicate Removal**: Automatic duplicate detection and removal

### 📈 Advanced Visualizations (3-5 Key EDA Graphs)
1. **Distribution Analysis**: Histograms, box plots, violin plots with statistical insights
2. **Correlation Analysis**: Interactive heatmaps, network plots, correlation strength analysis
3. **Bivariate Analysis**: Scatter plots with trend lines, contingency tables
4. **Multivariate Analysis**: Parallel coordinates, scatter matrices
5. **Outlier Analysis**: Box plots with outlier identification and treatment

### 🤖 AI-Powered Insights
- **LLM Integration**: Uses HuggingFace transformers for intelligent insights
- **Technical Analysis**: Statistical significance, distribution patterns, business implications
- **Fallback System**: Robust fallback insights when LLM is unavailable
- **Comprehensive Reporting**: Detailed technical insights with specific numbers and patterns

### 📄 Professional Reporting
- **Word Document Generation**: Professional reports with embedded visualizations
- **Interactive Dashboards**: Real-time exploration and analysis
- **Custom Visualizations**: Tailored plots based on data characteristics

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd sinchhh_ai_data_storyteller
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate virtual environment**
   
   **Windows:**
   ```bash
   .\venv\Scripts\Activate.ps1

   ```
   
   **macOS/Linux:**
   ```bash
   source venv/bin/activate.ps1
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### Running the Application

1. **Start the Streamlit dashboard**
   ```bash
   streamlit run dashboard/app.py
   ```

2. **Open your browser** and navigate to `http://localhost:8501`

3. **Upload a CSV file** or try the sample dataset

### Sample Dataset

The project includes a sample dataset (`data/vehicle_performance.csv`) with 398 records containing:
- Acceleration
- Name of the Vehicle
- Cylinders
- Weights etc.

### Analysis Workflow

1. **Upload Data**: Upload your CSV file through the web interface
2. **Data Validation**: Automatic validation with issue detection
3. **Data Cleaning**: Optional comprehensive data cleaning
4. **Statistical Analysis**: Automatic univariate, bivariate, and multivariate analysis
5. **Visualization**: Interactive plots and charts
6. **AI Insights**: LLM-powered technical insights and recommendations
7. **Report Generation**: Download comprehensive reports

## 📁 Project Structure

```
sinchhh_ai_data_storyteller/
├── dashboard/
│   └── app.py                 # Main Streamlit application
├── src/
│   ├── __init__.py
│   ├── eda_utils.py          # EDA analysis functions
│   ├── report_utils.py       # Visualization and reporting
│   └── llm_utils.py          # AI insights generation
├── data/
│   └── vehicle_performance.csv        # Sample dataset
├── report/                   # Generated reports directory
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## 🔧 Configuration

### Analysis Controls
- **Analysis Type**: Comprehensive EDA, Quick Analysis, Custom Analysis
- **Data Cleaning**: Toggle automatic data cleaning features
- **Visualization Options**: Control which visualizations to display
- **Outlier Handling**: Configure outlier detection and treatment

### Customization Options
- **Plot Types**: Choose from histogram, box plot, violin plot
- **Correlation Thresholds**: Adjust strong/moderate correlation thresholds
- **Groupby Analysis**: Configure categorical-numeric combinations
- **Report Format**: Choose between Word documents and JSON exports

## 📊 Analysis Capabilities

### Univariate Analysis
- **Numeric Variables**: Mean, median, std, skewness, kurtosis, outliers
- **Categorical Variables**: Frequency tables, unique counts, distribution analysis
- **Missing Value Analysis**: Comprehensive missing data assessment

### Bivariate Analysis
- **Numeric-Numeric**: Pearson/Spearman correlations with significance tests
- **Categorical-Categorical**: Chi-square tests with contingency tables
- **Mixed Types**: ANOVA tests and group statistics

### Multivariate Analysis
- **Correlation Matrices**: Advanced correlation analysis with strength classification
- **Groupby Operations**: Statistical aggregations across categorical groups
- **Parallel Coordinates**: Multi-dimensional data visualization

## 🤖 AI Insights Features

### Technical Analysis
- **Data Quality Assessment**: Missing value analysis and data completeness
- **Distribution Patterns**: Skewness, kurtosis, and normality assessment
- **Relationship Analysis**: Correlation strength and statistical significance
- **Business Implications**: Actionable insights for decision-making

### LLM Integration
- **Model**: Google FLAN-T5-small for efficient text generation
- **Prompt Engineering**: Structured prompts for consistent technical insights
- **Fallback System**: Robust fallback when LLM services are unavailable
- **Customizable Output**: Adjustable insight depth and technical detail

## 📈 Visualization Types

### Distribution Analysis
- **Histograms**: With marginal box plots and statistical overlays
- **Box Plots**: Outlier identification and quartile analysis
- **Violin Plots**: Distribution shape and density analysis

### Correlation Analysis
- **Heatmaps**: Interactive correlation matrices with color coding
- **Network Plots**: Strong correlation relationships visualization
- **Strength Analysis**: Horizontal bar charts for correlation comparison

### Bivariate Analysis
- **Scatter Plots**: With trend lines and marginal histograms
- **Contingency Tables**: Categorical relationship visualization
- **Box Plots**: Numeric-categorical relationship analysis

### Multivariate Analysis
- **Parallel Coordinates**: Multi-dimensional data exploration
- **Scatter Matrices**: Pairwise relationship analysis
- **Groupby Visualizations**: Statistical aggregation displays

## 🛡️ Error Handling

- **Data Validation**: Comprehensive input validation with detailed error messages
- **Graceful Degradation**: Fallback systems when advanced features fail
- **User Feedback**: Clear error messages and progress indicators
- **Robust Processing**: Handles various data types and edge cases

## 🔄 Dependencies

### Core Libraries
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **streamlit**: Web application framework
- **plotly**: Interactive visualizations

### Statistical Libraries
- **scipy**: Statistical functions and tests
- **scikit-learn**: Machine learning utilities
- **matplotlib/seaborn**: Additional plotting capabilities

### AI/ML Libraries
- **transformers**: HuggingFace transformers for LLM integration
- **torch**: PyTorch for deep learning models

### Reporting Libraries
- **python-docx**: Word document generation
- **kaleido**: Static image export for Plotly

## 🚀 Getting Started

1. **Quick Start**: Upload a CSV file and explore the automatic analysis
2. **Sample Data**: Try the included sample dataset to understand features
3. **Custom Analysis**: Use the sidebar controls to customize your analysis
4. **Report Generation**: Download comprehensive reports for sharing

## 📞 Support

For issues, questions, or contributions:
1. Check the error messages in the application
2. Review the console output for detailed error information
3. Ensure all dependencies are properly installed
4. Verify your CSV file format and data quality

## 🔮 Future Enhancements

- **Additional ML Models**: Integration with more advanced LLM models
- **Real-time Analysis**: Live data streaming and analysis
- **Custom Visualizations**: User-defined plot types and configurations
- **Database Integration**: Direct database connectivity for large datasets
- **API Endpoints**: RESTful API for programmatic access

---

**🤖 AI Data Storyteller** - Transforming raw data into actionable insights through advanced analytics and artificial intelligence.


