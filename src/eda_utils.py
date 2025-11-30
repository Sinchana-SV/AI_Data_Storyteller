# src/eda_utils.py
import pandas as pd
import numpy as np
import os
from scipy import stats
from scipy.stats import chi2_contingency, pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

def read_csv(uploaded_file):
    try:
        return pd.read_csv(uploaded_file)
    except Exception as e:
        raise ValueError(f"Error reading CSV: {e}")

min_rows = 10
min_cols = 2

def validate_dataset(df):
    """
    Validate dataset shape, dtypes, and missing values.
    Returns (is_valid: bool, issues: list, dtypes: dict)
    """
    issues = []
    rows, cols = df.shape
    if rows < min_rows:
        issues.append(f"Too few rows: {rows} rows (less than {min_rows}).")
    if cols < min_cols:
        issues.append(f"Too few columns: {cols} columns (need at least {min_cols}).")

    # all-null columns
    null_cols = df.columns[df.isna().all()].tolist()
    if null_cols:
        issues.append(f"Columns with all nulls: {null_cols}")

    # duplicates
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        issues.append(f"Found {dup_count} duplicate rows.")

    # very high cardinality for supposed categorical: (just warn)
    for c in df.columns:
        if df[c].nunique() >= 0.9 * rows and df[c].dtype == object:
            issues.append(f"Column '{c}' has very high cardinality (unique ≈ rows).")

    # basic datatypes info
    dtype_summary = df.dtypes.apply(lambda x: str(x)).to_dict()

    return (len(issues) == 0), issues, dtype_summary

def clean_dataset(df):
    """
    Comprehensive data cleaning function
    """
    df_cleaned = df.copy()
    
    # 1. Handle missing values
    missing_strategy = {}
    for col in df_cleaned.columns:
        if df_cleaned[col].dtype in ['object', 'category']:
            # For categorical: mode or 'Unknown'
            mode_val = df_cleaned[col].mode()
            if len(mode_val) > 0:
                df_cleaned[col].fillna(mode_val[0], inplace=True)
                missing_strategy[col] = f"Filled with mode: {mode_val[0]}"
            else:
                df_cleaned[col].fillna('Unknown', inplace=True)
                missing_strategy[col] = "Filled with 'Unknown'"
        else:
            # For numerical: median
            median_val = df_cleaned[col].median()
            df_cleaned[col].fillna(median_val, inplace=True)
            missing_strategy[col] = f"Filled with median: {median_val:.2f}"
    
    # 2. Remove duplicates
    initial_rows = len(df_cleaned)
    df_cleaned = df_cleaned.drop_duplicates()
    duplicates_removed = initial_rows - len(df_cleaned)
    
    # 3. Handle outliers using IQR method
    outlier_info = {}
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df_cleaned[(df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)]
        if len(outliers) > 0:
            # Cap outliers instead of removing
            df_cleaned[col] = np.where(df_cleaned[col] < lower_bound, lower_bound, df_cleaned[col])
            df_cleaned[col] = np.where(df_cleaned[col] > upper_bound, upper_bound, df_cleaned[col])
            outlier_info[col] = f"Capped {len(outliers)} outliers"
    
    # 4. Convert data types
    type_conversion = {}
    for col in df_cleaned.columns:
        if df_cleaned[col].dtype == 'object':
            # Try to convert to numeric
            try:
                pd.to_numeric(df_cleaned[col], errors='raise')
                df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
                type_conversion[col] = "Converted to numeric"
            except:
                # Try to convert to datetime
                try:
                    pd.to_datetime(df_cleaned[col], errors='raise')
                    df_cleaned[col] = pd.to_datetime(df_cleaned[col], errors='coerce')
                    type_conversion[col] = "Converted to datetime"
                except:
                    type_conversion[col] = "Kept as categorical"
    
    return df_cleaned, {
        'missing_strategy': missing_strategy,
        'duplicates_removed': duplicates_removed,
        'outlier_info': outlier_info,
        'type_conversion': type_conversion
    }

def get_comprehensive_stats(df):
    """
    Get comprehensive statistical analysis with improved variable type detection
    """
    # Use the same robust variable type detection as in the dashboard
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Check for object columns that might actually be numeric
    for col in df.columns:
        if col not in numeric_cols and col not in categorical_cols:
            try:
                numeric_test = pd.to_numeric(df[col], errors='coerce')
                if numeric_test.notna().sum() > 0.5 * len(df[col].dropna()):
                    numeric_cols.append(col)
                else:
                    categorical_cols.append(col)
            except:
                if col not in categorical_cols:
                    categorical_cols.append(col)
    
    # Get the actual data for analysis
    numeric = df[numeric_cols] if numeric_cols else pd.DataFrame()
    categorical = df[categorical_cols] if categorical_cols else pd.DataFrame()
    
    stats = {
        "dataset_info": {
            "n_rows": df.shape[0],
            "n_cols": df.shape[1],
            "memory_usage": df.memory_usage(deep=True).sum() / 1024**2,  # MB
            "duplicate_rows": df.duplicated().sum()
        },
        "missing_values": {
            "total_missing": df.isna().sum().sum(),
            "missing_per_col": df.isna().sum().to_dict(),
            "missing_percentage": (df.isna().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        },
        "data_types": {
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "datetime_cols": list(df.select_dtypes(include=['datetime64']).columns)
        }
    }
    
    # Numeric statistics
    if not numeric.empty:
        stats["numeric_stats"] = {
            "descriptive": numeric.describe().round(3).to_dict(),
            "skewness": numeric.skew().to_dict(),
            "kurtosis": numeric.kurtosis().to_dict(),
            "correlation_matrix": numeric.corr().round(3).to_dict()
        }
    
    # Categorical statistics
    if not categorical.empty:
        stats["categorical_stats"] = {}
        for col in categorical.columns:
            value_counts = df[col].value_counts()
            stats["categorical_stats"][col] = {
                "unique_count": df[col].nunique(),
                "most_frequent": value_counts.head(1).to_dict(),
                "top_5_values": value_counts.head(5).to_dict(),
                "frequency_distribution": (value_counts / len(df) * 100).round(2).to_dict()
            }
    
    return stats

def univariate_analysis(df, column):
    """
    Comprehensive univariate analysis with improved type detection
    """
    if column not in df.columns:
        return {"error": f"Column '{column}' not found"}
    
    col_data = df[column]
    analysis = {"column": column, "dtype": str(col_data.dtype)}
    
    # Enhanced numeric detection
    is_numeric = False
    if pd.api.types.is_numeric_dtype(col_data):
        is_numeric = True
    elif col_data.dtype in ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']:
        is_numeric = True
    elif col_data.dtype == 'object':
        try:
            numeric_converted = pd.to_numeric(col_data, errors='coerce')
            non_null_original = col_data.dropna()
            non_null_converted = numeric_converted.dropna()
            
            if len(non_null_converted) > 0.5 * len(non_null_original):
                col_data = numeric_converted
                is_numeric = True
        except:
            pass
    
    if is_numeric:
        # Numeric analysis
        clean_data = col_data.dropna()
        if len(clean_data) == 0:
            return {"type": "numeric", "error": "No valid numeric data"}
        
        try:
            Q1 = clean_data.quantile(0.25)
            Q3 = clean_data.quantile(0.75)
            IQR = Q3 - Q1
            outliers_count = len(clean_data[
                (clean_data < Q1 - 1.5 * IQR) | (clean_data > Q3 + 1.5 * IQR)
            ])
            
            analysis.update({
                "type": "numeric",
                "count": int(col_data.count()),
                "mean": float(clean_data.mean()),
                "median": float(clean_data.median()),
                "std": float(clean_data.std()),
                "min": float(clean_data.min()),
                "max": float(clean_data.max()),
                "q25": float(Q1),
                "q75": float(Q3),
                "skewness": float(clean_data.skew()),
                "kurtosis": float(clean_data.kurtosis()),
                "missing_count": int(col_data.isna().sum()),
                "missing_percentage": float((col_data.isna().sum() / len(df)) * 100),
                "outliers": int(outliers_count),
                "unique_count": int(clean_data.nunique())
            })
        except Exception as e:
            analysis.update({"type": "numeric", "error": f"Failed to compute statistics: {e}"})
    else:
        # Categorical analysis
        clean_data = col_data.dropna()
        if len(clean_data) == 0:
            return {"type": "categorical", "error": "No valid categorical data"}
            
        try:
            value_counts = clean_data.value_counts()
            analysis.update({
                "type": "categorical",
                "unique_count": int(clean_data.nunique()),
                "most_frequent": value_counts.head(1).to_dict(),
                "frequency_table": value_counts.head(10).to_dict(),
                "missing_count": int(col_data.isna().sum()),
                "missing_percentage": float((col_data.isna().sum() / len(df)) * 100),
                "total_count": int(col_data.count())
            })
        except Exception as e:
            analysis.update({"type": "categorical", "error": f"Failed to compute statistics: {e}"})
    
    return analysis

def bivariate_analysis(df, col1, col2):
    """
    Comprehensive bivariate analysis between two columns
    """
    analysis = {"column1": col1, "column2": col2}
    
    # Create a copy of the dataframe to avoid modifying the original
    df_clean = df.copy()
    
    # Helper function to convert object columns to numeric if possible
    def convert_to_numeric_if_possible(series):
        """Convert object series to numeric, handling common missing value indicators"""
        if series.dtype in [np.number]:
            return series
        
        # Try to convert to numeric, replacing common missing value indicators
        series_clean = series.replace(['?', 'N/A', 'n/a', 'NA', 'na', '', ' '], np.nan)
        numeric_series = pd.to_numeric(series_clean, errors='coerce')
        
        # If more than 50% of values are successfully converted, use numeric
        if numeric_series.notna().sum() / len(series) > 0.5:
            return numeric_series
        else:
            return series
    
    # Convert columns to appropriate types
    col1_converted = convert_to_numeric_if_possible(df_clean[col1])
    col2_converted = convert_to_numeric_if_possible(df_clean[col2])
    
    # Update the dataframe with converted columns
    df_clean[col1] = col1_converted
    df_clean[col2] = col2_converted
    
    # Check if both are numeric
    if col1_converted.dtype in [np.number] and col2_converted.dtype in [np.number]:
        # Numeric-Numeric correlation
        corr_pearson, p_pearson = pearsonr(col1_converted.dropna(), col2_converted.dropna())
        corr_spearman, p_spearman = spearmanr(col1_converted.dropna(), col2_converted.dropna())
        
        analysis.update({
            "type": "numeric_numeric",
            "pearson_correlation": corr_pearson,
            "pearson_p_value": p_pearson,
            "spearman_correlation": corr_spearman,
            "spearman_p_value": p_spearman,
            "correlation_strength": "strong" if abs(corr_pearson) > 0.7 else "moderate" if abs(corr_pearson) > 0.3 else "weak"
        })
    
    # Check if both are categorical
    elif col1_converted.dtype in ['object', 'category'] and col2_converted.dtype in ['object', 'category']:
        # Categorical-Categorical chi-square test
        contingency_table = pd.crosstab(col1_converted, col2_converted)
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        analysis.update({
            "type": "categorical_categorical",
            "chi2_statistic": chi2,
            "p_value": p_value,
            "degrees_of_freedom": dof,
            "contingency_table": contingency_table.to_dict(),
            "association_strength": "strong" if p_value < 0.001 else "moderate" if p_value < 0.05 else "weak"
        })
    
    # Mixed types
    else:
        # Numeric-Categorical analysis
        if col1_converted.dtype in [np.number]:
            numeric_col, categorical_col = col1, col2
        else:
            numeric_col, categorical_col = col2, col1
        
        # Ensure numeric column is actually numeric for aggregation
        numeric_data = df_clean[numeric_col]
        if numeric_data.dtype not in [np.number]:
            numeric_data = pd.to_numeric(numeric_data, errors='coerce')
        
        # Filter out NaN values for groupby operations
        valid_data = df_clean[[categorical_col, numeric_col]].dropna()
        
        if len(valid_data) == 0:
            analysis.update({
                "type": "numeric_categorical",
                "error": "No valid data for analysis after cleaning"
            })
            return analysis
        
        group_stats = valid_data.groupby(categorical_col)[numeric_col].agg(['mean', 'median', 'std', 'count']).round(3)
        
        # Prepare data for ANOVA test
        groups = [group[numeric_col].values for name, group in valid_data.groupby(categorical_col)]
        
        try:
            anova_result = stats.f_oneway(*groups)
            anova_f_statistic = anova_result.statistic
            anova_p_value = anova_result.pvalue
        except Exception as e:
            anova_f_statistic = None
            anova_p_value = None
        
        analysis.update({
            "type": "numeric_categorical",
            "group_statistics": group_stats.to_dict(),
            "anova_f_statistic": anova_f_statistic,
            "anova_p_value": anova_p_value
        })
    
    return analysis

def multivariate_analysis(df):
    """
    Comprehensive multivariate analysis
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    analysis = {}
    
    # Correlation analysis for numeric columns
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        analysis["correlation_analysis"] = {
            "correlation_matrix": corr_matrix.round(3).to_dict(),
            "strong_correlations": [],
            "moderate_correlations": []
        }
        
        # Find strong and moderate correlations
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    analysis["correlation_analysis"]["strong_correlations"].append({
                        "pair": f"{corr_matrix.columns[i]} - {corr_matrix.columns[j]}",
                        "correlation": corr_val
                    })
                elif abs(corr_val) > 0.3:
                    analysis["correlation_analysis"]["moderate_correlations"].append({
                        "pair": f"{corr_matrix.columns[i]} - {corr_matrix.columns[j]}",
                        "correlation": corr_val
                    })
    
    # Groupby analysis for categorical and numeric combinations
    if len(categorical_cols) > 0 and len(numeric_cols) > 0:
        analysis["groupby_analysis"] = {}
        for cat_col in categorical_cols[:3]:  # Limit to first 3 categorical columns
            for num_col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                group_stats = df.groupby(cat_col)[num_col].agg(['count', 'mean', 'median', 'std', 'min', 'max']).round(3)
                analysis["groupby_analysis"][f"{cat_col}_vs_{num_col}"] = group_stats.to_dict()
    
    return analysis

def get_basic_stats(df):
    """
    Legacy function for backward compatibility
    """
    return get_comprehensive_stats(df)

def detect_target_variable(df):
    """
    Automatically detect potential target variables in the dataset
    """
    target_candidates = []
    
    # Check for common target variable names
    common_target_names = ['target', 'label', 'y', 'class', 'category', 'outcome', 'result', 'prediction']
    
    for col in df.columns:
        col_lower = col.lower()
        if any(target_name in col_lower for target_name in common_target_names):
            target_candidates.append({
                'column': col,
                'reason': 'Common target variable naming pattern',
                'confidence': 'high'
            })
    
    # Check for binary categorical variables (potential classification targets)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        unique_count = df[col].nunique()
        if 2 <= unique_count <= 10:  # Binary to small categorical
            target_candidates.append({
                'column': col,
                'reason': f'Categorical variable with {unique_count} unique values (potential classification target)',
                'confidence': 'medium'
            })
    
    # Check for numeric variables that could be regression targets
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        # Check if values are in a reasonable range for a target
        if df[col].min() >= 0 and df[col].max() <= 1000:  # Common target ranges
            target_candidates.append({
                'column': col,
                'reason': f'Numeric variable with range {df[col].min():.2f} to {df[col].max():.2f} (potential regression target)',
                'confidence': 'low'
            })
    
    return target_candidates

def generate_plot_insights(df, plot_type, column1, column2=None, analysis_result=None):
    """
    Generate specific insights for each plot type
    """
    insights = []
    
    if plot_type == "distribution":
        if df[column1].dtype in [np.number]:
            # Numeric distribution insights
            mean_val = df[column1].mean()
            std_val = df[column1].std()
            skewness = df[column1].skew()
            kurtosis = df[column1].kurtosis()
            
            insights.append(f"📊 **Distribution Analysis for {column1}:**")
            insights.append(f"• Mean: {mean_val:.2f}, Standard Deviation: {std_val:.2f}")
            
            if abs(skewness) > 1:
                skew_direction = "right" if skewness > 0 else "left"
                insights.append(f"• Distribution is {skew_direction}-skewed (skewness: {skewness:.2f})")
            else:
                insights.append(f"• Distribution is approximately normal (skewness: {skewness:.2f})")
            
            if abs(kurtosis) > 3:
                kurtosis_type = "heavy-tailed" if kurtosis > 0 else "light-tailed"
                insights.append(f"• Distribution has {kurtosis_type} tails (kurtosis: {kurtosis:.2f})")
            
            # Outlier analysis
            Q1 = df[column1].quantile(0.25)
            Q3 = df[column1].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[column1] < Q1 - 1.5 * IQR) | (df[column1] > Q3 + 1.5 * IQR)]
            if len(outliers) > 0:
                insights.append(f"• {len(outliers)} outliers detected ({len(outliers)/len(df)*100:.1f}% of data)")
        
        else:
            # Categorical distribution insights
            value_counts = df[column1].value_counts()
            most_common = value_counts.iloc[0]
            most_common_pct = (most_common / len(df)) * 100
            
            insights.append(f"📊 **Distribution Analysis for {column1}:**")
            insights.append(f"• {df[column1].nunique()} unique categories")
            insights.append(f"• Most common: '{value_counts.index[0]}' ({most_common_pct:.1f}% of data)")
            
            if most_common_pct > 50:
                insights.append(f"• High concentration in one category - potential class imbalance")
            elif most_common_pct < 20:
                insights.append(f"• Well-distributed categories - good for analysis")
    
    elif plot_type == "correlation":
        if analysis_result and 'correlation_analysis' in analysis_result:
            corr_analysis = analysis_result['correlation_analysis']
            
            insights.append(f"🔗 **Correlation Analysis Insights:**")
            
            if corr_analysis['strong_correlations']:
                insights.append(f"• {len(corr_analysis['strong_correlations'])} strong correlations found (|r| > 0.7)")
                for corr in corr_analysis['strong_correlations'][:3]:  # Show top 3
                    insights.append(f"  - {corr['pair']}: r = {corr['correlation']:.3f}")
                insights.append("• Strong correlations may indicate multicollinearity")
            
            if corr_analysis['moderate_correlations']:
                insights.append(f"• {len(corr_analysis['moderate_correlations'])} moderate correlations found (0.3 < |r| < 0.7)")
                for corr in corr_analysis['moderate_correlations'][:3]:  # Show top 3
                    insights.append(f"  - {corr['pair']}: r = {corr['correlation']:.3f}")
                insights.append("• Moderate correlations suggest interesting relationships to explore")
            
            if not corr_analysis['strong_correlations'] and not corr_analysis['moderate_correlations']:
                insights.append("• No significant correlations found - variables appear independent")
    
    elif plot_type == "bivariate":
        if analysis_result:
            analysis_type = analysis_result.get('type', '')
            
            if analysis_type == "numeric_numeric":
                pearson_r = analysis_result.get('pearson_correlation', 0)
                p_value = analysis_result.get('pearson_p_value', 1)
                strength = analysis_result.get('correlation_strength', 'weak')
                
                insights.append(f"🔗 **Bivariate Analysis: {column1} vs {column2}**")
                insights.append(f"• Pearson correlation: r = {pearson_r:.3f}")
                insights.append(f"• Correlation strength: {strength}")
                insights.append(f"• Statistical significance: p = {p_value:.3f}")
                
                if p_value < 0.05:
                    insights.append("• Statistically significant relationship (p < 0.05)")
                else:
                    insights.append("• No statistically significant relationship (p ≥ 0.05)")
                
                if abs(pearson_r) > 0.7:
                    insights.append("• Strong linear relationship - good for predictive modeling")
                elif abs(pearson_r) > 0.3:
                    insights.append("• Moderate linear relationship - worth investigating further")
                else:
                    insights.append("• Weak linear relationship - may need non-linear analysis")
            
            elif analysis_type == "categorical_categorical":
                chi2 = analysis_result.get('chi2_statistic', 0)
                p_value = analysis_result.get('p_value', 1)
                association = analysis_result.get('association_strength', 'weak')
                
                insights.append(f"🔗 **Bivariate Analysis: {column1} vs {column2}**")
                insights.append(f"• Chi-square statistic: {chi2:.3f}")
                insights.append(f"• Association strength: {association}")
                insights.append(f"• Statistical significance: p = {p_value:.3f}")
                
                if p_value < 0.05:
                    insights.append("• Statistically significant association (p < 0.05)")
                else:
                    insights.append("• No statistically significant association (p ≥ 0.05)")
            
            elif analysis_type == "numeric_categorical":
                anova_f = analysis_result.get('anova_f_statistic', 0)
                p_value = analysis_result.get('anova_p_value', 1)
                
                insights.append(f"🔗 **Bivariate Analysis: {column1} vs {column2}**")
                insights.append(f"• ANOVA F-statistic: {anova_f:.3f}")
                insights.append(f"• Statistical significance: p = {p_value:.3f}")
                
                if p_value < 0.05:
                    insights.append("• Statistically significant difference between groups (p < 0.05)")
                else:
                    insights.append("• No statistically significant difference between groups (p ≥ 0.05)")
    
    return "\n".join(insights)

def load_sample_data(filename="vehicle_performance.csv"):
    # Get the directory of the current file (src/eda_utils.py)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to project root, then into data folder
    project_root = os.path.dirname(current_dir)
    data_path = os.path.join(project_root, "data", filename)
    try:
        df = pd.read_csv(data_path)
        print(f"✅ Loaded dataset with shape {df.shape}")
        return df
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print(f"Tried to load from: {data_path}")
    return None



