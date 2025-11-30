# src/llm_utils.py
from transformers import pipeline
import pandas as pd
import numpy as np

def generate_comprehensive_insights(df, stats, analysis_results, target_variable=None):
    """
    Generate comprehensive technical insights using LLM
    """
    try:
        # Load text generation pipeline
        generator = pipeline("text2text-generation", model="google/flan-t5-small")
        
        # Prepare comprehensive prompt
        prompt = prepare_insight_prompt(df, stats, analysis_results, target_variable)
        
        result = generator(prompt, max_length=500, do_sample=True, temperature=0.7)
        return result[0]['generated_text']
    except Exception as e:
        return generate_fallback_insights(df, stats, analysis_results, target_variable)

def prepare_insight_prompt(df, stats, analysis_results, target_variable=None):
    """
    Prepare a comprehensive prompt for LLM analysis
    """
    prompt = f"""
    You are a senior data scientist providing technical insights. Analyze this dataset and provide 5 key insights:
    
    DATASET OVERVIEW:
    - Shape: {df.shape[0]} rows, {df.shape[1]} columns
    - Memory usage: {stats.get('dataset_info', {}).get('memory_usage', 0):.2f} MB
    - Missing values: {stats.get('missing_values', {}).get('total_missing', 0)} ({stats.get('missing_values', {}).get('missing_percentage', 0):.1f}%)
    """
    
    if target_variable:
        prompt += f"\nTARGET VARIABLE: {target_variable}\n"
        target_info = df[target_variable].describe() if target_variable in df.columns else {}
        if not target_info.empty:
            prompt += f"- Target statistics: mean={target_info.get('mean', 0):.2f}, std={target_info.get('std', 0):.2f}\n"
    
    prompt += f"""
    DATA TYPES:
    - Numeric columns: {stats.get('data_types', {}).get('numeric_cols', [])}
    - Categorical columns: {stats.get('data_types', {}).get('categorical_cols', [])}
    
    STATISTICAL FINDINGS:
    """
    
    # Add numeric statistics
    if 'numeric_stats' in stats:
        prompt += "\nNUMERIC STATISTICS:\n"
        for col, desc in stats['numeric_stats']['descriptive'].items():
            prompt += f"- {col}: mean={desc.get('mean', 0):.2f}, std={desc.get('std', 0):.2f}, skewness={stats['numeric_stats']['skewness'].get(col, 0):.2f}\n"
    
    # Add categorical statistics
    if 'categorical_stats' in stats:
        prompt += "\nCATEGORICAL STATISTICS:\n"
        for col, cat_stats in stats['categorical_stats'].items():
            prompt += f"- {col}: {cat_stats['unique_count']} unique values, most frequent: {list(cat_stats['most_frequent'].keys())[0] if cat_stats['most_frequent'] else 'N/A'}\n"
    
    # Add correlation insights
    if 'correlation_analysis' in analysis_results:
        corr_analysis = analysis_results['correlation_analysis']
        if corr_analysis['strong_correlations']:
            prompt += f"\nSTRONG CORRELATIONS FOUND: {len(corr_analysis['strong_correlations'])} pairs\n"
        if corr_analysis['moderate_correlations']:
            prompt += f"MODERATE CORRELATIONS FOUND: {len(corr_analysis['moderate_correlations'])} pairs\n"
    
    if target_variable:
        prompt += f"""
    
    TARGET-FOCUSED ANALYSIS:
    Focus on how features impact the target variable '{target_variable}' and provide insights on:
    1. Key drivers of the target variable
    2. Feature importance and relationships
    3. Predictive patterns and opportunities
    4. Business implications for the target variable
    5. Recommendations for improving target variable outcomes
    
    Format your response as a numbered list with each insight on a separate line.
    """
    else:
        prompt += """
    
    Provide 5 technical insights focusing on:
    1. Data quality and completeness
    2. Distribution patterns and outliers
    3. Relationships between variables
    4. Business implications
    5. Recommendations for further analysis
    
    Format your response as a numbered list with each insight on a separate line.
    """
    
    prompt += """
    
    Use professional data science terminology and be specific with numbers and patterns.
    """
    
    return prompt

def generate_fallback_insights(df, stats, analysis_results, target_variable=None):
    """
    Generate fallback insights when LLM is not available
    """
    insights = []
    
    # Data quality insights
    missing_pct = stats.get('missing_values', {}).get('missing_percentage', 0)
    if missing_pct > 10:
        insights.append(f"⚠️ Data Quality Alert: {missing_pct:.1f}% of values are missing, indicating potential data collection issues.")
    elif missing_pct > 5:
        insights.append(f"📊 Data Quality: {missing_pct:.1f}% missing values detected, consider imputation strategies.")
    else:
        insights.append(f"✅ Data Quality: Excellent data completeness with only {missing_pct:.1f}% missing values.")
    
    # Target variable specific insights
    if target_variable and target_variable in df.columns:
        target_type = "numeric" if df[target_variable].dtype in [np.number] else "categorical"
        insights.append(f"🎯 Target Variable Analysis: {target_variable} is {target_type} with {df[target_variable].nunique()} unique values.")
        
        if target_type == "numeric":
            target_mean = df[target_variable].mean()
            target_std = df[target_variable].std()
            insights.append(f"📊 Target Statistics: Mean = {target_mean:.2f}, Std = {target_std:.2f}")
        else:
            most_common = df[target_variable].mode().iloc[0] if not df[target_variable].mode().empty else "N/A"
            insights.append(f"📊 Target Distribution: Most common value is '{most_common}'")
    
    # Distribution insights
    numeric_cols = stats.get('data_types', {}).get('numeric_cols', [])
    if numeric_cols:
        insights.append(f"📈 Numeric Analysis: {len(numeric_cols)} numeric variables available for statistical analysis.")
        
        # Check for skewed distributions
        for col in numeric_cols[:3]:  # Check first 3 numeric columns
            skewness = stats.get('numeric_stats', {}).get('skewness', {}).get(col, 0)
            if abs(skewness) > 1:
                insights.append(f"📊 Distribution Alert: {col} shows {'right' if skewness > 0 else 'left'} skewness ({skewness:.2f}), indicating non-normal distribution.")
    
    # Categorical insights
    cat_cols = stats.get('data_types', {}).get('categorical_cols', [])
    if cat_cols:
        insights.append(f"📋 Categorical Analysis: {len(cat_cols)} categorical variables with varying cardinality levels.")
        
        # Check for high cardinality
        for col in cat_cols[:3]:
            unique_count = stats.get('categorical_stats', {}).get(col, {}).get('unique_count', 0)
            if unique_count > len(df) * 0.8:
                insights.append(f"🔍 High Cardinality: {col} has {unique_count} unique values ({unique_count/len(df)*100:.1f}% of rows), consider grouping or feature engineering.")
    
    # Correlation insights
    if 'correlation_analysis' in analysis_results:
        strong_corr = analysis_results['correlation_analysis'].get('strong_correlations', [])
        moderate_corr = analysis_results['correlation_analysis'].get('moderate_correlations', [])
        
        if strong_corr:
            insights.append(f"🔗 Strong Relationships: {len(strong_corr)} variable pairs show strong correlation (|r| > 0.7), indicating potential multicollinearity.")
        if moderate_corr:
            insights.append(f"📊 Moderate Relationships: {len(moderate_corr)} variable pairs show moderate correlation (0.3 < |r| < 0.7), suggesting interesting patterns to explore.")
    
    # Target-focused business insights
    if target_variable:
        insights.append(f"💼 Target-Focused Analysis: With {target_variable} as the target variable, focus on identifying key drivers and predictive patterns for better outcomes.")
    else:
        insights.append(f"💼 Business Impact: Dataset contains {df.shape[0]} records across {df.shape[1]} dimensions, suitable for comprehensive analysis and modeling.")
    
    # Format insights as a numbered list with each point on a separate line
    formatted_insights = []
    for i, insight in enumerate(insights, 1):
        formatted_insights.append(f"{i}. {insight}")
    
    return "\n\n".join(formatted_insights)

def generate_insights(stats):
    """
    Legacy function for backward compatibility
    """
    return generate_fallback_insights(pd.DataFrame(), stats, {})

def generate_comprehensive_summary(df, stats, analysis_results, target_variable=None):
    """
    Generate a comprehensive, easy-to-understand summary with target variable focus
    """
    summary_sections = []
    
    # 1. Executive Summary
    summary_sections.append("## 🎯 Executive Summary")
    summary_sections.append(f"Your dataset contains **{df.shape[0]:,} records** across **{df.shape[1]} variables**.")
    
    if target_variable:
        summary_sections.append(f"The analysis focuses on **{target_variable}** as the target variable, helping identify key patterns and relationships.")
    else:
        summary_sections.append("This is a comprehensive exploratory analysis to understand data patterns and relationships.")
    
    # 2. Data Quality Assessment
    summary_sections.append("\n## 📊 Data Quality Assessment")
    missing_pct = stats.get('missing_values', {}).get('missing_percentage', 0)
    duplicate_count = stats.get('dataset_info', {}).get('duplicate_rows', 0)
    
    if missing_pct < 5:
        summary_sections.append("✅ **Excellent Data Quality**: Less than 5% missing values")
    elif missing_pct < 15:
        summary_sections.append("⚠️ **Good Data Quality**: Some missing values present but manageable")
    else:
        summary_sections.append("❌ **Data Quality Concerns**: High percentage of missing values detected")
    
    summary_sections.append(f"• Missing Values: {missing_pct:.1f}% of all data points")
    summary_sections.append(f"• Duplicate Rows: {duplicate_count:,} records")
    summary_sections.append(f"• Memory Usage: {stats.get('dataset_info', {}).get('memory_usage', 0):.2f} MB")
    
    # 3. Variable Overview
    summary_sections.append("\n## 🔍 Variable Overview")
    numeric_cols = stats.get('data_types', {}).get('numeric_cols', [])
    categorical_cols = stats.get('data_types', {}).get('categorical_cols', [])
    
    summary_sections.append(f"• **Numeric Variables**: {len(numeric_cols)} columns for statistical analysis")
    summary_sections.append(f"• **Categorical Variables**: {len(categorical_cols)} columns for grouping and classification")
    
    if numeric_cols:
        summary_sections.append(f"  - Key numeric variables: {', '.join(numeric_cols[:3])}{'...' if len(numeric_cols) > 3 else ''}")
    if categorical_cols:
        summary_sections.append(f"  - Key categorical variables: {', '.join(categorical_cols[:3])}{'...' if len(categorical_cols) > 3 else ''}")
    
    # 4. Target Variable Analysis (if selected)
    if target_variable and target_variable in df.columns:
        summary_sections.append(f"\n## 🎯 Target Variable Analysis: {target_variable}")
        
        target_type = "numeric" if df[target_variable].dtype in [np.number] else "categorical"
        summary_sections.append(f"**Type**: {target_type.title()}")
        summary_sections.append(f"**Unique Values**: {df[target_variable].nunique():,}")
        
        if target_type == "numeric":
            target_mean = df[target_variable].mean()
            target_std = df[target_variable].std()
            target_min = df[target_variable].min()
            target_max = df[target_variable].max()
            summary_sections.append(f"**Range**: {target_min:.2f} to {target_max:.2f}")
            summary_sections.append(f"**Average**: {target_mean:.2f} (±{target_std:.2f})")
            
            # Distribution insights
            if abs(df[target_variable].skew()) > 1:
                skew_direction = "right" if df[target_variable].skew() > 0 else "left"
                summary_sections.append(f"**Distribution**: {skew_direction}-skewed (non-normal)")
            else:
                summary_sections.append("**Distribution**: Approximately normal")
        else:
            most_common = df[target_variable].mode().iloc[0] if not df[target_variable].mode().empty else "N/A"
            most_common_pct = (df[target_variable].value_counts().iloc[0] / len(df)) * 100
            summary_sections.append(f"**Most Common**: '{most_common}' ({most_common_pct:.1f}% of data)")
            
            if most_common_pct > 50:
                summary_sections.append("**⚠️ Class Imbalance**: One category dominates the data")
            else:
                summary_sections.append("**✅ Balanced Distribution**: Good variety across categories")
    
    # 5. Key Statistical Findings
    summary_sections.append("\n## 📈 Key Statistical Findings")
    
    # Numeric variable insights
    if numeric_cols:
        summary_sections.append("### Numeric Variables")
        for col in numeric_cols[:3]:  # Show top 3 numeric variables
            if col in stats.get('numeric_stats', {}).get('descriptive', {}):
                desc = stats['numeric_stats']['descriptive'][col]
                mean_val = desc.get('mean', 0)
                std_val = desc.get('std', 0)
                summary_sections.append(f"• **{col}**: Average = {mean_val:.2f}, Variability = {std_val:.2f}")
    
    # Categorical variable insights
    if categorical_cols:
        summary_sections.append("### Categorical Variables")
        for col in categorical_cols[:3]:  # Show top 3 categorical variables
            if col in stats.get('categorical_stats', {}):
                cat_stats = stats['categorical_stats'][col]
                unique_count = cat_stats.get('unique_count', 0)
                most_frequent = list(cat_stats.get('most_frequent', {}).keys())[0] if cat_stats.get('most_frequent') else 'N/A'
                summary_sections.append(f"• **{col}**: {unique_count} categories, most common: '{most_frequent}'")
    
    # 6. Relationship Analysis
    summary_sections.append("\n## 🔗 Relationship Analysis")
    
    if 'correlation_analysis' in analysis_results:
        corr_analysis = analysis_results['correlation_analysis']
        strong_corr = corr_analysis.get('strong_correlations', [])
        moderate_corr = corr_analysis.get('moderate_correlations', [])
        
        if strong_corr:
            summary_sections.append(f"**Strong Relationships Found**: {len(strong_corr)} variable pairs show strong correlation")
            for corr in strong_corr[:2]:  # Show top 2
                summary_sections.append(f"  - {corr['pair']}: r = {corr['correlation']:.3f}")
        
        if moderate_corr:
            summary_sections.append(f"**Moderate Relationships**: {len(moderate_corr)} variable pairs show moderate correlation")
            for corr in moderate_corr[:2]:  # Show top 2
                summary_sections.append(f"  - {corr['pair']}: r = {corr['correlation']:.3f}")
        
        if not strong_corr and not moderate_corr:
            summary_sections.append("**Independent Variables**: No strong correlations found - variables appear independent")
    
    # 7. Target-Focused Insights (if target variable selected)
    if target_variable and target_variable in df.columns:
        summary_sections.append(f"\n## 🎯 Target-Focused Insights for {target_variable}")
        
        # Find features with strongest relationship to target
        feature_relationships = []
        for col in df.columns:
            if col != target_variable:
                try:
                    if df[target_variable].dtype in [np.number] and df[col].dtype in [np.number]:
                        # Numeric correlation
                        corr = df[target_variable].corr(df[col])
                        if not pd.isna(corr):
                            feature_relationships.append((col, abs(corr), corr))
                    elif df[target_variable].dtype in [np.number] and df[col].dtype in ['object', 'category']:
                        # Categorical vs numeric target
                        group_means = df.groupby(col)[target_variable].mean()
                        if len(group_means) > 1:
                            range_effect = group_means.max() - group_means.min()
                            feature_relationships.append((col, range_effect, range_effect))
                except:
                    continue
        
        # Sort by relationship strength
        feature_relationships.sort(key=lambda x: x[1], reverse=True)
        
        if feature_relationships:
            summary_sections.append("**Top Features Affecting Target Variable:**")
            for i, (feature, strength, value) in enumerate(feature_relationships[:3]):
                if df[target_variable].dtype in [np.number] and df[feature].dtype in [np.number]:
                    summary_sections.append(f"{i+1}. **{feature}**: Correlation = {value:.3f}")
                else:
                    summary_sections.append(f"{i+1}. **{feature}**: Range effect = {value:.2f}")
    
    # 8. Recommendations
    summary_sections.append("\n## 💡 Recommendations")
    
    if missing_pct > 10:
        summary_sections.append("• **Data Collection**: Address missing data issues before modeling")
    
    if target_variable:
        summary_sections.append(f"• **Modeling Focus**: Use {target_variable} as target for predictive modeling")
        if feature_relationships:
            top_feature = feature_relationships[0][0]
            summary_sections.append(f"• **Key Predictor**: {top_feature} shows strongest relationship with target")
    
    if 'correlation_analysis' in analysis_results:
        strong_corr = analysis_results['correlation_analysis'].get('strong_correlations', [])
        if len(strong_corr) > 0:
            summary_sections.append("• **Multicollinearity**: Consider feature selection to address highly correlated variables")
    
    if numeric_cols and categorical_cols:
        summary_sections.append("• **Mixed Analysis**: Combine numeric and categorical approaches for comprehensive insights")
    
    summary_sections.append("• **Next Steps**: Consider advanced modeling techniques based on your specific goals")
    
    # 9. Quick Action Items
    summary_sections.append("\n## ⚡ Quick Action Items")
    summary_sections.append("1. **Review Data Quality**: Check the data quality assessment above")
    summary_sections.append("2. **Explore Relationships**: Use the correlation analysis to identify key patterns")
    if target_variable:
        summary_sections.append(f"3. **Focus on Target**: Analyze how other variables relate to {target_variable}")
    summary_sections.append("4. **Generate Visualizations**: Use the advanced visualization tabs for deeper insights")
    summary_sections.append("5. **Export Results**: Download the summary and statistics for further analysis")
    
    return "\n".join(summary_sections)

# Ensure the function is accessible
# at bottom of src/llm_utils.py (optional)
__all__ = [
    'generate_comprehensive_summary',
    'generate_comprehensive_insights',
    'generate_fallback_insights',
    'generate_insights',
    'prepare_insight_prompt'
]
