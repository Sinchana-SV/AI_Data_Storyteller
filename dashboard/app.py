# dashboard/app.py
import streamlit as st
import pandas as pd
import numpy as np
import io
import sys
import os
import plotly.express as px
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import eda_utils, report_utils, llm_utils
from src.report_utils import build_dataset_stats, simple_template_insight

# Page configuration
st.set_page_config(
    layout="wide", 
    page_title="AI Data Storyteller", 
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'current_dataframe' not in st.session_state:
    st.session_state.current_dataframe = None
if 'data_source' not in st.session_state:
    st.session_state.data_source = None

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .insight-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
        white-space: pre-line;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🤖 AI-Powered Data Storyteller</h1>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <p style="font-size: 1.2rem; color: #666;">
        Upload a CSV file and get comprehensive EDA, advanced visualizations, 
        statistical analysis, and AI-powered insights with downloadable reports.
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar for controls
with st.sidebar:
    st.header("🔧 Analysis Controls")
    
    # Show current dataset info
    if st.session_state.data_loaded and st.session_state.current_dataframe is not None:
        st.success(f"📊 Dataset Loaded: {st.session_state.data_source}")
        st.info(f"Shape: {st.session_state.current_dataframe.shape[0]} rows × {st.session_state.current_dataframe.shape[1]} columns")
        
        # Reset button
        if st.button("🔄 Load New Dataset", type="secondary"):
            st.session_state.data_loaded = False
            st.session_state.current_dataframe = None
            st.session_state.data_source = None
            st.rerun()
    
    # Analysis type selection
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["Comprehensive EDA", "Quick Analysis", "Custom Analysis"]
    )
    
    # Data cleaning options
    st.subheader("Data Cleaning Options")
    clean_data = st.checkbox("Apply Data Cleaning", value=True)
    handle_outliers = st.checkbox("Handle Outliers", value=True)
    fill_missing = st.checkbox("Fill Missing Values", value=True)
    
    # Visualization options
    st.subheader("Visualization Options")
    show_distributions = st.checkbox("Show Distribution Plots", value=True)
    show_correlations = st.checkbox("Show Correlation Analysis", value=True)
    show_bivariate = st.checkbox("Show Bivariate Analysis", value=True)
    show_multivariate = st.checkbox("Show Multivariate Analysis", value=True)

def run_comprehensive_analysis(df, data_source="uploaded_file", clean_data=True, handle_outliers=True, fill_missing=True, show_distributions=True, show_correlations=True, show_bivariate=True, show_multivariate=True):
    """
    Run comprehensive analysis on a dataframe
    """
    try:
        # Reset summary state when new analysis starts
        if 'summary_generated' in st.session_state:
            del st.session_state['summary_generated']
        if 'comprehensive_summary' in st.session_state:
            del st.session_state['comprehensive_summary']
            
        # Data validation
        is_valid, issues, dtypes = eda_utils.validate_dataset(df)
        
        if not is_valid:
            st.warning("⚠️ Dataset validation flagged issues:")
            for issue in issues:
                st.write(f"- {issue}")
            st.info("You can still continue, but consider fixing these issues first.")

        # Data cleaning
        if clean_data:
            with st.spinner("Cleaning data..."):
                df_cleaned, cleaning_info = eda_utils.clean_dataset(df)
                df = df_cleaned
                
                # Show cleaning results
                st.success("✅ Data cleaning completed!")
                with st.expander("View Cleaning Details"):
                    st.json(cleaning_info)
        else:
            df_cleaned = df
            cleaning_info = {}
    
        # Target Variable Selection
        st.header("🎯 Target Variable Selection")
        st.info("Select a target variable to enable feature analysis and comparison insights.")
        
        # Let user select target variable from all columns
        target_options = ["None"] + list(df.columns)
        selected_target = st.selectbox("Select Target Variable for Analysis", target_options, key=f"target_selection_{data_source}")
        
        if selected_target == "None":
            selected_target = None
            st.warning("⚠️ No target variable selected. Some advanced analyses will be limited.")
        else:
            st.success(f"✅ Target variable selected: **{selected_target}**")
            
            # Show target variable info
            target_info = eda_utils.univariate_analysis(df, selected_target)
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Data Type", target_info.get('type', 'Unknown'))
                st.metric("Unique Values", target_info.get('unique_count', 'N/A'))
            
            with col2:
                if target_info.get('type') == 'numeric':
                    st.metric("Mean", f"{target_info.get('mean', 0):.2f}")
                    st.metric("Std Dev", f"{target_info.get('std', 0):.2f}")
                else:
                    st.metric("Most Frequent", list(target_info.get('most_frequent', {}).keys())[0] if target_info.get('most_frequent') else 'N/A')
        
        
        # Dataset Overview
        st.header("📊 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Rows", f"{df.shape[0]:,}")
        with col2:
            st.metric("Columns", f"{df.shape[1]:,}")
        with col3:
            memory_usage = df.memory_usage(deep=True).sum() / 1024**2
            st.metric("Memory Usage", f"{memory_usage:.2f} MB")
        with col4:
            missing_pct = (df.isna().sum().sum() / (df.shape[0] * df.shape[1])) * 100
            st.metric("Missing Values", f"{missing_pct:.1f}%")
        
        # Data preview
        st.subheader("📋 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Comprehensive Statistics
        st.header("📈 Comprehensive Statistical Analysis")
        
        with st.spinner("Computing comprehensive statistics..."):
            stats = eda_utils.get_comprehensive_stats(df)
        
        
        # Data types and basic info
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Data Types")
            dtype_df = pd.DataFrame({
                'Column': df.columns,
                'Data Type': [str(dtype) for dtype in df.dtypes],
                'Non-Null Count': df.count(),
                'Null Count': df.isna().sum()
            })
            st.dataframe(dtype_df, use_container_width=True)
        
        with col2:
            st.subheader("Missing Values Analysis")
            missing_df = pd.DataFrame({
                'Column': df.columns,
                'Missing Count': df.isna().sum(),
                'Missing %': (df.isna().sum() / len(df) * 100).round(2)
            }).sort_values('Missing %', ascending=False)
            st.dataframe(missing_df, use_container_width=True)
        
        # Univariate Analysis
        st.header("🔍 Univariate Analysis")

        # Improved column type detection
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

        if numeric_cols:
            st.subheader("📊 Numeric Variables Analysis")
            selected_numeric = st.selectbox("Select Numeric Column for Analysis", numeric_cols, key=f"numeric_col_{data_source}")
            
            if selected_numeric:
                # Debug information in expander
                with st.expander("🔍 Debug Information", expanded=False):
                    st.write(f"**Column**: {selected_numeric}")
                    st.write(f"**Data type**: {df[selected_numeric].dtype}")
                    st.write(f"**Non-null count**: {df[selected_numeric].count()}")
                    st.write(f"**Unique values**: {df[selected_numeric].nunique()}")
                    sample_vals = df[selected_numeric].dropna().head(5).tolist()
                    st.write(f"**Sample values**: {sample_vals}")
                
                # Analysis and plotting
                univariate_result = eda_utils.univariate_analysis(df, selected_numeric)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Statistical Summary")
                    if "error" in univariate_result:
                        st.error(f"Error: {univariate_result['error']}")
                    else:
                        if univariate_result.get('type') == 'numeric':
                            st.metric("Count", univariate_result.get('count', 'N/A'))
                            st.metric("Mean", f"{univariate_result.get('mean', 0):.3f}")
                            st.metric("Median", f"{univariate_result.get('median', 0):.3f}")
                            st.metric("Std Dev", f"{univariate_result.get('std', 0):.3f}")
                
                with col2:
                    st.subheader("Distribution Plot")
                    if show_distributions:
                        plot_type = st.selectbox("Plot Type", ["histogram", "box", "violin"], 
                                               key=f"numeric_plot_{selected_numeric}")
                        
                        try:
                            fig_dist = report_utils.create_distribution_plot(df, selected_numeric, plot_type)
                            if fig_dist is not None:
                                st.plotly_chart(fig_dist, use_container_width=True, 
                                              key=f"plot_{selected_numeric}_{plot_type}")
                                
                                # Insights
                                plot_insights = eda_utils.generate_plot_insights(df, "distribution", selected_numeric)
                                st.markdown("**📊 Plot Insights:**")
                                st.markdown(plot_insights)
                            else:
                                st.error("Could not create plot. Check debug information above.")
                                
                                # Fallback matplotlib plot
                                try:
                                    import matplotlib.pyplot as plt
                                    clean_data = df[selected_numeric].dropna()
                                    if len(clean_data) > 0:
                                        fig_mpl, ax = plt.subplots(figsize=(8, 4))
                                        ax.hist(clean_data, bins=30, alpha=0.7)
                                        ax.set_title(f"Distribution of {selected_numeric}")
                                        ax.set_xlabel(selected_numeric)
                                        ax.set_ylabel("Frequency")
                                        st.pyplot(fig_mpl)
                                        plt.close()
                                except Exception as fallback_error:
                                    st.error(f"Fallback plot failed: {fallback_error}")
                                    
                        except Exception as plot_error:
                            st.error(f"Plot error: {plot_error}")
        
        # Bivariate Analysis
        if show_bivariate and len(df.columns) >= 2:
            st.header("🔗 Bivariate Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                var1 = st.selectbox("Select First Variable", df.columns, key=f"var1_{data_source}")
            with col2:
                var2 = st.selectbox("Select Second Variable", df.columns, key=f"var2_{data_source}")
            
            if var1 != var2:
                # Bivariate analysis (run in backend but don't display results)
                bivariate_result = eda_utils.bivariate_analysis(df, var1, var2)
                
                # Bivariate plots
                fig_bivariate = report_utils.create_bivariate_analysis(df, var1, var2)
                if fig_bivariate:
                    st.plotly_chart(fig_bivariate, use_container_width=True, key=f"bivariate_plot_{var1}_{var2}")
                    
                    # Generate and display plot-specific insights
                    plot_insights = eda_utils.generate_plot_insights(df, "bivariate", var1, var2, bivariate_result)
                    st.markdown("**📊 Plot Insights:**")
                    st.markdown(plot_insights)
                else:
                    st.warning(f"Could not create bivariate plot for {var1} vs {var2}")
        
        # Multivariate Analysis
        if show_multivariate:
            st.header("🌐 Multivariate Analysis")
            
            with st.spinner("Computing multivariate analysis..."):
                multivariate_result = eda_utils.multivariate_analysis(df)
        else:
            multivariate_result = {}
        
        
        # Show correlation analysis
        if 'correlation_analysis' in multivariate_result:
            st.subheader("📊 Correlation Analysis")
            
            corr_analysis = multivariate_result['correlation_analysis']
            
            col1, col2 = st.columns(2)
            
            with col1:
                if corr_analysis['strong_correlations']:
                    st.write("**Strong Correlations (|r| > 0.7):**")
                    for corr in corr_analysis['strong_correlations']:
                        st.write(f"- {corr['pair']}: {corr['correlation']:.3f}")
            
            with col2:
                if corr_analysis['moderate_correlations']:
                    st.write("**Moderate Correlations (0.3 < |r| < 0.7):**")
                    for corr in corr_analysis['moderate_correlations']:
                        st.write(f"- {corr['pair']}: {corr['correlation']:.3f}")
            
            # Advanced correlation visualization
            if show_correlations:
                fig_corr = report_utils.create_correlation_analysis(df)
                if fig_corr:
                    st.plotly_chart(fig_corr, use_container_width=True, key="correlation_analysis_main")
                    
                    # Generate and display plot-specific insights
                    plot_insights = eda_utils.generate_plot_insights(df, "correlation", "correlation_matrix", analysis_result=multivariate_result)
                    st.markdown("**📊 Plot Insights:**")
                    st.markdown(plot_insights)
            
            # Target-Based Feature Analysis
            st.subheader("🎯 Target-Based Feature Analysis")
            
            if selected_target:
                st.info(f"🎯 Analyzing features against target variable: **{selected_target}**")
                
                # Select feature for analysis
                feature_options = [col for col in df.columns if col != selected_target]
                selected_feature = st.selectbox("Select Feature for Target Analysis", feature_options, key=f"target_feature_{data_source}")
                
                if selected_feature:
                    # Perform target-based analysis
                    with st.spinner("Performing target-based analysis..."):
                        analysis_result = report_utils.create_target_based_groupby_analysis(df, selected_target, selected_feature)
                    
                    if analysis_result:
                        st.subheader(f"📈 {selected_feature} → {selected_target} Analysis")
                        
                        # Display statistics
                        if analysis_result['type'] in ['numeric_target_categorical_feature', 'categorical_target_numeric_feature']:
                            st.dataframe(analysis_result['stats'], use_container_width=True)
                        elif analysis_result['type'] == 'categorical_target_categorical_feature':
                            st.dataframe(analysis_result['stats'], use_container_width=True)
                        else:  # numeric_target_numeric_feature
                            st.metric("Correlation Coefficient", f"{analysis_result['stats']['correlation']:.3f}")
                        
                        # Display visualization
                        st.plotly_chart(analysis_result['visualization'], use_container_width=True, 
                                      key=f"target_analysis_{selected_feature}_{selected_target}")
                        
                        # Generate and display 2-pointer insights
                        st.markdown("**🎯 Target Variable Impact Insights:**")
                        comparison_insights = report_utils.generate_target_comparison_insights(
                            df, selected_target, selected_feature, analysis_result
                        )
                        st.markdown(comparison_insights)
                        
                        # Additional feature analysis for all features
                        st.subheader("🔍 All Features vs Target Analysis")
                        
                        # Create a summary table for all features
                        feature_impact_summary = []
                        
                        for feature in feature_options[:10]:  # Limit to first 10 features for performance
                            try:
                                feature_analysis = report_utils.create_target_based_groupby_analysis(df, selected_target, feature)
                                if feature_analysis:
                                    if feature_analysis['type'] == 'numeric_target_numeric_feature':
                                        corr = feature_analysis['stats']['correlation']
                                        feature_impact_summary.append({
                                            'Feature': feature,
                                            'Impact_Type': 'Correlation',
                                            'Impact_Value': f"{corr:.3f}",
                                            'Strength': 'High' if abs(corr) > 0.7 else 'Medium' if abs(corr) > 0.3 else 'Low'
                                        })
                                    elif feature_analysis['type'] in ['numeric_target_categorical_feature', 'categorical_target_numeric_feature']:
                                        stats = feature_analysis['stats']
                                        if 'mean' in stats.columns:
                                            impact_range = stats['mean'].max() - stats['mean'].min()
                                            feature_impact_summary.append({
                                                'Feature': feature,
                                                'Impact_Type': 'Range',
                                                'Impact_Value': f"{impact_range:.2f}",
                                                'Strength': 'High' if impact_range > stats['std'].mean() else 'Medium' if impact_range > stats['std'].mean()/2 else 'Low'
                                            })
                            except:
                                continue
                        
                        if feature_impact_summary:
                            impact_df = pd.DataFrame(feature_impact_summary)
                            st.dataframe(impact_df, use_container_width=True)
                            
                            # Show top impacting features
                            st.markdown("**🏆 Top Impacting Features:**")
                            if 'Impact_Value' in impact_df.columns:
                                # Sort by absolute impact value
                                impact_df['abs_impact'] = impact_df['Impact_Value'].str.extract(r'([-+]?\d*\.?\d+)').astype(float).abs()
                                top_features = impact_df.nlargest(5, 'abs_impact')
                                
                                for idx, row in top_features.iterrows():
                                    st.markdown(f"• **{row['Feature']}**: {row['Impact_Type']} = {row['Impact_Value']} ({row['Strength']} impact)")
                    else:
                        st.warning(f"Could not perform analysis for {selected_feature} vs {selected_target}")
            else:
                st.info("Select a target variable above to enable target-based feature analysis.")
        
        
        # Advanced Visualizations
        st.header("📊 Advanced Visualizations")
        
        # Create tabs for different visualization types
        tab1, tab2, tab3, tab4 = st.tabs(["Distribution Analysis", "Correlation Analysis", "Bivariate Analysis", "Multivariate Analysis"])
        
        with tab1:
            if numeric_cols:
                selected_col = st.selectbox("Select Column for Distribution Analysis", numeric_cols, key=f"dist_col_{data_source}")
                if selected_col:
                    col1, col2 = st.columns(2)
                    with col1:
                        fig_hist = report_utils.create_distribution_plot(df, selected_col, 'histogram')
                        if fig_hist:
                            st.plotly_chart(fig_hist, use_container_width=True, key=f"tab_hist_{selected_col}")
                            # Plot insights
                            plot_insights = eda_utils.generate_plot_insights(df, "distribution", selected_col)
                            st.markdown("**📊 Histogram Insights:**")
                            st.markdown(plot_insights)
                    with col2:
                        fig_box = report_utils.create_distribution_plot(df, selected_col, 'box')
                        if fig_box:
                            st.plotly_chart(fig_box, use_container_width=True, key=f"tab_box_{selected_col}")
                            # Plot insights
                            plot_insights = eda_utils.generate_plot_insights(df, "distribution", selected_col)
                            st.markdown("**📊 Box Plot Insights:**")
                            st.markdown(plot_insights)
        
        with tab2:
            if len(numeric_cols) >= 2:
                fig_corr_advanced = report_utils.create_correlation_analysis(df)
                if fig_corr_advanced:
                    st.plotly_chart(fig_corr_advanced, use_container_width=True, key="tab_correlation_advanced")
                    # Plot insights
                    plot_insights = eda_utils.generate_plot_insights(df, "correlation", "correlation_matrix", analysis_result=multivariate_result)
                    st.markdown("**📊 Correlation Analysis Insights:**")
                    st.markdown(plot_insights)
            else:
                st.info("Need at least 2 numeric columns for correlation analysis.")
        
        with tab3:
            if len(df.columns) >= 2:
                col1, col2 = st.columns(2)
                with col1:
                    var1_biv = st.selectbox("Variable 1", df.columns, key=f"biv_var1_{data_source}")
                with col2:
                    var2_biv = st.selectbox("Variable 2", df.columns, key=f"biv_var2_{data_source}")
                
                if var1_biv != var2_biv:
                    # Perform bivariate analysis
                    bivariate_result_tab = eda_utils.bivariate_analysis(df, var1_biv, var2_biv)
                    
                    fig_biv = report_utils.create_bivariate_analysis(df, var1_biv, var2_biv)
                    if fig_biv:
                        st.plotly_chart(fig_biv, use_container_width=True, key=f"tab_bivariate_{var1_biv}_{var2_biv}")
                        
                        # Plot insights
                        plot_insights = eda_utils.generate_plot_insights(df, "bivariate", var1_biv, var2_biv, bivariate_result_tab)
                        st.markdown("**📊 Bivariate Analysis Insights:**")
                        st.markdown(plot_insights)
        
        with tab4:
            if len(numeric_cols) >= 2:
                fig_multi = report_utils.create_multivariate_analysis(df)
                if fig_multi:
                    st.plotly_chart(fig_multi, use_container_width=True, key="tab_multivariate")
                    # Plot insights
                    st.markdown("**📊 Multivariate Analysis Insights:**")
                    st.markdown("• **Parallel Coordinates Plot**: Shows relationships between multiple numeric variables")
                    st.markdown("• **Each line** represents one data point across all dimensions")
                    st.markdown("• **Clustering patterns** indicate similar data points")
                    st.markdown("• **Crossing lines** suggest complex relationships between variables")
            else:
                st.info("Need at least 2 numeric columns for multivariate analysis.")
        
        # Report Generation
        st.header("📄 Generate Comprehensive Report")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Show status if summary has been generated
            if st.session_state.get('summary_generated', False):
                st.info("✅ Summary statistics are ready for the report!")
            
            if st.button("📊 Generate Word Report", type="primary"):
                with st.spinner("Generating comprehensive report..."):
                    try:
                        # Create image buffers for report
                        img_buffers = {}
                        
                        # Add distribution plots
                        if numeric_cols:
                            selected_col = numeric_cols[0]
                            fig_dist = report_utils.create_distribution_plot(df, selected_col, 'histogram')
                            if fig_dist:
                                buf = io.BytesIO()
                                fig_dist.write_image(buf, format='png', scale=2)
                                buf.seek(0)
                                img_buffers[f"Distribution: {selected_col}"] = buf
                        
                        # Add correlation plot
                        if len(numeric_cols) >= 2:
                            fig_corr = report_utils.create_correlation_analysis(df)
                            if fig_corr:
                                buf = io.BytesIO()
                                fig_corr.write_image(buf, format='png', scale=2)
                                buf.seek(0)
                                img_buffers["Correlation Analysis"] = buf
                        
                        # Add categorical analysis
                        if categorical_cols:
                            selected_cat = categorical_cols[0]
                            fig_cat = report_utils.create_categorical_analysis(df, selected_cat)
                            if fig_cat:
                                buf = io.BytesIO()
                                fig_cat.write_image(buf, format='png', scale=2)
                                buf.seek(0)
                                img_buffers[f"Categorical Analysis: {selected_cat}"] = buf
                        
                        # Use stored comprehensive summary or generate new one
                        if 'comprehensive_summary' in st.session_state and st.session_state.get('summary_generated', False):
                            comprehensive_summary = st.session_state['comprehensive_summary']
                        else:
                            # Generate comprehensive summary for report if not already generated
                            comprehensive_summary = llm_utils.generate_comprehensive_summary(
                                df, stats, multivariate_result, selected_target
                            )
                        
                        # Generate report
                        report_path = report_utils.create_word_report(comprehensive_summary, img_buffers)
                        st.success(f"✅ Report generated successfully: {report_path}")
                        
                        # Provide download link
                        with open(report_path, "rb") as file:
                            st.download_button(
                                label="📥 Download Report",
                                data=file.read(),
                                file_name="comprehensive_data_analysis_report.docx",
                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                            )

                    except Exception as e:
                        st.error(f"Error generating report: {e}")
        
        with col2:
            if st.button("📈 Generate Summary Statistics"):
                with st.spinner("Generating comprehensive summary with insights..."):
                    # Generate comprehensive summary using the new function
                    comprehensive_summary = llm_utils.generate_comprehensive_summary(
                        df, stats, multivariate_result, selected_target
                    )
                    
                    # Store the summary in session state for later use in Word report
                    st.session_state['comprehensive_summary'] = comprehensive_summary
                    st.session_state['summary_generated'] = True
                    
                    # Show success message with green tick
                    st.success("✅ Summary Generated Successfully!")
                    
                    # Add a nice visual indicator
                    st.markdown("""
                    <div style="text-align: center; padding: 2rem; background-color: #f0f8ff; border-radius: 10px; border: 2px solid #4CAF50;">
                        <h3 style="color: #4CAF50; margin-bottom: 1rem;">📊 Summary Statistics Generated</h3>
                        <p style="color: #666; font-size: 1.1rem;">Your comprehensive data analysis summary has been prepared and is ready for inclusion in the Word report.</p>
                        <p style="color: #888; font-size: 0.9rem;">Click "Generate Word Report" to create a downloadable report with the summary.</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; margin-top: 2rem;">
            <p>🤖 AI Data Storyteller - Comprehensive Data Analysis Platform</p>
            <p>Powered by Streamlit, Plotly, and Advanced Machine Learning</p>
        </div>
        """, unsafe_allow_html=True)
        
        return True
        
    except Exception as e:
        st.error(f"❌ Error processing data: {e}")
        st.exception(e)
        return False

# Main content
uploaded_file = st.file_uploader(
    "📁 Upload CSV File", 
    type=["csv"],
    help="Upload a CSV file to begin comprehensive data analysis"
)

# Handle file upload
if uploaded_file is not None:
    try:
        # Load and validate data
        with st.spinner("Loading and validating data..."):
            df = eda_utils.read_csv(uploaded_file)
        
        # Store in session state
        st.session_state.current_dataframe = df
        st.session_state.data_loaded = True
        st.session_state.data_source = "uploaded_file"
        
        # Run comprehensive analysis
        run_comprehensive_analysis(df, "uploaded_file", clean_data, handle_outliers, fill_missing, show_distributions, show_correlations, show_bivariate, show_multivariate)
    
    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
        st.exception(e)

# Handle sample data loading
elif st.button("🎯 Try Sample Dataset", type="primary"):
    try:
        with st.spinner("Loading sample dataset..."):
            sample_df = eda_utils.load_sample_data()
        
        if sample_df is not None:
            st.success("✅ Sample dataset loaded successfully!")
            
            # Store in session state
            st.session_state.current_dataframe = sample_df
            st.session_state.data_loaded = True
            st.session_state.data_source = "sample_dataset"
            
            # Run comprehensive analysis on sample data
            run_comprehensive_analysis(sample_df, "sample_dataset", clean_data, handle_outliers, fill_missing, show_distributions, show_correlations, show_bivariate, show_multivariate)
        else:
            st.error("❌ Failed to load sample dataset. Please check if the data file exists.")
            st.info("💡 You can still upload your own CSV file using the file uploader above.")
    except Exception as e:
        st.error(f"❌ Error loading sample data: {e}")
        st.info("💡 You can still upload your own CSV file using the file uploader above.")

# Show analysis if data is loaded
elif st.session_state.data_loaded and st.session_state.current_dataframe is not None:
    # Run comprehensive analysis on stored data
    run_comprehensive_analysis(st.session_state.current_dataframe, st.session_state.data_source, clean_data, handle_outliers, fill_missing, show_distributions, show_correlations, show_bivariate, show_multivariate)

# Show welcome message if no data loaded
else:
    # Welcome message and sample data option
    st.markdown("""
    <div style="text-align: center; padding: 2rem;">
        <h2>Welcome to AI Data Storyteller! 🚀</h2>
        <p style="font-size: 1.1rem; margin: 1rem 0;">
            Upload a CSV file to get started with comprehensive data analysis, 
            or try our sample dataset to explore the features.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📊 Features
        - **Comprehensive EDA**: Univariate, bivariate, and multivariate analysis
        - **Advanced Visualizations**: Interactive plots with Plotly
        - **Data Cleaning**: Automatic missing value handling and outlier detection
        - **Statistical Analysis**: Correlation, distribution, and significance tests
        - **AI Insights**: LLM-powered technical insights and recommendations
        """)
    
    with col2:
        st.markdown("""
        ### 🔧 Analysis Types
        - **Distribution Analysis**: Histograms, box plots, violin plots
        - **Correlation Analysis**: Heatmaps, network plots, strength analysis
        - **Bivariate Analysis**: Scatter plots, contingency tables
        - **Multivariate Analysis**: Parallel coordinates, scatter matrices
        - **Outlier Detection**: IQR-based outlier identification and treatment
        """)
    
    with col3:
        st.markdown("""
        ### 📄 Reports
        - **Word Reports**: Professional downloadable reports with visualizations
        - **Interactive Dashboards**: Real-time exploration and analysis
        - **Custom Visualizations**: Tailored plots based on data characteristics
        """)