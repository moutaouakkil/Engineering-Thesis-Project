import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import os

# Create necessary directories
def create_directories():
    """Create directories for organizing analysis results."""
    base_dir = 'analysis_results'
    subdirs = ['csv_files', 'plots', 'plots/performance', 'plots/categories', 'plots/correlation']
    
    for subdir in subdirs:
        os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)
    
    return base_dir

def load_and_prepare_data():
    """Load and prepare the data for analysis."""
    # Read the CSV file
    df = pd.read_csv('processed_results.csv')
    
    # List of metrics to analyze
    en_metrics = ['EN_Cross-Model Agreement', 'EN_Set Coverage', 'EN_Response Accuracy Rate',
                 'EN_Accuracy_Scale', 'EN_Cultural_Sensitivity', 'EN_Language_Quality',
                 'EN_Contextual_Relevance']
    
    ar_metrics = ['AR_Cross-Model Agreement', 'AR_Set Coverage', 'AR_Response Accuracy Rate',
                 'AR_Accuracy_Scale', 'AR_Cultural_Sensitivity', 'AR_Language_Quality',
                 'AR_Contextual_Relevance']
    
    return df, en_metrics, ar_metrics

def calculate_model_performance(df, metrics, language="English"):
    """Calculate average performance metrics for each model."""
    model_performance = df.groupby('Model')[metrics].mean()
    
    # Create a summary DataFrame
    summary = pd.DataFrame()
    summary[f'{language} Overall Score'] = model_performance.mean(axis=1)
    
    # Add individual metrics
    for metric in metrics:
        summary[metric] = model_performance[metric]
    
    return summary.round(2)

def analyze_category_performance(df, metrics, language="English"):
    """Analyze performance across different categories."""
    return df.groupby('Category')[metrics].mean().round(2)

def calculate_language_disparity_stats(df):
    """Calculate statistics about language disparity."""
    disparity_stats = {
        'Mean Disparity': df['Language Disparity'].mean(),
        'Median Disparity': df['Language Disparity'].median(),
        'Max Disparity': df['Language Disparity'].max(),
        'Min Disparity': df['Language Disparity'].min(),
        'Std Disparity': df['Language Disparity'].std()
    }
    return pd.Series(disparity_stats).round(2)

def plot_model_performance(df, metrics, language="English", base_dir='analysis_results'):
    """Create a heatmap of model performance across metrics."""
    plt.figure(figsize=(12, 8))
    performance_data = df.groupby('Model')[metrics].mean()
    
    # Create heatmap
    sns.heatmap(performance_data, annot=True, cmap='YlOrRd', fmt='.2f')
    plt.title(f'{language} Language Model Performance Across Metrics')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(base_dir, 'plots', 'performance', f'{language.lower()}_model_performance.png')
    plt.savefig(save_path)
    plt.close()

def plot_category_comparison(df, metrics, language="English", base_dir='analysis_results'):
    """Create a box plot comparing performance across categories."""
    plt.figure(figsize=(15, 8))
    
    # Melt the DataFrame for easier plotting
    melted_df = pd.melt(df, 
                        id_vars=['Category'], 
                        value_vars=metrics,
                        var_name='Metric', 
                        value_name='Score')
    
    # Create box plot
    sns.boxplot(x='Category', y='Score', data=melted_df)
    plt.xticks(rotation=45, ha='right')
    plt.title(f'{language} Performance by Category')
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(base_dir, 'plots', 'categories', f'{language.lower()}_category_performance.png')
    plt.savefig(save_path)
    plt.close()

def analyze_correlation_matrix(df, metrics, language="English", base_dir='analysis_results'):
    """Create and plot a correlation matrix between metrics."""
    correlation_matrix = df[metrics].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title(f'{language} Metrics Correlation Matrix')
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(base_dir, 'plots', 'correlation', f'{language.lower()}_correlation_matrix.png')
    plt.savefig(save_path)
    plt.close()
    
    return correlation_matrix

def generate_summary_report(en_performance, ar_performance, disparity_stats, base_dir='analysis_results'):
    """Generate a summary report of the analysis."""
    with open(os.path.join(base_dir, 'analysis_summary.txt'), 'w', encoding='utf-8') as f:
        f.write("Analysis Summary Report\n")
        f.write("=====================\n\n")
        
        f.write("1. English Language Performance\n")
        f.write("--------------------------\n")
        f.write(str(en_performance))
        f.write("\n\n")
        
        f.write("2. Arabic Language Performance\n")
        f.write("-------------------------\n")
        f.write(str(ar_performance))
        f.write("\n\n")
        
        f.write("3. Language Disparity Statistics\n")
        f.write("----------------------------\n")
        f.write(str(disparity_stats))
        f.write("\n\n")
        
        # Add best performing models
        f.write("4. Best Performing Models\n")
        f.write("---------------------\n")
        f.write(f"Best English Model: {en_performance['English Overall Score'].idxmax()} ")
        f.write(f"(Score: {en_performance['English Overall Score'].max():.2f})\n")
        f.write(f"Best Arabic Model: {ar_performance['Arabic Overall Score'].idxmax()} ")
        f.write(f"(Score: {ar_performance['Arabic Overall Score'].max():.2f})\n")

def main():
    # Create directory structure
    base_dir = create_directories()
    
    # Load and prepare data
    df, en_metrics, ar_metrics = load_and_prepare_data()
    
    # Calculate model performance for both languages
    en_performance = calculate_model_performance(df, en_metrics, "English")
    ar_performance = calculate_model_performance(df, ar_metrics, "Arabic")
    
    # Analyze category performance
    en_category_perf = analyze_category_performance(df, en_metrics, "English")
    ar_category_perf = analyze_category_performance(df, ar_metrics, "Arabic")
    
    # Calculate language disparity statistics
    disparity_stats = calculate_language_disparity_stats(df)
    
    # Create visualizations
    plot_model_performance(df, en_metrics, "English", base_dir)
    plot_model_performance(df, ar_metrics, "Arabic", base_dir)
    plot_category_comparison(df, en_metrics, "English", base_dir)
    plot_category_comparison(df, ar_metrics, "Arabic", base_dir)
    
    # Calculate correlation matrices
    en_correlation = analyze_correlation_matrix(df, en_metrics, "English", base_dir)
    ar_correlation = analyze_correlation_matrix(df, ar_metrics, "Arabic", base_dir)
    
    # Save results to CSV files
    csv_dir = os.path.join(base_dir, 'csv_files')
    en_performance.to_csv(os.path.join(csv_dir, 'english_model_performance.csv'))
    ar_performance.to_csv(os.path.join(csv_dir, 'arabic_model_performance.csv'))
    en_category_perf.to_csv(os.path.join(csv_dir, 'english_category_performance.csv'))
    ar_category_perf.to_csv(os.path.join(csv_dir, 'arabic_category_performance.csv'))
    disparity_stats.to_csv(os.path.join(csv_dir, 'language_disparity_stats.csv'))
    
    # Generate summary report
    generate_summary_report(en_performance, ar_performance, disparity_stats, base_dir)
    
    print(f"\nAnalysis completed! Results have been saved to the '{base_dir}' directory.")
    print("\nDirectory structure:")
    print(f"- {base_dir}/")
    print("  ├── csv_files/")
    print("  └── plots/")
    print("      ├── performance/")
    print("      ├── categories/")
    print("      └── correlation/")

if __name__ == "__main__":
    main() 