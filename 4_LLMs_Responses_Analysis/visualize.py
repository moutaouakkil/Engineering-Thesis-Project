import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('3_LLMs_Responses_Metrics_Calculation/processed_results.csv')

# Create output directory if it doesn't exist
import os
if not os.path.exists('4_LLMs_Responses_Analysis/visualizations'):
    os.makedirs('4_LLMs_Responses_Analysis/visualizations')

# 1. MODEL PERFORMANCE RADAR CHART
def radar_chart():
    # Calculate average scores per model
    metrics = ['EN_Response Accuracy Rate', 'EN_Cultural_Sensitivity', 
              'EN_Language_Quality', 'EN_Contextual_Relevance']
    model_perf = df.groupby('Model')[metrics].mean().reset_index()
    
    # Create radar chart
    _, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
    
    # Number of variables
    categories = ['Accuracy', 'Cultural Sensitivity', 'Language Quality', 'Relevance']
    N = len(categories)
    
    # Create angles for each metric
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Set up colors for each model
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_perf['Model'])))
    
    # Plot each model
    for i, (_, row) in enumerate(model_perf.iterrows()):
        # Scale values to 0-100 range
        values = row[metrics].values.tolist()
        values = [v for v in values]  # No scaling needed as values are already in 0-100 range
        values += values[:1]  # Close the loop
        
        # Plot the line
        ax.plot(angles, values, 'o-', linewidth=2, label=row['Model'], color=colors[i])
        # Fill with lower alpha
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    
    # Set category labels
    plt.xticks(angles[:-1], categories, size=12)
    
    # Add legend with better positioning
    plt.legend(loc='center left', bbox_to_anchor=(1.2, 0.5))
    
    # Set title
    ax.set_title('Model Performance Comparison', size=15, pad=20)
    
    # Add gridlines and adjust their appearance
    ax.grid(True, alpha=0.3)
    
    # Set y-axis limits to better show differences
    ax.set_ylim(0, 100)
    
    # Add y-axis labels
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
    
    # Add radial grid lines
    ax.set_rgrids([20, 40, 60, 80, 100], angle=0)
    
    plt.tight_layout()
    plt.savefig('4_LLMs_Responses_Analysis/visualizations/model_radar_chart.png', dpi=300, bbox_inches='tight')
    plt.close()

# 2. LANGUAGE DISPARITY CHART
def language_disparity():
    # Calculate language disparity as absolute difference between EN and AR accuracy
    disparity = df.groupby('Model').apply(lambda x: abs(
        x['EN_Response Accuracy Rate'].mean() - x['AR_Response Accuracy Rate'].mean()
    )).reset_index()
    disparity.columns = ['Model', 'Language Disparity']
    disparity = disparity.sort_values('Language Disparity', ascending=True)
    
    # Create bar chart
    plt.figure(figsize=(12, 6))
    sns.barplot(data=disparity, x='Model', y='Language Disparity', hue='Model', legend=False)
    
    # Add value labels
    for i, v in enumerate(disparity['Language Disparity']):
        plt.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=10)
    
    plt.title('Language Performance Disparity by Model\n(Lower is better - shows absolute difference between EN and AR accuracy)', fontsize=14)
    plt.ylabel('Accuracy Difference (%)')
    plt.xlabel('Model')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('4_LLMs_Responses_Analysis/visualizations/language_disparity.png', dpi=300, bbox_inches='tight')
    plt.close()

# 3. CATEGORY PERFORMANCE HEATMAP
def category_heatmap():
    # Create pivot table of model performance by category
    category_perf = df.groupby(['Model', 'Category'])['EN_Response Accuracy Rate'].mean().reset_index()
    heatmap_data = category_perf.pivot(index='Model', columns='Category', values='EN_Response Accuracy Rate')
    
    # Create heatmap
    plt.figure(figsize=(14, 8))
    sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', fmt='.1f', 
                linewidths=.5, cbar_kws={'label': 'Accuracy Rate (%)'})
    
    plt.title('Model Performance by Category', fontsize=15)
    plt.tight_layout()
    
    plt.savefig('4_LLMs_Responses_Analysis/visualizations/category_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

# 4. EN vs AR PERFORMANCE COMPARISON
def language_comparison():
    # Calculate average EN and AR metrics by model
    en_metrics = df.groupby('Model')['EN_Response Accuracy Rate'].mean().reset_index()
    ar_metrics = df.groupby('Model')['AR_Response Accuracy Rate'].mean().reset_index()
    
    # Merge data
    comparison = pd.merge(en_metrics, ar_metrics, on='Model')
    comparison = comparison.sort_values('EN_Response Accuracy Rate', ascending=False)
    
    # Create comparison bar chart
    plt.figure(figsize=(14, 7))
    
    x = np.arange(len(comparison))
    width = 0.35
    
    plt.bar(x - width/2, comparison['EN_Response Accuracy Rate'], width, 
            label='English', color='royalblue')
    plt.bar(x + width/2, comparison['AR_Response Accuracy Rate'], width, 
            label='Arabic/Darija', color='tomato')
    
    plt.xlabel('Model')
    plt.ylabel('Accuracy Rate (%)')
    plt.title('English vs Arabic/Darija Performance by Model', fontsize=15)
    plt.xticks(x, comparison['Model'], rotation=45)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig('4_LLMs_Responses_Analysis/visualizations/language_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

# 5. CULTURAL SENSITIVITY vs ACCURACY SCATTER PLOT
def sensitivity_accuracy():
    # Calculate average metrics by model
    model_metrics = df.groupby('Model').agg({
        'EN_Response Accuracy Rate': 'mean',
        'EN_Cultural_Sensitivity': 'mean',
        'EN_Contextual_Relevance': 'mean'
    }).reset_index()
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    
    sizes = model_metrics['EN_Contextual_Relevance'] * 5
    
    plt.scatter(model_metrics['EN_Cultural_Sensitivity'], 
              model_metrics['EN_Response Accuracy Rate'],
              s=sizes, alpha=0.7,
              c=range(len(model_metrics)), cmap='viridis')
    
    # Add model labels
    for i, model in enumerate(model_metrics['Model']):
        plt.annotate(model, 
                   (model_metrics['EN_Cultural_Sensitivity'][i], 
                    model_metrics['EN_Response Accuracy Rate'][i]),
                   xytext=(7, -5), textcoords='offset points')
    
    plt.xlabel('Cultural Sensitivity Score')
    plt.ylabel('Accuracy Rate (%)')
    plt.title('Cultural Sensitivity vs Accuracy by Model\n(Bubble size = Contextual Relevance)', fontsize=14)
    plt.tight_layout()
    
    plt.savefig('4_LLMs_Responses_Analysis/visualizations/sensitivity_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()

# 6. CORRELATION MATRIX OF EVALUATION METRICS
def correlation_matrix():
    # Select metric columns
    metric_cols = [col for col in df.columns if 
                  any(x in col for x in ['Accuracy', 'Sensitivity', 'Quality', 'Relevance', 'Coverage', 'Agreement'])]
    
    # Calculate correlation matrix
    corr = df[metric_cols].corr()
    
    # Create correlation heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                annot=True, fmt='.2f', square=True, linewidths=.5)
    
    plt.title('Correlation Matrix of Evaluation Metrics', fontsize=16)
    plt.tight_layout()
    
    plt.savefig('4_LLMs_Responses_Analysis/visualizations/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

# 7. CATEGORY DIFFICULTY ANALYSIS
def category_difficulty():
    # Calculate average accuracy and standard deviation by category
    cat_diff = df.groupby('Category')['EN_Response Accuracy Rate'].agg(['mean', 'std']).reset_index()
    cat_diff = cat_diff.sort_values('mean')
    
    # Create error bar plot
    plt.figure(figsize=(12, 6))
    plt.errorbar(x=cat_diff['Category'], y=cat_diff['mean'],
                yerr=cat_diff['std'], fmt='o', capsize=5, ecolor='red', capthick=2)
    
    plt.title('Category Difficulty Ranking with Uncertainty', fontsize=15)
    plt.ylabel('Average Accuracy Rate (%)')
    plt.xlabel('Category')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig('4_LLMs_Responses_Analysis/visualizations/category_difficulty.png', dpi=300, bbox_inches='tight')
    plt.close()

# 8. MODEL PERFORMANCE BY CATEGORY
def model_category_performance():
    # Create recommendation system for different categories
    categories = df['Category'].unique()
    
    fig, axes = plt.subplots(4, 2, figsize=(18, 20))
    axes = axes.flatten()
    
    for i, category in enumerate(categories):
        if i < len(axes):
            # Filter data for this category
            cat_data = df[df['Category'] == category]
            
            # Calculate average accuracy by model for this category
            model_perf = cat_data.groupby('Model')['EN_Response Accuracy Rate'].mean().reset_index()
            model_perf = model_perf.sort_values('EN_Response Accuracy Rate', ascending=False)
            
            # Plot
            sns.barplot(x='Model', y='EN_Response Accuracy Rate', data=model_perf, 
                      palette='viridis', ax=axes[i])
            
            axes[i].set_title(f'Model Performance on {category} Questions', fontsize=12)
            axes[i].set_ylabel('Accuracy Rate (%)')
            axes[i].set_xlabel('')
            axes[i].tick_params(axis='x', rotation=45)
    
    # Remove any unused subplots
    for j in range(len(categories), len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.savefig('4_LLMs_Responses_Analysis/visualizations/model_by_category.png', dpi=300, bbox_inches='tight')
    plt.close()

# 9. PAIRWISE MODEL COMPARISON OF CROSS-MODEL AGREEMENT
def agreement_analysis():
    # Calculate average cross-model agreement by category
    agreement = df.groupby('Category')['EN_Cross-Model Agreement'].mean().reset_index()
    agreement = agreement.sort_values('EN_Cross-Model Agreement', ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Category', y='EN_Cross-Model Agreement', data=agreement, palette='Blues_d')
    
    plt.title('Cross-Model Agreement by Category', fontsize=15)
    plt.ylabel('Average Agreement Score (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig('4_LLMs_Responses_Analysis/visualizations/cross_model_agreement.png', dpi=300, bbox_inches='tight')
    plt.close()

# 10. MODEL CONSISTENCY ANALYSIS
def model_consistency():
    # Calculate variance in performance across categories for each model
    model_var = df.groupby('Model')['EN_Response Accuracy Rate'].agg(['mean', 'std']).reset_index()
    model_var['coefficient_of_variation'] = (model_var['std'] / model_var['mean']) * 100
    model_var = model_var.sort_values('coefficient_of_variation')
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Model', y='coefficient_of_variation', data=model_var, palette='viridis')
    
    plt.title('Model Consistency Analysis\n(Lower values indicate more consistent performance)', 
             fontsize=14)
    plt.ylabel('Coefficient of Variation (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig('4_LLMs_Responses_Analysis/visualizations/model_consistency.png', dpi=300, bbox_inches='tight')
    plt.close()

# Run all visualization functions
print("Creating visualizations...")
radar_chart()
language_disparity()
category_heatmap()
language_comparison()
sensitivity_accuracy()
correlation_matrix()
category_difficulty()
model_category_performance()
agreement_analysis()
model_consistency()

print("All visualizations complete! Images saved in the 'visualizations' folder.")