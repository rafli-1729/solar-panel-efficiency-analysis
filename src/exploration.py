import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# To plot correlation coefficient for each feature with each others
def plot_numerical_correlation(X: pd.DataFrame, numerical_features: list) -> None:
    """
    Computes and visualizes the Pearson correlation matrix for numerical features
    using a heatmap.

    Args:
        X (pd.DataFrame): Input dataframe.
        numerical_features (list): List of numerical column names to correlate.
    """
    # Calculate the pairwise correlation of columns, excluding NA/null values
    correlation_matrix = X[numerical_features].corr(method='pearson')

    plt.figure(figsize=(15, 12))
    sns.heatmap(
        correlation_matrix,
        cmap='coolwarm',      # Diverging colormap (Red for pos, Blue for neg)
        vmin=-1, vmax=1,      # Anchor the colormap range
        center=0,             # Center the colormap at 0
        linewidths=.5,
        cbar_kws={'label': 'Correlation Coefficient'}
    )

    plt.title('Numerical Features Correlation Heatmap (Pearson)',
              fontsize=14, fontweight='bold')

    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.show()

def compare_similar_features(df, cols):
    """
    Membandingkan dua fitur dengan Boxplot berdampingan dan menghitung korelasi.
    Args:
        df (pd.DataFrame): Dataframe sumber.
        cols (list): List berisi dua nama kolom, e.g. ['tempC', 'Temperature']
    """
    # Validasi input biar gak error
    if len(cols) != 2:
        print("Error: List must be consist of two columns.")
        return

    col1, col2 = cols[0], cols[1]

    # Pearson Correlation
    corr_val = df[col1].corr(df[col2])

    # Plot Setup (1 Row, 2 Columns)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: First Boxplot
    sns.boxplot(y=df[col1], ax=axes[0], color='skyblue', width=0.4)
    axes[0].set_title(f'Distribusi: {col1}', fontsize=12, fontweight='bold')
    axes[0].grid(True, axis='y', alpha=0.3)

    # Plot 2: Second Boxplot
    sns.boxplot(y=df[col2], ax=axes[1], color='salmon', width=0.4)
    axes[1].set_title(f'Distribusi: {col2}', fontsize=12, fontweight='bold')
    axes[1].grid(True, axis='y', alpha=0.3)

    # Main title and correlation score
    plt.suptitle(f'Perbandingan Fitur: {col1} vs {col2}\nCorrelation Score: {corr_val:.4f}',
                 fontsize=16, y=1.05)

    plt.tight_layout()
    plt.show()


# To plot target
def check_transformations(series, bins=50):
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # --- Original Data ---
    sns.histplot(series, bins=bins, kde=True, ax=axes[0], color='skyblue', edgecolor='black')
    axes[0].set_title(f'Original\n(Skew: {series.skew():.2f})', fontsize=14)

    # --- Log Transformation ---
    series_log = np.log(series[series > 0])
    sns.histplot(series_log, bins=bins, kde=True, ax=axes[1], color='salmon', edgecolor='black')
    axes[1].set_title(f'Log Transform\n(Skew: {series_log.skew():.2f})', fontsize=14)

    # --- Square Root Transformation ---
    series_sqrt = np.sqrt(series)
    sns.histplot(series_sqrt, bins=bins, kde=True, ax=axes[2], color='lightgreen', edgecolor='black')
    axes[2].set_title(f'Square Root Transform\n(Skew: {series_sqrt.skew():.2f})', fontsize=14)

    plt.suptitle(f'{TARGET} Distribution', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.show()