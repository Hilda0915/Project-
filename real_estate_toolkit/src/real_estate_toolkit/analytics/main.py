import sys
from pathlib import Path

# Add the `src` directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from real_estate_toolkit.analytics.exploratory import MarketAnalyzer


def main():
    # Path to the dataset
    dataset_path = "src/real_estate_toolkit/data/data_files/train.csv"
    
    # Initialize MarketAnalyzer
    analyzer = MarketAnalyzer(data_path=dataset_path)

    # Step 1: Clean Data
    analyzer.clean_data()
    print("Data cleaned successfully!")

    # Step 2: Generate Price Distribution Analysis
    print("Generating price distribution analysis...")
    price_stats = analyzer.generate_price_distribution_analysis()
    print("Price Distribution Analysis Results:")
    print(f"Mean SalePrice: {price_stats['Mean'][0]}")
    print(f"Median SalePrice: {price_stats['Median'][0]}")
    print(f"StdDev of SalePrice: {price_stats['StdDev'][0]}")
    print(f"Min SalePrice: {price_stats['Min'][0]}")
    print(f"Max SalePrice: {price_stats['Max'][0]}")

    # Step 3: Neighborhood Price Comparison
    print("Generating neighborhood price comparison...")
    neighborhood_stats = analyzer.neighborhood_price_comparison()
    print("Neighborhood Price Comparison Results:")
    print(neighborhood_stats)

    # Step 4: Feature Correlation Heatmap
    print("Generating feature correlation heatmap...")
    numerical_features = ["SalePrice", "LotArea", "YearBuilt", "GrLivArea"]  # Example features
    analyzer.feature_correlation_heatmap(variables=numerical_features)
    print("Feature correlation heatmap saved.")

    # Step 5: Create Scatter Plots
    print("Creating scatter plots...")
    scatter_plots = analyzer.create_scatter_plots()
    print(f"Scatter plots created: {list(scatter_plots.keys())}")
    for key, plot in scatter_plots.items():
        print(f"Plot saved as: {key}.html")

if __name__ == "__main__":
    main()
