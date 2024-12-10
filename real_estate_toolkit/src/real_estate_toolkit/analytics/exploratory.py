from typing import List, Dict
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
import os

class MarketAnalyzer:
    def __init__(self, data_path: str):
        """
        Initialize the analyzer with data from a CSV file.
        """
        # Treat 'NA' as null values
        self.real_state_data = pl.read_csv(data_path, null_values="NA")
        self.real_state_clean_data = None

    def clean_data(self) -> None:
        """
        Perform comprehensive data cleaning.
        """
        # Replace nulls in numeric columns with their respective mean
        numeric_columns = self.real_state_data.select(pl.col(pl.Float64, pl.Int64)).columns
        self.real_state_clean_data = self.real_state_data.with_columns(
            [
                pl.when(pl.col(col).is_null())
                .then(pl.col(col).mean())
                .otherwise(pl.col(col))
                .alias(col)
                for col in numeric_columns
            ]
        )

    def generate_price_distribution_analysis(self) -> pl.DataFrame:
        """
        Analyze sale price distribution using clean data.
        """
        stats = self.real_state_clean_data.select([
            pl.col("SalePrice").mean().alias("Mean"),
            pl.col("SalePrice").median().alias("Median"),
            pl.col("SalePrice").std().alias("StdDev"),
            pl.col("SalePrice").min().alias("Min"),
            pl.col("SalePrice").max().alias("Max")
        ])
        
        # Plotting the price distribution
        fig = px.histogram(
            self.real_state_clean_data.to_pandas(),
            x="SalePrice",
            title="Sale Price Distribution",
            nbins=50
        )
        
        output_path = "src/real_estate_toolkit/analytics/outputs/price_distribution.html"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.write_html(output_path)
        
        return stats

    def neighborhood_price_comparison(self) -> pl.DataFrame:
        """
        Create a boxplot comparing house prices across neighborhoods.
        """
        stats = self.real_state_clean_data.group_by("Neighborhood").agg([
            pl.col("SalePrice").mean().alias("MeanPrice"),
            pl.col("SalePrice").median().alias("MedianPrice")
        ])
        
        fig = px.box(
            self.real_state_clean_data.to_pandas(),
            x="Neighborhood",
            y="SalePrice",
            title="Neighborhood Price Comparison"
        )
        output_path = "src/real_estate_toolkit/analytics/outputs/neighborhood_prices.html"
        fig.write_html(output_path)
        
        return stats

    def feature_correlation_heatmap(self, variables: List[str]) -> None:
        """
        Generate a correlation heatmap for input variables.
        """
        correlation_matrix = self.real_state_clean_data.select(variables).corr()
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.to_numpy(),
            x=variables,
            y=variables,
            colorscale="Viridis"
        ))
        output_path = "src/real_estate_toolkit/analytics/outputs/correlation_heatmap.html"
        fig.write_html(output_path)

    def create_scatter_plots(self) -> Dict[str, go.Figure]:
        """
        Create scatter plots exploring feature relationships.
        """
        plots = {}
        relationships = [
            ("TotalSquareFootage", "SalePrice"),
            ("YearBuilt", "SalePrice"),
            ("OverallQuality", "SalePrice")
        ]
        for x, y in relationships:
            fig = px.scatter(
                self.real_state_clean_data.to_pandas(),
                x=x, y=y,
                title=f"{x} vs {y}",
                trendline="ols"
            )
            output_path = f"src/real_estate_toolkit/analytics/outputs/{x}_vs_{y}.html"
            fig.write_html(output_path)
            plots[f"{x}_vs_{y}"] = fig
        return plots
