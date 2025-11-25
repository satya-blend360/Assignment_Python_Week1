class SalesAnalyzer:
    """
    A class to analyze retail sales data and compute key performance indicators.
    """
    
    def __init__(self, dataframe):
        """
        Initialize the SalesAnalyzer with a pandas DataFrame.
        
        Parameters:
        -----------
        dataframe : pd.DataFrame
            Cleaned sales data
        """
        self.df = dataframe.copy()
        self._preprocess()
    
    def _preprocess(self):
        """Private method to preprocess data."""
        # Ensure Date is datetime
        self.df['Date'] = pd.to_datetime(self.df['Date'], errors='coerce')
        
        # Ensure Amount is numeric
        self.df['Amount'] = pd.to_numeric(self.df['Amount'], errors='coerce')
        
        # Add profit calculations
        self.df['Cost'] = self.df['Amount'] * 0.65
        self.df['Profit'] = self.df['Amount'] - self.df['Cost']
        self.df['Profit_Margin_%'] = (self.df['Profit'] / self.df['Amount']) * 100
        self.df['Profit_Margin_%'].fillna(0, inplace=True)
    
    def monthly_revenue(self):
        """Calculate monthly revenue."""
        return self.df.groupby(['Year', 'Month'])['Amount'].sum().reset_index()
    
    def region_sales(self, top_n=10):
        """
        Calculate region-wise sales.
        
        Parameters:
        -----------
        top_n : int
            Number of top regions to return
        """
        return self.df.groupby('ship-state')['Amount'].sum().sort_values(
            ascending=False).head(top_n).reset_index()
    
    def average_order_value(self):
        """Calculate average order value."""
        return self.df[self.df['Amount'] > 0]['Amount'].mean()
    
    def profit_margin(self):
        """Calculate average profit margin percentage."""
        return self.df[self.df['Amount'] > 0]['Profit_Margin_%'].mean()
    
    def cancellation_rate(self):
        """Calculate order cancellation rate."""
        total = len(self.df)
        cancelled = self.df['Status'].str.contains('Cancel', case=False, na=False).sum()
        return (cancelled / total) * 100
    
    def top_categories(self, top_n=10):
        """Get top selling categories by quantity."""
        return self.df.groupby('Category')['Qty'].sum().sort_values(
            ascending=False).head(top_n)
    
    def b2b_vs_b2c_sales(self):
        """Compare B2B and B2C sales."""
        return self.df.groupby('B2B')['Amount'].sum()
    
    def get_all_kpis(self):
        """Get all KPIs in a dictionary."""
        return {
            'Average Order Value': f"₹{self.average_order_value():.2f}",
            'Average Profit Margin': f"{self.profit_margin():.2f}%",
            'Cancellation Rate': f"{self.cancellation_rate():.2f}%",
            'Total Revenue': f"₹{self.df['Amount'].sum():.2f}",
            'Total Orders': len(self.df),
            'Total Quantity Sold': self.df['Qty'].sum()
        }
    
    def plot_monthly_revenue(self):
        """Plot monthly revenue trend."""
        plt.figure(figsize=(14, 6))
        monthly_rev = self.df.groupby(['Year', 'Month'])['Amount'].sum()
        monthly_rev.plot(kind='line', marker='o', color='#2E86C1', linewidth=2)
        plt.title('Monthly Revenue Trend', fontsize=16, fontweight='bold')
        plt.xlabel('Year - Month', fontsize=12)
        plt.ylabel('Revenue (INR)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_region_sales(self, top_n=10):
        """Plot top regions by sales."""
        plt.figure(figsize=(12, 6))
        top_regions = self.df.groupby('ship-state')['Amount'].sum().sort_values(
            ascending=False).head(top_n)
        top_regions.plot(kind='bar', color='#28B463')
        plt.title(f'Top {top_n} States by Sales Revenue', fontsize=16, fontweight='bold')
        plt.xlabel('State', fontsize=12)
        plt.ylabel('Revenue (INR)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    
    def plot_outliers(self):
        """Plot boxplot to identify outliers in sales amount."""
        plt.figure(figsize=(8, 6))
        df_positive = self.df[self.df['Amount'] > 0]
        plt.boxplot(df_positive['Amount'], vert=True, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', color='blue'),
                    medianprops=dict(color='red', linewidth=2))
        plt.title('Sales Amount - Outlier Detection', fontsize=16, fontweight='bold')
        plt.ylabel('Amount (INR)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def generate_report(self):
        """Generate a comprehensive analysis report."""
        print("="*60)
        print("SALES ANALYSIS REPORT")
        print("="*60)
        
        kpis = self.get_all_kpis()
        for key, value in kpis.items():
            print(f"{key}: {value}")
        
        print("\n" + "="*60)
        print("TOP 5 STATES BY REVENUE:")
        print("="*60)
        print(self.region_sales(5))
        
        print("\n" + "="*60)
        print("TOP 5 CATEGORIES BY QUANTITY:")
        print("="*60)
        print(self.top_categories(5))
        
        