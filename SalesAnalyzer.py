import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class SalesAnalyzer:
    
    def __init__(self, dataframe):
        self.df = dataframe.copy()
        self.setup_data()
    
    def setup_data(self):
        self.df['Date'] = pd.to_datetime(self.df['Date'], errors='coerce')
        self.df['Amount'] = pd.to_numeric(self.df['Amount'], errors='coerce')
        self.df['Qty'] = pd.to_numeric(self.df['Qty'], errors='coerce')
        self.df['Amount'].fillna(0, inplace=True)
        self.df['Qty'].fillna(0, inplace=True)
        
        self.df['Cost'] = self.df['Amount'] * 0.65
        self.df['Profit'] = self.df['Amount'] - self.df['Cost']
        self.df['Profit_Margin_%'] = np.where(self.df['Amount'] > 0, 
                                               (self.df['Profit'] / self.df['Amount']) * 100, 0)
    
    def monthly_revenue(self):
        monthly = self.df.groupby(['Year', 'Month', 'MonthName']).agg({
            'Amount': ['sum', 'mean', 'count'],
            'Qty': 'sum'
        }).reset_index()
        monthly.columns = ['Year', 'Month', 'MonthName', 'Total_Revenue', 
                          'Avg_Order_Value', 'Order_Count', 'Total_Qty']
        return monthly.sort_values(['Year', 'Month'])
    
    def region_sales(self, top_n=10):
        regions = self.df.groupby('ship-state').agg({
            'Amount': ['sum', 'mean', 'count'],
            'Qty': 'sum'
        }).reset_index()
        regions.columns = ['State', 'Total_Revenue', 'Avg_Order_Value', 
                          'Order_Count', 'Total_Qty']
        regions['Revenue_%'] = (regions['Total_Revenue'] / regions['Total_Revenue'].sum()) * 100
        return regions.sort_values('Total_Revenue', ascending=False).head(top_n)
    
    def average_order_value(self):
        completed = self.df[(self.df['Amount'] > 0) & 
                           (~self.df['Status'].str.contains('Cancel', case=False, na=False))]
        return completed['Amount'].mean()
    
    def profit_margin(self):
        return self.df[self.df['Amount'] > 0]['Profit_Margin_%'].mean()
    
    def total_profit(self):
        return self.df['Profit'].sum()
    
    def cancellation_rate(self):
        total = len(self.df)
        cancelled = self.df['Status'].str.contains('Cancel', case=False, na=False).sum()
        return (cancelled / total) * 100
    
    def fulfillment_rate(self):
        total = len(self.df)
        shipped = self.df['Status'].str.contains('Shipped', case=False, na=False).sum()
        return (shipped / total) * 100
    
    def top_categories(self, top_n=10):
        cats = self.df.groupby('Category').agg({
            'Qty': 'sum',
            'Amount': ['sum', 'mean'],
            'Order ID': 'count'
        }).reset_index()
        cats.columns = ['Category', 'Total_Qty', 'Total_Revenue', 'AOV', 'Orders']
        return cats.sort_values('Total_Revenue', ascending=False).head(top_n)
    
    def b2b_vs_b2c_sales(self):
        b2b = self.df.groupby('B2B').agg({
            'Amount': ['sum', 'mean', 'count'],
            'Qty': 'sum'
        }).reset_index()
        b2b.columns = ['B2B', 'Total_Revenue', 'AOV', 'Orders', 'Total_Qty']
        b2b['B2B'] = b2b['B2B'].map({True: 'B2B', False: 'B2C'})
        b2b['Revenue_%'] = (b2b['Total_Revenue'] / b2b['Total_Revenue'].sum()) * 100
        return b2b
    
    def get_all_kpis(self):
        kpis = {
            'Total Revenue': f"₹{self.df['Amount'].sum():,.2f}",
            'Total Orders': f"{len(self.df):,}",
            'Total Quantity Sold': f"{int(self.df['Qty'].sum()):,}",
            'Average Order Value': f"₹{self.average_order_value():.2f}",
            'Average Profit Margin': f"{self.profit_margin():.2f}%",
            'Total Profit': f"₹{self.total_profit():,.2f}",
            'Cancellation Rate': f"{self.cancellation_rate():.2f}%",
            'Fulfillment Rate': f"{self.fulfillment_rate():.2f}%",
            'Number of States': f"{self.df['ship-state'].nunique()}",
            'Number of Categories': f"{self.df['Category'].nunique()}"
        }
        return kpis
    
    def plot_monthly_revenue(self):
        plt.figure(figsize=(12, 5))
        monthly_rev = self.df.groupby(['Year', 'Month'])['Amount'].sum()
        monthly_rev.plot(kind='line', marker='o', color='#2E86C1', linewidth=2)
        plt.title('Monthly Revenue Trend', fontsize=14, fontweight='bold')
        plt.xlabel('Year - Month')
        plt.ylabel('Revenue (INR)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_region_sales(self, top_n=10):
        plt.figure(figsize=(11, 6))
        top_regions = self.df.groupby('ship-state')['Amount'].sum().sort_values(ascending=False).head(top_n)
        top_regions.plot(kind='bar', color='#28B463')
        plt.title(f'Top {top_n} States by Sales', fontsize=14, fontweight='bold')
        plt.xlabel('State')
        plt.ylabel('Revenue (INR)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    
    def plot_categories(self, top_n=8):
        plt.figure(figsize=(10, 6))
        cats = self.df.groupby('Category')['Amount'].sum().sort_values(ascending=True).tail(top_n)
        cats.plot(kind='barh', color='#E74C3C')
        plt.title(f'Top {top_n} Categories by Revenue', fontsize=14, fontweight='bold')
        plt.xlabel('Revenue (INR)')
        plt.ylabel('Category')
        plt.tight_layout()
        plt.show()
    
    def plot_outliers(self):
        plt.figure(figsize=(8, 6))
        positive_amounts = self.df[self.df['Amount'] > 0]
        plt.boxplot(positive_amounts['Amount'], vert=True, patch_artist=True,
                   boxprops=dict(facecolor='lightblue'))
        plt.title('Sales Amount - Outlier Detection', fontsize=14, fontweight='bold')
        plt.ylabel('Amount (INR)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_order_status(self):
        plt.figure(figsize=(9, 6))
        status = self.df['Status'].value_counts()
        plt.pie(status, labels=status.index, autopct='%1.1f%%', startangle=90)
        plt.title('Order Status Distribution', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_b2b_comparison(self):
        plt.figure(figsize=(8, 6))
        b2b_rev = self.df.groupby('B2B')['Amount'].sum()
        labels = ['B2C', 'B2B']
        plt.bar(labels, b2b_rev.values, color=['#3498DB', '#E67E22'])
        plt.title('B2B vs B2C Sales', fontsize=14, fontweight='bold')
        plt.ylabel('Revenue (INR)')
        plt.tight_layout()
        plt.show()
    
    def generate_report(self):
        print("\n" + "="*70)
        print("SALES ANALYSIS REPORT")
        print("="*70 + "\n")
        
        kpis = self.get_all_kpis()
        for key, value in kpis.items():
            print(f"{key:.<45} {value}")
        
        print("\n" + "-"*70)
        print("TOP 5 STATES:")
        print("-"*70)
        print(self.region_sales(5).to_string(index=False))
        
        print("\n" + "-"*70)
        print("TOP 5 CATEGORIES:")
        print("-"*70)
        print(self.top_categories(5).to_string(index=False))
        
        print("\n" + "-"*70)
        print("B2B vs B2C:")
        print("-"*70)
        print(self.b2b_vs_b2c_sales().to_string(index=False))
        
        print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    df = pd.read_csv('Cleaned_Amazon_Sale_Report.csv')
    
    analyzer = SalesAnalyzer(df)
    
    analyzer.generate_report()
    
    print("Generating visualizations...\n")
    analyzer.plot_monthly_revenue()
    analyzer.plot_region_sales()
    analyzer.plot_categories()
    analyzer.plot_outliers()
    analyzer.plot_order_status()
    analyzer.plot_b2b_comparison()
    
    print("Analysis complete!")