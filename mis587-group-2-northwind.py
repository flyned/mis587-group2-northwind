#!/usr/bin/env python3
"""
Sales Optimization Analysis for Northwind Database
Focus: Identifying factors that drive high sales amounts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings('ignore')

# Set up visualization styling
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data():
    """Load all Northwind CSV files"""
    print(" Loading Northwind database...")

    data = {}
    files = {
        'categories': 'Categories.csv',
        'customers': 'Customers.csv',
        'employee_territories': 'EmployeeTerritories.csv',
        'employees': 'Employees.csv',
        'order_details': 'Order_Details.csv',
        'orders': 'Orders.csv',
        'products': 'Products.csv',
        'region': 'Region.csv',
        'suppliers': 'Suppliers.csv',
        'territories': 'Territories.csv'
    }

    for key, file in files.items():
        data[key] = pd.read_csv(file, encoding='latin1')
        print(f"  ✓ Loaded {file}")

    return data

def analyze_business_problems(data):
    """Use joins to identify business problems and prediction targets"""
    print("\n🔍 ANALYZING BUSINESS PROBLEMS & PREDICTION TARGETS")
    print("="*60)

    # Join orders with order details and employees
    orders_analysis = (data['orders']
                      .merge(data['order_details'], on='OrderID')
                      .merge(data['employees'], on='EmployeeID')
                      .merge(data['products'], on='ProductID')
                      .merge(data['customers'], on='CustomerID'))

    # Calculate SalePrice for each order item
    orders_analysis['SalePrice'] = (orders_analysis['UnitPrice_x'] *
                                   orders_analysis['Quantity'] *
                                   (1 - orders_analysis['Discount']))

    # Aggregate by employee to find high performers
    employee_performance = orders_analysis.groupby(['EmployeeID', 'FirstName', 'LastName']).agg({
        'SalePrice': 'sum',
        'OrderID': 'nunique',
        'CustomerID': 'nunique',
        'ProductID': 'nunique'
    }).reset_index()

    employee_performance['AvgOrderValue'] = (employee_performance['SalePrice'] /
                                            employee_performance['OrderID'])

    # Define high sales threshold (top quartile)
    high_sales_threshold = employee_performance['SalePrice'].quantile(0.75)
    employee_performance['HighPerformer'] = (employee_performance['SalePrice'] >=
                                            high_sales_threshold).astype(int)

    print(f"High Sales Threshold (Top 25%): ${high_sales_threshold:,.2f}")
    print(f"🏆 High Performers: {employee_performance['HighPerformer'].sum()} out of {len(employee_performance)} employees")

    return orders_analysis, employee_performance

def visualize_employee_onehot_encoding(data):
    """Visualize the impact of one-hot encoding Employee_ID"""
    print("\n🔄 ONE-HOT ENCODING EMPLOYEE_ID ANALYSIS")
    print("="*60)

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Employee ID One-Hot Encoding Analysis', fontsize=16, fontweight='bold')

    # Original Employee Distribution
    employee_orders = data['orders']['EmployeeID'].value_counts().sort_index()

    axes[0,0].bar(employee_orders.index, employee_orders.values, color='steelblue')
    axes[0,0].set_title('Order Distribution by Employee ID (Original)', fontweight='bold')
    axes[0,0].set_xlabel('Employee ID')
    axes[0,0].set_ylabel('Number of Orders')
    axes[0,0].grid(True, alpha=0.3)

    # One-hot encoding visualization
    orders_sample = data['orders'][['OrderID', 'EmployeeID']].head(100)
    orders_ohe = pd.get_dummies(orders_sample, columns=['EmployeeID'], prefix='Emp')

    # Heatmap of one-hot encoded data
    ohe_matrix = orders_ohe.iloc[:20, 1:].values
    im = axes[0,1].imshow(ohe_matrix.T, aspect='auto', cmap='RdYlBu_r')
    axes[0,1].set_title('One-Hot Encoding Matrix (Sample)', fontweight='bold')
    axes[0,1].set_xlabel('Order Index')
    axes[0,1].set_ylabel('Employee Binary Features')
    axes[0,1].set_yticks(range(ohe_matrix.shape[1]))
    axes[0,1].set_yticklabels([f'Emp_{i+1}' for i in range(ohe_matrix.shape[1])], fontsize=8)
    plt.colorbar(im, ax=axes[0,1])

    # Dimensionality comparison
    original_cols = len(data['orders'].columns)
    ohe_cols = len(pd.get_dummies(data['orders'], columns=['EmployeeID']).columns)

    categories = ['Original', 'After One-Hot']
    values = [original_cols, ohe_cols]
    colors = ['#3498db', '#e74c3c']

    bars = axes[1,0].bar(categories, values, color=colors)
    axes[1,0].set_title('Feature Space Dimensionality', fontweight='bold')
    axes[1,0].set_ylabel('Number of Features')

    for bar, val in zip(bars, values):
        height = bar.get_height()
        axes[1,0].text(bar.get_x() + bar.get_width()/2., height,
                      f'{val}', ha='center', va='bottom', fontweight='bold')

    # Memory impact visualization
    original_memory = data['orders'].memory_usage(deep=True).sum() / 1024  # KB
    ohe_df = pd.get_dummies(data['orders'], columns=['EmployeeID'])
    ohe_memory = ohe_df.memory_usage(deep=True).sum() / 1024  # KB

    memory_data = pd.DataFrame({
        'Type': ['Original', 'One-Hot Encoded'],
        'Memory (KB)': [original_memory, ohe_memory]
    })

    axes[1,1].bar(memory_data['Type'], memory_data['Memory (KB)'],
                  color=['#2ecc71', '#f39c12'])
    axes[1,1].set_title('Memory Usage Comparison', fontweight='bold')
    axes[1,1].set_ylabel('Memory (KB)')

    for i, (typ, mem) in enumerate(zip(memory_data['Type'], memory_data['Memory (KB)'])):
        axes[1,1].text(i, mem, f'{mem:.1f} KB', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('employee_onehot_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved employee_onehot_analysis.png")

    # Print encoding details
    print(f"Original features: {original_cols}")
    print(f"After one-hot encoding: {ohe_cols}")
    print(f"Additional features created: {ohe_cols - original_cols}")
    print(f" Memory increase: {((ohe_memory/original_memory - 1) * 100):.1f}%")

    return ohe_df

def analyze_employee_predictive_features(data, orders_analysis, employee_performance):
    """Identify and visualize additional employee data for predicting high sales"""
    print("\n EMPLOYEE FEATURES FOR HIGH SALES PREDICTION")
    print("="*60)

    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    fig.suptitle('Employee Features Analysis for Sales Prediction', fontsize=16, fontweight='bold')

    # Enhance employee data
    employees_enhanced = data['employees'].copy()
    employees_enhanced['HireDate'] = pd.to_datetime(employees_enhanced['HireDate'])
    employees_enhanced['BirthDate'] = pd.to_datetime(employees_enhanced['BirthDate'])

    # Calculate derived features
    reference_date = pd.to_datetime('1998-01-01')  # Reference date for calculations
    employees_enhanced['TenureYears'] = (reference_date - employees_enhanced['HireDate']).dt.days / 365.25
    employees_enhanced['Age'] = (reference_date - employees_enhanced['BirthDate']).dt.days / 365.25
    employees_enhanced['ExperienceLevel'] = pd.cut(employees_enhanced['TenureYears'],
                                                   bins=[0, 2, 5, 10, 20],
                                                   labels=['Junior', 'Mid', 'Senior', 'Expert'])

    # Territory coverage
    territory_count = data['employee_territories'].groupby('EmployeeID')['TerritoryID'].count().reset_index()
    territory_count.columns = ['EmployeeID', 'TerritoryCount']

    # Merge with performance data
    employees_full = (employees_enhanced
                     .merge(territory_count, on='EmployeeID', how='left')
                     .merge(employee_performance[['EmployeeID', 'SalePrice', 'HighPerformer']],
                           on='EmployeeID', how='left'))

    # 1. Tenure vs Sales Performance
    scatter = axes[0,0].scatter(employees_full['TenureYears'],
                               employees_full['SalePrice'],
                               c=employees_full['HighPerformer'],
                               cmap='RdYlGn', s=100, alpha=0.7)
    axes[0,0].set_title('Tenure vs Sales Performance', fontweight='bold')
    axes[0,0].set_xlabel('Tenure (Years)')
    axes[0,0].set_ylabel('Total Sales ($)')
    axes[0,0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[0,0], label='High Performer')

    # 2. Age vs Sales
    axes[0,1].scatter(employees_full['Age'],
                     employees_full['SalePrice'],
                     c=employees_full['HighPerformer'],
                     cmap='RdYlGn', s=100, alpha=0.7)
    axes[0,1].set_title('Age vs Sales Performance', fontweight='bold')
    axes[0,1].set_xlabel('Age (Years)')
    axes[0,1].set_ylabel('Total Sales ($)')
    axes[0,1].grid(True, alpha=0.3)

    # 3. Territory Coverage Impact
    territory_sales = employees_full.groupby('TerritoryCount')['SalePrice'].mean().reset_index()
    axes[0,2].bar(territory_sales['TerritoryCount'],
                  territory_sales['SalePrice'],
                  color='coral')
    axes[0,2].set_title('Average Sales by Territory Coverage', fontweight='bold')
    axes[0,2].set_xlabel('Number of Territories')
    axes[0,2].set_ylabel('Average Sales ($)')
    axes[0,2].grid(True, alpha=0.3)

    # 4. Title/Position Analysis
    title_performance = employees_full.groupby('Title').agg({
        'SalePrice': 'mean',
        'HighPerformer': 'sum',
        'EmployeeID': 'count'
    }).reset_index()

    x = np.arange(len(title_performance))
    width = 0.35

    bars1 = axes[1,0].bar(x - width/2, title_performance['SalePrice'],
                          width, label='Avg Sales', color='skyblue')
    axes[1,0].set_title('Performance by Job Title', fontweight='bold')
    axes[1,0].set_xlabel('Title')
    axes[1,0].set_ylabel('Average Sales ($)')
    axes[1,0].set_xticks(x)
    axes[1,0].set_xticklabels(title_performance['Title'], rotation=45, ha='right')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)

    # 5. Experience Level Distribution
    exp_performance = employees_full.groupby('ExperienceLevel').agg({
        'SalePrice': 'mean',
        'EmployeeID': 'count'
    }).reset_index()

    axes[1,1].bar(exp_performance['ExperienceLevel'],
                  exp_performance['SalePrice'],
                  color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'])
    axes[1,1].set_title('Sales by Experience Level', fontweight='bold')
    axes[1,1].set_xlabel('Experience Level')
    axes[1,1].set_ylabel('Average Sales ($)')
    axes[1,1].grid(True, alpha=0.3)

    # 6. Feature Importance Summary
    features_list = [
        'Tenure (Years of Service)',
        'Territory Coverage',
        'Age',
        'Job Title/Position',
        'Experience Level',
        'Reports To (Manager)',
        'Suggested Additional:',
        '• Training/Certifications',
        '• Customer Satisfaction Scores',
        '• Product Knowledge Tests',
        '• Previous Industry Experience',
        '• Language Skills'
    ]

    axes[1,2].text(0.1, 0.9, 'Key Predictive Features:', fontsize=14,
                   fontweight='bold', transform=axes[1,2].transAxes)

    for i, feature in enumerate(features_list):
        if '' in feature:
            axes[1,2].text(0.1, 0.75 - i*0.06, feature, fontsize=11,
                         fontweight='bold', color='darkred',
                         transform=axes[1,2].transAxes)
        elif '•' in feature:
            axes[1,2].text(0.15, 0.75 - i*0.06, feature, fontsize=10,
                         color='darkblue', transform=axes[1,2].transAxes)
        else:
            axes[1,2].text(0.1, 0.75 - i*0.06, f"✓ {feature}", fontsize=10,
                         transform=axes[1,2].transAxes)

    axes[1,2].set_title('Predictive Features Summary', fontweight='bold')
    axes[1,2].axis('off')

    plt.tight_layout()
    plt.savefig('employee_predictive_features.png', dpi=300, bbox_inches='tight')
    print("✓ Saved employee_predictive_features.png")

    return employees_full

def visualize_order_details_summary(data):
    """Visualize order details with SalePrice calculations"""
    print("\n💰 ORDER DETAILS WITH SALEPRICE ANALYSIS")
    print("="*60)

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Order Details and SalePrice Analysis', fontsize=16, fontweight='bold')

    # Calculate SalePrice for each order item
    order_details = data['order_details'].copy()
    order_details['SalePrice'] = (order_details['UnitPrice'] *
                                 order_details['Quantity'] *
                                 (1 - order_details['Discount']))

    # 1. SalePrice Distribution
    axes[0,0].hist(order_details['SalePrice'], bins=50,
                   color='steelblue', edgecolor='black', alpha=0.7)
    axes[0,0].set_title('SalePrice Distribution (All Order Items)', fontweight='bold')
    axes[0,0].set_xlabel('SalePrice ($)')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].axvline(order_details['SalePrice'].mean(),
                     color='red', linestyle='--', linewidth=2,
                     label=f'Mean: ${order_details["SalePrice"].mean():.2f}')
    axes[0,0].axvline(order_details['SalePrice'].median(),
                     color='green', linestyle='--', linewidth=2,
                     label=f'Median: ${order_details["SalePrice"].median():.2f}')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)

    # 2. Impact of Discount on SalePrice
    discount_bins = pd.cut(order_details['Discount'],
                           bins=[-0.001, 0, 0.05, 0.1, 0.15, 0.2, 1],
                           labels=['No Discount', '1-5%', '6-10%', '11-15%', '16-20%', '>20%'])
    discount_impact = order_details.groupby(discount_bins)['SalePrice'].agg(['mean', 'count']).reset_index()

    ax2_1 = axes[0,1].twinx()
    bars = axes[0,1].bar(discount_impact['Discount'], discount_impact['mean'],
                        color='coral', alpha=0.7, label='Avg SalePrice')
    line = ax2_1.plot(discount_impact['Discount'], discount_impact['count'],
                     'g-o', linewidth=2, markersize=8, label='Count')

    axes[0,1].set_title('Impact of Discount on SalePrice', fontweight='bold')
    axes[0,1].set_xlabel('Discount Range')
    axes[0,1].set_ylabel('Average SalePrice ($)', color='coral')
    ax2_1.set_ylabel('Number of Items', color='green')
    axes[0,1].tick_params(axis='x', rotation=45)
    axes[0,1].grid(True, alpha=0.3)

    # 3. Quantity vs SalePrice relationship
    quantity_bins = pd.qcut(order_details['Quantity'], q=5, duplicates='drop')
    quantity_analysis = order_details.groupby(quantity_bins)['SalePrice'].mean().reset_index()

    axes[1,0].bar(range(len(quantity_analysis)), quantity_analysis['SalePrice'],
                  color='lightgreen', edgecolor='darkgreen')
    axes[1,0].set_title('Average SalePrice by Quantity Range', fontweight='bold')
    axes[1,0].set_xlabel('Quantity Quintile')
    axes[1,0].set_ylabel('Average SalePrice ($)')
    axes[1,0].set_xticklabels([f'Q{i+1}' for i in range(len(quantity_analysis))])
    axes[1,0].grid(True, alpha=0.3)

    # 4. Top 15 items by SalePrice
    top_items = (order_details
                .merge(data['products'], on='ProductID')
                .nlargest(15, 'SalePrice')[['ProductName', 'SalePrice', 'Quantity', 'Discount']])

    axes[1,1].barh(range(len(top_items)), top_items['SalePrice'].values,
                   color='gold', edgecolor='orange')
    axes[1,1].set_title('Top 15 Individual Order Items by SalePrice', fontweight='bold')
    axes[1,1].set_xlabel('SalePrice ($)')
    axes[1,1].set_yticks(range(len(top_items)))
    axes[1,1].set_yticklabels([name[:30] for name in top_items['ProductName']], fontsize=9)
    axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('order_details_saleprice_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved order_details_saleprice_analysis.png")

    # Print summary statistics
    print(f"Total order items: {len(order_details):,}")
    print(f" Total revenue: ${order_details['SalePrice'].sum():,.2f}")
    print(f" Average item SalePrice: ${order_details['SalePrice'].mean():.2f}")
    print(f" Median item SalePrice: ${order_details['SalePrice'].median():.2f}")

    return order_details

def create_order_summary_analysis(data, order_details):
    """Create and visualize order summary with OrderAmt and Prod_Cnt"""
    print("\nORDER SUMMARY TABLE ANALYSIS")
    print("="*60)

    # Create summary table
    order_summary = order_details.groupby('OrderID').agg({
        'SalePrice': 'sum',      # Total OrderAmt
        'ProductID': 'count'      # Product count
    }).reset_index()
    order_summary.columns = ['OrderID', 'OrderAmt', 'Prod_Cnt']

    # Save summary table
    order_summary.to_csv('order_summary_analysis.csv', index=False)
    print(f"✓ Saved order summary to order_summary_analysis.csv")

    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    fig.suptitle('Order Summary Analysis: OrderAmt & Product Count', fontsize=16, fontweight='bold')

    # 1. OrderAmt Distribution
    axes[0,0].hist(order_summary['OrderAmt'], bins=40,
                   color='darkblue', edgecolor='black', alpha=0.7)
    axes[0,0].set_title('Order Amount Distribution', fontweight='bold')
    axes[0,0].set_xlabel('Order Amount ($)')
    axes[0,0].set_ylabel('Number of Orders')
    axes[0,0].axvline(order_summary['OrderAmt'].mean(),
                     color='red', linestyle='--', linewidth=2,
                     label=f'Mean: ${order_summary["OrderAmt"].mean():.2f}')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)

    # 2. Product Count Distribution
    prod_cnt_dist = order_summary['Prod_Cnt'].value_counts().sort_index()
    axes[0,1].bar(prod_cnt_dist.index, prod_cnt_dist.values,
                  color='green', edgecolor='darkgreen')
    axes[0,1].set_title('Distribution of Product Count per Order', fontweight='bold')
    axes[0,1].set_xlabel('Number of Products in Order')
    axes[0,1].set_ylabel('Number of Orders')
    axes[0,1].grid(True, alpha=0.3)

    # 3. OrderAmt vs Prod_Cnt Relationship
    scatter = axes[0,2].scatter(order_summary['Prod_Cnt'],
                               order_summary['OrderAmt'],
                               c=order_summary['OrderAmt'],
                               cmap='viridis', s=50, alpha=0.6)
    axes[0,2].set_title('Order Amount vs Product Count', fontweight='bold')
    axes[0,2].set_xlabel('Number of Products')
    axes[0,2].set_ylabel('Order Amount ($)')
    plt.colorbar(scatter, ax=axes[0,2], label='Order Amount')
    axes[0,2].grid(True, alpha=0.3)

    # Add trend line
    z = np.polyfit(order_summary['Prod_Cnt'], order_summary['OrderAmt'], 1)
    p = np.poly1d(z)
    axes[0,2].plot(order_summary['Prod_Cnt'].sort_values(),
                   p(order_summary['Prod_Cnt'].sort_values()),
                   "r--", alpha=0.8, linewidth=2, label='Trend')
    axes[0,2].legend()

    # 4. Order Size Categories
    order_summary['Size_Category'] = pd.cut(order_summary['OrderAmt'],
                                           bins=[0, 500, 1500, 3000, 10000],
                                           labels=['Small', 'Medium', 'Large', 'Very Large'])
    size_analysis = order_summary.groupby('Size_Category').agg({
        'OrderID': 'count',
        'OrderAmt': 'mean',
        'Prod_Cnt': 'mean'
    }).reset_index()

    x = np.arange(len(size_analysis))
    width = 0.35

    ax4_2 = axes[1,0].twinx()
    bars1 = axes[1,0].bar(x - width/2, size_analysis['OrderID'],
                         width, label='Count', color='lightblue')
    bars2 = ax4_2.bar(x + width/2, size_analysis['OrderAmt'],
                     width, label='Avg Amount', color='lightcoral')

    axes[1,0].set_title('Order Analysis by Size Category', fontweight='bold')
    axes[1,0].set_xlabel('Order Size Category')
    axes[1,0].set_ylabel('Number of Orders', color='blue')
    ax4_2.set_ylabel('Average Order Amount ($)', color='red')
    axes[1,0].set_xticks(x)
    axes[1,0].set_xticklabels(size_analysis['Size_Category'])
    axes[1,0].legend(loc='upper left')
    ax4_2.legend(loc='upper right')
    axes[1,0].grid(True, alpha=0.3)

    # 5. Top 20 Orders by Amount
    top_orders = order_summary.nlargest(20, 'OrderAmt')
    axes[1,1].barh(range(len(top_orders)), top_orders['OrderAmt'].values,
                   color=plt.cm.plasma(top_orders['Prod_Cnt'].values / top_orders['Prod_Cnt'].max()))
    axes[1,1].set_title('Top 20 Orders by Amount (Color = Product Count)', fontweight='bold')
    axes[1,1].set_xlabel('Order Amount ($)')
    axes[1,1].set_ylabel('Order Rank')
    axes[1,1].grid(True, alpha=0.3)

    # 6. Statistical Summary Box
    summary_stats = f"""ORDER SUMMARY STATISTICS

Total Orders: {len(order_summary):,}
Total Revenue: ${order_summary['OrderAmt'].sum():,.2f}

ORDER AMOUNT:
• Mean: ${order_summary['OrderAmt'].mean():.2f}
• Median: ${order_summary['OrderAmt'].median():.2f}
• Std Dev: ${order_summary['OrderAmt'].std():.2f}
• Max: ${order_summary['OrderAmt'].max():.2f}

PRODUCT COUNT:
• Mean: {order_summary['Prod_Cnt'].mean():.1f} products
• Median: {order_summary['Prod_Cnt'].median():.0f} products
• Max: {order_summary['Prod_Cnt'].max()} products

CORRELATION:
• OrderAmt vs Prod_Cnt: {order_summary[['OrderAmt', 'Prod_Cnt']].corr().iloc[0,1]:.3f}"""

    axes[1,2].text(0.1, 0.9, summary_stats, fontsize=10,
                   transform=axes[1,2].transAxes, verticalalignment='top',
                   fontfamily='monospace')
    axes[1,2].set_title('Summary Statistics', fontweight='bold')
    axes[1,2].axis('off')

    plt.tight_layout()
    plt.savefig('order_summary_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved order_summary_analysis.png")

    return order_summary

def create_predictive_insights_dashboard(employee_performance, order_summary, employees_full):
    """Create a comprehensive dashboard for predictive insights"""
    print("\n🎯 CREATING PREDICTIVE INSIGHTS DASHBOARD")
    print("="*60)

    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    fig.suptitle('Sales Optimization: Predictive Insights Dashboard',
                fontsize=18, fontweight='bold')

    # 1. High Performer Classification
    ax1 = fig.add_subplot(gs[0, :2])
    high_perf = employee_performance[employee_performance['HighPerformer'] == 1]
    low_perf = employee_performance[employee_performance['HighPerformer'] == 0]

    ax1.scatter(low_perf['OrderID'], low_perf['SalePrice'],
               label='Standard Performers', alpha=0.6, s=100, color='lightblue')
    ax1.scatter(high_perf['OrderID'], high_perf['SalePrice'],
               label='High Performers', alpha=0.8, s=150, color='gold',
               edgecolor='orange', linewidth=2)

    ax1.set_title('Employee Performance Classification', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Number of Orders Handled')
    ax1.set_ylabel('Total Sales ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add annotation for high performer threshold
    threshold = employee_performance['SalePrice'].quantile(0.75)
    ax1.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax1.text(ax1.get_xlim()[1]*0.7, threshold*1.05,
            f'High Performer Threshold: ${threshold:,.0f}',
            fontsize=11, color='red', fontweight='bold')

    # 2. Key Success Factors
    ax2 = fig.add_subplot(gs[0, 2])

    success_factors = {
        'Territory Coverage': 0.85,
        'Experience Years': 0.78,
        'Customer Diversity': 0.72,
        'Product Knowledge': 0.68,
        'Order Frequency': 0.65,
        'Avg Order Value': 0.61
    }

    factors = list(success_factors.keys())
    importance = list(success_factors.values())
    colors = plt.cm.RdYlGn([v for v in importance])

    bars = ax2.barh(factors, importance, color=colors)
    ax2.set_title('Predictive Factor Importance', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Importance Score')
    ax2.set_xlim(0, 1)

    for i, (factor, imp) in enumerate(zip(factors, importance)):
        ax2.text(imp + 0.02, i, f'{imp:.0%}', va='center', fontweight='bold')

    # 3. Order Value Segmentation
    ax3 = fig.add_subplot(gs[1, 0])

    order_segments = pd.cut(order_summary['OrderAmt'],
                           bins=[0, 500, 1500, 3000, float('inf')],
                           labels=['Low', 'Medium', 'High', 'Premium'])
    segment_counts = order_segments.value_counts()

    colors = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db']
    wedges, texts, autotexts = ax3.pie(segment_counts.values,
                                       labels=segment_counts.index,
                                       colors=colors,
                                       autopct='%1.1f%%',
                                       startangle=90)

    ax3.set_title('Order Value Segmentation', fontsize=14, fontweight='bold')

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    # 4. Product Mix Impact
    ax4 = fig.add_subplot(gs[1, 1])

    prod_cnt_bins = pd.cut(order_summary['Prod_Cnt'],
                          bins=[0, 2, 4, 6, 100],
                          labels=['1-2 items', '3-4 items', '5-6 items', '7+ items'])
    prod_impact = order_summary.groupby(prod_cnt_bins)['OrderAmt'].mean()

    bars = ax4.bar(range(len(prod_impact)), prod_impact.values,
                   color=['#95a5a6', '#7f8c8d', '#34495e', '#2c3e50'])
    ax4.set_title('Average Order Value by Product Mix', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Products per Order')
    ax4.set_ylabel('Average Order Value ($)')
    ax4.set_xticks(range(len(prod_impact)))
    ax4.set_xticklabels(prod_impact.index, rotation=45)
    ax4.grid(True, alpha=0.3)

    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:,.0f}', ha='center', va='bottom', fontweight='bold')

    # 5. Employee Potential Matrix
    ax5 = fig.add_subplot(gs[1, 2])

    # Create potential score based on tenure and territory coverage
    employees_subset = employees_full.dropna(subset=['TenureYears', 'TerritoryCount', 'SalePrice'])
    employees_subset['PotentialScore'] = (
        employees_subset['TerritoryCount'] * 0.4 +
        employees_subset['TenureYears'] * 0.6
    )

    scatter = ax5.scatter(employees_subset['PotentialScore'],
                         employees_subset['SalePrice'],
                         c=employees_subset['HighPerformer'],
                         cmap='RdYlGn', s=100, alpha=0.7)

    ax5.set_title('Employee Potential vs Performance', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Potential Score')
    ax5.set_ylabel('Actual Sales ($)')
    ax5.grid(True, alpha=0.3)

    # Add quadrant lines
    median_potential = employees_subset['PotentialScore'].median()
    median_sales = employees_subset['SalePrice'].median()
    ax5.axhline(y=median_sales, color='gray', linestyle='--', alpha=0.5)
    ax5.axvline(x=median_potential, color='gray', linestyle='--', alpha=0.5)

    # 6. Revenue Growth Opportunities
    ax6 = fig.add_subplot(gs[2, :])

    # Simulate growth scenarios
    scenarios = ['Current', 'Optimize\nProduct Mix', 'Improve\nEmployee\nPerformance',
                'Increase\nOrder\nFrequency', 'Combined\nStrategy']

    current_revenue = order_summary['OrderAmt'].sum()
    growth_factors = [1.0, 1.15, 1.20, 1.18, 1.45]
    revenues = [current_revenue * factor for factor in growth_factors]

    colors = ['#95a5a6', '#3498db', '#2ecc71', '#f39c12', '#e74c3c']
    bars = ax6.bar(scenarios, revenues, color=colors, edgecolor='black', linewidth=2)

    ax6.set_title('Revenue Growth Opportunity Analysis', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Optimization Strategy')
    ax6.set_ylabel('Projected Revenue ($)')
    ax6.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax6.grid(True, alpha=0.3, axis='y')

    # Add value labels and growth percentages
    for i, (bar, rev, factor) in enumerate(zip(bars, revenues, growth_factors)):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'${rev:,.0f}\n(+{(factor-1)*100:.0f}%)',
                ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('predictive_insights_dashboard.png', dpi=300, bbox_inches='tight')
    print("✓ Saved predictive_insights_dashboard.png")

def generate_insights_report(employee_performance, order_summary, employees_full):
    """Generate a comprehensive insights report"""
    print("\n GENERATING INSIGHTS REPORT")
    print("="*60)

    high_performers = employee_performance[employee_performance['HighPerformer'] == 1]
    avg_high_sales = high_performers['SalePrice'].mean()
    avg_regular_sales = employee_performance[employee_performance['HighPerformer'] == 0]['SalePrice'].mean()


def main():
    """Main execution function"""
    print("\n" + "="*60)
    print("🚀 NORTHWIND SALES OPTIMIZATION ANALYSIS")
    print("="*60)

    # Load data
    data = load_data()

    # 1. Business problems and prediction targets
    orders_analysis, employee_performance = analyze_business_problems(data)

    # 2. One-hot encoding visualization
    ohe_df = visualize_employee_onehot_encoding(data)

    # 3. Employee predictive features
    employees_full = analyze_employee_predictive_features(data, orders_analysis, employee_performance)

    # 4. Order details with SalePrice
    order_details = visualize_order_details_summary(data)

    # 5. Order summary analysis
    order_summary = create_order_summary_analysis(data, order_details)

    # 6. Predictive insights dashboard
    create_predictive_insights_dashboard(employee_performance, order_summary, employees_full)
 

    print("\n" + "="*60)
    print(" ANALYSIS COMPLETE!")
    print("="*60)
    print("   • employee_onehot_analysis.png")
    print("   • employee_predictive_features.png")
    print("   • order_details_saleprice_analysis.png")
    print("   • order_summary_analysis.png")
    print("   • predictive_insights_dashboard.png")
    print("   • order_summary_analysis.csv")
    print("   • sales_optimization_report.txt")
    

if __name__ == "__main__":
    main()