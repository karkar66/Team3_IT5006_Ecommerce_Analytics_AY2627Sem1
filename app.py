import os
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Team 3 IT5006 Dashboard")

DATA_DIR='Olist_data'

# name = st.text_input("Enter your name")
# if name:
#     st.write(f"Hello, {name}!")

# data = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
# st.line_chart(data)


def main():
    # Load data
    olist = load_olist_data(DATA_DIR)

    # Products with category translation
    products = olist['products'].merge(
        olist['category_translation'],
        on='product_category_name',
        how='left'
    )
    products['product_category_name_english'] = products['product_category_name_english'].fillna('Unknown')

    print("PRODUCTS TABLE (with English categories)")
    print("=" * 50)
    print(f"Shape: {products.shape}")
    print(f"\nTop 10 categories:")
    print(products['product_category_name_english'].value_counts().head(10))
    products.head(3)





    # Build consolidated analysis dataset
    # Step 1: Start with order_items (line-level)
    df = olist['order_items'].copy()
    print(f"Step 1 - order_items: {df.shape}")

    # Step 2: Add product details
    df = df.merge(
        products[['product_id', 'product_category_name_english', 'product_weight_g']],
        on='product_id',
        how='left'
    )
    print(f"Step 2 - + products: {df.shape}")

    # Step 3: Add order details
    df = df.merge(
        olist['orders'][['order_id', 'customer_id', 'order_status',
                        'order_purchase_timestamp', 'order_delivered_customer_date',
                        'order_estimated_delivery_date']],
        on='order_id',
        how='left'
    )
    print(f"Step 3 - + orders: {df.shape}")

    # Step 4: Add customer details
    df = df.merge(
        olist['customers'][['customer_id', 'customer_state', 'customer_city']],
        on='customer_id',
        how='left'
    )
    print(f"Step 4 - + customers: {df.shape}")

    # Step 5: Add seller details
    df = df.merge(
        olist['sellers'][['seller_id', 'seller_state']],
        on='seller_id',
        how='left'
    )
    print(f"Step 5 - + sellers: {df.shape}")

    print(f"\n✅ Final consolidated dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    df.head()


    # Add derived columns for analysis
    df['order_date'] = df['order_purchase_timestamp'].dt.date
    df['year_month'] = df['order_purchase_timestamp'].dt.to_period('M').astype(str)
    df['year'] = df['order_purchase_timestamp'].dt.year
    df['month'] = df['order_purchase_timestamp'].dt.month
    df['day_of_week'] = df['order_purchase_timestamp'].dt.day_name()

    # Delivery metrics
    df['delivery_days'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days
    df['is_on_time'] = (df['order_delivered_customer_date'] <= df['order_estimated_delivery_date'])

    # Revenue per item
    df['item_revenue'] = df['price'] + df['freight_value']

    print("Added derived columns:")
    print(df[['order_id', 'year_month', 'day_of_week', 'delivery_days', 'is_on_time', 'item_revenue']].head())


    # Set up plotting style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig_color = '#f8f9fa'

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('SmartCommerce Business Overview', fontsize=16, fontweight='bold')

    # 1. Orders over time
    monthly_orders = df.groupby('year_month')['order_id'].nunique()
    axes[0, 0].plot(monthly_orders.index, monthly_orders.values, marker='o', linewidth=2, color='#2E86AB')
    axes[0, 0].set_title('Monthly Orders', fontweight='bold')
    axes[0, 0].set_xlabel('Month')
    axes[0, 0].set_ylabel('Number of Orders')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # 2. Revenue by category (top 10)
    cat_revenue = df.groupby('product_category_name_english')['price'].sum().nlargest(10)
    axes[0, 1].barh(cat_revenue.index, cat_revenue.values, color='#A23B72')
    axes[0, 1].set_title('Top 10 Categories by Revenue', fontweight='bold')
    axes[0, 1].set_xlabel('Revenue (R$)')
    axes[0, 1].invert_yaxis()

    # 3. Order value distribution
    order_values = df.groupby('order_id')['price'].sum()
    axes[1, 0].hist(order_values[order_values < 1000], bins=50, color='#F18F01', edgecolor='white')
    axes[1, 0].set_title('Order Value Distribution (< R$1000)', fontweight='bold')
    axes[1, 0].set_xlabel('Order Value (R$)')
    axes[1, 0].set_ylabel('Frequency')

    # 4. Orders by state (top 10)
    state_orders = df.groupby('customer_state')['order_id'].nunique().nlargest(10)
    axes[1, 1].bar(state_orders.index, state_orders.values, color='#C73E1D')
    axes[1, 1].set_title('Top 10 States by Orders', fontweight='bold')
    axes[1, 1].set_xlabel('State')
    axes[1, 1].set_ylabel('Number of Orders')

    plt.tight_layout()
    plt.show()

    st.pyplot(fig)


    # Delivery and Review Analysis
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. Delivery days distribution
    delivered_data = df[df['delivery_days'].notna() & (df['delivery_days'] < 60)]
    axes[0].hist(delivered_data['delivery_days'], bins=40, color='#2E86AB', edgecolor='white')
    axes[0].axvline(delivered_data['delivery_days'].mean(), color='red', linestyle='--', label=f"Mean: {delivered_data['delivery_days'].mean():.1f}")
    axes[0].set_title('Delivery Days Distribution', fontweight='bold')
    axes[0].set_xlabel('Days to Deliver')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()

    # 2. Review score distribution
    score_counts = olist['reviews']['review_score'].value_counts().sort_index()
    colors = ['#C73E1D', '#F18F01', '#F7B538', '#A8D5BA', '#2E86AB']
    axes[1].bar(score_counts.index, score_counts.values, color=colors)
    axes[1].set_title('Review Score Distribution', fontweight='bold')
    axes[1].set_xlabel('Review Score')
    axes[1].set_ylabel('Count')

    # 3. Payment type distribution
    payment_dist = olist['payments'].groupby('payment_type')['payment_value'].sum().sort_values(ascending=True)
    axes[2].barh(payment_dist.index, payment_dist.values, color='#A23B72')
    axes[2].set_title('Revenue by Payment Type', fontweight='bold')
    axes[2].set_xlabel('Total Payment Value (R$)')

    plt.tight_layout()
    plt.show()

    st.pyplot(fig)


# Load all Olist tables
@st.cache_data
def load_olist_data(data_dir):
    """Load all Olist CSV files into a dictionary of DataFrames."""

    tables = {
        'orders': 'olist_orders_dataset.csv',
        'order_items': 'olist_order_items_dataset.csv',
        'customers': 'olist_customers_dataset.csv',
        'products': 'olist_products_dataset.csv',
        'sellers': 'olist_sellers_dataset.csv',
        'payments': 'olist_order_payments_dataset.csv',
        'reviews': 'olist_order_reviews_dataset.csv',
        'geolocation': 'olist_geolocation_dataset.csv',
        'category_translation': 'product_category_name_translation.csv'
    }

    # Date columns to parse
    date_cols = {
        'orders': ['order_purchase_timestamp', 'order_approved_at',
                   'order_delivered_carrier_date', 'order_delivered_customer_date',
                   'order_estimated_delivery_date'],
        'order_items': ['shipping_limit_date'],
        'reviews': ['review_creation_date', 'review_answer_timestamp']
    }

    data = {}
    for name, filename in tables.items():
        filepath = os.path.join(data_dir, filename)
        parse_dates = date_cols.get(name, None)
        data[name] = pd.read_csv(filepath, parse_dates=parse_dates)
        print(f"Loaded {name}: {data[name].shape[0]:,} rows × {data[name].shape[1]} cols")

    return data


if __name__ == "__main__":
    main()