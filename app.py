import json
import os
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

st.set_page_config(page_title="Team 3 IT5006 Dashboard", layout="wide")

DATA_DIR = "Olist_data"
UNTRANSLATED_LABEL = "untranslated / missing"

API_BASE_URL = "https://team3-it5006-ecommerce-analytics.onrender.com"
PREDICT_URL = f"{API_BASE_URL}/predict"
RUN_SUMMARY_PATH = os.path.join("artifacts", "regression_run_summary.json")
# Order counts per temporal split, from the split table in cell 12 of
# notebooks/phase2 regression edits.ipynb (not recorded in the run summary).
SPLIT_SIZES = {"Train": 68_318, "Validation": 12_845, "Test": 12_507}
CATEGORICAL_FEATURES = {
    "purchase_month", "purchase_day_of_week", "purchase_hour", "main_product_category",
    "farthest_seller_state", "is_weekend", "is_interstate",
}
# Seller states absent from training: the API rejects them as unseen categories.
UNSEEN_SELLER_STATES = {"AC"}

# ---------------------------------------------------------------------------
# Design tokens (validated categorical + sequential palette)
# ---------------------------------------------------------------------------
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE = CATEGORICAL[0]      # primary metric
ORANGE = CATEGORICAL[1]    # secondary / comparison metric
ORDINAL_BLUE = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]  # light -> dark, ordered scores
SEQUENTIAL_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]  # continuous magnitude (heatmaps, choropleths)
SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
INK = "#0b0b0b"
MUTED = "#898781"


def style_fig(fig, title=None, xaxis_title=None, yaxis_title=None, showlegend=None, height=380):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        font_color=INK,
        plot_bgcolor=SURFACE,
        paper_bgcolor=SURFACE,
        legend=dict(font_color=MUTED, bgcolor="rgba(0,0,0,0)"),
        title_font_color=INK,
    )
    fig.update_xaxes(showgrid=False, color=MUTED, linecolor=GRID)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, color=MUTED, zeroline=False)
    if title is not None:
        fig.update_layout(title=title)
    if xaxis_title is not None:
        fig.update_layout(xaxis_title=xaxis_title)
    if yaxis_title is not None:
        fig.update_layout(yaxis_title=yaxis_title)
    if showlegend is not None:
        fig.update_layout(showlegend=showlegend)
    return fig


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
TABLES = {
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}

DATE_COLS = {
    "orders": ["order_purchase_timestamp", "order_approved_at",
               "order_delivered_carrier_date", "order_delivered_customer_date",
               "order_estimated_delivery_date"],
    "order_items": ["shipping_limit_date"],
    "reviews": ["review_creation_date", "review_answer_timestamp"],
}


@st.cache_data
def load_olist_data(data_dir):
    """Load the Olist CSV files needed for this dashboard into a dict of DataFrames."""
    data = {}
    for name, filename in TABLES.items():
        filepath = os.path.join(data_dir, filename)
        data[name] = pd.read_csv(filepath, parse_dates=DATE_COLS.get(name))
    return data


@st.cache_data
def load_data(data_dir):
    """Load and merge the Olist tables into one item-level analysis dataframe."""
    olist = load_olist_data(data_dir)

    products = olist["products"].merge(
        olist["category_translation"], on="product_category_name", how="left"
    )
    products["product_category_name_english"] = products["product_category_name_english"].fillna(UNTRANSLATED_LABEL)
    products["product_volume_cm3"] = (
        products["product_length_cm"] * products["product_height_cm"] * products["product_width_cm"]
    )
    products["product_weight_g"] = products["product_weight_g"].fillna(0)
    products["product_volume_cm3"] = products["product_volume_cm3"].fillna(0)
    products["product_weight_volume"] = products["product_weight_g"] * products["product_volume_cm3"]

    # A single order can have several payment rows (split payments) - collapse to one row per order.
    payments = olist["payments"]
    dominant_payment = (
        payments.sort_values("payment_value", ascending=False)
        .drop_duplicates(subset="order_id", keep="first")[["order_id", "payment_type"]]
    )
    installments = payments.groupby("order_id", as_index=False)["payment_installments"].max()

    # Keep one review per order.
    reviews = olist["reviews"].drop_duplicates(subset="order_id", keep="first")[["order_id", "review_score"]]

    df = (
        olist["order_items"]
        .merge(
            products[["product_id", "product_category_name_english", "product_weight_volume"]],
            on="product_id", how="left",
        )
        .merge(
            olist["orders"][["order_id", "customer_id", "order_status",
                              "order_purchase_timestamp", "order_delivered_customer_date",
                              "order_estimated_delivery_date"]],
            on="order_id", how="left",
        )
        .merge(
            olist["customers"][["customer_id", "customer_unique_id", "customer_state"]],
            on="customer_id", how="left",
        )
        .merge(olist["sellers"][["seller_id", "seller_state"]], on="seller_id", how="left")
        .merge(dominant_payment, on="order_id", how="left")
        .merge(installments, on="order_id", how="left")
        .merge(reviews, on="order_id", how="left")
    )

    df["product_category_name_english"] = df["product_category_name_english"].fillna(UNTRANSLATED_LABEL)
    df["payment_type"] = df["payment_type"].fillna("not_defined")
    df["item_revenue"] = df["price"] + df["freight_value"]
    df["delivery_days"] = (df["order_delivered_customer_date"] - df["order_purchase_timestamp"]).dt.days
    df["estimated_days"] = (df["order_estimated_delivery_date"] - df["order_purchase_timestamp"]).dt.days
    df["delivery_delay_days"] = (df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]).dt.days
    df["is_on_time"] = df["order_delivered_customer_date"] <= df["order_estimated_delivery_date"]

    return df


@st.cache_data
def load_zip_coordinates(data_dir):
    """Median lat/lng per 5-digit zip prefix, as in the regression notebook's distance feature."""
    geo = pd.read_csv(
        os.path.join(data_dir, "olist_geolocation_dataset.csv"),
        usecols=["geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng"],
        dtype={"geolocation_zip_code_prefix": "string"},
    )
    geo["geolocation_zip_code_prefix"] = geo["geolocation_zip_code_prefix"].str.zfill(5)
    return geo.groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]].median()


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


@st.cache_data
def load_run_summary(path):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
def get_filter_options(df):
    return {
        "min_date": df["order_purchase_timestamp"].min().date(),
        "max_date": df["order_purchase_timestamp"].max().date(),
        "statuses": sorted(df["order_status"].dropna().unique()),
        "categories": sorted(df["product_category_name_english"].dropna().unique()),
        "states": sorted(df["customer_state"].dropna().unique()),
        "seller_states": sorted(df["seller_state"].dropna().unique()),
        "payment_types": sorted(df["payment_type"].dropna().unique()),
        "review_scores": REVIEW_SCORE_LABELS,
        "min_weight_volume": int(df["product_weight_volume"].min()),
        "max_weight_volume": int(df["product_weight_volume"].max()),
    }


ALL_LABEL = "ALL"
NO_SCORE_LABEL = "No score"
REVIEW_SCORE_LABELS = [NO_SCORE_LABEL, "1", "2", "3", "4", "5"]


def _toggle_all(key):
    """Keep ALL and specific values mutually exclusive: picking one drops the other."""
    selected = st.session_state.get(key, [])
    if not selected:
        st.session_state[key] = [ALL_LABEL]
        return
    if selected[-1] == ALL_LABEL:
        st.session_state[key] = [ALL_LABEL]
    elif ALL_LABEL in selected:
        st.session_state[key] = [v for v in selected if v != ALL_LABEL]


def multiselect_all(label, options, key):
    """Multiselect that shows a single 'ALL' tag by default instead of every option."""
    st.session_state.setdefault(key, [ALL_LABEL])
    selected = st.sidebar.multiselect(
        label, [ALL_LABEL] + options, key=key,
        on_change=_toggle_all, args=(key,),
    )
    if not selected or ALL_LABEL in selected:
        return options
    return selected


def render_sidebar(df):
    opts = get_filter_options(df)
    st.sidebar.header("Filters")

    st.sidebar.markdown(
        f"📅 **Data available:** {opts['min_date']} → {opts['max_date']}  \n"
        "<small>Based on approved orders</small>",
        unsafe_allow_html=True,
    )
    start_date = st.sidebar.date_input(
        "Start date", value=opts["min_date"], min_value=opts["min_date"], max_value=opts["max_date"],
        format="YYYY-MM-DD", key="start_date",
    )
    end_date = st.sidebar.date_input(
        "End date", value=opts["max_date"], min_value=opts["min_date"], max_value=opts["max_date"],
        format="YYYY-MM-DD", key="end_date",
    )
    if start_date > end_date:
        st.sidebar.error("Start date must be on or before end date.")
        start_date, end_date = opts["min_date"], opts["max_date"]
    date_range = (start_date, end_date)

    statuses = multiselect_all("Order status", opts["statuses"], key="statuses")
    categories = multiselect_all("Product category", opts["categories"], key="categories")
    states = multiselect_all("Customer state", opts["states"], key="states")
    seller_states = multiselect_all("Seller state", opts["seller_states"], key="seller_states")
    payment_types = multiselect_all("Payment method", opts["payment_types"], key="payment_types")
    review_scores = multiselect_all("Review score", opts["review_scores"], key="review_scores")

    weight_volume_range = st.sidebar.slider(
        "Product weight × volume (g·cm³)",
        min_value=opts["min_weight_volume"], max_value=opts["max_weight_volume"],
        value=(opts["min_weight_volume"], opts["max_weight_volume"]), key="weight_volume_range",
    )

    return {
        "date_range": date_range,
        "statuses": statuses,
        "categories": categories,
        "states": states,
        "seller_states": seller_states,
        "payment_types": payment_types,
        "review_scores": review_scores,
        "weight_volume_range": weight_volume_range,
    }


def apply_filters(df, filters):
    start_date, end_date = filters["date_range"]

    selected_scores = filters["review_scores"]
    numeric_scores = [int(s) for s in selected_scores if s != NO_SCORE_LABEL]
    score_mask = df["review_score"].isin(numeric_scores)
    if NO_SCORE_LABEL in selected_scores:
        score_mask |= df["review_score"].isna()

    min_weight_volume, max_weight_volume = filters["weight_volume_range"]
    weight_volume_mask = df["product_weight_volume"].between(min_weight_volume, max_weight_volume)

    mask = (
        (df["order_purchase_timestamp"].dt.date >= start_date)
        & (df["order_purchase_timestamp"].dt.date <= end_date)
        & (df["order_status"].isin(filters["statuses"]))
        & (df["product_category_name_english"].isin(filters["categories"]))
        & (df["customer_state"].isin(filters["states"]))
        & (df["seller_state"].isin(filters["seller_states"]))
        & (df["payment_type"].isin(filters["payment_types"]))
        & score_mask
        & weight_volume_mask
    )
    return df.loc[mask].copy()


# ---------------------------------------------------------------------------
# KPI banner
# ---------------------------------------------------------------------------
def format_currency_compact(value):
    """Abbreviate large currency values (R$ 15.2M) so KPI tiles don't overflow on narrow screens."""
    if value >= 1_000_000:
        return f"R$ {value / 1_000_000:,.2f}M"
    if value >= 1_000:
        return f"R$ {value / 1_000:,.1f}K"
    return f"R$ {value:,.0f}"


def render_kpis(df):
    if df.empty:
        st.warning("No data matches the selected filters.")
        return

    total_revenue = df["item_revenue"].sum()
    total_orders = df["order_id"].nunique()
    aov = df.groupby("order_id")["item_revenue"].sum().mean()
    avg_review = df["review_score"].mean()

    delivered = df[df["order_delivered_customer_date"].notna()].drop_duplicates("order_id")
    on_time_rate = delivered["is_on_time"].mean() if not delivered.empty else float("nan")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Revenue", format_currency_compact(total_revenue))
    c2.metric("Total Orders", f"{total_orders:,}")
    c3.metric("Avg Order Value", f"R$ {aov:,.2f}")
    c4.metric("Avg Review Score", f"{avg_review:.2f} / 5" if pd.notna(avg_review) else "N/A")
    c5.metric("On-Time Delivery", f"{on_time_rate:.1%}" if pd.notna(on_time_rate) else "N/A")


# ---------------------------------------------------------------------------
# Tab 1: Sales & Revenue Trends
# ---------------------------------------------------------------------------
def render_sales_tab(df):
    if df.empty:
        st.info("Adjust filters to see sales trends.")
        return

    granularity = st.radio("Trend granularity", ["Monthly", "Daily"], horizontal=True, key="rev_granularity")
    trend_df = df.copy()
    trend_df["period"] = trend_df["order_purchase_timestamp"].dt.to_period(
        "M" if granularity == "Monthly" else "D"
    ).astype(str)
    revenue_trend = trend_df.groupby("period", as_index=False)["item_revenue"].sum()

    fig_trend = px.line(revenue_trend, x="period", y="item_revenue", markers=True,
                         color_discrete_sequence=[BLUE])
    fig_trend.update_traces(line_width=2, marker_size=5)
    style_fig(fig_trend, title=f"{granularity} Revenue Trend", xaxis_title="", yaxis_title="Revenue (R$)")
    st.plotly_chart(fig_trend, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        payment_revenue = (
            df.groupby("payment_type", as_index=False)["item_revenue"]
            .sum()
            .sort_values("item_revenue", ascending=False)
        )
        fig_donut = px.pie(payment_revenue, names="payment_type", values="item_revenue", hole=0.55,
                            color_discrete_sequence=CATEGORICAL)
        fig_donut.update_traces(textinfo="percent+label")
        style_fig(fig_donut, title="Revenue by Payment Method")
        st.plotly_chart(fig_donut, use_container_width=True)

    with col2:
        installments = (
            df.drop_duplicates("order_id")["payment_installments"]
            .value_counts()
            .sort_index()
            .reset_index()
        )
        installments.columns = ["installments", "orders"]
        fig_inst = px.bar(installments, x="installments", y="orders", color_discrete_sequence=[BLUE])
        style_fig(fig_inst, title="Orders by Payment Installments", xaxis_title="Installments", yaxis_title="Orders")
        st.plotly_chart(fig_inst, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 2: Logistics & Regional Insights
# ---------------------------------------------------------------------------
def render_logistics_tab(df):
    if df.empty:
        st.info("Adjust filters to see logistics insights.")
        return

    col1, col2 = st.columns(2)
    with col1:
        state_orders = (
            df.groupby("customer_state", as_index=False)["order_id"]
            .nunique()
            .rename(columns={"order_id": "orders"})
            .nlargest(10, "orders")
            .sort_values("orders")
        )
        fig_state_orders = px.bar(state_orders, x="orders", y="customer_state", orientation="h",
                                   color_discrete_sequence=[BLUE])
        style_fig(fig_state_orders, title="Top 10 States by Order Volume", xaxis_title="Orders", yaxis_title="")
        st.plotly_chart(fig_state_orders, use_container_width=True)

    with col2:
        state_revenue = (
            df.groupby("customer_state", as_index=False)["item_revenue"]
            .sum()
            .nlargest(10, "item_revenue")
            .sort_values("item_revenue")
        )
        fig_state_rev = px.bar(state_revenue, x="item_revenue", y="customer_state", orientation="h",
                                color_discrete_sequence=[ORANGE])
        style_fig(fig_state_rev, title="Top 10 States by Revenue", xaxis_title="Revenue (R$)", yaxis_title="")
        st.plotly_chart(fig_state_rev, use_container_width=True)

    st.subheader("Order Volume: Customer State vs. Seller State")
    top_customer_states = df.groupby("customer_state")["order_id"].nunique().nlargest(10).index.tolist()
    top_seller_states = df.groupby("seller_state")["order_id"].nunique().nlargest(10).index.tolist()

    def anchor_order(states, first, last):
        """Alphabetical order with `first`/`last` pinned to the opposite ends (if present)."""
        middle = sorted(s for s in states if s not in (first, last))
        return [s for s in [first] if s in states] + middle + [s for s in [last] if s in states]

    customer_order = anchor_order(top_customer_states, "BA", "SP")
    seller_order = anchor_order(top_seller_states, "SP", "BA")

    state_pairs = (
        df[df["customer_state"].isin(top_customer_states) & df["seller_state"].isin(top_seller_states)]
        .groupby(["seller_state", "customer_state"], as_index=False)["order_id"]
        .nunique()
        .rename(columns={"order_id": "orders"})
    )
    if state_pairs.empty:
        st.info("No data matches the selected filters.")
    else:
        pivot = (
            state_pairs.pivot(index="seller_state", columns="customer_state", values="orders")
            .reindex(index=seller_order, columns=customer_order)
            .fillna(0)
            .astype(int)
        )
        fig_heatmap = px.imshow(pivot, color_continuous_scale=SEQUENTIAL_BLUE, aspect="auto", text_auto=True)
        fig_heatmap.update_traces(hovertemplate="Customer: %{x}<br>Seller: %{y}<br>Orders: %{z:,.0f}<extra></extra>")
        style_fig(fig_heatmap, title="Order Volume: Top 10 Customer States vs. Top 10 Seller States",
                  xaxis_title="Customer State", yaxis_title="Seller State", height=420)
        st.plotly_chart(fig_heatmap, use_container_width=True)

    st.subheader("Delivery Performance: Customer State vs. Seller State")
    delivered_orders = df[df["order_status"] == "delivered"]
    performance_pairs = (
        delivered_orders.groupby(["seller_state", "customer_state"], as_index=False)["is_on_time"]
        .mean()
        .rename(columns={"is_on_time": "on_time_rate"})
    )
    if performance_pairs.empty:
        st.info("No delivered orders in the current filter selection.")
    else:
        perf_pivot = (
            performance_pairs.pivot(index="seller_state", columns="customer_state", values="on_time_rate")
            .reindex(index=seller_order, columns=customer_order)
        )
        fig_perf = px.imshow(perf_pivot, color_continuous_scale=SEQUENTIAL_BLUE, aspect="auto",
                              text_auto=".1%", zmin=0, zmax=1)
        fig_perf.update_traces(hovertemplate="Customer: %{x}<br>Seller: %{y}<br>On-time: %{z:.1%}<extra></extra>")
        style_fig(fig_perf, title="Delivery Performance (% On-Time) Between Top 10 Customer and Seller States",
                  xaxis_title="Customer State", yaxis_title="Seller State", height=420)
        st.plotly_chart(fig_perf, use_container_width=True)

    st.subheader("Delivery Lead Time: Actual vs. Estimated")
    delivered = df[df["order_delivered_customer_date"].notna()].drop_duplicates("order_id")
    if delivered.empty:
        st.info("No delivered orders in the current filter selection.")
        return

    lead_time = pd.concat([
        delivered[["delivery_days"]].rename(columns={"delivery_days": "days"}).assign(type="Actual"),
        delivered[["estimated_days"]].rename(columns={"estimated_days": "days"}).assign(type="Estimated"),
    ])
    fig_lead = px.box(lead_time, x="type", y="days", color="type",
                       category_orders={"type": ["Actual", "Estimated"]},
                       color_discrete_sequence=[BLUE, ORANGE])
    style_fig(fig_lead, title="Delivery Lead Time: Actual vs. Estimated (days)",
              xaxis_title="", yaxis_title="Days", showlegend=False)
    st.plotly_chart(fig_lead, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 3: Product Performance & Reviews
# ---------------------------------------------------------------------------
def render_products_tab(df):
    if df.empty:
        st.info("Adjust filters to see product performance.")
        return

    cat_revenue = df.groupby("product_category_name_english", as_index=False)["item_revenue"].sum()

    col1, col2 = st.columns(2)
    with col1:
        top10 = cat_revenue.nlargest(10, "item_revenue").sort_values("item_revenue")
        fig_top = px.bar(top10, x="item_revenue", y="product_category_name_english", orientation="h",
                          color_discrete_sequence=[BLUE])
        style_fig(fig_top, title="Top 10 Categories by Revenue", xaxis_title="Revenue (R$)", yaxis_title="")
        st.plotly_chart(fig_top, use_container_width=True)

    with col2:
        bottom10 = cat_revenue.nsmallest(10, "item_revenue").sort_values("item_revenue", ascending=False)
        fig_bottom = px.bar(bottom10, x="item_revenue", y="product_category_name_english", orientation="h",
                             color_discrete_sequence=[ORANGE])
        style_fig(fig_bottom, title="Bottom 10 Categories by Revenue", xaxis_title="Revenue (R$)", yaxis_title="")
        st.plotly_chart(fig_bottom, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        review_dist = df.drop_duplicates("order_id")["review_score"].dropna().astype(int).astype(str)
        review_dist = review_dist.value_counts().reset_index()
        review_dist.columns = ["review_score", "orders"]
        score_order = [str(i) for i in range(1, 6)]
        fig_review = px.bar(review_dist, x="review_score", y="orders", color="review_score",
                             category_orders={"review_score": score_order},
                             color_discrete_sequence=ORDINAL_BLUE)
        style_fig(fig_review, title="Review Score Distribution",
                  xaxis_title="Review Score", yaxis_title="Orders", showlegend=False)
        st.plotly_chart(fig_review, use_container_width=True)

    with col4:
        delayed = (
            df[df["order_delivered_customer_date"].notna() & df["review_score"].notna()]
            .drop_duplicates("order_id")
            .copy()
        )
        delayed["review_score"] = delayed["review_score"].astype(int).astype(str)
        fig_scatter = px.box(delayed, x="review_score", y="delivery_delay_days", color="review_score",
                              category_orders={"review_score": score_order},
                              color_discrete_sequence=ORDINAL_BLUE)
        fig_scatter.add_hline(y=0, line_dash="dash", line_color=MUTED)
        style_fig(fig_scatter, title="Review Score vs. Delivery Delay",
                  xaxis_title="Review Score", yaxis_title="Delay past estimate (days)", showlegend=False)
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Customer Repeat Rate")
    orders_per_customer = df.groupby("customer_unique_id")["order_id"].nunique()
    total_customers = len(orders_per_customer)
    if total_customers == 0:
        st.info("No customers match the selected filters.")
        return

    repeat_customers = int((orders_per_customer > 1).sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Unique Customers", f"{total_customers:,}")
    c2.metric("Repeat Customers", f"{repeat_customers:,}")
    c3.metric("Repeat Rate", f"{repeat_customers / total_customers:.2%}")

    distribution = orders_per_customer.value_counts().sort_index().reset_index()
    distribution.columns = ["orders_placed", "customers"]
    distribution["orders_placed"] = distribution["orders_placed"].astype(str)
    fig_repeat = px.bar(distribution, x="orders_placed", y="customers", text="customers",
                        color_discrete_sequence=[BLUE], log_y=True)
    fig_repeat.update_traces(texttemplate="%{text:,}", textposition="outside",
                             hovertemplate="Orders placed: %{x}<br>Customers: %{y:,}<extra></extra>")
    style_fig(fig_repeat, title="Orders per Customer",
              xaxis_title="Orders placed", yaxis_title="Customers (log scale)")
    st.plotly_chart(fig_repeat, use_container_width=True)
    st.caption(
        "One-time buyers dominate, so the customer axis uses a log scale to keep the repeat tail visible. "
        "Figures reflect the current filters — a customer's other orders may fall outside them."
    )


# ---------------------------------------------------------------------------
# Tab 4: Delivery-time prediction (calls the deployed FastAPI /predict)
# ---------------------------------------------------------------------------
def zip_distance_km(customer_zip, seller_zip):
    """Haversine km between two zip prefixes, or None if either is unknown."""
    coords = load_zip_coordinates(DATA_DIR)
    customer_zip, seller_zip = customer_zip.strip().zfill(5), seller_zip.strip().zfill(5)
    if customer_zip not in coords.index or seller_zip not in coords.index:
        return None
    (lat1, lon1), (lat2, lon2) = coords.loc[customer_zip], coords.loc[seller_zip]
    return round(float(haversine_km(lat1, lon1, lat2, lon2)), 2)


# Feature values of real orders from the reserved test period (Jul-Aug 2018, unseen in training),
# aggregated with the notebook's feature builder. Keys match the prediction form's widget keys.
SAMPLE_ORDERS = [
    {"label": "Small: 1 health & beauty item", "order_id": "038f8b45fafa1130d2557e5b8dfd5d71", "values": {
         "pred_purchased_at": "2018-07-22 13:00:04", "pred_n_items": 1, "pred_n_unique_products": 1,
         "pred_n_product_categories": 1, "pred_n_sellers": 1, "pred_n_seller_states": 1,
         "pred_main_product_category": "health_beauty", "pred_total_price": 19.99, "pred_total_freight": 7.40,
         "pred_total_product_weight_g": 50.0, "pred_total_physical_score": 225000.0,
         "pred_customer_zip": "05466", "pred_seller_zip": "13481",
         "pred_farthest_seller_state": "SP", "pred_is_interstate": "False"}},
    {"label": "Heavy single: 1 musical instrument (30 kg)", "order_id": "029c545413b10a6bd7540401f6eef0a3", "values": {
         "pred_purchased_at": "2018-07-10 17:16:33", "pred_n_items": 1, "pred_n_unique_products": 1,
         "pred_n_product_categories": 1, "pred_n_sellers": 1, "pred_n_seller_states": 1,
         "pred_main_product_category": "musical_instruments", "pred_total_price": 1350.00,
         "pred_total_freight": 294.76, "pred_total_product_weight_g": 30000.0,
         "pred_total_physical_score": 6480000000.0, "pred_customer_zip": "71680", "pred_seller_zip": "18060",
         "pred_farthest_seller_state": "SP", "pred_is_interstate": "True"}},
    {"label": "Multiple: 3 identical computer accessories", "order_id": "00a57dfbb049fbaae10763e2cf15f797", "values": {
         "pred_purchased_at": "2018-07-20 11:09:30", "pred_n_items": 3, "pred_n_unique_products": 1,
         "pred_n_product_categories": 1, "pred_n_sellers": 1, "pred_n_seller_states": 1,
         "pred_main_product_category": "computers_accessories", "pred_total_price": 269.70,
         "pred_total_freight": 67.68, "pred_total_product_weight_g": 1551.0,
         "pred_total_physical_score": 4879446.0, "pred_customer_zip": "79002", "pred_seller_zip": "87114",
         "pred_farthest_seller_state": "PR", "pred_is_interstate": "True"}},
    {"label": "Mixed: 4 items from 2 sellers in 2 states", "order_id": "0c0aeabd6ae9d7a303af615334a220d0", "values": {
         "pred_purchased_at": "2018-08-20 14:45:48", "pred_n_items": 4, "pred_n_unique_products": 2,
         "pred_n_product_categories": 2, "pred_n_sellers": 2, "pred_n_seller_states": 2,
         "pred_main_product_category": "housewares", "pred_total_price": 264.84, "pred_total_freight": 77.53,
         "pred_total_product_weight_g": 2300.0, "pred_total_physical_score": 21569800.0,
         "pred_customer_zip": "01456", "pred_seller_zip": "95020",
         "pred_farthest_seller_state": "RS", "pred_is_interstate": "True"}},
    {"label": "Big: 6 phones shipped 2,300 km", "order_id": "03aba68b07658f28f29612641f08d4ba", "values": {
         "pred_purchased_at": "2018-08-08 14:27:15", "pred_n_items": 6, "pred_n_unique_products": 1,
         "pred_n_product_categories": 1, "pred_n_sellers": 1, "pred_n_seller_states": 1,
         "pred_main_product_category": "telephony", "pred_total_price": 1020.00, "pred_total_freight": 182.64,
         "pred_total_product_weight_g": 4200.0, "pred_total_physical_score": 42336000.0,
         "pred_customer_zip": "91170", "pred_seller_zip": "42738",
         "pred_farthest_seller_state": "BA", "pred_is_interstate": "True"}},
]


def sample_purchased_at(original):
    """Today's date at the sample's original time of day, so samples read as new checkouts."""
    return datetime.combine(datetime.now().date(), pd.Timestamp(original).time())


def load_sample_into_form(sample):
    """Button callback: runs before the rerun, so the form widgets pick up these values."""
    values = dict(sample["values"])
    purchased_at = sample_purchased_at(values.pop("pred_purchased_at"))
    st.session_state["pred_purchase_date"] = purchased_at.date()
    st.session_state["pred_purchase_time"] = purchased_at.time().replace(second=0)
    st.session_state.update(values)
    st.session_state["pred_loaded_sample"] = sample["order_id"]


def render_sample_orders():
    st.subheader("Sample Orders")
    st.caption("Example orders to try. Click Load sample to fill the form, then press Predict Delivery Days.")
    table = pd.DataFrame([
        # Purchase date/time is left out here; Load sample fills it into the form instead.
        {"Sample": s["label"],
         **{k.removeprefix("pred_"): v for k, v in s["values"].items() if k != "pred_purchased_at"}}
        for s in SAMPLE_ORDERS
    ])
    st.dataframe(table, hide_index=True, use_container_width=True)
    cols = st.columns(len(SAMPLE_ORDERS))
    for i, (col, sample) in enumerate(zip(cols, SAMPLE_ORDERS), start=1):
        col.button(f"Load sample {i}", key=f"load_sample_{i}", on_click=load_sample_into_form,
                   args=(sample,), use_container_width=True)


def render_prediction_tab():
    olist = load_olist_data(DATA_DIR)
    categories = sorted(olist["category_translation"]["product_category_name_english"].dropna().unique())
    seller_states = sorted(set(olist["sellers"]["seller_state"].dropna()) - UNSEEN_SELLER_STATES)

    # Defaults go through session state (not `value=`) so sample buttons can overwrite them.
    now = datetime.now()
    defaults = {
        "pred_purchase_date": now.date(), "pred_purchase_time": now.time().replace(second=0, microsecond=0),
        "pred_n_items": 0, "pred_n_unique_products": 0, "pred_n_product_categories": 0,
        "pred_n_sellers": 0, "pred_n_seller_states": 0, "pred_main_product_category": categories[0],
        "pred_total_price": 0.0, "pred_total_freight": 0.0, "pred_total_product_weight_g": 0.0,
        "pred_total_physical_score": 0.0, "pred_customer_zip": "", "pred_seller_zip": "",
        "pred_farthest_seller_state": "SP" if "SP" in seller_states else seller_states[0],
        "pred_is_interstate": "True",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    st.subheader("Predict Delivery Days")
    st.caption("Estimate purchase-to-delivery days for a new order at checkout.")
    st.markdown(
        f"API: [GET /health]({API_BASE_URL}/health) · [GET /docs (Swagger UI)]({API_BASE_URL}/docs#/)"
    )

    loaded = next((s for s in SAMPLE_ORDERS if s["order_id"] == st.session_state.get("pred_loaded_sample")), None)
    if loaded:
        st.info(f"Loaded sample: {loaded['label']}.")

    with st.form("prediction_form"):
        st.markdown("**Purchase**")
        c1, c2 = st.columns(2)
        purchase_date = c1.date_input("Purchase date", format="YYYY-MM-DD", key="pred_purchase_date")
        purchase_time = c2.time_input("Purchase time", step=60, key="pred_purchase_time")

        st.markdown("**Basket**")
        c1, c2, c3 = st.columns(3)
        n_items = c1.number_input("No. of items", min_value=0, step=1, key="pred_n_items")
        n_unique_products = c2.number_input("No. of unique items", min_value=0, step=1, key="pred_n_unique_products")
        n_product_categories = c3.number_input("No. of product categories", min_value=0, step=1,
                                               key="pred_n_product_categories")
        c1, c2 = st.columns(2)
        n_sellers = c1.number_input("No. of sellers", min_value=0, step=1, key="pred_n_sellers")
        n_seller_states = c2.number_input("No. of seller states", min_value=0, step=1, key="pred_n_seller_states")
        main_product_category = st.selectbox(
            "Main product category", categories, key="pred_main_product_category",
            help="Most common category among the order's items; ties go to the alphabetically first.",
        )

        st.markdown("**Price & physical size**")
        c1, c2 = st.columns(2)
        total_price = c1.number_input("Total price (R$)", min_value=0.0, step=1.0, format="%.2f",
                                      key="pred_total_price")
        total_freight = c2.number_input("Total freight (R$)", min_value=0.0, step=1.0, format="%.2f",
                                        key="pred_total_freight")
        c1, c2 = st.columns(2)
        total_product_weight_g = c1.number_input("Total product weight (g)", min_value=0.0, step=100.0,
                                                 key="pred_total_product_weight_g")
        total_physical_score = c2.number_input(
            "Total physical score", min_value=0.0, step=1000.0, key="pred_total_physical_score",
            help="Sum over items of weight (g) × volume (cm³).",
        )

        st.markdown("**Shipping**")
        c1, c2 = st.columns(2)
        customer_zip = c1.text_input("Customer zip prefix (5 digits)", max_chars=5, placeholder="e.g. 01310",
                                     key="pred_customer_zip")
        seller_zip = c2.text_input("Farthest seller zip prefix (5 digits)", max_chars=5, placeholder="e.g. 13023",
                                   key="pred_seller_zip")
        c1, c2 = st.columns(2)
        farthest_seller_state = c1.selectbox("Farthest seller state", seller_states,
                                             key="pred_farthest_seller_state")
        is_interstate = c2.selectbox("Interstate order?", ["True", "False"], key="pred_is_interstate")

        submitted = st.form_submit_button("Predict Delivery Days", type="primary")

    if submitted:
        submit_prediction(
            purchase_date, purchase_time, n_items, n_unique_products, n_product_categories, n_sellers,
            n_seller_states, main_product_category, total_price, total_freight, total_product_weight_g,
            total_physical_score, customer_zip, seller_zip, farthest_seller_state, is_interstate,
        )

    st.divider()
    render_sample_orders()


def submit_prediction(purchase_date, purchase_time, n_items, n_unique_products, n_product_categories, n_sellers,
                      n_seller_states, main_product_category, total_price, total_freight, total_product_weight_g,
                      total_physical_score, customer_zip, seller_zip, farthest_seller_state, is_interstate):

    counts = {
        "No. of items": n_items, "No. of unique items": n_unique_products,
        "No. of product categories": n_product_categories, "No. of sellers": n_sellers,
        "No. of seller states": n_seller_states,
    }
    missing = [label for label, value in counts.items() if value < 1]
    if missing:
        st.error(f"These must be at least 1: {', '.join(missing)}.")
        return

    distance_km = None
    if customer_zip.strip() and seller_zip.strip():
        distance_km = zip_distance_km(customer_zip, seller_zip)
        if distance_km is None:
            st.warning("Zip prefix not found in the geolocation data; the model will use the typical distance.")
    else:
        st.info("No zip prefixes given; the model will use the typical seller–buyer distance.")

    purchased_at = datetime.combine(purchase_date, purchase_time)
    payload = {
        "n_items": int(n_items),
        "n_unique_products": int(n_unique_products),
        "n_product_categories": int(n_product_categories),
        "n_sellers": int(n_sellers),
        "n_seller_states": int(n_seller_states),
        "total_freight": float(total_freight),
        "total_product_weight_g": float(total_product_weight_g),
        "total_physical_score": float(total_physical_score),
        "max_seller_buyer_distance_km": distance_km,
        # Same +1 smoothing as the notebook, so zero-price orders don't divide by zero.
        "freight_ratio": float(total_freight) / (float(total_price) + 1),
        "purchase_month": purchased_at.month,
        "purchase_day_of_week": purchased_at.strftime("%A"),
        "purchase_hour": purchased_at.hour,
        "main_product_category": main_product_category,
        "farthest_seller_state": farthest_seller_state,
        "is_weekend": str(purchased_at.weekday() >= 5),
        "is_interstate": is_interstate,
    }

    # The free Render instance sleeps when idle, so the first call can take ~30-60s.
    with st.spinner("Calling the prediction API (may take up to a minute if it is waking up)..."):
        try:
            response = requests.post(PREDICT_URL, json=payload, timeout=90)
        except requests.RequestException as exc:
            st.error(f"Could not reach the prediction API: {exc}")
            return

    if response.status_code != 200:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        st.error(f"API returned {response.status_code}.")
        st.json(detail)
        return

    result = response.json()
    days = result["predicted_delivery_days"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted delivery", f"{days:.1f} days")
    c2.metric("Expected arrival", (purchased_at + pd.Timedelta(days=days)).strftime("%Y-%m-%d"))
    c3.metric("Seller–buyer distance", f"{distance_km:,.0f} km" if distance_km is not None else "typical")
    st.caption(f"Model: {result['model_name']} · trained on purchases before {result['training_purchase_cutoff']}")
    with st.expander("Request payload sent to the API"):
        st.json(payload)


# ---------------------------------------------------------------------------
# Tab 5: Model statistics (from artifacts/regression_run_summary.json)
# ---------------------------------------------------------------------------
def render_model_stats_tab():
    try:
        summary = load_run_summary(RUN_SUMMARY_PATH)
    except FileNotFoundError:
        st.error(f"Run summary not found at {RUN_SUMMARY_PATH}.")
        return
    results = summary["results"]

    st.subheader("Data Splits")
    st.caption("Temporal split by purchase date: the model is trained on the past and evaluated on later orders.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Train orders", f"{SPLIT_SIZES['Train']:,}")
    c2.metric("Validation orders", f"{SPLIT_SIZES['Validation']:,}")
    c3.metric("Test orders", f"{SPLIT_SIZES['Test']:,}")
    c4.metric("Total", f"{sum(SPLIT_SIZES.values()):,}")

    st.subheader("Model")
    params = summary["model_parameters"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Selected model", summary["selected_model"])
    c2.metric("Loss", params["loss"])
    c3.metric("Trees × depth", f"{params['n_estimators']} × {params['max_depth']}")
    c4.metric("Learning rate", params["learning_rate"])
    with st.expander("All model parameters"):
        st.dataframe(
            pd.DataFrame({"parameter": list(params), "value": [str(v) for v in params.values()]}),
            hide_index=True, use_container_width=True,
        )
    with st.expander("Training library versions"):
        st.json(summary["versions"])

    st.subheader("Features")
    features = summary["features"]
    categorical = [f for f in features if f in CATEGORICAL_FEATURES]
    numeric = [f for f in features if f not in CATEGORICAL_FEATURES]
    c1, c2, c3 = st.columns(3)
    c1.metric("Total features", len(features))
    c2.metric("Numeric", len(numeric))
    c3.metric("Categorical", len(categorical))
    c1, c2 = st.columns(2)
    c1.markdown("**Numeric** (median-imputed)\n" + "\n".join(f"- `{f}`" for f in numeric))
    c2.markdown("**Categorical** (one-hot encoded)\n" + "\n".join(f"- `{f}`" for f in categorical))

    st.subheader("Model Metrics")
    st.markdown("**Final test set** (held-out, most recent orders)")
    test = {row["Model"]: row for row in results["final_test"]}
    model_row = test[summary["selected_model"]]
    olist_row = test.get("Olist checkout estimate")
    c1, c2, c3 = st.columns(3)
    for col, metric, unit in [(c1, "MAE", " days"), (c2, "RMSE", " days"), (c3, "R2", "")]:
        delta = None
        if olist_row is not None:
            delta = f"{model_row[metric] - olist_row[metric]:+.2f} vs. Olist estimate"
        # Lower is better for MAE/RMSE, higher for R2.
        col.metric(f"Test {metric.replace('R2', 'R²')}", f"{model_row[metric]:.2f}{unit}", delta=delta,
                   delta_color="normal" if metric == "R2" else "inverse")

    st.markdown("**Validation set** (model selection)")
    validation = pd.DataFrame(results["validation"]).round(3)
    st.dataframe(validation, hide_index=True, use_container_width=True)

    with st.expander("Cross-validation screening (all candidate models)"):
        st.dataframe(pd.DataFrame(results["screening"]).round(3), hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    st.title("Team 3 IT5006 Dashboard")
    st.caption("SmartCommerce · Olist Brazilian e-commerce interactive analytics")

    df = load_data(DATA_DIR)
    filters = render_sidebar(df)
    filtered = apply_filters(df, filters)

    render_kpis(filtered)
    st.divider()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Sales & Revenue Trends",
        "Logistics & Regional Insights",
        "Product Performance & Reviews",
        "Prediction",
        "Model Stats",
    ])
    with tab1:
        render_sales_tab(filtered)
    with tab2:
        render_logistics_tab(filtered)
    with tab3:
        render_products_tab(filtered)
    with tab4:
        render_prediction_tab()
    with tab5:
        render_model_stats_tab()


if __name__ == "__main__":
    main()
