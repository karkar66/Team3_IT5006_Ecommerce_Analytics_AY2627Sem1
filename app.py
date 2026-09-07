import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Team 3 IT5006 Dashboard", layout="wide")

DATA_DIR = "Olist_data"
UNTRANSLATED_LABEL = "untranslated / missing"

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
        .merge(products[["product_id", "product_category_name_english"]], on="product_id", how="left")
        .merge(
            olist["orders"][["order_id", "customer_id", "order_status",
                              "order_purchase_timestamp", "order_delivered_customer_date",
                              "order_estimated_delivery_date"]],
            on="order_id", how="left",
        )
        .merge(olist["customers"][["customer_id", "customer_state"]], on="customer_id", how="left")
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

    # Data-quality note: products missing an English category translation.
    missing_products = products[products["product_category_name_english"] == UNTRANSLATED_LABEL]
    translation_quality = {
        "total_products": len(products),
        "missing_count": len(missing_products),
        "missing_no_category": int(products["product_category_name"].isna().sum()),
        "missing_breakdown": missing_products.loc[
            missing_products["product_category_name"].notna(), "product_category_name"
        ].value_counts(),
    }

    return df, translation_quality


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

    st.sidebar.markdown(f"📅 **Data available:** {opts['min_date']} → {opts['max_date']}")
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

    return {
        "date_range": date_range,
        "statuses": statuses,
        "categories": categories,
        "states": states,
        "seller_states": seller_states,
        "payment_types": payment_types,
        "review_scores": review_scores,
    }


def apply_filters(df, filters):
    start_date, end_date = filters["date_range"]

    selected_scores = filters["review_scores"]
    numeric_scores = [int(s) for s in selected_scores if s != NO_SCORE_LABEL]
    score_mask = df["review_score"].isin(numeric_scores)
    if NO_SCORE_LABEL in selected_scores:
        score_mask |= df["review_score"].isna()

    mask = (
        (df["order_purchase_timestamp"].dt.date >= start_date)
        & (df["order_purchase_timestamp"].dt.date <= end_date)
        & (df["order_status"].isin(filters["statuses"]))
        & (df["product_category_name_english"].isin(filters["categories"]))
        & (df["customer_state"].isin(filters["states"]))
        & (df["seller_state"].isin(filters["seller_states"]))
        & (df["payment_type"].isin(filters["payment_types"]))
        & score_mask
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
def render_products_tab(df, translation_quality):
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

    with st.expander(
        f"Data Quality: {translation_quality['missing_count']} products missing an English category translation"
    ):
        st.write(
            f"Out of {translation_quality['total_products']:,} products, "
            f"{translation_quality['missing_count']:,} have no English category translation "
            f"(shown as \"{UNTRANSLATED_LABEL}\" throughout this dashboard)."
        )
        st.write(f"- {translation_quality['missing_no_category']:,} products have no category name at all")
        if not translation_quality["missing_breakdown"].empty:
            st.write("- Products whose category name isn't in the translation table:")
            st.dataframe(
                translation_quality["missing_breakdown"].rename_axis("product_category_name").rename("count"),
                use_container_width=True,
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    st.title("Team 3 IT5006 Dashboard")
    st.caption("SmartCommerce · Olist Brazilian e-commerce interactive analytics")

    df, translation_quality = load_data(DATA_DIR)
    filters = render_sidebar(df)
    filtered = apply_filters(df, filters)

    render_kpis(filtered)
    st.divider()

    tab1, tab2, tab3 = st.tabs([
        "Sales & Revenue Trends",
        "Logistics & Regional Insights",
        "Product Performance & Reviews",
    ])
    with tab1:
        render_sales_tab(filtered)
    with tab2:
        render_logistics_tab(filtered)
    with tab3:
        render_products_tab(filtered, translation_quality)


if __name__ == "__main__":
    main()
