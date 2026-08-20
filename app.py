from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# Page configuration must be the first Streamlit command
# ============================================================

st.set_page_config(
    page_title="SmartBuy AI",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Load recommendation modules
# ============================================================

try:
    from models.content_based import (
        get_product,
        products,
        recommend_similar,
        search_products,
    )
    from models.collaborative import users_also_purchased
    from models.hybrid import hybrid_for_product
except Exception as error:
    st.error("SmartBuy AI could not load the recommendation models.")
    st.exception(error)
    st.stop()


PROJECT_DIR = Path(__file__).resolve().parent
EVALUATION_PATH = PROJECT_DIR / "evaluation" / "evaluation_results.csv"


# ============================================================
# Visual design
# ============================================================

st.markdown(
    """
    <style>
    :root {
        --smartbuy-navy: #131921;
        --smartbuy-blue: #232f3e;
        --smartbuy-orange: #ff9900;
        --smartbuy-light: #f3f4f6;
    }

    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        background: #f5f7fa;
        color: #172033;
    }

    /* Keep Streamlit text readable even when the browser/OS uses dark mode. */
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stWidgetLabel"] p,
    [data-testid="stMetricLabel"] p,
    [data-testid="stMetricValue"],
    [data-baseweb="tab"] p,
    [data-testid="stExpander"] summary,
    [data-testid="stDataFrame"] {
        color: #26364a !important;
    }

    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] p,
    .stCaptionContainer,
    small {
        color: #52657a !important;
        opacity: 1 !important;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #172033 !important;
    }

    .smartbuy-header,
    .smartbuy-header * {
        color: #ffffff !important;
    }

    .smartbuy-logo span { color: #ff9900 !important; }

    /* Product cards need a visible surface instead of blending into the page. */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff;
        border-color: #d8e0e9 !important;
        box-shadow: 0 3px 10px rgba(15, 23, 42, 0.05);
    }

    /* Search input remains readable under Streamlit light or dark themes. */
    .stTextInput input {
        background: #ffffff !important;
        color: #172033 !important;
        border-color: #aab7c4 !important;
        caret-color: #172033 !important;
    }

    .stTextInput input::placeholder {
        color: #66788a !important;
        opacity: 1 !important;
    }

    .smartbuy-header {
        background: linear-gradient(105deg, #131921 0%, #232f3e 75%, #31445d 100%);
        border-radius: 0 0 18px 18px;
        padding: 22px 30px;
        margin: -1rem -1rem 1.2rem -1rem;
        color: white;
        box-shadow: 0 5px 20px rgba(15, 23, 42, 0.18);
    }

    .smartbuy-logo {
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.03em;
    }

    .smartbuy-tagline {
        color: #dbe4ee !important;
        margin-top: 3px;
        font-size: 0.95rem;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 750;
        color: #172033;
        margin: 0.4rem 0 0.8rem 0;
    }

    .product-icon {
        font-size: 3.6rem;
        text-align: center;
        padding: 10px 0 2px 0;
        min-height: 82px;
    }

    .product-name {
        font-size: 0.98rem;
        font-weight: 700;
        color: #172033;
        min-height: 48px;
        line-height: 1.35;
    }

    .product-meta {
        color: #64748b;
        font-size: 0.78rem;
        margin: 5px 0;
    }

    .product-rating {
        color: #b45309;
        font-size: 0.87rem;
        font-weight: 650;
    }

    .current-price {
        color: #b12704;
        font-size: 1.25rem;
        font-weight: 800;
    }

    .original-price {
        color: #7b8794;
        text-decoration: line-through;
        font-size: 0.8rem;
        margin-left: 6px;
    }

    .discount-badge {
        display: inline-block;
        padding: 2px 7px;
        border-radius: 999px;
        background: #dcfce7;
        color: #166534;
        font-size: 0.72rem;
        font-weight: 700;
        margin-top: 5px;
    }

    .reason-box {
        margin-top: 8px;
        padding: 7px 9px;
        border-radius: 8px;
        background: #eff6ff;
        color: #1e3a5f;
        font-size: 0.75rem;
        min-height: 48px;
    }

    .selected-panel {
        padding: 20px;
        border-radius: 14px;
        background: white;
        border: 1px solid #dbe3ec;
        box-shadow: 0 5px 16px rgba(15, 23, 42, 0.07);
        margin-bottom: 16px;
    }

    .metric-note {
        border-left: 4px solid #ff9900;
        background: #fff8e8;
        padding: 11px 14px;
        border-radius: 8px;
        color: #62420c;
    }

    div.stButton > button {
        border-radius: 9px;
        font-weight: 650;
        color: #ffffff;
        background: #172033;
        border-color: #172033;
    }

    div.stButton > button:hover {
        color: #172033;
        background: #ffb22e;
        border-color: #e88900;
    }

    div.stButton > button p {
        color: #ffffff !important;
    }

    div.stButton > button:hover p {
        color: #172033 !important;
    }

    div[data-testid="stFormSubmitButton"] button {
        background: #ff9900;
        border-color: #e88900;
        color: #172033 !important;
        font-weight: 750;
    }

    div[data-testid="stFormSubmitButton"] button p {
        color: #172033 !important;
    }

    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e2e8f0;
        color: #172033;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span {
        color: #26364a !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Session state
# ============================================================

# Use a fixed customer-facing range instead of changing the slider endpoints
# whenever the dataset changes. The selected values are still applied directly
# to the real product prices during filtering.
price_min = 0.0
price_max = 1_000_000.0

DEFAULT_STATE = {
    "active_query": "",
    "search_input": "",
    "selected_product_id": None,
    "view_history": [],
    "category_filter": "All",
    "subcategory_filter": "All",
    "brand_filter": "All",
    "price_filter": (price_min, price_max),
    "rating_filter": 0.0,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


def select_product(product_id: str) -> None:
    st.session_state.selected_product_id = str(product_id)
    history = [item for item in st.session_state.view_history if item != product_id]
    st.session_state.view_history = ([str(product_id)] + history)[:20]


def clear_selected_product() -> None:
    st.session_state.selected_product_id = None


def reset_filters() -> None:
    st.session_state.search_input = ""
    st.session_state.active_query = ""
    st.session_state.category_filter = "All"
    st.session_state.subcategory_filter = "All"
    st.session_state.brand_filter = "All"
    st.session_state.price_filter = (price_min, price_max)
    st.session_state.rating_filter = 0.0
    st.session_state.selected_product_id = None


# ============================================================
# Cached model calls
# ============================================================

@st.cache_data(show_spinner=False)
def cached_search(
    keyword: str,
    category: str,
    subcategory: str,
    brand: str,
    minimum_price: float,
    maximum_price: float,
    minimum_rating: float,
    limit: int,
) -> pd.DataFrame:
    return search_products(
        keyword=keyword,
        category=category,
        subcategory=subcategory,
        brand=brand,
        min_price=minimum_price,
        max_price=maximum_price,
        min_rating=minimum_rating,
        limit=limit,
    )


@st.cache_data(show_spinner=False)
def cached_similar(product_id: str, number: int) -> pd.DataFrame:
    return recommend_similar(product_id, number)


@st.cache_data(show_spinner=False)
def cached_also_purchased(product_id: str, number: int) -> pd.DataFrame:
    return users_also_purchased(product_id, number)


@st.cache_data(show_spinner=False)
def cached_product_hybrid(product_id: str, number: int) -> pd.DataFrame:
    return hybrid_for_product(product_id, number)


# ============================================================
# Product presentation helpers
# ============================================================

def product_icon(subcategory: object) -> str:
    icons = {
        "Mobile": "📱",
        "Laptop": "💻",
        "Camera": "📷",
        "Headphones": "🎧",
        "Outdoor": "🏕️",
        "Fitness": "🏋️",
        "Cycling": "🚴",
        "Men": "👔",
        "Women": "👗",
        "Kids": "🧸",
        "Kitchen Appliances": "🍳",
        "Furniture": "🛋️",
        "Home Decor": "🏠",
        "Makeup": "💄",
        "Skincare": "🧴",
        "Haircare": "💇",
    }
    return icons.get(str(subcategory), "🛍️")


def money(value: object) -> str:
    try:
        return f"₹{float(value):,.2f}"
    except (TypeError, ValueError):
        return "Price unavailable"


def render_product_card(
    row: pd.Series,
    key_prefix: str,
    show_button: bool = True,
) -> None:
    name = html.escape(str(row.get("product_name", "Unnamed product")))
    brand = html.escape(str(row.get("brand", "Unknown")))
    category = html.escape(str(row.get("category", "Unknown")))
    subcategory = html.escape(str(row.get("subcategory", "Unknown")))
    rating = float(row.get("rating", 0) or 0)
    review_count = int(row.get("review_count", 0) or 0)
    original_price = float(row.get("price", 0) or 0)
    final_price = float(row.get("final_price", original_price) or original_price)
    discount = float(row.get("discount", 0) or 0)
    reason = html.escape(str(row.get("recommendation_reason", "")))

    score_labels = [
        ("hybrid_score", "Hybrid"),
        ("similarity_score", "Similarity"),
        ("collaborative_score", "Collaborative"),
        ("search_score", "Search"),
    ]
    score_text = ""
    for column, label in score_labels:
        if column in row.index and pd.notna(row[column]):
            score_text = f"{label} score: {float(row[column]):.3f}"
            break

    reason_html = ""
    if reason or score_text:
        details = " · ".join(item for item in [score_text, reason] if item)
        reason_html = f'<div class="reason-box">🤖 {details}</div>'

    st.markdown(
        f"""
        <div class="product-icon">{product_icon(subcategory)}</div>
        <div class="product-name">{name}</div>
        <div class="product-meta">{brand} · {category} · {subcategory}</div>
        <div class="product-rating">⭐ {rating:.2f} &nbsp; ({review_count:,} reviews)</div>
        <div style="margin-top:8px;">
            <span class="current-price">{money(final_price)}</span>
            <span class="original-price">{money(original_price)}</span>
        </div>
        <div class="discount-badge">{discount:.0f}% OFF</div>
        {reason_html}
        """,
        unsafe_allow_html=True,
    )

    if show_button:
        st.button(
            "View product & recommendations",
            key=f"{key_prefix}_{row['product_id']}",
            use_container_width=True,
            on_click=select_product,
            args=(str(row["product_id"]),),
        )


def render_product_grid(
    frame: pd.DataFrame,
    key_prefix: str,
    show_button: bool = True,
    columns_per_row: int = 3,
) -> None:
    if frame is None or frame.empty:
        st.info("No products are available for this section.")
        return

    frame = frame.reset_index(drop=True)
    for start in range(0, len(frame), columns_per_row):
        columns = st.columns(columns_per_row)
        for offset, column in enumerate(columns):
            index = start + offset
            if index >= len(frame):
                continue
            with column:
                with st.container(border=True):
                    render_product_card(
                        frame.iloc[index],
                        key_prefix=f"{key_prefix}_{index}",
                        show_button=show_button,
                    )


# ============================================================
# Header and sidebar
# ============================================================

st.markdown(
    """
    <div class="smartbuy-header">
        <div class="smartbuy-logo">🛒 SmartBuy <span>AI</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("### 🛒 SmartBuy AI")
page = st.sidebar.radio(
    "Navigation",
    ["🛍️ Shop", "📊 Evaluation"],
    key="page_navigation",
)

st.sidebar.markdown("---")


# ============================================================
# Shop page
# ============================================================

if page == "🛍️ Shop":
    with st.form("product_search_form", clear_on_submit=False):
        search_column, button_column = st.columns([5, 1])
        with search_column:
            st.text_input(
                "Search SmartBuy",
                key="search_input",
                placeholder="Try laptop, Samsung, wireless headphones, cycling...",
                label_visibility="collapsed",
            )
        with button_column:
            submitted = st.form_submit_button("🔍 Search", use_container_width=True)

    if submitted:
        st.session_state.active_query = st.session_state.search_input.strip()
        st.session_state.selected_product_id = None

    # ============================================================
    # Main page filters
    # ============================================================

    # Product filters area

    filter_col1, filter_col2, filter_col3 = st.columns(3)

    with filter_col1:
        categories = ["All", *sorted(products["category"].dropna().unique().tolist())]
        st.selectbox("Department", categories, key="category_filter")

    category_products = products
    if st.session_state.category_filter != "All":
        category_products = category_products[
            category_products["category"].eq(st.session_state.category_filter)
        ]

    with filter_col2:
        subcategories = [
            "All",
            *sorted(category_products["subcategory"].dropna().unique().tolist())
        ]

        if st.session_state.subcategory_filter not in subcategories:
            st.session_state.subcategory_filter = "All"

        st.selectbox(
            "Product category",
            subcategories,
            key="subcategory_filter"
        )

    brand_products = category_products

    if st.session_state.subcategory_filter != "All":
        brand_products = brand_products[
            brand_products["subcategory"].eq(
                st.session_state.subcategory_filter
            )
        ]

    with filter_col3:
        brands = [
            "All",
            *sorted(brand_products["brand"].dropna().unique().tolist())
        ]

        if st.session_state.brand_filter not in brands:
            st.session_state.brand_filter = "All"

        st.selectbox(
            "Brand",
            brands,
            key="brand_filter"
        )

    price_col, rating_col, reset_col = st.columns(3)

    with price_col:
        st.slider(
            "Price range (₹)",
            min_value=price_min,
            max_value=price_max,
            key="price_filter",
            step=1000.0,
            format="₹%.0f"
        )

    with rating_col:
        st.slider(
            "Minimum rating",
            min_value=0.0,
            max_value=5.0,
            step=0.5,
            key="rating_filter"
        )

    with reset_col:
        st.write("")
        st.write("")
        st.button(
            "Reset Filters",
            on_click=reset_filters,
            use_container_width=True
        )

    st.divider()


    if st.session_state.selected_product_id:
        selected_id = st.session_state.selected_product_id
        try:
            selected = get_product(selected_id)
        except ValueError:
            st.session_state.selected_product_id = None
            st.warning("The selected product is no longer available.")
        else:
            st.button("← Back to search results", on_click=clear_selected_product)
            st.markdown('<div class="section-title">Selected product</div>', unsafe_allow_html=True)

            with st.container(border=True):
                detail_icon, detail_info, detail_stats = st.columns([1, 4, 2])
                with detail_icon:
                    st.markdown(
                        f'<div class="product-icon" style="font-size:5rem;">'
                        f'{product_icon(selected["subcategory"])}</div>',
                        unsafe_allow_html=True,
                    )
                with detail_info:
                    st.subheader(str(selected["product_name"]))
                    st.write(
                        f"**{selected['brand']}** · {selected['category']} · "
                        f"{selected['subcategory']}"
                    )
                    st.write(f"Product ID: `{selected['product_id']}`")
                    st.write(f"⭐ {float(selected['rating']):.2f} from {int(selected['review_count']):,} reviews")
                with detail_stats:
                    st.metric("SmartBuy price", money(selected["final_price"]))
                    st.caption(
                        f"Original {money(selected['price'])} · "
                        f"{float(selected['discount']):.0f}% discount"
                    )

            similar_tab, also_tab, hybrid_tab = st.tabs(
                ["🔎 Similar Products", "🧑‍🤝‍🧑 Users Also Purchased", "🤖 Hybrid Picks"]
            )

            with similar_tab:
                st.caption("TF-IDF + cosine similarity using product content.")
                with st.spinner("Finding similar products..."):
                    similar = cached_similar(selected_id, 6)
                render_product_grid(similar, "similar", columns_per_row=3)

            with also_tab:
                st.caption(
                    "Products purchased by the same users. These are not necessarily from the same order."
                )
                with st.spinner("Analysing purchase patterns..."):
                    also = cached_also_purchased(selected_id, 6)
                render_product_grid(also, "also", columns_per_row=3)

            with hybrid_tab:
                st.caption("Combined content, collaborative, rating and popularity signals.")
                with st.spinner("Blending recommendation signals..."):
                    hybrid = cached_product_hybrid(selected_id, 6)
                render_product_grid(hybrid, "hybrid_product", columns_per_row=3)

    else:
        low_price, high_price = st.session_state.price_filter
        with st.spinner("Searching the SmartBuy catalog..."):
            results = cached_search(
                st.session_state.active_query,
                st.session_state.category_filter,
                st.session_state.subcategory_filter,
                st.session_state.brand_filter,
                float(low_price),
                float(high_price),
                float(st.session_state.rating_filter),
                60,
            )

        if st.session_state.active_query:
            heading = f'Results for “{html.escape(st.session_state.active_query)}”'
        else:
            heading = "Browse products"

        st.markdown(f'<div class="section-title">{heading}</div>', unsafe_allow_html=True)

        if results.empty:
            st.warning("No products match the current search and filters. Try resetting the filters.")
        else:
            st.caption(f"Showing {min(len(results), 12)} of the top {len(results)} matching products")
            render_product_grid(results.head(12), "search_result", columns_per_row=3)

        if st.session_state.view_history:
            viewed = products[
                products["product_id"].isin(st.session_state.view_history)
            ].copy()
            order = {product_id: index for index, product_id in enumerate(st.session_state.view_history)}
            viewed["_order"] = viewed["product_id"].map(order)
            viewed = viewed.sort_values("_order").drop(columns="_order").head(6)

            st.markdown('<div class="section-title">Recently viewed</div>', unsafe_allow_html=True)
            render_product_grid(viewed, "recent", columns_per_row=3)


# ============================================================
# Evaluation page
# ============================================================

elif page == "📊 Evaluation":

    st.markdown(
        """
        <div class="selected-panel">
            <h2>📊 Recommendation Model Comparison</h2>
            <p>
            Compare the three SmartBuy AI recommendation models using the same
            five evaluation metrics before selecting the best-performing model.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not EVALUATION_PATH.exists():
        st.warning(
            "Evaluation results are not available. Run "
            "`python evaluation/evaluate.py` first."
        )
    else:
        evaluation_metrics = pd.read_csv(EVALUATION_PATH)
        required_models = ["Content-Based", "Collaborative", "Hybrid"]
        required_metrics = ["RMSE", "MAE", "Precision", "Recall", "F1-score"]

        # Keep the Evaluation page focused on the five metrics required for
        # comparing the recommendation models.
        comparison = evaluation_metrics[
            evaluation_metrics["model"].isin(required_models)
            & evaluation_metrics["metric"].isin(required_metrics)
        ].copy()
        comparison["model"] = pd.Categorical(
            comparison["model"], categories=required_models, ordered=True
        )
        comparison["metric"] = pd.Categorical(
            comparison["metric"], categories=required_metrics, ordered=True
        )
        comparison = comparison.sort_values(["metric", "model"])

        st.markdown("### 📈 Five-Metric Model Comparison")
        st.caption(
            "RMSE and MAE: lower is better. Precision, Recall and F1-score: higher is better."
        )

        comparison_table = comparison.pivot(
            index="metric", columns="model", values="value"
        ).reindex(index=required_metrics, columns=required_models)
        comparison_table = comparison_table.rename(
            columns={
                "Content-Based": "Content-Based Model",
                "Collaborative": "Collaborative Model",
                "Hybrid": "Hybrid Model",
            }
        )
        st.dataframe(
            comparison_table.style.format("{:.4f}"),
            use_container_width=True,
        )

        st.markdown("### 🧪 Evaluation Method")
        st.info(
            """
            • The same leave-one-out test set is used for all three models.

            • Ratings of 4.0 or above are treated as relevant recommendations.

            • RMSE and MAE measure rating prediction error.

            • Precision, Recall and F1-score measure recommendation effectiveness.

            • The models are compared using the same test users and held-out ratings,
              so the comparison is consistent and fair.
            """
        )

        # Rank all five metrics together to identify one overall best model.
        ranking = comparison.pivot(index="metric", columns="model", values="value").reindex(required_metrics)
        rank_table = pd.DataFrame(index=required_models)
        for metric in required_metrics:
            rank_table[metric] = ranking.loc[metric].rank(
                ascending=metric in ["RMSE", "MAE"], method="average"
            )
        rank_table["Average Rank"] = rank_table.mean(axis=1)
        best_model = rank_table["Average Rank"].idxmin()

        st.markdown("### 🏆 Best Performing Model")

        # Show the winning model and its actual five evaluation results instead
        # of exposing the internal average-rank calculation to the user.
        best_metrics = comparison_table[f"{best_model} Model"]
        best_col1, best_col2 = st.columns([1.25, 2.75])
        with best_col1:
            st.success(
                f"**{best_model} Model**\n\n"
                "is the best overall performer based on the balanced comparison "
                "of all five evaluation metrics."
            )
        with best_col2:
            best_metric_cols = st.columns(5)
            for metric_col, metric in zip(best_metric_cols, required_metrics):
                with metric_col:
                    metric_col.metric(
                        metric,
                        f"{float(best_metrics.loc[metric]):.4f}"
                    )

        st.caption(
            f"{best_model} provides the strongest overall balance across RMSE, MAE, "
            "Precision, Recall and F1-score. Lower RMSE/MAE indicate smaller prediction "
            "errors, while higher Precision/Recall/F1-score indicate better recommendation quality."
        )

st.markdown("---")
st.caption(
    "SmartBuy AI · Academic recommender-system prototype · "
    "Synthetic product display names are identified by the SB- prefix."
)