
"""SocialPulse - Large-Scale Social Media Intelligence Platform."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st


# ==========================================================
# CONFIGURATION
# ==========================================================

ROOT = Path(__file__).resolve().parents[2]

YOUTUBE_DIR = ROOT / "data" / "processed" / "youtube"
TOPIC_DIR = YOUTUBE_DIR / "topics"
GRAPH_DIR = ROOT / "data" / "processed" / "graph"
MISINFO_DIR = ROOT / "data" / "processed" / "misinformation"

st.set_page_config(
    page_title="SocialPulse | Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================================
# PROFESSIONAL THEME
# ==========================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #F4F7FB;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #10233F 0%, #18365C 100%);
    }

    [data-testid="stSidebar"] * {
        color: #F1F5FC !important;
    }

    [data-testid="stSidebar"] hr {
        border-color: #385273;
    }

    .main-title {
        color: #142B49;
        font-size: 30px;
        font-weight: 800;
        margin-bottom: 2px;
    }

    .subtitle {
        color: #62748B;
        font-size: 14px;
        margin-bottom: 22px;
    }

    .section-title {
        color: #193653;
        font-size: 21px;
        font-weight: 700;
        margin-top: 15px;
        margin-bottom: 12px;
    }

    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2EAF3;
        border-radius: 13px;
        padding: 17px 19px;
        min-height: 112px;
        box-shadow: 0 2px 9px rgba(22, 48, 78, 0.045);
    }

    .metric-label {
        color: #65778D;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .metric-value {
        color: #173B65;
        font-size: 27px;
        font-weight: 800;
        margin-top: 8px;
    }

    .metric-note {
        color: #8493A5;
        font-size: 11px;
        margin-top: 3px;
    }

    .info-card {
        background: #FFFFFF;
        border: 1px solid #E2EAF3;
        border-left: 4px solid #3983D5;
        border-radius: 9px;
        padding: 14px 17px;
        margin: 9px 0 15px 0;
        color: #42566F;
        font-size: 13px;
        line-height: 1.65;
    }

    .success-card {
        background: #EAF6F0;
        border-left: 4px solid #27966A;
        border-radius: 8px;
        padding: 12px 15px;
        color: #246B4E;
        font-size: 13px;
        margin: 10px 0;
    }

    .warning-card {
        background: #FFF6E8;
        border-left: 4px solid #D89A32;
        border-radius: 8px;
        padding: 12px 15px;
        color: #805B20;
        font-size: 13px;
        margin: 10px 0;
    }

    .workflow-step {
        background: #FFFFFF;
        border: 1px solid #E1E9F2;
        border-radius: 10px;
        padding: 14px;
        min-height: 125px;
    }

    .workflow-number {
        color: #3983D5;
        font-size: 12px;
        font-weight: 800;
    }

    .workflow-heading {
        color: #193653;
        font-size: 14px;
        font-weight: 700;
        margin: 7px 0;
    }

    .workflow-description {
        color: #687A90;
        font-size: 12px;
        line-height: 1.5;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #E2EAF3;
        padding: 12px 15px;
        border-radius: 11px;
    }

    div[data-testid="stMetricLabel"] {
        color: #62748B;
    }

    div[data-testid="stMetricValue"] {
        color: #173B65;
        font-weight: 750;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #E2EAF3;
        border-radius: 9px;
    }

    .footer {
        text-align: center;
        color: #8A99AA;
        font-size: 11px;
        padding: 15px;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================================
# DATA LOADERS
# ==========================================================

@st.cache_data
def load_csv(path_string):
    path = Path(path_string)

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_json(path_string):
    path = Path(path_string)

    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return {}


videos = load_csv(str(YOUTUBE_DIR / "cleaned_videos.csv"))
comments = load_csv(str(YOUTUBE_DIR / "cleaned_comments.csv"))
sentiment = load_csv(str(YOUTUBE_DIR / "sentiment_results.csv"))

topic_summary = load_csv(str(TOPIC_DIR / "topic_summary.csv"))
topic_trends = load_csv(str(TOPIC_DIR / "topic_trends.csv"))
topic_keywords = load_csv(str(TOPIC_DIR / "topic_keywords.csv"))
comment_topics = load_csv(str(TOPIC_DIR / "comment_topics.csv"))
topic_report = load_json(str(TOPIC_DIR / "topic_model_report.json"))

graph_report = load_json(str(GRAPH_DIR / "graph_report.json"))
pagerank = load_csv(str(GRAPH_DIR / "pagerank_scores.csv"))
communities = load_csv(str(GRAPH_DIR / "community_assignments.csv"))
edges = load_csv(str(GRAPH_DIR / "graph_edges.csv"))

misinfo_metrics = load_json(
    str(MISINFO_DIR / "misinformation_metrics.json")
)


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def section(title):
    st.markdown(
        f'<div class="section-title">{title}</div>',
        unsafe_allow_html=True
    )


def info_box(title, message):
    st.markdown(
        f"""
        <div class="info-card">
            <b>{title}</b><br>{message}
        </div>
        """,
        unsafe_allow_html=True
    )


def metric_card(label, value, note=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def safe_number(value, default=0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


def format_count(value):
    return f"{int(safe_number(value)):,}"


def show_empty(message):
    st.info(message)


def clean_sentiment_data():
    if sentiment.empty or "sentiment" not in sentiment.columns:
        return pd.DataFrame()

    data = sentiment.copy()
    data["sentiment"] = (
        data["sentiment"].astype(str).str.lower().str.strip()
    )
    return data


def show_explanation(title, method, interpretation):
    info_box(
        title,
        f"<b>How it was generated:</b> {method}<br>"
        f"<b>How to interpret:</b> {interpretation}"
    )


def show_chart(data, title, explanation=None):
    section(title)

    if data is None or len(data) == 0:
        show_empty("No data available for this visualization.")
        return

    st.bar_chart(data, use_container_width=True)

    if explanation:
        st.caption(explanation)


# ==========================================================
# SIDEBAR NAVIGATION
# ==========================================================

st.sidebar.markdown(
    """
    <div style="padding: 10px 0 18px 0;">
        <div style="font-size:25px;font-weight:800;">📊 SocialPulse</div>
        <div style="font-size:12px;color:#C0D0E2;">
            Social Media Intelligence
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Overview",
        "Sentiment Analysis",
        "Topic Analysis",
        "Graph Analytics",
        "Misinformation Detection",
        "Methodology & Limitations",
    ],
)

st.sidebar.divider()

st.sidebar.markdown("**DATA SOURCES**")
st.sidebar.caption("YouTube Data API v3")
st.sidebar.caption("LIAR benchmark dataset")

st.sidebar.divider()

st.sidebar.caption("NetworkX | Neo4j")
st.sidebar.caption("BERTopic | XLM-RoBERTa")
st.sidebar.caption("TF-IDF | Logistic Regression")

if st.sidebar.button("🔄 Refresh dashboard data"):
    st.cache_data.clear()
    st.rerun()


# ==========================================================
# GLOBAL HEADER
# ==========================================================

st.markdown(
    '<div class="main-title">🌐 Social Media Intelligence Platform</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Understand online conversations through multilingual sentiment, '
    'topic discovery, interaction networks and misinformation classification.'
    '</div>',
    unsafe_allow_html=True
)


# ==========================================================
# OVERVIEW
# ==========================================================

if page == "Overview":

    section("Executive Overview")

    st.markdown("""
    <div class="info-card">
        <b>Project objective</b><br>
        This platform analyzes social media conversations to identify
        audience sentiment, discover discussion topics, understand
        interaction patterns and classify potentially misleading claims.
        It combines Natural Language Processing, Machine Learning and
        Graph Analytics in one analytical workflow.
    </div>
    """, unsafe_allow_html=True)

    total_videos = len(videos)
    total_comments = len(comments)
    total_sentiment = len(sentiment)

    total_authors = graph_report.get("nodes", len(pagerank))
    total_topics = 0

    if "Topic" in topic_summary.columns:
        topic_ids = pd.to_numeric(
            topic_summary["Topic"], errors="coerce"
        )
        total_topics = int(topic_ids[topic_ids != -1].nunique())
    elif not topic_summary.empty:
        total_topics = len(topic_summary)

    total_communities = graph_report.get("communities_detected", 0)

    section("Platform at a Glance")

    columns = st.columns(4)

    with columns[0]:
        metric_card("Videos Collected", format_count(total_videos),
                    "Cleaned video records")

    with columns[1]:
        metric_card("Comments Cleaned", format_count(total_comments),
                    "Preprocessed comment records")

    with columns[2]:
        metric_card("Sentiment Predictions", format_count(total_sentiment),
                    "Model-processed comments")

    with columns[3]:
        metric_card("Topics Discovered", format_count(total_topics),
                    "BERTopic, excluding outlier topic")

    st.write("")

    columns = st.columns(3)

    with columns[0]:
        metric_card("Graph Authors", format_count(total_authors),
                    "Pseudonymized author nodes")

    with columns[1]:
        metric_card("Communities", format_count(total_communities),
                    "NetworkX detected groups")

    with columns[2]:
        metric_card(
            "Reply Interactions",
            format_count(graph_report.get("directed_edges", 0)),
            "Directed graph edges"
        )

    section("How the Platform Works")

    workflow = [
        ("01", "Data Collection",
         "Collect video and comment data through YouTube Data API v3."),
        ("02", "Preprocessing",
         "Clean text, remove duplicates and detect languages."),
        ("03", "Sentiment Analysis",
         "Classify comments using multilingual XLM-RoBERTa."),
        ("04", "Topic Discovery",
         "Discover discussion themes using BERTopic."),
        ("05", "Graph Analytics",
         "Analyze reply interactions using NetworkX and Neo4j."),
        ("06", "Misinformation",
         "Evaluate a TF-IDF and Logistic Regression baseline on LIAR."),
    ]

    columns = st.columns(3)

    for index, item in enumerate(workflow):
        with columns[index % 3]:
            st.markdown(
                f"""
                <div class="workflow-step">
                    <div class="workflow-number">STEP {item[0]}</div>
                    <div class="workflow-heading">{item[1]}</div>
                    <div class="workflow-description">{item[2]}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    section("Sentiment Snapshot")

    sentiment_data = clean_sentiment_data()

    if not sentiment_data.empty:

        counts = sentiment_data["sentiment"].value_counts()

        sentiment_colors = {
            "positive": "#239B73",
            "neutral": "#E4A33B",
            "negative": "#D85B65",
        }

        chart_counts = pd.DataFrame({
            "Comments": counts
        })

        st.bar_chart(chart_counts, color="#3983D5")

        total = len(sentiment_data)

        cols = st.columns(3)

        for index, label in enumerate(
            ["positive", "neutral", "negative"]
        ):
            count = int(counts.get(label, 0))
            percentage = count / total * 100 if total else 0

            with cols[index]:
                metric_card(
                    label.title(),
                    f"{percentage:.1f}%",
                    f"{count:,} comments"
                )

        show_explanation(
            "Understanding the sentiment results",
            "Each processed comment receives a sentiment prediction "
            "from the multilingual XLM-RoBERTa model.",
            "The percentages show the distribution of model predictions, "
            "not verified opinions or a representative survey."
        )

    else:
        show_empty("Sentiment results have not been generated yet.")

    section("Most Frequent Discussion Topics")

    if not topic_summary.empty and "Count" in topic_summary.columns:

        display_topics = topic_summary.copy()
        display_topics["Count"] = pd.to_numeric(
            display_topics["Count"], errors="coerce"
        )
        display_topics = display_topics.dropna(subset=["Count"])
        display_topics = display_topics[
            display_topics["Topic"].astype(str) != "-1"
        ] if "Topic" in display_topics.columns else display_topics

        display_topics = display_topics.sort_values(
            "Count", ascending=False
        ).head(10)

        if not display_topics.empty:
            chart_data = display_topics.set_index(
                "Name" if "Name" in display_topics.columns else "Topic"
            )[["Count"]]

            st.bar_chart(chart_data, color="#3983D5")

            cols = [
                col for col in ["Topic", "Name", "Count"]
                if col in display_topics.columns
            ]

            st.dataframe(
                display_topics[cols],
                use_container_width=True,
                hide_index=True
            )

    else:
        show_empty("Topic summary is not available.")

    st.markdown("""
    <div class="warning-card">
        <b>Data-source transparency:</b>
        Reddit API access approval was unavailable. Therefore, this
        implementation uses data collected through YouTube Data API v3
        for social-media text and interaction analysis. The LIAR dataset
        is used separately for misinformation-classifier evaluation.
    </div>
    """, unsafe_allow_html=True)


# ==========================================================
# SENTIMENT ANALYSIS
# ==========================================================

elif page == "Sentiment Analysis":

    section("😊 Multilingual Sentiment Analysis")

    st.markdown("""
    <div class="info-card">
        <b>Purpose:</b> Identify the overall emotional tone of comments
        and understand how positive, neutral and negative discussions
        are distributed across the collected dataset.
    </div>
    """, unsafe_allow_html=True)

    data = clean_sentiment_data()

    if data.empty:
        show_empty("Sentiment results are not available.")
    else:

        counts = data["sentiment"].value_counts()

        total = len(data)
        positive = int(counts.get("positive", 0))
        neutral = int(counts.get("neutral", 0))
        negative = int(counts.get("negative", 0))

        columns = st.columns(4)

        for column, label, count in zip(
            columns,
            ["Total Analyzed", "Positive", "Neutral", "Negative"],
            [total, positive, neutral, negative]
        ):
            with column:
                metric_card(label, format_count(count),
                            "Model prediction count")

        section("Sentiment Distribution")

        chart_data = pd.DataFrame({
            "Number of Comments": counts
        })

        st.bar_chart(chart_data, color="#3983D5")

        percentage_data = (
            counts / total * 100
        ).round(2).rename("Percentage")

        section("Sentiment Percentage")

        st.dataframe(
            percentage_data.to_frame(),
            use_container_width=True
        )

        show_explanation(
            "What does the distribution tell us?",
            "The XLM-RoBERTa multilingual sentiment model assigns each "
            "comment to a sentiment category.",
            "A higher positive share means more comments were classified "
            "as positive by the model. It does not prove that all users "
            "or the entire platform have that opinion."
        )

        if "confidence" in data.columns:

            data["confidence"] = pd.to_numeric(
                data["confidence"], errors="coerce"
            )

            average_confidence = data["confidence"].mean()

            section("Model Confidence")

            if pd.notna(average_confidence):
                st.metric(
                    "Average Prediction Confidence",
                    f"{average_confidence:.3f}"
                )

            st.caption(
                "Confidence is the model's reported confidence score. "
                "It is not the same as independently measured accuracy."
            )

        if "language" in data.columns:

            section("Language Distribution")

            language_counts = (
                data["language"]
                .fillna("unknown")
                .astype(str)
                .value_counts()
                .head(15)
            )

            st.bar_chart(
                language_counts.to_frame("Comments"),
                color="#3983D5"
            )

            st.caption(
                "Language labels are generated during preprocessing."
            )

        if "published_at" in data.columns:

            section("Sentiment Trends Over Time")

            time_data = data.copy()

            time_data["published_at"] = pd.to_datetime(
                time_data["published_at"],
                errors="coerce",
                utc=True
            )

            time_data = time_data.dropna(
                subset=["published_at"]
            )

            if not time_data.empty:

                time_data["date"] = (
                    time_data["published_at"].dt.date
                )

                trend = (
                    time_data.groupby(["date", "sentiment"])
                    .size()
                    .unstack(fill_value=0)
                    .sort_index()
                )

                st.line_chart(trend)

                st.caption(
                    "Daily number of comments assigned to each sentiment."
                )
            else:
                show_empty("Valid publication dates are unavailable.")

        section("Model Information")

        st.markdown("""
        - **Model:** CardiffNLP multilingual XLM-RoBERTa
        - **Model identifier:** `cardiffnlp/twitter-xlm-roberta-base-sentiment`
        - **Task:** Three-class sentiment classification
        - **Output:** Positive, Neutral and Negative
        - **Input:** Cleaned social-media comments
        """)

        st.warning(
            "These are automated predictions. Sarcasm, context, slang "
            "and mixed-language expressions can affect model performance."
        )


# ==========================================================
# TOPIC ANALYSIS
# ==========================================================

elif page == "Topic Analysis":

    section("🔍 BERTopic - Discussion Topic Discovery")

    st.markdown("""
    <div class="info-card">
        <b>Purpose:</b> Discover recurring themes in social-media
        conversations without manually defining every topic in advance.
        BERTopic groups semantically similar comments and extracts
        representative keywords.
    </div>
    """, unsafe_allow_html=True)

    if topic_summary.empty:

        show_empty("Topic summary is not available. Run the topic model first.")

    else:

        summary = topic_summary.copy()

        if "Count" in summary.columns:
            summary["Count"] = pd.to_numeric(
                summary["Count"], errors="coerce"
            )

        if "Topic" in summary.columns:
            topic_ids = pd.to_numeric(
                summary["Topic"], errors="coerce"
            )
            non_outlier = summary[topic_ids != -1]
            topic_count = non_outlier["Topic"].nunique()
        else:
            non_outlier = summary
            topic_count = len(summary)

        assignments = len(comment_topics)

        columns = st.columns(3)

        with columns[0]:
            metric_card("Topics Discovered", format_count(topic_count),
                        "Excluding outlier topic")

        with columns[1]:
            metric_card("Comment Assignments", format_count(assignments),
                        "BERTopic output records")

        with columns[2]:
            metric_card("Topic Trend Records", format_count(len(topic_trends)),
                        "Available daily topic records")

        section("Most Frequent Topics")

        if "Count" in non_outlier.columns:

            top_topics = non_outlier.dropna(
                subset=["Count"]
            ).sort_values("Count", ascending=False).head(20)

            if not top_topics.empty:

                index_col = (
                    "Name" if "Name" in top_topics.columns
                    else "Topic"
                )

                chart = top_topics.set_index(index_col)[["Count"]]

                st.bar_chart(chart, color="#3983D5")

                st.caption(
                    "The chart displays the 20 most frequent non-outlier topics."
                )

        section("Topic Summary Table")

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )

        show_explanation(
            "How should we interpret the topics?",
            "BERTopic combines multilingual sentence embeddings, "
            "dimensionality reduction, clustering and keyword extraction.",
            "Topic names and keywords summarize recurring language patterns. "
            "They are not guaranteed to represent perfectly separated themes."
        )

    section("Topic Keywords")

    if not topic_keywords.empty:
        st.dataframe(
            topic_keywords,
            use_container_width=True,
            hide_index=True
        )
    else:
        show_empty("Topic keywords are not available.")

    section("Topic Trends Over Time")

    if not topic_trends.empty and {
        "date", "topic_id", "comment_count"
    }.issubset(topic_trends.columns):

        trends = topic_trends.copy()

        trends["date"] = pd.to_datetime(
            trends["date"], errors="coerce"
        )

        trends["comment_count"] = pd.to_numeric(
            trends["comment_count"], errors="coerce"
        )

        trends = trends.dropna(
            subset=["date", "comment_count"]
        )

        if not trends.empty:

            pivot = trends.pivot_table(
                index="date",
                columns="topic_id",
                values="comment_count",
                aggfunc="sum",
                fill_value=0
            ).sort_index()

            options = list(pivot.columns)

            selected = st.multiselect(
                "Choose topics for trend comparison",
                options=options,
                default=options[:min(5, len(options))]
            )

            if selected:
                st.line_chart(pivot[selected])

            st.caption(
                "Daily comment volume assigned to each selected topic."
            )

    else:
        show_empty("Topic trend data is not available.")

    if topic_report:

        with st.expander("View BERTopic model report"):
            st.json(topic_report)

    st.markdown("""
    <div class="warning-card">
        <b>Important:</b> BERTopic may assign some comments to topic -1,
        which represents outliers that were not assigned to a regular
        topic cluster. Not every discovered cluster is necessarily
        a meaningful discussion theme.
    </div>
    """, unsafe_allow_html=True)


# ==========================================================
# GRAPH ANALYTICS
# ==========================================================

elif page == "Graph Analytics":

    section("🕸️ Social Interaction Graph Analytics")

    st.markdown("""
    <div class="info-card">
        <b>Purpose:</b> Understand how authors interact through replies.
        The graph represents reply relationships between pseudonymized
        channel authors. NetworkX calculates graph metrics and identifies
        communities; Neo4j stores and visualizes the graph.
    </div>
    """, unsafe_allow_html=True)

    nodes = graph_report.get("nodes", len(pagerank))
    active = graph_report.get("active_authors", 0)
    isolates = graph_report.get("isolated_authors", 0)
    directed_edges = graph_report.get("directed_edges", len(edges))
    community_count = graph_report.get("communities_detected", 0)

    columns = st.columns(4)

    metrics = [
        ("Total Authors", nodes, "All author nodes"),
        ("Active Authors", active, "Authors in reply relationships"),
        ("Reply Interactions", directed_edges, "Directed graph edges"),
        ("Communities", community_count, "Detected groups"),
    ]

    for column, metric in zip(columns, metrics):
        with column:
            metric_card(
                metric[0],
                format_count(metric[1]),
                metric[2]
            )

    section("Network Structure")

    columns = st.columns(3)

    with columns[0]:
        metric_card(
            "Isolated Authors",
            format_count(isolates),
            "Authors without active reply links"
        )

    with columns[1]:
        metric_card(
            "Average Degree",
            f"{safe_number(graph_report.get('average_degree', 0)):.4f}",
            "Average connectivity"
        )

    with columns[2]:
        metric_card(
            "Graph Modularity",
            f"{safe_number(graph_report.get('modularity', 0)):.4f}",
            "Community structure measure"
        )

    show_explanation(
        "Understanding graph metrics",
        "Authors become nodes and reply relationships become directed edges. "
        "NetworkX calculates PageRank and uses greedy modularity community "
        "detection on the undirected active graph.",
        "PageRank estimates structural importance within this graph. "
        "Modularity measures how strongly the detected groups are separated. "
        "Neither metric proves real-world influence."
    )

    section("Top Authors by PageRank")

    if not pagerank.empty and {
        "anonymous_author_id", "pagerank_score"
    }.issubset(pagerank.columns):

        ranking = pagerank.copy()

        ranking["pagerank_score"] = pd.to_numeric(
            ranking["pagerank_score"], errors="coerce"
        )

        ranking = ranking.dropna(
            subset=["pagerank_score"]
        ).sort_values(
            "pagerank_score", ascending=False
        ).head(10)

        if not ranking.empty:

            chart = ranking.set_index(
                "anonymous_author_id"
            )[["pagerank_score"]]

            st.bar_chart(chart, color="#3983D5")

            st.dataframe(
                ranking,
                use_container_width=True,
                hide_index=True
            )

    else:
        show_empty("PageRank results are not available.")

    section("Community Distribution")

    if not communities.empty and "community_id" in communities.columns:

        community_counts = (
            communities["community_id"]
            .value_counts()
            .sort_index()
        )

        st.bar_chart(
            community_counts.to_frame("Authors"),
            color="#3983D5"
        )

        st.caption(
            "Community ID -1 represents isolated authors not assigned "
            "to a detected interaction community."
        )

    else:
        show_empty("Community assignments are not available.")

    section("Reply Interaction Records")

    if not edges.empty:

        st.dataframe(
            edges.head(100),
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            f"Showing up to 100 of {len(edges):,} interaction records."
        )

    else:
        show_empty("Graph edge data is not available.")

    st.markdown("""
    <div class="warning-card">
        <b>Privacy and interpretation:</b> Author IDs are pseudonymized
        using SHA-256. PageRank reflects structural position in the
        collected reply graph, not personal identity or verified influence.
        A sparse graph can also affect how community metrics should be interpreted.
    </div>
    """, unsafe_allow_html=True)


# ==========================================================
# MISINFORMATION DETECTION
# ==========================================================

elif page == "Misinformation Detection":

    section("🚩 Misinformation Classification")

    st.markdown("""
    <div class="info-card">
        <b>Purpose:</b> Evaluate a baseline text-classification model
        trained on the LIAR benchmark dataset. The model learns linguistic
        patterns associated with the dataset's grouped labels.
    </div>
    """, unsafe_allow_html=True)

    if not misinfo_metrics:

        show_empty("Misinformation evaluation metrics are not available.")

    else:

        test = misinfo_metrics.get("test", {})
        validation = misinfo_metrics.get("validation", {})

        section("Test Set Performance")

        accuracy = safe_number(test.get("accuracy", 0))
        precision = safe_number(test.get("misleading_precision", 0))
        recall = safe_number(test.get("misleading_recall", 0))
        f1 = safe_number(test.get("misleading_f1", 0))

        columns = st.columns(4)

        results = [
            ("Test Accuracy", accuracy * 100, "Overall correct predictions"),
            ("Misleading Precision", precision * 100, "Correct positive predictions"),
            ("Misleading Recall", recall * 100, "Misleading cases identified"),
            ("Misleading F1", f1 * 100, "Precision-recall balance"),
        ]

        for column, result in zip(columns, results):
            with column:
                metric_card(
                    result[0],
                    f"{result[1]:.2f}%",
                    result[2]
                )

        section("Validation vs Test Accuracy")

        accuracy_comparison = pd.DataFrame({
            "Accuracy (%)": {
                "Validation": safe_number(
                    validation.get("accuracy", 0)
                ) * 100,
                "Test": accuracy * 100,
            }
        })

        st.bar_chart(
            accuracy_comparison,
            color="#3983D5"
        )

        section("Test Confusion Matrix")

        matrix = test.get("confusion_matrix", [])

        if (
            isinstance(matrix, list)
            and len(matrix) == 2
            and all(isinstance(row, list) and len(row) == 2
                    for row in matrix)
        ):

            matrix_df = pd.DataFrame(
                matrix,
                index=[
                    "Actual: Misleading",
                    "Actual: Not classified as misleading",
                ],
                columns=[
                    "Predicted: Misleading",
                    "Predicted: Not classified as misleading",
                ]
            )

            st.dataframe(
                matrix_df,
                use_container_width=True
            )

            st.caption(
                "Rows represent actual labels; columns represent model predictions."
            )

        else:
            show_empty("A valid confusion matrix is not available.")

        show_explanation(
            "How to read the confusion matrix",
            "The test dataset is passed through the trained TF-IDF and "
            "Logistic Regression pipeline. Predictions are compared "
            "against the grouped benchmark labels.",
            "Correct predictions appear on the diagonal. Off-diagonal "
            "values represent classification errors."
        )

        section("Test Dataset Class Balance")

        balance = test.get("class_balance", {})

        if balance:

            balance_series = pd.Series(
                balance, name="Number of Samples"
            )

            st.bar_chart(
                balance_series.to_frame(),
                color="#3983D5"
            )

        else:
            show_empty("Test class-balance data is not available.")

        section("Model Configuration")

        st.markdown("""
        - **Dataset:** LIAR benchmark dataset
        - **Feature extraction:** TF-IDF
        - **Classifier:** Logistic Regression
        - **Task:** Binary classification using grouped labels
        - **Positive class:** Misleading
        """)

        st.markdown("""
        <div class="warning-card">
            <b>Important limitation:</b> This is a baseline classifier,
            not an independent fact-checking system. Its output reflects
            patterns learned from LIAR dataset labels. It does not establish
            whether a real-world statement is true or false. Predictions
            should be reviewed by a human fact-checker.
        </div>
        """, unsafe_allow_html=True)


# ==========================================================
# METHODOLOGY AND LIMITATIONS
# ==========================================================

elif page == "Methodology & Limitations":

    section("Project Methodology")

    st.markdown("""
    This project combines data collection, Natural Language Processing,
    unsupervised topic modelling, graph analytics and supervised
    classification into a single analytical workflow.
    """)

    section("Technology Stack")

    technology = pd.DataFrame([
        ["Data Collection", "YouTube Data API v3"],
        ["Data Processing", "Python, Pandas"],
        ["Language Detection", "langdetect"],
        ["Sentiment Analysis", "CardiffNLP XLM-RoBERTa"],
        ["Topic Modelling", "BERTopic, Sentence Transformers, UMAP, HDBSCAN"],
        ["Graph Analysis", "NetworkX, PageRank, Community Detection"],
        ["Graph Database", "Neo4j"],
        ["Misinformation Baseline", "LIAR, TF-IDF, Logistic Regression"],
        ["Dashboard", "Streamlit"],
    ], columns=["Component", "Technology"])

    st.dataframe(
        technology,
        use_container_width=True,
        hide_index=True
    )

    section("Data Sources")

    st.markdown("""
    **1. YouTube Data API v3**

    Video and comment records are collected using the official API.
    The collected records are cleaned and used for sentiment, topic
    and reply-network analysis.

    **2. LIAR benchmark dataset**

    Used separately to train and evaluate the misinformation
    classification baseline.
    """)

    section("Project Limitations")

    limitations = [
        (
            "Reddit API access",
            "Reddit API approval was unavailable. Unauthorized Reddit "
            "collection was not attempted. The implemented social-media "
            "analysis therefore uses YouTube API data."
        ),
        (
            "Sentiment classification",
            "Automated sentiment models can misinterpret sarcasm, slang, "
            "context and mixed-language comments."
        ),
        (
            "Topic modelling",
            "Some comments may be assigned to the outlier topic. "
            "Not every topic cluster is necessarily meaningful."
        ),
        (
            "Graph sparsity",
            "The reply graph contains isolated authors and relatively "
            "few reply relationships compared with the total number "
            "of authors."
        ),
        (
            "Misinformation baseline",
            "LIAR labels are grouped into two classes for a simplified "
            "classification task. The model is not a real-world "
            "fact-verification system."
        ),
        (
            "Data coverage",
            "YouTube API access, disabled comments, pagination and "
            "collection settings affect which records are available."
        ),
    ]

    for title, description in limitations:
        info_box(title, description)

    section("Responsible Interpretation")

    st.markdown("""
    - Model predictions should not be treated as verified facts.
    - Pseudonymized graph identifiers are used instead of displaying
      channel identities.
    - Dataset limitations should be considered before drawing conclusions.
    - The dashboard reports computed results rather than inventing
      performance values.
    """)


# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.markdown("""
<div class="footer">
    <b>SocialPulse</b> | Large-Scale Social Media Intelligence Platform<br>
    Academic Data Science Project | YouTube Data API v3 + LIAR Benchmark<br>
    Built with Python, Streamlit, NLP, Machine Learning and Graph Analytics
</div>
""", unsafe_allow_html=True)
