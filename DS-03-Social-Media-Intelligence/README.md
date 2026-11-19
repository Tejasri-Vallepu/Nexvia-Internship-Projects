# SocialPulse – Large-Scale Social Media Intelligence Platform

An end-to-end Data Science project that analyzes social media conversations using Natural Language Processing, Machine Learning, Topic Modeling, and Graph Analytics.

SocialPulse transforms collected social media comments into meaningful insights by identifying sentiment patterns, discovering discussion topics, analyzing interaction networks, and evaluating a baseline misinformation classifier.

---

## Project Overview

Social media platforms generate large volumes of user discussions every day. Understanding these conversations manually is difficult.

SocialPulse addresses this challenge through a modular analytical pipeline that:

- Collects video and comment data using the YouTube Data API v3.
- Cleans text, removes duplicates, and detects languages.
- Performs multilingual sentiment classification.
- Discovers discussion topics using BERTopic.
- Analyzes author interaction networks using NetworkX and Neo4j.
- Evaluates a misinformation classification baseline using the LIAR dataset.
- Presents analytical results through an interactive Streamlit dashboard.

## Dashboard

The Streamlit dashboard provides a unified interface for exploring the project's analytical results.

### Dashboard modules

1. **Overview** – Project summary, data statistics, workflow, and sentiment snapshot.
2. **Sentiment Analysis** – Sentiment distribution, confidence, language distribution, and trends.
3. **Topic Analysis** – Discovered topics, keywords, and topic trends.
4. **Graph Analytics** – Author interactions, PageRank, and community distribution.
5. **Misinformation Detection** – Classification metrics and confusion matrix.
6. **Methodology & Limitations** – Technology stack, data sources, and responsible interpretation.

To launch the dashboard:

```bash
streamlit run src/dashboard/app.py
```

## Architecture and Workflow

```text
             YouTube Data API v3
                      |
                      v
             Raw JSON Data Storage
                      |
                      v
                Preprocessing
           Cleaning and Language Detection
                      |
          +-----------+-----------+
          |           |           |
          v           v           v
      Sentiment     Topic       Graph
      Analysis     Modeling    Analytics
          |           |           |
          v           v           v
       XLM-R       BERTopic    NetworkX
                                Neo4j

          LIAR Benchmark Dataset
                      |
                      v
             Text Classification
             TF-IDF + Logistic
                Regression

                      |
                      v
              Streamlit Dashboard
                      |
                      v
             Analytical Insights
```

## Technology Stack

| Component | Technologies |
|---|---|
| Programming | Python |
| Data Collection | YouTube Data API v3 |
| Data Processing | Pandas, NumPy |
| Language Detection | langdetect |
| Sentiment Analysis | Transformers, XLM-RoBERTa |
| Topic Modeling | BERTopic, Sentence Transformers, UMAP, HDBSCAN |
| Graph Analysis | NetworkX, PageRank, Community Detection |
| Graph Database | Neo4j |
| Misinformation Classification | Scikit-learn, TF-IDF, Logistic Regression |
| Dashboard | Streamlit |
| Data Storage | JSON, CSV |

## Implemented Modules

### 1. Data Collection

Collected video and comment records using the YouTube Data API v3.

Collection includes:
- Video metadata.
- Comment text.
- Publication timestamps.
- Like counts.
- Available author-channel identifiers.
- Top-level comments and replies returned by the API.

The collector stores raw records as JSON for downstream processing.

### 2. Data Preprocessing

Implemented a preprocessing pipeline to:
- Clean comment text.
- Remove duplicate records.
- Detect comment languages.
- Prepare structured CSV files for analysis.

Main outputs:

```text
data/processed/youtube/
    cleaned_videos.csv
    cleaned_comments.csv
    data_quality_report.json
```

### 3. Multilingual Sentiment Analysis

**Model:** CardiffNLP multilingual XLM-RoBERTa

Model identifier:

`cardiffnlp/twitter-xlm-roberta-base-sentiment`

The model classifies comments into:

- Positive
- Neutral
- Negative

The pipeline stores sentiment predictions and confidence scores.

Main output:

```text
data/processed/youtube/sentiment_results.csv
```

Previously confirmed evaluation snapshot:

| Sentiment | Comments | Percentage |
|---|---:|---:|
| Positive | 2,173 | 46.2% |
| Neutral | 1,393 | 29.6% |
| Negative | 1,142 | 24.3% |
| **Total** | **4,708** | **100%** |

These figures represent an earlier confirmed sentiment run. The dashboard displays the records currently available in the output CSV.

### 4. Topic Modeling with BERTopic

BERTopic is used to discover recurring themes in social media discussions.

Components:
- Multilingual sentence embeddings.
- UMAP dimensionality reduction.
- HDBSCAN clustering.
- CountVectorizer keyword extraction.

Embedding model:

`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

Previously confirmed run:

| Metric | Result |
|---|---:|
| Unique comments used | 4,368 |
| Topics discovered | 119 |
| Outlier comments | 1,027 |
| Assigned comments | 3,341 |
| Outlier percentage | 23.51% |

Outlier topic `-1` represents comments that were not assigned to regular topic clusters.

Main outputs:

```text
data/processed/youtube/topics/
    comment_topics.csv
    topic_keywords.csv
    topic_summary.csv
    topic_trends.csv
    topic_model_report.json
```

### 5. Graph Analytics

A directed reply-interaction graph is constructed using author relationships.

Implemented analytics:
- Author nodes and reply edges.
- Interaction counts.
- PageRank scores.
- Community detection using greedy modularity.
- Graph storage and visualization using Neo4j.

NetworkX graph results from the latest confirmed graph report:

| Metric | Result |
|---|---:|
| Author nodes | 4,352 |
| Active authors | 1,013 |
| Isolated authors | 3,339 |
| Directed edges | 917 |
| Valid reply relationships | 994 |
| Communities detected | 172 |
| Modularity | 0.9140 |

Author identifiers are pseudonymized using SHA-256.

**Interpretation:** PageRank estimates structural importance within the collected interaction graph. It does not establish real-world influence. The graph is sparse, so community metrics should be interpreted carefully.

Main outputs:

```text
data/processed/graph/
    graph_nodes.csv
    graph_edges.csv
    pagerank_scores.csv
    community_assignments.csv
    graph_report.json
```

### 6. Misinformation Classification

A baseline text-classification pipeline is trained and evaluated using the LIAR benchmark dataset.

**Model pipeline:**
- TF-IDF feature extraction.
- Logistic Regression classifier.
- Binary label grouping.

Label mapping:

| Original LIAR labels | Grouped label |
|---|---|
| pants-fire, false, barely-true | misleading |
| half-true, mostly-true, true | not_classified_as_misleading |

Test-set results:

| Metric | Result |
|---|---:|
| Accuracy | 59.94% |
| Misleading Precision | 53.61% |
| Misleading Recall | 56.12% |
| Misleading F1-score | 54.83% |

These results represent a baseline model with moderate classification performance.

**Important limitation:** The classifier learns patterns associated with grouped LIAR dataset labels. It is not an independent fact-checking system and cannot establish whether a real-world statement is true or false. Human review is required.

Saved model:

```text
models/misinformation/liar_tfidf_logistic_regression.joblib
```

Evaluation outputs:

```text
data/processed/misinformation/
    misinformation_metrics.json
    test_classification_report.txt
    test_confusion_matrix.csv
    validation_classification_report.txt
    validation_confusion_matrix.csv
```

---

## Project Structure

```text
Large-Scale Social Media Intelligence Platform/
│
├── data/
│   ├── raw/
│   │   └── youtube/
│   ├── processed/
│   │   ├── youtube/
│   │   │   └── topics/
│   │   ├── graph/
│   │   └── misinformation/
│   └── external/
│       └── liar/
│
├── models/
│   ├── bertopic/
│   └── misinformation/
│
├── src/
│   ├── preprocessing/
│   ├── sentiment/
│   ├── topics/
│   ├── graph/
│   ├── misinformation/
│   └── dashboard/
│       └── app.py
│
├── docs/
├── reports/
├── tests/
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Installation and Setup

### Requirements

- Python 3.10 or 3.11
- pip
- Git
- YouTube Data API v3 credentials for data collection
- Neo4j Desktop for graph visualization (optional for dashboard viewing)

### 1. Clone the repository

```bash
git clone https://github.com/Tejasri-Vallepu/Large-Scale-Social-Media-Intelligence-Platform.git
cd Large-Scale-Social-Media-Intelligence-Platform
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
YOUTUBE_API_KEY=your_youtube_api_key
```

Never upload API keys or secrets to GitHub.

### 5. Run the dashboard

```bash
streamlit run src/dashboard/app.py
```

The dashboard reads the generated CSV and JSON analysis files from the project data directories.

## Running the Analysis Modules

Run the implemented modules from the project root with the virtual environment activated.

```bash
python -m src.preprocessing.process_youtube_data
```

```bash
python -m src.sentiment.sentiment_analyzer
```

```bash
python -m src.topics.topic_model
```

```bash
python -m src.graph.graph_analyzer
```

```bash
python -m src.misinformation.fake_news_classifier
```

Data collection should be run using the configured YouTube collector module before preprocessing. Keep API credentials in `.env`.

## Data Source and Access Limitations

### YouTube Data API v3

YouTube Data API v3 is used for video and comment collection, subject to API availability, quota limits and comment availability.

### LIAR Benchmark Dataset

The LIAR dataset is used specifically for misinformation-classification training and evaluation.

### Reddit API limitation

Reddit API access approval was not available during project development. To comply with platform access policies, unauthorized Reddit collection was not attempted.

Therefore, the implemented social-media text and interaction analysis uses YouTube API data rather than claiming Reddit API integration.

## Privacy and Responsible Data Usage

- API credentials are stored in environment variables.
- Private or restricted content is not intentionally collected.
- Author identifiers in graph outputs are pseudonymized.
- Raw data should not be redistributed unless permitted by its applicable terms and license.
- Model predictions are not treated as verified facts.
- Analytical findings should be interpreted in the context of data coverage and model limitations.

## Current Project Status

| Module | Status |
|---|---|
| YouTube API data collection | Implemented |
| Text preprocessing | Implemented |
| Language detection | Implemented |
| Multilingual sentiment analysis | Implemented |
| BERTopic topic modeling | Implemented |
| NetworkX graph analytics | Implemented |
| Neo4j graph visualization | Implemented |
| LIAR baseline classifier | Implemented |
| Streamlit dashboard | Implemented |
| Elasticsearch and Kibana | Not implemented |

## Future Improvements

- Expand permitted data-source coverage.
- Improve topic interpretation and outlier handling.
- Evaluate sentiment performance using manually reviewed samples.
- Explore more advanced graph analysis.
- Improve misinformation classification with additional evaluation and explainability.
- Add Elasticsearch and Kibana for search and monitoring.
- Add automated tests and reproducible experiment tracking.

## Author

**Tejasri Vallepu**

- GitHub: https://github.com/Tejasri-Vallepu
- LinkedIn: https://www.linkedin.com/in/tejasri-vallepu/

## Project Purpose

Developed as a Data Science specialization project to demonstrate practical skills in API-based data collection, Natural Language Processing, Machine Learning, unsupervised learning, graph analytics, and interactive data visualization.
