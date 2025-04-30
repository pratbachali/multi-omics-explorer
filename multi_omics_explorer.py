# multi_omics_explorer.py

import streamlit as st
import pandas as pd
import spacy
import requests
import gzip
from Bio import Entrez
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import os
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go

# === CONFIGURATION ===
Entrez.email = "your-email@domain.com"  # Replace with your actual email
MATRIX_DIR = "matrix_files"
os.makedirs(MATRIX_DIR, exist_ok=True)
nlp = spacy.load("en_core_web_sm")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# === Predefined Disease Search Terms ===
disease_search_terms = {
    "Systemic Lupus Erythematosus": '(lupus OR "systemic lupus erythematosus" OR SLE)',
    "Rheumatoid Arthritis": '("rheumatoid arthritis" OR RA)',
    "Multiple Sclerosis": '("multiple sclerosis" OR MS)',
    "Type 1 Diabetes": '("type 1 diabetes" OR "juvenile diabetes" OR "insulin-dependent diabetes")',
    "Scleroderma": '("scleroderma" OR "systemic sclerosis")',
    "Psoriasis": '("psoriasis" OR "psoriatic disease")',
    "COVID-19": '("COVID-19" OR "SARS-CoV-2" OR "coronavirus disease 2019")',
    "Alzheimer's Disease": '("Alzheimer disease" OR "Alzheimer\'s disease" OR "AD" OR "dementia, Alzheimer type")',
    "Crohn's Disease": '("Crohn disease" OR "Crohn\'s disease" OR "regional enteritis")',
    "Inflammatory Bowel Disease (IBD)": '("inflammatory bowel disease" OR "IBD" OR "ulcerative colitis" OR "Crohn disease")'
}

# === PubMed Functions ===
def fetch_pubmed_metadata_custom(search_term, max_results, email, keyword_filter="", override_search_term=False):
    Entrez.email = email
    if override_search_term:
        search_term = f"{keyword_filter}[Author]"
    search_handle = Entrez.esearch(db="pubmed", term=search_term, retmax=max_results)
    ids = Entrez.read(search_handle).get("IdList", [])
    search_handle.close()
    if not ids:
        return []
    fetch_handle = Entrez.efetch(db="pubmed", id=",".join(ids), rettype="xml", retmode="xml")
    records = Entrez.read(fetch_handle)
    fetch_handle.close()

    results = []
    for article in records['PubmedArticle']:
        try:
            citation = article['MedlineCitation']
            article_data = citation['Article']
            abstract_parts = article_data.get('Abstract', {}).get('AbstractText', [])
            abstract = " ".join(map(str, abstract_parts)) if abstract_parts else "No Abstract"

            authors_list = []
            affiliations_list = []
            for author in article_data.get('AuthorList', []):
                name = f"{author.get('ForeName', '')} {author.get('LastName', '')}".strip()
                if name:
                    authors_list.append(name)
                for aff in author.get('AffiliationInfo', []):
                    aff_text = aff.get("Affiliation")
                    if aff_text:
                        affiliations_list.append(aff_text)
            combined_text = f"{abstract} {' '.join(authors_list)} {' '.join(affiliations_list)}"
            if keyword_filter and not override_search_term and keyword_filter.lower() not in combined_text.lower():
                continue
            journal_title = article_data.get("Journal", {}).get("Title", "N/A")
            pub_year = article_data.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {}).get("Year", "N/A")

            results.append({
                "pmid": str(citation['PMID']),
                "title": article_data.get("ArticleTitle", "No Title"),
                "abstract": abstract,
                "authors": ", ".join(authors_list) or "N/A",
                "affiliations": ", ".join(affiliations_list) or "N/A",
                "journal": journal_title,
                "date": pub_year
            })
        except Exception as e:
            print(f"Error processing article: {e}")
            continue
    return results

def preprocess_text(text):
    return " ".join([token.lemma_ for token in nlp(text) if not token.is_stop and token.is_alpha])

def reduce_dimensions(embeddings):
    return UMAP(n_neighbors=15, min_dist=0.1, metric='cosine').fit_transform(embeddings)

# === GEO Functions ===
def get_gse_metadata(keyword, max_results, start_year, start_month, end_year, end_month):
    date_filter = f" AND ({start_year}/{start_month:02d}/01[PDAT] : {end_year}/{end_month:02d}/31[PDAT])"
    full_query = f"{keyword}{date_filter}"
    search_handle = Entrez.esearch(db="gds", term=full_query, retmax=max_results)
    search_results = Entrez.read(search_handle)
    search_handle.close()
    uid_list = search_results.get("IdList", [])
    if not uid_list:
        return []
    summary_handle = Entrez.esummary(db="gds", id=",".join(uid_list))
    summary_list = Entrez.read(summary_handle)
    summary_handle.close()
    results = []
    for doc in summary_list:
        if doc.get("Accession", "").startswith("GSE"):
            gse_id = doc.get("Accession")
            pmid = doc.get("PubMedId", "")
            if not pmid:
                try:
                    link_handle = Entrez.elink(dbfrom="gds", db="pubmed", id=gse_id)
                    link_result = Entrez.read(link_handle)
                    link_handle.close()

                    # Extract PMID if exists
                    link_sets = link_result[0].get("LinkSetDb", [])
                    if link_sets:
                        pmid = link_sets[0]["Link"][0]["Id"]
                except Exception as e:
                    print(f"Error fetching PMID for {gse_id}: {e}")

            results.append({
                "GSE_ID": gse_id,
                "Title": doc.get("title", ""),
                "Organism": doc.get("taxon", "Unknown"),
                "Platform": doc.get("GPL", "Unknown"),
                "PMID": pmid,
                "PubDate": doc.get("PDAT", "Unknown")
            })
    return results

def download_series_matrix(gse_id):
    prefix = gse_id[:-3] + "nnn"
    url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/{gse_id}/matrix/{gse_id}_series_matrix.txt.gz"
    output_path = os.path.join(MATRIX_DIR, f"{gse_id}_series_matrix.txt.gz")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return output_path
    return None

def parse_series_matrix_to_df(matrix_path):
    sample_metadata = {}
    characteristics = []
    with gzip.open(matrix_path, 'rt') as f:
        for line in f:
            if line.startswith("!Sample_"):
                parts = line.strip().split('\t')
                field = parts[0].replace("!Sample_", "")
                values = parts[1:]
                if field == "characteristics_ch1":
                    characteristics = values
                else:
                    sample_metadata[field] = values
            if line.startswith("!series_matrix_table_begin"):
                break
    df = pd.DataFrame(sample_metadata)
    parsed = []
    for char in characteristics:
        parsed_dict = {}
        for item in char.split(";"):
            if ":" in item:
                key, value = item.split(":", 1)
                parsed_dict[key.strip().lower()] = value.strip()
        parsed.append(parsed_dict)
    char_df = pd.DataFrame(parsed)
    df_full = pd.concat([df, char_df], axis=1)
    if "geo_accession" in df:
        df_full.index = df["geo_accession"]
        df_full.index.name = "GSM_ID"
    return df_full

def run_dimensionality_reduction(df, method="UMAP", color_by=None):
    numeric_df = pd.get_dummies(df.select_dtypes(include=["number", "category", "bool", "object"]))
    scaled = StandardScaler().fit_transform(numeric_df)
    if method == "UMAP":
        reducer = UMAP(n_components=2, random_state=42)
    elif method == "PCA":
        reducer = PCA(n_components=2)
    elif method == "t-SNE":
        reducer = TSNE(n_components=2, random_state=42)
    else:
        raise ValueError("Invalid method")
    embedding = reducer.fit_transform(scaled)
    df_vis = pd.DataFrame(embedding, columns=["X", "Y"], index=df.index)
    df_vis[color_by] = df[color_by] if color_by in df.columns else "Unknown"
    return df_vis

def plot_embedding(df_vis, color_by, save_path="geo_embedding.png"):
    fig = px.scatter(
        df_vis,
        x="X", y="Y",
        color=color_by,
        title=f"Sample Clustering by '{color_by}'",
        hover_name=df_vis.index,
        template="simple_white",  # Clean style with minimal background
        color_discrete_sequence=px.colors.qualitative.Safe,
    )

    # Larger marker size with border
    fig.update_traces(marker=dict(size=14, line=dict(width=1, color='DarkSlateGrey')))

    # Remove gridlines and enhance fonts
    fig.update_layout(
        autosize=True,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=30, r=30, t=50, b=30),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            title="X",
            title_font=dict(size=16),
            tickfont=dict(size=14)
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            title="Y",
            title_font=dict(size=16),
            tickfont=dict(size=14)
        ),
        legend_title=dict(text=color_by, font=dict(size=16)),
        legend=dict(font=dict(size=14)),
        title=dict(font=dict(size=20)),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Save image and enable download
    fig.write_image(save_path)
    with open(save_path, "rb") as f:
        st.download_button(
            "📸 Download Clustering Plot (PNG)",
            f,
            file_name=save_path,
            mime="image/png"
        )


def plot_summary(df, column):
    st.subheader(f"Summary of '{column}'")
    value_counts = df[column].value_counts().dropna()
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "bar"}, {"type": "pie"}]])
    fig.add_trace(go.Bar(x=value_counts.index, y=value_counts.values, name="Bar"), row=1, col=1)
    fig.add_trace(go.Pie(labels=value_counts.index, values=value_counts.values, name="Pie"), row=1, col=2)
    fig.update_layout(height=400, title_text=f"Distribution of '{column}'", showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

# === MAIN APP UI ===
def main():
    st.set_page_config(layout="wide")
    st.title("🧬 Multi-Omics Explorer: PubMed + GEO Analysis")

    tab1, tab2 = st.tabs(["🔬 PubMed Topic Modeling", "📊 GEO Dataset Exploration"])

    # === TAB 1: PUBMED ===
    with tab1:
        st.markdown("### 🔍 PubMed Topic Modeling via BERTopic")

        col1, col2 = st.columns([1.5, 1])
        with col1:
            disease = st.selectbox("Select Disease", list(disease_search_terms.keys()))
            custom_query = st.text_input("Or enter custom PubMed query")
        with col2:
            email = st.text_input("Email (required)", "")

        search_term = custom_query if custom_query.strip() else disease_search_terms[disease]
        max_results = st.slider("Max results", 50, 3000, 200)
        keyword_filter = st.text_input("Optional keyword filter (e.g. author or topic)")
        exclude_mouse = st.checkbox("Exclude mouse studies")

        if st.button("🔎 Run PubMed Search & Topic Modeling"):
            if not email:
                st.warning("Email is required for PubMed access.")
                return

            override = bool(keyword_filter and all(w[0].isupper() for w in keyword_filter.split()))
            data = fetch_pubmed_metadata_custom(search_term, max_results, email, keyword_filter, override)

            if exclude_mouse:
                data = [
                    entry for entry in data
                    if not any(x in entry["abstract"].lower() for x in ["mouse", "mice", "murine", "rodent"])
                ]

            df_pubmed = pd.DataFrame(data)
            if df_pubmed.empty:
                st.warning("No articles found.")
                return

            st.dataframe(df_pubmed[["pmid", "title", "date", "journal", "authors", "abstract"]])
            st.download_button("📥 Download Abstracts CSV", df_pubmed.to_csv(index=False).encode("utf-8"), "pubmed_abstracts.csv", "text/csv")

            if len(df_pubmed) < 10:
                st.warning("Not enough abstracts for topic modeling.")
                return

            abstracts_raw = df_pubmed["abstract"].tolist()
            preprocessed_abstracts = [preprocess_text(ab) for ab in abstracts_raw]
            filtered = [(p, a) for p, a in zip(preprocessed_abstracts, abstracts_raw) if p.strip()]
            if not filtered:
                st.warning("No valid abstracts after preprocessing.")
                return

            preprocessed_abstracts, abstracts_cleaned = map(list, zip(*filtered))
            embeddings = embedding_model.encode(preprocessed_abstracts, show_progress_bar=True)
            stopwords = set(["autoimmune", "disease", "patients", "inflammation", "treatment", "clinical", "study", "response"])
            all_stopwords = list(text.ENGLISH_STOP_WORDS.union(stopwords))

            vectorizer = CountVectorizer(stop_words=all_stopwords)
            topic_model = BERTopic(vectorizer_model=vectorizer)
            topics, _ = topic_model.fit_transform(preprocessed_abstracts)

            reduced_coords = reduce_dimensions(embeddings)
            df_reduced = pd.DataFrame(reduced_coords, columns=["x", "y"])
            df_reduced["Topic"] = topics
            df_reduced["Abstract"] = abstracts_cleaned

            for col in ["title", "pmid", "journal", "authors", "affiliations", "date"]:
                df_reduced[col] = df_pubmed[col].values[:len(df_reduced)]

            fig = px.scatter(
                df_reduced,
                x="x", y="y",
                color=df_reduced["Topic"].astype(str),
                hover_data=["Abstract"],
                title="BERTopic Clustering of PubMed Abstracts"
            )
            st.plotly_chart(fig)

            st.dataframe(topic_model.get_topic_info())
            with st.expander("📄 View Abstracts and Metadata by Topic"):
                for topic in sorted(df_reduced["Topic"].unique()):
                    st.markdown(f"### Topic {topic}")
                    topic_df = df_reduced[df_reduced["Topic"] == topic]
                    for _, row in topic_df.iterrows():
                        st.markdown(f"**🔗 PMID:** [{row['pmid']}]")
                        st.markdown(f"**📄 Title:** {row['title']}")
                        st.markdown(f"**🏥 Journal:** {row['journal']}")
                        st.markdown(f"**👨‍🔬 Authors:** {row['authors']}")
                        st.markdown(f"**📅 Year:** {row['date']}")
                        st.markdown(f"**📝 Abstract:** {row['Abstract']}")
                        st.markdown("---")

    # === TAB 2: GEO ===
    with tab2:
        st.markdown("### 📊 Explore GEO Datasets (Gene Expression Omnibus)")

        col1, col2, col3 = st.columns(3)
        with col1:
            keyword = st.text_input("🧬 Disease keyword (e.g. lung cancer):")
        with col2:
            start_date = st.text_input("Start date (YYYY-MM):", "2015-01")
        with col3:
            end_date = st.text_input("End date (YYYY-MM):", "2020-12")

        if keyword and start_date and end_date:
            try:
                start_y, start_m = map(int, start_date.split("-"))
                end_y, end_m = map(int, end_date.split("-"))
                metadata_list = get_gse_metadata(keyword, 200, start_y, start_m, end_y, end_m)
            except:
                st.error("⚠️ Invalid date format. Use YYYY-MM.")
                return

            if metadata_list:
                df_gse = pd.DataFrame(metadata_list)
                
                # Add clickable PubMed links
                #df_gse["PMID_Link"] = df_gse["PMID"].apply(lambda x: f"[{x}](https://pubmed.ncbi.nlm.nih.gov/{x}/)" if x else "N/A")

                st.markdown("#### 📚 Matching GEO Series")
                species = sorted(df_gse["Organism"].dropna().unique())
                selected_species = st.multiselect("🐾 Filter by species:", species)
                if selected_species:
                    df_gse = df_gse[df_gse["Organism"].isin(selected_species)]

                st.dataframe(df_gse, use_container_width=True)
                #st.markdown("#### 📚 Matching GEO Series (Click PMID to view PubMed)")
                #display_cols = ["GSE_ID", "Title", "Organism", "Platform", "PubDate", "PMID_Link"]
                #st.markdown(df_gse[display_cols].to_markdown(index=False), unsafe_allow_html=True)
                st.download_button("📥 Download GSE Metadata CSV", df_gse.to_csv(index=False).encode("utf-8"), "filtered_gse_metadata.csv", "text/csv")

                selected_gse = st.selectbox("Select GSE to explore:", df_gse["GSE_ID"])
                if selected_gse:
                    path = download_series_matrix(selected_gse)
                    if path:
                        df_samples = parse_series_matrix_to_df(path)

                        st.markdown("#### 🧪 Sample Metadata")
                        st.dataframe(df_samples, use_container_width=True)

                        st.download_button(
                            "📥 Download Sample Metadata CSV",
                            df_samples.to_csv().encode("utf-8"),
                            f"{selected_gse}_samples.csv",
                            "text/csv"
                        )

                        if not df_samples.empty:
                            st.markdown("### 📈 Explore Sample Metadata")
                            col1, col2 = st.columns([1.3, 1])
                            with col1:
                                col_to_summarize = st.selectbox("Summarize column:", df_samples.columns)
                            with col2:
                                method = st.selectbox("Dimensionality reduction:", ["UMAP", "PCA", "t-SNE"])

                            color_by = st.selectbox("Color samples by:", df_samples.columns)

                            st.markdown("---")
                            if st.button("🚀 Run Clustering"):
                                try:
                                    df_vis = run_dimensionality_reduction(df_samples, method, color_by)
                                    plot_embedding(df_vis, color_by)
                                except Exception as e:
                                    st.error(f"Clustering failed: {e}")

                            if col_to_summarize:
                                plot_summary(df_samples, col_to_summarize)
                    else:
                        st.error(f"Failed to download series matrix for {selected_gse}.")
            else:
                st.warning("No GEO datasets found for the given criteria.")

if __name__ == "__main__":
    main()
