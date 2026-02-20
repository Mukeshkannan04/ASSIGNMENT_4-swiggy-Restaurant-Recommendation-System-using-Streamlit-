import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Page Config
st.set_page_config(page_title="Swiggy Recommender", layout="wide")

# Load Data
@st.cache_data
def load_data():
    try:
        clean_df = pd.read_csv('cleaned_data.csv')
        enc_df = pd.read_csv('encoded_data.csv')
        return clean_df, enc_df
    except:
        return None, None

cleaned_df, encoded_df = load_data()

# App Interface
st.title("🍔 Swiggy Restaurant Recommender")

if cleaned_df is None:
    st.error("⚠️ Data files not found. Please run 'preprocess.py' first!")
    st.stop()

# Sidebar Filters
st.sidebar.header("Your Preferences")
city = st.sidebar.selectbox("Select City", sorted(cleaned_df['city'].unique()))

city_cuisines = sorted(cleaned_df[cleaned_df['city'] == city]['cuisine'].unique())
cuisine = st.sidebar.selectbox("Select Cuisine", city_cuisines)

budget = st.sidebar.slider("Max Budget (₹)", 100, 2000, 500)

if st.sidebar.button("Find Food"):
    # Filter by City
    city_indices = cleaned_df[cleaned_df['city'] == city].index
    city_vectors = encoded_df.iloc[city_indices]
    
    # Target Restaurant (Matching your cuisine choice)
    ref_rest = cleaned_df[(cleaned_df['city'] == city) & (cleaned_df['cuisine'] == cuisine)]
    
    if not ref_rest.empty:
        target_idx = ref_rest.index[0]
        target_vector = encoded_df.iloc[target_idx].values.reshape(1, -1)
        
        # Calculate Similarity
        scores = cosine_similarity(target_vector, city_vectors)[0]
        
        # Get results
        results = pd.DataFrame({'idx': city_indices, 'score': scores})
        top_recs = cleaned_df.iloc[results.sort_values(by='score', ascending=False)['idx']]
        
        # Filter by budget
        final_recs = top_recs[top_recs['cost'] <= budget].head(5)
        
        st.success(f"Top matches for {cuisine} in {city}:")
        for _, row in final_recs.iterrows():
            with st.container():
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.metric("Rating", f"⭐ {row['rating']}")
                with col2:
                    st.subheader(row['name'])
                    st.write(f"**Cost:** ₹{row['cost']} | **Address:** {row['address']}")
                    st.link_button("Order on Swiggy 🚀", str(row['link']).strip())
                st.divider()
    else:
        st.warning("No matches found for that specific cuisine in this city.")