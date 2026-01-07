



# main.py - Streamlit Interface for Content-Based Recommender System
import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import hstack, csr_matrix, load_npz, save_npz
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PATH CONFIGURATION
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# The dataset lives in the sibling `data/` directory next to `code/`.
DATA_DIR = os.path.normpath(os.path.join(BASE_DIR, '..', 'data'))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
TABLES_DIR = os.path.join(RESULTS_DIR, 'tables')
CONTENT_BASED_DIR = os.path.join(TABLES_DIR, 'content_based')

# Create directories if they don't exist
os.makedirs(CONTENT_BASED_DIR, exist_ok=True)
DATA_FILE = os.path.join(DATA_DIR, 'cleaned_financial_data.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================================
# CACHE DECORATORS FOR PERFORMANCE
# ============================================================================
@st.cache_data
def load_data():
    """Load and cache the dataset"""
    df = pd.read_csv(DATA_FILE)
    return df

@st.cache_resource
def load_or_create_models():
    """Load or create the feature models"""
    # Define paths for saved models
    tfidf_path = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')
    ohe_path = os.path.join(MODEL_DIR, 'onehot_encoder.pkl')
    item_features_path = os.path.join(MODEL_DIR, 'item_features.npz')
    item_ids_path = os.path.join(MODEL_DIR, 'item_ids.pkl')
    df_path = os.path.join(MODEL_DIR, 'processed_df.pkl')
    
    # Load existing models if available
    if all([os.path.exists(p) for p in [tfidf_path, ohe_path, item_features_path, 
                                        item_ids_path, df_path]]):
        with open(tfidf_path, 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        with open(ohe_path, 'rb') as f:
            ohe = pickle.load(f)
        item_features = load_npz(item_features_path)
        with open(item_ids_path, 'rb') as f:
            item_ids = pickle.load(f)
        df = pd.read_pickle(df_path)
        
        print("✅ Loaded pre-trained models from cache")
        return df, tfidf_vectorizer, ohe, item_features, item_ids
    
    # If models don't exist, create them
    print("🔄 Creating new models (first run)")
    df = load_data()
    
    # Prepare text features
    text_cols = ['title', 'description', 'summary']
    df['text_combined'] = df[text_cols].fillna('').agg(' '.join, axis=1)
    
    # TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
    text_features = tfidf_vectorizer.fit_transform(df['text_combined'])
    
    # One-Hot Encoder
    cat_cols = ['primary_topic', 'subtopic', 'difficulty', 'content_type']
    try:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    except:
        ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
    categorical_features = ohe.fit_transform(df[cat_cols])
    
    # Numerical features
    knowledge_map = {'beginner': 1, 'intermediate': 2, 'advanced': 3}
    df['financial_knowledge_num'] = df['financial_knowledge'].map(knowledge_map)
    num_features = df[['financial_knowledge_num']].values
    
    # Combine all features
    item_features = hstack([text_features, categorical_features, num_features]).tocsr()
    item_ids = df['item_id'].values
    
    # Save models for future use
    with open(tfidf_path, 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
    with open(ohe_path, 'wb') as f:
        pickle.dump(ohe, f)
    save_npz(item_features_path, item_features)
    with open(item_ids_path, 'wb') as f:
        pickle.dump(item_ids, f)
    df.to_pickle(df_path)
    
    print("✅ Models created and saved to cache")
    return df, tfidf_vectorizer, ohe, item_features, item_ids

@st.cache_data
def load_precomputed_recommendations():
    """Load precomputed recommendations if available"""
    top10_path = os.path.join(CONTENT_BASED_DIR, 'top_10_recommendations.csv')
    top20_path = os.path.join(CONTENT_BASED_DIR, 'top_20_recommendations.csv')
    
    if os.path.exists(top10_path) and os.path.exists(top20_path):
        top10_df = pd.read_csv(top10_path)
        top20_df = pd.read_csv(top20_path)
        return top10_df, top20_df
    return None, None


def inject_sidebar_style():
    """Inject CSS to style the Streamlit sidebar/navigation."""
    st.markdown(
        """
        <style>
        /* Sidebar background and padding */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f1724 0%, #071028 100%);
            padding: 18px 12px 24px 18px;
            color: #e6eef8;
        }

        /* Sidebar title style */
        .sidebar-title {
            font-size: 20px;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 8px;
        }

        /* Radio / widget labels inside sidebar */
        [data-testid="stSidebar"] label, [data-testid="stSidebar"] .streamlit-expanderHeader {
            color: #dbeafe !important;
            font-weight: 600;
        }

        /* Make selectboxes and textareas look lighter */
        [data-testid="stSidebar"] .stTextArea, [data-testid="stSidebar"] .stSelectbox {
            color: #dbeafe;
        }

        /* Nice rounded buttons */
        [data-testid="stSidebar"] .stButton>button {
            background-color: #2563eb !important;
            color: white !important;
            border-radius: 8px !important;
            padding: 6px 10px !important;
        }

        /* Subtle hover for radio choices */
        [data-testid="stSidebar"] .stRadio>div>label:hover {
            background: rgba(255,255,255,0.03);
            border-radius: 6px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def sidebar_navigation():
    """Render a modern-looking sidebar navigation using buttons and return page selection."""
    # Ensure a persistent page state
    if 'page' not in st.session_state:
        st.session_state['page'] = 'Home'

    # Title already injected via CSS-styled markdown, but keep a clear label
    st.sidebar.markdown("<div class='sidebar-title'>  System </div> ", unsafe_allow_html=True)

    # Render modern buttons (full-width, rounded, with icons)
    if st.sidebar.button("  Home", key="nav_home" , width=250 , ):
        st.session_state['page'] = 'Home'
    if st.sidebar.button("  New User Registration", key="nav_register", width=250):
        st.session_state['page'] = 'New User Registration'
    if st.sidebar.button("  Recommendations", key="nav_recs", width=250):
        st.session_state['page'] = 'Recommendations'
    if st.sidebar.button("  System Info", key="nav_info", width=250):
        st.session_state['page'] = 'System Info'

    # A small spacer and subtle footer in the sidebar
    st.sidebar.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.sidebar.markdown("<div style='font-size:12px;color:#9fb3d8;margin-top:6px'>v1 • Content-based Recommender</div>", unsafe_allow_html=True)

    return st.session_state['page']

# ============================================================================
# COLD-START USER HANDLING FUNCTIONS
# ============================================================================
def create_cold_start_user_profile(user_preferences, df, tfidf_vectorizer, ohe):
    """
    Create a user profile for a cold-start user based on preferences
    """
    # Extract user preferences
    financial_knowledge = user_preferences.get('financial_knowledge', '')
    primary_topic = user_preferences.get('primary_topic', '')
    subtopic = user_preferences.get('subtopic', '')
    difficulty = user_preferences.get('difficulty', '')
    content_type = user_preferences.get('content_type', '')
    interests = user_preferences.get('interests', '')

    # Build text parts: include only fields that are not 'non'
    text_parts = []
    if primary_topic and primary_topic != 'non':
        text_parts.append(f"User Preference: {primary_topic}")
    # Always include free-text 'interests' if provided
    if isinstance(interests, str) and interests.strip():
        text_parts.append(interests.strip())
    if subtopic and subtopic != 'non':
        text_parts.append(str(subtopic))
    if primary_topic and primary_topic != 'non' and subtopic and subtopic != 'non':
        text_parts.append(f"Interests in {primary_topic}, {subtopic}")

    # Fallback to an empty string if nothing provided
    text_combined = ' '.join(text_parts).strip() if text_parts else ''

    # Transform using existing TF-IDF model (empty string will produce zero vector)
    text_features = tfidf_vectorizer.transform([text_combined])

    # Prepare categorical data: for any 'non' selection, pass an unknown value so
    # OneHotEncoder will produce zeros (handle_unknown='ignore')
    cat_vals = [primary_topic if primary_topic != 'non' else '',
                subtopic if subtopic != 'non' else '',
                difficulty if difficulty != 'non' else '',
                content_type if content_type != 'non' else '']

    try:
        categorical_features = ohe.transform([cat_vals])
    except Exception:
        # If transform fails for any reason, fall back to zeros of appropriate width
        try:
            n_cat_cols = sum(len(c) for c in getattr(ohe, 'categories_', []))
        except Exception:
            n_cat_cols = 0
        categorical_features = np.zeros((1, n_cat_cols))

    # Prepare numerical feature: if user selected 'non', set numeric to 0 to minimize influence
    knowledge_map = {'beginner': 1, 'intermediate': 2, 'advanced': 3}
    if financial_knowledge and financial_knowledge != 'non':
        financial_knowledge_num = knowledge_map.get(financial_knowledge, 2)
    else:
        financial_knowledge_num = 0
    num_features = np.array([[financial_knowledge_num]])

    # Combine features
    user_profile = hstack([text_features, categorical_features, num_features])

    return user_profile

def get_recommendations_for_cold_start(user_profile, item_features, item_ids, df, top_n=10):
    """
    Get recommendations for a cold-start user with diverse items
    """
    # Calculate similarity between user profile and all items
    similarities = cosine_similarity(user_profile, item_features).flatten()
    
    # Create a DataFrame with item info and similarities
    item_sim_df = pd.DataFrame({
        'item_id': item_ids,
        'score': similarities
    })
    
    # Merge with original df to get item details
    item_details = df[['item_id', 'title', 'primary_topic', 'subtopic', 
                       'difficulty', 'content_type']].drop_duplicates(subset=['item_id'])
    item_sim_df = item_sim_df.merge(item_details, on='item_id', how='left')
    
    # Sort by score descending
    item_sim_df = item_sim_df.sort_values('score', ascending=False)
    
    # Remove duplicates to ensure unique items
    item_sim_df = item_sim_df.drop_duplicates(subset=['item_id'])
    
    # Ensure we have enough items
    if len(item_sim_df) < top_n:
        # If not enough items, take what we have
        top_n = len(item_sim_df)
    
    # Get top N items - this ensures different items
    top_items = item_sim_df.head(top_n).copy()
    
    # Reset index for display
    top_items.reset_index(drop=True, inplace=True)
    
    return top_items

# ============================================================================
# STREAMLIT INTERFACE
# ============================================================================
def main():
    st.set_page_config(
        page_title="Financial Literacy Recommender System",
        
        layout="wide"
    )
    
    # Load data and models
    with st.spinner("Loading recommender system..."):
        df, tfidf_vectorizer, ohe, item_features, item_ids = load_or_create_models()
        top10_df, top20_df = load_precomputed_recommendations()
    
    # Sidebar for navigation (styled)
    inject_sidebar_style()
    # Use modern sidebar navigation buttons (replaces the old radio/checkpoint style)
    page = sidebar_navigation()
    
    # Main content
    st.title("Financial Literacy Recommender System")
    
    if page == "Home":
        st.header("Welcome to the Financial Literacy Recommender")
        st.markdown("""
        ### About This System
        
        This system provides personalized financial literacy content recommendations 
        based on your knowledge level and interests.
        
        **Features:**
        - Content-based filtering using TF-IDF and categorical features
        - Cold-start user handling
        - Top-N recommendations
        - Item-based collaborative filtering (KNN)
        
        **Get Started:**
        1. Go to **New User Registration** to create your profile
        2. Receive personalized recommendations
        3. Explore different financial topics
        """)
        
        # Show dataset statistics
        st.subheader("Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Users", df['user_id'].nunique())
        with col2:
            st.metric("Total Items", df['item_id'].nunique())
        with col3:
            st.metric("Topics", df['primary_topic'].nunique())
        with col4:
            st.metric("Avg Rating", f"{df['rating'].mean():.2f}")
    
    elif page == "New User Registration":
        st.header("👤 New User Registration")
        st.markdown("Please fill in your preferences to get personalized recommendations:")
        
        with st.form("user_preferences_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                financial_knowledge = st.selectbox(
                    "Financial Knowledge Level",
                    ["non", "beginner", "intermediate", "advanced"],
                    help="Select your current financial knowledge level"
                )
                
                primary_topic_options = ["non"] + df['primary_topic'].dropna().unique().tolist()
                primary_topic = st.selectbox(
                    "Primary Topic of Interest",
                    primary_topic_options,
                    help="Select your main area of interest"
                )
                
                # Filter subtopics based on primary topic; always include 'non'
                if primary_topic == "non":
                    subtopics = ["non"]
                else:
                    subtopics = ["non"] + df[df['primary_topic'] == primary_topic]['subtopic'].dropna().unique().tolist()
                subtopic = st.selectbox(
                    "Subtopic of Interest",
                    subtopics,
                    help="Select specific subtopic"
                )
            
            with col2:
                difficulty_options = ["non"] + df['difficulty'].dropna().unique().tolist()
                difficulty = st.selectbox(
                    "Preferred Difficulty Level",
                    difficulty_options,
                    help="Select preferred difficulty"
                )
                
                content_type_options = ["non"] + df['content_type'].dropna().unique().tolist()
                content_type = st.selectbox(
                    "Preferred Content Type",
                    content_type_options,
                    help="Select type of content you prefer"
                )
                
                interests = st.text_area(
                    "Additional Interests (Optional)",
                    placeholder="e.g., stocks, retirement planning, crypto investments...",
                    help="Describe your specific interests in your own words"
                )
            
            # Submit button
            submitted = st.form_submit_button("Get Recommendations")
            
            if submitted:
                with st.spinner("Creating your profile and generating recommendations..."):
                    # Create user preferences dictionary
                    user_preferences = {
                        'financial_knowledge': financial_knowledge,
                        'primary_topic': primary_topic,
                        'subtopic': subtopic,
                        'difficulty': difficulty,
                        'content_type': content_type,
                        'interests': interests
                    }
                    
                    # Store in session state
                    st.session_state['user_preferences'] = user_preferences
                    
                    # Generate a temporary user ID for new user
                    existing_user_ids = df['user_id'].unique()
                    new_user_id = max(existing_user_ids) + 1 if len(existing_user_ids) > 0 else 100000
                    st.session_state['temp_user_id'] = new_user_id
                    
                    # Create user profile
                    user_profile = create_cold_start_user_profile(
                        user_preferences, df, tfidf_vectorizer, ohe
                    )
                    
                    # Get recommendations
                    recommendations = get_recommendations_for_cold_start(
                        user_profile, item_features, item_ids, df, top_n=10
                    )
                    
                    # Store recommendations in session
                    st.session_state['recommendations'] = recommendations
                    
                    st.success("✅ Profile created successfully!")
                    st.balloons()
                    
                    # Show message below the form/button
                    st.markdown("---")
                    st.success("🎯 **Go to the 'Recommendations' page in the sidebar to see your personalized recommendations!**")
                    
                    # Show user profile summary
                    st.subheader("Your Profile Summary")
                    profile_cols = st.columns(3)
                    with profile_cols[0]:
                        st.info(f"**Knowledge:** {financial_knowledge.title()}")
                    with profile_cols[1]:
                        st.info(f"**Topic:** {primary_topic}")
                    with profile_cols[2]:
                        st.info(f"**Difficulty:** {difficulty.title()}")
                    
                    # Remove the auto-redirect (st.rerun())
                    # User can now manually click on Recommendations in sidebar
    
    elif page == "Recommendations":
        st.header("📚 Your Personalized Recommendations")
        
        if st.session_state.get('recommendations') is None:
            st.warning("Please complete the New User Registration first!")
            st.info("Go to the **New User Registration** page to create your profile.")
        else:
            recommendations = st.session_state['recommendations']
            user_prefs = st.session_state.get('user_preferences', {})
            
            # Display user preferences
            with st.expander("Your Preferences", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Financial Knowledge:** {user_prefs.get('financial_knowledge', 'N/A')}")
                    st.write(f"**Primary Topic:** {user_prefs.get('primary_topic', 'N/A')}")
                with col2:
                    st.write(f"**Difficulty:** {user_prefs.get('difficulty', 'N/A')}")
                    st.write(f"**Content Type:** {user_prefs.get('content_type', 'N/A')}")
            
            # Display recommendations in the format matching your code
            st.subheader(f"Top {len(recommendations)} Recommended Items")
            
            # Create a DataFrame that matches the format from your code
            display_columns = ['item_id', 'title', 'primary_topic', 'subtopic', 
                              'difficulty', 'content_type', 'score']
            
            # Ensure we have all required columns
            recommendations_display = recommendations.copy()
            
            # Select and reorder columns to match your format
            available_columns = [col for col in display_columns if col in recommendations_display.columns]
            recommendations_display = recommendations_display[available_columns]
            
            # Add user_id column (for new user, we'll use the temporary ID)
            temp_user_id = st.session_state.get('temp_user_id', 'new_user')
            recommendations_display.insert(0, 'user_id', temp_user_id)
            
            # Check if items are unique
            unique_items = recommendations_display['item_id'].nunique()
            total_items = len(recommendations_display)
            
            if unique_items < total_items:
                st.warning(f"Warning: Found {total_items - unique_items} duplicate items. Removing duplicates...")
                recommendations_display = recommendations_display.drop_duplicates(subset=['item_id'])
            
            # Display the table
            st.dataframe(
                recommendations_display,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "user_id": st.column_config.TextColumn("User ID"),
                    "item_id": st.column_config.TextColumn("Item ID"),
                    "title": st.column_config.TextColumn("Title"),
                    "primary_topic": st.column_config.TextColumn("Primary Topic"),
                    "subtopic": st.column_config.TextColumn("Subtopic"),
                    "difficulty": st.column_config.TextColumn("Difficulty"),
                    "content_type": st.column_config.TextColumn("Content Type"),
                    "score": st.column_config.NumberColumn(
                        "Score",
                        format="%.4f",
                        help="Similarity score between user profile and item"
                    )
                }
            )
            
            # (Removed sample output block to avoid duplicate/tabley preview)
            
            # Verify we have 10 different items
            st.subheader("Recommendation Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Recommendations", len(recommendations_display))
            with col2:
                st.metric("Unique Items", recommendations_display['item_id'].nunique())
            with col3:
                st.metric("Avg Score", f"{recommendations_display['score'].mean():.4f}")
            
            # Also show the data in a more structured way
            st.subheader("Detailed Item Information:")
            for idx, row in recommendations_display.iterrows():
                with st.expander(f"{idx + 1}. {row['title']}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Item ID:** {row['item_id']}")
                        st.write(f"**Primary Topic:** {row['primary_topic']}")
                        st.write(f"**Subtopic:** {row['subtopic']}")
                    with col2:
                        st.write(f"**Difficulty:** {row['difficulty']}")
                        st.write(f"**Content Type:** {row['content_type']}")
                        st.write(f"**Relevance Score:** {row['score']:.4f}")
                    
                    # Get full description from original dataset
                    item_details = df[df['item_id'] == row['item_id']]
                    if not item_details.empty:
                        item_details = item_details.iloc[0]
                        if 'description' in item_details:
                            st.write(f"**Description:** {str(item_details['description'])[:200]}...")
            
            # Download recommendations
            csv = recommendations_display.to_csv(index=False)
            st.download_button(
                label="📥 Download Recommendations (CSV)",
                data=csv,
                file_name="my_recommendations.csv",
                mime="text/csv",
            )
            
            # Show system explanation
            with st.expander("How these recommendations were generated"):
                st.markdown("""
                **Recommendation Method: Content-Based Filtering**
                
                1. **Your Profile Creation**: We created a virtual profile based on your preferences
                2. **Feature Extraction**: Converted your preferences to the same features as our content
                3. **Similarity Calculation**: Used cosine similarity to find items matching your profile
                4. **Top-N Selection**: Selected the 10 most similar unique items
                
                **Features Used:**
                - Text similarity (title, description, summary)
                - Topic matching (primary_topic, subtopic)
                - Difficulty level alignment
                - Content type preference
                - Financial knowledge level
                
                **Note**: The system ensures all recommended items are unique.
                """)
    
    elif page == "System Info":
        st.header("System Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Information")
            st.write(f"**Total Users:** {df['user_id'].nunique()}")
            st.write(f"**Total Items:** {df['item_id'].nunique()}")
            st.write(f"**Features per Item:** {item_features.shape[1]}")
            st.write(f"**Unique Topics:** {df['primary_topic'].nunique()}")
            
            st.subheader("Feature Engineering")
            st.markdown("""
            - **Text Features**: TF-IDF on title, description, summary (500 dimensions)
            - **Categorical Features**: One-hot encoding of topic, difficulty, content type
            - **Numerical Features**: Financial knowledge level (1-3 scale)
            """)
        
        with col2:
            st.subheader("Model Information")
            st.markdown("""
            **Content-Based Filtering Approach:**
            1. User profile creation from preferences
            2. Cosine similarity calculation
            3. Top-N item recommendation with uniqueness guarantee
            
            **For Existing Users:**
            - Recommendations are precomputed
            - Based on their rating history
            
            **For New Users (Cold Start):**
            - Profile created from explicit preferences
            - Uses same feature space as items
            - Ensures 10 different items are recommended
            """)
            
            st.subheader("Performance")
            if top10_df is not None:
                st.write(f"**Precomputed Recommendations:** {len(top10_df)} user-item pairs")
                # Show sample of precomputed recommendations
                with st.expander("View Precomputed Top-10 Sample"):
                    st.dataframe(top10_df.head(10))
            st.write(f"**Feature Matrix Size:** {item_features.shape}")
        
        # Show sample data
        with st.expander("View Sample Data"):
            st.dataframe(df.head(10))
        
        # Technical details
        with st.expander("Technical Implementation Details"):
            st.code("""
            Key Components:
            1. TF-IDF Vectorizer: Converts text to numerical features
            2. One-Hot Encoder: Handles categorical variables
            3. Feature Combination: All features combined into sparse matrix
            4. Cosine Similarity: Measures similarity between user and items
            5. Deduplication: Ensures unique items in recommendations
            6. Session State: Maintains user state across interactions
            """)

# ============================================================================
# RUN THE APPLICATION
# ============================================================================
if __name__ == "__main__":
    # Initialize session state
    if 'user_preferences' not in st.session_state:
        st.session_state['user_preferences'] = {}
    if 'recommendations' not in st.session_state:
        st.session_state['recommendations'] = None
    if 'temp_user_id' not in st.session_state:
        st.session_state['temp_user_id'] = None
    
    # Run the app
    main()
