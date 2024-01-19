import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the data and perform preprocessing
df_lyrics = pd.read_csv('data/lyrics-data.csv')
df_lyrics = df_lyrics[df_lyrics['language'] == 'en']
df_lyrics.rename(columns={'SName': 'title',
                 'Lyric': 'lyrics', 'ALink': 'artist_id'}, inplace=True)
df_lyrics.drop(columns=['SLink', 'language'], inplace=True)
df_lyrics.dropna(inplace=True)

df_artists = pd.read_csv('data/artists-data.csv')
df_artists.drop(columns=['Genres', 'Songs', 'Popularity'], inplace=True)
df_artists.rename(columns={'Artist': 'artist',
                  'Link': 'artist_id'}, inplace=True)
df_artists.columns = df_artists.columns.str.lower()
df_artists.dropna(inplace=True)
df_artists = df_artists[df_artists['artist_id'].isin(
    df_lyrics['artist_id'].unique())]

df = pd.merge(df_lyrics, df_artists, on='artist_id')
df = df[['artist_id', 'artist', 'title', 'lyrics']]
df.dropna(inplace=True)

# TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['lyrics'])

# Recommender function


def recommender(input_text, top_n=10):
    input_tfidf_vector = tfidf_vectorizer.transform([input_text])
    similarities = cosine_similarity(input_tfidf_vector, tfidf_matrix)
    similar_songs_indices = similarities.argsort()[0][-top_n:][::-1]
    recommended_songs = df.iloc[similar_songs_indices][['artist', 'title']]
    recommended_songs['similarity_score'] = similarities[0,
                                                         similar_songs_indices]
    return recommended_songs

# Streamlit app


def main():
    st.title("Music Recommender System")
    st.sidebar.header("Input Lyrics")
    input_text = st.sidebar.text_area("Enter lyrics here:")

    if st.sidebar.button("Get Recommendations"):
        if input_text:
            st.header("Recommended Songs:")
            recommended_songs = recommender(input_text)
            st.table(
                recommended_songs[['artist', 'title', 'similarity_score']])
        else:
            st.warning("Please enter lyrics to get recommendations.")


if __name__ == "__main__":
    main()
