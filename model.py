import pandas as pd

from sys import argv
import pickle
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import random
from sklearn.neighbors import NearestNeighbors


class MovieRecommender:
    def __init__(
            self,
            data,
            model_type='knn',
            n_neighbors=5
            ):
        self.data = data
        self.model_type = model_type
        self.n_neighbors = n_neighbors
        self.model = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree')
        self.model.fit(data)


    def return_similar_movies(self, movie_name, n_movies=5):
        idx = self.data.index.get_loc(movie_name)
        # get the 5 most similar movies
        similar_movies = self.model.kneighbors(self.data.iloc[idx, :].values.reshape(1, -1), return_distance=False)
        # get the names of the movies
        similar_movies = [self.data.index[i] for i in similar_movies[0]]
        return similar_movies


    def test_similarity(self, data):
        for i in range(5):
            movie = random.choice(data.index)
            print('Movie selected:', movie)
            print('Similar movies:')
            print(self.return_similar_movies(movie))
            print()

def main():

    # Check command-line arguments
    if len(argv) not in [2, 3]:
        print("Usage: python model.py <data.csv> [model]")
        exit(1)

    if len(argv) == 2:
        _, DATA = argv

    else:
        _, DATA, MODEL = argv

    # Load data from csv file
    data = pd.read_csv(DATA)

    # clean data
    fn = lambda x: x.translate(str.maketrans('', '', string.punctuation)).lower()

    data['description'] = data['description'].apply(fn)
    data['review_text'] = data['review_text'].apply(fn)
    data['genre'] = data['genre'].apply(fn)
    data['keywords'] = data['keywords'].apply(fn)
    data['director'] = data['director'].apply(fn)
    data['actor'] = data['actor'].apply(fn)
    data['creator'] = data['creator'].apply(fn)

    # Vectorize the data
    ct = CountVectorizer(stop_words='english', max_features=1000)
    tfidf = TfidfVectorizer(stop_words='english', max_features=1000)

    # fit and transform on description
    tfidf_des = pd.DataFrame(tfidf.fit_transform(data['description']).toarray())
    tfidf_des.index = data['name']
    tfidf_rev = pd.DataFrame(tfidf.fit_transform(data['review_text']).toarray())
    tfidf_rev.index = data['name']
    ct_genre = pd.DataFrame(ct.fit_transform(data['genre']).toarray())
    ct_genre.index = data['name']
    tfidf_keywords = pd.DataFrame(tfidf.fit_transform(data['keywords']).toarray())
    tfidf_keywords.index = data['name']
    ct_director = pd.DataFrame(ct.fit_transform(data['director']).toarray())
    ct_director.index = data['name']
    ct_actor = pd.DataFrame(ct.fit_transform(data['actor']).toarray())
    ct_actor.index = data['name']
    ct_creator = pd.DataFrame(ct.fit_transform(data['creator']).toarray())
    ct_creator.index = data['name']

    master_frame = pd.concat([tfidf_des,
        tfidf_rev,
        ct_genre,
        tfidf_keywords,
        ct_director,
        ct_actor,
        ct_creator],
        axis=1
        )        
    master_frame.index = data['name']
    
    random.seed(1)
    
    model = MovieRecommender(master_frame)
    model.test_similarity(master_frame)


if __name__ == "__main__":
    main()