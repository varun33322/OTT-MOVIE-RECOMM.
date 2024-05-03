#importing the libraries to be used 
import sys
import json
import numpy as np 
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from rapidfuzz import process

movie=pd.read_csv('movies.csv')
rating=pd.read_csv('ratings.csv', nrows=6753444)

#extracting year from the title column and creating a separate year column
movie['year'] = movie.title.str.extract(r'(\(\d\d\d\d\))',expand=False)
#Removing the parentheses
movie['year'] = movie.year.str.extract(r'(\d\d\d\d)',expand=False)

#Removing the years from the 'title' column
movie['title'] = movie.title.str.replace(r'(\(\d\d\d\d\))', '')
#Applying the strip function to get rid of whitespace characters
movie['title'] = movie['title'].str.strip()

rating.groupby('movieId').rating.mean()
df=movie.join(rating,lsuffix='N', rsuffix='K')

#remove unnecessary columns
df=df.drop(['movieIdK','genres','year','timestamp'],axis=1)

#Removing NAN values 
movie_users=df.pivot(index='movieIdN', columns='userId',values='rating').fillna(0)

matrix_movies_users=csr_matrix(movie_users.values)

#making ai model
knn= NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20 , n_jobs=-1)
knn.fit(matrix_movies_users)

#recommendation function to find the output even if there is a mis-spell or letter case issue
def recommender(movie_name, data,model, n_recommendations ):
    model.fit(data)
    print("=")
    idx=process.extractOne(movie_name, df['title'])[2]
    distances, indices=model.kneighbors(data[idx], n_neighbors=n_recommendations) 

    for i in indices:
        print((df['title'][i].where(i!=idx))+"=")

name=sys.argv[1]

recommender(name, matrix_movies_users, knn,5)
