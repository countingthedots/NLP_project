import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, classification_report


df = pd.read_csv(r'C:\Users\Nastasja\Desktop\PJN\PROJEKT\IMDb_All_Genres_etf_clean1.csv')


#""""""DATA CLEANING""""""
y = df['Rating']

#actors 

actors = df['Actors'].str.split(',').explode().unique()

actors = ['actor_' + actor for actor in actors]
actors_values = []
for actor in actors:
    actors_values.append(df['Actors'].str.contains(actor).astype(int))
actors_df = pd.DataFrame(dict(zip(actors, actors_values)))
df = pd.concat([df, actors_df], axis=1)


#directors

df['Director'].apply(
    lambda director: director.replace('Directors:', '')
)
df['Director'] = df['Director'].str.split(',').apply(lambda x: [i.strip() for i in x])

# Calculate the average rating of each director
director_ratings = df.explode('Director').groupby('Director')['Rating'].mean()

# For each movie, calculate the average rating of its directors
df['Director'] = df['Director'].apply(lambda directors: np.mean([director_ratings[director.strip()] for director in directors]))


#genres
genres = df['main_genre'].unique()
genres_df = pd.DataFrame()
for genre in genres:
    genres_df['genre_' + genre] = df['main_genre'].str.contains(genre).astype(int)
df = pd.concat([df, genres_df], axis=1)

df = df.drop(columns=['Movie_Title', 'Censor', 'Total_Gross', 'side_genre', 'main_genre', 'Actors'], inplace=False)
X = df.drop(columns=['Rating'], inplace=False)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

alpha = 0.1
ridge = Ridge(alpha=alpha)

ridge.fit(X_train_scaled, y_train)

y_pred = ridge.predict(X_test_scaled)

accuracy = ridge.score(X_test_scaled, y_test)
print(f'Accuracy: {accuracy}')