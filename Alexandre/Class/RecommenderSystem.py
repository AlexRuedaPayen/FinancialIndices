from typing import Dict, Text

import numpy as np
import tensorflow as tf

import tensorflow_recommenders as tfrs
import tensorflow_datasets as tfds


class RecommenderSystem(tf.keras.Model):
    def __init__(self):
        ratings=tfds.load("movielens/100k-ratings",split="train")
        ratings=ratings.map(lambda x:{
            "movie_title":x['movie_title'],
            "user_id":x['user_id'],
            "user_rating":x["user_rating"]
        })
        tf.random.set_seed(42)
        shuffled=ratings.shuffle(100_000,seed=42,reshuffle_each_iteration=False)

        train=shuffled.take(80_000)
        test=shuffled.skip(80_000).take(20_000)

        movies_titles=ratings.batch(1_000_000).map(lambda x:x["movie_title"])
        user_ids=ratings.batch(1_000_000).map(lambda x:x["user_id"])

        unique_movie_titles=np.unique(np.concatenatelist(movies_titles))
        unique_user_ids=np.unique(np.concatenatelist(user_ids))

        embedding_dimension=32

        super().__init__()
        self.user_embeddings=tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.StringLookup(
                vocabulary=unique_user_ids, mask_token=None
            ),
            tf.keras.layers.Embedding(len(unique_user_ids)+1,embedding_dimension)
        ])
        self.movie.embeddings=tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.StringLookup(
                vocabulary=unique_movie_titles, mask_token=None
            ),
            tf.keras.layers.Embedding(len(unique_movie_titles)+1,embedding_dimension)
        ])