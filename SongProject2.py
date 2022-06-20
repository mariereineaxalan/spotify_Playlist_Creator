import pyspark.ml.clustering as clus
from pyspark.ml import Pipeline

import pyspark.sql.types as types
import pyspark.ml.feature as ft
from pyspark.sql import functions as fn
from pyspark.sql import Window
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import StringIndexer
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler


from pyspark.sql import SparkSession

spark= SparkSession.builder.appName("PySpark app S3").master("spark://spark-master:7077").getOrCreate()
sc = spark.sparkContext

##Main Genres we built our model off of.
genres = spark.read.options(inferSchema = True, multiLine = True, escape = '\"').csv('s3://msbx5420-spr22/dockdock_goose/genres_v2.csv', header=True)

## Reading in second dataset and tring to cluster with more data
data=spark.read.options(inferSchema = True, multiLine = True, escape = '\"').csv('s3://msbx5420-spr22/dockdock_goose/data.csv', header=True)

data.printSchema()

genres.printSchema()

genres2.groupBy('genre').agg(fn.count('*')).collect()

##These columns have the most NAs 
genres=genres.drop("Unnamed: 0","title",'song_name')

genres=genres.na.drop()

data=data.na.drop()

genres.show(vertical=True,truncate=0)

##Didn't think these were relevant to the analysis
genres=genres.drop('id','analysis_url','track_href','duration_ms')

genres.columns

genres.summary().show()

##Starting with Feature creator for our pipeline
featurecreator=ft.VectorAssembler(inputCols=['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'],outputCol='features')

##Started different transformations to maximize sillhouette score to find K
from pyspark.ml.feature import StandardScaler
scale=StandardScaler(inputCol='features',outputCol='standardized')

from pyspark.ml.feature import MinMaxScaler
scaler = MinMaxScaler(min=0, max=1, inputCol='features', outputCol='features_minmax')

from pyspark.ml.feature import PCA
pca = PCA(k=2, inputCol='features', outputCol='features_pca')

genres_train,genres_test =genres.randomSplit([.7,.3], seed=123)

data_train,data_test =data.randomSplit([.7,.3], seed=123)

genres_train.show(2)

##Here we are testing different K to get the most seperated clusters.
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
silhouette_score=[]
evaluator = ClusteringEvaluator(predictionCol='prediction', featuresCol='features_pca', \
                                metricName='silhouette', distanceMeasure='squaredEuclidean')
for i in range(2,50):
    
    kmeans=clus.KMeans(k=i,featuresCol='features_pca')
    
    pipeline=Pipeline(stages=[featurecreator,pca,kmeans])
    
    model=pipeline.fit(data_train)
    output=model.transform(data_train)
    
    
    
    score=evaluator.evaluate(output)
    
    silhouette_score.append(score)
    
    print('K=', i, "Silhouette Score:",score)

## BUilding the kmeans model
kmeans=clus.KMeans(k=9,featuresCol='features_pca')
pipeline=Pipeline(stages=[featurecreator,pca,kmeans])

model=pipeline.fit(data_train)
train=model.transform(data_train)

train.groupBy('prediction').agg(fn.count('genre'),fn.avg('danceability'), fn.avg('energy'),fn.avg('key'),fn.avg('loudness'),fn.avg('mode'),fn.avg('speechiness'),fn.avg('acousticness'),fn.avg('instrumentalness'),fn.avg('liveness'),fn.avg('valence'),fn.avg('tempo')).collect()

##Confirming there are a mix of genres in each cluster
train.groupBy('prediction','genre').agg(fn.count('genre')).sort('prediction').show()

train.show()

test=model.transform(data_test)
test.groupBy('prediction').agg(fn.count('*')).collect()

testP=test.groupBy('prediction').agg(fn.count('genre'),fn.avg('danceability'), fn.avg('energy'),fn.avg('key'),fn.avg('loudness'),fn.avg('mode'),fn.avg('speechiness'),fn.avg('acousticness'),fn.avg('instrumentalness'),fn.avg('liveness'),fn.avg('valence'),fn.avg('tempo')).sort("prediction")

testP=testP.toPandas()

testP.head()

##Using pandas to plot each feature
import pandas as pd
import matplotlib.pyplot as plt


testP.plot.bar(x="prediction", y="avg(danceability)", color = '#1DB954')
plt.show()

%matplot plt

import pandas as pd
import matplotlib.pyplot as plt


testP.plot.bar(x="prediction", y="avg(energy)",color = '#1DB954')
plt.show()
%matplot plt


import pandas as pd
import matplotlib.pyplot as plt


testP.plot.bar(x="prediction", y="avg(key)",color = '#1DB954')
plt.show()
%matplot plt

import pandas as pd
import matplotlib.pyplot as plt


testP.plot.bar(x="prediction", y="avg(loudness)",color = '#1DB954')
plt.show()
%matplot plt

import pandas as pd
import matplotlib.pyplot as plt


testP.plot.bar(x="prediction", y="avg(mode)",color = '#1DB954')
plt.show()
%matplot plt

import pandas as pd
import matplotlib.pyplot as plt


testP.plot.bar(x="prediction", y="avg(speechiness)",color = '#1DB954')
plt.show()
%matplot plt

import pandas as pd
import matplotlib.pyplot as plt


testP.plot.bar(x="prediction", y="avg(acousticness)",color = '#1DB954')
plt.show()
%matplot plt

testP.plot.bar(x="prediction", y="avg(instrumentalness)",color = '#1DB954')
plt.show()
%matplot plt

testP.plot.bar(x="prediction", y="avg(liveness)",color = '#1DB954')
plt.show()
%matplot plt

testP.plot.bar(x="prediction", y="avg(valence)",color = '#1DB954')
plt.show()
%matplot plt

testP.plot.bar(x="prediction", y="avg(tempo)",color = '#1DB954')
plt.show()
%matplot plt

cluster0=test.filter("prediction=0")

cluster0.filter("genre = 'RnB'").show(vertical=True, truncate=0)

cluster0.show(vertical=True, truncate=0)

cluster0.describe().show(vertical=True)

cluster0.groupBy('genre').agg(fn.count('genre')).show()

cluster0.agg(fn.count('genre'),fn.avg('danceability'), fn.avg('energy'),fn.avg('key'),fn.avg('loudness'),fn.avg('mode'),fn.avg('speechiness'),fn.avg('acousticness'),fn.avg('instrumentalness'),fn.avg('liveness'),fn.avg('valence'),fn.avg('tempo')).show(vertical=True)

cluster1=test.filter("prediction=1")

cluster1.filter("genre = 'Pop'").show(vertical=True, truncate=0)

cluster1.groupBy('genre').agg(fn.count('genre')).show()

cluster1.agg(fn.count('genre'),fn.avg('danceability'), fn.avg('energy'),fn.avg('key'),fn.avg('loudness'),fn.avg('mode'),fn.avg('speechiness'),fn.avg('acousticness'),fn.avg('instrumentalness'),fn.avg('liveness'),fn.avg('valence'),fn.avg('tempo')).show(vertical=True)

cluster2=test.filter("prediction=2")

cluster2.filter("genre = 'RnB'").show(vertical=True, truncate=0)

cluster2.groupBy('genre').agg(fn.count('genre')).show()

cluster2.agg(fn.count('genre'),fn.avg('danceability'), fn.avg('energy'),fn.avg('key'),fn.avg('loudness'),fn.avg('mode'),fn.avg('speechiness'),fn.avg('acousticness'),fn.avg('instrumentalness'),fn.avg('liveness'),fn.avg('valence'),fn.avg('tempo')).show(vertical=True)

cluster3=test.filter("prediction=3")

cluster3.filter("genre = 'Pop'").show(vertical=True, truncate=0)

cluster3.groupBy('genre').agg(fn.count('genre')).show()

cluster3.agg(fn.count('genre'),fn.avg('danceability'), fn.avg('energy'),fn.avg('key'),fn.avg('loudness'),fn.avg('mode'),fn.avg('speechiness'),fn.avg('acousticness'),fn.avg('instrumentalness'),fn.avg('liveness'),fn.avg('valence'),fn.avg('tempo')).show(vertical=True)

cluster4=test.filter("prediction=4")

cluster4.filter("genre = 'RnB'").show(vertical=True, truncate=0)

cluster4.groupBy('genre').agg(fn.count('genre')).show()

cluster4.agg(fn.count('genre'),fn.avg('danceability'), fn.avg('energy'),fn.avg('key'),fn.avg('loudness'),fn.avg('mode'),fn.avg('speechiness'),fn.avg('acousticness'),fn.avg('instrumentalness'),fn.avg('liveness'),fn.avg('valence'),fn.avg('tempo')).show(vertical=True)

cluster5=test.filter("prediction=5")

cluster5.filter("genre = 'Pop'").show(vertical=True, truncate=0)

cluster5.groupBy('genre').agg(fn.count('genre')).show()

cluster5.agg(fn.count('genre'),fn.avg('danceability'), fn.avg('energy'),fn.avg('key'),fn.avg('loudness'),fn.avg('mode'),fn.avg('speechiness'),fn.avg('acousticness'),fn.avg('instrumentalness'),fn.avg('liveness'),fn.avg('valence'),fn.avg('tempo')).show(vertical=True)

cluster6=test.filter("prediction=6")

cluster6.filter("genre = 'Rap'").show(vertical=True, truncate=0)

cluster6.groupBy('genre').agg(fn.count('genre')).show()

cluster6.agg(fn.count('genre'),fn.avg('danceability'), fn.avg('energy'),fn.avg('key'),fn.avg('loudness'),fn.avg('mode'),fn.avg('speechiness'),fn.avg('acousticness'),fn.avg('instrumentalness'),fn.avg('liveness'),fn.avg('valence'),fn.avg('tempo')).show(vertical=True)

cluster7=test.filter("prediction=7")

cluster7.filter("genre = 'Emo'").show(vertical=True, truncate=0)

cluster7.groupBy('genre').agg(fn.count('genre')).show()

cluster7.agg(fn.count('genre'),fn.avg('danceability'), fn.avg('energy'),fn.avg('key'),fn.avg('loudness'),fn.avg('mode'),fn.avg('speechiness'),fn.avg('acousticness'),fn.avg('instrumentalness'),fn.avg('liveness'),fn.avg('valence'),fn.avg('tempo')).show(vertical=True)

cluster8=test.filter("prediction=8")

cluster8.filter("genre = 'Dark Trap'").show(vertical=True, truncate=0)

cluster8.show()

cluster8.groupBy('genre').agg(fn.count('genre')).show()

cluster8.agg(fn.count('genre'),fn.avg('danceability'), fn.avg('energy'),fn.avg('key'),fn.avg('loudness'),fn.avg('mode'),fn.avg('speechiness'),fn.avg('acousticness'),fn.avg('instrumentalness'),fn.avg('liveness'),fn.avg('valence'),fn.avg('tempo')).show(vertical=True)
