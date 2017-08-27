# -*- coding: utf-8 -*-
from __future__ import print_function
from pyspark.sql import SparkSession
from pyspark.ml.feature import Word2Vec
import sys


if __name__ == "__main__":
    
    model_name = 'model1' #sys.argv[1] 
    spark = None
    
    try:
        spark = SparkSession\
        .builder\
        .appName("Word2Vec_arabic_data")\
        .getOrCreate()
        sc = spark._sc
    
        ##loading model
        model = Word2Vec.load(model_name)
        
        ##############testing similarity
        synonyms = model.findSynonyms('مواليد', 5)
        
        for word, cosine_distance in synonyms.collect():
            print("{}: {}".format(word, cosine_distance))
        ##################################################
        
    finally:
        spark.stop()
        print('spark session has been stopped')