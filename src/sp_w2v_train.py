# -*- coding: utf-8 -*-
from __future__ import print_function
from pyspark.sql import SparkSession
from pyspark.ml.feature import Word2Vec
import sys
import utils

#####################################################################
#reads all files from input directory
#apply pre-processing: remove non arabic text
#return dataframe object of result text 
def read_corpus(dir_path):
    #inp = sc.textFile(dir_path).map(lambda row: (row.split(" "),))
    corpus = sc.textFile(dir_path).map(utils.extract_arabic)
    corpus = corpus.filter(lambda x: len(x[0])>1)
    corpus = corpus.toDF(['text'])
    return corpus
#####################################################################

#####################################################################
def train_w2v(inp_df, vectorSize=50, minCount=0, inputCol="text", 
                        outputCol="result", windowSize=5, maxIter=50):
    #word2vec = Word2Vec(vectorSize=50, minCount=0, numPartitions=1, stepSize=0.025, 
    #maxIter=20, seed=None, inputCol="text", outputCol=None, windowSize=5,
    #maxSentenceLength=1000)
    word2vec = Word2Vec(vectorSize=vectorSize, minCount=minCount, inputCol=inputCol, 
                        outputCol=outputCol, windowSize=windowSize, maxIter=maxIter)
    model = word2vec.fit(inp_df)
    return model
#####################################################################


if __name__ == "__main__":
    
    dir_path = sys.argv[1]+'/*/*' #"../resources/test_dir/*/*"
    model_name = sys.argv[2] #'saved_model'
    vectorSize = int(sys.argv[3]) #50
    maxIter = int(sys.argv[4]) #50
    windowSize = int(sys.argv[5]) #5
    minCount = int(sys.argv[6]) #0
    
    spark = None
    
    try:
        spark = SparkSession\
        .builder\
        .appName("Word2Vec_arabic_data")\
        .getOrCreate()
        sc = spark._sc
        
        df = read_corpus(dir_path)

        model = train_w2v(df, vectorSize=vectorSize, minCount=minCount,
                        maxIter=maxIter, windowSize=windowSize)
        
        #saving model
        model.save(model_name)
        
        ##loading model
        #model = Word2Vec.load(model_name)
        
        ##############testing similarity
        synonyms = model.findSynonyms('مواليد', 5)
        
        for word, cosine_distance in synonyms.collect():
            print("{}: {}".format(word, cosine_distance))
        ##################################################
    
    finally:
        spark.stop()
        print('spark session has been stopped')