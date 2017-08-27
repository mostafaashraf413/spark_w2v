#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import print_function

#from pyspark import SparkContext
from pyspark.sql import SparkSession
# $example on$
#from pyspark.mllib.feature import Word2Vec
from pyspark.ml.feature import Word2Vec
# $example off$

if __name__ == "__main__":
    
    spark = None
    try:
        #sc = SparkContext(appName="Word2VecExample")  # SparkContext
        spark = SparkSession\
        .builder\
        .appName("Word2VecExample")\
        .getOrCreate()
        sc = spark._sc
        
        # $example on$
        #inp = sc.textFile("../resources/small_test_txt.gz").map(lambda row: (row.split(" "),))
        #inp = sc.wholeTextFiles("../resources/test_dir/*/*").map(lambda x: x[1].split("\n"))
        inp = sc.textFile("../resources/test_dir/*/*").map(lambda row: (row.split(" "),))
        #print(inp.collect())
        inp = inp.toDF(['text'])
        

        #word2vec = Word2Vec(vectorSize=50, minCount=0, numPartitions=1, stepSize=0.025, 
        #maxIter=20, seed=None, inputCol="text", outputCol=None, windowSize=5,
        #maxSentenceLength=1000)
        word2vec = Word2Vec(vectorSize=50, minCount=0, inputCol="text", 
                            outputCol="result", maxIter=50)
        
        model = word2vec.fit(inp)
        
        model_name = 'saved_model'
        #saving model
        model.save(model_name)
        
        ##loading model
        #model = Word2Vec.load(model_name)
        
        #testing similarity
        synonyms = model.findSynonyms('graph', 5)
        
        for word, cosine_distance in synonyms.collect():
            print("{}: {}".format(word, cosine_distance))
        # $example off$
    
    finally:
        print('stopping spark session')
        spark.stop()