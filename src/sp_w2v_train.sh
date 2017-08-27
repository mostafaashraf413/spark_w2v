#!/bin/sh
# pyspark w2v for arabic_web_data

DATASET_DIR='../resources/test_ar_dir'
SAVED_MODEL_PATH='model1'
VECTOR_SIZE=50
MAX_ITER=50
WINDOW_SIZE=5
MIN_COUNT=0
#query parameter is optional, it is just for testing
QUERY='المستشفى'

spark-submit sp_w2v_train.py ${DATASET_DIR} ${SAVED_MODEL_PATH} ${VECTOR_SIZE} ${MAX_ITER} ${WINDOW_SIZE} ${MIN_COUNT} ${QUERY}

