#!/usr/bin/python

import pandas as pd
import json
from uuid import uuid4
import time, sys, os, shutil, glob, io, requests
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline, Model, PipelineModel
from pyspark.sql import SQLContext
import dsx_core_utils
from dsx_ml.ml import save_evaluation_metrics


# setup dsxr environmental vars from command line input
from dsx_ml.ml import dsxr_setup_environment
dsxr_setup_environment()

# define variables
args = {"dataset": "/datasets/GraduateAdmissionsSparkMLEval.csv", "published": "false", "threshold": {"metric": "areaUnderROC", "min_value": 0.3, "mid_value": 0.83}, "evaluator_type": "binary", "execution_type": "DSX", "remoteHost": "", "remoteHostImage": "", "livyVersion": "livyspark2"}
model_path = os.path.join(os.getenv("DSX_PROJECT_DIR"), "models", os.getenv("DEF_DSX_MODEL_NAME", "GraduateAdmissionsClassificationSparkML"), os.getenv("DEF_DSX_MODEL_VERSION", "2"), "model")

# create spark context
spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

# load the input data

input_data = os.getenv("DEF_DSX_DATASOURCE_INPUT_FILE", os.getenv("DSX_PROJECT_DIR") + args.get("dataset"))
dataframe = SQLContext(sc).read.csv(input_data , header="true", inferSchema = "true")

# load the model from disk 
model_rf = PipelineModel.load(model_path)


startTime = int(time.time())

# generate predictions
predictions = model_rf.transform(dataframe)

threshold = {'metric': 'areaUnderROC', 'min_value': 0.3, 'mid_value': 0.83}

# replace "label" below with the numeric representation of
# the label column that you defined while training the model
labelCol = "label"

# create evaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator(labelCol=labelCol)

# compute evaluations
eval_fields = {
        "accuracyScore": predictions.rdd.filter(lambda x: x[labelCol] == x["prediction"]).count() * 1.0 / predictions.count(),
        "areaUnderPR": evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderPR"}),
        "areaUnderROC": evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"}),
        "thresholdMetric": threshold["metric"],
        "thresholdMinValue": threshold["min_value"],
        "thresholdMidValue": threshold["mid_value"]
    }

# feel free to customize to your own performance logic using the values of "good", "poor", and "fair".
if(eval_fields[eval_fields["thresholdMetric"]] >= threshold.get('mid_value', 0.70)):
    eval_fields["performance"] = "good"
elif(eval_fields[eval_fields["thresholdMetric"]] <= threshold.get('min_value', 0.25)):
    eval_fields["performance"] = "poor"
else:
    eval_fields["performance"] = "fair"

save_evaluation_metrics(eval_fields, "GraduateAdmissionsClassificationSparkML", "2", startTime)