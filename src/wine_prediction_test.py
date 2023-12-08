import os
import sys
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql.functions import col

def clean_dataframe(dataframe):
    return dataframe.select(*(col(col_name).cast("double").alias(col_name.strip("\"")) for col_name in dataframe.columns))

if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName('wine_prediction_spark_app') \
        .getOrCreate()
    spark.sparkContext.setLogLevel('ERROR')

    current_dir = os.getcwd()

    # Handling input arguments for data and model paths
    if len(sys.argv) > 3:
        sys.exit(-1)
    elif len(sys.argv) > 1:
        input_path = sys.argv[1]
        if not ("/" in input_path):
            input_path = os.path.join(current_dir, input_path)
        model_path = os.path.join(current_dir, "trained.model")
        print("Test data file location:")
        print(input_path)
    else:
        print("Current directory:")
        print(current_dir)
        input_path = os.path.join(current_dir, "testdata.csv")
        model_path = os.path.join(current_dir, "trained.model")

    # Reading CSV file into DataFrame
    df = (spark.read
          .format("csv")
          .option('header', 'true')
          .option("sep", ";")
          .option("inferschema", 'true')
          .load(input_path))

    cleaned_df = clean_dataframe(df)

    # Defining features for the model
    selected_features = [
        'fixed acidity',
        'volatile acidity',
        'citric acid',
        'chlorides',
        'total sulfur dioxide',
        'density',
        'sulphates',
        'alcohol',
    ]

    # Loading the pre-trained model
    trained_model = PipelineModel.load(model_path)

    # Making predictions on the data
    predictions = trained_model.transform(cleaned_df)
    print("Sample predictions:")
    print(predictions.show(5))

    # Evaluating the model
    results = predictions.select(['prediction', 'label'])
    evaluator = MulticlassClassificationEvaluator(
        labelCol='label',
        predictionCol='prediction',
        metricName='accuracy'
    )
    accuracy = evaluator.evaluate(predictions)
    print('Test Accuracy of the wine prediction model:', accuracy)

    metrics = MulticlassMetrics(results.rdd.map(tuple))
    print('Weighted F1 score of the wine prediction model:', metrics.weightedFMeasure())
    sys.exit(0)
