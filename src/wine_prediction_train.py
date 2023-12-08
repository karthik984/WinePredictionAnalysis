from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.mllib.evaluation import MulticlassMetrics

def clean_dataframe(dataframe):
    return dataframe.select(*(col(col_name).cast("double").alias(col_name.strip("\"")) for col_name in dataframe.columns))

if __name__ == "__main__":
    spark = SparkSession.builder.appName('wine_prediction').getOrCreate()
    spark.sparkContext.setLogLevel('ERROR')

    # File paths: This should be run from AWS S3 
    input_path_train = "s3://sk3374-winepred/TrainingDataset.csv"
    input_path_valid = "s3://sk3374-winepred/ValidationDataset.csv"
    output_path= "s3://sk3374-winepred/trained.model"

    # Reading and cleaning data
    df_train = spark.read.format("csv").option('header', 'true').option("sep", ";").option("inferschema",'true').load(input_path_train)
    train_data = clean_dataframe(df_train)

    df_valid = spark.read.format("csv").option('header', 'true').option("sep", ";").option("inferschema",'true').load(input_path_valid)
    valid_data = clean_dataframe(df_valid)

    # Define features
    all_features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']

    # Assembling features and indexing label
    assembler = VectorAssembler(inputCols=all_features, outputCol='features')
    indexer = StringIndexer(inputCol="quality", outputCol="label")

    train_data.cache()
    valid_data.cache()

    # Random Forest Classifier setup
    rf = RandomForestClassifier(labelCol='label', featuresCol='features', numTrees=150, maxBins=8, maxDepth=15, seed=150, impurity='gini')
    pipeline = Pipeline(stages=[assembler, indexer, rf])
    model = pipeline.fit(train_data)

    # Evaluate initial model
    predictions = model.transform(valid_data)
    evaluator = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction', metricName='accuracy')
    accuracy = evaluator.evaluate(predictions)
    print('Initial Model Test Accuracy:', accuracy)

    metrics = MulticlassMetrics(predictions.select(['prediction', 'label']).rdd.map(tuple))
    print('Initial Model Weighted f1 score:', metrics.weightedFMeasure())

    # Define parameter grid for cross-validation
    param_grid = ParamGridBuilder() \
        .addGrid(rf.maxBins, [8, 4]) \
        .addGrid(rf.maxDepth, [25, 6]) \
        .addGrid(rf.numTrees, [500, 50]) \
        .addGrid(rf.minInstancesPerNode, [6]) \
        .addGrid(rf.seed, [100, 200]) \
        .addGrid(rf.impurity, ["entropy", "gini"]) \
        .build()

    # Cross-validation setup
    crossval = CrossValidator(estimator=pipeline, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=2)
    cv_model = crossval.fit(train_data)

    # Retrieve best model from cross-validation
    best_model = cv_model.bestModel
    print('Best Model:', best_model)

    # Evaluate best model
    predictions_best = best_model.transform(valid_data)
    accuracy_best = evaluator.evaluate(predictions_best)
    print('Best Model Test Accuracy:', accuracy_best)

    metrics_best = MulticlassMetrics(predictions_best.select(['prediction', 'label']).rdd.map(tuple))
    print('Best Model Weighted f1 score:', metrics_best.weightedFMeasure())

    # Save the best model
    best_model.write().overwrite().save(output_path)
