package org.example;


import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.*;
import org.apache.spark.ml.regression.RandomForestRegressor;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;


public class PipelineTest {

    public static void main(String args[]){
        SparkSession spark = SparkSession
                .builder()
                .config("spark.master", "local[*]")
                .appName("test")
                .getOrCreate();

        Dataset<Row> data = spark.read().format("csv").option("header", "true").option("inferSchema", true)
                .load("SparkML_Practice/insurance.csv");
        data.show(5);
        data.columns();
        //data = data.withColumn("charges", data.col("charges").cast("double"));
        //data = data.withColumn("age", data.col("age").cast("double"));

// Automatically identify categorical features, and index them.

        String [] inputCols = {"sex", "smoker", "region"};
        String [] outputCols = {"sexIndexed", "smokerIndexed", "regionIndexed"};
        StringIndexer indexer = (StringIndexer) new StringIndexer()
                .setInputCols(inputCols)
                .setOutputCols(outputCols);
        //data = indexer.fit(data).transform(data);

        //A standard scaler instance to scale the data for improved performance
        StandardScaler scaler = new StandardScaler()
                .setInputCol("features")
                .setOutputCol("scaledFeatures");

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"age",   "sexIndexed",  "bmi","children","smokerIndexed",   "regionIndexed"})
                .setOutputCol("features");
        VectorAssembler assembler2 = new VectorAssembler()
                .setInputCols(new String[]{"charges"})
                .setOutputCol("chargesS");

// Split the data into training and test sets (30% held out for testing).
        Dataset<Row>[] splits = data.randomSplit(new double[]{0.7, 0.3});
        Dataset<Row> trainingData = splits[0];
        Dataset<Row> testData = splits[1];

// Train a DecisionTree model.
        RandomForestRegressor dt = new RandomForestRegressor()
                .setFeaturesCol("features").setLabelCol("charges").setNumTrees(180).setMaxDepth(30).setMaxBins(300);

        //TODO:Implement a scaler for the output column and transform the scaled values back to primitive double as required
        // Chain indexer and tree in a Pipeline.
        org.apache.spark.ml.Pipeline pipeline = new org.apache.spark.ml.Pipeline()
                .setStages(new PipelineStage[]{indexer,assembler, dt});

// Train model. This also runs the indexer.
        PipelineModel model = pipeline.fit(trainingData);

// Make predictions.
        Dataset<Row> predictions = model.transform(testData);

// Select example rows to display.
        predictions.select("charges", "features").show(5);

// Test how good the prediction metrics are
        RegressionEvaluator evaluator = new RegressionEvaluator()
                .setLabelCol("charges")
                .setPredictionCol("prediction")
                .setMetricName("r2");
        RegressionEvaluator rmse = new RegressionEvaluator()
                .setLabelCol("charges")
                .setPredictionCol("prediction")
                .setMetricName("rmse");

        double r2 = evaluator.evaluate(predictions);
        double rmsed = rmse.evaluate(predictions);




        System.out.println("R2 Score on test data = " + r2);
        System.out.println("Root Mean Squared Error (RMSE) on test data = " + rmsed);


    }
}
