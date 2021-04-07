package org.example;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.feature.VectorIndexer;
import org.apache.spark.ml.feature.VectorIndexerModel;
import org.apache.spark.ml.regression.DecisionTreeRegressionModel;
import org.apache.spark.ml.regression.DecisionTreeRegressor;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

//Requires data in libsvm format
public class testLIBSVM {

    public static void main(String args[]){
        SparkSession spark = SparkSession
                .builder()
                .config("spark.master", "local[*]")
                .appName("test")
                .getOrCreate();

// Load the data stored in LIBSVM format as a DataFrame.
        Dataset<Row> data = spark.read().format("libsvm")
                .load("/home/elidor/Documents/libsvm.data");
        data.show(50);

// Automatically identify categorical features, and index them.
// Set maxCategories so features with > 4 distinct values are treated as continuous.
        VectorIndexerModel featureIndexer = new VectorIndexer()
                .setInputCol("features")
                .setOutputCol("indexedFeatures")
                .setMaxCategories(7)
                .fit(data);



// Split the data into training and test sets (30% held out for testing).
        Dataset<Row>[] splits = data.randomSplit(new double[]{0.7, 0.3});
        Dataset<Row> trainingData = splits[0];
        Dataset<Row> testData = splits[1];

// Train a DecisionTree model.
        DecisionTreeRegressor dt = new DecisionTreeRegressor()
                .setFeaturesCol("features");

// Chain indexer and tree in a Pipeline.
        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[]{featureIndexer, dt});

// Train model. This also runs the indexer.
        PipelineModel model = pipeline.fit(trainingData);

// Make predictions.
        Dataset<Row> predictions = model.transform(testData);

// Select example rows to display.
        predictions.select("label", "features").show(5);

// Select (prediction, true label) and compute test error.
        RegressionEvaluator evaluator = new RegressionEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("r2");
        RegressionEvaluator rmse = new RegressionEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("rmse");
        RegressionEvaluator mse = new RegressionEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("mse");
        RegressionEvaluator mae = new RegressionEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("mae");
        double r2 = evaluator.evaluate(predictions);
        double rmsed = rmse.evaluate(predictions);
        double msed = mae.evaluate(predictions);
        double maed = mae.evaluate(predictions);


        System.out.println("R2 Score on test data = " + r2);
        System.out.println("Root Mean Squared Error (RMSE) on test data = " + rmsed);
        System.out.println(" Mean Squared Error (MSE) on test data = " + msed);
        System.out.println("Mean absolute error: "+ maed);


        DecisionTreeRegressionModel treeModel =
                (DecisionTreeRegressionModel) (model.stages()[1]);
        System.out.println("Learned regression tree model:\n" + treeModel.toDebugString());


    }
}
