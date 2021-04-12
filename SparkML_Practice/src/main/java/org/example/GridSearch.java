package org.example;


import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.regression.RandomForestRegressor;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;


public class GridSearch {



    public static void main(String args[]){
            SparkSession spark = SparkSession
                    .builder()
                    .config("spark.master", "local[*]")
                    .appName("GridSearch")
                    .getOrCreate();



            Dataset<Row> data = spark.read().format("csv").option("header", "true").option("inferSchema", true)
                    .load("SparkML_Practice/insurance2.csv");
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
            Dataset<Row>[] splits = data.randomSplit(new double[]{0.8, 0.2});
            Dataset<Row> trainingData = splits[0];
            Dataset<Row> testData = splits[1];

            // Train a DecisionTree model.
            RandomForestRegressor dt = new RandomForestRegressor();
                    //.setFeaturesCol("features").setLabelCol("charges");//.setNumTrees(180).setMaxDepth(30).setMaxBins(300);

            //TODO:Implement a scaler for the output column and transform the scaled values back to primitive double as required
            // Chain indexer and tree in a Pipeline.
            org.apache.spark.ml.Pipeline pipeline = new org.apache.spark.ml.Pipeline()
                    .setStages(new PipelineStage[]{indexer,assembler, dt});

            // Train model. This also runs the indexer.
            //PipelineModel model = pipeline.fit(trainingData);


            ParamMap[] paramGrid = new ParamGridBuilder()
                    .addGrid(dt.numTrees(), new int[] {105, 157, 250,45,55,98,99,300})
                    .addGrid(dt.maxDepth(), new int[]{18,25, 30})
                    .addGrid(dt.maxBins(),new int[]{ 75, 144,188, 205} )
                    .build();

            // We now treat the Pipeline as an Estimator, wrapping it in a CrossValidator instance.
            // This will allow us to jointly choose parameters for all Pipeline stages.
            // A CrossValidator requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
            // Note that the evaluator here is a BinaryClassificationEvaluator and its default metric
            // is areaUnderROC.
            CrossValidator cv = new CrossValidator()
                    .setEstimator(pipeline)
                    .setEvaluator(new RegressionEvaluator())
                    .setEstimatorParamMaps(paramGrid)
                    .setNumFolds(10)  // Use 3+ in practice
                    .setParallelism(25);  // Evaluate up to 2 parameter settings in parallel

            // Run cross-validation, and choose the best set of parameters.
            CrossValidatorModel cvModel = cv.fit(trainingData);
            Dataset<Row> predictions = cvModel.transform(testData);

            predictions.select("label", "features").show(5);

            // Generate model metrics
            RegressionEvaluator evaluator = new RegressionEvaluator()
                    .setLabelCol("label")
                    .setPredictionCol("prediction")
                    .setMetricName("r2");
            RegressionEvaluator rmse = new RegressionEvaluator()
                    .setLabelCol("label")
                    .setPredictionCol("label")
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
            double msed = mse.evaluate(predictions);
            double maed = mae.evaluate(predictions);


            //Print the metrics
            System.out.println("R2 Score on test data = " + r2);
            System.out.println("Root Mean Squared Error (RMSE) on test data = " + rmsed);
            System.out.println(" Mean Squared Error (MSE) on test data = " + msed);
            System.out.println("Mean absolute error: "+ maed);



        }
    }


