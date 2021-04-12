package MovieLens.Notebooks;

import org.apache.spark.sql.AnalysisException;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import static org.apache.spark.sql.functions.col;

public class MovieLensSolution {

    public static void filterMoviesByYear(int year,  Dataset<Row> df){
        df.printSchema();

        df.select("title").show();

        df.select(col("title"), col("movie_id").plus(1)).show();

// Select people older than 21
        df.filter(col("movie_id").gt(21)).show();

        df.groupBy("movie_id").count().show();

    }
    //Accepts movies dataframe
    public static void filterMoviesByID(SparkSession spark, int year, Dataset<Row> df) throws AnalysisException {
        df.createGlobalTempView("movies");
        spark.sql("SELECT * FROM global_temp.movies").show();


        spark.newSession().sql("SELECT * FROM global_temp.movies Where movie_id >"+year).show();

    }

    static void occupationStats(SparkSession spark,int occCode,  Dataset<Row> df) throws AnalysisException {

        df.createGlobalTempView("stats");
        spark.sql("SELECT * FROM global_temp.stats").show();


        spark.newSession().sql("SELECT  Count(occupation)  AS occ FROM global_temp.stats Where occupation ="+occCode+" AND gender Like 'F'").show();


    }


    public static void main(String args[]) throws AnalysisException {
        SparkSession spark = SparkSession
                .builder()
                .config("spark.master", "local[*]")
                .appName("test")
                .getOrCreate();

        Dataset<Row> users = spark.read().format("csv").option("header", "true").option("inferSchema", true)
                .load("/home/elidor/Documents/Notebooks/prepareSparkData/users.csv");

        Dataset<Row> movies = spark.read().format("csv").option("header", "true").option("inferSchema", true)
                .load("/home/elidor/Documents/Notebooks/prepareSparkData/movies.csv");
        occupationStats(spark,6, users);
        filterMoviesByID(spark,6, movies);



    }
}
