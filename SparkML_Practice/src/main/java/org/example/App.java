package org.example;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

/**
 * Hello world!
 *A simple class to experiment with the given data format
 *
 */
public class App 
{
    public static void main( String[] args )
    {
        SparkSession spark = SparkSession
                .builder()
                .config("spark.master", "local[*]")
                .appName("test")
                .getOrCreate();

// Load the data stored in LIBSVM format as a DataFrame.
        Dataset<Row> data = spark.read().format("csv").option("header", "true")
                .load("/home/elidor/Documents/data.csv");
        data.show(10);


        System.out.println( "Hello World!" );
    }
}
