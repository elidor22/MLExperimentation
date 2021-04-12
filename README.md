# MLExperimentation
A quick project that experiments with Spark MLlib and various ML libraries. Also includes benchmarking for different tools used                   
The project is divided in **two main subdirectories**:

**`-PythonBenchmark`** subdirectory includes Python scripts using xgboost as the main ML library. It also includes simple API endpoints implemented
using Flask and Fast API. For load testing purposes [Vegeta](https://github.com/tsenart/vegeta) was used and the testing configuration files 
are included in the subdirectory `PythonBenchmark/VegetaFiles`. To install required libraries use the included [`requirements.txt`](PythonBenchmark/requirements.txt )
file with **`pip`** package manager. Also the pretrained XGBoost model weight are provided in Binary Format and a [converter script](PythonBenchmark/saveXG2JSON.py)
that converts
the values to JSON is provided.



**`-SparkML_Practice`** subdirectory includes a Java project and Maven is used for build automation. This module uses `Apache Spark` to solve 
Machine Learning problems. The approach used in here is a more straightforward that the one in the Python as Spark is the only library needed 
for an end to end solution(with Spark it means all the needed modules included in different Spark libraries available in Maven central and included
in [pom.xml](SparkML_Practice/pom.xml). To run the project first install the dependencies using Maven, then excecute it via an IDE that 
supports Java or manually compile the code using `mvn` commands. 
**`PipelineTest`** class creates a pipeline consisting of a `VectorAssembler` that prepares the raw input to be fed into the following Spark
utilities, a *`StandardScaler`* that standardizes features by removing the mean and scaling to unit variance using column summary statistics on 
the samples in the training set, and a **`StringIndexer`** that **encodes** string inputs into numerical format.

**`GridSearch`** class is similar to the above mentioned class, except that it has an extra feature, which is the implementation of GridSearch
algorithm. This change makes the class more suitable for hyperparameter tunning as the process of finding the best combination of parameters is automated 
and the developer should input only the variables that shall be tested to find the possible combination for the input variables.
***Attention:* grid search algorithms is helpful in many scenarios for both Regression and Classification tasks, but it is resource intensive as
the final number of runs in which a model is trained from scratch will be equal to the multiplication result of all input dimensions(eg 4 possible 
values for MaxDepth times 4 values for MaxBins times 2 values for Learning rate will result in the training process being repeated 32 times
which), so it's strongly recommended to use the class with a smaller dataset if a big one is provided.
