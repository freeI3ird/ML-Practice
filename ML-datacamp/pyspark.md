## Spark
`spark: SparkSession object`
`sc: SparkContext object`

### Preamble
1. Spark is a platform for cluster computing.
2. If your data is too big, spark can be used.
   - It spreads your data over multiple nodes, both data processing and computation(algorithm run on data) happens parallely over the  multiple nodes.
3. Don't work in every situation, only in specific ones
   1. Your data is too big to be handled on a single Machine
   2. Your task(algorithm) can be parallelized.

### Connecting to Spark cluster using pyspark
1. We will connect to master node.
   - Master node is connected with other worker nodes
   - Master node sends data and algorithm to worker nodes to run, they send back the results.
2. Connection from pyspark
   - Instantiate the class `SparkContext()` with details of cluster.
     - This will make a connection with Spark Cluster.
   - Details of cluster are collected in object of `SparkConf()` and send to `SparkContext()`.

### RDD (Resilient Distributed Dataset)
1. RDD is the core data_structures of spark.
2. Spark Dataframe : abstraction built over RDD
   1. Many operations are optimized for Spark Dataframe.
   2. Behave as SQL tables
   3. To work with spark DataFrame need `SparkSession`
   4. Spark Dataframe are **Immutable**
3. `SparkSession`
   1. `SparkContext` act as connection to cluster
   2. `SparkSession` act as interface for this connection.
      1. Used to send and recieve data over this connection.

### Creating spark session
1. `from pyspark.sql import SparkSession`
2. `spark = SparkSession.builder.getOrCreate()`
   1. This returns an existing SparkSession if there's already one in the environment, or creates a new one if necessary!

### Spark Attributes
1. `catalog`
   - It lists all the data inside the cluster
   - It has several method to do so.
   - Methods
     - `spark.catalog.listTables()`: List all the tables in cluster
2. `read`
   - To read from different data sources into spark dataframe
   - It has several methods to do so
   - Methods
     - `spark.read.csv(file_path, header=True)`: To read csv file.

### Query a spark table
1. We can query a spark DataFrame as sql table
2. Query Table
   1. `query_stg = "SELECT * FROM flights LIMIT 10"`
   2. `result_df = spark.sql(query_stg)`
3. Show Spark DataFrame
   1. `result_df.show()`

### Spark and Pandas DataFrame
1. Spark to Pandas
   1. `pandas_df = spark_df.toPandas()`
2. Pandas to Spark
   1. `spark_df = spark.createDataFrame(pandas_df)`
   2. This dataframe is present only locally and not registered in the catalog, so we can't query on this.

### Registering a table in spark.catalog
1. Register Temporarily
   1. Temporary meaning this table is visible only in this SparkSession.
   2. Methods
      1. `spark_df.createTempView("table_name")`
      2. `spark_df.createOrReplaceTempView("table_name")`
         1. Updates table if already exists

### Spark DataFrame Operations

#### Creating a spark DF from table listed in catalog
1. `spark_df = spark.table("table_name")`

#### Column operations  
`Spark DF is immutable, everytime a new DF is created`
1. Extracting a column
   1. `col_obj = df.col_name`
   2. `col_obj` is an object of class `Column`
2. Adding a new column
   1. `new_df = df.withColumn("new_col_name", col_obj )`
      1. `col_obj` is an object of class `Column`
   2. Adding a column using a previous column
      1. `new_df = df.withColumn("col_name", df.old_col +1) `
   3. Overwriting a df with new df
      1. `df = df.withColumn("col_name", df.old_col + 1)`
   4. Overwriting a existing column
      1. `df = df.withColumn("old_col", df.old_col + 1)`
      2. Keep the column name arguement same as the old column name.
   5. Using two columns to form a third column
      1. `df = df.withCOlumn("plane_age", df.year - df.plane_year)`
      2. `year` and `plane_year` are column of the table.
   6. Creating a boolean column( mostly used to create labels for classification)
      1. `df = df.withColumn("is_late", df.dep_delay > 0)`
      2. Before using it for ML models, we have to convert boolean to numberic data type
         1. `df = df.withColumn("label", df.is_late.cast("integer"))`
3. Renaming a Column
   1. `new_df = df.withColumnRenamed("old_col_name", "new_col_name")`

### Analogous to SQL
1. `df.filter() ~ WHERE clause`
   1. Takes 2 types of arguements
      1. String expression, which forms the part of WHERE clause
         - `new_df = df.filter("air_time > 600")`
      2. `Column` class object, column of booleans  
         - `new_df = df.filter( df.air_time > 600)`
           - df.air_time > 600, it returns a column of booleans
   2. Analogous SQL query
      1. df corresponds to flights table
      2. `SELECT * FROM flights WHERE air_time > 600`
2. `df.select() ~ SELECT `
   1. Selects column from a df.
   2. Take 2 types of arguements
      1. Column names as a string
         - `new_df = df.select("origin", "dest")`
      2. `Column` class objects
         - `new_df = df.select(df.origin, df.dest)`
   3. Analogous SQL query
      1. df correponds to flights table
      2. `SELECT origin, dest FROM flights`
   4. Column wise Transformations can be applied using `.select()`
      1. `new_df = df.select(df.air_time/60)`
      2. Changing the name of column
         1. `new_df = df.select( (df.air_time/60).alias(time_in_hours) )`
         2. `.alias()` is method of `Column` class
   5. This is not allowed in when `.select()` takes string arguements
      1. `new_df = df.select("origin", "air_time/60")`
         1. Transformations when passing string arguements
         2. For this type of functionality `.selectExpr()` is used.
3. `df.selectExpr()`
   1. To select columns and apply transformation on them
   2. Takes column names as string arguements
   3. `new_df = df.selectExpr("origin", "dest", "air_time/60 as time_in_hours")`0
      1. `as` is similiar to `.alias()`
   4. Does the same work as `.select()` do for Column transformation.

4. `df.groupBy() ~ GROUP BY`
   1. Same as `GROUP BY` of SQL
   2. `obj = df.groupBy('col1', 'col2',...)`
      - returns object of class `GroupData`
      - `GroupData` class has many aggregation methods such as
        - `.min(), .max(), .count(), .avg(), .sum()`
        - These methods take arguements as column name of the aggregated data.
        - returns a new DataFrame.
   3. Code
      1. No arguement is passed to `.groupBy()`
         1. Then only 1 group is created
         2. Find the min air_time in flights table
            - `df = flights.groupBy().min("air_time")`
      2. Arguements passed `.groupBy("col1", "col2")`
         1. This creates the different groups on the basis of different combinations of arguements passed ([col1, col2]) exactly similar to SQL GROUP BY
         2. Find the avg air_time of flights with different origin
            - `df = flights.groupBy("origin").avg("air_time")`
   4. `.agg()` method of `GroupData`
      1. Its objective is to apply wide variety of aggregation functions on Grouped Data.
      2. The sub module`pyspark.sql.functions` has many aggregation functions.
         - These functions take arguements as `col_name` in GroupData table.
         - returns a new DataFrame
         - `pyspark.sql.functions.stddev('col_name')`
      3. Group flight data by `dest` and then find the `standard Deviation` in the `dep_delay`(departure delay)
         1. `grp_by_dest = flights.groupBy('dest')``
         2. `grp_by_dest.avg("dep_delay")`
            - Mean of departure delay
         3. `import pyspark.sql.functions as F`
         4. `df= grp_by_dest.agg( F.stddev('dep_delay')) )`
5. `df.join() ~ SQL joins`
   1. Arguements: `df.join( df2, on="col_key", how="joinType")`
      1. First Arguement is the another DataFrame which we want to join.
      2. `on=`: This takes column_key, join is made on the basis of this column.
      3. `how=`: Specify the type of join
         - `how= "leftouter"`
   2. Join flights table with airports table on key "dest"
      1. `flights.join(airports, on="dest", how="leftouter")`

### Datatypes in spark
`Spark requires numeric data for modeling`
1. Spark only handles/works-on NUMERIC data
   1. NUMERIC
      - integer
      - double
2. When we load data in spark DF, spark tries to guess the data type of columns, but its guess sometimes may not work. It can treat a column of numeric as string.
   1. So the columns that we need, we can explicitly `type-cast` them to correct data type
3. Type-casting
   1. use `.cast("data_type")` method of `Column` class
      - Arguement: single string arguement denotes the type in which you want to cast your data
      - "integer" or "double"
      - It is the method of class `Column`
   2. E.g:
      - `df = df.withColumn("air_time", df.air_time.cast("integer"))`

### ML Pipeline in pyspark
`First we will have a DATA-PIPELINE then MODELLING PIPELINE`
#### ML Algos
1. Pyspark has a module `pyspark.ml`, it has several classes for performing ML tasks
2. 2 basic Types of Classes in `pyspark.ml` sub-module.
   1. Estimator
      - Implements `.fit()` method.
      - E.g: RandomForestModel for classification & regression
   2. Transformer
      - Implements `.transform()` method.
      - E.g: PCA, Bucketizer
3.

#### Handling Categorical Features
1. Spark requires numeric data for modeling, but there can be a feature which
   - we want to use in modeling
   - is in form of string
   - e.g flight_destination: 'London', 'Paris'.
2. Approaches to Handle categorical data
   1. OneHotEncoder
      - It  is a way of representing a categorical feature where every observation has a vector in which all elements are zero except for at most one element, which has a value of one.
3. How to encode Categorical feature in pyspark
   1. `pyspark.ml.features` It is a sub-module to handle these features.
   2. Process
      - First we find all unique values in a column, create a mapping from these values to integer values.
      - Create a new column with integer values corresponding to string column, using the mapping we created.
      - Convert these integer values into OneHotEncoder.
   3. Steps Involved
      1. Create `StringIndexer`
         - `StringIndexer` is a class
         - Members of this class
           - `Estimator`
             - Take a DataFrame with a column of strings and create a mapping which maps each unique string to a number.
             -  returns a Transformer, that we defined below and this mapping is available to this Transformer.
           - `Transformer`
             - Takes a DataFrame, attaches the mapping to it as metadata, and returns a new DataFrame with a numeric column corresponding to the string column(numeric values for each string).

      2. Create `OneHotEncoder`
         1. `OneHotEncoder` is a class.
         2. Encode this above numeric column as a one-hot vector.
         3. Members of this class
            - `Estimator`
            - `Transformer`
         4. `?? Role of these Estimator and Transformer is not clear ??`
            1. Guess
               - Estimator calls the `.fit()` method on column, it derives the meta data from data.
                 - For `StringIndexer`, metadata= No. of unique strings, a mapping from strings to integer.
                 - For `OneHotEncoder`, metadata= No. of unique integers, length of the vector.
               - Transformer uses the meta data derived by the Estimator and calls `.transform()` method on column.
                 - For `StringIndexer`, it transformed every string to integer using the mapping.
                 - For `OneHotEncoder`, it transformed every integer to a vector of fixed length.

      3. End Result
         - A column that encodes your categorical feature as a vector that's suitable for machine learning routines.
      4. Code
          - Convert destination feature 'dest' to OneHotEncoder
          - `dest_indexer = StringIndexer(inputCol="dest", outputCol="dest_index")`
          - `dest_encoder = OneHotEncoder(inputCol="dest_index", outputCol="dest_fact")`
            - `inputCol` is the name of the column you want to index or encode.
            - `outputCol` is the name of the new column that the Transformer should create.

4. `?? Functioning of Estimator and transformer is not much clear ??`

#### Final form of Data
1. In spark, we need to combine all features into a vector. So a/c to models perspective:
   1. Each observation is a vector, which contains all information
   2. Each element of this vector = feature.
   3. This is how spark modelling routines expects data.
2. How to do in pyspark
   1. Use Transformer class `VectorAssembler` of sub-module `pyspark.ml.feature`
   2. Code
      1. `vec_assembler = VectorAssembler(inputCols=["month", "air_time", "carrier_fact", "dest_fact", "plane_age"], outputCol="features")`
   3. This is the last step of a Data Pipeline.
   4. Features are not combined right now, only an object is created, which will be passed to Pipeline.

#### DATA-PIPELINE
1. This pipeline has many stages of Estimators and Transformers to process and transform the data, before modelling(model learning) takes place.
2. This lets you reuse the same modeling process over and over again by wrapping it up in one simple object
3. How to do in pyspark
   1. Class `Pipeline` from submodule `pyspark.ml`
      1. It combines all the estimators and transformers that we created.
   2. `flights_pipe = Pipeline(stages=[dest_indexer, dest_encoder, carr_indexer, carr_encoder, vec_assembler])`
4. Words from documentation:
   -  A Pipeline consists of a sequence of stages, each of which is either an :py:class:`Estimator` or :py:class:`Transformer`.
   - When :py:meth:`Pipeline.fit` is called, the stages are executed in order.
   - If a stage is an :py:class:`Estimator`, its
    :py:meth:`Estimator.fit` method will be called on the input
    dataset to fit a model. Then the model, which is a transformer,
    will be used to transform the dataset as the input to the next
    stage.
   - If a stage is a :py:class:`Transformer`, its
    :py:meth:`Transformer.transform` method will be called to produce the dataset for the next stage.
   - The fitted model from a :py:class:`Pipeline` is a :py:class:`PipelineModel`, which consists of fitted models and transformers, corresponding to the pipeline stages.
   - If stages is an empty list, the pipeline acts as an
    identity transformer.
4. Pass Data to Pipeline
   1. `flights_pipe_model = flights_pipe.fit(model_data)`
   2. `model_data = flights_pipe_model.transform(model_data)`

5. Before Modelling, split the data in train and test sets
   1. `training, test = piped_data.randomSplit([0.6, 0.4])`
   2. 60% training , 40% test
   3. In Spark it's important to make sure you split the data after all the transformations.
       - This is because operations like StringIndexer don't always produce the same index even when given the same list of strings.

### Creating ML-Model(MODELLING)
1. First step: Instance of a estimator
   1. Logistic Regression
      1. It is an Estimator
   2. Code
      1. `from pyspark.ml.classification import LogisticRegression`
      2. `lr = LogisticRegression()`
      3.
2. HyperParameter Tunning
   1. Use k-fold cross validation.
      1. Default k=3.
      2. Cross validation error give good idea of error on unseen data.
   2. Define Evaluation Metric
      1. Two compare different models, like to select best model in HyperParameter Tunning, we need some evaluation metric.
      2. Evaluation in pyspark
         1. pyspark module `pyspark.ml.evaluation` has Evaluation classes for different type of metrics.
         2. For Binary classification
            1. `pyspark.ml.evaluation.BinaryClassificationEvaluator`
            2. This evaluator calculates the area under the ROC. This is a metric that combines the two kinds of errors a binary classifier can make (false positives and false negatives) into a simple number.
            3. `import pyspark.ml.evaluation as evals`
            4. `evaluator = evals.BinaryClassificationEvaluator(metricName="areaUnderROC")`
   3. Create the grid of values to tune the model
      1. Use module `pyspark.ml.tuning`
      2. Class `ParamGridBuilder` of this module.
      3. Code
         1. `grid = ParamGridBuilder()`
         2. `grid.add(lr.regParam, np.arrange(0,1, 0.1))`
         3. `grid.add(lr.elasticNetParam, [0,1])`
            -  takes a model parameter (an attribute of the model Estimator, LogisticRegression().regParam) and a list of values that you want to try.
         4. `grid.build()`
            - method takes no arguments, it just returns the grid that we'll use later.
   4. Perform Cross Validation
      1. Create the validator
      2. Class `CrossValidator` from `pyspark.ml.tuning`
      3. This is an Estimator
      4. It takes the modeler you want to fit, the grid of hyperparameters you created, and the evaluator you want to use to compare your models.
      5. Code
         1. `import pyspark.ml.tuning as tune`
         2. `cv = tune.CrossValidator(estimator= lr, estimatorParamMaps= grid, evaluator=evaluator)`
   5. Fitting the model and selecting Best one
      1. `cv.fit(training_data)`
      2. `best_model  = cv.bestModel`
      3. Cross validation is computationally intensive task.
3. Evaluating Model on test set
   1. `test_results = best_model.transform(test_data)`
   2. `evaluator.evaluate(test_results)`
