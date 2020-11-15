#! /usr/bin/python3.6
import findspark
findspark.init('/usr/lib/spark-current')
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("Python Spark with DataFrame").getOrCreate()
from pyspark.sql.types import *
from pyspark.ml.feature import Binarizer

from pyspark.sql.functions import col, when
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.sql.window import Window
import sys

#借用了李丰老师的函数：
def get_sdummies(sdf, dummy_columns, keep_top, replace_with='other'):
    """    Index string columns and group all observations that occur in less then a keep_top% of the rows in sdf per column.
    :param sdf: A pyspark.sql.dataframe.DataFrame
    :param dummy_columns: String columns that need to be indexed
    :param keep_top: List [1, 0.8, 0.8]
    :param replace_with: String to use as replacement for the observations that need to be grouped.
    """
    total = sdf.count()
    column_i = 0
    for string_col in dummy_columns:

        # Descending sorting with counts
        sdf_column_count = sdf.groupBy(string_col).count().orderBy(
            'count', ascending=False)
        sdf_column_count = sdf_column_count.withColumn(
            "cumsum",
            F.sum("count").over(Window.partitionBy().orderBy().rowsBetween(
                -sys.maxsize, 0)))

        # Obtain top dummy factors
        sdf_column_top_dummies = sdf_column_count.withColumn(
            "cumperc", sdf_column_count['cumsum'] /
            total).filter(col('cumperc') <= keep_top[column_i])
        keep_list = sdf_column_top_dummies.select(string_col).rdd.flatMap(
            lambda x: x).collect()
        sdf = sdf.withColumn(
            string_col,
            when((col(string_col).isin(keep_list)),
                 col(string_col)).otherwise(replace_with))

        # Apply string indexer
        pipeline = Pipeline(stages=[
            StringIndexer(inputCol=string_col, outputCol="IDX_" + string_col)
        ])
        sdf = pipeline.fit(sdf).transform(sdf)

        encoder = OneHotEncoder(inputCol="IDX_" + string_col,
                                outputCol="ONEHOT_" + string_col)
        encoder.setDropLast(True)  # only keep 2^n-n dummies to keep dummy independent.
        sdf = encoder.transform(sdf)

        column_i += 1

    ## Drop intermediate columns
    drop_columns = ["IDX_" +x for x in dummy_columns] # +  dummy_columns
    sdf = sdf.drop(*drop_columns)

    return sdf

schema_sdf = StructType([
        StructField('Year', IntegerType(), True),
        StructField('Month', IntegerType(), True),
        StructField('DayofMonth', IntegerType(), True),
        StructField('DayOfWeek', IntegerType(), True),
        StructField('DepTime', DoubleType(), True),
        StructField('CRSDepTime', DoubleType(), True),
        StructField('ArrTime', DoubleType(), True),
        StructField('CRSArrTime', DoubleType(), True),
        StructField('UniqueCarrier', StringType(), True),
        StructField('FlightNum', StringType(), True),
        StructField('TailNum', StringType(), True),
        StructField('ActualElapsedTime', DoubleType(), True),
        StructField('CRSElapsedTime',  DoubleType(), True),
        StructField('AirTime',  DoubleType(), True),
        StructField('ArrDelay',  DoubleType(), True),
        StructField('DepDelay',  DoubleType(), True),
        StructField('Origin', StringType(), True),
        StructField('Dest',  StringType(), True),
        StructField('Distance',  DoubleType(), True),
        StructField('TaxiIn',  DoubleType(), True),
        StructField('TaxiOut',  DoubleType(), True),
        StructField('Cancelled',  IntegerType(), True),
        StructField('CancellationCode',  StringType(), True),
        StructField('Diverted',  IntegerType(), True),
        StructField('CarrierDelay', DoubleType(), True),
        StructField('WeatherDelay',  DoubleType(), True),
        StructField('NASDelay',  DoubleType(), True),
        StructField('SecurityDelay',  DoubleType(), True),
        StructField('LateAircraftDelay',  DoubleType(), True)
    ])
air = spark.read.options(header='true').schema(schema_sdf).csv("/data/airdelay_small.csv")
#air2=air.na.drop()
air1 = air.select(["ArrDelay","Year","DayofMonth","DayofWeek","DepTime","CRSDepTime","CRSArrTime","UniqueCarrier","ActualElapsedTime","Origin","Dest","Distance"])
air3=air1.na.drop()
binarizer = Binarizer(threshold=0, inputCol="ArrDelay", outputCol="Delay_feature")
air2 = binarizer.transform(air3)
air2.show()

df=get_sdummies(air2,air2.columns,[1,1,1,1,1,1,1,0.8,1,0.5,0.6,1,1])
df1=df.select(["Delay_feature","Year","DayofMonth","DayofWeek","DepTime","CRSDepTime","CRSArrTime","UniqueCarrier","ActualElapsedTime","Origin","Dest","Distance"])


#binarizer = Binarizer(threshold=0, inputCol="ArrDelay", outputCol="Delay_feature")
#df2 = binarizer.transform(df1)

categories_UC = df1.select("UniqueCarrier").distinct().rdd.flatMap(lambda x: x).collect()
exprs_UC= [F.when(F.col("UniqueCarrier") == category, 1).otherwise(0).alias(category) for category in categories_UC]

categories_OG = df1.select("Origin").distinct().rdd.flatMap(lambda x: x).collect()
exprs_OG= [F.when(F.col("Origin") == category, 1).otherwise(0).alias(category) for category in categories_OG]

categories_DS = df1.select("Dest").distinct().rdd.flatMap(lambda x: x).collect()
exprs_DS= [F.when(F.col("Dest") == category, 1).otherwise(0).alias(category) for category in categories_DS]

df2=df1.select("Delay_feature","Year","DayofMonth","DayofWeek","DepTime","CRSDepTime","CRSArrTime","ActualElapsedTime" ,"Distance",*exprs_UC,*exprs_OG,*exprs_DS)
df2.show()
