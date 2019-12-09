package com.mage.ml.kmeans

import org.apache.spark.ml.clustering.{KMeans, KMeansModel}

import org.apache.spark.ml.linalg.{ SQLDataTypes, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types._
import org.apache.spark.sql._

/**
  * Kmeans聚类算法
  */
object KMeans1 {

    def main(args: Array[String]) {

        val spark = SparkSession
            .builder
            .master("local")
            .appName("Kmeans")
            .getOrCreate()

        val data: RDD[String] = spark.sparkContext.textFile("kmeans_data.txt")

        //数据转化...
        val rdd: RDD[Row] = data.map(s => Row(Vectors.dense(s.split(' ').map(_.toDouble))))

        val schema = StructType(List(
            StructField("features", SQLDataTypes.VectorType, nullable = false)
        ))

        val df: DataFrame = spark.createDataFrame(rdd, schema)

        val kmeans = new KMeans()
        //聚类数...
        kmeans.setK(3)
        //最大迭代次数
        kmeans.setMaxIter(20)
        //随机种子...
        kmeans.setSeed(1L)

        val model: KMeansModel = kmeans.fit(df)

        //中心点位置....
        model.clusterCenters.foreach(println)

        //评估模型的好坏，使用平方欧式距离测度
        val errors: Double = model.computeCost(df)

        println("平方误差: " + errors)

        model.save("model/kmeans")

        spark.close()

    }
}
