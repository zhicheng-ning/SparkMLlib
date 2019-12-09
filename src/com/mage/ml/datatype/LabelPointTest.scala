package com.mage.ml.datatype

import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
  * 创建LabelPoint(标注点)的两种方式：
  *    1.通过稀疏向量或稠密向量来创建LabelPoint
  *    2.通过加载LibSVM文件来创建
  */

object LabelPointTest {


    def main(args: Array[String]): Unit = {

//        //通过稠密向量，创建标注点
//        val lp1: LabeledPoint = LabeledPoint(1.0, Vectors.dense(1.0, 0.0, 3.0))
//        //通过稀疏向量，创建标注点
//        val lp2: LabeledPoint = LabeledPoint(0.0, Vectors.sparse(3, Array(0, 2), Array(1.0, 3.0)))
//
//        println("label: " + lp1.label)
//        println("features: " + lp1.features)

        /**
          * 加载LIBSVM文件来创建...
          */
        val spark = SparkSession
            .builder
            .master("local")
            .appName("LabelPoint")
            .getOrCreate()

        val data: DataFrame = spark.read.format("libsvm").load("健康状况训练集.txt")
//
        data.show()

        data.printSchema()

    }
}