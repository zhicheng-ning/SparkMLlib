package com.mage.ml.lr

import java.util

import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel, LogisticRegressionSummary}

import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

object LogisticRegression2 {
    def main(args: Array[String]) {

        val spark = SparkSession
            .builder
            .master("local")
            .appName("LogisticRegression2")
            .getOrCreate()

        val data: DataFrame = spark.read.format("libsvm").load("w0测试数据.txt")

        val splits: util.List[Dataset[Row]] = data.randomSplitAsList(Array(0.8, 0.2), seed = 1L)

        val (trainingData, testData) = (splits.get(0), splits.get(1))

        val lr = new LogisticRegression()
            .setFeaturesCol("features")
            .setLabelCol("label")
            //最大迭代次数
            .setMaxIter(100)
            //设置截距
            .setFitIntercept(false)

        val startTime = System.nanoTime()

        val model: LogisticRegressionModel = lr.fit(trainingData)
        //训练模型所消耗的时间
        val elapsedTime = (System.nanoTime() - startTime) / 1e9

        println("Training time: " + elapsedTime +"seconds")
        //权重.
        println("Weights: " + model.coefficients)
        //截距.
        println("Intercept:" +  model.intercept)

        val summary: LogisticRegressionSummary = model.evaluate(testData)

        val predictions: DataFrame = summary.predictions
        //打印测试结果
        predictions.show()

        predictions.createOrReplaceTempView("result")

        //计算正确率
        val accuracy: DataFrame = spark.sql("select (1- (sum(abs(label-prediction)))/count(label)) as accuracy from result")

        accuracy.show()

        spark.stop()
    }
}