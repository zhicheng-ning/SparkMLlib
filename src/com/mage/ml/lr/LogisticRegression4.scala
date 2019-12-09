package com.mage.ml.lr

import java.util

import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel, LogisticRegressionSummary}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}


object LogisticRegression4 {

    def main(args: Array[String]) {
        val spark = SparkSession
            .builder
            .master("local")
            .appName("LogisticRegression4")
            .getOrCreate()
        //w0测试数据.txt
        val data: DataFrame = spark.read.format("libsvm").load("健康状况训练集.txt")

        val splits: util.List[Dataset[Row]] = data.randomSplitAsList(Array(0.8, 0.2), seed = 1L)

        val (trainingData, testData) = (splits.get(0), splits.get(1))

        val lr = new LogisticRegression()
            .setFeaturesCol("features")
            .setLabelCol("label")
            //最大迭代次数
            .setMaxIter(100)
            //设置截距
            .setFitIntercept(true)

        val model: LogisticRegressionModel = lr.fit(trainingData)

        // 癌症病人宁愿错判断出得癌症也别错过一个得癌症的病人
        model.setThreshold(0.3)

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
