package com.mage.ml.lr

import java.util

import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel, LogisticRegressionSummary}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

/**
  * 正则化: .setElasticNetParam(0)   L2正则化
  *        .setElasticNetParam(1)   L1正则化
  *
  * 惩罚系数 : setRegParam(0.1)
  *
  */
object LogisticRegression5 {

    def main(args: Array[String]) {

        val spark = SparkSession
            .builder
            .master("local")
            .appName("LogisticRegression1")
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
            //设置正则化: 0代表L2正则化  1代表L1正则化，0~1之间则会结合使用两种正则化
            .setElasticNetParam(1)
            // 这块设置的是我们的lambda,越大越看重这个模型的推广能力,一般不会超过1
            .setRegParam(0.1)
            //设置截距
            .setFitIntercept(true)

        val startTime = System.nanoTime()

        val model: LogisticRegressionModel = lr.fit(trainingData)
        //训练模型所消耗的时间
        val elapsedTime = (System.nanoTime() - startTime) / 1e9

        println("Training time: " + elapsedTime + "seconds")
        //权重.
        println("Weights: " + model.coefficients.toDense)
        //截距.
        println("Intercept:" + model.intercept)

        val summary: LogisticRegressionSummary = model.evaluate(testData)

        val predictions: DataFrame = summary.predictions
        //打印测试结果
        predictions.show()

        predictions.createOrReplaceTempView("result")

        //计算正确率
        val accuracy: DataFrame = spark.sql("select (1- (sum(abs(label-prediction)))/count(label)) as accuracy from result")

        accuracy.show()

        spark.stop()

        /**
          * 0.1 惩罚系数：  0.7617 正确率..
          *
          *     0.05159150816897747,0.2222755137929913,0.0035876562369314266,-0.016202006580241048,0.011691895227040026,-0.005507356003182153
          *
          * 0.4 惩罚系数：  0.7607 正确率..
          *     0.025438178618154554,0.12154325053347248,-7.038659228247561E-5,-0.007973885238669962,0.0044269437497883,-0.0026903235333777796
          *
          * 由此得出：惩罚系数越大，正确率越低，但推广能力越强，即牺牲正确率，来提高推广能力
          *
          * 0.1 惩罚系数：L1正则化  0.7631正确率
          *      0.04767093543883304,0.0,0.0,0.0,0.0,0.0
          *
          * 默认是L2正则化，由此得出L1正则化，偏向特征值取0; L2正则化，偏向整体偏小
          *
          */

    }
}
