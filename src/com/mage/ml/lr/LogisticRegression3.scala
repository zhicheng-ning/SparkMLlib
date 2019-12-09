package com.mage.ml.lr

import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel, LogisticRegressionSummary}
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vectors}
import org.apache.spark.sql._


object LogisticRegression3 {

    def main(args: Array[String]) {

        val spark = SparkSession
            .builder
            .master("local")
            .appName("LogisticRegression3")
            .getOrCreate()

        import spark.implicits._


        val df: DataFrame = spark.read.format("libsvm").load("线性不可分数据集.txt")
        // 解决线性不可分我们来升维,升维有代价,计算复杂度变大了
        val ds: Dataset[LabeledPoint] = df.map(row => {

            //x:Row
            val label: Double = row.getAs[Double]("label")
            val features: SparseVector = row.getAs[SparseVector]("features")


            val array = features.toArray
            //升维，生成第三维度，第三维度的值，可根据业务经验来定。
            val convertFeatures: linalg.Vector = Vectors.dense(array(0),array(1),  array(0) - array(1) )

            LabeledPoint(label, convertFeatures)
        })

        val splits = ds.randomSplit(Array(0.8, 0.2), 1)

        val (trainingData, testData) = (splits(0), splits(1))

        val lr = new LogisticRegression()
            .setFeaturesCol("features")
            .setLabelCol("label")

        val startTime = System.nanoTime()

        val model: LogisticRegressionModel = lr.fit(trainingData)
        //训练模型所消耗的时间
        val elapsedTime = (System.nanoTime() - startTime) / 1e9

        println("Training time: " + elapsedTime + "seconds")
        //权重.
        println("Weights: " + model.coefficients)
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
    }
}
