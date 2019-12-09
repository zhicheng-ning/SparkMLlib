package com.mage.ml.lr

import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.feature.{StandardScaler, StandardScalerModel}
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}


/**
  * 数值优化：
  *     均值归一化
  *     标准差归一化
  */
object LogisticRegression6 {

    def main(args: Array[String]) {
        val spark = SparkSession
            .builder
            .master("local")
            .appName("LogisticRegression1")
            .getOrCreate()

        var df: DataFrame = spark.read.format("libsvm").load("环境分类数据.txt")

        /*******************   数值优化代码开始：   ******************/

        //数值标准优化器.
        val scaler = new StandardScaler()
        //均值归一化
        scaler.setWithMean(true)
        //标准差归一化
        scaler.setWithStd(true)
        //设置输入字段，即要对哪个字段的值进行数值优化
        scaler.setInputCol("features")
        //设置输出字段，即数值优化后的新字段名称..
        scaler.setOutputCol("standard_features")
        //通过指定的数据集，构建模型。PS：scaler数值标准优化器就会对数据集进行计算得出它的平均值和标准差..
        val scalerModel: StandardScalerModel = scaler.fit(df)

        println(scalerModel.mean)

        println(scalerModel.std)

        //用模型对数据进行数值优化，得出新的值。
        df = scalerModel.transform(df)
//
        df.show()
//        //删除原有数据字段..
        df = df.drop("features")
//        //将数值优化后的字段，改名为features
        df = df.withColumnRenamed("standard_features", "features")
//        //改名后的dataframe
        df.show()
//
        /*******************   数值优化代码结束   ******************/
////
        val splits = df.randomSplit(Array(0.8, 0.2), 1L)

        val (trainingData, testData) = (splits(0), splits(1))
        val lr = new LogisticRegression

//        lr.setFeaturesCol("standard_features")

        val startTime = System.nanoTime()

        val model: LogisticRegressionModel = lr.fit(trainingData)

        //训练模型所消耗的时间
        val elapsedTime = (System.nanoTime() - startTime) / 1e9

        println("Training time: " + elapsedTime + "seconds")

        //权重.
        println("Weights: " + model.coefficients.toDense)

        val summary = model.evaluate(testData)

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
