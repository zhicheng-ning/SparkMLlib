package com.mage.ml.lr

import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel, LinearRegressionSummary}
import org.apache.spark.sql.{Dataset, Encoders, SparkSession}

/**
  * 线性回归
  */
object LinearRegression1 {

    def main(args: Array[String]) {
        val spark = SparkSession
            .builder
            .master("local")
            .appName("LinearRegression")
            .getOrCreate()
        //scala. 隐式转换
        import spark.implicits._
        //读取样本数据
        val data_path1 = "lpsa.data"

        val data1: Dataset[String] = spark.read.textFile(data_path1)


        val data2: Dataset[LabeledPoint] = data1.map { line =>
            val parts = line.split(',')

            val features: Array[Double] = parts(1).split(' ').map(_.toDouble)

            LabeledPoint(parts(0).toDouble, Vectors.dense(features));
        }

//        //1 为随机种子
        val train2TestData: Array[Dataset[LabeledPoint]] = data2.randomSplit(Array(0.8, 0.2), 1)


        // 迭代次数
        val numIterations = 100

        val lir: LinearRegression = new LinearRegression()
            .setFeaturesCol("features")
            .setLabelCol("label")
            //收敛的值，越小结果越精确，但迭代次数也越大，花费更多时间
            .setTol(1E-1)
            //迭代次数
            .setMaxIter(numIterations)
            //是否需要截距，默认true
            .setFitIntercept(false)

        val startTime = System.nanoTime()

        val model: LinearRegressionModel = lir.fit(train2TestData(0))
        //训练模型所消耗的时间
        val elapsedTime = (System.nanoTime() - startTime) / 1e9

        println("Training time: " + elapsedTime +" seconds")
        //权重.
        println("Weights: " + model.coefficients)
        //截距.
        println("Intercept:" +  model.intercept)

        //用测试集数据去评估模型，得到一个评估结果。
        val summary: LinearRegressionSummary = model.evaluate(train2TestData(1))
//
//        //打印测试结果
        summary.predictions.show()


        //平均绝对误差，预测数据和原始数据对应点误差绝对值和的均值
        println("平均绝对值误差: " + summary.meanAbsoluteError)
//        //均方差，预测数据和原始数据对应点误差的平方和的均值
        println("均方差: " + summary.meanSquaredError)
//        //测试集的数据条目
        println(summary.numInstances)

        /**
          * 训练完之后，可以将模型进行保存..
          *  model.save("model/lir.model")
          *  模型训练完毕后，以后用的时候可以直接加载模型，无需再训练
          *  val model = LinearRegressionModel.load("model/lir.model")
          */

        spark.stop()


    }

}
