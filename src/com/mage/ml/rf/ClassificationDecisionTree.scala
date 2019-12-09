package com.mage.ml.rf

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
  * 决策树...
  */

object ClassificationDecisionTree {

    def main(args: Array[String]): Unit = {

        val spark = SparkSession
            .builder
            .master("local")
            .appName("classificationDecisionTree")
            .getOrCreate()

        val df = spark.read.format("libsvm").load("汽车数据样本.txt")

        val splits = df.randomSplit(Array(0.8, 0.2), 33L)

        val (trainingData, testData) = (splits(0), splits(1))


        //创建决策树...
        val decisionTree = new DecisionTreeClassifier()

        //决策树的最大深度..太深运算量大也没有必要  剪枝
        decisionTree.setMaxDepth(3)
        //设置离散化程度,连续数据需要离散化,分成32个区间,默认其实就是32,分割的区间保证数量差不多  这个参数也可以进行剪枝
        decisionTree.setMaxBins(6)
        //设定评判标准,只能选择"entropy"(熵) 或 "gini"(基尼系数),默认是通过基尼系数
        decisionTree.setImpurity("entropy")

        val model = decisionTree.fit(trainingData)

        val predictions = model.transform(testData)

        predictions.show()

        val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")

        val accuracy = evaluator.evaluate(predictions)

        println("正确率: " + accuracy)

        println("决策树构造: " + model.toDebugString)

        spark.stop()
    }
}
