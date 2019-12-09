package com.mage.ml.nb

import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

/**
  * 贝叶斯算法
  */
object Naive_bayes {
    def main(args: Array[String]) {

        val spark = SparkSession
            .builder
            .master("local")
            .appName("NaiveBayesExample")
            .getOrCreate()

        val textFile: RDD[String] = spark.sparkContext.textFile("sms_spam.txt")

        val rdd: RDD[Row] = textFile.map(x=>{

            val values: Array[String] = x.split(",")

            //1.0为正常邮件 0.0为垃圾邮件
            val label: Double = if (values(0).equals("ham")) 1.0 else 0.0

            val features: Array[String] = values(1).split(" ").map(_.trim)
            Row(label,features)
        })

        val schema = StructType(List(
            StructField("label", DoubleType, nullable = false),
            StructField("words", ArrayType(StringType, true), nullable = false)
        ))

        val data: DataFrame = spark.createDataFrame(rdd,schema)

        //词表构造器..
        val countVectorizer = new CountVectorizer()
        //设置要进行构造词表的输入字段和构造后的输出字段
        countVectorizer.setInputCol("words").setOutputCol("features")

        //构建词汇表/词袋 【i hate you love dont 】
        val vectorizerModel: CountVectorizerModel = countVectorizer.fit(data)

        //查看词汇表
        vectorizerModel.vocabulary.take(100).foreach(println)

//        //文本向量化 CountVector (1,2,1) IdfVector  WordVertor
        val df: DataFrame = vectorizerModel.transform(data)
//
//        /**
//          *
//          * lable |                           words                                                               |           features
//          *
//          * 1.0   | [00, 00, 00, 00, 00, Hope, Hope, Hope, you, are, having, a, good, week., Just, checking, in]  |(13045,[1,3,6,20,79,82,231,327,705,997,1062],[1.0,1.0,1.0,1.0,1.0,1.0,3.0,1.0,1.0,1.0,5.0])
//          *
//          */

        df.show()
//
        val Array(trainingData, testData) = df.randomSplit(Array(0.8, 0.2), seed = 1234L)
//
        val nb: NaiveBayes = new NaiveBayes()
//        //训练模型...
//
        val model = nb.fit(trainingData)
//
//        //对训练后的模型进行测试,得到测试结果集。
        val predictions: DataFrame = model.transform(testData)
//        //打印测试结果集...
        predictions.show()
//
//        //多分类的评估器...
        val evaluator = new MulticlassClassificationEvaluator()
            //原始数据的结果值.label字段(y值)
            .setLabelCol("label")
            //预测字段名称..
            .setPredictionCol("prediction")
            //需要评估的指标名称，在这里我们要评估正确率，所以设置字段名为："accuracy"
            .setMetricName("accuracy")

////        //评估准确度...
        val accuracy: Double = evaluator.evaluate(predictions)

        println("准确率 = " + accuracy)

        spark.stop()

    }
}
