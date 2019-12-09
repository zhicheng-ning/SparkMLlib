/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.mage.ml.association_rules

import org.apache.spark.ml.fpm.{FPGrowth, FPGrowthModel}
import org.apache.spark.sql.{DataFrame, Dataset}
// $example off$
import org.apache.spark.sql.SparkSession

/**
  * 关联规则....
  */
object FPGrowthExample {

    def main(args: Array[String]): Unit = {
        val spark: SparkSession = SparkSession
      .builder
      .master("local")
      .appName("FPGrowth")
      .getOrCreate()
    import spark.implicits._

    //加载数据
    val shoppings: Dataset[String] = spark.read.textFile("shopping_cart")

    //把数据通过空格分割,转成DataFrame

    val df: DataFrame = shoppings.map(_.split(",")).toDF("items")

    val growth = new FPGrowth().setItemsCol("items")
    //设置支持度和置信度
    growth.setMinConfidence(0.8)
    growth.setMinSupport(0.3)
    //设置分区数
    growth.setNumPartitions(2)

    val model: FPGrowthModel = growth.fit(df)
    //打印频繁项集
    model.freqItemsets.show();

    //打印符合置信度和支持度条件的关联规则
    model.associationRules.show()

    spark.stop()
}
}


