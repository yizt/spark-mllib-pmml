package com.es.analyze.utils

import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.{Dataset, Row}

/**
  * Created by mick.yi on 2018/9/13.
  * 评估工具类
  */
object EvaluatorUtils {
  /**
    * 获取MulticlassMetrics对象，默认MulticlassClassificationEvaluator评估器，一次只能输出一个评估指标
    * @param dataset
    * @param evaluator
    * @return 返回MulticlassMetrics对象
    */
  def getMulticlassMetrics(dataset: Dataset[_],evaluator: MulticlassClassificationEvaluator):MulticlassMetrics={
    val schema = dataset.schema
    val predictionAndLabels =
      dataset.select(evaluator.getPredictionCol, evaluator.getLabelCol).rdd.map {
        case Row(prediction: Double, label: Double) => (prediction, label)
      }
    val metrics = new MulticlassMetrics(predictionAndLabels)
    metrics
  }

  /**
    * 根据MulticlassMetrics一次性获取常的评估指标结果
    * @param metrics
    * @return
    */
  def getCommonMetrics(metrics: MulticlassMetrics):Map[String,Double]={
    Map("f1" -> metrics.weightedFMeasure,
      "weightedPrecision" -> metrics.weightedPrecision,
      "weightedRecall" -> metrics.weightedRecall,
      "accuracy" -> metrics.accuracy)
  }

}
