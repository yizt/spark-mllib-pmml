package com.es.analyze.utils

import org.apache.spark.ml.evaluation.{Evaluator, BinaryClassificationEvaluator, RegressionEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, RegressionMetrics, MulticlassMetrics}
import org.apache.spark.sql.{Dataset, Row}

/**
  * Created by mick.yi on 2018/9/13.
  * 评估工具类
  */
object EvaluatorUtils {
  /**
    * 获取MulticlassMetrics对象，默认MulticlassClassificationEvaluator评估器，一次只能输出一个评估指标
    *
    * @param dataset
    * @param evaluator
    * @return 返回MulticlassMetrics对象
    */
  def getMulticlassMetrics(dataset: Dataset[_], evaluator: MulticlassClassificationEvaluator): MulticlassMetrics = {
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
    *
    * @param metrics
    * @return
    */
  def getMulticlassMetricsValue(metrics: MulticlassMetrics): Map[String, Double] = {
    Map("f1" -> metrics.weightedFMeasure,
      "weightedPrecision" -> metrics.weightedPrecision,
      "weightedRecall" -> metrics.weightedRecall,
      "accuracy" -> metrics.accuracy)
  }

  /**
    * RegressionMetrics，RegressionEvaluator，一次只能输出一个评估指标
    *
    * @param dataset
    * @param evaluator
    * @return
    */
  def getRegressionMetrics(dataset: Dataset[_], evaluator: RegressionEvaluator): RegressionMetrics = {
    val schema = dataset.schema
    val predictionAndLabels =
      dataset.select(evaluator.getPredictionCol, evaluator.getLabelCol).rdd.map {
        case Row(prediction: Double, label: Double) => (prediction, label)
      }
    val metrics = new RegressionMetrics(predictionAndLabels)
    metrics
  }

  /**
    * 根据RegressionMetrics一次性获取常的评估指标结果
    *
    * @param metrics
    * @return
    */
  def getRegressionMetricsValue(metrics: RegressionMetrics): Map[String, Double] = {
    Map(
      "rmse" -> metrics.rootMeanSquaredError,
      "mse" -> metrics.meanSquaredError,
      "r2" -> metrics.r2,
      "mae" -> metrics.meanAbsoluteError
    )
  }


  /**
    * BinaryClassificationMetrics，BinaryClassificationEvaluator，一次只能输出一个评估指标
    *
    * @param dataset
    * @param evaluator
    * @return
    */
  def getBinaryClassificationMetrics(dataset: Dataset[_], evaluator: BinaryClassificationEvaluator): BinaryClassificationMetrics = {
    val schema = dataset.schema
    val predictionAndLabels =
      dataset.select(evaluator.getRawPredictionCol, evaluator.getLabelCol).rdd.map {
        case Row(prediction: Double, label: Double) => (prediction, label)
      }
    val metrics = new BinaryClassificationMetrics(predictionAndLabels)
    metrics
  }

  /**
    * 根据BinaryClassificationMetrics一次性获取常的评估指标结果
    *
    * @param metrics
    * @return
    */
  def getBinaryClassificationMetricsValue(metrics: BinaryClassificationMetrics): Map[String, Double] = {
    Map(
      "areaUnderROC" -> metrics.areaUnderROC(),
      "areaUnderPR" -> metrics.areaUnderPR()
    )
  }

  def getMetrics(dataset: Dataset[_], evaluator: Evaluator): Object = {
    val metrics = evaluator match {
      case e: BinaryClassificationEvaluator => getBinaryClassificationMetrics(dataset, e.asInstanceOf[BinaryClassificationEvaluator])
      case e: MulticlassClassificationEvaluator => getMulticlassMetrics(dataset, e.asInstanceOf[MulticlassClassificationEvaluator])
      case e: RegressionEvaluator => getRegressionMetrics(dataset, e.asInstanceOf[RegressionEvaluator])
    }
    metrics
  }

  def getMetricsValue(metrics: Object): Map[String, Double] = {
    metrics match {
      case m: BinaryClassificationMetrics => getBinaryClassificationMetricsValue(m)
      case m: MulticlassMetrics => getMulticlassMetricsValue(m)
      case m: RegressionMetrics => getRegressionMetricsValue(m)
    }
  }


}
