package com.es.analyze.utils

import com.es.analyze.train.TrainMain.Params
import ml.dmlc.xgboost4j.scala.spark.{XGBoostRegressor, XGBoostClassifier}
import org.apache.spark.ml.classification.{GBTClassifier, RandomForestClassifier, DecisionTreeClassifier}
import org.apache.spark.ml.regression.{GBTRegressor, RandomForestRegressor, DecisionTreeRegressor}

/**
  * Created by mick.yi on 2018/9/14.
  * 模型工具类
  */
object ModelUtils {
  def getDecisionTreeClassifier(p: Params): DecisionTreeClassifier = {
    val classifier: DecisionTreeClassifier = new DecisionTreeClassifier
    classifier
  }

  def getDecisionTreeRegressor(p: Params): DecisionTreeRegressor = {
    val rgs: DecisionTreeRegressor = new DecisionTreeRegressor()
    rgs
  }

  def getRandomForestClassifier(p: Params): RandomForestClassifier = {
    val classifier: RandomForestClassifier = new RandomForestClassifier
    classifier
  }

  def getRandomForestRegressor(p: Params): RandomForestRegressor = {
    val rgs: RandomForestRegressor = new RandomForestRegressor
    rgs
  }

  def getGBTClassifier(p: Params): GBTClassifier = {
    val classifier: GBTClassifier = new GBTClassifier
    classifier
  }

  def getGBTRegressor(p: Params): GBTRegressor = {
    val rgs: GBTRegressor = new GBTRegressor
    rgs
  }
  /**
    * xgb可以设置的参数非常多,以下是部分参数默认设置
    *
    * a) 提升树默认参数
    * (eta -> 0.3, gamma -> 0, maxDepth -> 6,
    * minChildWeight -> 1, maxDeltaStep -> 0,
    * growPolicy -> "depthwise", maxBins -> 16,
    * subsample -> 1, colsampleBytree -> 1, colsampleBylevel -> 1,
    * lambda -> 1, alpha -> 0, treeMethod -> "auto", sketchEps -> 0.03,
    * scalePosWeight -> 1.0, sampleType -> "uniform", normalizeType -> "tree",
    * rateDrop -> 0.0, skipDrop -> 0.0, lambdaBias -> 0, treeLimit -> 0)
    *
    * b) 通用默认参数
    * (numRound -> 1, numWorkers -> 1, nthread -> 1,
    * useExternalMemory -> false, silent -> 0,
    * customObj -> null, customEval -> null, missing -> Float.NaN,
    * trackerConf -> TrackerConf(), seed -> 0, timeoutRequestWorkers -> 30 * 60 * 1000L,
    * checkpointPath -> "", checkpointInterval -> -1
    * )
    * c) 以下参数必须指定
    * objective:分类可选目标函数"binary:logistic","binary:logitraw", "count:poisson", "multi:softmax", "multi:softprob"
    * num_class: 类别数
    */
  def getXGBoostClassifier(p: Params): XGBoostClassifier = {
    val xgbParam = parse_kwargs(p.kwargs)
    val xgbClassifier = new XGBoostClassifier(xgbParam)
    xgbClassifier
  }

  /**
    * xgb可以设置的参数非常多,以下是部分参数默认设置
    *
    * a) 提升树默认参数
    * (eta -> 0.3, gamma -> 0, maxDepth -> 6,
    * minChildWeight -> 1, maxDeltaStep -> 0,
    * growPolicy -> "depthwise", maxBins -> 16,
    * subsample -> 1, colsampleBytree -> 1, colsampleBylevel -> 1,
    * lambda -> 1, alpha -> 0, treeMethod -> "auto", sketchEps -> 0.03,
    * scalePosWeight -> 1.0, sampleType -> "uniform", normalizeType -> "tree",
    * rateDrop -> 0.0, skipDrop -> 0.0, lambdaBias -> 0, treeLimit -> 0)
    *
    * b) 通用默认参数
    * (numRound -> 1, numWorkers -> 1, nthread -> 1,
    * useExternalMemory -> false, silent -> 0,
    * customObj -> null, customEval -> null, missing -> Float.NaN,
    * trackerConf -> TrackerConf(), seed -> 0, timeoutRequestWorkers -> 30 * 60 * 1000L,
    * checkpointPath -> "", checkpointInterval -> -1
    * )
    * c) 训练参数
    * objective:分类可选目标函数"reg:linear", "reg:logistic"，"reg:gamma", "reg:tweedie"；默认为：reg:linear
    *
    * @param p
    * @return
    */
  def getXGBoostRegressor(p:Params):XGBoostRegressor={
    val xgbParam = parse_kwargs(p.kwargs)
    val xgbRgr=new XGBoostRegressor(xgbParam)
    xgbRgr
  }

  /**
    * 转换值类型
    * @param kwargs
    *
    * @return 转换为Double,Int,Boolean后的Map
    */
  def parse_kwargs(kwargs: Map[String,String]) = {

   kwargs.map{ case (key, value) => {
      var v: Any = value
      try
        v = value.toInt
      catch {
        case _ => {
          try v = value.toDouble
          catch {
            case _ => {
              try v = value.toBoolean
              catch {
                case _ => {
                  v = value
                }
              }
            }
          }
        }
      }
      (key, v)
    }
    }
  }


}
