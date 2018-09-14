package com.es.analyze.train

import com.es.analyze.utils.{EvaluatorUtils, PmmlUtils, ModelUtils}
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator, RegressionEvaluator, Evaluator}
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator, TrainValidationSplit}
import org.apache.spark.ml.{PipelineModel, PipelineStage, Pipeline, Predictor}
import org.apache.spark.ml.feature.RFormula
import org.apache.spark.sql.{SparkSession, DataFrame}
import scopt.OptionParser

/**
  * Created by mick.yi on 2018/9/13.
  * 训练入口类
  */
object TrainMain {

  /** 命令行参数 */
  case class Params(modelName: String = "", //模型名称
                    tableName: String = "", //表名
                    featureCols: Seq[String] = Seq.empty, //特征列
                    targetCol: String = "", //目标列
                    filters: String = "", //过滤条件
                    outputPmml: String = "", //pmml输出路径
                    modelSavePath: String = "", //模型保存路径
                    crossValid: Boolean = false, //是否交叉验证
                    folds: Int = 3, //交叉验证数据分为多少折
                    splitRatio: Seq[Double] = Seq(3, 1), //训练测试分割比例
                   trainRatio:Double=0.8, //训练验证比例
                   parallel:Int=1, //并行度
                    kwargs: Map[String, String] = Map.empty, //其它个性化参数
                    appName: String = "TrainMain"
                   )

  val parser = new OptionParser[Params]("Train_Spark_Model") {
    head("Train Spark Model:.")
    opt[String]('m', "model")
      .required()
      .text("模型名称")
      .action((x, c) => c.copy(modelName = x))
      .validate(x => {
        if ("DTCls,DTRgr,RFCls,RFRgr,GBTCls,GBTRgr,XGBCls,XGBRgr".contains(x)) success
        else failure("Option --模型名称必须是DTCls,DTRgr,RFCls,RFRgr,GBTCls,GBTRgr,XGBCls,XGBRgr中一个")
      })
    //输入数据
    opt[String]('t', "tablename")
      .required()
      .text("表名")
      .action((x, c) => c.copy(tableName = x))
    opt[Seq[String]]('f', "featureColumns").valueName("col1,<col2,<col3,<...>>>")
      .required()
      .text("特征列")
      .action((x, c) => c.copy(featureCols = x))
    opt[String]("targetColumn")
      .required()
      .text("目标列")
      .action((x, c) => c.copy(targetCol = x))
    opt[String]("filters")
      .optional()
      .text("过滤条件")
      .action((x, c) => c.copy(filters = x))

    //训练参数
    opt[Seq[Double]]("splitRatio")
      .valueName("<trainweight,testweight>")
      .text("分割比例,如:3,1")
      .action((x, c) => c.copy(splitRatio = x))

    //输出参数
    opt[String]("pmml")
      .optional()
      .text("pmml文件输出路径")
      .action((x, c) => c.copy(outputPmml = x))
    opt[String]('s', "modelSavePath")
      .optional()
      .text("模型保存路径")
      .action((x, c) => c.copy(modelSavePath = x))


    //交叉验证
    opt[Unit]("crossvalid").action((_, c) =>
      c.copy(crossValid = true)).text("crossvalid是标志参数,代表启用交叉验证")
    opt[Int]('k', "folds")
      .text("k折交叉验证")
      .action((x, c) => c.copy(folds = x))

    //其它参数
    opt[Map[String, String]]("kwargs")
      .valueName("k1=v1,k2=v2...").
      action((x, c) => c.copy(kwargs = x)).text("其它参数")

  }

  def main(args: Array[String]) {
    val spark = SparkSession.builder.
      master(args(0)).
      appName("DecisionTreeRegressionModel").
      getOrCreate()
    val mpgData = spark.read.option("header", "true").option("delimiter", " ").csv("")

    parser.parse(args, new Params()).map(p => {
      run(p,mpgData)
    }) getOrElse {
      System.exit(1)
    }
  }
  def run(p: Params,df:DataFrame): Unit = {
    //分割数据集
    val Array(trainDF, testDF) = df.randomSplit(p.splitRatio.toArray)


    //配置pipeline
    val rFormual=getRFormula(p)
    val model=getModel(p).
      setLabelCol(rFormual.getLabelCol).
      setFeaturesCol(rFormual.getFeaturesCol)
    val pipeline = new Pipeline()
      .setStages(Array(rFormual, model.asInstanceOf[PipelineStage]))

    //训练评估
    val paramGrid = new ParamGridBuilder()
      .build()
    val (trainModel,evaluator)=if(p.crossValid){ //交叉验证
      val cv = new CrossValidator()
        .setEstimator(pipeline)
        .setEvaluator(getEvaluator(p))
        .setEstimatorParamMaps(paramGrid)
        .setNumFolds(p.folds)
        .setParallelism(p.parallel)
      val cm=cv.fit(trainDF)
        (cm.bestModel.asInstanceOf[PipelineModel],cm.getEstimator)
    }else{ //训练验证分割
      val trainValidationSplit = new TrainValidationSplit()
        .setEstimator(pipeline)
        .setEvaluator(getEvaluator(p))
        .setEstimatorParamMaps(paramGrid)
        .setTrainRatio(p.trainRatio)
        .setParallelism(p.parallel)
      val tm=trainValidationSplit.fit(trainDF)
      (tm.bestModel.asInstanceOf[PipelineModel],tm.getEvaluator)
    }

    //查看预测结果
    val prediction = trainModel.transform(testDF)

    //val multiMetrics=EvaluatorUtils.getMulticlassMetrics(prediction,evaluator)
    //保存模型
    if(!"".equals(p.modelSavePath))
      trainModel.save(p.modelSavePath)
    //导出pmml
    if(!"".equals(p.outputPmml))
      PmmlUtils.save(df.schema,trainModel,p.outputPmml)
  }

  /**
    * 获取RFormula
    *
    * @param p
    * @return
    */
  def getRFormula(p:Params):RFormula={
    val rFormula = new RFormula
    rFormula.setFormula(s"${p.targetCol} ~ ${p.featureCols.mkString(" + ")}")
    rFormula
  }

  /**
    * 获取模型
    *
    * @param p
    * @return
    */
  def getModel(p:Params)={
    val model=p.modelName match {
      case "DTCls" => ModelUtils.getDecisionTreeClassifier(p)
      case "DTRgr" => ModelUtils.getDecisionTreeRegressor(p)

      case "RFCls" => ModelUtils.getRandomForestClassifier(p)
      case "RFRgr" => ModelUtils.getRandomForestRegressor(p)

      case "GBTCls" => ModelUtils.getGBTClassifier(p)
      case "GBTRgr" => ModelUtils.getGBTRegressor(p)

      case "XGBCls" => ModelUtils.getXGBoostClassifier(p)
      case "XGBRgr" => ModelUtils.getXGBoostRegressor(p)
    }
    model
  }
  def getEvaluator(p:Params):Evaluator={
    if(p.modelName.endsWith("Rgr"))
      new RegressionEvaluator()
    else
      new MulticlassClassificationEvaluator()
  }

}
