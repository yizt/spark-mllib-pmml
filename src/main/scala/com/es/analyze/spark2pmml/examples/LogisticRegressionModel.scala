package com.es.analyze.spark2pmml.examples

import java.io.FileOutputStream
import javax.xml.transform.stream.StreamResult

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{LogisticRegression}
import org.apache.spark.ml.feature.RFormula
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
import org.jpmml.model.JAXBUtil
import org.jpmml.sparkml.{PMMLBuilder, ConverterUtil}

/**
  * Created by mick.yi on 2017/11/3.
  * 逻辑回归
  */
object LogisticRegressionModel {
  def main(args: Array[String]) {
    //args eg:   local ./src/main/resources/data/iris.csv out/spark2pmml_logistic.pmml
    val master=args(0)
    val irisPath=args(1)
    val outPmmlFile=args(2)

    //    val irisPath="hdfs://master:8020/user/hdfs/data/dataset/iris.csv"
    //    val outPmmlFile="/home/hdfs/pmml/spark2pmml_logistic.pmml"

    val spark = SparkSession.builder.
      master(args(0)).
      appName("GradientBoostedTreeClassifier").
      getOrCreate()

    //原本列名含有点号，Spark DataFrame不支持列名含有点号
    val schema = new StructType(Array(
      StructField("SepalLength", DoubleType),
      StructField("SepalWidth", DoubleType),
      StructField("PetalLength", DoubleType),
      StructField("PetalWidth", DoubleType),
      StructField("Species", StringType)
    ))

    val irisData =spark.read.
      option("header", "true").option("delimiter", " ").
      schema(schema).
      csv(irisPath)

    //定义目标分类列和特征列
    val rFormula = new RFormula
    val formula: RFormula = rFormula.setFormula("Species ~ SepalLength + SepalWidth + PetalLength + PetalWidth")

    val logit = new LogisticRegression().
      setLabelCol(formula.getLabelCol).
      setFeaturesCol(formula.getFeaturesCol)

    val pipeline: Pipeline = new Pipeline setStages Array(formula, logit)
    val pipelineModel = pipeline.fit(irisData)

    //使用jpmml-sparkml导出为pmml模型
    val pmml = new PMMLBuilder(irisData.schema, pipelineModel).build()
    JAXBUtil.marshalPMML(pmml, new StreamResult(new FileOutputStream(outPmmlFile)))

    spark.stop()
  }

}
