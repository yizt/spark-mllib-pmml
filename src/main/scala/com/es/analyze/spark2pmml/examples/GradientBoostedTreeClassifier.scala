package com.es.analyze.spark2pmml.examples

import java.io.FileOutputStream
import javax.xml.transform.stream.StreamResult

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{DecisionTreeClassifier, GBTClassificationModel, GBTClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, RFormula, StringIndexer, VectorIndexer}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
import org.jpmml.model.JAXBUtil
import org.jpmml.sparkml.ConverterUtil
/**
  * Created by mick.yi on 2017/11/2.
  * 梯度提升分类树(目前只支持二分类)
  * java.lang.IllegalArgumentException: requirement failed: GBTClassifier was given dataset with invalid label 2.0.  Labels must be in {0,1}; note that GBTClassifier currently only supports binary classification.
  */
object GradientBoostedTreeClassifier {
  def main(args: Array[String]): Unit = {
    //args eg:   local ./src/main/resources/data/iris.csv out/spark2pmml_gbt.pmml
    val master=args(0)
    val irisPath=args(1)
    val outPmmlFile=args(2)

    //    val irisPath="hdfs://master:8020/user/hdfs/data/dataset/iris.csv"
    //    val outPmmlFile="/home/hdfs/pmml/spark2pmml_gbt.pmml"

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

    val gbt = new GBTClassifier().
      setLabelCol(formula.getLabelCol).
      setFeaturesCol(formula.getFeaturesCol)

    val pipeline: Pipeline = new Pipeline setStages Array(formula, gbt)
    val pipelineModel = pipeline.fit(irisData)

    //使用jpmml-sparkml导出为pmml模型
    val pmml = ConverterUtil.toPMML(irisData.schema, pipelineModel)
    JAXBUtil.marshalPMML(pmml, new StreamResult(new FileOutputStream(outPmmlFile)))

    spark.stop()
  }

}
