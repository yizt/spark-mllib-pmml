package com.es.analyze.xgboost2pmml.examples

import java.io.{FileOutputStream, File, PrintWriter}
import javax.xml.transform.stream.StreamResult

import org.apache.spark.ml.feature.{IndexToString, VectorAssembler, StringIndexer}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
import ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier
import org.jpmml.model.JAXBUtil
import org.jpmml.sparkml.PMMLBuilder

/**
  * Created by mick.yi on 2018/9/4.
  */
object XGBoost_RFormula {
  def main(args: Array[String]) {
    //args eg:   local[4] ./src/main/resources/data/iris.csv out/xgboost2pmml.pmml
    val master=args(0)
    val irisPath=args(1)
    val outPmmlFile=args(2)


    val spark = SparkSession.builder.
      master(args(0)).
      appName("XGBoostModel").
      getOrCreate()

    //原本列名含有点号，Spark DataFrame不支持列名含有点号
    val schema = new StructType(Array(
      StructField("SepalLength", DoubleType),
      StructField("SepalWidth", DoubleType),
      StructField("PetalLength", DoubleType),
      StructField("PetalWidth", DoubleType),
      StructField("class", StringType)
    ))
    val rawInput = spark.read.
      option("header", "true").option("delimiter", " ").
      schema(schema).
      csv(irisPath)

    val stringIndexer = new StringIndexer().
      setInputCol("class").
      setOutputCol("classIndex").
      fit(rawInput)
    //val labelTransformed = stringIndexer.transform(rawInput).drop("class")



    val vectorAssembler = new VectorAssembler().
      setInputCols(Array("SepalLength", "SepalWidth", "PetalLength", "PetalWidth")).
      setOutputCol("features")


    val xgbParam = Map("eta" -> 0.1f,
      "max_depth" -> 2,
      "objective" -> "multi:softprob",
      "num_class" -> 3,
      "num_round" -> 100,
      "num_workers" -> 2)

    val xgbClassifier = new XGBoostClassifier(xgbParam).
      setFeaturesCol("features").
      setLabelCol("classIndex").setProbabilityCol("origin_prob")

    val labelConverter = new IndexToString()
      .setInputCol("classIndex")
      .setOutputCol("realLabel")
      .setLabels(stringIndexer.labels)

    import org.apache.spark.ml.Pipeline
    import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

    val Array(training, test) = rawInput.randomSplit(Array(0.8, 0.2), 123)


    val pipeline = new Pipeline()
      .setStages(Array(stringIndexer,vectorAssembler, xgbClassifier, labelConverter))

    val model = pipeline.fit(training)

    val prediction = model.transform(test)
    val evaluator = new MulticlassClassificationEvaluator() setLabelCol "classIndex"
    evaluator.setMetricName("f1")
    val accuracy = evaluator.evaluate(prediction)
    print(accuracy)
    prediction.show()

    // 导出pmml
    val pmml = new PMMLBuilder(rawInput.schema, model).build()
    JAXBUtil.marshalPMML(pmml, new StreamResult(new FileOutputStream(outPmmlFile)))
    print(pmml.toString)
  }
}
