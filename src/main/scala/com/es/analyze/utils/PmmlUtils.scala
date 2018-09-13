package com.es.analyze.utils

import java.io.FileOutputStream
import javax.xml.transform.stream.StreamResult

import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.types.StructType
import org.jpmml.model.JAXBUtil
import org.jpmml.sparkml.PMMLBuilder

/**
  * Created by mick.yi on 2018/9/12.
  * Pmml模型导出工具类
  */
object PmmlUtils {
  /**
    * 将模型保存为pmml文件
    * @param schema 输入数据的schema
    * @param pipelineModel 训练好的PipelineModel
    * @param outFile pmml输出文件
    */
  def save(schema:StructType ,pipelineModel:PipelineModel,outFile:String): Unit ={
    val pmml = new PMMLBuilder(schema, pipelineModel).build()
    JAXBUtil.marshalPMML(pmml, new StreamResult(new FileOutputStream(outFile)))
  }

}
