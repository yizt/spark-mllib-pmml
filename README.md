# spark-mllib-pmml

####1：打包
mvn clean package -DskipTests

###例子，在192.168.1.218上执行
export HDP_VERSION=2.6.0.3-8
/usr/hdp/2.6.0.3-8/spark2/bin/spark-submit \
--master local \
--class com.es.analyze.spark2pmml.examples.DecisionTreeClassificationModel \
spark-mllib-pmml-0.01-SNAPSHOT.jar local hdfs://master:8020/user/hdfs/data/dataset/iris.csv /home/hdfs/pmml/spark2pmml_decision_tree.pmml
