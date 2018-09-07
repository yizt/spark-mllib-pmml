# spark-mllib-pmml
### 首先执行src/main/resources/install.txt中相关操作

### 1：打包
mvn clean package -DskipTests

### 2: 例子，将jar包上传到192.168.1.218的/opt/pmml目录下，并执行如下：

export HDP_VERSION=2.6.0.3-8

/usr/hdp/2.6.0.3-8/spark2/bin/spark-submit \
--master local \
--class com.es.analyze.spark2pmml.examples.DecisionTreeClassificationModel \
spark-mllib-pmml-0.01-SNAPSHOT.jar local hdfs://master:8020/user/hdfs/data/dataset/iris.csv /home/hdfs/pmml/spark2pmml_decision_tree.pmml
