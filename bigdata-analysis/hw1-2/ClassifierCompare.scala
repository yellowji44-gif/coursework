// Student number: 20240207
import java.io.File
import java.util.Locale

import org.apache.spark.ml.classification.{NaiveBayes, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.SparkSession

object ClassifierCompare {
  def main(args: Array[String]): Unit = {
    val seed = 207L // 학번 마지막 네 자리(0207)

    val spark = SparkSession.builder().appName("ClassifierCompare").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    val path = new File("sample_libsvm_data.txt").getAbsoluteFile.toURI.toString
    val data = spark.read.format("libsvm").load(path)

    // randomSplit은 한 번만 호출하고, 모든 모델이 같은 train/test를 사용
    val Array(train, test) = data.randomSplit(Array(0.7, 0.3), seed)
    train.cache()
    test.cache()
    train.count() // 캐시를 미리 채워 학습 시간에 데이터 로딩이 섞이지 않게 함
    test.count()

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    println("model trees accuracy fit(s)")

    var importances100 = Array.empty[Double]

    for (n <- Array(1, 10, 50, 100)) {
      val t0 = System.nanoTime()
      val model = new RandomForestClassifier()
        .setNumTrees(n)
        .setSeed(seed)
        .fit(train)
      val sec = (System.nanoTime() - t0) / 1e9
      val acc = evaluator.evaluate(model.transform(test))
      println("RandomForest %d %.4f %.3f".formatLocal(Locale.US, n, acc, sec))
      if (n == 100) importances100 = model.featureImportances.toArray
    }

    // NaiveBayes: 한 번만 학습, 비교 기준
    val t0 = System.nanoTime()
    val nbModel = new NaiveBayes().fit(train)
    val nbSec = (System.nanoTime() - t0) / 1e9
    val nbAcc = evaluator.evaluate(nbModel.transform(test))
    println("NaiveBayes - %.4f %.3f".formatLocal(Locale.US, nbAcc, nbSec))

    // numTrees = 100 모델의 featureImportances 상위 5개
    val top = importances100.zipWithIndex
      .sortBy(p => -p._1)
      .take(5)
      .map(p => "%d(%.4f)".formatLocal(Locale.US, p._2, p._1))
      .mkString(" ")
    println("top features: " + top)
    println("(seed = " + seed + ")")

    spark.stop()
  }
}
