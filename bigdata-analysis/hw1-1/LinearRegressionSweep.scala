// Student number: 20240207
import java.io.File
import java.util.Locale

import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SparkSession

object LinearRegressionSweep {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().appName("LinearRegressionSweep").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    // 실행 위치(프로젝트 디렉터리)의 데이터 파일을 절대 경로 URI로 읽는다
    val path = new File("sample_linear_regression_data.txt").getAbsoluteFile.toURI.toString
    val data = spark.read.format("libsvm").load(path).cache()
    data.count()

    val regParams   = Array(0.0, 0.1, 0.3, 1.0)
    val elasticNets = Array(0.0, 0.5, 1.0)

    println("reg enet RMSE r2 zeros")

    // 4 x 3 = 12개 조합을 한 번의 실행에서 모두 학습
    val results = for (reg <- regParams; enet <- elasticNets) yield {
      val lr = new LinearRegression()
        .setMaxIter(100)
        .setRegParam(reg)
        .setElasticNetParam(enet)
      val model = lr.fit(data)

      val rmse  = model.summary.rootMeanSquaredError
      val r2    = model.summary.r2
      val zeros = model.coefficients.toArray.count(c => math.abs(c) < 1e-8)

      println("%.2f %.2f %.4f %.4f %d".formatLocal(Locale.US, reg, enet, rmse, r2, zeros))
      (reg, enet, rmse)
    }

    val best = results.minBy(_._3)
    println("Best: regParam=%.2f, elasticNetParam=%.2f, RMSE=%.4f"
      .formatLocal(Locale.US, best._1, best._2, best._3))

    spark.stop()
  }
}
