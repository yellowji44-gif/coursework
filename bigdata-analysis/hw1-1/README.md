# HW 1-1: Linear Regression Regularization Sweep

## 1. 과제 요약
* Apache Spark MLlib의 `LinearRegression`을 사용하여 `sample_linear_regression_data.txt` 데이터셋을 학습합니다[cite: 1].
* `regParam` (0.0, 0.1, 0.3, 1.0)과 `elasticNetParam` (0.0, 0.5, 1.0)의 총 12가지 정규화 하이퍼파라미터 조합에 대해 RMSE, $R^2$, 절대값이 $1e-8$ 이하인 계수의 개수를 평가하고 최적의 조합을 산출합니다[cite: 1].

## 2. 실행 방법 (`hw1_1_cmd.txt`)
```bash
sbt package
/opt/spark/bin/spark-submit --class LinearRegressionSweep --master local[*] target/scala-2.12/hw1-1_2.12-1.0.jar
```[cite: 1]

## 3. 실행 결과
![HW 1-1 실행 결과](./result_hw1_1.png)

## 4. 결과 해석
* 정규화 파라미터(`regParam`) 값이 커질수록 계수 가중치가 억제되어 0에 가까운 계수 수가 증가하며, 해당 데이터셋에서는 정규화를 적용하지 않았을 때(`regParam=0.0`, `elasticNetParam=0.0`) 가장 낮은 RMSE를 기록하여 최적의 성능을 보였습니다[cite: 1].
