# HW 1-2: RandomForest vs NaiveBayes Classifier Comparison

## 1. 과제 요약
* Apache Spark MLlib의 `RandomForestClassifier`와 `NaiveBayes` 모델을 `sample_libsvm_data.txt` 데이터셋에 대해 비교 평가합니다[cite: 1].
* 동일한 7:3 시드 기반 분할 조건 하에서 Random Forest의 `numTrees`(1, 10, 50, 100) 변화에 따른 정확도와 학습 시간을 Naive Bayes 기준 모델과 비교하고, `numTrees=100` 모델의 상위 5개 주요 피처 중요도를 출력합니다[cite: 1].

## 2. 실행 방법 (`hw1_2_cmd.txt`)
```bash
sbt package
/opt/spark/bin/spark-submit --class ClassifierCompare --master local[*] target/scala-2.12/hw1-2_2.12-1.0.jar
```[cite: 1, 2]

## 3. 실행 결과
![HW 1-2 실행 결과](./result_hw1_2.png)

## 4. 결과 해석
* Random Forest 모델은 트리의 개수(`numTrees`)가 늘어남에 따라 학습 시간은 증가하지만 예측 정확도가 1.0까지 향상되어, 베이스라인인 Naive Bayes 모델 대비 더 뛰어난 분류 성능을 나타냈습니다[cite: 1, 2].
