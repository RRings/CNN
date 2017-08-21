## 1. Clone해서 실행하기

Clone 받아서 실행하기 전에 설치 해야할 것들.

> pip install tensorflow
>
> pip install numpy
>
> pip install pandas

설치후 실행.

오류가 나거나 잘 동작이 안할경우 직접 가상환경을 만들어서 작업해보세요.



---



## 2. 가상환경 만들고 실행파일과 데이터만 추가

1. ### 가상환경

   1. **가상환경 만들기**

      Mac 쓰시는분들 

      > virtualenv -p python3 <name>

   2. **가상환경 실행**

      위에서 만든 폴더 안에서

      > source bin/activate

2. ### 실행파일, 데이터 가져오기

   1. **실행파일**
      * cnn_sentence_classification.py
      * cnn_tool.py
   2. **데이터**
      * ratings.csv

3. ### 실행준비하기

   > pip install tensorflow
   >
   > pip install pandas
   >
   > pip install numpy

4. ### 실행 

   실행하기 전에 `cnn_sentence_classification.py` 파일안의 데이터 경로를 수정해준다. ( data_path )

   ​

이미 작업하시던 폴더가 있다면 데이터파일과 실행파일만 가져와서 실행하셔도 되요 . 

보통 작업하시는 폴더는 tensorflow 와 numpy 가 설치되어 있으실 꺼니 pandas 만 설치 하시면 될겁니다.