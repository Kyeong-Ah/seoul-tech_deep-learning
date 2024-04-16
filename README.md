# 서울과학기술대학교 2024 상반기 인공신경망과 딥러닝

## 1. The number of model parameters of LeNet-5 and custom MLP and how to compute them.
### LeNet-5 모델의 파라미터 수
#### 총 파라미터 개수: 61,706개
* 첫 번째 컨볼루션 층: 입력 채널 1, 출력 채널 6, 커널 사이즈 5x5   
   파라미터 수: (1 x 6 x 5 x 5) + 6 = 156    
* 두 번째 컨볼루션 층: 입력 채널 6, 출력 채널 16, 커널 사이즈 5x5   
   파라미터 수: (6 x 16 x 5 x 5) + 16 = 2,416   
   
* 첫 번째 완전 연결 층: 입력 크기 1655=400, 출력 크기 120   
   파라미터 수: (400 x 120) + 120 = 48,120   
* 두 번째 완전 연결 층: 입력 크기 120, 출력 크기 84   
   파라미터 수: (120 x 84) + 84 = 10,164   
* 세 번째 완전 연결 층: 입력 크기 84, 출력 크기 10   
   파라미터 수: (84 x 10) + 10 = 850   
따라서, 이 LeNet-5 모델의 총 파라미터 수는 156 + 2,416 + 48,120 + 10,164 + 850 = 61,706개 입니다.   

### Custom MLP 모델의 파라미터 수
#### 총 파라미터 개수: 61,520개
CustomMLP 모델의 구조는 다음과 같습니다:   

* 첫 번째 완전 연결 층: 입력 크기 32*32=1024, 출력 크기 57   
   파라미터 수: (1024 x 57) + 57 = 58,361   
* 두 번째 완전 연결 층: 입력 크기 57, 출력 크기 47   
   파라미터 수: (57 x 47) + 47 = 2,679   
* 세 번째 완전 연결 층: 입력 크기 47, 출력 크기 10   
   파라미터 수: (47 x 10) + 10 = 480   
따라서, 이 CustomMLP 모델의 총 파라미터 수는 58,361 + 2,679 + 480 = 61,520개 입니다.   
   
앞서 계산한 LeNet-5 모델의 총 파라미터 수가 61,706개였던 것을 고려하면, CustomMLP 모델의 파라미터 개수가 LeNet-5와 비슷한 수준임을 확인할 수 있습니다.   

## 2. Plots for each model.

## 3. Compare the predictive performances of LeNet-5 and custom MLP.
LeNet-5 ACC(%) = 
   
CustomMLP ACC(%) = 
   

## 4. Regularization techniques to improve LeNet-5   
* Dropout
* L2 regularization(Weight decay)

LeNet-5 = 
   
Regularization 후 LeNet-5 = 
   
정규화 후 성능이 향상할 것이라고 기대했던 결과에 반해, 성능이 소폭 감소하는 것을 확인할 수 있습니다.
