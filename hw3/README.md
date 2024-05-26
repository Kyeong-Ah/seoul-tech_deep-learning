# Character-Level Language Modeling
Shakespeare 데이터셋을 활용한 문자 기반의 Neural Network modeling
## Models
### hyper parameters
RNN과 LSTM은 둘 다 동일한 hyper parameter로 학습하였습니다.
- batch_size = 64
- seq_length = 30
- epochs = 50
- lr = 0.001
- patience = 5
- vocab_size = len(dataset.chars)
- hidden_size = 256
- n_layers = 2
모델 학습 중에 성능이 개선되지 않으면 학습이 중단되도록 early_stopping을 설정하였습니다.
#### 과적합
처음에는 hidden_size=512, learning_rate=0.002였으나 과적합이 발생하여 이를 해결하기 위해 아래와 같이 값을 조정하였습니다. 그 결과 validation loss가 감소하였고, 변경한 상태로 최종 학습을 진행하였습니다.
    - hidden_size = 256
    - learning_rate = 0.001

### Performance
RNN loss plot
![RNN_loss_plot](https://github.com/Kyeong-Ah/seoul-tech_deep-learning/assets/97220162/ec9759f3-5676-4c5f-870e-da967c8dceb0)

LSTM loss plot
![LSTM_loss_plot](https://github.com/Kyeong-Ah/seoul-tech_deep-learning/assets/97220162/7e149ed5-baaa-47f0-990f-e0dffcd80606)



## Generate samples
### 5. Write generate.py to generate characters with your trained model. Choose the model showing the best validation performance. You should provide at least 100 length of 5 different samples generated from different seed characters.
seed character별 sample
### 6. (Report) Softmax function with a temperature parameter *T* can be written as: 
$$y_i = \frac{\exp(z_i/T)}{\displaystyle\sum \exp(z_i/T)}$$
Try different temperatures when you generate characters, and discuss what difference the temperature makes and why it helps to generate more plausible results.
    
