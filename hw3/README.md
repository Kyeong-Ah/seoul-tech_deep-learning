# Character-Level Language Modeling
Shakespeare 데이터셋을 활용한 문자 기반의 Neural Network modeling
## Models
### RNN
설명
### LSTM
LSTM에서 과적합이 발생.
learing_rate를 늘려서 해결 시도. : 0.02 -> 0.1
### Performance
RNN plot
LSTM plot
## Generate samples
### 5. Write generate.py to generate characters with your trained model. Choose the model showing the best validation performance. You should provide at least 100 length of 5 different samples generated from different seed characters.
seed character별 sample
### 6. (Report) Softmax function with a temperature parameter *T* can be written as: 
$$y_i = \frac{\exp(z_i/T)}{\displaystyle\sum \exp(z_i/T)}$$
Try different temperatures when you generate characters, and discuss what difference the temperature makes and why it helps to generate more plausible results.
    
