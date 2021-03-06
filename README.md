# Использование нейросетевой архитектуры YOLO для распоснания лиц

Данная работа нацелена на то, чтобы узнать, как изменится точность распознавания лиц, если вместо класической задачи распознавания решать комбинированную задачу детектирования и распознавания.

Моя гипотеза заключается в том, что при решении такой комбинированной задачи методами deep learning алгоритм обучится извлекать в том числе пространственные фичи, которые при традиционной постановки задачи он мог бы и проигнорировать. Тут также можно вспомнить проблему инвариантности сверточных сетей относительно сдвига элементов на изображении.

## Pipeline

Для решения задачи будет использован подход transfer learning. По изображениям будут сначала получаться эмбеддинги лиц с помощью алгоритма, обученного решать другую задачу.

Для построения этих эмбеддингов я планирую использовать сверточную нейронную сеть архитектуры [YOLO v3](https://pjreddie.com/darknet/yolov2/), написанную на PyTorch. На основе bounding box'ов, полученных этой сетью, из последнего слоя сети вырезается некоторый регион, который и будет выступать эмбеддингом.

Далее на этих эмбеддингах будет обучаться простой классификатор.

## Текущие результаты

- [X] Написан код для сети.
- [X] Сеть протестирована на датасете из 228 лиц в разном положении, с разным фоном и освещенностью. Сеть тратила в среднем 0.2 секунды на обработку одного изображения (на GPU) не распознав 1 лицо.
- [ ] По точности сеть (очевидно) выигрывает встроенным в OpenCV (Виола-Джонс) и Dlib (HOG) методам, хотя точное исследование я еще не сделал.
