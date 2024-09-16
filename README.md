# Предсказание необходимой для декодирования длины синдрома для различных кодов

## Код Хэмминга
### Запуск
_Инструкции для запуска на Linux._

Создайте виртуальное окружение

`python3 -m venv env && source env/bin/activate`

Установите зависимости

`pip install -r requirements.txt`

Для запуска тестов для кода Хэмминга:

`python -m unittest tests/test_codec.py`

Для генерации файла `out.csv`, содержащего список необходимых для декодирования длин синдромов для соответствующих ошибок:
_Занимает до нескольких минут._

`python src/simulation.py`

Для запуска NN-модели, предсказывающей длину синдрома:
_Занимает до нескольких минут из-за поиска оптимальных параметров._

`python src/model.py 2>/dev/null`

### Результаты
```
Random Guess Accuracy: 0.559748427672956
Best parameters found:
 {'activation': 'tanh', 'alpha': 0.0001, 'hidden_layer_sizes': (100,), 'learning_rate': 'adaptive', 'solver': 'adam'}
MLP Accuracy: 0.8427672955974843
```
