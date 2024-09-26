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

`python src/model.py 2>/dev/null`

### Результаты
```
Random Guess MAE: 1521.3041260670682
Random Guess MAPE: 2.0145481685022957
Best parameters found:
 {'activation': 'relu', 'alpha': 0.0001, 'hidden_layer_sizes': (100,), 'learning_rate': 'constant', 'solver': 'adam'}
MLP MAE: 26.5181715590117
MLP MAPE: 0.0234864184142573
```
