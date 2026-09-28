# Запуск

Python 3.11. Данные (`train.parquet`, `benchmark_items.parquet`, `benchmark_queries.parquet`) лежат в этой же папке.

```bash
pip install -r requirements.txt
./run.sh
```

`run.sh` по очереди запускает:

1. `prepare.py` — таблица объявлений и симуляция бенчмарка на train;
2. `validate.py` — кандидаты, обучение ранкера, печатает Recall@50 на симуляции;
3. `submit.py` — пишет `answer.csv`.

## Пересчёт эмбеддингов

```bash
pip install torch sentence-transformers datasets pandas
python train_encoder.py sim
python train_encoder.py full
```

Обучение на GPU не детерминировано, поэтому после пересчета несколько объявлений в `answer.csv` могут поменяться.
