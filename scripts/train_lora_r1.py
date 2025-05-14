import ast
import re

import pandas as pd
import pytorch_lightning as pl
import torch
from datasets import Dataset
from lightning.pytorch.loggers import MLFlowLogger
from peft import LoraConfig, TaskType, get_peft_model
import pytorch_lightning as L
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from torch.utils.data import DataLoader
from transformers import (
    MT5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)

import mlflow_settings

# Доля от общего размера датасета, которая будет поделена
# на тренировочную и тестовую части
DATASET_SIZE = 0.9

# Доля тестовой части от DATASET_SIZE
TEST_SIZE = 0.1

# Доля валидационной выборки от датасета, которая не была
# взята для тренировки и оценки, т.е. часть от доли (1 - DATASET_SIZE)
VAL_SIZE = 0.9

# Установка затравки для повторяемости результатов
SEED = 42

# Константа, которая определяет будет ли проводиться дообучение модели
ENABLE_FINE_TUNE = True

# Каталог для сохранения результатов обучения
SAVE_DIRECTORY = "./t5_lora_r1"

# Имя предобученного токенизатора
TOKENIZER_NAME = "kavlab/review-t5"

# В этот раз в качестве базовой будет модель, которая ранее была обучена
# на всем датасете
MODEL_NAME = "kavlab/review-t5"

# Загружаем очищенный и подготовленный датасет
print("Загружаем датасет...")
data = pd.read_csv(
    "./data/kw_cleared_dataset.csv", sep=";", encoding="utf-8-sig", index_col=0
)
data["rubrics_list"] = data["rubrics_list"].apply(ast.literal_eval)
data["key_words"] = data["key_words"].apply(ast.literal_eval)

data.columns = ["rating", "rubrics_list", "text", "key_words"]

# Из датасета отбираем только те отзывы, у которых рейтинг равен 1 или 2
data = data.query("rating == 1 or rating == 2")

data["text"] = data["text"].apply(lambda text: re.sub(r"\s+", " ", text).strip())


def make_query(row):
    query = (
        f"<rubrics>: {', '.join(row['rubrics_list'])} | "
        f"<raiting>: {row['rating']} | "
        f"<keywords>: {', '.join(row['key_words'])}"
    )
    return query


# Формируем строку запроса для обучения модели
data["query"] = data.apply(make_query, axis=1)

data.dropna(inplace=True)

# Вычислим длины отзывов, чтобы определиться с параметрами токенизации
df_lens = data["text"].apply(lambda row: len(row))

data = data[df_lens <= 512]

print(f"Dataset size: {data.shape[0]}")

# print(data[["query", "text"]].head())

set_seed(SEED)

# Создание датасета Hugging Face
source_dataset = Dataset.from_pandas(data[["query", "text"]])

# Делим общий датасет на две части, одна будет использоваться для
# тернировки и тестирования, а другая для валидации
test_split_dataset = source_dataset.train_test_split(
    test_size=DATASET_SIZE,
    shuffle=True,
    seed=SEED,
)

# В данном случае "train" - это оставшаяся часть датасета (1 - DATASET_SIZE),
# из нее берем часть для валидационной выборки
val_dataset = test_split_dataset["train"].train_test_split(
    test_size=VAL_SIZE,
    shuffle=True,
    seed=SEED,
)

# Делим на тренировочную и тестовую части
dataset = test_split_dataset["test"].train_test_split(
    test_size=TEST_SIZE,
    shuffle=True,
    seed=SEED,
)

# Добавляем в dataset валидационный набор
dataset["validation"] = val_dataset["test"]

# Загружаем с Hugging Face токенизатор
tokenizer = T5Tokenizer.from_pretrained(TOKENIZER_NAME)

# Максимальная длина входных данных
# Во входных данным мало символов, основная часть в рубриках, т.к.
# может быть перечислено несколько значений
max_input_length = 512

# Максимальная длина выходных данных (отзыва)
# Берем значение 512, т.к. более 75% отзывов имееют меньшую длину
max_target_length = 512

label_pad_token_id = -100


def preprocess_data(data):
    """Функция предобработки для токенизации"""
    inputs = data["query"]
    texts = data["text"]

    model_inputs = tokenizer(
        inputs, max_length=max_input_length, padding="max_length", truncation=True
    )

    labels = tokenizer(
        texts, max_length=max_target_length, padding="max_length", truncation=True
    ).input_ids

    # Заменяем индекс токенов заполнения на -100,
    # чтобы они не учитывались в CrossEntropyLoss
    labels_with_ignore_index = []
    for labels_example in labels:
        labels_example = [
            label if label != 0 else label_pad_token_id for label in labels_example
        ]
        labels_with_ignore_index.append(labels_example)

    model_inputs["labels"] = labels_with_ignore_index

    return model_inputs


# Выполняем предобработку всех частей датасета
print("Выполняем предобработку всех частей датасета...")
dataset = dataset.map(
    preprocess_data,
    batched=True,
    cache_file_names={
        "train": "./cache/datasets_train_r1",
        "test": "./cache/datasets_test_r1",
        "validation": "./cache/datasets_val_r1",
    },
)

dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Создаем объекты DataLoader с размерами батчей, чтобы они помещались в
# видеопамяти. Значение batch_size=8 для тренировочной выборки было
# подобрано для NVidia RTX 4070Ti с 12 Гб видеопамяти. На этой карте
# проводилось обучение модели.
train_dataloader = DataLoader(
    dataset["train"], shuffle=True, batch_size=8, num_workers=9
)
valid_dataloader = DataLoader(dataset["validation"], batch_size=8, num_workers=9)
test_dataloader = DataLoader(dataset["test"], batch_size=8, num_workers=9)

torch.set_float32_matmul_precision("medium")


# Создаем класс модели с наследованием от pytorch_lightning.LightningModule
# PyTorch Lightning упрощает процесс обучения, в т.ч. следит за тем, чтобы
# все тензоры были на одном устройсте
class ReviewT5(pl.LightningModule):
    def __init__(self, model, lr=5e-5, num_train_epochs=15, warmup_steps=1000):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=["model"])

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        return outputs

    def common_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log("training_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

        num_train_optimization_steps = self.hparams.num_train_epochs * len(
            train_dataloader
        )
        lr_scheduler = {
            "scheduler": get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.warmup_steps,
                num_training_steps=num_train_optimization_steps,
            ),
            "name": "learning_rate",
            "interval": "step",
            "frequency": 1,
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def train_dataloader(self):
        return train_dataloader

    def val_dataloader(self):
        return valid_dataloader

    def test_dataloader(self):
        return test_dataloader


peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=16,
    lora_alpha=1,
    lora_dropout=0.01,
    target_modules=["k", "q", "v"],
)

# Загружаем базовую модель
model = MT5ForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    revision="1.0.0"
)

# Подготавливаем: замораживаем все веса исходной модели, добавляем LoRa
peft_model = get_peft_model(model, peft_config)

# Выводим количество обучаемых параметров
peft_model.print_trainable_parameters()

# Создаем объект модели
model = ReviewT5(model=peft_model)

callbacks = [
    EarlyStopping(
        monitor="validation_loss", patience=3, strict=False, verbose=False, mode="min"
    ),
    LearningRateMonitor(logging_interval="step"),
    ModelCheckpoint(save_top_k=1, mode="min", monitor="validation_loss"),
]

mlf_logger = MLFlowLogger(
    experiment_name="t5_training_logs",
    tracking_uri=mlflow_settings.MLFLOW_TRACKING_URI,
    log_model=False,
)

trainer = L.Trainer(
    default_root_dir="./checkpoints_r1",
    callbacks=callbacks,
    logger=mlf_logger,
    max_epochs=15,
)

# Запускаем процесс обучения
print("Запускаем процесс обучения...")
trainer.fit(model)

# Сохраняем результаты в каталог
model.model.save_pretrained(SAVE_DIRECTORY)

print(f"Обученная модель сохранена в каталоге {SAVE_DIRECTORY}")
