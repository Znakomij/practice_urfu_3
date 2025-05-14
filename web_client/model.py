from transformers import T5Tokenizer, MT5ForConditionalGeneration

# Загрузка токенизатора
tokenizer = T5Tokenizer.from_pretrained("kavlab/review-t5", revision="1.0.0")

# Загрузка модели
model = MT5ForConditionalGeneration.from_pretrained("kavlab/review-t5", revision="1.0.0")

# Максимальная длина сгенерированного текста
max_target_length = 512

# Функция для генерации отзыва
def generate_review(rubrics, rating, keywords):

    # Формирование запроса для модели
    query = f"Рубрики: {rubrics}. Рейтинг: {rating}. Ключевые слова: {keywords}."

    # Токенизация запроса
    input_ids = tokenizer(query, return_tensors="pt").input_ids

    # Генерация текста с использованием модели
    outputs = model.generate(
        input_ids,
        max_length=max_target_length,
        do_sample=True,
        temperature=0.8,
        repetition_penalty=1.4,
        num_beams=4,
        top_k=30,
        top_p=0.9,
    )

    # Декодирование ответа
    generated_review = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_review