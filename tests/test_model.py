import pytest
from transformers import T5ForConditionalGeneration, T5Tokenizer
from web_client.app import app as flask_app


@pytest.fixture
def loaded_model():
    # Загрузка модели и токенизатора (предполагается, что они сохранены локально или в Hugging Face Hub)
    model_name = "your_fine_tuned_t5_model"
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    return model, tokenizer


@pytest.fixture
def client():
    flask_app.config['TESTING'] = True
    with flask_app.test_client() as client:
        yield client


def test__model_generation(loaded_model):
    """Тест генерации отзыва моделью на стандартных параметрах"""
    model, tokenizer = loaded_model
    input_text = "Ресторан 5 вкусно, уютно, обслуживание"

    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    assert len(generated_text) > 0
    assert "ресторан" in generated_text.lower()


def test_model_with_different_ratings(loaded_model):
    """Тест генерации с разными рейтингами"""
    model, tokenizer = loaded_model

    for rating in [1, 3, 5]:
        input_text = f"Кафе {rating} кофе, завтрак"
        inputs = tokenizer(input_text, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=100)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        assert str(rating) in generated_text or ("плох" in generated_text.lower() if rating == 1 else "")


def test_flask_api(client):
    """Тест API веб-приложения"""
    test_data = {
        "category": "Ресторан",
        "rating": 4,
        "keywords": "пицца, паста, вино"
    }

    response = client.post('/generate', json=test_data)
    assert response.status_code == 200
    assert "text" in response.json
    assert len(response.json["text"]) > 0


def test_empty_keywords(loaded_model):
    """Тест генерации без ключевых слов"""
    model, tokenizer = loaded_model
    input_text = "Бар 3 "

    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    assert len(generated_text) > 0
    assert "бар" in generated_text.lower()


def test_invalid_input(client):
    """Тест обработки некорректных входных данных"""
    invalid_data = {
        "category": "",
        "rating": 6,  # рейтинг вне диапазона
        "keywords": ""
    }

    response = client.post('/generate', json=invalid_data)
    assert response.status_code == 400
    assert "error" in response.json