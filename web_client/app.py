from flask import Flask, render_template, request
from model import generate_review

app = Flask(__name__)

@app.route('/')
def about_project():
    return render_template('about.html')  # Страница "О проекте"

@app.route('/app', methods=['GET', 'POST'])
def app_page():

    # Сброс значений при GET-запросе
    if request.method == 'GET':
        return render_template('app.html', review=None, category="", rating="", keywords="")

    if request.method == 'POST':
        rubrics = request.form.get('category')   # Получаем рубрику
        rating = int(request.form['rating'])     # Получаем рейтинг
        keywords = request.form.get('keywords')  # Получаем ключевые слова

        # Передаем ключевые слова в функцию генерации
        review = generate_review(rubrics, rating, keywords)

    return render_template('app.html', review=review)

@app.route('/about-us')
def about_us():
    return render_template('about_us.html')  # Страница "О нас"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)