import gradio as gr
import pypdf
import json
import os
import zipfile
import time
import random
from datetime import datetime
from openai import OpenAI

# --- Конфигурация ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
MAX_CHARS = 6000        # для бибописания (данные в начале статьи)
MAX_CHARS_REFERAT = 8000  # для реферата нужен больший охват

client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

AGENT_PROMPT = """Ты — библиотечный ИИ-агент для извлечения библиографических данных.
Извлеки: 1. Название журнала 2. Том 3. Выпуск 4. Год 5. Название статьи 6. Автора 7. Организацию 8. Название статьи на англ 9. Автора на англ.
Если поля нет, ставь "Не определено". Верни СТРОГО JSON:
{"journal_title_ru": "...", "volume": "...", "issue": "...", "year": "...", "article_title_ru": "...", "authors_ru": "...", "affiliation_ru": "...", "article_title_en": "...", "authors_en": "..."}
Текст статьи:
"""

REFERAT_PROMPT = """Ты — научный референт. Прочитай фрагмент статьи и подготовь реферат на русском языке со СТРОГО следующей структурой:

БИБЛИОГРАФИЧЕСКАЯ ЗАПИСЬ:
КРАТКАЯ АННОТАЦИЯ: (2-3 предложения)
ЦЕЛЬ РАБОТЫ:
МЕТОДЫ:
ОСНОВНЫЕ РЕЗУЛЬТАТЫ:
ВЫВОДЫ:

Используй только факты из текста. Если информация отсутствует — напиши «Не раскрыто в тексте».
Текст статьи:
"""

# --- Извлечение текста и вызов LLM ---
def extract_text_from_pdf(pdf_path, max_pages=3):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = pypdf.PdfReader(file)
        for page in reader.pages[:max_pages]:
            text += page.extract_text() + "\n"
    return text

def call_biblio_agent(text):
    try:
        response = client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": AGENT_PROMPT + text[:MAX_CHARS]}],
            temperature=0.1,
            format="json"
        )
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        return {"error": "Модель вернула некорректный JSON. Попробуйте другую модель."}
    except Exception as e:
        return {"error": f"Ошибка Ollama: {e}. Проверьте: ollama run {OLLAMA_MODEL}"}

def call_referat_agent(text):
    try:
        response = client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": REFERAT_PROMPT + text[:MAX_CHARS_REFERAT]}],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return None

# --- Демо-данные ---
def get_demo_data():
    return {
        "journal_title_ru": "Биофизика", "volume": "XXIX", "issue": "6", "year": "1984",
        "article_title_ru": "Определение частоты спинового обмена нитроксильных радикалов и кислорода методом непрерывного насыщения спектров ЭПР радикалов, присоединенных к белкам",
        "authors_ru": "Юданова Е. И., Куликов А. В.",
        "affiliation_ru": "Отделение института химической физики АН СССР, Черноголовка (Московская область)",
        "article_title_en": "Determination of Spin-Exchange Frequency of Nitroxide Radicals and Oxygen by the Method of Continuous Saturation of Radical ESR Spectra Bound to Proteins",
        "authors_en": "Yudanova E. I., Kulikov A. V."
    }

def get_demo_referat():
    return """БИБЛИОГРАФИЧЕСКАЯ ЗАПИСЬ:
Юданова Е. И., Куликов А. В. Определение частоты спинового обмена нитроксильных радикалов и кислорода методом непрерывного насыщения спектров ЭПР радикалов, присоединенных к белкам // Биофизика. — 1984. — Т. XXIX, вып. 6.

КРАТКАЯ АННОТАЦИЯ:
Методом непрерывного насыщения спектров ЭПР определены частоты спинового обмена нитроксильных радикалов, ковалентно связанных с белками, с парамагнитными центрами и молекулярным кислородом. Показано, что измеряемые частоты позволяют количественно оценивать локальную концентрацию кислорода и динамику белковых структур.

ЦЕЛЬ РАБОТЫ:
Разработка и апробация ЭПР-методики количественного определения частоты спинового обмена спиновых меток в белковых системах.

МЕТОДЫ:
Спектроскопия ЭПР в режиме непрерывного насыщения; нитроксильные радикалы в качестве спиновых меток; варьирование концентрации кислорода и парамагнитных центров.

ОСНОВНЫЕ РЕЗУЛЬТАТЫ:
Получены зависимости частоты спинового обмена от концентрации кислорода; продемонстрирована чувствительность метода к конформационным изменениям белков.

ВЫВОДЫ:
Предложенный подход применим для мониторинга кислородного режима и структурной динамики белковых систем в биофизических экспериментах."""

# --- Форматирование вывода ---
def format_to_txt(data):
    if "error" in data:
        return data["error"]
    return f"""Название журнала: {data.get('journal_title_ru', 'Не определено')}
Том: {data.get('volume', 'Не определено')}
Выпуск (номер): {data.get('issue', 'Не определено')}
Год: {data.get('year', 'Не определено')}
Название статьи: {data.get('article_title_ru', 'Не определено')}
Автор(ы): {data.get('authors_ru', 'Не определено')}
Организация: {data.get('affiliation_ru', 'Не определено')}
Название статьи на английском: {data.get('article_title_en', 'Не определено')}
Автор(ы) на английском: {data.get('authors_en', 'Не определено')}"""

def format_to_bibtex(data):
    if "error" in data:
        return None
    authors = data.get('authors_ru', '').strip()
    first_author = "unknown"
    if authors and authors != "Не определено":
        first_author = authors.split(',')[0].strip().split()[-1]
    year = data.get('year', '').strip()
    if not year or year == "Не определено":
        year = "nd"
    return (
        f"@article{{{first_author}{year},\n"
        f"  journal = {{{data.get('journal_title_ru', '')}}},\n"
        f"  volume  = {{{data.get('volume', '')}}},\n"
        f"  number  = {{{data.get('issue', '')}}},\n"
        f"  year    = {{{year}}},\n"
        f"  title   = {{{data.get('article_title_ru', '')}}},\n"
        f"  author  = {{{authors}}}\n"
        f"}}"
    )

# --- Консоль (тёмная) ---
def log_line(msg):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    tag = ''.join(random.choice('0123456789ABCDEF') for _ in range(4))
    return f"<div class='log-line'><span class='log-time'>[{ts}]</span> <span class='log-tag'>[0x{tag}]</span> {msg}</div>"

def log_success(msg):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    return f"<div class='log-success'><span class='log-time'>[{ts}]</span> > {msg}</div>"

def log_error(msg):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    return f"<div class='log-error'><span class='log-time'>[{ts}]</span> > {msg}</div>"

def render_console(content):
    return f"<div class='telemetry-console'>{content}</div>"

DEMO_STEPS = [
    "Инициализация ядра агента [OK]",
    "Загрузка файлов в изолированную среду",
    "Распознавание структуры PDF",
    "Семантический анализ контекста",
    "Извлечение признаков документа",
    "Преобразование данных в структурированный формат",
    "Сборка итогового документа [DONE]"
]

def unique_path(directory, base, ext, used):
    name, n = base, 1
    while f"{name}.{ext}" in used:
        name = f"{base}_{n}"
        n += 1
    used.add(f"{name}.{ext}")
    return os.path.join(directory, f"{name}.{ext}"), name

# --- Агент 1: Библио-Граф ---
def process_pdfs(pdf_files, is_demo, progress=gr.Progress()):
    output_dir = "output_biblio"
    os.makedirs(output_dir, exist_ok=True)
    output_files, used = [], set()

    if is_demo:
        files_to_process = [{"name": "ben01000360110-4.pdf", "is_demo": True}]
    else:
        if not pdf_files:
            yield "Ошибка: загрузите PDF", None, render_console(log_error("Файлы не загружены"))
            return
        files_to_process = [
            {"name": os.path.basename(getattr(f, "orig_name", None) or f.name),
             "path": f.name, "is_demo": False}
            for f in pdf_files
        ]

    progress(0, desc="Инициализация...")
    console_html = ""
    yield "Подготовка...", None, render_console(console_html)
    if is_demo:
        time.sleep(0.4)

    for i, file_info in enumerate(files_to_process):
        fname = file_info["name"]

        if file_info["is_demo"]:
            for j, step in enumerate(DEMO_STEPS):
                console_html += log_line(step)
                progress((i + j / len(DEMO_STEPS)) / len(files_to_process), desc=f"Обработка: {fname}")
                yield f"Обработка {i + 1} из {len(files_to_process)}: {fname}", None, render_console(console_html)
                time.sleep(0.35)
            biblio_data = get_demo_data()
        else:
            console_html += log_line(f"Извлечение текста: {fname}")
            yield f"Обработка {i + 1} из {len(files_to_process)}: {fname}", None, render_console(console_html)
            try:
                text = extract_text_from_pdf(file_info["path"])
            except Exception as e:
                console_html += log_error(f"Не удалось прочитать {fname}: {e}")
                yield f"Ошибка чтения: {fname}", None, render_console(console_html)
                continue

            console_html += log_line(f"Запрос к локальной модели {OLLAMA_MODEL}...")
            progress((i + 0.5) / len(files_to_process), desc=f"LLM: {fname}")
            yield f"Обработка {i + 1} из {len(files_to_process)}: {fname}", None, render_console(console_html)
            biblio_data = call_biblio_agent(text)

            if "error" in biblio_data:
                console_html += log_error(biblio_data["error"])
                yield f"Ошибка: {fname}", None, render_console(console_html)
                continue

        txt_path, out_name = unique_path(output_dir, os.path.splitext(fname)[0], "txt", used)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(format_to_txt(biblio_data))
        output_files.append(txt_path)

        bib = format_to_bibtex(biblio_data)
        if bib:
            bib_path, _ = unique_path(output_dir, out_name, "bib", used)
            with open(bib_path, "w", encoding="utf-8") as f:
                f.write(bib)
            output_files.append(bib_path)

        console_html += log_success(f"{out_name}: запись сохранена (TXT + BibTeX)")
        yield f"Готово: {out_name}", None, render_console(console_html)

    if not output_files:
        yield "❌ Ни один файл не обработан", None, render_console(console_html)
        return

    zip_path = os.path.join(output_dir, "biblio_results.zip")
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for fp in output_files:
            zipf.write(fp, os.path.basename(fp))

    console_html += log_success(f"Архив собран: {len(output_files)} файлов")
    yield "✅ Бибописания готовы! Скачайте архив.", zip_path, render_console(console_html)

# --- Агент 2: Реферат-Про ---
def process_referats(pdf_files, is_demo, progress=gr.Progress()):
    output_dir = "output_referat"
    os.makedirs(output_dir, exist_ok=True)
    output_files, used = [], set()
    last_text = ""

    if is_demo:
        files_to_process = [{"name": "demo_article.pdf", "is_demo": True}]
    else:
        if not pdf_files:
            yield "Ошибка: загрузите PDF", "", None, render_console(log_error("Файлы не загружены"))
            return
        files_to_process = [
            {"name": os.path.basename(getattr(f, "orig_name", None) or f.name),
             "path": f.name, "is_demo": False}
            for f in pdf_files
        ]

    progress(0, desc="Инициализация...")
    console_html = ""
    yield "Подготовка...", "", None, render_console(console_html)
    if is_demo:
        time.sleep(0.4)

    for i, file_info in enumerate(files_to_process):
        fname = file_info["name"]

        if file_info["is_demo"]:
            for j, step in enumerate(DEMO_STEPS):
                console_html += log_line(step)
                progress((i + j / len(DEMO_STEPS)) / len(files_to_process), desc=f"Обработка: {fname}")
                yield f"Обработка {i + 1} из {len(files_to_process)}: {fname}", last_text, None, render_console(console_html)
                time.sleep(0.35)
            referat_text = get_demo_referat()
        else:
            console_html += log_line(f"Извлечение текста: {fname}")
            yield f"Обработка {i + 1} из {len(files_to_process)}: {fname}", last_text, None, render_console(console_html)
            try:
                text = extract_text_from_pdf(file_info["path"])
            except Exception as e:
                console_html += log_error(f"Не удалось прочитать {fname}: {e}")
                yield f"Ошибка чтения: {fname}", last_text, None, render_console(console_html)
                continue

            console_html += log_line(f"Генерация реферата ({OLLAMA_MODEL}, локально)...")
            progress((i + 0.5) / len(files_to_process), desc=f"LLM: {fname}")
            yield f"Обработка {i + 1} из {len(files_to_process)}: {fname}", last_text, None, render_console(console_html)
            referat_text = call_referat_agent(text)

            if not referat_text:
                console_html += log_error(f"Модель не вернула результат для {fname}")
                yield f"Ошибка: {fname}", last_text, None, render_console(console_html)
                continue

        txt_path, out_name = unique_path(output_dir, os.path.splitext(fname)[0] + "_referat", "txt", used)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(referat_text)
        output_files.append(txt_path)
        last_text = referat_text

        console_html += log_success(f"{out_name}: реферат сохранён")
        yield f"Готово: {out_name}", last_text, None, render_console(console_html)

    zip_path = os.path.join(output_dir, "referat_results.zip")
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for fp in output_files:
            zipf.write(fp, os.path.basename(fp))

    console_html += log_success(f"Архив собран: {len(output_files)} файлов")
    yield "✅ Рефераты готовы! Скачайте архив.", last_text, zip_path, render_console(console_html)

# --- Данные экосистемы ---
AGENTS_DATA = [
    {"cluster": "1. Создание библиографических записей", "agents": [
        {"name": "Библио-Граф", "desc": "Извлечение метаданных и формирование бибописаний.", "active": True, "page": "biblio"},
        {"name": "Мета-Мастер", "desc": "Парсинг и нормализация метаданных из разнородных источников."},
        {"name": "Реестр-Бот", "desc": "Ведение реестра оцифрованных изданий с авто-идентификаторами."},
        {"name": "Рукопись-ИИ", "desc": "Перевод имидж-каталогов в электронный вид."}
    ]},
    {"cluster": "2. Создание реферативных статей", "agents": [
        {"name": "Реферат-Про", "desc": "Генерация кратких и развёрнутых аннотаций.", "active": True, "page": "referat"},
        {"name": "Саммари-Бот", "desc": "Создание структурированных саммари статей и монографий."},
        {"name": "Абстракт-Генератор", "desc": "Формирование аннотаций на рус. и англ. языках."},
        {"name": "Резюме-Мастер", "desc": "Извлечение основных положений и выводов."}
    ]},
    {"cluster": "3. Формирование коллекций", "agents": [
        {"name": "Коллектор-ИИ", "desc": "Автоматическая тематическая кластеризация."},
        {"name": "Тема-Селектор", "desc": "Группировка по предметным рубрикам и дисциплинам."},
        {"name": "Хроно-Коллектор", "desc": "Формирование коллекций по временным периодам."},
        {"name": "Автор-Куратор", "desc": "Сбор и организация изданий по авторам."},
        {"name": "Типо-Классификатор", "desc": "Разделение по типам документов (книги, статьи)."}
    ]},
    {"cluster": "4. Подбор изданий по каталогу", "agents": [
        {"name": "Каталог-Навигатор", "desc": "Интеллектуальный подбор по параметрам каталога."},
        {"name": "Фильтр-Поиск", "desc": "Многокритериальный отбор по полям."},
        {"name": "Рубрикатор-ИИ", "desc": "Навигация по предметным рубрикам."},
        {"name": "Библио-Селектор", "desc": "Персонализированный подбор под запрос."},
        {"name": "Мета-Поисковик", "desc": "Агрегация результатов из разных разделов."}
    ]},
    {"cluster": "5. Семантический поиск", "agents": [
        {"name": "Семантик-Поиск", "desc": "Поиск по смыслу, а не по ключевым словам."},
        {"name": "Вектор-Навигатор", "desc": "Использование векторных представлений."},
        {"name": "Контекст-ИИ", "desc": "Учёт контекста запроса для расширения результатов."},
        {"name": "Подобие-Бот", "desc": "Выявление документов со схожей тематикой."},
        {"name": "Нейро-Поиск", "desc": "Глубокое семантическое сопоставление."}
    ]},
    {"cluster": "6. Аудит электронного каталога", "agents": [
        {"name": "Аудит-Каталог", "desc": "Проверка полноты и корректности записей."},
        {"name": "Валидатор-Мета", "desc": "Выявление пропущенных полей и ошибок."},
        {"name": "Контроль-Качества", "desc": "Мониторинг качества и выявление дубликатов."},
        {"name": "Ревизор-ИИ", "desc": "Автоматическая сверка с эталонными источниками."},
        {"name": "Комплект-Бот", "desc": "Оценка заполненности обязательных полей."}
    ]},
    {"cluster": "7. Формирование базы знаний", "agents": [
        {"name": "Наука-БД", "desc": "Построение базы знаний с аннотациями."},
        {"name": "Знание-Граф", "desc": "Создание семантической сети знаний."},
        {"name": "Реф-Хранилище", "desc": "Организация ссылок на полнотекстовые файлы."},
        {"name": "Аннот-Наука", "desc": "Агрегация коротких аннотаций и метаданных."},
        {"name": "Библио-Знание", "desc": "Интеграция данных и полнотекстовых ресурсов."}
    ]},
    {"cluster": "8. Обработка запроса на естественном языке", "agents": [
        {"name": "Диалог-Библиотекарь", "desc": "Понимание запросов и выдача результатов."},
        {"name": "Запрос-Интерпретатор", "desc": "Семантический разбор запроса в критерии."},
        {"name": "НЛП-Поисковик", "desc": "Использование NLP для извлечения сущностей."},
        {"name": "Голос-Каталог", "desc": "Обработка голосовых и текстовых запросов."},
        {"name": "Семант-Ассистент", "desc": "Интеллектуальный помощник для навигации."}
    ]},
    {"cluster": "9. Проверка реестров запрещенной литературы", "agents": [
        {"name": "Инквизитор-ИИ", "desc": "Проверка изданий по реестрам запрещенной литературы."},
        {"name": "Следователь-ИИ", "desc": "Поиск на выявление запрещенной информации."}
    ]}
]

def render_cluster_html(cluster):
    html = f"<div class='cluster-block'><h3>{cluster['cluster']}</h3><div class='agents-grid'>"
    for agent in cluster['agents']:
        if agent.get('active'):
            badge = "<span class='badge-active'>Активен</span>"
        else:
            badge = "<span class='badge-beta'>В разработке</span>"
        html += f"""
            <div class='agent-card'>
                <h4>{agent['name']}</h4>
                <p>{agent['desc']}</p>
                <div class='card-footer'>{badge}</div>
            </div>
        """
    html += "</div></div>"
    return html

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=PT+Sans&family=PT+Serif&family=JetBrains+Mono:wght@400;700&display=swap');
body { background-color: #f0f2f5; }
.gradio-container {
    max-width: 1300px !important; margin: 20px auto !important;
    background-color: #ffffff !important; border-top: 6px solid #2a4d6e;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1) !important; padding: 0 !important;
    font-family: 'PT Sans', 'Arial', sans-serif !important; border-radius: 0 !important;
}
h1, h2, h3 { font-family: 'PT Serif', 'Georgia', serif !important; color: #2a4d6e !important; }

.sidebar { background: #f8f9fa; border-right: 2px solid #e0e0e0; padding: 30px 20px !important; min-height: 600px; }
.sidebar-title { font-family: 'PT Serif', serif; color: #2a4d6e !important; font-weight: bold; font-size: 18px; text-align: center; margin-bottom: 5px; }
.sidebar-subtitle { font-size: 11px; color: #888 !important; text-align: center; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 25px; }
.sidebar-sep { border-top: 1px solid #e0e0e0; margin: 20px 0; }
.sidebar-note { font-size: 11px; color: #999 !important; text-align: center; margin-top: 15px; line-height: 1.5; }

.nav-btn { width: 100% !important; justify-content: flex-start !important; text-align: left !important; margin-bottom: 10px !important; font-size: 15px !important; padding: 12px 16px !important; }
.nav-btn-idle { background: transparent !important; color: #2a4d6e !important; border: 1px solid #d0d7de !important; }
.nav-btn-idle:hover { border-color: #2a4d6e !important; background: #eef2f6 !important; }
.nav-btn-disabled { background: transparent !important; color: #aab3ba !important; border: 1px dashed #d8dde2 !important; cursor: not-allowed !important; font-size: 13px !important; padding: 8px 16px !important; margin-bottom: 6px !important; }

.main-area { padding: 40px !important; }

.big-card { white-space: pre-line !important; text-align: left !important; justify-content: flex-start !important; padding: 25px 22px !important; height: 110px !important; border: 1px solid #e0e0e0 !important; border-radius: 4px !important; background: #fbfcfd !important; font-size: 16px !important; color: #2a4d6e !important; transition: all 0.25s ease; }
.big-card:hover { transform: translateY(-3px); box-shadow: 0 6px 18px rgba(42,77,110,0.18); border-color: #2a4d6e !important; }

.cta-btn { max-width: 440px !important; margin: 0 auto 30px auto !important; display: flex !important; justify-content: center !important; }

.open-agent-btn { background: #eef2f6 !important; border: 1px solid #2a4d6e !important; color: #2a4d6e !important; font-weight: bold !important; }
.open-agent-btn:hover { background: #2a4d6e !important; color: white !important; }

.hero-section { text-align: center; padding: 40px 20px; background: #f8f9fa; border: 1px solid #e0e0e0; margin-bottom: 30px; }
.hero-section h1 { font-size: 34px !important; margin-bottom: 10px !important; }
.hero-section p { font-size: 17px; color: #555; max-width: 820px; margin: 0 auto 15px auto; line-height: 1.6; text-align: justify; text-align-last: center; }
.badge { display: inline-block; background: #2a4d6e; color: white; padding: 5px 15px; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 20px; }
.local-badge { display: inline-block; background: #d4edda; color: #155724; padding: 6px 18px; font-size: 13px; font-weight: bold; border-radius: 20px; }

.cluster-block { margin-bottom: 30px; }
.cluster-block h3 { border-bottom: 2px solid #2a4d6e; padding-bottom: 10px; margin-bottom: 20px; }
.agents-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 20px; }
.agent-card { border: 1px solid #e0e0e0; padding: 20px; background: #fff; border-radius: 4px; display: flex; flex-direction: column; justify-content: space-between; transition: all 0.3s; }
.agent-card:hover { transform: translateY(-3px); box-shadow: 0 5px 15px rgba(42,77,110,0.15); border-color: #2a4d6e; }
.agent-card h4 { margin-top: 0; color: #2a4d6e; font-size: 18px; font-family: 'PT Serif', serif; }
.agent-card p { font-size: 14px; color: #666; line-height: 1.5; margin-bottom: 15px; }
.card-footer { margin-top: auto; }
.badge-active { background: #d4edda; color: #155724; padding: 4px 10px; font-size: 12px; font-weight: bold; border-radius: 4px; }
.badge-beta { background: #fff3cd; color: #856404; padding: 4px 10px; font-size: 12px; font-weight: bold; border-radius: 4px; }

.telemetry-console { background-color: #1e1e2e; border: 1px solid #11111b; border-radius: 4px; padding: 20px; height: 280px; overflow-y: auto; font-family: 'JetBrains Mono', monospace; font-size: 13px; line-height: 1.6; color: #cdd6f4; box-shadow: inset 0 0 20px rgba(0,0,0,0.5); }
.telemetry-console::-webkit-scrollbar { width: 8px; }
.telemetry-console::-webkit-scrollbar-track { background: #1e1e2e; }
.telemetry-console::-webkit-scrollbar-thumb { background: #45475a; border-radius: 4px; }
.log-line { color: #a6adc8; }
.log-time { color: #585b70; margin-right: 10px; }
.log-tag { color: #89b4fa; margin-right: 10px; font-weight: bold; }
.log-success { color: #a6e3a1; font-weight: bold; margin-top: 5px; margin-bottom: 10px; border-bottom: 1px dashed #313244; padding-bottom: 5px; }
.log-error { color: #f38ba8; font-weight: bold; }

.gr-button-primary { background-color: #2a4d6e !important; border: none !important; border-radius: 4px !important; color: white !important; font-weight: bold !important; }
.gr-button-secondary { background-color: #ffffff !important; border: 2px solid #2a4d6e !important; border-radius: 4px !important; color: #2a4d6e !important; font-weight: bold !important; }
label { color: #4c5a62 !important; font-weight: bold !important; }
"""

# --- Навигация ---
NAV_KEYS = ["home", "clusters", "biblio", "referat"]

with gr.Blocks(css=custom_css) as demo:
    with gr.Row():
        # ===== БОКОВАЯ ПАНЕЛЬ =====
        with gr.Column(scale=1, min_width=230, elem_classes=["sidebar"]):
            gr.HTML("<div class='sidebar-title'>Платформа<br>ИИ-Агентов</div><div class='sidebar-subtitle'>БЕН РАН</div>")

            nav_home = gr.Button("🏠 Главная", variant="primary", elem_classes=["nav-btn"])
            nav_clusters = gr.Button("🗂 Кластеры агентов", variant="secondary", elem_classes=["nav-btn", "nav-btn-idle"])
            nav_biblio = gr.Button("📄 Библио-Граф", variant="secondary", elem_classes=["nav-btn", "nav-btn-idle"])
            nav_referat = gr.Button("📝 Реферат-Про", variant="secondary", elem_classes=["nav-btn", "nav-btn-idle"])

            gr.HTML("<div class='sidebar-sep'></div><div class='sidebar-subtitle' style='margin-bottom:10px;'>В разработке</div>")
            for name in ["Мета-Мастер", "Реестр-Бот", "Рукопись-ИИ"]:
                gr.Button(f"• {name}", interactive=False, elem_classes=["nav-btn-disabled"])
            gr.HTML("<div class='sidebar-note'>…и ещё 28 агентов.<br>Полный перечень — в разделе<br>«Кластеры агентов».</div>")

        # ===== ОСНОВНАЯ ОБЛАСТЬ =====
        with gr.Column(scale=4, elem_classes=["main-area"]):

            # --- Страница: Главная ---
            with gr.Column(visible=True) as page_home:
                gr.HTML("""
                <div class="hero-section">
                    <div class="badge">Платформа ИИ-Агентов БЕН РАН</div>
                    <h1>Нейросетевой агрегатор библиотечной деятельности</h1>
                    <p>Единая интеллектуальная платформа БЕН РАН, объединяющая специализированные ИИ-агенты для автоматизации ключевых процессов современной научной библиотеки. Агрегатор представляет собой модульную экосистему, в которой каждый агент решает узкоспециализированную задачу: от создания библиографических записей и генерации аннотаций до семантического поиска, аудита каталога и формирования тематических коллекций.</p>
                    <p>В основе лежит многокомпонентная нейросетевая модель, построенная на архитектуре трансформеров с поддержкой многоязычной обработки естественного языка (NLP), векторных представлений документов, механизмов извлечения структурированных метаданных и глубокого семантического анализа.</p>
                    <div class="local-badge">🔒 100% локально — данные не покидают ваш компьютер</div>
                </div>
                """)
                home_cta = gr.Button("Перейти к экосистеме ИИ-агентов →", variant="primary", elem_classes=["cta-btn"])
                with gr.Row():
                    card_biblio = gr.Button("📄  Библио-Граф  · активен\nБиблиографические записи из PDF (TXT + BibTeX)", elem_classes=["big-card"])
                    card_referat = gr.Button("📝  Реферат-Про  · активен\nСтруктурированные рефераты научных статей", elem_classes=["big-card"])
                gr.HTML("<div class='cluster-block' style='margin-top:40px;'><h3>⚙️ Технологии</h3></div>")
                with gr.Row():
                    gr.HTML("""
                    <div class='agents-grid' style='width:100%;'>
                        <div class='agent-card'><h4>Backend Core</h4><p>Python 3.10+ · Gradio UI<br>pypdf · OpenAI SDK (Ollama)</p></div>
                        <div class='agent-card'><h4>AI Engine</h4><p>Ollama · локальный инференс LLaMA 3<br>JSON output · Temperature 0.1</p></div>
                        <div class='agent-card'><h4>Конвейер</h4><p>PDF Parsing → LLM Extraction<br>→ Validation → TXT / BibTeX / ZIP</p></div>
                    </div>
                    """)

            # --- Страница: Кластеры ---
            with gr.Column(visible=False) as page_clusters:
                gr.Markdown("## 🤖 Перечень нейросетевых кластеров")
                gr.Markdown("Платформа включает 9 ключевых кластеров, объединяющих более 30 специализированных ИИ-агентов для автоматизации библиотечной деятельности БЕН РАН.")
                cluster_nav_buttons = []
                for cluster in AGENTS_DATA:
                    gr.HTML(render_cluster_html(cluster))
                    for a in cluster["agents"]:
                        if a.get("active"):
                            btn = gr.Button(f"▶  Открыть агента {a['name']}", elem_classes=["open-agent-btn"])
                            cluster_nav_buttons.append((a["page"], btn))

            # --- Страница: Библио-Граф ---
            with gr.Column(visible=False) as page_biblio:
                gr.Markdown("## 📄 Библио-Граф — пакетная обработка научных статей")
                gr.Markdown("Загрузите один или несколько PDF-файлов. Агент извлечёт библиографические данные и упакует их в **TXT** и **BibTeX**.")
                with gr.Row():
                    pdf_input = gr.Files(label="Исходные файлы (PDF)", file_count="multiple", file_types=[".pdf"])
                with gr.Row():
                    run_btn = gr.Button("🧠 Запустить ИИ-Агента", variant="primary")
                    demo_btn = gr.Button("⚙️ Демо-режим (Витрина)", variant="secondary")
                gr.Markdown("### Журнал обработки")
                biblio_console = gr.HTML(render_console("<div class='log-line'>> Ожидание инициализации системы...</div>"))
                with gr.Row():
                    with gr.Column(scale=1):
                        biblio_status = gr.Textbox(label="Статус выполнения", interactive=False, placeholder="Ожидание...")
                    with gr.Column(scale=1):
                        biblio_file = gr.File(label="Готовые файлы (ZIP)", interactive=False)
                run_btn.click(fn=process_pdfs, inputs=[pdf_input, gr.State(False)], outputs=[biblio_status, biblio_file, biblio_console])
                demo_btn.click(fn=process_pdfs, inputs=[pdf_input, gr.State(True)], outputs=[biblio_status, biblio_file, biblio_console])

            # --- Страница: Реферат-Про ---
            with gr.Column(visible=False) as page_referat:
                gr.Markdown("## 📝 Реферат-Про — генерация рефератов статей")
                gr.Markdown("Загрузите PDF научной статьи. Агент сформирует структурированный реферат: библиографическая запись, аннотация, цель, методы, результаты, выводы.")
                with gr.Row():
                    ref_pdf_input = gr.Files(label="Исходные файлы (PDF)", file_count="multiple", file_types=[".pdf"])
                with gr.Row():
                    ref_run_btn = gr.Button("🧠 Сгенерировать рефераты", variant="primary")
                    ref_demo_btn = gr.Button("⚙️ Демо-режим (Витрина)", variant="secondary")
                gr.Markdown("### Журнал обработки")
                ref_console = gr.HTML(render_console("<div class='log-line'>> Ожидание инициализации системы...</div>"))
                gr.Markdown("### Предпросмотр реферата")
                ref_preview = gr.Textbox(label="Содержание", lines=14, interactive=False, placeholder="Здесь появится текст реферата...")
                with gr.Row():
                    with gr.Column(scale=1):
                        ref_status = gr.Textbox(label="Статус выполнения", interactive=False, placeholder="Ожидание...")
                    with gr.Column(scale=1):
                        ref_file = gr.File(label="Готовые файлы (ZIP)", interactive=False)
                ref_run_btn.click(fn=process_referats, inputs=[ref_pdf_input, gr.State(False)], outputs=[ref_status, ref_preview, ref_file, ref_console])
                ref_demo_btn.click(fn=process_referats, inputs=[ref_pdf_input, gr.State(True)], outputs=[ref_status, ref_preview, ref_file, ref_console])


    # --- Логика навигации (внутри Blocks-контекста!) ---
    NAV_BUTTONS = [("home", nav_home), ("clusters", nav_clusters), ("biblio", nav_biblio), ("referat", nav_referat)]
    NAV_PAGES = [("home", page_home), ("clusters", page_clusters), ("biblio", page_biblio), ("referat", page_referat)]

    def navigate(target):
        def _nav():
            btn_updates = [gr.update(variant="primary" if k == target else "secondary") for k, _ in NAV_BUTTONS]
            page_updates = [gr.update(visible=(k == target)) for k, _ in NAV_PAGES]
            return btn_updates + page_updates
        return _nav

    NAV_OUTPUTS = [b for _, b in NAV_BUTTONS] + [p for _, p in NAV_PAGES]

    nav_home.click(navigate("home"), inputs=None, outputs=NAV_OUTPUTS)
    nav_clusters.click(navigate("clusters"), inputs=None, outputs=NAV_OUTPUTS)
    nav_biblio.click(navigate("biblio"), inputs=None, outputs=NAV_OUTPUTS)
    nav_referat.click(navigate("referat"), inputs=None, outputs=NAV_OUTPUTS)
    home_cta.click(navigate("clusters"), inputs=None, outputs=NAV_OUTPUTS)
    card_biblio.click(navigate("biblio"), inputs=None, outputs=NAV_OUTPUTS)
    card_referat.click(navigate("referat"), inputs=None, outputs=NAV_OUTPUTS)
    for target, btn in cluster_nav_buttons:
        btn.click(navigate(target), inputs=None, outputs=NAV_OUTPUTS)

if __name__ == "__main__":
    demo.launch(inbrowser=True)
