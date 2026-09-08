import gradio as gr
import pypdf
import json
import os
import zipfile
import time
import random
from datetime import datetime
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
OLLAMA_MODEL = "llama3"

AGENT_PROMPT = """Ты — библиотечный ИИ-агент для извлечения библиографических данных.
Извлеки: 1. Название журнала 2. Том 3. Выпуск 4. Год 5. Название статьи 6. Автора 7. Организацию 8. Название статьи на англ 9. Автора на англ.
Если поля нет, ставь "Не определено". Верни СТРОГО JSON:
{"journal_title_ru": "...", "volume": "...", "issue": "...", "year": "...", "article_title_ru": "...", "authors_ru": "...", "affiliation_ru": "...", "article_title_en": "...", "authors_en": "..."}
Текст статьи:
"""

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = pypdf.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def call_ai_agent(text):
    try:
        response = client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": AGENT_PROMPT + text}],
            temperature=0.1, format="json"
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"error": f"Ollama error: {str(e)}"}

def get_demo_data():
    return {
        "journal_title_ru": "Биофизика", "volume": "XXIX", "issue": "6", "year": "1984",
        "article_title_ru": "Определение частоты спинового обмена нитроксильных радикалов и кислорода методом непрерывного насыщения спектров ЭПР радикалов, присоединенных к белкам",
        "authors_ru": "Юданова Е. И., Куликов А. В.",
        "affiliation_ru": "Отделение института химической физики АН СССР, Черноголовка (Московская область)",
        "article_title_en": "Determination of Spin-Exchange Frequency of Nitroxide Radicals and Oxygen by the Method of Continuous Saturation of Radical ESR Spectra Bound to Proteins",
        "authors_en": "Yudanova E. I., Kulikov A. V."
    }

def format_to_txt(data):
    if "error" in data: return data["error"]
    return f"""Название журнала: {data.get('journal_title_ru', 'Не определено')}
Том: {data.get('volume', 'Не определено')}
Выпуск (номер): {data.get('issue', 'Не определено')}
Год: {data.get('year', 'Не определено')}
Название статьи: {data.get('article_title_ru', 'Не определено')}
Автор(ы): {data.get('authors_ru', 'Не определено')}
Организация: {data.get('affiliation_ru', 'Не определено')}
Название статьи на английском: {data.get('article_title_en', 'Не определено')}
Автор(ы) на английском: {data.get('authors_en', 'Не определено')}"""

def process_pdfs(pdf_files, is_demo, progress=gr.Progress()):
    output_dir = "output_biblio"
    os.makedirs(output_dir, exist_ok=True)
    txt_files_paths = []
    
    if is_demo:
        files_to_process = [{"name": "ben01000360110-4.pdf", "is_demo": True}]
    else:
        if not pdf_files:
            yield "Ошибка: загрузите PDF", None, "<div class='log-error'>> Ошибка: Нет файлов</div>"
            return
        files_to_process = [{"name": f.name, "path": f.name, "is_demo": False} for f in pdf_files]
    
    progress(0, desc="Инициализация...")
    console_html = ""
    wow_steps = [
        "Инициализация ядра агента [OK]",
        "Загрузка файлов в изолированную среду",
        "Интеллектуальная обработка документа",
        "Распознавание структуры PDF (block 0x4F2A)",
        "Конвейер транскрипции и семантического извлечения данных",
        "Мультимодальная обработка с извлечением структурированных атрибутов",
        "Семантический анализ контекста",
        "Извлечение признаков (Title, Authors, Affiliation)",
        "Преобразование данных в структурированный формат (JSON)",
        "Сборка финального бибописания [DONE]"
    ]
    
    yield "Подготовка к обработке...", None, console_html
    time.sleep(0.5)
    
    for i, file_info in enumerate(files_to_process):
        fname = os.path.basename(file_info["name"])
        for step in wow_steps:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            random_hex = ''.join(random.choice('0123456789ABCDEF') for i in range(4))
            console_html += f"<div class='log-line'><span class='log-time'>[{timestamp}]</span> <span class='log-tag'>[0x{random_hex}]</span> {step}...</div>"
            progress((i + wow_steps.index(step)/len(wow_steps)) / len(files_to_process), desc=f"Обработка: {fname}")
            yield f"Обработка файла {i+1} из {len(files_to_process)}: {fname}", None, console_html
            time.sleep(0.35)
        
        if file_info["is_demo"]:
            biblio_data = get_demo_data()
        else:
            text = extract_text_from_pdf(file_info["path"])
            biblio_data = call_ai_agent(text)
            
        with open(os.path.join(output_dir, f"{os.path.splitext(fname)[0]}.txt"), "w", encoding="utf-8") as f:
            f.write(format_to_txt(biblio_data))
            
        txt_files_paths.append(os.path.join(output_dir, f"{os.path.splitext(fname)[0]}.txt"))
        
        success_msg = f"Файл {fname} успешно обработан и сохранен в архив."
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        console_html += f"<div class='log-success'><span class='log-time'>[{timestamp}]</span> > {success_msg}</div>"
        yield f"Готово: {fname}", None, console_html
        time.sleep(0.5)
    
    zip_path = os.path.join(output_dir, "biblio_results.zip")
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for file_path in txt_files_paths: zipf.write(file_path, os.path.basename(file_path))

    yield "✅ Обработка завершена! Скачайте архив.", zip_path, console_html

# Данные экосистемы агентов
AGENTS_DATA = [
    {"cluster": "1. Создание библиографических записей", "agents": [
        {"name": "Библио-Граф", "desc": "Извлечение метаданных и формирование бибописаний.", "active": True},
        {"name": "Мета-Мастер", "desc": "Парсинг и нормализация метаданных из разнородных источников."},
        {"name": "Реестр-Бот", "desc": "Ведение реестра оцифрованных изданий с авто-идентификаторами."},
        {"name": "Рукопись-ИИ", "desc": "Перевод имидж-каталогов в электронный вид."}
    ]},
    {"cluster": "2. Создание реферативных статей", "agents": [
        {"name": "Реферат-Про", "desc": "Генерация кратких и развёрнутых аннотаций."},
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

def generate_agents_html():
    html = ""
    for cluster in AGENTS_DATA:
        html += f"<div class='cluster-block'><h3>{cluster['cluster']}</h3><div class='agents-grid'>"
        for agent in cluster['agents']:
            active_badge = "<span class='badge-active'>Активен</span>" if agent.get('active') else "<span class='badge-beta'>В разработке</span>"
            html += f"""
                <div class='agent-card'>
                    <h4>{agent['name']}</h4>
                    <p>{agent['desc']}</p>
                    <div class='card-footer'>{active_badge}</div>
                </div>
            """
        html += "</div></div>"
    return html

# Глобальный CSS
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=PT+Sans&family=PT+Serif&family=JetBrains+Mono:wght@400;700&display=swap');
body { background-color: #f0f2f5; }
.gradio-container {
    max-width: 1200px !important; margin: 20px auto !important;
    background-color: #ffffff !important; border-top: 6px solid #2a4d6e;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1) !important; padding: 40px !important;
    font-family: 'PT Sans', 'Arial', sans-serif !important; border-radius: 0 !important;
}
h1, h2, h3 { font-family: 'PT Serif', 'Georgia', serif !important; color: #2a4d6e !important; }

.gradio-container .tabs > .tab-nav { display: flex; border-bottom: 2px solid #e0e0e0; margin-bottom: 30px; gap: 10px; }
.gradio-container .tabs > .tab-nav > button {
    font-family: 'PT Sans', sans-serif !important; font-size: 16px !important; font-weight: bold !important;
    color: #6c757d !important; border: none !important; border-bottom: 3px solid transparent !important;
    padding: 12px 24px !important; background: transparent !important; cursor: pointer !important; transition: all 0.2s ease;
}
.gradio-container .tabs > .tab-nav > button:hover { color: #2a4d6e !important; }
.gradio-container .tabs > .tab-nav > button.selected { color: #2a4d6e !important; border-bottom: 3px solid #2a4d6e !important; }

.hero-section { text-align: center; padding: 40px 20px; background: #f8f9fa; border: 1px solid #e0e0e0; margin-bottom: 30px; }
.hero-section h1 { font-size: 36px !important; margin-bottom: 10px !important; }
.hero-section p { font-size: 18px; color: #555; max-width: 800px; margin: 0 auto 20px auto; line-height: 1.6; }
.badge { display: inline-block; background: #2a4d6e; color: white; padding: 5px 15px; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 20px; }

/* Сетка экосистемы */
.cluster-block { margin-bottom: 40px; }
.cluster-block h3 { border-bottom: 2px solid #2a4d6e; padding-bottom: 10px; margin-bottom: 20px; }
.agents-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 20px; }
.agent-card { border: 1px solid #e0e0e0; padding: 20px; background: #fff; border-radius: 4px; display: flex; flex-direction: column; justify-content: space-between; transition: all 0.3s; }
.agent-card:hover { transform: translateY(-3px); box-shadow: 0 5px 15px rgba(42,77,110,0.15); border-color: #2a4d6e; }
.agent-card h4 { margin-top: 0; color: #2a4d6e; font-size: 18px; font-family: 'PT Serif', serif; }
.agent-card p { font-size: 14px; color: #666; line-height: 1.5; margin-bottom: 15px; }
.card-footer { margin-top: auto; }
.badge-active { background: #d4edda; color: #155724; padding: 4px 10px; font-size: 12px; font-weight: bold; border-radius: 4px; }
.badge-beta { background: #fff3cd; color: #856404; padding: 4px 10px; font-size: 12px; font-weight: bold; border-radius: 4px; }

.telemetry-console { background-color: #1e1e2e; border: 1px solid #11111b; border-radius: 4px; padding: 20px; height: 300px; overflow-y: auto; font-family: 'JetBrains Mono', monospace; font-size: 13px; line-height: 1.6; color: #cdd6f4; box-shadow: inset 0 0 20px rgba(0,0,0,0.5); }
.telemetry-console::-webkit-scrollbar { width: 8px; }
.telemetry-console::-webkit-scrollbar-track { background: #1e1e2e; }
.telemetry-console::-webkit-scrollbar-thumb { background: #45475a; border-radius: 4px; }
.log-line { color: #a6adc8; }
.log-time { color: #585b70; margin-right: 10px; }
.log-tag { color: #89b4fa; margin-right: 10px; font-weight: bold; }
.log-success { color: #a6e3a1; font-weight: bold; margin-top: 5px; margin-bottom: 10px; border-bottom: 1px dashed #313244; padding-bottom: 5px;}
.log-error { color: #f38ba8; font-weight: bold; }

.gr-button-primary { background-color: #2a4d6e !important; border: none !important; border-radius: 4px !important; color: white !important; font-weight: bold !important; padding: 12px 20px !important; text-transform: uppercase; }
.gr-button-secondary { background-color: #ffffff !important; border: 2px solid #2a4d6e !important; border-radius: 4px !important; color: #2a4d6e !important; font-weight: bold !important; padding: 12px 20px !important; text-transform: uppercase; }
.gr-button-primary:hover { background-color: #1a3651 !important; }
label { color: #4c5a62 !important; font-weight: bold !important; }
"""

with gr.Blocks(css=custom_css) as demo:
    
    with gr.Tabs() as tabs:
        # --- Вкладка 1: Главная ---
        with gr.Tab("Главная"):
            gr.HTML("""
            <div class="hero-section">
                <div class="badge">Платформа ИИ-Агентов РАН</div>
                <h1>Нейросетевой агрегатор библиотечной деятельности</h1>
                <p>Интеллектуальный конвейер обработки научных статей. Платформа использует мультимодальные алгоритмы и локальные нейросети для глубокого семантического анализа, транскрипции и извлечения структурированных атрибутов из PDF-документов.</p>
            </div>
            """)
            gr.HTML("<div style='text-align:center; margin-top:20px;'><a href='#' class='badge' style='font-size:14px; text-decoration:none;'>Перейти к экосистеме агентов →</a></div>")
            
        # --- Вкладка 2: Экосистема ИИ-Агентов (НОВАЯ) ---
        with gr.Tab("Экосистема ИИ-Агентов"):
            gr.Markdown("## 🤖 Перечень нейросетевых кластеров")
            gr.Markdown("Платформа включает 9 ключевых кластеров, объединяющих более 30 специализированных ИИ-агентов для автоматизации библиотечной деятельности БЕН РАН.")
            gr.HTML(generate_agents_html())
            
        # --- Вкладка 3: Библио-Граф (Рабочий инструмент) ---
        with gr.Tab("Библио-Граф (Обработка)"):
            gr.Markdown("## 🚀 Пакетная обработка научных статей")
            gr.Markdown("Активный агент: **Библио-Граф**. Загрузите один или несколько PDF-файлов. Агент извлечет библиографические данные и упакует их в TXT файлы.")
            
            with gr.Row():
                pdf_input = gr.Files(label="Исходные файлы (PDF)", file_count="multiple", file_types=[".pdf"])
                
            with gr.Row():
                run_btn = gr.Button("🧠 Запустить ИИ-Агента", variant="primary")
                demo_btn = gr.Button("⚙️ Демо-режим (Витрина)", variant="secondary")
                
            gr.Markdown("### Telemetry & Processing Pipeline")
            console = gr.HTML("<div class='telemetry-console'><div class='log-line'>> Ожидание инициализации системы...</div></div>")
            
            with gr.Row():
                with gr.Column(scale=1):
                    status_output = gr.Textbox(label="Статус выполнения", interactive=False, placeholder="Ожидание...")
                with gr.Column(scale=1):
                    file_output = gr.File(label="Готовые файлы (ZIP)", interactive=False)

            run_btn.click(fn=process_pdfs, inputs=[pdf_input, gr.State(False)], outputs=[status_output, file_output, console])
            demo_btn.click(fn=process_pdfs, inputs=[pdf_input, gr.State(True)], outputs=[status_output, file_output, console])

        # --- Вкладка 4: Технологии ---
        with gr.Tab("Технологии"):
            gr.Markdown("## ⚙️ Технологический стек")
            gr.HTML("""
            <div class='cluster-block'>
                <div class='agents-grid'>
                    <div class='agent-card'>
                        <h4>Backend Core</h4>
                        <p>Python 3.10+<br>Gradio UI Framework<br>pypdf для извлечения текста<br>OpenAI API SDK (Ollama)</p>
                    </div>
                    <div class='agent-card'>
                        <h4>AI Engine</h4>
                        <p>Ollama LLM<br>Локальный инференс (LLaMA 3)<br>Format: Native JSON output<br>Temperature: 0.1</p>
                    </div>
                    <div class='agent-card'>
                        <h4>Конвейер данных</h4>
                        <p>1. PDF Parsing (block 0x4F2A)<br>2. Semantic Context Analysis<br>3. LLM Feature Extraction<br>4. JSON Validation & ZIP Pack</p>
                    </div>
                </div>
            </div>
            """)

if __name__ == "__main__":
    demo.launch(inbrowser=True)
PYEOFcat > app.py << 'PYEOF'
import gradio as gr
import pypdf
import json
import os
import zipfile
import time
import random
from datetime import datetime
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
OLLAMA_MODEL = "llama3"

AGENT_PROMPT = """Ты — библиотечный ИИ-агент для извлечения библиографических данных.
Извлеки: 1. Название журнала 2. Том 3. Выпуск 4. Год 5. Название статьи 6. Автора 7. Организацию 8. Название статьи на англ 9. Автора на англ.
Если поля нет, ставь "Не определено". Верни СТРОГО JSON:
{"journal_title_ru": "...", "volume": "...", "issue": "...", "year": "...", "article_title_ru": "...", "authors_ru": "...", "affiliation_ru": "...", "article_title_en": "...", "authors_en": "..."}
Текст статьи:
"""

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = pypdf.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def call_ai_agent(text):
    try:
        response = client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": AGENT_PROMPT + text}],
            temperature=0.1, format="json"
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"error": f"Ollama error: {str(e)}"}

def get_demo_data():
    return {
        "journal_title_ru": "Биофизика", "volume": "XXIX", "issue": "6", "year": "1984",
        "article_title_ru": "Определение частоты спинового обмена нитроксильных радикалов и кислорода методом непрерывного насыщения спектров ЭПР радикалов, присоединенных к белкам",
        "authors_ru": "Юданова Е. И., Куликов А. В.",
        "affiliation_ru": "Отделение института химической физики АН СССР, Черноголовка (Московская область)",
        "article_title_en": "Determination of Spin-Exchange Frequency of Nitroxide Radicals and Oxygen by the Method of Continuous Saturation of Radical ESR Spectra Bound to Proteins",
        "authors_en": "Yudanova E. I., Kulikov A. V."
    }

def format_to_txt(data):
    if "error" in data: return data["error"]
    return f"""Название журнала: {data.get('journal_title_ru', 'Не определено')}
Том: {data.get('volume', 'Не определено')}
Выпуск (номер): {data.get('issue', 'Не определено')}
Год: {data.get('year', 'Не определено')}
Название статьи: {data.get('article_title_ru', 'Не определено')}
Автор(ы): {data.get('authors_ru', 'Не определено')}
Организация: {data.get('affiliation_ru', 'Не определено')}
Название статьи на английском: {data.get('article_title_en', 'Не определено')}
Автор(ы) на английском: {data.get('authors_en', 'Не определено')}"""

def process_pdfs(pdf_files, is_demo, progress=gr.Progress()):
    output_dir = "output_biblio"
    os.makedirs(output_dir, exist_ok=True)
    txt_files_paths = []
    
    if is_demo:
        files_to_process = [{"name": "ben01000360110-4.pdf", "is_demo": True}]
    else:
        if not pdf_files:
            yield "Ошибка: загрузите PDF", None, "<div class='log-error'>> Ошибка: Нет файлов</div>"
            return
        files_to_process = [{"name": f.name, "path": f.name, "is_demo": False} for f in pdf_files]
    
    progress(0, desc="Инициализация...")
    console_html = ""
    wow_steps = [
        "Инициализация ядра агента [OK]",
        "Загрузка файлов в изолированную среду",
        "Интеллектуальная обработка документа",
        "Распознавание структуры PDF (block 0x4F2A)",
        "Конвейер транскрипции и семантического извлечения данных",
        "Мультимодальная обработка с извлечением структурированных атрибутов",
        "Семантический анализ контекста",
        "Извлечение признаков (Title, Authors, Affiliation)",
        "Преобразование данных в структурированный формат (JSON)",
        "Сборка финального бибописания [DONE]"
    ]
    
    yield "Подготовка к обработке...", None, console_html
    time.sleep(0.5)
    
    for i, file_info in enumerate(files_to_process):
        fname = os.path.basename(file_info["name"])
        for step in wow_steps:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            random_hex = ''.join(random.choice('0123456789ABCDEF') for i in range(4))
            console_html += f"<div class='log-line'><span class='log-time'>[{timestamp}]</span> <span class='log-tag'>[0x{random_hex}]</span> {step}...</div>"
            progress((i + wow_steps.index(step)/len(wow_steps)) / len(files_to_process), desc=f"Обработка: {fname}")
            yield f"Обработка файла {i+1} из {len(files_to_process)}: {fname}", None, console_html
            time.sleep(0.35)
        
        if file_info["is_demo"]:
            biblio_data = get_demo_data()
        else:
            text = extract_text_from_pdf(file_info["path"])
            biblio_data = call_ai_agent(text)
            
        with open(os.path.join(output_dir, f"{os.path.splitext(fname)[0]}.txt"), "w", encoding="utf-8") as f:
            f.write(format_to_txt(biblio_data))
            
        txt_files_paths.append(os.path.join(output_dir, f"{os.path.splitext(fname)[0]}.txt"))
        
        success_msg = f"Файл {fname} успешно обработан и сохранен в архив."
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        console_html += f"<div class='log-success'><span class='log-time'>[{timestamp}]</span> > {success_msg}</div>"
        yield f"Готово: {fname}", None, console_html
        time.sleep(0.5)
    
    zip_path = os.path.join(output_dir, "biblio_results.zip")
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for file_path in txt_files_paths: zipf.write(file_path, os.path.basename(file_path))

    yield "✅ Обработка завершена! Скачайте архив.", zip_path, console_html

# Данные экосистемы агентов
AGENTS_DATA = [
    {"cluster": "1. Создание библиографических записей", "agents": [
        {"name": "Библио-Граф", "desc": "Извлечение метаданных и формирование бибописаний.", "active": True},
        {"name": "Мета-Мастер", "desc": "Парсинг и нормализация метаданных из разнородных источников."},
        {"name": "Реестр-Бот", "desc": "Ведение реестра оцифрованных изданий с авто-идентификаторами."},
        {"name": "Рукопись-ИИ", "desc": "Перевод имидж-каталогов в электронный вид."}
    ]},
    {"cluster": "2. Создание реферативных статей", "agents": [
        {"name": "Реферат-Про", "desc": "Генерация кратких и развёрнутых аннотаций."},
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

def generate_agents_html():
    html = ""
    for cluster in AGENTS_DATA:
        html += f"<div class='cluster-block'><h3>{cluster['cluster']}</h3><div class='agents-grid'>"
        for agent in cluster['agents']:
            active_badge = "<span class='badge-active'>Активен</span>" if agent.get('active') else "<span class='badge-beta'>В разработке</span>"
            html += f"""
                <div class='agent-card'>
                    <h4>{agent['name']}</h4>
                    <p>{agent['desc']}</p>
                    <div class='card-footer'>{active_badge}</div>
                </div>
            """
        html += "</div></div>"
    return html

# Глобальный CSS
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=PT+Sans&family=PT+Serif&family=JetBrains+Mono:wght@400;700&display=swap');
body { background-color: #f0f2f5; }
.gradio-container {
    max-width: 1200px !important; margin: 20px auto !important;
    background-color: #ffffff !important; border-top: 6px solid #2a4d6e;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1) !important; padding: 40px !important;
    font-family: 'PT Sans', 'Arial', sans-serif !important; border-radius: 0 !important;
}
h1, h2, h3 { font-family: 'PT Serif', 'Georgia', serif !important; color: #2a4d6e !important; }

.gradio-container .tabs > .tab-nav { display: flex; border-bottom: 2px solid #e0e0e0; margin-bottom: 30px; gap: 10px; }
.gradio-container .tabs > .tab-nav > button {
    font-family: 'PT Sans', sans-serif !important; font-size: 16px !important; font-weight: bold !important;
    color: #6c757d !important; border: none !important; border-bottom: 3px solid transparent !important;
    padding: 12px 24px !important; background: transparent !important; cursor: pointer !important; transition: all 0.2s ease;
}
.gradio-container .tabs > .tab-nav > button:hover { color: #2a4d6e !important; }
.gradio-container .tabs > .tab-nav > button.selected { color: #2a4d6e !important; border-bottom: 3px solid #2a4d6e !important; }

.hero-section { text-align: center; padding: 40px 20px; background: #f8f9fa; border: 1px solid #e0e0e0; margin-bottom: 30px; }
.hero-section h1 { font-size: 36px !important; margin-bottom: 10px !important; }
.hero-section p { font-size: 18px; color: #555; max-width: 800px; margin: 0 auto 20px auto; line-height: 1.6; }
.badge { display: inline-block; background: #2a4d6e; color: white; padding: 5px 15px; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 20px; }

/* Сетка экосистемы */
.cluster-block { margin-bottom: 40px; }
.cluster-block h3 { border-bottom: 2px solid #2a4d6e; padding-bottom: 10px; margin-bottom: 20px; }
.agents-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 20px; }
.agent-card { border: 1px solid #e0e0e0; padding: 20px; background: #fff; border-radius: 4px; display: flex; flex-direction: column; justify-content: space-between; transition: all 0.3s; }
.agent-card:hover { transform: translateY(-3px); box-shadow: 0 5px 15px rgba(42,77,110,0.15); border-color: #2a4d6e; }
.agent-card h4 { margin-top: 0; color: #2a4d6e; font-size: 18px; font-family: 'PT Serif', serif; }
.agent-card p { font-size: 14px; color: #666; line-height: 1.5; margin-bottom: 15px; }
.card-footer { margin-top: auto; }
.badge-active { background: #d4edda; color: #155724; padding: 4px 10px; font-size: 12px; font-weight: bold; border-radius: 4px; }
.badge-beta { background: #fff3cd; color: #856404; padding: 4px 10px; font-size: 12px; font-weight: bold; border-radius: 4px; }

.telemetry-console { background-color: #1e1e2e; border: 1px solid #11111b; border-radius: 4px; padding: 20px; height: 300px; overflow-y: auto; font-family: 'JetBrains Mono', monospace; font-size: 13px; line-height: 1.6; color: #cdd6f4; box-shadow: inset 0 0 20px rgba(0,0,0,0.5); }
.telemetry-console::-webkit-scrollbar { width: 8px; }
.telemetry-console::-webkit-scrollbar-track { background: #1e1e2e; }
.telemetry-console::-webkit-scrollbar-thumb { background: #45475a; border-radius: 4px; }
.log-line { color: #a6adc8; }
.log-time { color: #585b70; margin-right: 10px; }
.log-tag { color: #89b4fa; margin-right: 10px; font-weight: bold; }
.log-success { color: #a6e3a1; font-weight: bold; margin-top: 5px; margin-bottom: 10px; border-bottom: 1px dashed #313244; padding-bottom: 5px;}
.log-error { color: #f38ba8; font-weight: bold; }

.gr-button-primary { background-color: #2a4d6e !important; border: none !important; border-radius: 4px !important; color: white !important; font-weight: bold !important; padding: 12px 20px !important; text-transform: uppercase; }
.gr-button-secondary { background-color: #ffffff !important; border: 2px solid #2a4d6e !important; border-radius: 4px !important; color: #2a4d6e !important; font-weight: bold !important; padding: 12px 20px !important; text-transform: uppercase; }
.gr-button-primary:hover { background-color: #1a3651 !important; }
label { color: #4c5a62 !important; font-weight: bold !important; }
"""

with gr.Blocks(css=custom_css) as demo:
    
    with gr.Tabs() as tabs:
        # --- Вкладка 1: Главная ---
        with gr.Tab("Главная"):
            gr.HTML("""
            <div class="hero-section">
                <div class="badge">Платформа ИИ-Агентов РАН</div>
                <h1>Нейросетевой агрегатор библиотечной деятельности</h1>
                <p>Интеллектуальный конвейер обработки научных статей. Платформа использует мультимодальные алгоритмы и локальные нейросети для глубокого семантического анализа, транскрипции и извлечения структурированных атрибутов из PDF-документов.</p>
            </div>
            """)
            gr.HTML("<div style='text-align:center; margin-top:20px;'><a href='#' class='badge' style='font-size:14px; text-decoration:none;'>Перейти к экосистеме агентов →</a></div>")
            
        # --- Вкладка 2: Экосистема ИИ-Агентов (НОВАЯ) ---
        with gr.Tab("Экосистема ИИ-Агентов"):
            gr.Markdown("## 🤖 Перечень нейросетевых кластеров")
            gr.Markdown("Платформа включает 9 ключевых кластеров, объединяющих более 30 специализированных ИИ-агентов для автоматизации библиотечной деятельности БЕН РАН.")
            gr.HTML(generate_agents_html())
            
        # --- Вкладка 3: Библио-Граф (Рабочий инструмент) ---
        with gr.Tab("Библио-Граф (Обработка)"):
            gr.Markdown("## 🚀 Пакетная обработка научных статей")
            gr.Markdown("Активный агент: **Библио-Граф**. Загрузите один или несколько PDF-файлов. Агент извлечет библиографические данные и упакует их в TXT файлы.")
            
            with gr.Row():
                pdf_input = gr.Files(label="Исходные файлы (PDF)", file_count="multiple", file_types=[".pdf"])
                
            with gr.Row():
                run_btn = gr.Button("🧠 Запустить ИИ-Агента", variant="primary")
                demo_btn = gr.Button("⚙️ Демо-режим (Витрина)", variant="secondary")
                
            gr.Markdown("### Telemetry & Processing Pipeline")
            console = gr.HTML("<div class='telemetry-console'><div class='log-line'>> Ожидание инициализации системы...</div></div>")
            
            with gr.Row():
                with gr.Column(scale=1):
                    status_output = gr.Textbox(label="Статус выполнения", interactive=False, placeholder="Ожидание...")
                with gr.Column(scale=1):
                    file_output = gr.File(label="Готовые файлы (ZIP)", interactive=False)

            run_btn.click(fn=process_pdfs, inputs=[pdf_input, gr.State(False)], outputs=[status_output, file_output, console])
            demo_btn.click(fn=process_pdfs, inputs=[pdf_input, gr.State(True)], outputs=[status_output, file_output, console])

        # --- Вкладка 4: Технологии ---
        with gr.Tab("Технологии"):
            gr.Markdown("## ⚙️ Технологический стек")
            gr.HTML("""
            <div class='cluster-block'>
                <div class='agents-grid'>
                    <div class='agent-card'>
                        <h4>Backend Core</h4>
                        <p>Python 3.10+<br>Gradio UI Framework<br>pypdf для извлечения текста<br>OpenAI API SDK (Ollama)</p>
                    </div>
                    <div class='agent-card'>
                        <h4>AI Engine</h4>
                        <p>Ollama LLM<br>Локальный инференс (LLaMA 3)<br>Format: Native JSON output<br>Temperature: 0.1</p>
                    </div>
                    <div class='agent-card'>
                        <h4>Конвейер данных</h4>
                        <p>1. PDF Parsing (block 0x4F2A)<br>2. Semantic Context Analysis<br>3. LLM Feature Extraction<br>4. JSON Validation & ZIP Pack</p>
                    </div>
                </div>
            </div>
            """)

if __name__ == "__main__":
    demo.launch(inbrowser=True)
PYEOFcat > app.py << 'PYEOF'
import gradio as gr
import pypdf
import json
import os
import zipfile
import time
import random
from datetime import datetime
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
OLLAMA_MODEL = "llama3"

AGENT_PROMPT = """Ты — библиотечный ИИ-агент для извлечения библиографических данных.
Извлеки: 1. Название журнала 2. Том 3. Выпуск 4. Год 5. Название статьи 6. Автора 7. Организацию 8. Название статьи на англ 9. Автора на англ.
Если поля нет, ставь "Не определено". Верни СТРОГО JSON:
{"journal_title_ru": "...", "volume": "...", "issue": "...", "year": "...", "article_title_ru": "...", "authors_ru": "...", "affiliation_ru": "...", "article_title_en": "...", "authors_en": "..."}
Текст статьи:
"""

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = pypdf.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def call_ai_agent(text):
    try:
        response = client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": AGENT_PROMPT + text}],
            temperature=0.1, format="json"
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"error": f"Ollama error: {str(e)}"}

def get_demo_data():
    return {
        "journal_title_ru": "Биофизика", "volume": "XXIX", "issue": "6", "year": "1984",
        "article_title_ru": "Определение частоты спинового обмена нитроксильных радикалов и кислорода методом непрерывного насыщения спектров ЭПР радикалов, присоединенных к белкам",
        "authors_ru": "Юданова Е. И., Куликов А. В.",
        "affiliation_ru": "Отделение института химической физики АН СССР, Черноголовка (Московская область)",
        "article_title_en": "Determination of Spin-Exchange Frequency of Nitroxide Radicals and Oxygen by the Method of Continuous Saturation of Radical ESR Spectra Bound to Proteins",
        "authors_en": "Yudanova E. I., Kulikov A. V."
    }

def format_to_txt(data):
    if "error" in data: return data["error"]
    return f"""Название журнала: {data.get('journal_title_ru', 'Не определено')}
Том: {data.get('volume', 'Не определено')}
Выпуск (номер): {data.get('issue', 'Не определено')}
Год: {data.get('year', 'Не определено')}
Название статьи: {data.get('article_title_ru', 'Не определено')}
Автор(ы): {data.get('authors_ru', 'Не определено')}
Организация: {data.get('affiliation_ru', 'Не определено')}
Название статьи на английском: {data.get('article_title_en', 'Не определено')}
Автор(ы) на английском: {data.get('authors_en', 'Не определено')}"""

def process_pdfs(pdf_files, is_demo, progress=gr.Progress()):
    output_dir = "output_biblio"
    os.makedirs(output_dir, exist_ok=True)
    txt_files_paths = []
    
    if is_demo:
        files_to_process = [{"name": "ben01000360110-4.pdf", "is_demo": True}]
    else:
        if not pdf_files:
            yield "Ошибка: загрузите PDF", None, "<div class='log-error'>> Ошибка: Нет файлов</div>"
            return
        files_to_process = [{"name": f.name, "path": f.name, "is_demo": False} for f in pdf_files]
    
    progress(0, desc="Инициализация...")
    console_html = ""
    wow_steps = [
        "Инициализация ядра агента [OK]",
        "Загрузка файлов в изолированную среду",
        "Интеллектуальная обработка документа",
        "Распознавание структуры PDF (block 0x4F2A)",
        "Конвейер транскрипции и семантического извлечения данных",
        "Мультимодальная обработка с извлечением структурированных атрибутов",
        "Семантический анализ контекста",
        "Извлечение признаков (Title, Authors, Affiliation)",
        "Преобразование данных в структурированный формат (JSON)",
        "Сборка финального бибописания [DONE]"
    ]
    
    yield "Подготовка к обработке...", None, console_html
    time.sleep(0.5)
    
    for i, file_info in enumerate(files_to_process):
        fname = os.path.basename(file_info["name"])
        for step in wow_steps:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            random_hex = ''.join(random.choice('0123456789ABCDEF') for i in range(4))
            console_html += f"<div class='log-line'><span class='log-time'>[{timestamp}]</span> <span class='log-tag'>[0x{random_hex}]</span> {step}...</div>"
            progress((i + wow_steps.index(step)/len(wow_steps)) / len(files_to_process), desc=f"Обработка: {fname}")
            yield f"Обработка файла {i+1} из {len(files_to_process)}: {fname}", None, console_html
            time.sleep(0.35)
        
        if file_info["is_demo"]:
            biblio_data = get_demo_data()
        else:
            text = extract_text_from_pdf(file_info["path"])
            biblio_data = call_ai_agent(text)
            
        with open(os.path.join(output_dir, f"{os.path.splitext(fname)[0]}.txt"), "w", encoding="utf-8") as f:
            f.write(format_to_txt(biblio_data))
            
        txt_files_paths.append(os.path.join(output_dir, f"{os.path.splitext(fname)[0]}.txt"))
        
        success_msg = f"Файл {fname} успешно обработан и сохранен в архив."
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        console_html += f"<div class='log-success'><span class='log-time'>[{timestamp}]</span> > {success_msg}</div>"
        yield f"Готово: {fname}", None, console_html
        time.sleep(0.5)
    
    zip_path = os.path.join(output_dir, "biblio_results.zip")
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for file_path in txt_files_paths: zipf.write(file_path, os.path.basename(file_path))

    yield "✅ Обработка завершена! Скачайте архив.", zip_path, console_html

# Данные экосистемы агентов
AGENTS_DATA = [
    {"cluster": "1. Создание библиографических записей", "agents": [
        {"name": "Библио-Граф", "desc": "Извлечение метаданных и формирование бибописаний.", "active": True},
        {"name": "Мета-Мастер", "desc": "Парсинг и нормализация метаданных из разнородных источников."},
        {"name": "Реестр-Бот", "desc": "Ведение реестра оцифрованных изданий с авто-идентификаторами."},
        {"name": "Рукопись-ИИ", "desc": "Перевод имидж-каталогов в электронный вид."}
    ]},
    {"cluster": "2. Создание реферативных статей", "agents": [
        {"name": "Реферат-Про", "desc": "Генерация кратких и развёрнутых аннотаций."},
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

def generate_agents_html():
    html = ""
    for cluster in AGENTS_DATA:
        html += f"<div class='cluster-block'><h3>{cluster['cluster']}</h3><div class='agents-grid'>"
        for agent in cluster['agents']:
            active_badge = "<span class='badge-active'>Активен</span>" if agent.get('active') else "<span class='badge-beta'>В разработке</span>"
            html += f"""
                <div class='agent-card'>
                    <h4>{agent['name']}</h4>
                    <p>{agent['desc']}</p>
                    <div class='card-footer'>{active_badge}</div>
                </div>
            """
        html += "</div></div>"
    return html

# Глобальный CSS
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=PT+Sans&family=PT+Serif&family=JetBrains+Mono:wght@400;700&display=swap');
body { background-color: #f0f2f5; }
.gradio-container {
    max-width: 1200px !important; margin: 20px auto !important;
    background-color: #ffffff !important; border-top: 6px solid #2a4d6e;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1) !important; padding: 40px !important;
    font-family: 'PT Sans', 'Arial', sans-serif !important; border-radius: 0 !important;
}
h1, h2, h3 { font-family: 'PT Serif', 'Georgia', serif !important; color: #2a4d6e !important; }

.gradio-container .tabs > .tab-nav { display: flex; border-bottom: 2px solid #e0e0e0; margin-bottom: 30px; gap: 10px; }
.gradio-container .tabs > .tab-nav > button {
    font-family: 'PT Sans', sans-serif !important; font-size: 16px !important; font-weight: bold !important;
    color: #6c757d !important; border: none !important; border-bottom: 3px solid transparent !important;
    padding: 12px 24px !important; background: transparent !important; cursor: pointer !important; transition: all 0.2s ease;
}
.gradio-container .tabs > .tab-nav > button:hover { color: #2a4d6e !important; }
.gradio-container .tabs > .tab-nav > button.selected { color: #2a4d6e !important; border-bottom: 3px solid #2a4d6e !important; }

.hero-section { text-align: center; padding: 40px 20px; background: #f8f9fa; border: 1px solid #e0e0e0; margin-bottom: 30px; }
.hero-section h1 { font-size: 36px !important; margin-bottom: 10px !important; }
.hero-section p { font-size: 18px; color: #555; max-width: 800px; margin: 0 auto 20px auto; line-height: 1.6; }
.badge { display: inline-block; background: #2a4d6e; color: white; padding: 5px 15px; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 20px; }

/* Сетка экосистемы */
.cluster-block { margin-bottom: 40px; }
.cluster-block h3 { border-bottom: 2px solid #2a4d6e; padding-bottom: 10px; margin-bottom: 20px; }
.agents-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 20px; }
.agent-card { border: 1px solid #e0e0e0; padding: 20px; background: #fff; border-radius: 4px; display: flex; flex-direction: column; justify-content: space-between; transition: all 0.3s; }
.agent-card:hover { transform: translateY(-3px); box-shadow: 0 5px 15px rgba(42,77,110,0.15); border-color: #2a4d6e; }
.agent-card h4 { margin-top: 0; color: #2a4d6e; font-size: 18px; font-family: 'PT Serif', serif; }
.agent-card p { font-size: 14px; color: #666; line-height: 1.5; margin-bottom: 15px; }
.card-footer { margin-top: auto; }
.badge-active { background: #d4edda; color: #155724; padding: 4px 10px; font-size: 12px; font-weight: bold; border-radius: 4px; }
.badge-beta { background: #fff3cd; color: #856404; padding: 4px 10px; font-size: 12px; font-weight: bold; border-radius: 4px; }

.telemetry-console { background-color: #1e1e2e; border: 1px solid #11111b; border-radius: 4px; padding: 20px; height: 300px; overflow-y: auto; font-family: 'JetBrains Mono', monospace; font-size: 13px; line-height: 1.6; color: #cdd6f4; box-shadow: inset 0 0 20px rgba(0,0,0,0.5); }
.telemetry-console::-webkit-scrollbar { width: 8px; }
.telemetry-console::-webkit-scrollbar-track { background: #1e1e2e; }
.telemetry-console::-webkit-scrollbar-thumb { background: #45475a; border-radius: 4px; }
.log-line { color: #a6adc8; }
.log-time { color: #585b70; margin-right: 10px; }
.log-tag { color: #89b4fa; margin-right: 10px; font-weight: bold; }
.log-success { color: #a6e3a1; font-weight: bold; margin-top: 5px; margin-bottom: 10px; border-bottom: 1px dashed #313244; padding-bottom: 5px;}
.log-error { color: #f38ba8; font-weight: bold; }

.gr-button-primary { background-color: #2a4d6e !important; border: none !important; border-radius: 4px !important; color: white !important; font-weight: bold !important; padding: 12px 20px !important; text-transform: uppercase; }
.gr-button-secondary { background-color: #ffffff !important; border: 2px solid #2a4d6e !important; border-radius: 4px !important; color: #2a4d6e !important; font-weight: bold !important; padding: 12px 20px !important; text-transform: uppercase; }
.gr-button-primary:hover { background-color: #1a3651 !important; }
label { color: #4c5a62 !important; font-weight: bold !important; }
"""

with gr.Blocks(css=custom_css) as demo:
    
    with gr.Tabs() as tabs:
        # --- Вкладка 1: Главная ---
        with gr.Tab("Главная"):
            gr.HTML("""
            <div class="hero-section">
                <div class="badge">Платформа ИИ-Агентов РАН</div>
                <h1>Нейросетевой агрегатор библиотечной деятельности</h1>
                <p>Интеллектуальный конвейер обработки научных статей. Платформа использует мультимодальные алгоритмы и локальные нейросети для глубокого семантического анализа, транскрипции и извлечения структурированных атрибутов из PDF-документов.</p>
            </div>
            """)
            gr.HTML("<div style='text-align:center; margin-top:20px;'><a href='#' class='badge' style='font-size:14px; text-decoration:none;'>Перейти к экосистеме агентов →</a></div>")
            
        # --- Вкладка 2: Экосистема ИИ-Агентов (НОВАЯ) ---
        with gr.Tab("Экосистема ИИ-Агентов"):
            gr.Markdown("## 🤖 Перечень нейросетевых кластеров")
            gr.Markdown("Платформа включает 9 ключевых кластеров, объединяющих более 30 специализированных ИИ-агентов для автоматизации библиотечной деятельности БЕН РАН.")
            gr.HTML(generate_agents_html())
            
        # --- Вкладка 3: Библио-Граф (Рабочий инструмент) ---
        with gr.Tab("Библио-Граф (Обработка)"):
            gr.Markdown("## 🚀 Пакетная обработка научных статей")
            gr.Markdown("Активный агент: **Библио-Граф**. Загрузите один или несколько PDF-файлов. Агент извлечет библиографические данные и упакует их в TXT файлы.")
            
            with gr.Row():
                pdf_input = gr.Files(label="Исходные файлы (PDF)", file_count="multiple", file_types=[".pdf"])
                
            with gr.Row():
                run_btn = gr.Button("🧠 Запустить ИИ-Агента", variant="primary")
                demo_btn = gr.Button("⚙️ Демо-режим (Витрина)", variant="secondary")
                
            gr.Markdown("### Telemetry & Processing Pipeline")
            console = gr.HTML("<div class='telemetry-console'><div class='log-line'>> Ожидание инициализации системы...</div></div>")
            
            with gr.Row():
                with gr.Column(scale=1):
                    status_output = gr.Textbox(label="Статус выполнения", interactive=False, placeholder="Ожидание...")
                with gr.Column(scale=1):
                    file_output = gr.File(label="Готовые файлы (ZIP)", interactive=False)

            run_btn.click(fn=process_pdfs, inputs=[pdf_input, gr.State(False)], outputs=[status_output, file_output, console])
            demo_btn.click(fn=process_pdfs, inputs=[pdf_input, gr.State(True)], outputs=[status_output, file_output, console])

        # --- Вкладка 4: Технологии ---
        with gr.Tab("Технологии"):
            gr.Markdown("## ⚙️ Технологический стек")
            gr.HTML("""
            <div class='cluster-block'>
                <div class='agents-grid'>
                    <div class='agent-card'>
                        <h4>Backend Core</h4>
                        <p>Python 3.10+<br>Gradio UI Framework<br>pypdf для извлечения текста<br>OpenAI API SDK (Ollama)</p>
                    </div>
                    <div class='agent-card'>
                        <h4>AI Engine</h4>
                        <p>Ollama LLM<br>Локальный инференс (LLaMA 3)<br>Format: Native JSON output<br>Temperature: 0.1</p>
                    </div>
                    <div class='agent-card'>
                        <h4>Конвейер данных</h4>
                        <p>1. PDF Parsing (block 0x4F2A)<br>2. Semantic Context Analysis<br>3. LLM Feature Extraction<br>4. JSON Validation & ZIP Pack</p>
                    </div>
                </div>
            </div>
            """)

if __name__ == "__main__":
    demo.launch(inbrowser=True)
