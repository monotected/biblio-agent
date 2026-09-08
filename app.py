import gradio as gr
import pypdf
import json
import os
import zipfile
import time
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
        if not pdf_files: return "Ошибка: загрузите PDF или включите демо-режим.", None
        files_to_process = [{"name": f.name, "path": f.name, "is_demo": False} for f in pdf_files]
    
    progress(0, desc="Начало обработки...")
    time.sleep(0.5)
    
    for i, file_info in enumerate(files_to_process):
        fname = os.path.basename(file_info["name"])
        progress((i) / len(files_to_process), desc=f"Файл {i+1} из {len(files_to_process)}: {fname}")
        time.sleep(1.2)
        
        if file_info["is_demo"]:
            biblio_data = get_demo_data()
        else:
            biblio_data = call_ai_agent(extract_text_from_pdf(file_info["path"]))
            
        with open(os.path.join(output_dir, f"{os.path.splitext(fname)[0]}.txt"), "w", encoding="utf-8") as f:
            f.write(format_to_txt(biblio_data))
            
        txt_files_paths.append(os.path.join(output_dir, f"{os.path.splitext(fname)[0]}.txt"))
        progress((i+1) / len(files_to_process), desc=f"Готово: {fname}")
        time.sleep(0.5)
    
    zip_path = os.path.join(output_dir, "biblio_results.zip")
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for file_path in txt_files_paths: zipf.write(file_path, os.path.basename(file_path))

    return "Обработка завершена! Скачайте архив.", zip_path

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=PT+Sans&family=PT+Serif&display=swap');
body { background-color: #e8ecef; }
.gradio-container { max-width: 900px !important; margin: 40px auto !important; background-color: #ffffff !important; border-top: 4px solid #2a4d6e; box-shadow: 0 4px 15px rgba(0,0,0,0.07) !important; padding: 40px !important; font-family: 'PT Sans', 'Arial', sans-serif !important; border-radius: 0 !important; }
h1, h2, h3 { font-family: 'PT Serif', 'Georgia', serif !important; color: #2a4d6e !important; }
.gr-button-primary { background-color: #2a4d6e !important; border: none !important; border-radius: 0 !important; color: white !important; font-weight: bold !important; padding: 12px 20px !important; }
.gr-button-secondary { background-color: #f8f9fa !important; border: 1px solid #2a4d6e !important; border-radius: 0 !important; color: #2a4d6e !important; font-weight: bold !important; padding: 12px 20px !important; }
.gr-button-primary:hover { background-color: #1a3651 !important; }
.gr-box, .gr-input, .gr-textbox { border-radius: 0 !important; border: 1px solid #ccd1d6 !important; }
label { color: #4c5a62 !important; font-weight: bold !important; }
"""

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("# ИИ-Агент для извлечения библиографических данных")
    gr.Markdown("Система пакетной обработки научных статей. Для работы с реальными файлами необходима запущенная Ollama.")
    with gr.Row():
        pdf_input = gr.Files(label="Исходные файлы (PDF)", file_count="multiple", file_types=[".pdf"])
    with gr.Row():
        run_btn = gr.Button("Запустить Ollama", variant="primary")
        demo_btn = gr.Button("Демо-режим (имитация)", variant="secondary")
    with gr.Row():
        status_output = gr.Textbox(label="Статус выполнения", interactive=False, placeholder="Ожидание...")
        file_output = gr.File(label="Готовые файлы (ZIP)", interactive=False)

    run_btn.click(fn=process_pdfs, inputs=[pdf_input, gr.State(False)], outputs=[status_output, file_output])
    demo_btn.click(fn=process_pdfs, inputs=[pdf_input, gr.State(True)], outputs=[status_output, file_output])

if __name__ == "__main__":
    demo.launch(inbrowser=True)
