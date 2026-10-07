import time
import os
from datetime import datetime
import streamlit as st
import streamlit.components.v1 as components
import pypdf
import joblib

# Настройка страницы
st.set_page_config(page_title="HBBA Local ML Engine", page_icon="🧬", layout="wide")

st.title("🧬 HBBA — Human Body Biomechanics Analyzer")
st.subheader("Автономная система первичного триажа на базе собственной ML-модели")

# Загрузка обученной локальной модели
@st.cache_resource
def load_custom_ml_model():
    if os.path.exists('hbba_model.pkl'):
        return joblib.load('hbba_model.pkl')
    return None

ml_bundle = load_custom_ml_model()

if "selected_zones" not in st.session_state:
    st.session_state.selected_zones = ["Голова / Виски / Затылок"]

def toggle_zone(zone_name):
    if zone_name in st.session_state.selected_zones:
        st.session_state.selected_zones.remove(zone_name)
    else:
        st.session_state.selected_zones.append(zone_name)

# Расширенная база знаний по биомеханике
BIOMECHANICS_KNOWLEDGE_BASE = {
    "Невролог": {
        "summary": "Развитие синдрома связано с нарушением регуляции нервно-мышечной передачи, локальным миофасциальным спазмом или иррадиацией боли по ходу периферических нервных стволов.",
        "causes": [
            "Длительное статическое напряжение (работа за ПК, неудобная поза)",
            "Компрессия или раздражение нервных корешков спинного мозга",
            "Рефлекторный мышечный спазм в ответ на стресс или переохлаждение",
            "Изменение тонуса церебральных или периферических сосудов"
        ],
        "anatomy": "Вовлечены **трапециевидная, ременная мышцы шеи**, **затылочные нервы (n. occipitalis)** или корешки спинномозговых нервов. Наблюдается гипертонус триггерных точек.",
        "actions": [
            "Ограничить резкие движения и высокие осевые нагрузки",
            "Обеспечить эргономичное положение тела при отдыхе и работе",
            "Вести дневник интенсивности и характера боли",
            "Запланировать очный осмотр невролога для проверки рефлексов"
        ]
    },
    "Травматолог-ортопед": {
        "summary": "Симптоматика указывает на механическое перенапряжение связочно-суставного аппарата, нарушение биомеханической оси нагрузки или микротравматизацию тканевых структур.",
        "causes": [
            "Перегрузка суставно-связочного аппарата при физических упражнениях",
            "Дегенеративно-дистрофические изменения суставных поверхностей",
            "Микронадрывы сухожилий или растяжение связок",
            "Нарушение осанки и паттернов ходьбы/движения"
        ],
        "anatomy": "Затронуты **суставные капсулы**, **связочные комплексы** и **межпозвоночные/периферические суставы**. Выявлено ограничение амплитуды движения.",
        "actions": [
            "Обеспечить функциональный покой пораженной области",
            "Исключить осевые и скручивающие нагрузки",
            "При необходимости использовать щадящую фиксацию/ортезирование",
            "Пройти инструментальную диагностику (рентгенография / МРТ)"
        ]
    },
    "Кардиолог / Терапевт": {
        "summary": "Картина может отражать изменения системной гемодинамики, реактивность сосудистого тонуса или отраженные висцеральные боли.",
        "causes": [
            "Колебания артериального давления (гипер- или гипотоническая реакция)",
            "Вегетативная дисфункция сосудистого тонуса",
            "Межреберная невралгия или отраженный мышечный синдром",
            "Психоэмоциональное перенапряжение"
        ],
        "anatomy": "Рефлекторный отклик **межреберных нервных сплетений**, **сосудистого русла** и мышечного каркаса грудной клетки.",
        "actions": [
            "Измерить и зафиксировать артериальное давление и пульс",
            "Исключить интенсивные физические нагрузки и стрессовые факторы",
            "Обеспечить приток свежего воздуха и покой",
            "Обратиться к терапевту или кардиологу для ЭКГ-диагностики"
        ]
    },
    "Терапевт / Общий специалист": {
        "summary": "Симптомокомплекс требует первичной дифференциальной диагностики для уточнения характера патологического процесса.",
        "causes": [
            "Общесоматические и воспалительные реакции",
            "Переутомление и физическое перенапряжение",
            "Системные изменения тонуса мышц и фасций"
        ],
        "anatomy": "Вовлечение **поверхностных фасциальных футляров** и системных рефлекторных дуг.",
        "actions": [
            "Зафиксировать точное время и обстоятельства появления симптомов",
            "Избегать самолечения до консультации со специалистом",
            "Пройти первичный осмотр и сдать базовые анализы крови"
        ]
    }
}

def generate_detailed_analysis(spec, urgency, zones_str, flags_str, user_text):
    info = BIOMECHANICS_KNOWLEDGE_BASE.get(spec, BIOMECHANICS_KNOWLEDGE_BASE["Терапевт / Общий специалист"])
    
    causes_list = "\n".join([f"  * {c}" for c in info['causes']])
    actions_list = "\n".join([f"  * {act}" for act in info['actions']])
    
    # Дополнительный анализ введённого текста
    text_insight = ""
    text_lower = user_text.lower()
    if "тян" in text_lower or "ноет" in text_lower:
        text_insight = "Характер боли (тянущий/ноющий) типичен для хрононического мышечного гипертонуса или фасциального натяжения."
    elif "стрел" in text_lower or "остр" in text_lower or "колет" in text_lower:
        text_insight = "Острый/стреляющий характер боли указывает на возможное раздражение нервных окончаний или корешковый синдром."
    elif "пульс" in text_lower:
        text_insight = "Пульсирующий характер симптоматики часто связан с изменениями сосудистого тонуса и кровотока."
    else:
        text_insight = "Зафиксирован комплексный характер симптомов, требующий оценки кинематики движений."

    return f"""
### 1. 💡 Краткая суть простыми словами
Локальный алгоритм **HBBA ML Engine** обработал жалобы для локализации: **{zones_str}**.

{info['summary']}

---

### 2. 🚨 Оценка опасных симптомов (Red Flags)
* **Статус симптомов:** `{flags_str}`
{"* ⚠️ **ВНИМАНИЕ:** Выявлены тревожные признаки! Настоятельно рекомендуется приоритетный осмотр врача для исключения острых состояний." if flags_str != "Отсутствуют" else "* ✅ Тревожных красных флагов по выбранным пунктам не зафиксировано."}

---

### 3. 🦴 Анатомический и биомеханический анализ
* **Анатомические структуры:** {info['anatomy']}
* **Анализ жалоб пациента:** {text_insight}
* **Возможные провоцирующие факторы:**
{causes_list}

---

### 4. ⚠️ Степень срочности и рекомендательный план
* **Рекомендуемый приоритет:** **{urgency}**
* **Первоочередные шаги:**
{actions_list}

---

### 5. 👨‍⚕️️ Рекомендуемый профильный специалист
Приоритетная первичная консультация: **{spec}**
"""

def build_download_report(zones, user_text, red_flags, triage_result):
    zones_str = ", ".join(zones) if zones else "Не указаны"
    flags_str = ", ".join(red_flags) if red_flags else "Отсутствуют"
    
    report = f"""==================================================
HBBA — HUMAN BODY BIOMECHANICS ANALYZER REPORT
==================================================
Дата и время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Движок обработки: Custom ML Engine (Scikit-Learn / Offline)
Выбранные зоны: {zones_str}
Опасные симптомы (Красные флаги): {flags_str}
--------------------------------------------------

ОПИСАНИЕ СИМПТОМОВ ОТ ПАЦИЕНТА:
{user_text if user_text.strip() else 'Не указано'}

--------------------------------------------------
РЕЗУЛЬТАТЫ ПЕРВИЧНОГО ТРИАЖА И АНАЛИЗА:
--------------------------------------------------

{triage_result}

==================================================
Примечание: Данный отчет сформирован автоматизированной системой 
первичного триажа HBBA и не является официальным медицинским диагнозом.
"""
    return report

# Боковая панель
with st.sidebar:
    st.header("⚙️ Режим работы")
    st.success("🟢 Оффлайн ML-режим активен")
    st.info("В проекте HBBA задействован локальный ИИ — классификатор на базе алгоритма Random Forest и векторизации TF-IDF. Модель обучена на подготовленном датасете биомеханических симптомов и в реальном времени определяет профиль врача, уровень риска и первичные рекомендации без обращения к сторонним облачным сервисам.")
    
    st.markdown("---")
    st.header("💡 О системе HBBA")
    st.markdown("""
    **HBBA** использует собственную обученную модель классификации для первичной оценки симптомов, категории риска и определения профильного специалиста.
    """)
    st.warning("⚠️ Сервис не заменяет очный прием врача.")

if ml_bundle is None:
    st.error("⚠️ Файл модели `hbba_model.pkl` не найден в папке проекта! Убедитесь, что запущен `train_model.py`.")
else:
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.markdown("### 1. 🧍 Карта тела и выбор зон")
        
        head_color = "#ff4b4b" if "Голова / Виски / Затылок" in st.session_state.selected_zones or "Шея / Шейный отдел позвоночника" in st.session_state.selected_zones else "#f38ba8"
        hands_color = "#ff4b4b" if "Плечевой сустав / Руки" in st.session_state.selected_zones else "#89b4fa"
        chest_color = "#ff4b4b" if "Грудной отдел / Грудная клетка" in st.session_state.selected_zones or "Живот / Абдоминальная область" in st.session_state.selected_zones else "#a6e3a1"
        back_color = "#ff4b4b" if "Поясница / Пояснично-крестцовый отдел" in st.session_state.selected_zones or "Таз / Тазобедренный сустав" in st.session_state.selected_zones else "#f9e2af"
        legs_color = "#ff4b4b" if "Коленный сустав / Ноги" in st.session_state.selected_zones or "Голеностоп / Стопа" in st.session_state.selected_zones else "#74c7ec"

        body_svg = f"""
        <div style="text-align: center; background: #1e1e2e; padding: 15px; border-radius: 12px; border: 1px solid #313244;">
            <svg width="180" height="330" viewBox="0 0 200 400" xmlns="http://www.w3.org/2000/svg">
                <circle cx="100" cy="45" r="28" fill="{head_color}" stroke="#ffffff" stroke-width="1.5"/>
                <rect x="91" y="75" width="18" height="18" fill="{head_color}" rx="4"/>
                <rect x="30" y="95" width="24" height="110" fill="{hands_color}" rx="10"/>
                <rect x="146" y="95" width="24" height="110" fill="{hands_color}" rx="10"/>
                <rect x="60" y="95" width="80" height="95" fill="{chest_color}" rx="8"/>
                <rect x="64" y="193" width="72" height="40" fill="{back_color}" rx="6"/>
                <rect x="68" y="236" width="28" height="145" fill="{legs_color}" rx="10"/>
                <rect x="104" y="236" width="28" height="145" fill="{legs_color}" rx="10"/>
            </svg>
        </div>
        """
        components.html(body_svg, height=350)

        st.caption("Нажмите на нужную зону для выбора:")
        b1, b2 = st.columns(2)
        with b1:
            if st.button("🧠 Голова / Шея", use_container_width=True):
                toggle_zone("Голова / Виски / Затылок")
            if st.button("💪 Плечи / Руки", use_container_width=True):
                toggle_zone("Плечевой сустав / Руки")
            if st.button("🩻 Поясница / Спина", use_container_width=True):
                toggle_zone("Поясница / Пояснично-крестцовый отдел")
        with b2:
            if st.button("🫁 Грудная клетка", use_container_width=True):
                toggle_zone("Грудной отдел / Грудная клетка")
            if st.button("🦴 Таз / Бедра", use_container_width=True):
                toggle_zone("Таз / Тазобедренный сустав")
            if st.button("🦵 Ноги / Суставы", use_container_width=True):
                toggle_zone("Коленный сустав / Ноги")

        st.session_state.selected_zones = st.multiselect(
            "Активные зоны:",
            [
                "Голова / Виски / Затылок",
                "Шея / Шейный отдел позвоночника",
                "Плечевой сустав / Руки",
                "Грудной отдел / Грудная клетка",
                "Живот / Абдоминальная область",
                "Поясница / Пояснично-крестцовый отдел",
                "Таз / Тазобедренный сустав",
                "Коленный сустав / Ноги",
                "Голеностоп / Стопа"
            ],
            default=st.session_state.selected_zones
        )

    with col2:
        st.markdown("### 2. 📝 Детализируйте симптомы и загрузите данные")
        
        user_input = st.text_area(
            "Описание характера боли и ограничений движения:",
            placeholder="Например: Боль тянущая, усиливается при повороте головы, отдаёт в плечо...",
            height=130
        )

        st.markdown("#### 🚨 Красные флаги (опасные симптомы):")
        flag_numbness = st.checkbox("Онемение или слабость в конечностях")
        flag_dizziness = st.checkbox("Головокружение, тошнота или потемнение в глазах")
        flag_acute = st.checkbox("Резкая острая боль при выдохе/вдохе или нагрузке")

        uploaded_file = st.file_uploader(
            "📎 Прикрепить файл (МРТ, КТ, УЗИ или медицинский отчет PDF/TXT):", 
            type=["pdf", "txt"]
        )

    st.markdown("---")

    if st.button("🔍 Провести локальный ML-анализ", use_container_width=True):
        if user_input.strip() or st.session_state.selected_zones or uploaded_file is not None:
            
            zones_str = ", ".join(st.session_state.selected_zones) if st.session_state.selected_zones else "Не указано"
            
            active_flags = []
            if flag_numbness: active_flags.append("Онемение/слабость в конечностях")
            if flag_dizziness: active_flags.append("Головокружение/тошнота")
            if flag_acute: active_flags.append("Резкая острая боль")
            flags_str = ", ".join(active_flags) if active_flags else "Отсутствуют"

            # Извлечение текста из PDF
            pdf_text = ""
            if uploaded_file is not None and uploaded_file.type == "application/pdf":
                try:
                    pdf_reader = pypdf.PdfReader(uploaded_file)
                    for page in pdf_reader.pages:
                        pdf_text += page.extract_text() or ""
                except Exception as pdf_err:
                    st.warning(f"Не удалось прочитать PDF-файл: {pdf_err}")

            with st.spinner("Локальный ML-модуль обрабатывает симптомы..."):
                time.sleep(0.2)
                
                # Запрос к обученной модели
                input_query = f"{user_input} {zones_str} {pdf_text[:500]}"
                pred_spec = ml_bundle["specialist_model"].predict([input_query])[0]
                pred_urgency = ml_bundle["urgency_model"].predict([input_query])[0]

                if active_flags:
                    pred_urgency = "🚨 ЭКСТРЕННО / ТРЕБУЕТСЯ СРОЧНОЕ ВНИМАНИЕ"

                # Генерируем глубокий разбор
                result_text = generate_detailed_analysis(
                    pred_spec, pred_urgency, zones_str, flags_str, user_input
                )

                st.markdown("## 📊 Результаты автономного ML-триажа:")
                st.success("Анализ завершен локально движком Scikit-Learn!")
                st.markdown(result_text)

                # Сохранение отчета на диск
                full_report_text = build_download_report(
                    st.session_state.selected_zones, user_input, active_flags, result_text
                )
                
                folder_path = "reports"
                os.makedirs(folder_path, exist_ok=True)
                filename = f"HBBA_Triage_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                full_file_path = os.path.join(folder_path, filename)
                with open(full_file_path, "w", encoding="utf-8") as f:
                    f.write(full_report_text)

                st.download_button("📥 Скачать отчет (.txt)", data=full_report_text, file_name=filename, mime="text/plain")

        else:
            st.warning("Пожалуйста, выберите хотя бы одну зону на теле, введите описание или прикрепите файл.")