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

# --- ФУНКЦИЯ ПРОВЕРКИ ПРОТИВОРЕЧИЙ (КРОСС-ВАЛИДАЦИЯ ВВОДА) ---
def validate_input_consistency(selected_zones, user_text, file_text=""):
    text_lower = (user_text + " " + file_text).lower().strip()
    if not text_lower:
        return True, ""

    zone_keywords = {
        "head_neck": ["голова", "головы", "висок", "виске", "затылок", "глаз", "ухо", "макушка", "шея", "шею", "шейный"],
        "upper_limbs": ["плечо", "плече", "рука", "руке", "руки", "локоть", "кисть", "предплечье"],
        "chest": ["грудь", "грудная", "межреберная", "грудина", "сердце"],
        "back_pelvis": ["поясница", "пояснице", "спина", "спине", "крестец", "таз", "ягодица", "бедро"],
        "lower_limbs": ["нога", "ноге", "ноги", "колено", "колене", "стопа", "голень", "пятка", "икра", "икроножная"]
    }

    selected_categories = set()
    for z in selected_zones:
        if "Голова" in z or "Шея" in z:
            selected_categories.add("head_neck")
        elif "Плечевой" in z or "Руки" in z:
            selected_categories.add("upper_limbs")
        elif "Грудной" in z or "Живот" in z:
            selected_categories.add("chest")
        elif "Поясница" in z or "Таз" in z:
            selected_categories.add("back_pelvis")
        elif "Коленный" in z or "Ноги" in z or "Голеностоп" in z:
            selected_categories.add("lower_limbs")

    text_categories = set()
    for cat, keywords in zone_keywords.items():
        if any(kw in text_lower for kw in keywords):
            text_categories.add(cat)

    if text_categories and selected_categories and not selected_categories.intersection(text_categories):
        cat_names = {
            "head_neck": "Голова / Шея",
            "upper_limbs": "Руки / Плечи",
            "chest": "Грудная клетка",
            "back_pelvis": "Поясница / Таз",
            "lower_limbs": "Ноги / Нижние конечности"
        }
        
        text_found = ", ".join([cat_names[c] for c in text_categories])
        zones_found = ", ".join(selected_zones)
        
        err_msg = (
            f"⚠️ **Обнаружено противоречие в данных!**\n\n"
            f"Вы выбрали на карте/в списке зону: **«{zones_found}»**, "
            f"но в текстовом описании упоминаются симптомы из области **«{text_found}»**.\n\n"
            f"Пожалуйста, скорректируйте выбранную зону на карте или уточните текстовое описание."
        )
        return False, err_msg

    return True, ""


def generate_detailed_analysis(spec, urgency, zones_list, flags_str, user_text, file_text=""):
    text_lower = user_text.lower()
    zones_formatted = ", ".join(zones_list) if zones_list else "Не указаны"

    insights = []
    causes = []
    what_to_do = []
    what_not_to_do = []

    if "глаз" in text_lower or "дерг" in text_lower or "тик" in text_lower:
        insights.append("• **Подергивание века / глаза (миокимия):** Связано с перевозбуждением лицевого или тройничного нерва на фоне переутомления, перенапряжения глазных мышц, стресса или нехватки магния.")
        causes.append("Перенапряжение мышц глазного яблока, спазм мимической мускулатуры.")
    
    if "ухо" in text_lower or "уше" in text_lower or "шум" in text_lower:
        insights.append("• **Боль / дискомфорт в ухе при здоровой барабанной перепонке:** Часто является отраженной болью (иррадиацией) из-за спазма глубоких мышц шеи, височно-нижнечелюстного сустава (ВНЧС) или затылочного нерва.")
        causes.append("Иррадиация импульсов по ветвям тройничного/затылочного нервов.")

    if "макушк" in text_lower or "затылок" in text_lower or "темя" in text_lower or "голов" in text_lower:
        insights.append("• **Боль в макушке / темени / затылке (тензионный паттерн):** Напоминает 'шлем' или 'каску'. Обычно вызвана спазмом сухожильного шлема головы (апоневроза) и спазмом трапециевидной мышцы шеи.")
        causes.append("Хронический спазм сухожильного шлема головы и мышц шеи.")

    if "тян" in text_lower or "ноет" in text_lower or "туп" in text_lower:
        insights.append("• **Ноющий/тянущий характер:** Типичный знак миофасциального синдрома — когда мышца из-за перегрузки не может расслабиться.")
    elif "стрел" in text_lower or "остр" in text_lower or "коле" in text_lower or "резк" in text_lower:
        insights.append("• **Простреливающий/острый характер:** Указывает на механическое раздражение нервных стволов или корешков.")

    # Проверка на ЛОР и простудные симптомы / отит / синусит
    if any(w in text_lower for w in ["ухо", "ухе", "отит", "простуд", "орви", "насморк", "заложил", "стреля", "горл", "синусит", "гайморит"]):
        insights.append("• **ЛОР-патологии / воспалительный процесс:** Отечность слизистой, евстахиит или воспаление пазух на фоне ОРВИ/простуды, дающие отраженную боль в ухо или голову.")
        causes.append("Воспаление ЛОР-органов (отит, евстахиит, синусит) или осложнения после простуды.")

    # Проверка на стоматологический фактор / зубы / челюсть / ВНЧС
    if any(w in text_lower for w in ["зуб", "зубы", "челюст", "внчс", "прикус", "жеван"]):
        insights.append("• **Стоматологический фактор / ВНЧС:** Проблемы с зубами верхнего ряда или дисфункция челюстного сустава (корни зубов граничат с гайморовой пазухой).")
        causes.append("Патология зубов или дисфункция ВНЧС, вызывающая иррадиацию боли в голову и ухо через ветви тройничного нерва.")
    if not insights:
        insights.append("• Зафиксирован мышечно-суставной дискомфорт в локализованной зоне, требующий снятия локального спазма и нормализации кинематики движений.")

    causes.extend([
        "Длительное статическое напряжение (работа за ПК, неудобная поза, наклон головы к смартфону)",
        "Психоэмоциональное перенапряжение и стресс (вызывает непроизвольное сжатие челюстей и плеч)",
        "Нарушение микроциркуляции и тонуса локальных сосудов или нервно-мышечной передачи"
    ])

    if spec == "Невролог":
        what_to_do = [
            "**Снизить зрительную и умственную нагрузку:** делать перерывы от экрана каждые 30-40 минут, дать глазам и голове отдых.",
            "**Мягкий самомассаж:** легкими круговыми движениями помассировать затылок, плечи и сухожильный шлем (кожу головы).",
            "**Нормализовать режим сна:** спать не менее 7-8 часов на удобной (желательно ортопедической) подушке.",
            "**Контролировать гидратацию:** пить достаточное количество чистой воды для улучшения тонуса сосудов.",
            "**Запланировать прием невролога:** для проверки нервной проводимости, чувствительности и рефлексов."
        ]
        what_not_to_do = [
            "❌ **НЕ греть место боли горячими грелками или компрессами** (при невралгии или скрытом воспалении это может усилить отек нерва).",
            "❌ **НЕ делать резких хрустов и скручиваний шеей** (опасно травматизацией межпозвоночных суставов и сосудов).",
            "❌ **НЕ злоупотреблять обезболивающими без назначения врача** (это маскирует симптомы и может вызывать 'абузусную' головную боль).",
            "❌ **НЕ игнорировать динамику:** если боль усиливается или добавляются новые симптомы, не откладывать визит к врачу."
        ]
    elif "Травматолог" in spec:
        what_to_do = [
            "**Обеспечить функциональный покой:** исключить физические нагрузки, поднятие тяжестей и резкие движения на пораженную зону.",
            "**Щадящий двигательный режим:** при необходимости использовать эластичный бинт или фиксирующий ортез.",
            "**Пройти диагностику:** сделать УЗИ или рентгенографию суставно-связочного аппарата по назначению врача."
        ]
        what_not_to_do = [
            "❌ **НЕ выполнять упражнения через острую боль** (это увеличивает риск микронадрывов сухожилий и связок).",
            "❌ **НЕ растирать разогревающими мазями при свежем отеке** (это усилит приток крови и увеличит воспаление)."
        ]
    else:
        what_to_do = [
            "**Измерить артериальное давление и пульс:** зафиксировать показатели в течение дня.",
            "**Обеспечить покой и приток свежего воздуха.**",
            "**Пройти первичный осмотр терапевта:** для исключения системных факторов и назначения базовых анализов."
        ]
        what_not_to_do = [
            "❌ **НЕ заниматься самолечением** до первичного осмотра врачом.",
            "❌ **НЕ перегружать организм тяжелыми физическими тренировками.**"
        ]

    insights_str = "\n".join(insights)
    causes_str = "\n".join([f"  * {c}" for c in causes])
    todo_str = "\n".join([f"  1. {act}" for act in what_to_do])
    not_todo_str = "\n".join([f"  * {nact}" for nact in what_not_to_do])

    return f"""
### 💡 1. Что с вами происходит (Суть простыми словами)
Локальный ML-алгоритм **HBBA** проанализировал ваши жалобы (`{user_text if user_text.strip() else 'Жалобы по выбранным зонам'}`) для зоны **{zones_formatted}**.

**Подробный разбор вашей симптоматики:**
{insights_str}

**Почему это возникло (главные причины):**
{causes_str}

---

### 🚨 2. Оценка опасности (Красные флаги)
* **Зафиксированные опасные признаки:** `{flags_str}`
{"* ⚠️ **ВНИМАНИЕ:** Выявлены симптомы, требующие срочного обращения за медицинской помощью!" if flags_str != "Отсутствуют" else "* ✅ Жизненно опасных симптомов (красных флагов) не обнаружено. Состояние требует планового внимания."}

---

### 🟢 3. Пошаговый план: Что нужно делать
* **Рекомендуемый приоритет:** **{urgency}**
* **Ваши первоочередные шаги:**
{todo_str}

---

### 🛑 4. Категорически НЕ рекомендуется делать
Чтобы не ухудшить состояние и не вызвать осложнений, соблюдайте ограничения:
{not_todo_str}

---

### 👨‍⚕️ 5. К какому врачу обратиться
* **Рекомендуемый профильный специалист:** **{spec}**
* **Зачем нужен очный прием:** Врач проведет пальпацию триггерных точек, проверит неврологические рефлексы и при необходимости направит на дополнительную диагностику (МРТ/КТ/УЗИ).
"""

def build_download_report(zones, user_text, red_flags, triage_result, file_text=""):
    zones_str = ", ".join(zones) if zones else "Не указаны"
    flags_str = ", ".join(red_flags) if red_flags else "Отсутствуют"
    file_block = f"\nИЗВЛЕЧЕННЫЙ ТЕКСТ ИЗ ПРИКРЕПЛЕННОГО ФАЙЛА:\n{file_text}\n--------------------------------------------------" if file_text.strip() else ""
    
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
{file_block}

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
    st.info("В проекте HBBA задействован локальный ИИ — классификатор на базе алгоритма Random Forest и векторизации TF-IDF. Модель обучена на подготовленном датасете биомеханических симптомов и в реальном времени определяет профиль врача и уровень риска без обращения к сторонним облачным сервисам.")
    
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
            extracted_file_text = ""
            if uploaded_file is not None:
                if uploaded_file.type == "application/pdf":
                    try:
                        pdf_reader = pypdf.PdfReader(uploaded_file)
                        for page in pdf_reader.pages:
                            extracted_file_text += (page.extract_text() or "") + "\n"
                    except Exception as pdf_err:
                        st.warning(f"Не удалось прочитать PDF-файл: {pdf_err}")
                elif uploaded_file.type == "text/plain":
                    try:
                        extracted_file_text = uploaded_file.read().decode("utf-8")
                    except Exception as txt_err:
                        st.warning(f"Не удалось прочитать TXT-файл: {txt_err}")
            
            # --- ПРОВЕРКА НЕПРОТИВОРЕЧИВОСТИ ВВОДА ---
            is_valid, error_message = validate_input_consistency(st.session_state.selected_zones, user_input, extracted_file_text)
            
            if not is_valid:
                st.error(error_message)
            else:
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
                    input_query = f"{user_input} {extracted_file_text[:500]} {zones_str}"
                    pred_spec = ml_bundle["specialist_model"].predict([input_query])[0]
                    pred_urgency = ml_bundle["urgency_model"].predict([input_query])[0]

                    if active_flags:
                        pred_urgency = "🚨 ЭКСТРЕННО / ТРЕБУЕТСЯ СРОЧНОЕ ВНИМАНИЕ"

                    # Генерируем глубокий разбор
                    result_text = generate_detailed_analysis(
    pred_spec, pred_urgency, st.session_state.selected_zones, flags_str, user_input, extracted_file_text
)

                    st.markdown("## 📊 Результаты автономного ML-триажа:")
                    st.success("Анализ завершен локально движком Scikit-Learn!")
                    st.markdown(result_text)
                    # Отображаем текст прикрепленного файла на экране, если он есть
                    if extracted_file_text.strip():
                       with st.expander("📄 Прочитанный текст из прикрепленного файла"):
                           st.text(extracted_file_text)

                    # Сохранение отчета на диск
                    full_report_text = build_download_report(
                        st.session_state.selected_zones, user_input, active_flags, result_text, extracted_file_text
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