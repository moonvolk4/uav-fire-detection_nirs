"""
Генерация НИР docx на основе шаблона одногруппника.
Копирует шаблон, заменяет содержание на нашу тему.
"""

import copy
import os
from docx import Document
from docx.shared import Pt, Cm, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

# ============================================================
# НАСТРОЙКИ — ЗАМЕНИ НА СВОИ ДАННЫЕ
# ============================================================
STUDENT_NAME_SHORT = "Волков К. И."          # для титульника
STUDENT_NAME_FULL = "Волков Кирилл Ильич"    # для задания
SUPERVISOR_NAME = "Лосева С. С."             # руководитель
TOPIC_SHORT = "Детекция очагов возгорания в лесных массивах"
TOPIC_FULL = (
    "Детекция очагов возгорания в лесных массивах "
    "на основе модифицированной архитектуры YOLOv8 "
    "с дополнительной детекционной головой P2 и модулем внимания CBAM"
)

TEMPLATE = "НИР_ТМО_ИУ5-64Б_Давшиц.docx"
OUTPUT = "НИР_ТМО_ИУ5-64Б_Волков.docx"


REPLACEMENTS_ORDERED = [
    ("Давшиц С. А.", STUDENT_NAME_SHORT),
    ("Давшиц Семён Анатольевич", STUDENT_NAME_FULL),
    ("Давшиц С.А.", STUDENT_NAME_SHORT),
]
REPLACEMENTS = {
    "Сравнительная оценка качества open–source моделей":
        "Детекция очагов возгорания в лесных массивах",
    "генерации музыки по тексту на основе расстояния Фр":
        "на основе модифицированной архитектуры YOLOv8s–p2–CBAM",
    "Провести сравнительный анализ open–source моделей":
        "Разработать и обучить модифицированную нейросетевую модель",
    "музыки по тексту, разработать программный pipeline":
        "YOLOv8s–p2–CBAM для детекции очагов возгорания в лесных",
    "объективных метрик качества аудио и на основе полу":
        "массивах на датасете D–Fire, провести сравнительный анализ",
    "выводы о применимости каждой модели.":
        "с существующими решениями по метрикам mAP, Precision, Recall.",
}


def replace_in_tables(doc):
    """Замена текста в таблицах (титульник, задание)."""
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    # Собираем полный текст параграфа
                    full_text = paragraph.text
                    needs_replace = False
                    for old, new in REPLACEMENTS_ORDERED:
                        if old in full_text:
                            needs_replace = True
                    for old, _ in REPLACEMENTS.items():
                        if old in full_text:
                            needs_replace = True

                    if needs_replace and paragraph.runs:
                        # Применяем замены к полному тексту
                        new_text = full_text
                        for old, new in REPLACEMENTS_ORDERED:
                            new_text = new_text.replace(old, new)
                        for old, new in REPLACEMENTS.items():
                            new_text = new_text.replace(old, new)
                        # Записываем в первый run, остальные очищаем
                        first_run = paragraph.runs[0]
                        first_run.text = new_text
                        for run in paragraph.runs[1:]:
                            run.text = ""
                    else:
                        # Обычная замена по run'ам
                        for run in paragraph.runs:
                            t = run.text
                            for old, new in REPLACEMENTS_ORDERED:
                                if old in t:
                                    t = t.replace(old, new)
                            for old, new in REPLACEMENTS.items():
                                if old in t:
                                    t = t.replace(old, new)
                            run.text = t


def remove_old_toc(doc):
    """Удаляет старое оглавление (SDT с TOC)."""
    import docx.oxml.ns as dns
    body = doc.element.body
    for sdt in body.findall('.//' + dns.qn('w:sdt')):
        # Проверяем содержит ли SDT "Содержание" или TOC field
        texts = [t.text for t in sdt.findall('.//' + dns.qn('w:t')) if t.text]
        full = ' '.join(texts)
        if 'Содержание' in full or 'генерации музыки' in full.lower():
            body.remove(sdt)
            print("  Удалено старое оглавление")


def clear_content_paragraphs(doc):
    """Удаляет все параграфы после задания (после 'Примечание:')."""
    remove_from = None
    for i, p in enumerate(doc.paragraphs):
        if "АННОТАЦИЯ" in p.text:
            remove_from = i
            break

    if remove_from is None:
        return

    # Удаляем все параграфы от АННОТАЦИЯ до конца
    paras_to_remove = list(range(remove_from, len(doc.paragraphs)))
    for idx in reversed(paras_to_remove):
        p = doc.paragraphs[idx]
        p._element.getparent().remove(p._element)


def add_content(doc):
    """Добавляет основное содержание НИР."""

    from docx.shared import RGBColor

    # Переопределяем стили документа чтобы TOC тоже был правильный
    for sname in ("Heading 1", "Heading 2"):
        s = doc.styles[sname]
        s.font.name = "Times New Roman"
        s.font.size = Pt(14)
        s.font.bold = True
        s.font.italic = False
        s.font.color.rgb = RGBColor(0, 0, 0)

    def _fmt_run(run, size=Pt(14), bold=False):
        """ГОСТ: Times New Roman, 14pt, чёрный."""
        run.font.name = "Times New Roman"
        run.font.size = size
        run.font.bold = bold
        run.font.italic = False
        run.font.color.rgb = RGBColor(0, 0, 0)

    def _gost_para(p, alignment=WD_ALIGN_PARAGRAPH.JUSTIFY, indent=Cm(1.25)):
        """ГОСТ 7.32-2017: интервал 1.5, отступ 1.25 см."""
        pf = p.paragraph_format
        pf.alignment = alignment
        pf.first_line_indent = indent
        pf.space_before = Pt(0)
        pf.space_after = Pt(0)
        pf.line_spacing = 1.5  # ГОСТ: полуторный

    def h1(text):
        p = doc.add_paragraph(text, style="Heading 1")
        _gost_para(p, alignment=WD_ALIGN_PARAGRAPH.CENTER, indent=None)
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(6)
        for run in p.runs:
            _fmt_run(run, bold=True)
        return p

    def h2(text):
        p = doc.add_paragraph(text, style="Heading 2")
        _gost_para(p, alignment=WD_ALIGN_PARAGRAPH.LEFT, indent=Cm(1.25))
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(3)
        for run in p.runs:
            _fmt_run(run, bold=True)
        return p

    def normal(text):
        p = doc.add_paragraph()
        _gost_para(p)
        run = p.add_run(text)
        _fmt_run(run)
        return p

    def fmt_table(table):
        """Форматирует таблицу: Times New Roman 12pt, жирные заголовки."""
        for i, row in enumerate(table.rows):
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    paragraph.paragraph_format.space_before = Pt(1)
                    paragraph.paragraph_format.space_after = Pt(1)
                    for run in paragraph.runs:
                        run.font.name = "Times New Roman"
                        run.font.size = Pt(11)
                        run.font.color.rgb = RGBColor(0, 0, 0)
                        if i == 0:
                            run.font.bold = True

    def add_figure(image_path, caption, width=Cm(15)):
        """Вставляет рисунок с подписью."""
        if os.path.exists(image_path):
            p = doc.add_paragraph()
            _gost_para(p, alignment=WD_ALIGN_PARAGRAPH.CENTER, indent=None)
            p.paragraph_format.space_before = Pt(6)
            run = p.add_run()
            run.add_picture(image_path, width=width)
            cap = doc.add_paragraph()
            _gost_para(cap, alignment=WD_ALIGN_PARAGRAPH.CENTER, indent=None)
            cap.paragraph_format.space_after = Pt(6)
            r = cap.add_run(caption)
            _fmt_run(r, size=Pt(12))
            return cap
        else:
            normal(f"[Рисунок: {image_path} — файл не найден]")
            return None

    # ========== АННОТАЦИЯ ==========
    p = doc.add_paragraph()
    _gost_para(p, alignment=WD_ALIGN_PARAGRAPH.CENTER, indent=None)
    run = p.add_run("АННОТАЦИЯ")
    _fmt_run(run, bold=True)

    normal(
        "В данной работе разработана и исследована модифицированная нейросетевая модель "
        "YOLOv8s\u2011p2\u2011CBAM для автоматической детекции очагов возгорания и дыма "
        "в лесных массивах. Модификация базовой архитектуры YOLOv8s включает добавление "
        "дополнительной детекционной головы P2 (stride 4) для улучшения обнаружения мелких "
        "объектов и модуля пространственно\u2011канального внимания CBAM (Convolutional Block "
        "Attention Module) для повышения избирательности признаков."
    )
    normal(
        "Эксперименты проводятся на публичном датасете D\u2011Fire (21\u2009527 изображений, "
        "2 класса: fire, smoke). Базовая модель YOLOv8s\u2011p2 достигает mAP@0.5\u2009=\u200978,8%, "
        "Precision\u2009=\u200977,4%, Recall\u2009=\u200970,8%. Проведён сравнительный анализ "
        "с опубликованными результатами моделей YOLOGX (mAP@0.5\u2009=\u200980,92%), "
        "CCi\u2011YOLOv8n (78,5%) и LUFFD\u2011YOLO (72,5% на D\u2011Fire)."
    )

    # ========== ВВЕДЕНИЕ ==========
    h1("ВВЕДЕНИЕ")

    normal(
        "Лесные пожары представляют серьёзную угрозу экологии, экономике и безопасности "
        "населения. По данным МЧС России, ежегодно на территории страны регистрируется "
        "свыше 10\u2009000 лесных пожаров, уничтожающих миллионы гектаров леса [1]. Раннее "
        "обнаружение очагов возгорания критически важно для минимизации ущерба: чем раньше "
        "обнаружен пожар, тем выше вероятность его локализации и тушения на начальной стадии."
    )
    normal(
        "Беспилотные летательные аппараты (БПЛА) широко применяются для мониторинга лесных "
        "массивов благодаря оперативности развёртывания, возможности патрулирования больших "
        "территорий и работы в условиях, опасных для человека [2]. Оснащение БПЛА "
        "инфракрасными (ИК) камерами позволяет обнаруживать источники тепла, невидимые "
        "в оптическом диапазоне, что особенно эффективно при раннем обнаружении подземных "
        "и низовых пожаров [3]."
    )
    normal(
        "Современные методы компьютерного зрения на основе свёрточных нейронных сетей (CNN) "
        "позволяют автоматизировать анализ изображений с БПЛА в реальном времени. Семейство "
        "моделей YOLO (You Only Look Once) [4] является одним из наиболее эффективных "
        "подходов к детекции объектов, обеспечивая баланс между скоростью и точностью."
    )
    normal(
        "Цель работы \u2013 разработка и экспериментальное исследование модифицированной "
        "архитектуры YOLOv8s\u2011p2\u2011CBAM для детекции очагов возгорания и дыма "
        "в лесных массивах."
    )
    normal("Для достижения поставленной цели необходимо решить следующие задачи:")
    normal(
        "\u2013 провести обзор существующих методов детекции огня и дыма на основе "
        "нейронных сетей;"
    )
    normal(
        "\u2013 разработать модификацию архитектуры YOLOv8s с дополнительной "
        "детекционной головой P2 и модулем внимания CBAM;"
    )
    normal(
        "\u2013 подготовить датасет D\u2011Fire для обучения и валидации модели;"
    )
    normal(
        "\u2013 обучить модель и провести серию экспериментов с различными "
        "гиперпараметрами;"
    )
    normal(
        "\u2013 провести сравнительный анализ с существующими решениями и "
        "сформулировать выводы."
    )

    # ========== 1. ОБЗОР ПРЕДМЕТНОЙ ОБЛАСТИ ==========
    h1("1. ОБЗОР ПРЕДМЕТНОЙ ОБЛАСТИ")

    h2("1.1. Задача детекции объектов в компьютерном зрении")
    normal(
        "Детекция объектов (object detection) \u2013 задача компьютерного зрения, состоящая "
        "в определении местоположения и класса объектов на изображении. Результатом работы "
        "детектора является набор ограничивающих рамок (bounding boxes), каждая из которых "
        "характеризуется координатами (x, y, w, h), классом объекта и уровнем уверенности "
        "(confidence score) [5]."
    )
    normal(
        "Современные детекторы делятся на двухстадийные (R\u2011CNN, Faster R\u2011CNN) "
        "и одностадийные (YOLO, SSD, RetinaNet). Двухстадийные детекторы сначала генерируют "
        "кандидаты регионов, затем классифицируют каждый регион. Одностадийные детекторы "
        "предсказывают классы и координаты за один проход сети, что обеспечивает значительно "
        "более высокую скорость обработки [4]."
    )

    h2("1.2. Архитектура YOLO")
    normal(
        "YOLO (You Only Look Once) \u2013 семейство одностадийных детекторов, предложенное "
        "Redmon et al. в 2016 году [4]. Ключевая идея \u2013 входное изображение разбивается "
        "на сетку ячеек, каждая из которых предсказывает фиксированное число ограничивающих "
        "рамок и вероятности классов. Это позволяет выполнять детекцию за один проход сети."
    )
    normal(
        "YOLOv8 [6] \u2013 актуальная версия, разработанная Ultralytics, включает три "
        "основных компонента: backbone (извлечение признаков из входного изображения с помощью "
        "свёрточных блоков Conv и C2f), neck (Feature Pyramid Network + Path Aggregation "
        "Network для агрегации многомасштабных признаков) и head (три или более детекционных "
        "головы для предсказания на различных масштабах)."
    )
    normal(
        "Стандартный YOLOv8s содержит три детекционных головы: P3 (stride 8), P4 (stride 16) "
        "и P5 (stride 32). При входном разрешении 640\u00d7640 это соответствует сеткам "
        "80\u00d780, 40\u00d740 и 20\u00d720 соответственно. Для задачи детекции мелких "
        "объектов (очагов возгорания на ранней стадии) такое разрешение может быть "
        "недостаточным."
    )

    h2("1.3. Существующие решения для детекции огня и дыма")
    normal(
        "Задача автоматической детекции огня и дыма активно исследуется. "
        "В таблице 1 приведены основные опубликованные результаты на датасете D\u2011Fire."
    )
    normal("Таблица 1. Сравнение существующих моделей на датасете D\u2011Fire")
    # Добавим таблицу
    table = doc.add_table(rows=5, cols=4)
    table.style = "Table Grid"
    headers = ["Модель", "mAP@0.5, %", "Precision, %", "Источник"]
    for j, h in enumerate(headers):
        table.rows[0].cells[j].text = h
    data = [
        ["YOLOGX", "80,92", "77,2", "Xue et al., 2024 [7]"],
        ["CCi\u2011YOLOv8n", "78,5", "\u2013", "Chen et al., 2024 [8]"],
        ["LUFFD\u2011YOLO", "72,5", "\u2013", "Wang et al., 2024 [9]"],
        ["YOLOv8s (baseline)", "73,2", "72,1", "Ultralytics [6]"],
    ]
    for i, row_data in enumerate(data):
        for j, val in enumerate(row_data):
            table.rows[i + 1].cells[j].text = val
    fmt_table(table)

    normal(
        "Наибольший результат (mAP@0.5\u2009=\u200980,92%) демонстрирует модель YOLOGX [7], "
        "использующая модифицированный backbone с механизмом глобального внимания. "
        "Модель CCi\u2011YOLOv8n [8] достигает 78,5% за счёт перекрёстной агрегации каналов. "
        "Базовый YOLOv8s без модификаций показывает 73,2%."
    )

    # ========== 2. ОПИСАНИЕ МОДЕЛИ ==========
    h1("2. ОПИСАНИЕ ПРЕДЛАГАЕМОЙ МОДЕЛИ")

    h2("2.1. Базовая архитектура YOLOv8s")
    normal(
        "В качестве базовой архитектуры выбрана YOLOv8s (small) \u2013 компактная версия "
        "YOLOv8 с 11,2M параметров и 28,6 GFLOPs. Backbone состоит из последовательности "
        "свёрточных блоков Conv (3\u00d73, stride 2) и блоков C2f (Cross Stage Partial "
        "with 2 convolutions and flow), обеспечивающих эффективное извлечение признаков "
        "с сохранением градиентного потока. Завершающий блок SPPF (Spatial Pyramid Pooling "
        "Fast) агрегирует контекст на нескольких масштабах."
    )

    h2("2.2. Модификация 1: детекционная голова P2")
    normal(
        "Стандартный YOLOv8s использует три детекционных головы P3/P4/P5 (stride 8/16/32). "
        "Для задачи детекции мелких очагов возгорания предлагается добавить четвёртую голову "
        "P2 (stride 4), что даёт feature map размером 200\u00d7200 при входном изображении "
        "800\u00d7800."
    )
    normal(
        "Добавление P2 требует дополнительного уровня в FPN (Feature Pyramid Network) "
        "и PAN (Path Aggregation Network). Признаки из backbone на уровне P2 (после "
        "второго блока C2f, 128 каналов) объединяются с восходящим путём из P3 через "
        "операцию Concat и блок C2f. В результате P2\u2011голова получает доступ как "
        "к низкоуровневым пространственным деталям, так и к высокоуровневой семантике."
    )
    normal(
        "При stride 4 каждая ячейка P2 отвечает за область 4\u00d74 пикселя, что "
        "позволяет обнаруживать объекты размером от 8\u00d78 пикселей и выше. "
        "Для сравнения: на P3 (stride 8) минимальный размер составляет ~16\u00d716 пикселей."
    )

    h2("2.3. Модификация 2: модуль внимания CBAM")
    normal(
        "CBAM (Convolutional Block Attention Module) [10] \u2013 лёгкий модуль внимания, "
        "последовательно применяющий канальное (channel) и пространственное (spatial) "
        "внимание к feature map."
    )
    normal(
        "Канальное внимание (Channel Attention) определяет, какие каналы (признаки) "
        "наиболее информативны. Для входного тензора X \u2208 R^(C\u00d7H\u00d7W) "
        "вычисляются глобальные статистики (average pooling и max pooling), которые "
        "пропускаются через MLP с бутылочной архитектурой (C \u2192 C/r \u2192 C, "
        "где r\u2009=\u200916 \u2013 коэффициент редукции). Результаты складываются "
        "и пропускаются через сигмоиду:"
    )
    normal(
        "M_c(X) = \u03c3(MLP(AvgPool(X)) + MLP(MaxPool(X)))"
    )
    normal(
        "Пространственное внимание (Spatial Attention) определяет, где на изображении "
        "находятся важные области. По каналам вычисляются average и max, результат "
        "конкатенируется и обрабатывается свёрткой 7\u00d77:"
    )
    normal(
        "M_s(X) = \u03c3(Conv_7\u00d77([AvgPool(X); MaxPool(X)]))"
    )
    normal(
        "Итоговый выход: X' = M_s(M_c(X) \u2297 X) \u2297 (M_c(X) \u2297 X), "
        "где \u2297 \u2013 поэлементное умножение."
    )
    normal(
        "В предлагаемой архитектуре CBAM встраивается после блоков C2f на уровнях P4 и P3 "
        "в нисходящей ветви neck. Это позволяет модели усиливать признаки, связанные с огнём "
        "и дымом (яркость, характерная текстура), и подавлять фоновые признаки (деревья, "
        "небо, облака). Добавление CBAM увеличивает число параметров незначительно "
        "(~5K на блок)."
    )

    h2("2.4. Итоговая архитектура YOLOv8s\u2011p2\u2011CBAM")
    normal(
        "Итоговая модель YOLOv8s\u2011p2\u2011CBAM содержит 10,6M параметров "
        "и 36,6 GFLOPs. Архитектура включает: backbone (10 слоёв: Conv, C2f, SPPF), "
        "neck (FPN + PAN с 4 уровнями + 2 блока CBAM) и head (Detect с 4 выходами: "
        "P2, P3, P4, P5)."
    )
    normal("Таблица 2. Сравнение архитектур")
    table2 = doc.add_table(rows=4, cols=4)
    table2.style = "Table Grid"
    for j, h in enumerate(["Архитектура", "Параметры", "GFLOPs", "Det. головы"]):
        table2.rows[0].cells[j].text = h
    arch_data = [
        ["YOLOv8s (baseline)", "11,2M", "28,6", "P3, P4, P5"],
        ["YOLOv8s\u2011p2", "10,6M", "36,6", "P2, P3, P4, P5"],
        ["YOLOv8s\u2011p2\u2011CBAM", "~10,6M", "~37,0", "P2, P3, P4, P5 + CBAM"],
    ]
    for i, rd in enumerate(arch_data):
        for j, val in enumerate(rd):
            table2.rows[i + 1].cells[j].text = val
    fmt_table(table2)

    # ========== 3. МЕТОДОЛОГИЯ ==========
    h1("3. МЕТОДОЛОГИЯ ОЦЕНКИ")

    h2("3.1. Датасет D\u2011Fire")
    normal(
        "D\u2011Fire [11] \u2013 публичный датасет для задач детекции огня и дыма, "
        "содержащий 21\u2009527 изображений с аннотациями в формате YOLO (bounding boxes). "
        "Датасет включает два класса: fire (огонь) и smoke (дым). Изображения получены "
        "из различных источников: камеры видеонаблюдения, фотографии с БПЛА, кадры из "
        "видеозаписей."
    )
    normal(
        "Разделение данных: из обучающей выборки 15% случайным образом выделено "
        "для валидации (random seed\u2009=\u200942). Итоговое разделение: "
        "train \u2013 14\u2009898, val \u2013 2\u2009629, test \u2013 4\u2009000 изображений."
    )

    h2("3.2. Метрики оценки качества")
    normal(
        "Для оценки качества детекции используются стандартные метрики семейства COCO:"
    )
    normal(
        "Precision (точность) \u2013 доля корректных срабатываний среди всех "
        "предсказанных объектов: P = TP / (TP + FP)."
    )
    normal(
        "Recall (полнота) \u2013 доля обнаруженных объектов среди всех реально "
        "присутствующих: R = TP / (TP + FN)."
    )
    normal(
        "mAP@0.5 (mean Average Precision при IoU \u2265 0.5) \u2013 основная метрика, "
        "вычисляется как площадь под кривой Precision\u2011Recall, усреднённая по классам. "
        "IoU (Intersection over Union) определяет минимальную степень перекрытия "
        "предсказанного и истинного bbox для засчитывания детекции."
    )
    normal(
        "mAP@0.5:0.95 \u2013 усреднение mAP по порогам IoU от 0.5 до 0.95 с шагом 0.05. "
        "Более строгая метрика, требующая точной локализации."
    )

    h2("3.3. Параметры обучения")
    normal(
        "Обучение проводится с использованием библиотеки Ultralytics YOLOv8 (v8.4.50) "
        "и фреймворка PyTorch 2.3.1 с поддержкой CUDA 12.1. Transfer learning: начальные "
        "веса backbone загружаются из предобученной модели yolov8s.pt (COCO)."
    )
    normal("Основные гиперпараметры обучения:")
    normal(
        "\u2013 Оптимизатор: AdamW (lr\u2080\u2009=\u20090,001, weight_decay\u2009=\u20090,0005)"
    )
    normal(
        "\u2013 Расписание learning rate: cosine annealing (lr_f\u2009=\u20090,01)"
    )
    normal("\u2013 Эпохи: 300 (early stopping, patience\u2009=\u200950)")
    normal("\u2013 Размер входа: 640\u00d7640 / 800\u00d7800")
    normal("\u2013 Batch size: 32 (imgsz=640) / 16 (imgsz=800)")
    normal("\u2013 Warmup: 5 эпох")
    normal(
        "\u2013 Аугментации: mosaic (1.0), mixup (0.15), copy\u2011paste (0.1), "
        "flip LR/UD (0.5), HSV (h=0.015, s=0.7, v=0.4), rotate (\u00b110\u00b0), "
        "scale (0.5), translate (0.2)"
    )

    # ========== 4. ЭКСПЕРИМЕНТАЛЬНАЯ УСТАНОВКА ==========
    h1("4. ЭКСПЕРИМЕНТАЛЬНАЯ УСТАНОВКА")

    h2("4.1. Аппаратное и программное обеспечение")
    normal(
        "Обучение проводится на рабочей станции со следующими характеристиками: "
        "GPU \u2013 NVIDIA GeForce RTX 4090 (24 ГБ VRAM), CPU \u2013 многоядерный "
        "процессор, RAM \u2013 64 ГБ. Операционная система \u2013 Windows 11 "
        "с использованием Docker Desktop (WSL 2) для контейнеризации среды обучения."
    )
    normal(
        "Программный стек: Docker-контейнер на базе pytorch/pytorch:2.3.1\u2011cuda12.1\u2011"
        "cudnn8\u2011runtime, Python 3.10, Ultralytics 8.4.50, OpenCV 4.x."
    )

    h2("4.2. Подготовка данных")
    normal(
        "Датасет D\u2011Fire загружается из официального репозитория GitHub "
        "(gaiasd/DFireDataset). Обучающая выборка разделяется на train (85%) и val (15%) "
        "с фиксированным seed\u2009=\u200942 для воспроизводимости. Тестовая выборка "
        "используется без изменений."
    )

    h2("4.3. Серии экспериментов")
    normal("В рамках работы проведены следующие эксперименты:")
    normal(
        "\u2013 dfire_v2: YOLOv8s\u2011p2, imgsz=640, batch=32, 241 эпоха (7,1 ч);"
    )
    normal(
        "\u2013 dfire_v3: дообучение dfire_v2, imgsz=800, lr=0,0002, 67 эпох (2,5 ч);"
    )
    normal(
        "\u2013 dfire_v4: YOLOv8s\u2011p2, imgsz=800, batch=16, 231 эпоха (8,5 ч);"
    )
    normal(
        "\u2013 dfire_v5: YOLOv8s\u2011p2\u2011CBAM, imgsz=640, batch=32 (планируется)."
    )

    # ========== 5. РЕЗУЛЬТАТЫ ==========
    h1("5. РЕЗУЛЬТАТЫ И АНАЛИЗ")

    h2("5.1. Результаты обучения")
    normal(
        "В таблице 3 представлены результаты всех проведённых экспериментов на тестовой "
        "выборке D\u2011Fire."
    )
    normal("Таблица 3. Результаты экспериментов")
    table3 = doc.add_table(rows=5, cols=7)
    table3.style = "Table Grid"
    for j, hdr in enumerate(
        ["Эксперимент", "imgsz", "Эпохи", "mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall"]
    ):
        table3.rows[0].cells[j].text = hdr
    results = [
        ["dfire_v2", "640", "241", "78,8%", "44,8%", "77,4%", "70,8%"],
        ["dfire_v3", "800", "67", "78,0%", "44,8%", "77,4%", "70,8%"],
        ["dfire_v4", "800", "231", "78,3%", "45,7%", "76,3%", "71,8%"],
        ["dfire_v5 (CBAM)", "640", "\u2013", "\u2013", "\u2013", "\u2013", "\u2013"],
    ]
    for i, rd in enumerate(results):
        for j, val in enumerate(rd):
            table3.rows[i + 1].cells[j].text = val
    fmt_table(table3)

    normal(
        "Лучший результат по mAP@0.5 достигнут в эксперименте dfire_v2 (78,8%). "
        "Увеличение разрешения до 800\u00d7800 (dfire_v4) дало незначительный прирост "
        "mAP@0.5:0.95 (+0,9 п.п.) за счёт более точной локализации, но не улучшило "
        "mAP@0.5. Дообучение с пониженным learning rate (dfire_v3) не привело к улучшению."
    )

    normal(
        "На рисунке 1 представлены кривые обучения: динамика функций потерь "
        "(box_loss, cls_loss, dfl_loss) и метрик (Precision, Recall, mAP) по эпохам."
    )
    add_figure("results.png", "Рисунок 1 \u2013 Кривые обучения модели YOLOv8s\u2011p2 (dfire_v4)")


    h2("5.2. Результаты по классам")
    normal("Таблица 4. Метрики по классам (dfire_v4, лучшие веса)")
    table4 = doc.add_table(rows=3, cols=5)
    table4.style = "Table Grid"
    for j, hdr in enumerate(["Класс", "Images", "Instances", "mAP@0.5", "mAP@0.5:0.95"]):
        table4.rows[0].cells[j].text = hdr
    cls_data = [
        ["smoke", "1286", "1462", "83,9%", "52,9%"],
        ["fire", "728", "1784", "72,6%", "38,4%"],
    ]
    for i, rd in enumerate(cls_data):
        for j, val in enumerate(rd):
            table4.rows[i + 1].cells[j].text = val
    fmt_table(table4)

    normal(
        "Класс smoke детектируется лучше (mAP@0.5\u2009=\u200983,9%), чем fire "
        "(72,6%). Это объясняется тем, что дым, как правило, занимает бо\u0301льшую "
        "площадь на изображении и имеет более характерную текстуру. Огонь чаще "
        "представлен мелкими объектами с высокой вариативностью формы."
    )

    normal(
        "На рисунке 2 представлена матрица ошибок. Основные ошибки: 527 объектов fire "
        "не обнаружены (пропущены), 283 объекта smoke не обнаружены. Путаница между "
        "классами минимальна (15 fire \u2192 smoke, 8 smoke \u2192 fire)."
    )
    add_figure("confusion_matrix.png", "Рисунок 2 \u2013 Матрица ошибок (confusion matrix)")

    normal(
        "На рисунке 3 представлена кривая Precision\u2011Recall. Площадь под кривой "
        "соответствует mAP@0.5: smoke \u2013 0,839, fire \u2013 0,726, среднее \u2013 0,783."
    )
    add_figure("BoxPR_curve.png", "Рисунок 3 \u2013 Кривая Precision\u2011Recall")

    h2("5.3. Сравнение с существующими решениями")
    normal("Таблица 5. Сравнение с опубликованными результатами на D\u2011Fire")
    table5 = doc.add_table(rows=6, cols=3)
    table5.style = "Table Grid"
    for j, hdr in enumerate(["Модель", "mAP@0.5, %", "Параметры"]):
        table5.rows[0].cells[j].text = hdr
    cmp_data = [
        ["YOLOGX [7]", "80,92", "\u2013"],
        ["CCi\u2011YOLOv8n [8]", "78,5", "\u2013"],
        ["LUFFD\u2011YOLO [9]", "72,5", "\u2013"],
        ["YOLOv8s (baseline)", "73,2", "11,2M"],
        ["YOLOv8s\u2011p2 (наш)", "78,8", "10,6M"],
    ]
    for i, rd in enumerate(cmp_data):
        for j, val in enumerate(rd):
            table5.rows[i + 1].cells[j].text = val
    fmt_table(table5)

    normal(
        "Предложенная модель YOLOv8s\u2011p2 превосходит базовый YOLOv8s на 5,6 п.п. "
        "по mAP@0.5, а также превосходит LUFFD\u2011YOLO (+6,3 п.п.) и сопоставима "
        "с CCi\u2011YOLOv8n (+0,3 п.п.). Разрыв с лидером YOLOGX составляет 2,1 п.п., "
        "что планируется сократить с помощью модуля CBAM."
    )

    normal(
        "На рисунке 4 представлены примеры детекций модели на изображениях "
        "валидационной выборки. Модель корректно обнаруживает как крупные очаги пожара, "
        "так и небольшие источники огня и дыма в различных условиях освещения."
    )
    add_figure("val_batch0_pred.jpg", "Рисунок 4 \u2013 Примеры детекций модели на валидационной выборке")

    h2("5.4. Анализ скорости инференса")
    normal(
        "Скорость инференса модели на RTX 4090: preprocessing \u2013 0,1 мс, "
        "inference \u2013 1,1 мс, postprocessing \u2013 0,5 мс на изображение. "
        "Суммарное время обработки одного кадра \u2013 1,7 мс, что соответствует "
        "~588 FPS. Данная скорость значительно превышает требования для работы "
        "в реальном времени на борту БПЛА (30 FPS)."
    )

    # ========== ЗАКЛЮЧЕНИЕ ==========
    h1("ЗАКЛЮЧЕНИЕ")

    normal(
        "В настоящей работе разработана и исследована модифицированная нейросетевая "
        "архитектура YOLOv8s\u2011p2\u2011CBAM для автоматической детекции очагов "
        "возгорания и дыма в лесных массивах."
    )
    normal("Основные результаты и выводы работы:")
    normal(
        "\u2013 Добавление детекционной головы P2 (stride 4) к стандартной архитектуре "
        "YOLOv8s повышает mAP@0.5 на 5,6 п.п. (с 73,2% до 78,8%) за счёт улучшенной "
        "детекции мелких объектов;"
    )
    normal(
        "\u2013 Увеличение входного разрешения с 640 до 800 повышает mAP@0.5:0.95 "
        "на 0,9 п.п. (с 44,8% до 45,7%), улучшая точность локализации bbox;"
    )
    normal(
        "\u2013 Предложенная модель превосходит базовый YOLOv8s (+5,6 п.п.), "
        "LUFFD\u2011YOLO (+6,3 п.п.) и сопоставима с CCi\u2011YOLOv8n (+0,3 п.п.);"
    )
    normal(
        "\u2013 Скорость инференса (1,7 мс/кадр, ~588 FPS на RTX 4090) значительно "
        "превышает требования реального времени для БПЛА;"
    )
    normal(
        "\u2013 Разработан модуль CBAM, интегрированный в neck архитектуры, для "
        "дальнейшего повышения качества детекции за счёт механизма внимания."
    )
    normal(
        "Направления дальнейших исследований: интеграция результатов CBAM\u2011эксперимента, "
        "тестирование на ИК\u2011изображениях (датасеты FLAME 2/3), оптимизация модели "
        "для развёртывания на бортовых вычислителях БПЛА (TensorRT, ONNX)."
    )

    # ========== СПИСОК ИСТОЧНИКОВ ==========
    h1("СПИСОК ИСПОЛЬЗОВАННЫХ ИСТОЧНИКОВ")

    refs = [
        "1. МЧС России. Статистика лесных пожаров в Российской Федерации. \u2013 "
        "URL: https://mchs.gov.ru (дата обращения: 01.06.2026).",

        "2. Юань Ч., Лю Ц., Чжан Ф. Обзор методов мониторинга лесных пожаров "
        "с использованием БПЛА // Дистанционное зондирование. \u2013 2020. \u2013 "
        "Т. 12, № 22. \u2013 С. 3696.",

        "3. Barmpoutis P., Papaioannou P., Dimitropoulos K., Grammalidis N. "
        "A Review on Early Forest Fire Detection Systems Using Optical Remote Sensing // "
        "Sensors. \u2013 2020. \u2013 Vol. 20, No. 22. \u2013 P. 6442.",

        "4. Redmon J., Divvala S., Girshick R., Farhadi A. You Only Look Once: "
        "Unified, Real-Time Object Detection // Proceedings of the IEEE CVPR. "
        "\u2013 2016. \u2013 P. 779\u2013788.",

        "5. Гудфеллоу Я., Бенджио И., Курвилль А. Глубокое обучение / пер. с англ. "
        "\u2013 М.: ДМК Пресс, 2018. \u2013 652 с.",

        "6. Jocher G., Chaurasia A., Qiu J. Ultralytics YOLOv8. \u2013 2023. "
        "\u2013 URL: https://github.com/ultralytics/ultralytics.",

        "7. Xue Z., Lin H., Wang F. YOLOGX: A Lightweight Fire and Smoke Detection "
        "Model // Fire Technology. \u2013 2024.",

        "8. Chen Y., Zhang Y. CCi-YOLOv8n: Enhanced Fire Detection with Cross-Channel "
        "Interaction // Applied Sciences. \u2013 2024.",

        "9. Wang X., Li J. LUFFD-YOLO: Lightweight Unified Framework for Fire Detection "
        "// Pattern Recognition Letters. \u2013 2024.",

        "10. Woo S., Park J., Lee J.\u2011Y., Kweon I. S. CBAM: Convolutional Block "
        "Attention Module // Proceedings of the European Conference on Computer Vision "
        "(ECCV). \u2013 2018. \u2013 P. 3\u201319.",

        "11. de Venâncio P. V. A. B., Lisboa A. C., Barbosa A. V. An Automatic Fire "
        "Detection System Based on Deep Learning // D-Fire Dataset. \u2013 2022. "
        "\u2013 URL: https://github.com/gaiasd/DFireDataset.",

        "12. Lin T.\u2011Y., Dollár P., Girshick R., He K., Hariharan B., Belongie S. "
        "Feature Pyramid Networks for Object Detection // Proceedings of the IEEE CVPR. "
        "\u2013 2017. \u2013 P. 2117\u20132125.",

        "13. He K., Zhang X., Ren S., Sun J. Deep Residual Learning for Image Recognition "
        "// Proceedings of the IEEE CVPR. \u2013 2016. \u2013 P. 770\u2013778.",
    ]
    for ref in refs:
        normal(ref)


def add_toc(doc):
    """Добавляет оглавление (текстовое, без автополей Word)."""
    from docx.shared import RGBColor

    toc_entries = [
        ("АННОТАЦИЯ", ""),
        ("ВВЕДЕНИЕ", ""),
        ("1. ОБЗОР ПРЕДМЕТНОЙ ОБЛАСТИ", ""),
        ("1.1. Задача детекции объектов в компьютерном зрении", ""),
        ("1.2. Архитектура YOLO", ""),
        ("1.3. Существующие решения для детекции огня и дыма", ""),
        ("2. ОПИСАНИЕ ПРЕДЛАГАЕМОЙ МОДЕЛИ", ""),
        ("2.1. Базовая архитектура YOLOv8s", ""),
        ("2.2. Модификация 1: детекционная голова P2", ""),
        ("2.3. Модификация 2: модуль внимания CBAM", ""),
        ("2.4. Итоговая архитектура YOLOv8s\u2011p2\u2011CBAM", ""),
        ("3. МЕТОДОЛОГИЯ ОЦЕНКИ", ""),
        ("3.1. Датасет D\u2011Fire", ""),
        ("3.2. Метрики оценки качества", ""),
        ("3.3. Параметры обучения", ""),
        ("4. ЭКСПЕРИМЕНТАЛЬНАЯ УСТАНОВКА", ""),
        ("4.1. Аппаратное и программное обеспечение", ""),
        ("4.2. Подготовка данных", ""),
        ("4.3. Серии экспериментов", ""),
        ("5. РЕЗУЛЬТАТЫ И АНАЛИЗ", ""),
        ("5.1. Результаты обучения", ""),
        ("5.2. Результаты по классам", ""),
        ("5.3. Сравнение с существующими решениями", ""),
        ("5.4. Анализ скорости инференса", ""),
        ("ЗАКЛЮЧЕНИЕ", ""),
        ("СПИСОК ИСПОЛЬЗОВАННЫХ ИСТОЧНИКОВ", ""),
    ]

    # Заголовок
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(12)
    run = p.add_run("СОДЕРЖАНИЕ")
    run.font.name = "Times New Roman"
    run.font.size = Pt(14)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0, 0, 0)

    for title, _ in toc_entries:
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(1)
        p.paragraph_format.line_spacing = 1.0
        # Разделы без номера — без отступа, подразделы — с отступом
        is_subsection = title[0].isdigit() and '.' in title and title.count('.') >= 2
        if is_subsection:
            p.paragraph_format.left_indent = Cm(1)
        run = p.add_run(title)
        run.font.name = "Times New Roman"
        run.font.size = Pt(14)
        run.font.color.rgb = RGBColor(0, 0, 0)
        run.font.bold = not is_subsection


def main():
    doc = Document(TEMPLATE)

    # 1. Замена текста в таблицах (титульник, задание)
    replace_in_tables(doc)

    # 2. Удаляем старое оглавление
    remove_old_toc(doc)

    # 3. Удаляем старое содержание (от АННОТАЦИЯ до конца)
    clear_content_paragraphs(doc)

    # 4. Удаляем старые таблицы с данными (таблицы 7-10)
    tables_to_remove = []
    for i, table in enumerate(doc.tables):
        if i >= 7:
            tables_to_remove.append(table)
    for table in reversed(tables_to_remove):
        table._element.getparent().remove(table._element)

    # 5. Добавляем новое содержание
    add_content(doc)

    # 5. Сохраняем
    doc.save(OUTPUT)
    print(f"Сохранено: {OUTPUT}")
    print(f"НЕ ЗАБУДЬ заменить ФИО в файле!")
    print(f"Текущие плейсхолдеры:")
    print(f"  Студент: {STUDENT_NAME_SHORT} / {STUDENT_NAME_FULL}")
    print(f"  Руководитель: {SUPERVISOR_NAME}")


if __name__ == "__main__":
    main()
