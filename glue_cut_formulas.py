# Установите: pip install rapidfuzz pillow
from rapidfuzz import fuzz
from PIL import Image, ImageOps, ImageDraw
import json
import os
import re

def tokenize_math_text(text):
    """
    Разбивает строку на элементы (текст и формулы).
    Формулой считается либо блок \begin...\end, либо блок \left...\right,
    причем приоритет отдается внешним конструкциям.
    """
    
    # 1. Паттерн для самых сложных/внешних структур: \left ... \right
    # Мы используем [^]*? или [\s\S]*? для захвата всего, включая переносы строк.
    # Важно: этот паттерн должен быть первым и "поглощать" внутренние \begin/\end.
    pattern_left_right = r'\\left\{.*?\\right[\s\S]*?[\(\)\]\}]?'
    
    # 2. Паттерн для обычных блоков \begin...\end
    pattern_begin_end = r'\\begin\{.*?\}.*?\\end\{.*?\}'
    
    # 3. Паттерн для простых инлайновых формул (если они есть в $...$)
    pattern_inline = r'\$.*?\$'

    # Объединяем в один паттерн. 
    # Порядок критически важен: сначала ищем самые длинные/внешние конструкции.
    combined_pattern = f'({pattern_left_right}|{pattern_begin_end}|{pattern_inline})'
    
    tokens = []
    last_pos = 0
    
    # Используем finditer, чтобы найти все совпадения
    for match in re.finditer(combined_pattern, text, flags=re.DOTALL):
        start, end = match.span()
        
        # Если между последней формулой и текущей есть текст — добавляем его
        if start > last_pos:
            text_segment = text[last_pos:start].strip()
            if text_segment:
                tokens.append(text_segment)
        
        # Добавляем саму формулу
        tokens.append(match.group(0).strip())
        last_pos = end
        
    # Добавляем остаток текста после последней формулы
    if last_pos < len(text):
        remaining_text = text[last_pos:].strip()
        if remaining_text:
            tokens.append(remaining_text)
            
    return tokens


class RobustFormulaReconstructor:
    def __init__(self, image_folder, canvas_size=(2000, 2000), similarity_threshold=70):
        self.image_folder = image_folder
        self.canvas_size = canvas_size
        self.similarity_threshold = similarity_threshold

    def _clean_text(self, text):
        """Очистка текста для максимально честного сравнения."""
        if not text: return ""
        # Убираем LaTeX спецсимволы, пробелы и приводим к нижнему регистру
        text = text.replace("\\", "").replace("{", "").replace("}", "").replace("(", "").replace(")", "")
        text = text.replace(";", "").replace(",", "").replace(";", "")
        return "".join(text.split()).lower()

    def get_bounding_box(self, locs):
        x_coords = locs[0::2]
        y_coords = locs[1::2]
        return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]

    def _ultra_clean_latex(self, text):
        """
        Превращает любой LaTeX в "голый" математический текст.
        Убирает окружения, скобки, команды и спецсимволы.
        """
        if not text:
            return ""
        
        # 1. Удаляем LaTeX окружения и структуры
        # Удаляем \begin{...}, \end{...}, \left, \right, \begin{array}, \end{array}
        text = re.sub(r'\\begin\{.*?\?\}', '', text)
        text = re.sub(r'\\end\{.*?\?\}', '', text)
        text = re.sub(r'\\left', '', text)
        text = re.sub(r'\\right', '', text)
        text = re.sub(r'array', '', text)
        text = re.sub(r'cases', '', text)
        text = re.sub(r'\\array\{.*?\}', '', text)
        text = re.sub(r'\\begin\{array\}\{.*?\}', '', text)
        
        # 2. Удаляем специфические команды (напр. \circ, \begin, \cases)
        text = re.sub(r'\\[a-zA-Z]+', '', text)
        
        # 3. Удаляем все спецсимволы LaTeX, которые могут мешать
        # (оставляем только буквы, цифры и базовые знаки операций)
        text = re.sub(r'[\\\{\}\(\)\[\]\%;,]', ' ', text)
        
        # 4. Очистка пробелов и приведение к нижнему регистру
        text = "".join(text.split()).lower()
        
        return text

    def is_already_correct(self, original_content, etalon_text):
        """
        Проверяет, является ли оригинальный контент уже верным.
        """
        clean_orig = self._ultra_clean_latex(original_content)
        clean_etalon = self._ultra_clean_latex(etalon_text)
        
        # Если после очистки строки идентичны - значит, теги просто разные, 
        # но формула уже распознана верно целиком.
        return fuzz.ratio(clean_orig, clean_etalon) > self.similarity_threshold

    def find_broken_fragments(self, fragments, etalon_texts):
        """
        Возвращает только те группы фрагментов, которые НУЖНО склеивать.
        """
        to_reconstruct = []
        
        # Сначала отфильтруем фрагменты, которые уже являются правильными целыми блоками
        # (Если фрагмент по контенту совпадает с эталоном - это не "сломанный" фрагмент)
        valid_fragments = []
        for frag in fragments:
            is_correct = False
            for etalon in etalon_texts:
                if self.is_already_correct(frag['content'], etalon):
                    etalon_texts.remove(etalon)
                    is_correct = True
                    break
            if not is_correct:
                valid_fragments.append(frag)
        
        return valid_fragments, etalon_texts

    def reconstruct(self, fragments, full_page_etalon_texts):
        """
        fragments: список словарей из JSON
        full_page_etalon_texts: список строк (эталонных систем), найденных на странице
        """
        reconstructed_groups = []

        # 1. Пробуем сопоставить каждый фрагмент с одной из эталонных систем
        # Создаем мапу: { index_эталона: [список_фрагментов] }
        etalon_to_frags = {i: [] for i in range(len(full_page_etalon_texts))}

        for frag in fragments:
            clean_frag = self._clean_text(frag['content'])
            if not clean_frag: continue

            best_match_idx = -1
            highest_score = 0

            for idx, etalon in enumerate(full_page_etalon_texts):
                clean_etalon = self._clean_text(etalon)
                
                # Используем partial_ratio, так как фрагмент — это часть эталона
                score = fuzz.partial_ratio(clean_frag, clean_etalon)
                
                if score > highest_score:
                    highest_score = score
                    best_match_idx = idx

            # Если сходство выше порога, приписываем фрагмент к этой системе
            if highest_score >= self.similarity_threshold:
                etalon_to_frags[best_match_idx].append(frag)

        # 2. Фильтрация и сборка
        for idx, group_frags in etalon_to_frags.items():
            if len(group_frags) > 1:
                # Сортируем фрагменты внутри группы по Y, чтобы они шли по порядку
                group_frags.sort(key=lambda f: self.get_bounding_box(f['locs'])[1])
                
                reconstructed_groups.append({
                    'text': full_page_etalon_texts[idx],
                    'fragments': group_frags
                })

        return self._visualize(reconstructed_groups)

    def _trim_white_borders(self, img):
        """
        Обрезает белые поля вокруг объекта.
        """
        # 1. Переводим в градации серого (L)
        gray_img = img.convert("L")
        
        # 2. Инвертируем изображение: белый станет черным, а всё остальное — белым.
        # Это нужно, потому что getbbox() ищет границы черных пикселей.
        inverted_img = ImageOps.invert(gray_img)
        
        # 3. Находим bounding box (границы) всех не-черных пикселей
        bbox = inverted_img.getbbox()
        
        if bbox:
            # 4. Обрезаем оригинал по найденным координатам
            # Добавим небольшой padding (2-3 пикселя), чтобы не резать символы слишком близко
            padding = 3
            left, top, right, bottom = bbox
            
            # Учитываем границы холста, чтобы не выйти за пределы
            left = max(0, left - padding)
            top = max(0, top - padding)
            right = min(img.width, right + padding)
            bottom = min(img.height, bottom + padding)
            
            return img.crop((left, top, right, bottom))
        else:
            # Если изображение полностью белое, возвращаем его как есть
            return img


    def _visualize(self, groups):
        results = []

        for group in groups:
            canvas = Image.new('RGB', self.canvas_size, color='white')
            draw = ImageDraw.Draw(canvas)
            
            # Считаем общие границы группы
            all_bboxes = [self.get_bounding_box(f['locs']) for f in group['fragments']]
            min_x = min(b[0] for b in all_bboxes)
            min_y = min(b[1] for b in all_bboxes)

            for frag in group['fragments']:
                # Логика поиска файла (замените на вашу!)
                # В примере используем order из JSON
                img_name = f"p71_{frag['order']:03d}.png" 
                img_path = os.path.join(self.image_folder, img_name)

                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    
                    # Смещение относительно верхнего левого угла системы
                    # Добавляем небольшой offset (50, 50), чтобы не прилипало к краю холста
                    offset_x = frag['locs'][0] - min_x + 50
                    offset_y = frag['locs'][1] - min_y + 50
                    
                    canvas.paste(img, (int(offset_x), int(offset_y)))
                    shape = [
                        int(offset_x), 
                        int(offset_y), 
                        int(offset_x + img.width), 
                        int(offset_y + img.height)
                    ]
                    # outline='red' — цвет, width=1 — толщина в 1 пиксель
                    draw.rectangle(shape, outline="red", width=1)
                else:
                    print(f"Warning: File {img_path} not found.")

            results.append({
                'text': group['text'],
                'image': self._trim_white_borders(canvas)
            })
        return results

# --- ТЕСТОВЫЙ СЦЕНАРИЙ ---

try:
    with open('ocr_output/p71_fullpage.latex', 'r', encoding='utf-8') as f:
        example_text = f.read()
    print("Файл p71_fullpage.latex успешно загружен.")
except FileNotFoundError:
    print("Ошибка: Файл p71_fullpage.latex не найден.")
    example_text = ""

try:
    with open('ocr_output/p71_raw.json', 'r', encoding='utf-8') as f:
        mock_fragments = json.load(f)
    print("Файл p71_raw.json успешно загружен.")
except FileNotFoundError:
    print("Ошибка: Файл p71_raw.json не найден.")
    mock_fragments = []
except json.JSONDecodeError:
    print("Ошибка: Файл p71_raw.json содержит некорректный JSON.")
    mock_fragments = []

# Эталон, который мы получили при полном распознавании страницы
# (Чистый LaTeX без мусора)
etalo_texts = [x for x in tokenize_math_text(example_text) if "$" in x]

# Запуск
reconstructor = RobustFormulaReconstructor(image_folder="ocr_output/crops", similarity_threshold=85)


final_results = reconstructor.reconstruct(*reconstructor.find_broken_fragments(mock_fragments, etalo_texts))
i=0
for res in final_results:
    print(f"Found System: {res['text']}")
    res['image'].save(f"final_reconstruction_{i}.png")
    print("Saved: final_reconstruction.png")
    i = i+1
