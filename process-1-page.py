#!/usr/bin/env python3
"""
Минимальный модуль: PNG -> PaddleOCR-VL (llama-server) -> вырезанные блоки + текст.
"""

import base64
import json
import re
from pathlib import Path
from dataclasses import dataclass
import logging

import requests
from rapidfuzz import fuzz
from PIL import Image, ImageOps, ImageDraw

from typing import List, Dict, Any

import numpy as np
import pandas as pd

# ======================== НАСТРОЙКИ ========================

API_URL = "http://127.0.0.1:8080/v1/chat/completions"
MODEL_NAME = "PaddlePaddle/PaddleOCR-VL-1.6-GGUF:GGUF"  # имя модели в llama-server (обычно совпадает с именем файла)
TIMEOUT = 300  # секунд; OCR больших изображений может быть медленным

# =========================================================

@dataclass
class Fragment:
    content: str
    locs: List[int]
    image: str
    available: bool

class MathPage:
    """
    Represents a mathematical page containing formulas and their corresponding image crops.
    
    Attributes:
        formulas: List of formula strings parsed from OCR output.
        paths: List of paths to cropped image files corresponding to each formula.
    """

    def __init__(self, *, out_dir, page_text: str, frag_content: List[str], locs_list: List[List[int]], path_list: List[str]) -> None:
        """
        Initialize a MathPage with a list of formulas and their corresponding image paths.
        
        Args:
            formula_list: List of formula strings parsed from OCR.
            path_list: List of paths to cropped image files.
        """
        self.formulas: List[str] = tokenize_math_text(page_text)
        self.fragments: List[Fragment] = [
            Fragment(x, y, z, True) for x, y, z in zip(frag_content, locs_list, path_list)
            ] 
        self.formula_to_fragment_map: Dict[int, int] = {}

        self.sim_threshold_match: int = 80
        self.sim_threshold_reconstruct: int = 80

        self.out_dir = out_dir

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

    def assign_fragments_to_formulas(self) -> None:
        """
        Assign fragments to their corresponding formulas. 
        1. Performs exact matching first.
        2. For unmatched formulas, finds a sequence of adjacent fragments 
        that, when combined, match the target formula.
        """
        # 1. Attempt Exact Matches
        # We assume find_exact_match marks the formula as 'assigned' or updates state
        for formula_idx, _ in enumerate(self.formulas):
            self.find_exact_match(formula_idx)

        # 2. Get Similarity Matrix (using the Pandas version from previous step)
        # Expected: Index = formula_idx, Columns = fragment_idx, Values = score
        sim_matrix = self.construct_similarity_matrix()

        # 3. Reconstruction Logic for unassigned formulas
        for formula_idx, formula_text in enumerate(self.formulas):
            # Skip if this formula was already paired by exact match
            if formula_idx in self.formula_to_fragment_map:
                continue

            clean_target = self._ultra_clean_latex(formula_text)
            if not clean_target:
                continue

            best_sequence = []
            highest_combined_score = 0
            
            # We look for "adjacent" fragments. 
            # We iterate through possible starting fragments and expand.
            for start_frag_idx in range(len(self.fragments)):
                if not self.fragments[start_frag_idx].available:
                    continue

                current_sequence = []
                current_combined_text = ""

                # Expand the sequence window (e.g., try combining up to 5 adjacent fragments)
                for window_size in range(1, 6): 
                    end_frag_idx = start_frag_idx + window_size
                    if end_frag_idx > len(self.fragments):
                        break
                    
                    # Get the next fragment in the sequence
                    next_frag = self.fragments[end_frag_idx - 1]
                    if not next_frag.available:
                        break # Stop sequence if we hit an unavailable fragment

                    current_sequence.append(next_frag)
                    # Glue the content together (simulating reconstruction)
                    current_combined_text += next_frag.content

                    # Calculate how well this "glued" string matches the target
                    # We use partial_ratio or ratio depending on how strict we want to be
                    combined_score = fuzz.ratio(clean_target, self._ultra_clean_latex(current_combined_text))

                    if combined_score > highest_combined_score:
                        highest_combined_score = combined_score
                        best_sequence = list(current_sequence)

            if highest_combined_score > 85:
                image = self.out_dir / f"reconstruct/r_{formula_idx}.png"
                glue_fragments(best_sequence, image)
                self.formula_to_fragment_map[formula_idx] = image
            else:
                logging.info(f"no perfect match for formula {formula_text} found")
#unfinished
        return

    def construct_similarity_matrix(self) -> pd.DataFrame:
        """
        Calculates fuzz.partial_ratio for each pair of formula and fragment 
        and returns a Pandas DataFrame.
        """
        data = []

        for formula_idx, formula in enumerate(self.formulas):
            clean_formula = self._clean_text(formula)
            if not clean_formula:
                continue

            for frag_idx, frag in enumerate(self.fragments):
                if not frag.available:
                    continue
                    
                clean_frag = self._clean_text(frag.content)
                if clean_frag:
                    score = fuzz.partial_ratio(clean_formula, clean_frag)
                    # Store as a flat record
                    data.append({
                        'formula_idx': formula_idx,
                        'fragment_idx': frag_idx,
                        'score': score
                    })

        # Create DataFrame from the list of records
        df = pd.DataFrame(data)

        if df.empty:
            return pd.DataFrame(columns=['formula_idx', 'fragment_idx', 'score'])

        # Pivot the data to get the matrix structure (rows=formula, cols=fragment)
        matrix_df = df.pivot(index='formula_idx', columns='fragment_idx', values='score')
        
        return matrix_df

    def find_exact_match(self, formula_id: int) -> bool:
        """
        Проверяет, является ли оригинальный контент уже верным.
        """
        clean_etalon = self._ultra_clean_latex(self.formulas[formula_id])

        for i, fragment in enumerate(self.fragments):

            content, picture_path, available = fragment.content, fragment.image, fragment.available
            if not available:
                continue
            clean_orig = self._ultra_clean_latex(content)
            match = fuzz.ratio(clean_orig, clean_etalon) > self.sim_threshold_match
            if match:
                self.formula_to_fragment_map[formula_id] = picture_path
                self.fragments[i].available = False
                logging.info(f"assigned fragment {content} to formula {self.formulas}")
                return True # Если после очистки строки идентичны - значит, теги просто разные, 
                            # но формула уже распознана верно целиком.
        logging.info(f"no fragment assigned to formula {self.formulas}")
        return False

    def __repr__(self) -> str:
        return (
            f"MathPage(formulas={len(self.formulas)}, "
            f"paths={len(self.paths)}, "
            f"fragments={len(self.fragments)})"
        )

def glue_fragments(fragment_sequence, out_path, canvas_size=(2000,2000)) -> Path:
    canvas = Image.new('RGB', canvas_size, color='white')
    draw = ImageDraw.Draw(canvas)
    
    # Считаем общие границы группы
    all_bboxes = [get_bounding_box(f.locs) for f in fragment_sequence]
    min_x = min(b[0] for b in all_bboxes)
    min_y = min(b[1] for b in all_bboxes)

    for frag in fragment_sequence:
        # Логика поиска файла (замените на вашу!)
        # В примере используем order из JSON
        img_path = Path(frag.image)

        if img_path.exists():
            img = Image.open(img_path)
            
            # Смещение относительно верхнего левого угла системы
            # Добавляем небольшой offset (50, 50), чтобы не прилипало к краю холста
            offset_x = frag.locs[0] - min_x + 50
            offset_y = frag.locs[1] - min_y + 50
            
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

    img = _trim_white_borders(canvas)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)

def get_bounding_box(locs):
        x_coords = locs[0::2]
        y_coords = locs[1::2]
        return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]

def _trim_white_borders(img: Image) -> Image:
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
def image_to_data_url(image_path: Path) -> str:
    """Кодирует PNG в data URL для OpenAI-совместимого API."""
    data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


LOC_RE = re.compile(r"<\|?LOC_(\d+)\|?>")
# Текст до восьми LOC-токенов (нежадно)
BLOCK_RE = re.compile(r"(?P<text>.*?)(?P<locs>(?:<\|?LOC_\d+\|?>\s*){8})", re.DOTALL)

def call_paddle_OCR(image_path: Path) -> str:
    """
    Отправляет изображение в PaddleOCR-VL с промптом 'OCR:'.
    Возвращает распозанный текст
    """
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
                    {"type": "text", "text": "OCR:"},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": 8192,
    }

    resp = requests.post(API_URL, json=payload, timeout=TIMEOUT)
    resp.raise_for_status()
    raw_content = resp.json()["choices"][0]["message"]["content"]

    return raw_content


def call_paddle_spotting(image_path: Path) -> list[dict]:
    """
    Отправляет изображение в PaddleOCR-VL с промптом 'Spotting:'.
    Возвращает список блоков: [{'content': str, 'locs': [8 int], 'order': int}, ...].
    """
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
                    {"type": "text", "text": "Spotting:"},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": 8192,
    }

    resp = requests.post(API_URL, json=payload, timeout=TIMEOUT)
    resp.raise_for_status()
    raw_content = resp.json()["choices"][0]["message"]["content"]

    # Парсим текст с LOC-токенами
    blocks = parse_spotting_output(raw_content)
    if not blocks:
        print("[!] Не удалось найти ни одного блока с LOC-токенами. Сырой ответ:")
        print(raw_content[:1000])
    return blocks


def parse_spotting_output(raw: str) -> list[dict]:
    """
    Извлекает блоки из сырого ответа Spotting.
    Каждый блок: content (текст до LOC) + locs (8 чисел).
    """
    blocks = []
    order = 0
    for m in BLOCK_RE.finditer(raw):
        content = m.group("text").strip()
        loc_str = m.group("locs")
        locs = [int(x) for x in LOC_RE.findall(loc_str)]
        if len(locs) != 8:
            continue
        order += 1
        blocks.append({
            "content": content,
            "locs": locs,   # нормализованные 0–1000
            "order": order,
        })
    return blocks


def locs_to_bbox_pixels(locs: list[int], width: int, height: int) -> tuple[int, int, int, int]:
    """
    Преобразует 8 нормализованных координат в пиксельный bbox.
    locs = [xTL, yTL, xTR, yTR, xBR, yBR, xBL, yBL]
    """
    xs = [locs[0], locs[2], locs[4], locs[6]]
    ys = [locs[1], locs[3], locs[5], locs[7]]
    x1 = min(xs) * width // 1000
    x2 = max(xs) * width // 1000
    y1 = min(ys) * height // 1000
    y2 = max(ys) * height // 1000
    # Защита от выхода за границы
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return x1, y1, x2, y2


def crop_blocks(
    image_path: Path,
    blocks: list[dict],
    out_dir: Path,
    prefix: str = "block",
) -> list[dict]:
    """
    Вырезает каждый блок по LOC-координатам и сохраняет PNG.
    Возвращает список с путями и текстом.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    img = Image.open(image_path)
    w, h = img.size
    results = []

    for block in blocks:
        order = block["order"]
        content = block["content"]
        if not content:
            continue
        x1, y1, x2, y2 = locs_to_bbox_pixels(block["locs"], w, h)
        crop = img.crop((x1, y1, x2, y2))
        crop_path = out_dir / f"{prefix}_{order:03d}.png"
        crop.save(crop_path)

        results.append({
            "order": order,
            "content": content,
            "bbox_pixels": [x1, y1, x2, y2],
            "locs_normalized": block["locs"],
            "image": str(crop_path),
        })
        print(f"[crop] {crop_path.name}  bbox=({x1},{y1},{x2},{y2})")

    return results


def write_text(results: list[dict], out_file: Path) -> None:
    """Записывает распознанный текст в порядке чтения."""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for item in sorted(results, key=lambda r: r["order"]):
        if item["content"]:
            lines.append(item["content"])
    out_file.write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    print(f"[text] Записано {len(lines)} блоков в {out_file}")


def process_png(image_path: str | Path, out_dir: str | Path) -> None:
    """
    Основная функция: PNG -> spotting -> crop -> текст.
    """
    image_path = Path(image_path)
    out_dir = Path(out_dir) if out_dir else image_path.name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ocr] Отправка {image_path.name} в {API_URL} ...")
    data = call_paddle_spotting(image_path)
    etalon_text = call_paddle_OCR(image_path)
    if not data:
        print("[!] В ответе нет blocks/layout_boxes. Сырой ответ:")
        print(json.dumps(data, ensure_ascii=False, indent=2)[:2000])
        return

    print(f"[ocr] Получено блоков: {len(data)}")
    results = crop_blocks(image_path, data, out_dir / "crops", prefix=image_path.stem)

    (out_dir / f"{image_path.stem}_etalon.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    write_text(results, out_dir / f"{image_path.stem}.md")
    # Сохраним и сырой JSON на случай отладки
    (out_dir / f"{image_path.stem}_raw.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    #import argparse

    #parser = argparse.ArgumentParser(description="PaddleOCR-VL: PNG -> блоки + текст")
    #parser.add_argument("image", help="Путь к PNG-файлу")
    #parser.add_argument("--out", default="ocr_output", help="Папка для результатов")
    #args = parser.parse_args()
 
    #process_png(args.image, args.out)

    with open("ocr_output/p71_fullpage.latex") as f:
        fullpage = f.read()

    with open("ocr_output/p71_raw.json") as f:
        formulas = json.load(f)
    
    images = [
        f for f in Path("ocr_output/crops").iterdir() 
        if f.suffix.lower() == ".png"
    ]
    m = MathPage(
        out_dir=Path("ocr_output"), 
        page_text=fullpage,
        path_list = images,
        locs_list=  [f['locs'] for f in formulas],
        frag_content = [f['content'] for f in formulas]
        )
    m.assign_fragments_to_formulas()
    pass