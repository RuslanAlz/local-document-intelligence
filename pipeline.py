import concurrent.futures
import hashlib
import json
import logging
import os
import re
import subprocess
import time
from enum import Enum
from typing import Any, Dict, List, Tuple, Union

import pymupdf4llm
import requests
from langchain_text_splitters import MarkdownTextSplitter
from pydantic import BaseModel, Field
from tqdm import tqdm
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

def detect_hardware() -> dict:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
        )
        vram_mb = int(out.decode().split('\n')[0].strip())
        vram_gb = vram_mb // 1024
    except Exception:
        vram_gb = int(os.environ.get("VRAM_GB", 12))

    # Qwen3:14B requires ~8.5GB VRAM; TranslateGemma:12B requires ~7.5GB
    qwen_workers = max(1, int((vram_gb - 8.5) / 1.0))
    gemma_workers = max(1, int((vram_gb - 7.5) / 1.0))
    keep_models = vram_gb >= 24

    logger.info(
        "Hardware autoscale: %s GB VRAM | Qwen workers: %s | Gemma workers: %s | Keep models: %s",
        vram_gb, qwen_workers, gemma_workers, keep_models,
    )
    return {
        "qwen_workers": qwen_workers,
        "gemma_workers": gemma_workers,
        "keep_models": keep_models,
    }

HW_CONFIG = detect_hardware()

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen3:14b"
TRANSLATE_MODEL = "translategemma:12b"
TARGET_LANGUAGE = "Azerbaijani"
CACHE_VERSION = 1

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")

def get_token_count(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))

def _try_repair_json(text: str) -> str:
    """Repairs truncated JSON responses by balancing brackets and braces."""
    text = text.rstrip()
    try:
        json.loads(text)
        return text
    except Exception:
        pass

    last_brace = text.rfind('}')
    if last_brace != -1:
        candidate = text[:last_brace + 1]
        opens = candidate.count('[') - candidate.count(']')
        closes_obj = candidate.count('{') - candidate.count('}')
        suffix = ']' * opens + '}' * closes_obj
        try:
            json.loads(candidate + suffix)
            return candidate + suffix
        except Exception:
            pass

    return text

def _detect_and_truncate_repetition(text: str) -> str:
    """Detects repetitive generation loops and truncates at the repetition boundary."""
    sentences = re.split(r'(?<=[.!?\n])\s+', text)
    if len(sentences) < 6:
        return text

    seen = {}
    cut_index = None
    for i, s in enumerate(sentences):
        s_norm = s.strip().lower()
        if len(s_norm) < 30:
            continue
        fp = s_norm[:80]
        seen[fp] = seen.get(fp, 0) + 1
        if seen[fp] >= 4:
            count = 0
            for j in range(i + 1):
                fj = sentences[j].strip().lower()[:80]
                if fj == fp:
                    count += 1
                    if count == 2:
                        cut_index = j
                        break
            break

    if cut_index is not None:
        cleaned = ' '.join(sentences[:cut_index]).strip()
        last_period = max(cleaned.rfind('.'), cleaned.rfind('\n'))
        if last_period > len(cleaned) * 0.5:
            cleaned = cleaned[:last_period + 1]
        return cleaned

    return text

def unload_model(model_name: str):
    if HW_CONFIG["keep_models"]:
        return
    logger.info("Evicting model %s from VRAM...", model_name)
    try:
        requests.post(OLLAMA_URL, json={"model": model_name, "keep_alive": 0}, timeout=10)
        time.sleep(2)
    except Exception as e:
        logger.warning("Could not evict model %s: %s", model_name, e)

def call_ollama(prompt: str, schema: dict = None, num_predict: int = 3072, system: str = None, num_ctx: int = 8192, temperature: float = 0.0, think_mode: bool = False, repetition_penalty: float = 1.0, model_override: str = None) -> str:
    payload = {
        "model": model_override if model_override else MODEL_NAME,
        "messages": [],
        "stream": False,
        "think": think_mode,
        "keep_alive": -1,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
            "num_ctx": num_ctx,
            "top_p": 0.1 if temperature == 0.0 else 0.9, 
            "top_k": 10 if temperature == 0.0 else 40,
            "repeat_penalty": repetition_penalty
        }
    }
    # Строго соблюдаем порядок для Prefix Caching: сначала system, потом user
    if system:
        payload["messages"].append({"role": "system", "content": system})
    payload["messages"].append({"role": "user", "content": prompt})
    
    if schema:
        # Аппаратный контроль JSON-формата (Strict Format) на уровне API
        payload["format"] = schema

    for attempt in range(3):
        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=600)
            response.raise_for_status()
            resp_data = response.json()
            content = resp_data.get("message", {}).get("content", "")

            done_reason = resp_data.get("done_reason", "")
            if done_reason == "length" and schema:
                logger.warning("Generation truncated (done_reason=length). Attempting JSON repair...")
                content = _try_repair_json(content)

            return content
        except Exception as e:
            delay = 2 ** (attempt + 1)
            if attempt < 2:
                logger.warning("Ollama request failed (%s). Retrying in %ss...", e, delay)
                time.sleep(delay)
            else:
                logger.error("Ollama request failed after 3 attempts: %s", e)
                return ""

def layout_aware_ingestion(pdf_path: str) -> str:
    import fitz
    doc = fitz.open(pdf_path)
    md_pages = []
    for page_num in tqdm(range(len(doc)), desc="Reading PDF", unit="page", leave=False):
        md_pages.append(pymupdf4llm.to_markdown(doc, pages=[page_num]))
    return "\n".join(md_pages)

def table_aware_chunking(markdown_text: str, source_name: str = None) -> List[str]:
    # chunk_size=2500 leaves ~5500 tokens for system prompt and generated JSON with num_ctx=8192
    splitter = MarkdownTextSplitter(chunk_size=2500, chunk_overlap=200, length_function=get_token_count)
    raw_chunks = splitter.split_text(markdown_text)

    safe_chunks = []
    separator_pattern = re.compile(r'^\s*\|[\s:\-]+\|.*$', re.MULTILINE)
    search_pos = 0

    for chunk in tqdm(raw_chunks, desc="Validating tables", unit="chunk", leave=False):
        lines = chunk.strip().split('\n')
        if not lines:
            safe_chunks.append(chunk)
            continue

        pos = markdown_text.find(chunk[:100], search_pos)
        if pos != -1:
            search_pos = pos

        first_line = lines[0]
        if first_line.strip().startswith('|'):
            is_header = False
            if len(lines) > 1 and separator_pattern.match(lines[1]):
                is_header = True

            if not is_header and pos != -1:
                text_before = markdown_text[:pos]
                separators = list(separator_pattern.finditer(text_before))
                if separators:
                    last_separator = separators[-1]
                    lines_before_sep = text_before[:last_separator.start()].strip().split('\n')
                    if lines_before_sep:
                        header = lines_before_sep[-1]
                        separator_line = last_separator.group().strip()
                        chunk = header + "\n" + separator_line + "\n" + chunk
        if source_name:
            chunk = f"[Source Document: {source_name}]\n\n{chunk}"
        safe_chunks.append(chunk)
    return safe_chunks

class ProfilingResult(BaseModel):
    detected_language: str = Field(description="Primary language of the document")
    categories: List[str] = Field(description="5 to 7 broad categories summarizing the entire document domain")

def sanitize_category(cat: str) -> str:
    cat = re.sub(r'[^a-zA-Z0-9_]', '_', cat)
    cat = re.sub(r'_+', '_', cat).strip('_')
    if not cat or not cat[0].isalpha():
        cat = "Cat_" + cat
    return cat

def dynamic_profiling(chunks: List[str]) -> tuple:
    logger.info("Starting dynamic document profiling...")
    if len(chunks) == 0:
        return "English", ["General"]

    sample_indices = [0, len(chunks) // 2, max(0, len(chunks) - 2)]
    sample_indices = sorted(list(set(sample_indices)))
    sampled_text = "\n\n---\n\n".join([chunks[i][:3000] for i in sample_indices])

    system_prompt = (
        "Analyze this document sample. "
        "1. Identify its primary language. "
        "2. Generate 5 to 7 broad categories that perfectly classify the information in this specific document type."
    )
    prompt = f"Document sample:\n{sampled_text}"

    response = call_ollama(
        prompt, schema=ProfilingResult.model_json_schema(), num_predict=512,
        system=system_prompt, num_ctx=4096, temperature=0.0, think_mode=False
    )
    try:
        data = json.loads(response)
        lang = data.get("detected_language", "English")
        raw_cats = data.get("categories", ["General_Info", "Financial_Data", "Legal_Issues", "Operations", "Risks"])

        safe_cats = [sanitize_category(c) for c in raw_cats]
        safe_cats = list(dict.fromkeys(safe_cats))
        if not safe_cats:
            safe_cats = ["General_Info"]

        logger.info("Profiled document language: %s, categories: %s", lang, safe_cats)
        return lang, safe_cats
    except Exception as e:
        logger.warning("Profiling parse failed (%s), using fallback categories", e)
        return "English", ["Macro_Financial", "Legal_and_Compliance", "Operational", "Risks", "Strategic"]

def create_dynamic_extraction_schema(categories: List[str]) -> dict:
    CategoryEnum = Enum('CategoryEnum', {cat: cat for cat in categories}, type=str)

    class Fact(BaseModel):
        category: CategoryEnum = Field(description="MUST be one of the provided categories. Pick the closest one.")
        details: str = Field(description="Detailed fact with exact numbers or names")
        importance: int = Field(description="Score 1-5")

    class ExtractionResult(BaseModel):
        summary_of_chunk: str = Field(description="A very brief 1-sentence summary of what this text block is about")
        important_facts: List[Fact] = Field(description="List of all important facts, numbers, or legal points extracted from the text")

    return ExtractionResult.model_json_schema()

def map_phase(chunks: List[str], schema: dict, cache_dir: str) -> List[dict]:
    results = []
    logger.info("Map phase: extracting facts using %s workers (cache: %s)...", HW_CONFIG["qwen_workers"], cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    def process_chunk(idx_chunk_tuple):
        i, chunk = idx_chunk_tuple
        cache_file = os.path.join(cache_dir, f"chunk_{i}.json")

        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        system_prompt = (
            "You are a strict data extractor. "
            "Extract ONLY critical facts and classify them STRICTLY into the allowed categories in the schema. "
            "CRITICAL RULES: "
            "1) Extract a MAXIMUM of 5 facts. "
            "2) Each 'details' field MUST be under 80 words. Be precise, not verbose. "
            "3) Assign importance 4-5 ONLY to core data (financial numbers, verdicts, key metrics). "
            "4) DO NOT extract Exhibit numbers, XBRL, filing dates, certificate references, or procedural SEC metadata. Ignore them completely. "
            "5) PRESERVE EXACT NUMBERS ($ amounts, percentages). Do not summarize numbers away. "
            "6) CRITICAL: Double-check magnitudes (Millions vs Billions) and acronyms before outputting. "
            "7) SOURCE TRACKING: The text begins with a [Source Document: filename] tag. You MUST include this filename context in the extracted 'details'."
        )
        prompt = (
            "Extract the most critical facts from this text. Max 5 facts. Keep details short.\n\n"
            f"Text:\n{chunk}"
        )
        response = call_ollama(prompt, schema=schema, system=system_prompt, num_ctx=4096, num_predict=512, temperature=0.0, think_mode=False)
        try:
            data = json.loads(response)
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return data
        except Exception as e:
            logger.warning("Chunk %s processing failed: %s", i, e)
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=HW_CONFIG["qwen_workers"]) as executor:
        for data in tqdm(executor.map(process_chunk, enumerate(chunks, 1)), total=len(chunks), desc="Map extraction", unit="chunk", leave=False):
            if data is not None:
                results.append(data)

    logger.info("Map phase completed: extracted %s JSON fact records", len(results))
    return results

def intra_bucket_aggregate(facts: List[dict], bucket_name: str, schema: dict) -> List[dict]:
    wrapped_data = {"summary_of_chunk": f"Consolidated bucket: {bucket_name}", "important_facts": facts}

    system_prompt = f"You are compressing facts exclusively for the [{bucket_name}] category. Do not change the topic. Keep all facts with importance 4 or 5."
    prompt = (
        "Combine these JSON objects into a single consolidated JSON object. "
        "Deduplicate exact facts. DO NOT remove critical facts with importance 4 or 5.\n\n"
        "OUTPUT FORMAT: Your response MUST be a single JSON object with EXACTLY two keys:\n"
        "1. \"summary_of_chunk\": A short string.\n"
        "2. \"important_facts\": A list of consolidated objects, each having \"category\", \"details\", and \"importance\".\n\n"
        f"Data:\n{json.dumps([wrapped_data])}"
    )
    response = call_ollama(prompt, schema=schema, system=system_prompt, num_ctx=8192, num_predict=4500, temperature=0.0, think_mode=False)
    try:
        data = json.loads(response)
        return data.get("important_facts", [])
    except Exception as e:
        logger.warning("Intra-bucket consolidation failed for %s: %s", bucket_name, e)
        return facts

def thematic_routing(extracted_jsons: List[dict], categories: List[str], schema: dict) -> Dict[str, List[List[dict]]]:
    logger.info("Thematic routing: grouping facts into category buckets...")

    buckets = {cat: [] for cat in categories}
    seen_facts = set()

    for obj in extracted_jsons:
        facts = obj.get("important_facts", [])
        for fact in facts:
            cat = fact.get("category")
            details = fact.get("details", "")

            fingerprint = re.sub(r'[^a-z0-9]', '', details.lower())[:100]
            if len(fingerprint) > 10:
                if fingerprint in seen_facts:
                    continue
                seen_facts.add(fingerprint)

            if cat in buckets:
                buckets[cat].append(fact)
            else:
                if len(categories) > 0:
                    buckets[categories[0]].append(fact)

    final_batches = {}
    for cat, facts in buckets.items():
        if not facts:
            continue

        batches = []
        current_batch = []
        current_tokens = 0
        for fact in facts:
            fact_tokens = get_token_count(json.dumps(fact))
            if current_tokens + fact_tokens > 2000 and current_batch:
                batches.append(current_batch)
                current_batch = [fact]
                current_tokens = fact_tokens
            else:
                current_batch.append(fact)
                current_tokens += fact_tokens
        if current_batch:
            batches.append(current_batch)

        final_batches[cat] = batches
        logger.info("Category %s: %s facts partitioned into %s sub-chapters", cat, len(facts), len(batches))

    return final_batches

def _translate_chapter(english_text: str, chapter_title: str) -> str:
    """Translates a generated chapter to target language via specialized LLM."""
    system_prompt = (
        f"You are a professional translator. Translate the following analytical text from English to {TARGET_LANGUAGE}. "
        "RULES: "
        "1) Preserve ALL numbers, percentages, dollar amounts EXACTLY as-is. "
        "2) Preserve Markdown formatting (headers, bold, lists). "
        "3) Translate naturally — do not transliterate English terms if a proper native term exists. "
        "4) Do NOT add, remove, or change any facts. Translate ONLY. "
        "CRITICAL NUMERICAL RULES: "
        "5) 'billion' MUST be translated as 'milyard'. 'million' MUST be translated as 'milyon'. NEVER confuse these. "
        "6) Double-check every financial figure's scale against the source before outputting. "
        "STYLE RULES: "
        "7) Write in MODERN, clear Azerbaijani. Use simple, short sentences. "
        "8) Avoid archaic Arabic/Persian loanwords — prefer contemporary Azerbaijani vocabulary. "
        "9) Do NOT use overly formal or literary constructions. Write as a modern analyst would speak. "
        "LANGUAGE ENFORCEMENT: "
        "10) ONLY use the Azerbaijani language and the Latin alphabet. "
        "11) NEVER mix in Armenian, Georgian, Russian, or any other languages/alphabets. "
        "12) Dates must be strictly in Azerbaijani format (e.g., '30 sentyabr 2023-cü il' instead of using foreign scripts)."
    )
    prompt = f"Translate this chapter to {TARGET_LANGUAGE}:\n\n{english_text}"

    translated = call_ollama(
        prompt, schema=None, num_predict=3000, system=system_prompt,
        num_ctx=4096, temperature=0.1, think_mode=False, repetition_penalty=1.2,
        model_override=TRANSLATE_MODEL
    )
    translated = _detect_and_truncate_repetition(translated)
    return translated

def multi_chapter_reduce(routed_batches: Dict[str, List[List[dict]]]) -> str:
    logger.info("Multi-chapter reduce: generating structured chapters...")

    full_report = "# Analytical Report\n\n"
    english_chapters = []

    tasks = []
    for cat, batches in routed_batches.items():
        for i, batch in enumerate(batches, 1):
            tasks.append((cat, i, batch, len(batches)))

    def generate_one_chapter(task):
        cat, i, batch, total_batches = task
        chapter_title = f"{cat} (Part {i})" if total_batches > 1 else cat

        system_prompt = (
            f"You are an expert financial analyst. "
            f"Write EXACTLY ONE chapter titled '{chapter_title}' for an analytical report. "
            "RULES: "
            "1) Write in clear, professional English. "
            "2) Use Markdown formatting with bullet points for key metrics. "
            "3) Preserve ALL exact numbers from the data. "
            "4) Do NOT write an introduction or conclusion. "
            "5) Keep it concise — max 400 words."
        )
        prompt = (
            f"Write the chapter '{chapter_title}' based on these extracted facts:\n\n"
            f"{json.dumps(batch, indent=1)}"
        )

        english_text = call_ollama(
            prompt, schema=None, num_predict=2048, system=system_prompt,
            num_ctx=8192, temperature=0.3, think_mode=False, repetition_penalty=1.2
        )
        english_text = _detect_and_truncate_repetition(english_text)
        return {"title": chapter_title, "text": english_text}

    actual_qwen_workers = max(1, min(HW_CONFIG["qwen_workers"], len(tasks) if tasks else 1))

    logger.info("Phase 1: Generating %s chapters in English (%s workers)...", len(tasks), actual_qwen_workers)
    with concurrent.futures.ThreadPoolExecutor(max_workers=actual_qwen_workers) as executor:
        for result in tqdm(executor.map(generate_one_chapter, tasks), total=len(tasks), desc="Generating (EN)", leave=False):
            if result["text"].strip():
                english_chapters.append(result)

    unload_model(MODEL_NAME)

    def translate_one_chapter(chap):
        translated = _translate_chapter(chap["text"], chap["title"])
        if not translated.strip() or len(translated) < len(chap["text"]) * 0.3:
            translated = chap["text"]
        return {"title": chap["title"], "text": translated}

    actual_gemma_workers = max(1, min(HW_CONFIG["gemma_workers"], len(english_chapters) if english_chapters else 1))

    logger.info("Phase 2: Translating %s chapters to %s (%s workers)...", len(english_chapters), TARGET_LANGUAGE, actual_gemma_workers)
    translated_chapters = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=actual_gemma_workers) as executor:
        for result in tqdm(executor.map(translate_one_chapter, english_chapters), total=len(english_chapters), desc=f"Translating ({TARGET_LANGUAGE})", leave=False):
            translated_chapters.append(result)

    for chap in translated_chapters:
        full_report += f"## {chap['title']}\n\n{chap['text']}\n\n---\n\n"

    unload_model(TRANSLATE_MODEL)

    logger.info("Pipeline processing completed successfully")
    return full_report

def process_document(pdf_paths: Union[str, List[str]], force_restart: bool = False) -> str:
    if isinstance(pdf_paths, str):
        pdf_paths = [pdf_paths]

    hasher = hashlib.md5()
    for path in sorted(pdf_paths):
        with open(path, 'rb') as f:
            hasher.update(f.read())
    file_hash = hasher.hexdigest()
    cache_dir = os.path.join(".cache", f"{file_hash}_v{CACHE_VERSION}")

    if force_restart and os.path.exists(cache_dir):
        import shutil
        shutil.rmtree(cache_dir)

    logger.info("Processing %s input document(s)...", len(pdf_paths))
    start_time = time.time()

    all_chunks = []
    for path in pdf_paths:
        source_name = os.path.basename(path)
        logger.info("Ingesting document: %s", source_name)
        markdown_text = layout_aware_ingestion(path)
        chunks = table_aware_chunking(markdown_text, source_name=source_name)
        all_chunks.extend(chunks)

    logger.info("Document ingestion finished in %.1fs (%s total chunks)", time.time() - start_time, len(all_chunks))

    doc_lang, categories = dynamic_profiling(all_chunks)
    schema = create_dynamic_extraction_schema(categories)

    extracted_jsons = map_phase(all_chunks, schema, cache_dir)
    routed_batches = thematic_routing(extracted_jsons, categories, schema)
    final_report = multi_chapter_reduce(routed_batches)

    return final_report
