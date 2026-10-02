"""
Проверка offline-инференса vLLM на маленькой русскоязычной модели.
Три промпта разной длины (~10, ~50, ~100 слов) — смотрим, как модель
справляется с генерацией при разной длине контекста на входе.

Установка (на VM с NVIDIA GPU):
    pip install vllm

Запуск:
    python vllm_context_test.py
"""

from vllm import LLM, SamplingParams
import time

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"  # самая маленькая модель с приличным русским
# Альтернатива, если качества 0.5B не хватит:
# MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# --- Три промпта возрастающей длины ---

prompt_short = [
    {
        "role": "system",
        "content": "Ты — ассистент кардиолога, который объясняет медицинские термины простым языком.",
    },
    {
        "role": "user",
        "content": "Объясни, что такое синдром удлинённого интервала QT, простыми словами за два предложения.",
    },
]  # ~20 слов

prompt_medium = [
    {
        "role": "system",
        "content": ("Ты — ассистент кардиолога в клинике, которая занимается дистанционным анализом ЭКГ. "
                    "Пациенту 58 лет, на записи ЭКГ: ЧСС 48 уд/мин, интервал PQ 240 мс, QRS 105 мс, "
                    "QT 460 мс, QTc 410 мс, RR вариабельны от 1100 до 1400 мс. Пациент жалуется на "
                    "головокружение и тревожится.")
    },
    {
        "role": "user",
        "content": ("Напиши понятный и структурированный ответ, который "
                    "включает описание отклонений в показателях, уточняющий вопрос про приём "
                    "лекарств (бета-блокаторы, антиаритмики) и рекомендацию, что делать до очной "
                    "консультации врача.")
    }
]  # ~55 слов

prompt_long = [
    {
        "role": "system",
        "content": (
            "Ты пишешь клиническое заключение для внутренней системы "
            "поддержки принятия решений в кардиологическом отделении, которая анализирует "
            "суточное холтеровское мониторирование. Данные пациента 67 лет с артериальной "
            "гипертензией и постинфарктным кардиосклерозом: средний RR 820 мс, SDNN 38 мс, "
            "RMSSD 17 мс, интервал PQ 215 мс, QRS 142 мс с морфологией блокады левой ножки "
            "пучка Гиса, QT 490 мс, QTc 520 мс, зарегистрировано 1 240 желудочковых "
            "экстрасистол за сутки и три эпизода неустойчивой желудочковой тахикардии по "
            "5–8 комплексов."
        ),
    },
    {
        "role": "user",
        "content": (
            "Опиши в трёх пунктах, какие патологии и риски следует "
            "заподозрить по этим показателям, какие дополнительные обследования необходимы "
            "и какую тактику наблюдения предложить. Ответ должен быть структурированным и "
            "практичным, ориентированным на лечащего врача, который будет принимать "
            "решение о госпитализации и подборе терапии."
        ),
    },
]  # ~100 слов


prompts = [
    ("SHORT (~10 слов)", prompt_short),
    ("MEDIUM (~50 слов)", prompt_medium),
    ("LONG (~100 слов)", prompt_long),
]

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=300,  # даём модели достаточно места для полного ответа
)

from transformers import AutoModelForCausalLM, \
                        AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
from time import time
for (label, prompt) in prompts:
    encodes = tokenizer.apply_chat_template(prompt, return_tensors="pt")
    s_t = time()
    generation = model.generate(max_new_tokens=256, **encodes)
    generation = generation[:, encodes["input_ids"].shape[1]:]
    e_t = time()
    decodes = tokenizer.batch_decode(generation, skip_special_tokens=True)
    print(36*"=")
    print(f""" GENERATION_RESULTS:
                label: {label},
                time: {e_t - s_t},
                generation: {decodes}""")