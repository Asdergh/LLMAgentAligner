from transformers import (PreTrainedModel,
                        AutoModelForCausalLM,
                        AutoConfig,
                        AutoTokenizer,
                        GPT2LMHeadModel, GPT2Config,
                        GPTNeoForCausalLM, GPTNeoConfig,
                        GPTNeoXForCausalLM, GPTNeoXConfig,
                        OPTForCausalLM, OPTConfig,
                        BloomForCausalLM, BloomConfig,
                        CodeGenForCausalLM, CodeGenConfig,
                        XGLMForCausalLM, XGLMConfig,
                        FalconForCausalLM, FalconConfig,
                        LlamaForCausalLM, LlamaConfig,
                        MistralForCausalLM, MistralConfig,
                        Qwen2ForCausalLM, Qwen2Config,
                        Qwen3ForCausalLM, Qwen3Config,
                        GemmaForCausalLM, GemmaConfig,
                        Gemma2ForCausalLM, Gemma2Config,
                        Gemma3ForCausalLM, Gemma3TextConfig,
                        PhiForCausalLM, PhiConfig,
                        Phi3ForCausalLM, Phi3Config,
                        StableLmForCausalLM, StableLmConfig,
                        OlmoForCausalLM, OlmoConfig,
                        Olmo2ForCausalLM, Olmo2Config,
                        GraniteForCausalLM, GraniteConfig,
                        Starcoder2ForCausalLM, Starcoder2Config,
                        GPTBigCodeForCausalLM, GPTBigCodeConfig,
                        SmolLM3ForCausalLM, SmolLM3Config,
                        MambaForCausalLM, MambaConfig,
                        Mamba2ForCausalLM, Mamba2Config)

_SMALL_CAUSAL_LMS = {
    "GPT2":    {"model": GPT2LMHeadModel,        "config": GPT2Config},       # 124M–1.5B
    "GPTNEO":  {"model": GPTNeoForCausalLM,      "config": GPTNeoConfig},     # 125M–2.7B
    "PYTHIA":  {"model": GPTNeoXForCausalLM,     "config": GPTNeoXConfig},    # 14M–12B (архитектура GPT-NeoX)
    "OPT":     {"model": OPTForCausalLM,         "config": OPTConfig},        # 125M–66B
    "BLOOM":   {"model": BloomForCausalLM,       "config": BloomConfig},      # 560M–7B
    "CODEGEN": {"model": CodeGenForCausalLM,     "config": CodeGenConfig},    # 350M–16B
    "XGLM":    {"model": XGLMForCausalLM,        "config": XGLMConfig},       # 564M–7.5B
    "SC":      {"model": GPTBigCodeForCausalLM,  "config": GPTBigCodeConfig}, # SantaCoder / StarCoder v1
    "FALCON":  {"model": FalconForCausalLM,      "config": FalconConfig},     # 1B–180B
    "LLAMA":   {"model": LlamaForCausalLM,       "config": LlamaConfig},      # TinyLlama-1.1B, Llama-3.2-1B/3B, SmolLM(1/2), MobileLLM
    "MISTRAL": {"model": MistralForCausalLM,     "config": MistralConfig},    # 7B+
    "QWEN2":   {"model": Qwen2ForCausalLM,       "config": Qwen2Config},      # 0.5B–72B (Qwen2 / Qwen2.5)
    "QWEN3":   {"model": Qwen3ForCausalLM,       "config": Qwen3Config},      # 0.6B–32B
    "GEMMA":   {"model": GemmaForCausalLM,       "config": GemmaConfig},      # 2B, 7B
    "GEMMA2":  {"model": Gemma2ForCausalLM,      "config": Gemma2Config},     # 2B–27B
    "GEMMA3":  {"model": Gemma3ForCausalLM,      "config": Gemma3TextConfig}, # 270M, 1B (текстовые версии)
    "PHI":     {"model": PhiForCausalLM,         "config": PhiConfig},        # Phi-1/1.5/2 (1.3B–2.7B)
    "PHI3":    {"model": Phi3ForCausalLM,        "config": Phi3Config},       # Phi-3-mini 3.8B, Phi-4-mini
    "STABLELM":{"model": StableLmForCausalLM,    "config": StableLmConfig},   # 1.6B, 3B
    "OLMO":    {"model": OlmoForCausalLM,        "config": OlmoConfig},       # 1B, 7B
    "OLMO2":   {"model": Olmo2ForCausalLM,       "config": Olmo2Config},      # 1B, 7B+
    "GRANITE": {"model": GraniteForCausalLM,     "config": GraniteConfig},    # 2B, 8B
    "SC2":     {"model": Starcoder2ForCausalLM,  "config": Starcoder2Config}, # 3B, 7B, 15B
    "SMOLLM3": {"model": SmolLM3ForCausalLM,     "config": SmolLM3Config},    # 3B

    # --- not transformers ---
    "MAMBA":   {"model": MambaForCausalLM,       "config": MambaConfig},      # 130M–2.8B
    "MAMBA2":  {"model": Mamba2ForCausalLM,      "config": Mamba2Config},     # 130M–2.7B
}
__all__ = ["_SMALL_CAUSAL_LMS"]
# --- Models ---