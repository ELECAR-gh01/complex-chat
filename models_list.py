MODELS = {
    # =======================
    # ---- OpenAI ----
    # =======================
    # Free
    "OpenAI / GPT OSS 20B (Free)"     : ("stima", "gpt-oss-20b:free"),

    # Paid (Text)
    "OpenAI / GPT-5 Chat"             : ("stima", "gpt-5-chat-latest"),
    "OpenAI / GPT-5"                  : ("stima", "gpt-5-2025-08-07"),
    "OpenAI / GPT-5 Mini"             : ("stima", "gpt-5-mini-2025-08-07"),
    "OpenAI / GPT-5 Nano"             : ("stima", "gpt-5-nano-2025-08-07"),
    "OpenAI / GPT OSS 120B"           : ("stima", "gpt-oss-120b"),
    "OpenAI / GPT-4o"                 : ("stima", "gpt-4o"),
    "OpenAI / GPT-4o Mini"            : ("stima", "gpt-4o-mini"),
    "OpenAI / GPT-4.1"                : ("stima", "gpt-4.1"),
    "OpenAI / GPT-4.1 Mini"           : ("stima", "gpt-4.1-mini"),
    "OpenAI / GPT-4.1 Nano"           : ("stima", "gpt-4.1-nano"),

    # Non-Text
    "OpenAI / GPT-4o Image Generation (Non-Text)" : ("stima", "gpt-4o-image"),
    "OpenAI / Whisper-1 (Non-Text)"               : ("stima", "whisper-1"),
    "OpenAI / TTS-1 (Non-Text)"                   : ("stima", "tts-1"),
    "OpenAI / TTS-1 HD (Non-Text)"                : ("stima", "tts-1-hd"),
    "OpenAI / DALL·E 3 (Non-Text)"                : ("stima", "dall-e-3"),
    "OpenAI / text-embedding-3-small (Non-Text)"  : ("stima", "text-embedding-3-small"),
    "OpenAI / text-embedding-3-large (Non-Text)"  : ("stima", "text-embedding-3-large"),

    # =======================
    # ---- Anthropic ----
    # =======================
    # (沒有 Free，全是付費)
    "Anthropic / Claude Opus 4.1 (Thinking)" : ("stima", "claude-opus-4-1-20250805-thinking"),
    "Anthropic / Claude Opus 4.1"            : ("stima", "claude-opus-4-1-20250805"),
    "Anthropic / Claude Opus 4 (Thinking)"   : ("stima", "claude-opus-4-20250514-thinking"),
    "Anthropic / Claude Opus 4"              : ("stima", "claude-opus-4-20250514"),
    "Anthropic / Claude Sonnet 4 (Thinking)" : ("stima", "claude-sonnet-4-20250514-thinking"),
    "Anthropic / Claude Sonnet 4"            : ("stima", "claude-sonnet-4-20250514"),
    "Anthropic / Claude 3.7 Sonnet Thinking" : ("stima", "claude-3.7-sonnet-thinking"),
    "Anthropic / Claude 3.7 Sonnet"          : ("stima", "claude-3.7-sonnet"),
    "Anthropic / Claude 3.5 Haiku"           : ("stima", "claude-3-5-haiku-20241022"),
    "Anthropic / Claude 3.5 Sonnet"          : ("stima", "claude-3-5-sonnet-20241022"),
    "Anthropic / Claude 3 Sonnet"            : ("stima", "claude-3-sonnet-20240229"),
    "Anthropic / Claude 3 Haiku"             : ("stima", "claude-3-haiku-20240307"),
    "Anthropic / Claude 3 Opus"              : ("stima", "claude-3-opus-20240229"),

    # =======================
    # ---- Google ----
    # =======================
    # Free
    "Google / Gemini 2.5 Flash Image Preview (Free)" : ("stima", "gemini-2.5-flash-image-preview:free"),
    "Google / Gemini 2.5 Pro Experimental (Free)"    : ("stima", "gemini-2.5-pro-exp-03-25:free"),
    "Google / Gemini 2.0 Flash Experimental (Free)"  : ("stima", "gemini-2.0-flash-exp:free"),
    "Google / Gemma 3n 2B (Free)"                   : ("stima", "gemma-3n-e2b-it:free"),
    "Google / Gemma 3n E4B (Free)"                  : ("stima", "gemma-3n-e4b-it:free"),
    "Google / Gemma 3 12B (Free)"                   : ("stima", "gemma-3-12b-it:free"),
    "Google / Gemma 3 4B (Free)"                    : ("stima", "gemma-3-4b-it:free"),
    "Google / Gemma 3 27B (Free)"                   : ("stima", "gemma-3-27b-it:free"),
    "Google / Gemma 2 9B (Free)"                    : ("stima", "gemma-2-9b-it:free"),

    # Paid (Text)
    "Google / Gemini 2.5 Flash Lite"        : ("stima", "gemini-2.5-flash-lite"),
    "Google / Gemini 2.5 Flash"             : ("stima", "gemini-2.5-flash"),
    "Google / Gemini 2.5 Pro"               : ("stima", "gemini-2.5-pro"),
    "Google / Gemini 2.5 Pro DeepSearch"    : ("stima", "gemini-2.5-pro-deepsearch"),
    "Google / Gemini 2.5 Flash DeepSearch"  : ("stima", "gemini-2.5-flash-deepsearch"),
    "Google / Gemini 1.5 Pro"               : ("stima", "gemini-1.5-pro-latest"),
    "Google / Gemini 1.5 Flash"             : ("stima", "gemini-1.5-flash-latest"),

    # Non-Text
    "Google / Gemini 2.0 Flash Exp Image Generation (Non-Text)" : ("stima", "gemini-2.0-flash-exp-image"),
    "Google / Veo 3 Fast (Non-Text)"                            : ("stima", "veo3-fast"),
    "Google / Veo 3 (Non-Text)"                                 : ("stima", "veo3"),
    "Google / Veo 3 Pro (Non-Text)"                             : ("stima", "veo3-pro"),
    "Google / Veo 3 Pro Frames (Non-Text)"                      : ("stima", "veo3-pro-frames"),

    # =======================
    # ---- Meta (Llama) ----
    # =======================
    # Free
    "Meta / Llama 3.3 8B Instruct (Free)"     : ("stima", "llama-3.3-8b-instruct:free"),
    "Meta / Llama 4 Scout (Free)"             : ("stima", "llama-4-scout:free"),
    "Meta / Llama 4 Maverick (Free)"          : ("stima", "llama-4-maverick:free"),
    "Meta / Llama 3.3 70B Instruct (Free)"    : ("stima", "llama-3.3-70b-instruct:free"),
    "Meta / Llama 3.2 3B Instruct (Free)"     : ("stima", "llama-3.2-3b-instruct:free"),
    "Meta / Llama 3.2 1B Instruct (Free)"     : ("stima", "llama-3.2-1b-instruct:free"),
    "Meta / Llama 3.2 11B Vision Instruct (Free)" : ("stima", "llama-3.2-11b-vision-instruct:free"),
    "Meta / Llama 3.1 8B Instruct (Free)"     : ("stima", "llama-3.1-8b-instruct:free"),
    "Meta / Llama 3.3 Nemotron Super 49B (Free)" : ("stima", "llama-3.3-nemotron-super-49b-v1:free"),
    "Meta / Llama 3.1 Nemotron Ultra 253B (Free)" : ("stima", "llama-3.1-nemotron-ultra-253b-v1:free"),

    # Paid (Text)
    "Meta / Llama 4 Scout"            : ("stima", "llama-4-scout"),
    "Meta / Llama 4 Maverick"         : ("stima", "llama-4-maverick"),
    "Meta / Llama 3.3 70B Instruct"   : ("stima", "llama-3.3-70b-instruct"),
    "Meta / Llama 3.2 3B Instruct"    : ("stima", "llama-3.2-3b-instruct"),
    "Meta / Llama 3.2 11B Vision Instruct" : ("stima", "llama-3.2-11b-vision-instruct"),
    "Meta / Llama 3.1 8B Instruct"    : ("stima", "llama-3.1-8b-instruct"),
    "Meta / Llama 3.1 70B Instruct"   : ("stima", "llama-3.1-70b-instruct"),
    "Meta / Llama 3.1 405B"           : ("stima", "llama-3.1-405b"),
    "Meta / Llama 3.1 405B Instruct"  : ("stima", "llama-3.1-405b-instruct"),
    "Meta / Llama 3 70B"              : ("stima", "llama-3-70b"),
    "Meta / Llama 3 8B"               : ("stima", "llama-3-8b"),

    # =======================
    # ---- Microsoft ----
    # =======================
    # Free
    "Microsoft / Phi 4 Reasoning Plus (Free)" : ("stima", "phi-4-reasoning-plus:free"),
    "Microsoft / Phi 4 Reasoning (Free)"      : ("stima", "phi-4-reasoning:free"),
    "Microsoft / MAI DS R1 (Free)"            : ("stima", "mai-ds-r1:free"),

    # Paid (Text)
    "Microsoft / Phi 4 Reasoning Plus"    : ("stima", "phi-4-reasoning-plus"),
    "Microsoft / Phi-4"                   : ("stima", "phi-4"),
    "Microsoft / Phi-3.5 Mini 128K Instruct" : ("stima", "phi-3.5-mini-128k-instruct"),
    "Microsoft / Phi 4 Multimodal Instruct (Non-Text)" : ("stima", "phi-4-multimodal-instruct"),

    # =======================
    # ---- Mistral AI ----
    # =======================
    # Free
    "Mistral / Small 3.2 24B (Free)"      : ("stima", "mistral-small-3.2-24b-instruct:free"),
    "Mistral / Devstral Small (Free)"     : ("stima", "devstral-small:free"),
    "Mistral / Small 3.1 24B (Free)"      : ("stima", "mistral-small-3.1-24b-instruct:free"),
    "Mistral / Small 3 (Free)"            : ("stima", "mistral-small-24b-instruct-2501:free"),
    "Mistral / 7B Instruct (Free)"        : ("stima", "mistral-7b-instruct:free"),

    # Paid (Text)
    "Mistral / Medium 3.1"                : ("stima", "mistral-medium-3.1"),
    "Mistral / Codestral 2508"            : ("stima", "codestral-2508"),
    "Mistral / Devstral Medium"           : ("stima", "devstral-medium"),
    "Mistral / Devstral Small 1.1"        : ("stima", "devstral-small-2507"),
    "Mistral / Magistral Small 2506"      : ("stima", "magistral-small-2506"),
    "Mistral / Magistral Medium 2506"     : ("stima", "magistral-medium-2506"),
    "Mistral / Magistral Medium 2506 (Thinking)" : ("stima", "magistral-medium-2506:thinking"),
    "Mistral / Medium 3"                  : ("stima", "mistral-medium-3"),
    "Mistral / Small 3.1 24B"             : ("stima", "mistral-small-3.1-24b-instruct-2503"),
    "Mistral / Saba"                      : ("stima", "mistral-saba"),
    "Mistral / Small 3"                   : ("stima", "mistral-small-24b-instruct-2501"),
    "Mistral / Codestral 2501"            : ("stima", "codestral-2501"),
    "Mistral / Large 2411"                : ("stima", "mistral-large-2411"),
    "Mistral / Large 2407"                : ("stima", "mistral-large-2407"),
    "Mistral / Pixtral Large 2411"        : ("stima", "pixtral-large-2411"),
    "Mistral / Ministral 3B"              : ("stima", "ministral-3b"),
    "Mistral / Ministral 8B"              : ("stima", "ministral-8b"),
    "Mistral / Pixtral 12B"               : ("stima", "pixtral-12b"),
    "Mistral / Mixtral 8x7B Instruct"     : ("stima", "mixtral-8x7b-instruct"),
    "Mistral / 7B Instruct"               : ("stima", "mistral-7b-instruct"),
    "Mistral / 7B Instruct v0.3"          : ("stima", "mistral-7b-instruct-v0.3"),

    # =======================
    # ---- Alibaba / Qwen ----
    # =======================
    # Free
    "Alibaba / Qwen3 Coder (Free)"               : ("stima", "qwen3-coder:free"),
    "Alibaba / Qwen3 235B A22B 2507 (Free)"      : ("stima", "qwen3-235b-a22b-07-25:free"),
    "Alibaba / Qwen3 0.6B (Free)"                : ("stima", "qwen3-0.6b-04-28:free"),
    "Alibaba / Qwen3 1.7B (Free)"                : ("stima", "qwen3-1.7b:free"),
    "Alibaba / Qwen3 4B (Free)"                  : ("stima", "qwen3-4b:free"),
    "Alibaba / Qwen3 30B A3B (Free)"             : ("stima", "qwen3-30b-a3b:free"),
    "Alibaba / Qwen3 8B (Free)"                  : ("stima", "qwen3-8b:free"),
    "Alibaba / Qwen3 14B (Free)"                 : ("stima", "qwen3-14b:free"),
    "Alibaba / Qwen3 32B (Free)"                 : ("stima", "qwen3-32b:free"),
    "Alibaba / Qwen3 235B A22B (Free)"           : ("stima", "qwen3-235b-a22b:free"),
    "Alibaba / Qwen2.5 VL 32B Instruct (Free)"   : ("stima", "qwen2.5-vl-32b-instruct:free"),
    "Alibaba / Qwen2.5 VL 72B Instruct (Free)"   : ("stima", "qwen2.5-vl-72b-instruct:free"),
    "Alibaba / QwQ 32B Preview (Free)"           : ("stima", "qwq-32b-preview:free"),

    # Paid (Text)
    "Alibaba / Qwen3 Next 80B A3B Thinking"      : ("stima", "qwen3-next-80b-a3b-thinking"),
    "Alibaba / Qwen3 Next 80B A3B Instruct"      : ("stima", "qwen3-next-80b-a3b-instruct"),
    "Alibaba / Qwen Plus 2025-07-28"             : ("stima", "qwen-plus-2025-07-28"),
    "Alibaba / Qwen Plus 2025-07-28 Thinking"    : ("stima", "qwen-plus-2025-07-28:thinking"),
    "Alibaba / Qwen3 Max"                        : ("stima", "qwen3-max"),
    "Alibaba / Qwen3 30B A3B Think"              : ("stima", "qwen3-30b-a3b-think"),
    "Alibaba / Qwen3 30B A3B"                    : ("stima", "qwen3-30b-a3b"),
    "Alibaba / Qwen3 Coder 480B A35B Instruct"   : ("stima", "qwen3-coder-480b-a35b-instruct"),
    "Alibaba / Qwen3 Coder Plus"                 : ("stima", "qwen3-coder-plus"),
    "Alibaba / Qwen3 Coder Plus 2025-07-22"      : ("stima", "qwen3-coder-plus-2025-07-22"),
    "Alibaba / Qwen2.5 Coder 7B Instruct"        : ("stima", "qwen2.5-coder-7b-instruct"),
    "Alibaba / Qwen2.5 VL 72B Instruct"          : ("stima", "qwen2.5-vl-72b-instruct"),
    "Alibaba / Qwen2.5 VL 3B Instruct"           : ("stima", "qwen2.5-vl-3b-instruct"),
    "Alibaba / Qwen VL Max"                      : ("stima", "qwen-vl-max"),
    "Alibaba / Qwen2.5 VL 32B Instruct"          : ("stima", "qwen2.5-vl-32b-instruct"),
    "Alibaba / QwQ 32B"                          : ("stima", "qwq-32b"),
    "Alibaba / Qwen2.5 32B Instruct"             : ("stima", "qwen2.5-32b-instruct"),
    "Alibaba / Qwen VL Plus"                     : ("stima", "qwen-vl-plus"),
    "Alibaba / Qwen Max"                         : ("stima", "qwen-max"),
    "Alibaba / Qwen Plus"                        : ("stima", "qwen-plus"),
    "Alibaba / Qwen Turbo"                       : ("stima", "qwen-turbo"),
    "Alibaba / Qwen2.5 Coder 32B Instruct"       : ("stima", "qwen-2.5-coder-32b-instruct"),
    "Alibaba / Qwen2-VL 72B Instruct"            : ("stima", "qwen-2-vl-72b-instruct"),
    "Alibaba / Qwen2.5 72B Instruct"             : ("stima", "qwen-2.5-72b-instruct"),

    # =======================
    # ---- NVIDIA ----
    # =======================
    # Free
    "NVIDIA / Nemotron Nano 9B V2 (Free)"          : ("stima", "nemotron-nano-9b-v2"),
    "NVIDIA / Llama 3.3 Nemotron Super 49B (Free)" : ("stima", "llama-3.3-nemotron-super-49b-v1:free"),
    "NVIDIA / Llama 3.1 Nemotron Ultra 253B (Free)": ("stima", "llama-3.1-nemotron-ultra-253b-v1:free"),

    # Paid (Text)
    "NVIDIA / Llama 3.3 Nemotron Super 49B"        : ("stima", "llama-3.3-nemotron-super-49b-v1"),
    "NVIDIA / Llama 3.1 Nemotron 70B Instruct"     : ("stima", "llama-3.1-nemotron-70b-instruct"),

    # =======================
    # ---- xAI (Grok) ----
    # =======================
    # (沒有 Free，全是付費)
    "xAI / Grok Code Fast 1"       : ("stima", "grok-code-fast-1"),
    "xAI / Grok 4"                 : ("stima", "grok-4"),
    "xAI / Grok 3 Beta"            : ("stima", "grok-3-beta"),
    "xAI / Grok 3 Mini Beta"       : ("stima", "grok-3-mini-beta"),
    "xAI / Grok 3 Search"          : ("stima", "grok-3-search"),
    "xAI / Grok 3 DeepSearch"      : ("stima", "grok-3-deepsearch"),
    "xAI / Grok 3 Reasoning"       : ("stima", "grok-3-reasoning"),
    "xAI / Grok 3"                 : ("stima", "grok-3"),
    "xAI / Grok 2 Vision 1212 (Non-Text)" : ("stima", "grok-2-vision-1212"),
    "xAI / Grok Vision Beta (Non-Text)"   : ("stima", "grok-vision-beta"),
    "xAI / Grok Beta"              : ("stima", "grok-beta"),
    "xAI / Grok 3 Image Generation (Non-Text)" : ("stima", "grok-3-image"),

    # =======================
    # ---- Amazon ----
    # =======================
    # (沒有 Free，全是付費)
    "Amazon / Nova Lite 1.0"  : ("stima", "nova-lite-v1"),
    "Amazon / Nova Micro 1.0" : ("stima", "nova-micro-v1"),
    "Amazon / Nova Pro 1.0"   : ("stima", "nova-pro-v1"),

    # =======================
    # ---- DeepSeek ----
    # =======================
    # Free
    "DeepSeek / R1 0528 Qwen3 8B (Free)" : ("stima", "deepseek-r1-0528-qwen3-8b:free"),
    "DeepSeek / R1 0528 (Free)"          : ("stima", "deepseek-r1-0528:free"),
    "DeepSeek / Prover V2 (Free)"        : ("stima", "deepseek-prover-v2:free"),
    "DeepSeek / V3 Base (Free)"          : ("stima", "deepseek-v3-base"),
    "DeepSeek / R1 Zero (Free)"          : ("stima", "deepseek-r1-zero"),
    "DeepSeek / R1 Distill Qwen 32B (Free)" : ("stima", "deepseek-r1-distill-qwen-32b:free"),
    "DeepSeek / R1 Distill Qwen 14B (Free)" : ("stima", "deepseek-r1-distill-qwen-14b:free"),
    "DeepSeek / R1 Distill Llama 70B (Free)": ("stima", "deepseek-r1-distill-llama-70b:free"),
    "DeepSeek / R1 (Free)"               : ("stima", "deepseek-r1:free"),
    "DeepSeek / V3 (Free)"               : ("stima", "deepseek-v3:free"),

    # Paid (Text)
    "DeepSeek / V3.1"             : ("stima", "deepseek-chat-v3.1"),
    "DeepSeek / V3.1 Base"        : ("stima", "deepseek-v3.1-base"),
    "DeepSeek / R1 Distill Qwen 7B" : ("stima", "deepseek-r1-distill-qwen-7b"),
    "DeepSeek / R1 0528 Qwen3 8B" : ("stima", "deepseek-r1-0528-qwen3-8b"),
    "DeepSeek / Prover V2"        : ("stima", "deepseek-prover-v2"),
    "DeepSeek / V3 Search"        : ("stima", "deepseek-v3-search"),
    "DeepSeek / V3 0324"          : ("stima", "deepseek-chat-v3-0324"),
    "DeepSeek / R1 Searching"     : ("stima", "deepseek-r1-searching"),
    "DeepSeek / R1 Distill Llama 8B" : ("stima", "deepseek-r1-distill-llama-8b"),
    "DeepSeek / R1 Distill Qwen 1.5B" : ("stima", "deepseek-r1-distill-qwen-1.5b"),

    # =======================
    # ---- Perplexity ----
    # =======================
    # (沒有 Free，全是付費)
    "Perplexity / Sonar Reasoning Pro" : ("stima", "sonar-reasoning-pro"),
    "Perplexity / Sonar Pro"           : ("stima", "sonar-pro"),
    "Perplexity / Sonar Deep Research" : ("stima", "sonar-deep-research"),
    "Perplexity / R1 1776"             : ("stima", "perplexity-r1-1776"),
    "Perplexity / Sonar Reasoning"     : ("stima", "sonar-reasoning"),
    "Perplexity / Sonar"               : ("stima", "sonar"),
    "Perplexity / Llama 3.1 Sonar 70B Online" : ("stima", "llama-3.1-sonar-large-128k-online"),
    "Perplexity / Llama 3.1 Sonar 8B Online"  : ("stima", "llama-3.1-sonar-small-128k-online"),

    # =======================
    # ---- Moonshot ----
    # =======================
    # Free
    "Moonshot / Kimi K2 (Free)"        : ("stima", "kimi-k2:free"),
    "Moonshot / Kimi Dev 72B (Free)"   : ("stima", "kimi-dev-72b:free"),
    "Moonshot / Kimi VL A3B Thinking (Free)" : ("stima", "kimi-vl-a3b-thinking:free"),
    "Moonshot / Moonlight 16B A3B Instruct (Free)" : ("stima", "moonlight-16b-a3b-instruct"),

    # Paid (Text)
    "Moonshot / Kimi K2 0905"          : ("stima", "kimi-k2-0905"),
    "Moonshot / Kimi K2 0711 Preview Search" : ("stima", "kimi-k2-0711-preview-search"),
    "Moonshot / Kimi K2"               : ("stima", "kimi-k2"),
    "Moonshot / Moonshot V1 128k"      : ("stima", "moonshot-v1-128k"),
    "Moonshot / Moonshot V1 32k"       : ("stima", "moonshot-v1-32k"),
    "Moonshot / Moonshot V1 8k"        : ("stima", "moonshot-v1-8k"),

    # =======================
    # ---- Cohere ----
    # =======================
    # (沒有 Free，全是付費)
    "Cohere / Command A"         : ("stima", "command-a"),
    "Cohere / Command R7B"       : ("stima", "command-r7b-12-2024"),
    "Cohere / Command R"         : ("stima", "command-r-08-2024"),
    "Cohere / Command R+"        : ("stima", "command-r-plus-08-2024"),
    "Cohere / Command"           : ("stima", "command"),
    "Cohere / Command R+"        : ("stima", "command-r-plus"),
    "Cohere / Command R (03-2024)" : ("stima", "command-r-03-2024"),

    # =======================
    # ---- Baidu ----
    # =======================
    # (沒有 Free，全是付費)
    "Baidu / ERNIE X1 32K"              : ("stima", "ernie-x1-32k"),
    "Baidu / ERNIE X1 Turbo 32K"        : ("stima", "ernie-x1-turbo-32k"),
    "Baidu / ERNIE 4.5 0.3B A3B"        : ("stima", "ernie-4.5-0.3b-a3b"),
    "Baidu / ERNIE 4.5 21B A3B"         : ("stima", "ernie-4.5-21b-a3b"),
    "Baidu / ERNIE 4.5 Turbo 128K"      : ("stima", "ernie-4.5-turbo-128k"),
    "Baidu / ERNIE 4.5 Turbo 32K"       : ("stima", "ernie-4.5-turbo-32k"),
    "Baidu / ERNIE 4.5 Turbo VL 32K"    : ("stima", "ernie-4.5-turbo-vl-32k"),
    "Baidu / ERNIE 4.5 VL 28B A3B"      : ("stima", "ernie-4.5-vl-28b-a3b"),
    "Baidu / ERNIE 4.5 300B A47B"       : ("stima", "ernie-4.5-300b-a47b"),
    "Baidu / ERNIE 4.0 8K"              : ("stima", "ERNIE-4.0-8K"),
    "Baidu / ERNIE Lite 8K 0308"        : ("stima", "ERNIE-Lite-8K-0308"),
    "Baidu / ERNIE Lite 8K 0922"        : ("stima", "ERNIE-Lite-8K-0922"),
    "Baidu / ERNIE Speed 128K"          : ("stima", "ERNIE-Speed-128K"),
    "Baidu / ERNIE Speed 8K"            : ("stima", "ERNIE-Speed-8K"),
    "Baidu / ERNIE 3.5 8K"              : ("stima", "ERNIE-3.5-8K"),

    # =======================
    # ---- Open R1 ----
    # =======================
    "OpenR1 / OlympicCoder 32B (Free)" : ("stima", "olympiccoder-32b"),

    # =======================
    # ---- Z.AI (GLM) ----
    # =======================
    # Free
    "Z.AI / GLM-4.5 Air (Free)" : ("stima", "glm-4.5-air:free"),

    # Paid (Text)
    "Z.AI / GLM-4.5 X"          : ("stima", "glm-4.5-x"),
    "Z.AI / GLM-4.5 Flash"      : ("stima", "glm-4.5-flash"),
    "Z.AI / GLM-4.5 AirX"       : ("stima", "glm-4.5-airx"),
    "Z.AI / GLM-4.5 Air"        : ("stima", "glm-4.5-air"),
    "Z.AI / GLM-4.5"            : ("stima", "glm-4.5"),
    "Z.AI / GLM-4"              : ("stima", "glm-4"),

    # =======================
    # ---- OpenRouter ----
    # =======================
    "OpenRouter / Sonoma Sky Alpha (Free)"  : ("stima", "sonoma-sky-alpha"),
    "OpenRouter / Sonoma Dusk Alpha (Free)" : ("stima", "sonoma-dusk-alpha"),

    # =======================
    # ---- Nano Banana ----
    # =======================
    "NanoBanana / Nano Banana (Non-Text)" : ("stima", "fal-ai/nano-banana"),

}
