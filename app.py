import os
import asyncio
import gradio as gr
import openai
from dotenv import load_dotenv
import httpx


# ---------- 0. 讀環境變數 ----------
load_dotenv()
"""
OR_KEY = os.getenv("OPENROUTER_API_KEY")
OA_KEY = os.getenv("OPENAI_API_KEY")
PP_KEY = os.getenv("PERPLEXITY_API_KEY")
"""
STIMA_KEY = os.getenv("STIMA_API_KEY")


# ---------- 1. 模型對照表 ----------
# 使用者下拉看到的名稱  →  (provider, 真正的 API 模型 ID)
"""
MODELS = {
    "DeepSeek-R1"               : ("openrouter", "deepseek/deepseek-r1:free"),
    "Gemini 2.0 Flash"          : ("openrouter", "google/gemini-2.0-flash-exp:free"),
    "Qwen2.5-VL-72B"            : ("openrouter", "qwen/qwen2.5-vl-72b-instruct:free"),
    "Mistral-Small-24B"         : ("openrouter", "mistralai/mistral-small-24b-instruct-2501:free"),
    "Dolphin-3.0-Mistral-24B"   : ("openrouter", "cognitivecomputations/dolphin3.0-r1-mistral-24b:free"),
    "Llama-3-70B-Instruct"      : ("openrouter", "meta-llama/llama-3.3-70b-instruct:free"),
    "GPT-4o-mini"               : ("openai",     "gpt-4o-mini"),
    "Sonar-Deep-Research"       : ("perplexity", "sonar-deep-research"),
    "Sonar-Reasoning-Pro"       : ("perplexity", "sonar-reasoning-pro"),
    "Sonar-Reasoning"           : ("perplexity", "sonar-reasoning"),
    "Sonar-Pro"                 : ("perplexity", "sonar-pro"),
    "Sonar"                     : ("perplexity", "sonar"),
    "r1-1776"                   : ("perplexity", "r1-1776")
}
"""

MODELS = {
    # ---- Qwen ----
    "Qwen / 3 Coder (Free)"            : ("stima", "qwen3-coder:free"),
    "Qwen / 3 235B A22B 2507 (Free)"   : ("stima", "qwen3-235b-a22b-07-25:free"),
    "Qwen / 3 0.6B (Free)"             : ("stima", "qwen3-0.6b-04-28:free"),
    "Qwen / 3 1.7B (Free)"             : ("stima", "qwen3-1.7b:free"),
    "Qwen / 3 4B (Free)"               : ("stima", "qwen3-4b:free"),
    "Qwen / 3 8B (Free)"               : ("stima", "qwen3-8b:free"),
    "Qwen / 3 14B (Free)"              : ("stima", "qwen3-14b:free"),
    "Qwen / 3 32B (Free)"              : ("stima", "qwen3-32b:free"),
    "Qwen / 3 30B A3B (Free)"          : ("stima", "qwen3-30b-a3b:free"),
    "Qwen / 3 235B A22B (Free)"        : ("stima", "qwen3-235b-a22b:free"),
    "Qwen / 2.5 VL 32B Instruct (Free)": ("stima", "qwen2.5-vl-32b-instruct:free"),
    "Qwen / 2.5 VL 72B Instruct (Free)": ("stima", "qwen2.5-vl-72b-instruct:free"),

    # ---- DeepSeek ----
    "DeepSeek / R1 (Free)"             : ("stima", "deepseek-r1:free"),
    "DeepSeek / R1 0528 (Free)"        : ("stima", "deepseek-r1-0528:free"),
    "DeepSeek / R1 0528 Qwen3 8B (Free)":("stima", "deepseek-r1-0528-qwen3-8b:free"),
    "DeepSeek / R1 Distill Qwen 14B (Free)" : ("stima", "deepseek-r1-distill-qwen-14b:free"),
    "DeepSeek / R1 Distill Qwen 32B (Free)" : ("stima", "deepseek-r1-distill-qwen-32b:free"),
    "DeepSeek / R1 Distill Llama 70B (Free)":("stima", "deepseek-r1-distill-llama-70b:free"),
    "DeepSeek / V3 (Free)"             : ("stima", "deepseek-v3:free"),
    "DeepSeek / Prover V2 (Free)"      : ("stima", "deepseek-prover-v2:free"),

    # ---- Llama ----
    "Llama / 3.1 8B Instruct (Free)"   : ("stima", "llama-3.1-8b-instruct:free"),
    "Llama / 3.2 1B Instruct (Free)"   : ("stima", "llama-3.2-1b-instruct:free"),
    "Llama / 3.2 3B Instruct (Free)"   : ("stima", "llama-3.2-3b-instruct:free"),
    "Llama / 3.2 11B Vision Instruct (Free)": ("stima", "llama-3.2-11b-vision-instruct:free"),
    "Llama / 3.3 8B Instruct (Free)"   : ("stima", "llama-3.3-8b-instruct:free"),
    "Llama / 3.3 70B Instruct (Free)"  : ("stima", "llama-3.3-70b-instruct:free"),
    "Llama / 3.3 Nemotron Super 49B (Free)": ("stima", "llama-3.3-nemotron-super-49b-v1:free"),
    "Llama / 3.1 Nemotron Ultra 253B (Free)":("stima", "llama-3.1-nemotron-ultra-253b-v1:free"),
    "Llama / 4 Scout (Free)"           : ("stima", "llama-4-scout:free"),
    "Llama / 4 Maverick (Free)"        : ("stima", "llama-4-maverick:free"),

    # ---- Mistral ----
    "Mistral / 7B Instruct (Free)"     : ("stima", "mistral-7b-instruct:free"),
    "Mistral / Small 3 (Free)"         : ("stima", "mistral-small-24b-instruct-2501:free"),
    "Mistral / Small 3.1 24B (Free)"   : ("stima", "mistral-small-3.1-24b-instruct:free"),
    "Mistral / Small 3.2 24B (Free)"   : ("stima", "mistral-small-3.2-24b-instruct:free"),

    # ---- Gemini ----
    "Gemini / 2.0 Flash Experimental (Free)": ("stima", "gemini-2.0-flash-exp:free"),
    "Gemini / 2.5 Flash Image Preview (Free)":("stima", "gemini-2.5-flash-image-preview:free"),
    "Gemini / 2.5 Pro Experimental (Free)"   :("stima", "gemini-2.5-pro-exp-03-25:free"),

    # ---- Gemma ----
    "Gemma / 2 9B (Free)"              : ("stima", "gemma-2-9b-it:free"),
    "Gemma / 3n 2B (Free)"             : ("stima", "gemma-3n-e2b-it:free"),
    "Gemma / 3n E4B (Free)"            : ("stima", "gemma-3n-e4b-it:free"),
    "Gemma / 3 4B (Free)"              : ("stima", "gemma-3-4b-it:free"),
    "Gemma / 3 12B (Free)"             : ("stima", "gemma-3-12b-it:free"),
    "Gemma / 3 27B (Free)"             : ("stima", "gemma-3-27b-it:free"),

    # ---- 其他常見 ----
    "OpenAI / GPT-4o-mini"             : ("openai", "gpt-4o-mini"),
    "OpenAI / GPT-4o"                  : ("openai", "gpt-4o"),
    "Anthropic / Claude 3.5 Sonnet"    : ("openrouter", "anthropic/claude-3.5-sonnet"),
    "Anthropic / Claude 3.5 Haiku"     : ("openrouter", "anthropic/claude-3.5-haiku"),
    "Google / Gemini 1.5 Pro"          : ("openrouter", "google/gemini-1.5-pro"),
    "Meta / Llama 3.1 70B Instruct"    : ("openrouter", "meta-llama/llama-3.1-70b-instruct"),
    "Mistral / Large 2"                : ("openrouter", "mistralai/mistral-large-2407"),
}


# ---------- 2. 建立對應 Provider 的 Async 客戶端 ----------
def get_client(provider):
    if provider == "openai":
        return openai.AsyncOpenAI(api_key=OA_KEY)    # 使用官方端點, 不用改 base_url
    elif provider == "openrouter":
        return openai.AsyncOpenAI(
            api_key=OR_KEY,
            base_url="https://openrouter.ai/api/v1"    # 其他全部走 OpenRouter
        )
    elif provider == "stima":
        return openai.AsyncOpenAI(
            api_key=STIMA_KEY,
            base_url="https://api.stima.tech/v1"
        )
    elif provider == "perplexity":
        return httpx.AsyncClient(
            base_url="https://api.perplexity.ai",
            headers={"Authorization": f"Bearer {PP_KEY}"}
        )

# ---------- 3. 呼叫一次模型做段落字句改寫 ----------
async def rewrite_once(model_key, text, system_prompt, temp):
    provider, full_id = MODELS[model_key]
    client  = get_client(provider)
    if provider in ["openai", "openrouter"]:
        resp = await client.chat.completions.create(
            model=full_id,
            messages=[
                {"role": "system", "content": system_prompt or
                    "You are a researcher."},
                {"role": "user", "content": text}
            ],
            temperature = temp,    # 用 UI 傳進來的溫度
        )
        return resp.choices[0].message.content.strip()
    elif provider == "perplexity":
        # 假設 Perplexity Chat API 長這樣，實際依官方調整
        payload = {
            "model": full_id,
            "messages": [
                {"role": "system", "content": system_prompt or
                    "You are a researcher."},
                {"role": "user", "content": text}
            ],
            "temperature": temp
        }
        # Perplexity 回傳 JSON，格式也許是 { "choices": [ { "message": { "content": "..." } } ] }
        r = await client.post(
            "/chat/completions", 
            json=payload,
            timeout=httpx.Timeout(600.0, connect=10.0)
        )
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()

    else:
        raise ValueError(f"Unknown provider: {provider}")

# ---------- 4. 三模型並行改寫 ----------
def rewrite_batch(text, model1, model2, model3, sys_prompt, temp):
    async def _run():
        return await asyncio.gather(    # gather 讓三個 coroutines 同時執行
            rewrite_once(model1, text, sys_prompt, temp),
            rewrite_once(model2, text, sys_prompt, temp),
            rewrite_once(model3, text, sys_prompt, temp),
        )
    return asyncio.run(_run())    # 把 async 結果變同步 (Gradio 需要)

# ---------- 5. Gradio 介面 ----------
model_list = list(MODELS.keys())    # 給 Dropdown 用

with gr.Blocks(theme=gr.themes.Soft(), title="Chat-API") as demo:
    gr.Markdown("### 📝 一次比較三種模型的輸出結果")

    src = gr.Textbox(label="輸入", lines=8, placeholder="貼上或輸入要改寫的內容")
    sys_prompt = gr.Textbox(label="自訂系統提示 (可空白)", placeholder="例：保持專業且具有科技感的語氣")

    temp = gr.Slider(0.0, 1.0, value=0.7,
                     step=0.05, label="Temperature")    # 新增溫度滑桿：0.0 ~ 1.0，預設 0.7
    
    with gr.Row():
        dd1 = gr.Dropdown(model_list, value=model_list[0],  label="模型 1")    # 預設 DeepSeek-R1
        dd2 = gr.Dropdown(model_list, value=model_list[1],  label="模型 2")    # 預設 Gemini 2.0 Flash
        dd3 = gr.Dropdown(model_list, value=model_list[12], label="模型 3")    # 預設 GPT-4o-mini

    btn = gr.Button("🌟查詢")

    with gr.Row():
        out1 = gr.Textbox(label="模型 1 輸出", lines=20)
        out2 = gr.Textbox(label="模型 2 輸出", lines=20)
        out3 = gr.Textbox(label="模型 3 輸出", lines=20)

    btn.click(
        rewrite_batch,
        inputs=[src, dd1, dd2, dd3, sys_prompt, temp],
        outputs=[out1, out2, out3]
    )

if __name__ == "__main__":
    demo.launch()
