<div align="center">

# complex-chat

A chatbot that can input a question and get answers from three models

[![Built with Stima API](https://img.shields.io/badge/Built%20with-Stima%20API-blueviolet?logo=robot)](https://api.stima.tech)
[![Hugging Face Space](https://img.shields.io/badge/Space-%20demo-yellow?logo=huggingface)](https://huggingface.co/spaces/ELECAR00/complex-chat-demo)

</div>

## 特點

1. 依照自己的選擇，選取三種不同/相同的聊天模型，輸入一次獲得三種結果
2. 無痛比較不同模型差別，節省時間
3. 支援即時切換，快速測試不同模型的優勢與限制
4. 簡單直觀介面，無需額外設定即可上手

## 程式介紹

### 一、依照使用習慣於本地或網頁端進行

程式支援 **本地端運行** 與 **線上 Demo** 兩種方式：  
- **本地端**：只要安裝依賴、設定 API key，即可在自己的電腦上運行，方便開發者自訂與修改。  
- **HuggingFace Space**：直接透過瀏覽器進入 Demo 頁面試玩，無需安裝任何東西，體驗快速輕鬆，也能設定自己的 API。  

整體介面基於 [Gradio](https://gradio.app) 開發，設計簡單直觀，即使沒有程式背景的人也能輕鬆使用。  


### 二、自由串接不同廠商 API

此專案並不綁定特定 API，可以自由選擇不同的模型供應商：  
- **Stima API**：目前的主要串接對象，支援多種模型，適合初學者與開發者
- **OpenRouter**：先前使用的多模型聚合平台，方便比較不同廠商的回覆
- **其他 API**：只要提供 OpenAI 相容的介面，即可快速接入

這樣的設計讓使用者能根據需求 **快速切換不同模型**，不論是字句比較、性能測試，或是想單純體驗不同 AI 的風格，都能一次完成。  

（目前正配合 **Stima 推廣計畫**，所以這個版本以 Stima API 為主要串接對象，順便測試看看它的使用體驗 XD）  


### 三、適用情境

- 開發者：快速比較不同 LLM 的表現
- 研究人員：於論文改寫、字句編譯時，可以一次比較多種模型輸出結果
- 教育用途：讓學生理解不同模型的思路差異
- 產品開發：在早期選擇模型時，能快速測試多種模型


## 🔮 未來計畫 / TODO

- [ ] **支援更多模型**：不限於三個，可自由擴充至四個或更多  
- [ ] **多語言介面**：支援切換中文、英文等多語言介面  
- [ ] **輸出格式**：支援表格化輸出、JSON API 介接  
- [ ] **對話歷史與記憶**：可保存上下文，模擬更真實的聊天  
- [ ] **整合更多應用**：結合翻譯、摘要、問答等功能  

---
