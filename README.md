# 📖 Biblical Companion AI - Your AI Guide to Christian Faith 🕊️

This project provides an interactive AI experience for engaging with the Bible and Christian spirituality from Catholic, Orthodox Christian, and Protestant traditions. It leverages large language models and LangChain, with RAG and a QDrant vector database, to offer personalized guidance, prayer support, virtual confession (for Catholic and Orthodox users), and tailored meditation scripts based on user input and the specific Christian perspective chosen.

The idea for this project arose from the desire to create an accessible and informative tool for individuals seeking deeper understanding, comfort, and spiritual growth within their chosen Christian faith. By grounding responses in biblical text and adhering to the distinct values of each tradition, Biblical Companion AI aims to offer a respectful and insightful experience.

## Overview 🧐

The core functionality of this project includes:

* **Tradition-Specific Guidance:** Offers distinct conversational agents tailored to Catholicism, Orthodox Christianity, and Protestantism.
* **Contextual Bible Coaching:** Allows users to ask questions about scripture and receive explanations, interpretations, and practical advice aligned with the chosen tradition.
* **Personalized Prayer Assistance:** Provides comforting, tailored prayers and relevant Bible verses in response to user needs.
* **Virtual Confession (Catholic & Orthodox):** Enables users to engage in a virtual confession, receiving compassionate guidance, relevant scripture, spiritual advice, and a suggested penance.
* **Emotionally Adaptive Meditation Generation:** Creates tailored meditation scripts designed for relaxation, spiritual growth, and connection with God within each Christian tradition.
* **Contextual Awareness:** Maintains a short-term memory of the conversation to provide more relevant and coherent responses.

## Technologies Used 💻

* **Python**
* **Langchain**
    * **VertexAI**
    * **HuggingFace Embeddings**
    * **Qdrant**
    * **Prompt Engineering**
    * **ConversationBufferWindowMemory**

## Key Features ✨

* **Faith-Based Perspectives:** Offers responses aligned with Catholic, Orthodox, and Protestant teachings.
* **Contextual Scriptural Grounding:** Utilizes relevant Bible passages to inform responses.
* **Personalized Interactions:** Tailors guidance, prayers, and meditations to user input.
* **Respectful and Supportive Tone:** Maintains a compassionate and understanding approach.
* **Structured Responses:** Provides clear and organized answers with relevant biblical references and practical advice (where applicable).

## Potential Improvements ✨

* **Text-to-Speech integration.**
* **User Interface:** Create a more user-friendly interface (e.g., using a web framework like Flask or FastAPI).
* **Expanded Knowledge Base:** Incorporate additional theological resources specific to each tradition.
* **More Advanced Personalization:** Allow users to specify denominations or specific theological viewpoints within the broader traditions.
