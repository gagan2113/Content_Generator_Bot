# 📜 Content_Generator_Bot

A **Streamlit + LangGraph (Groq API)** application to quickly generate engaging social-media scripts. The app allows users to specify **Topic**, **Duration**, **Tone**, and **Platform**, then automatically creates a ready-to-use script along with a caption and hashtags. It can also store the generated scripts as nodes/edges in a **LangGraph** database using the **Groq API**.

---

## 🚀 Features

* **Social Media Script Generator**: Enter a topic, duration (seconds/minutes), tone, and optional platform to create:

  * Final script (ready to record or post)
  * Suggested caption
  * Relevant hashtags

* **Groq LangGraph Integration**: Automatically saves generated content into a knowledge graph as nodes (topic, script) and edges (relationships).

* **Customizable Output**: Choose between:

  * **Detailed Outline** mode: Hook, main points, call-to-action.
  * **Simple Script** mode: Direct, concise script with caption and hashtags.

* **Streamlit UI**: User-friendly interface to input content details and view generated output.

---

## 🏗️ Project Structure

```
Content_Generator_Bot/
├── LangGraph_Groq_Streamlit_App.py   # Main Streamlit application
├── README.md                         # Project documentation
├── requirements.txt                   # Python dependencies
└── ... (optional config files)
```

---

## ⚙️ Installation

1️⃣ **Clone the repository**

```bash
git clone https://github.com/yourusername/Content_Generator_Bot.git
cd Content_Generator_Bot
```

2️⃣ **Create a virtual environment (recommended)**

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

3️⃣ **Install dependencies**

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the root directory and add:

```
GROQ_API_KEY=your_groq_api_key
GROQ_PROJECT=your_groq_project_id
LLM_API_KEY=your_preferred_llm_key   # Optional if using external LLMs
```

---

## ▶️ Usage

Run the Streamlit app:

```bash
streamlit run LangGraph_Groq_Streamlit_App.py
```

Open your browser at `http://localhost:8501` to access the UI.

---

## 💡 How It Works

1. **User Input**: Topic, duration, tone, and platform.
2. **Script Generation**: A deterministic template (or optional LLM call) creates a short social-media script, caption, and hashtags.
3. **LangGraph Storage**: If enabled, the app calls the Groq API to create nodes and edges linking the topic to the generated script.
4. **Output Display**: Script and hashtags are shown in a clean Streamlit interface.

---

## 🔄 Modes of Operation

* **Detailed Outline Mode**: Generates a full outline with intro, main points, and call-to-action.
* **Simple Final Script Mode**: Outputs a direct, ready-to-use script with caption and hashtags.

Toggle modes by changing the prompt variable in `LangGraph_Groq_Streamlit_App.py`.

---

## 🛠️ Tech Stack

* **Python 3.9+**
* **Streamlit**: Front-end UI
* **Groq API**: LangGraph storage of nodes and edges
* **Optional LLM**: Replace the deterministic script builder with your favorite LLM (e.g., OpenAI, Anthropic) for more creativity.

---

## 📌 Example Prompt for Simple Script

```python
simple_script_prompt = f"""
Write a short and simple {state['duration']}-second social media video script in {state['language']}
about "{state['topic']}".
Tone: {state['tone']}
Platform: {state.get('platform', 'Any')}

Keep it conversational, energetic, and easy to understand.
Just give:
1. The final script (no outlines or timing notes)
2. One engaging caption
3. 5-6 relevant hashtags
"""
```

---

## 🧩 Customization

* **Replace pseudo-functions** `groq_create_node` and `groq_create_edge` with real API calls to the Groq LangGraph endpoint.
* Integrate your own **LLM provider** to enhance creativity.
* Adjust the **speaking rate** or **word count** estimation in `build_script()`.

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

