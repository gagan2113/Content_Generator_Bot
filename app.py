import streamlit as st
import requests
from typing import TypedDict, Dict, Any
from langgraph.graph import StateGraph, END
from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

# --- Groq API Configuration ---
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- LangGraph State Definition ---
class ScriptState(TypedDict):
    topic: str
    duration: int
    tone: str
    platform: str
    language: str
    script_outline: str
    final_script: str
    hashtags: str
    error: str

# --- Groq API Helper Function ---
def call_groq_api(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1024,
        "temperature": 0.8
    }
    response = requests.post(GROQ_API_URL, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"].strip()
    else:
        return f"Error: {response.status_code} - {response.text}"

# --- LangGraph Nodes ---
def create_outline_node(state: ScriptState) -> ScriptState:
    """Create a script outline based on user inputs"""
    outline_prompt = f"""
    Create a detailed outline for a {state['duration']}-second social media script about "{state['topic']}" in {state['language']}.
    Tone: {state['tone']}
    Platform: {state.get('platform', 'Any')}
    
    Provide a structured outline with:
    1. Hook/Intro (first 5-10 seconds)
    2. Main content points (middle section)
    3. Call-to-action (last 5-10 seconds)
    
    Consider the platform's audience and format requirements.
    """
    
    try:
        outline = call_groq_api(outline_prompt)
        state["script_outline"] = outline
    except Exception as e:
        state["error"] = f"Error creating outline: {str(e)}"
    
    return state

def generate_script_node(state: ScriptState) -> ScriptState:
    """Generate the full script based on the outline"""
    if state.get("error"):
        return state
        
    script_prompt = f"""
    Based on this outline: {state['script_outline']}
    
    Write a complete {state['duration']}-second social media script about "{state['topic']}" in {state['language']}.
    Tone: {state['tone']}
    Platform: {state.get('platform', 'Any')}
    
    Make it engaging, conversational, and appropriate for the platform.
    Include timing cues and natural pauses.
    Ensure it fits within {state['duration']} seconds when spoken at normal pace.
    """
    
    try:
        script = call_groq_api(script_prompt)
        state["final_script"] = script
    except Exception as e:
        state["error"] = f"Error generating script: {str(e)}"
    
    return state

def generate_hashtags_node(state: ScriptState) -> ScriptState:
    """Generate relevant hashtags and captions"""
    if state.get("error"):
        return state
        
    hashtag_prompt = f"""
    Based on this script: {state['final_script']}
    
    Generate:
    1. 8-10 relevant hashtags for {state.get('platform', 'social media')}
    2. A compelling caption/description
    
    Topic: {state['topic']}
    Platform: {state.get('platform', 'Any')}
    Language: {state['language']}
    
    Make hashtags trending and platform-appropriate.
    """
    
    try:
        hashtags = call_groq_api(hashtag_prompt)
        state["hashtags"] = hashtags
    except Exception as e:
        state["error"] = f"Error generating hashtags: {str(e)}"
    
    return state

# --- LangGraph Workflow ---
def create_script_workflow():
    workflow = StateGraph(ScriptState)
    
    # Add nodes
    workflow.add_node("create_outline", create_outline_node)
    workflow.add_node("generate_script", generate_script_node)
    workflow.add_node("generate_hashtags", generate_hashtags_node)
    
    # Define edges
    workflow.set_entry_point("create_outline")
    workflow.add_edge("create_outline", "generate_script")
    workflow.add_edge("generate_script", "generate_hashtags")
    workflow.add_edge("generate_hashtags", END)
    
    return workflow.compile()

# --- Main Generation Function ---
def generate_script(topic, duration, tone, platform, language):
    app = create_script_workflow()
    
    initial_state = ScriptState(
        topic=topic,
        duration=duration,
        tone=tone,
        platform=platform,
        language=language,
        script_outline="",
        final_script="",
        hashtags="",
        error=""
    )
    
    try:
        result = app.invoke(initial_state)
        
        if result.get("error"):
            return f"❌ {result['error']}"
        
        # Format the complete response
        response = f"""
## 📝 Script Outline
{result.get('script_outline', 'No outline generated')}

## 🎬 Final Script
{result.get('final_script', 'No script generated')}

## 🏷️ Hashtags & Caption
{result.get('hashtags', 'No hashtags generated')}
        """
        
        return response.strip()
        
    except Exception as e:
        return f"❌ Workflow error: {str(e)}"

# --- Streamlit UI ---
st.set_page_config(page_title="AI Social Media Script Generator", page_icon="📝")
st.title("AI Social Media Script Generator")
st.write("Generate compelling, platform-specific social media scripts in seconds!")

with st.form("script_form"):
    topic = st.text_input("Topic", help="What is the main topic or subject?")
    duration = st.number_input("Duration (seconds)", min_value=10, max_value=600, value=60, step=10)
    tone = st.selectbox("Tone", ["Friendly", "Professional", "Inspirational", "Humorous", "Serious", "Casual"])
    platform = st.selectbox("Platform (optional)", ["", "Instagram", "TikTok", "YouTube", "LinkedIn", "Facebook", "Twitter/X"])
    language = st.selectbox("Language", ["English", "Hindi"])
    submitted = st.form_submit_button("Generate Script")

if submitted:
    if not topic.strip():
        st.error("Please enter a topic!")
    else:
        with st.spinner("🔄 Creating script outline..."):
            script = generate_script(topic, duration, tone, platform, language)
        
        st.subheader("📋 Generated Content")
        st.markdown(script)
        st.success("✅ Script generated! Copy and use it for your next post.")

st.markdown("---")
st.caption("Made by Gagan Verma")