import os
import pandas as pd
import streamlit as st
import time
from dotenv import load_dotenv
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_anthropic import ChatAnthropic
from langchain.agents.agent_types import AgentType

# Load environment variables (expects ANTHROPIC_API_KEY in .env or env)
load_dotenv()

st.set_page_config(page_title="CSV Chat with LangChain", page_icon="🦜")
st.title("📊 Ask questions about your CSV with LangChain + Anthropic")

# File uploader – falls back to default file in repo
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    default_path = "stackadapt-4-1-7-14.csv"
    if os.path.exists(default_path):
        df = pd.read_csv(default_path)
        st.info(f"Using default CSV file: {default_path}")
    else:
        st.warning("Please upload a CSV file to get started.")
        st.stop()

st.subheader("Data preview")
st.dataframe(df.head())

# Cache the agent so it only constructs once per session
@st.cache_resource(show_spinner=False)
def get_agent(dataframe: pd.DataFrame):
    """Create a LangChain agent capable of answering questions about the DataFrame."""
    llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)  # Reads ANTHROPIC_API_KEY from env
    agent = create_pandas_dataframe_agent(
        llm,
        dataframe,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        allow_dangerous_code=True,
        max_iterations=60,
        early_stopping_method="generate",
    )
    return agent

agent = get_agent(df)

st.markdown("### Ask a question about the data")
question = st.text_input("Your question")
ask_button = st.button("Ask")

if ask_button and question:
    start_time = time.time()
    with st.spinner("Thinking..."):
        try:
            response = agent.invoke(question)
            end_time = time.time()
            duration = end_time - start_time
            
            # Extract just the answer from the response
            if isinstance(response, dict) and 'output' in response:
                answer = response['output']
            else:
                answer = str(response)
            
            st.success(answer)
            st.info(f"⏱️ Response time: {duration:.2f} seconds")
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            st.error(f"❌ Error: {e}")
            st.info(f"⏱️ Time before error: {duration:.2f} seconds") 