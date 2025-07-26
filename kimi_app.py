import os
import pandas as pd
import streamlit as st
import time
from dotenv import load_dotenv
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_groq import ChatGroq
from langchain.agents.agent_types import AgentType

# Load environment variables (expects GROQ_API_KEY and Groq_model in .env or env)
load_dotenv()

st.set_page_config(page_title="CSV Chat with LangChain Kimi", page_icon="🦙")
st.title("📊 Ask questions about your CSV with LangChain + Kimi")

# File uploader – falls back to default file in repo
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    default_path = "AI Tool GEO Data Test Set - All Channels Geo Data - Actual.csv"
    if os.path.exists(default_path):
        df = pd.read_csv(default_path)
        st.info(f"Using default CSV file: {default_path}")
    else:
        st.warning("Please upload a CSV file to get started.")
        st.stop()

st.subheader("Data preview")
st.dataframe(df.head())

def format_response(text: str) -> str:
    """Format the agent response for better readability."""
    # If it contains pandas Series output, try to format it better
    lines = text.split('\n')
    formatted_lines = []
    
    for line in lines:
        # Clean up pandas Series formatting
        if 'dtype:' in line:
            continue  # Skip dtype lines
        if 'Name:' in line and ', dtype:' in line:
            continue  # Skip name/dtype lines
        
        # Format region data better
        if any(state in line for state in ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California']):
            # This looks like state data, format it nicely
            parts = line.strip().split()
            if len(parts) >= 2 and parts[-1].isdigit():
                state = ' '.join(parts[:-1])
                value = int(parts[-1])
                formatted_lines.append(f"**{state}**: {value:,}")
            else:
                formatted_lines.append(line)
        else:
            formatted_lines.append(line)
    
    return '\n'.join(formatted_lines)

# Cache the agent so it only constructs once per session
@st.cache_resource(show_spinner=False)
def get_agent(dataframe: pd.DataFrame):
    """Create a LangChain agent capable of answering questions about the DataFrame."""
    model_name = os.getenv("Groq_model", "moonshotai/kimi-k2-instruct")  
    llm = ChatGroq(model=model_name, temperature=0.1)  # Reads GROQ_API_KEY from env
    
    # Custom prompt to improve formatting
    prefix = """
You are working with a pandas dataframe in Python. The name of the dataframe is `df`.
When presenting results:
1. Format numbers with appropriate commas for thousands separators
2. If showing data by region/category, present it in a clean, readable table format using tabulate
3. For numerical results, round to 2 decimal places where appropriate
4. If the result is a pandas Series or DataFrame, convert it to a more readable format
5. Always provide context and interpretation of the results
6. Use proper formatting for better readability
7. IMPORTANT: Include any formatted tables directly in your Final Answer, not just references to them

When creating tables, use this format:
```python
from tabulate import tabulate
# your data processing code here
print(tabulate(your_dataframe, headers='keys', tablefmt='psql'))
```

You should use the tools below to answer the question posed of you:
"""
    
    agent = create_pandas_dataframe_agent(
        llm,
        dataframe,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        allow_dangerous_code=True,
        prefix=prefix,
    )
    return agent

agent = get_agent(df)

st.markdown("### Ask a question about the data")

# Add example questions
with st.expander("💡 Example questions for better formatted results"):
    st.markdown("""
    - "Show me the top 10 regions by clicks in a formatted table"
    - "What are the total clicks by region, formatted nicely?"
    - "Create a summary of impressions by region with proper formatting"
    - "Show me click-through rates by region as percentages"
    """)

question = st.text_input("Your question")
ask_button = st.button("Ask")

if ask_button and question:
    start_time = time.time()
    with st.spinner("Thinking..."):
        try:
            response = agent.invoke(question)
            end_time = time.time()
            duration = end_time - start_time
            
            # Extract both the intermediate steps and final answer
            if isinstance(response, dict):
                answer = response.get('output', str(response))
                # Try to get intermediate steps that might contain formatted tables
                intermediate_steps = response.get('intermediate_steps', [])
            else:
                answer = str(response)
                intermediate_steps = []
            
            # Display the formatted answer
            st.markdown("### 📊 Results")
            
            # Check if the final answer itself contains a formatted table
            answer_has_table = '+----+' in answer or '|----+' in answer
            
            if answer_has_table:
                # Extract table from final answer
                lines = answer.split('\n')
                table_lines = []
                other_lines = []
                in_table = False
                
                for line in lines:
                    if '+----+' in line or '|----+' in line:
                        in_table = True
                        table_lines.append(line)
                    elif in_table and ('|' in line or '+' in line):
                        table_lines.append(line)
                    elif in_table and not ('|' in line or '+' in line) and line.strip():
                        in_table = False
                        other_lines.append(line)
                    elif not in_table:
                        other_lines.append(line)
                
                # Display table if found
                if table_lines:
                    st.markdown("**Formatted Table:**")
                    st.code('\n'.join(table_lines), language=None)
                
                # Display other text
                if other_lines:
                    st.markdown("**Summary:**")
                    clean_text = '\n'.join(other_lines).strip()
                    st.markdown(format_response(clean_text))
            else:
                # Look for formatted tables in intermediate steps
                table_found = False
                for step in intermediate_steps:
                    if len(step) >= 2:
                        action, observation = step[0], step[1]
                        # Check if observation contains a formatted table
                        if '+----+' in str(observation) or '|----+' in str(observation):
                            st.markdown("**Formatted Table:**")
                            st.code(observation, language=None)
                            table_found = True
                            break
                
                # Display the final answer
                formatted_answer = format_response(answer)
                if not table_found:
                    st.markdown(formatted_answer)
                else:
                    st.markdown("**Summary:**")
                    st.markdown(formatted_answer)
            
            # Display response time
            st.info(f"⏱️ Response time: {duration:.2f} seconds")
            
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            st.error(f"❌ Error: {e}")
            st.info(f"⏱️ Time before error: {duration:.2f} seconds")
            st.info("💡 Try rephrasing your question or check if your data contains the columns you're asking about.") 