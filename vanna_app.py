import streamlit as st
import pandas as pd
import os
from vanna.openai import OpenAI_Chat
from vanna.chromadb import ChromaDB_VectorStore
from dotenv import load_dotenv
import sqlite3
import plotly.express as px
import plotly.graph_objects as go

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Vanna AI - Marketing Data Analysis",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #333;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

class VannaApp:
    def __init__(self):
        self.csv_path = "/Users/eugeneleychenko/Downloads/SIO Langchain Experiments 7-16/only 2025 AI Tool GEO Data Test Set - All Channels Geo Data - Actual.csv"
        self.db_path = "marketing_data.db"
        
    def setup_database(self):
        """Load CSV data into SQLite database"""
        if not os.path.exists(self.csv_path):
            st.error(f"CSV file not found: {self.csv_path}")
            return False
            
        try:
            # Load CSV data
            df = pd.read_csv(self.csv_path)
            
            # Create SQLite connection
            conn = sqlite3.connect(self.db_path)
            
            # Store data in SQLite
            df.to_sql('marketing_campaigns', conn, if_exists='replace', index=False)
            conn.close()
            
            st.success(f"✅ Loaded {len(df)} rows into database")
            st.session_state.db_initialized = True
            return True
        except Exception as e:
            st.error(f"Error setting up database: {str(e)}")
            return False
    
    def setup_vanna(self):
        """Initialize Vanna with OpenAI"""
        try:
            # Check for OpenAI API key
            openai_key = os.getenv('OPENAI_API_KEY')
            if not openai_key:
                st.error("⚠️ Please set your OPENAI_API_KEY in the environment variables")
                st.info("You can set it in a .env file: OPENAI_API_KEY=your_key_here")
                return False

            # Initialize Vanna with OpenAI + ChromaDB (local)
            class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
                def __init__(self, config=None):
                    ChromaDB_VectorStore.__init__(self, config=config)
                    OpenAI_Chat.__init__(self, config=config)

            vn = MyVanna(config={'api_key': openai_key, 'model': 'gpt-4.1'})

            # Connect to SQLite database
            vn.connect_to_sqlite(self.db_path)
            
            # Allow LLM to see data for better SQL generation
            vn.allow_llm_to_see_data = True

            # Train Vanna on the schema
            self.train_vanna_on_schema(vn)

            # Store in session state
            st.session_state.vanna_model = vn
            st.session_state.vanna_initialized = True
            st.success("✅ Vanna initialized successfully!")
            return True

        except Exception as e:
            st.error(f"Error initializing Vanna: {str(e)}")
            return False
    
    def train_vanna_on_schema(self, vn):
        """Train Vanna on the database schema and sample data"""
        
        # Define the schema documentation
        ddl = """
        CREATE TABLE marketing_campaigns (
            "Region (User location)" TEXT,
            Month TEXT,
            Week TEXT,
            Campaign TEXT,
            Clicks INTEGER,
            "Impr." INTEGER,
            "Currency code" TEXT,
            Cost TEXT,
            Conversions INTEGER,
            CPC TEXT,
            CPA TEXT,
            Channel TEXT,
            Funnel TEXT,
            "Allocated Spend" TEXT,
            "Actual Spend" TEXT
        );
        """
        
        # Add documentation
        documentation = """
        This table contains marketing campaign performance data by US region/state with the following fields:
        - Region (User location): US state/region where the campaign was shown
        - Month: Campaign month (Jan-25 format)
        - Week: Campaign week (if applicable)
        - Campaign: Campaign name/description
        - Clicks: Number of clicks received
        - Impr.: Number of impressions (views)
        - Currency code: Currency used (USD)
        - Cost: Total campaign cost
        - Conversions: Number of conversions achieved
        - CPC: Cost per click
        - CPA: Cost per acquisition
        - Channel: Marketing channel (e.g., TikTok DTC, DeepIntent DTC)
        - Funnel: Marketing funnel stage (tof, mof, bof)
        - Allocated Spend: Budget allocated
        - Actual Spend: Actual amount spent
        """
        
        # Sample queries for training
        sample_queries = [
            ("What is the total number of clicks across all regions?", "SELECT SUM(Clicks) AS total_clicks FROM marketing_campaigns;"),
            ("Which region has the highest number of conversions?", "SELECT \"Region (User location)\", SUM(Conversions) AS total_conversions FROM marketing_campaigns GROUP BY \"Region (User location)\" ORDER BY total_conversions DESC LIMIT 1;"),
            ("Show monthly click trends", "SELECT Month, SUM(Clicks) AS total_clicks FROM marketing_campaigns GROUP BY Month ORDER BY Month;"),
            ("Which channel has the best performance?", "SELECT Channel, SUM(Clicks) AS total_clicks, SUM(Conversions) AS total_conversions FROM marketing_campaigns GROUP BY Channel;"),
            ("Average cost per region", "SELECT \"Region (User location)\", COUNT(*) AS campaigns FROM marketing_campaigns GROUP BY \"Region (User location)\";"),
            # NEW ADVANCED SAMPLE
            (
                "For each DTC TOF channel, show the top 10 states with the highest average cost per impression in Jun-25, the % above channel average, and total spent",
                "WITH cleaned AS ( SELECT \"Region (User location)\" AS state, Channel, Funnel, Month, CAST(REPLACE(REPLACE(Cost, '$', ''), ',', '') AS REAL) AS cost_value, CAST(REPLACE(\"Impr.\", ',', '') AS REAL) AS impressions FROM marketing_campaigns WHERE Month = 'Jun-25' AND LOWER(Channel) LIKE '%dtc%' AND LOWER(Funnel) = 'tof' AND impressions > 0 ), state_stats AS ( SELECT Channel, state, AVG(cost_value / impressions) AS avg_cpi, SUM(cost_value) AS total_budget FROM cleaned GROUP BY Channel, state ), overall_avg AS ( SELECT Channel, AVG(cost_value / impressions) AS avg_cpi_all FROM cleaned GROUP BY Channel ), ranked AS ( SELECT ss.Channel, ss.state, ss.avg_cpi, ((ss.avg_cpi - oa.avg_cpi_all) / oa.avg_cpi_all) * 100 AS pct_above_avg, ss.total_budget, ROW_NUMBER() OVER (PARTITION BY ss.Channel ORDER BY ss.avg_cpi DESC) AS rn FROM state_stats ss JOIN overall_avg oa ON ss.Channel = oa.Channel ) SELECT Channel, state, ROUND(avg_cpi, 4) AS avg_cost_per_impression, ROUND(pct_above_avg, 2) AS pct_above_avg, ROUND(total_budget, 2) AS total_budget FROM ranked WHERE rn <= 10 ORDER BY Channel, avg_cost_per_impression DESC;"
            )
        ]
        
        try:
            # Train on DDL
            vn.train(ddl=ddl)
            
            # Train on documentation
            vn.train(documentation=documentation)
            
            # Train on sample queries
            for question, sql in sample_queries:
                vn.train(question=question, sql=sql)
                
        except Exception as e:
            st.warning(f"Training warning: {str(e)}")

def main():
    st.markdown('<h1 class="main-header">📊 Vanna AI Marketing Analytics</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ask questions about your marketing campaign data in natural language!</p>', unsafe_allow_html=True)
    
    # Initialize session state variables
    if 'db_initialized' not in st.session_state:
        st.session_state.db_initialized = False
    if 'vanna_initialized' not in st.session_state:
        st.session_state.vanna_initialized = False
    if 'vanna_model' not in st.session_state:
        st.session_state.vanna_model = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Initialize the app
    app = VannaApp()
    
    # Sidebar for setup
    with st.sidebar:
        st.header("🚀 Setup")
        
        if st.button("Initialize Database", type="primary"):
            with st.spinner("Setting up database..."):
                app.setup_database()
        
        if st.button("Initialize Vanna AI", type="primary"):
            with st.spinner("Setting up Vanna..."):
                app.setup_vanna()
        
        # Show status
        st.markdown("---")
        st.markdown("### 📊 Status")
        st.write(f"Database: {'✅ Ready' if st.session_state.db_initialized else '❌ Not initialized'}")
        st.write(f"Vanna AI: {'✅ Ready' if st.session_state.vanna_initialized else '❌ Not initialized'}")
        
        st.markdown("---")
        st.markdown("### 💡 Sample Questions")
        st.markdown("""
        - What is the total number of clicks by region?
        - Which month has the highest number of conversions?
        - Show me the trend of clicks over time
        - Which regions have the best conversion rates?
        - What is the performance by marketing channel?
        - Which campaigns have the lowest cost per click?
        """)
    
    # Main content area
    if not st.session_state.vanna_initialized or st.session_state.vanna_model is None:
        st.info("👆 Please initialize the database and Vanna AI using the sidebar buttons.")
        
        # Show data preview
        if os.path.exists(app.csv_path):
            st.markdown("### 📋 Data Preview")
            df = pd.read_csv(app.csv_path)
            st.dataframe(df.head(10))
            
            # Show basic stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Rows", f"{len(df):,}")
            with col2:
                st.metric("Total Clicks", f"{df['Clicks'].sum():,}")
            with col3:
                st.metric("Total Conversions", f"{df['Conversions'].sum():,}")
            with col4:
                st.metric("Unique Regions", f"{df['Region (User location)'].nunique():,}")
        
    else:
        # Chat interface
        st.markdown("### 💬 Ask Your Question")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "sql" in message:
                    st.code(message["sql"], language="sql")
                if "chart" in message:
                    st.plotly_chart(message["chart"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your marketing data..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing your question..."):
                    try:
                        # Get the vanna model from session state
                        vanna_model = st.session_state.vanna_model
                        
                        if vanna_model is None:
                            st.error("Vanna model not found. Please reinitialize.")
                            return
                        
                        # Generate SQL
                        st.write("🔄 Generating SQL query...")
                        sql = vanna_model.generate_sql(prompt)
                        st.code(sql, language="sql")
                        
                        # Execute SQL
                        st.write("🔄 Executing query...")
                        df = vanna_model.run_sql(sql)
                        
                        if df is not None and not df.empty:
                            st.success(f"Found {len(df)} results")
                            st.dataframe(df)
                            
                            # Try to create a chart
                            try:
                                chart = create_chart(df, prompt)
                                if chart:
                                    st.plotly_chart(chart)
                                    
                                    # Add to chat history with chart
                                    st.session_state.messages.append({
                                        "role": "assistant", 
                                        "content": f"Here are the results for: {prompt}",
                                        "sql": sql,
                                        "chart": chart
                                    })
                                else:
                                    # Add to chat history without chart
                                    st.session_state.messages.append({
                                        "role": "assistant", 
                                        "content": f"Here are the results for: {prompt}",
                                        "sql": sql
                                    })
                            except Exception as chart_error:
                                st.warning(f"Could not create chart: {chart_error}")
                                st.session_state.messages.append({
                                    "role": "assistant", 
                                    "content": f"Here are the results for: {prompt}",
                                    "sql": sql
                                })
                        else:
                            st.warning("No results found for your query.")
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": "No results found for your query.",
                                "sql": sql
                            })
                            
                    except Exception as e:
                        error_msg = f"Error processing your question: {str(e)}"
                        st.error(error_msg)
                        st.write(f"Debug info: {type(e).__name__}")
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": error_msg
                        })

def create_chart(df, question):
    """Create appropriate chart based on the data and question"""
    if df.empty or len(df.columns) < 2:
        return None
    
    # Get column names
    cols = df.columns.tolist()
    
    # Determine chart type based on data
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if len(numeric_cols) >= 1 and len(cols) >= 2:
        x_col = cols[0]
        y_col = numeric_cols[0]
        
        # Create appropriate chart
        if df[x_col].dtype == 'object' and len(df) <= 20:
            # Bar chart for categorical data
            fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
        elif len(numeric_cols) >= 2:
            # Scatter plot for two numeric columns
            fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], 
                           title=f"{numeric_cols[1]} vs {numeric_cols[0]}")
        else:
            # Line chart for time series or ordered data
            fig = px.line(df, x=x_col, y=y_col, title=f"{y_col} over {x_col}")
        
        fig.update_layout(height=400)
        return fig
    
    return None

if __name__ == "__main__":
    main() 