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
        self.csv_path = "/Users/eugeneleychenko/Downloads/SIO Langchain Experiments 7-16/stackadapt-4-1-7-14.csv"
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

            vn = MyVanna(config={'api_key': openai_key, 'model': 'gpt-3.5-turbo'})

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
            Region TEXT,
            Data_Date TEXT,
            Month_Day_MonthName TEXT,
            Month TEXT,
            Campaign TEXT,
            Clicks INTEGER,
            Impr INTEGER,
            Unique_Impr INTEGER,
            Current_Code TEXT,
            Cost REAL,
            Click_Conversions INTEGER,
            View_Conversion INTEGER,
            CPC REAL,
            CPA REAL,
            Channel TEXT,
            Funnel_Stage TEXT
        );
        """
        
        # Add documentation
        documentation = """
        This table contains marketing campaign performance data with the following key metrics:
        - Clicks: Number of clicks on ads
        - Impr: Total impressions
        - Unique_Impr: Unique impressions 
        - Cost: Total cost in USD
        - Click_Conversions: Conversions from clicks
        - View_Conversion: View-through conversions
        - CPC: Cost per click
        - CPA: Cost per acquisition
        - Channel: Marketing channel (e.g., 'Slynd DTC Video StackAapt')
        - Funnel_Stage: TOF (Top of Funnel) or BOF (Bottom of Funnel)
        - Region: US states where campaigns ran
        """
        
        # Sample queries for training
        sample_queries = [
            ("What is the total cost across all campaigns?", "SELECT SUM(Cost) as total_cost FROM marketing_campaigns;"),
            ("Which region has the highest number of clicks?", "SELECT Region, SUM(Clicks) as total_clicks FROM marketing_campaigns GROUP BY Region ORDER BY total_clicks DESC LIMIT 1;"),
            ("What's the average CPC by channel?", "SELECT Channel, AVG(CPC) as avg_cpc FROM marketing_campaigns WHERE CPC > 0 GROUP BY Channel;"),
            ("Show performance by funnel stage", "SELECT Funnel_Stage, SUM(Clicks) as total_clicks, SUM(Cost) as total_cost, AVG(CPC) as avg_cpc FROM marketing_campaigns GROUP BY Funnel_Stage;"),
            ("Which campaigns have the best conversion rates?", "SELECT Campaign, SUM(Click_Conversions) as conversions, SUM(Clicks) as clicks, CASE WHEN SUM(Clicks) > 0 THEN (SUM(Click_Conversions) * 100.0 / SUM(Clicks)) ELSE 0 END as conversion_rate FROM marketing_campaigns GROUP BY Campaign HAVING SUM(Clicks) > 0 ORDER BY conversion_rate DESC;")
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
        - What is the total cost by region?
        - Which campaigns have the highest CTR?
        - Show me performance by funnel stage
        - What's the average CPC by channel?
        - Which region generates the most conversions?
        - Compare TOF vs BOF performance
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
                st.metric("Total Cost", f"${df['Cost'].sum():,.2f}")
            with col3:
                st.metric("Total Clicks", f"{df['Clicks'].sum():,}")
            with col4:
                st.metric("Total Impressions", f"{df['Impr'].sum():,}")
        
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