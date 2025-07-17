# Vanna AI Marketing Analytics

This application uses Vanna AI to enable natural language queries on your marketing campaign data using a beautiful Streamlit interface.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up OpenAI API Key
Create a `.env` file in the project root with:
```
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Run the Application
```bash
streamlit run vanna_app.py
```

## 📊 Features

- **Natural Language Queries**: Ask questions about your marketing data in plain English
- **Automatic SQL Generation**: Vanna converts your questions to SQL automatically
- **Interactive Charts**: Visualizations are generated automatically based on your queries
- **Chat Interface**: Conversation-style interaction with your data
- **Data Preview**: View your dataset before querying

## 💡 Example Questions

Try asking these questions once the app is running:

- "What is the total cost by region?"
- "Which campaigns have the highest click-through rate?"
- "Show me performance by funnel stage"
- "What's the average cost per click by channel?"
- "Which region generates the most conversions?"
- "Compare top of funnel vs bottom of funnel performance"
- "What are the top 10 most expensive campaigns?"
- "Show me campaigns with zero conversions"

## 🔧 How It Works

1. **Database Setup**: Your CSV data is loaded into a SQLite database
2. **Schema Training**: Vanna learns about your data structure and common queries
3. **Query Processing**: When you ask a question:
   - Vanna generates appropriate SQL
   - The query is executed against your data
   - Results are displayed with automatic visualizations

## 📁 Data Schema

Your marketing campaign data includes:

- **Region**: US states where campaigns ran
- **Campaign**: Campaign names and targeting
- **Clicks**: Number of ad clicks
- **Impr**: Total impressions served
- **Cost**: Campaign costs in USD
- **Conversions**: Click and view conversions
- **CPC/CPA**: Cost per click and acquisition metrics
- **Channel**: Marketing channels (DTC Video, Display, etc.)
- **Funnel_Stage**: TOF (Top of Funnel) or BOF (Bottom of Funnel)

## 🔑 API Key Setup

You'll need an OpenAI API key to use Vanna. Get one at: https://platform.openai.com/api-keys

Alternative LLM providers can also be configured in the code.

## 🛠 Troubleshooting

- **"CSV file not found"**: Ensure the CSV path in `vanna_app.py` is correct
- **"OpenAI API key not found"**: Check your `.env` file setup
- **"No results found"**: Try rephrasing your question or asking simpler queries
- **Import errors**: Make sure all dependencies are installed with `pip install -r requirements.txt`

## 🎯 Tips for Better Results

1. **Be specific**: "Cost by region" works better than "show me costs"
2. **Use data terminology**: Reference actual column names when possible
3. **Start simple**: Begin with basic aggregations before complex analyses
4. **Check the SQL**: Review the generated SQL to understand what Vanna created

Enjoy exploring your marketing data with natural language! 🚀 