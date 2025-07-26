#!/usr/bin/env python3
"""
Setup script for Vanna AI Marketing Analytics
"""

import os
import sys

def create_env_file():
    """Create .env file with OpenAI API key"""
    env_path = ".env"
    
    if os.path.exists(env_path):
        print(f"✅ {env_path} already exists!")
        return
    
    print("🔑 Setting up your OpenAI API key...")
    print("You can get your API key from: https://platform.openai.com/api-keys")
    
    api_key = input("Please enter your OpenAI API key: ").strip()
    
    if not api_key:
        print("❌ No API key provided. Exiting...")
        return
    
    env_content = f"""# OpenAI API Key (required for Vanna AI)
OPENAI_API_KEY={api_key}

# Alternative: You can also use other LLM providers
# ANTHROPIC_API_KEY=your_anthropic_key_here
# GROQ_API_KEY=your_groq_key_here
"""
    
    try:
        with open(env_path, 'w') as f:
            f.write(env_content)
        print(f"✅ Created {env_path} successfully!")
        print("\n🚀 You can now run: streamlit run vanna_app.py")
    except Exception as e:
        print(f"❌ Error creating {env_path}: {e}")

def main():
    print("🏗️  Vanna AI Setup Script")
    print("=" * 50)
    
    # Check if CSV file exists
    csv_path = "Incremental_NRx_TRx_Data.csv"
    if os.path.exists(csv_path):
        print(f"✅ CSV file found: {csv_path}")
    else:
        print(f"❌ CSV file not found: {csv_path}")
        print("Please ensure your CSV file is in the current directory.")
        return
    
    create_env_file()
    
    print("\n📊 Next steps:")
    print("1. Run: streamlit run vanna_app.py")
    print("2. Click 'Initialize Database' in the sidebar")
    print("3. Click 'Initialize Vanna AI' in the sidebar")
    print("4. Start asking questions about your marketing data!")

if __name__ == "__main__":
    main() 