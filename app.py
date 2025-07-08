# Soccer LLM App with OpenAI API
# Requirements: pip install flask openai python-dotenv

import os
from flask import Flask, request, jsonify, render_template
from openai import OpenAI
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Soccer-focused system prompt
SOCCER_SYSTEM_PROMPT = """You are a specialized AI assistant focused exclusively on soccer (football). You are knowledgeable about:

- Players (current and historical)
- Teams and clubs worldwide
- Rules and regulations
- Tactics and formations
- Tournaments and competitions
- Soccer history and statistics
- Training and fitness
- Equipment and gear

Please provide accurate, helpful, and engaging responses about soccer. If a question is not related to soccer, politely redirect the conversation back to soccer topics. Keep responses conversational and informative, suitable for soccer fans of all knowledge levels.

Always respond in a friendly, enthusiastic tone that reflects your passion for the beautiful game."""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Check if API key is set
        if not os.getenv('OPENAI_API_KEY'):
            return jsonify({'error': 'API key not configured. Please set OPENAI_API_KEY environment variable.'}), 500
        
        # Make request to OpenAI API
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": SOCCER_SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        
        ai_response = response.choices[0].message.content
        
        return jsonify({'response': ai_response})
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model': 'gpt-4'})

if __name__ == '__main__':
    # Check if API key is set
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  WARNING: OPENAI_API_KEY environment variable is not set!")
        print("Please set it before running the app:")
        print("export OPENAI_API_KEY='your_api_key_here'")
        print("or create a .env file with: OPENAI_API_KEY=your_api_key_here")
    
    print("🚀 Starting Soccer AI Assistant...")
    print("⚽ Powered by OpenAI GPT-4")
    print("🌐 Access the app at: http://localhost:8000")
    
    app.run(debug=True, host='0.0.0.0', port=8000)