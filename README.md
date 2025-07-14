# Soccer Agent - Intelligent Football Player Analysis

An intelligent conversational agent for comprehensive football player analysis using StatsBomb data, Transfermarkt information, and advanced analytics.

## 🚀 Features

- **Conversational Interface**: Interactive agent that handles multiple search results and user selection
- **Player Performance Analysis**: Complete evaluation of a player's performance using StatsBomb data
- **Career Statistics**: Comprehensive career overview with aggregated metrics
- **Transfer Market Data**: Player market value and transfer history from Transfermarkt
- **Multi-Source Data**: StatsBomb Open Data + Transfermarkt + web scraping
- **Session Management**: Maintains conversation state for multiple users
- **Telegram Bot Support**: Ready-to-use Telegram bot implementation

## 📋 Requirements

- Python 3.8+
- OpenAI API Key
- Internet connection for StatsBomb data

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd soccer_agent
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Configure environment variables**:
```bash
cp env_example.txt .env
```

Edit the `.env` file and add your API keys:
```
# Optional: For Transfermarkt data
RAPID_API_KEY=your_rapidapi_key_here

# Optional: For Telegram bot
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here

# Optional: For future ML features
OPENAI_API_KEY=your_openai_api_key_here
```

## 🎯 Quick Start

### Demo Notebook
```python
# Open the demo notebook
jupyter notebook notebooks/soccer_agent_demo.ipynb
```

### Conversational Usage
```python
from src.soccer_agent import SoccerAgent

# Initialize the agent
agent = SoccerAgent()

# Handle user messages
user_id = "user123"
response = agent.handle_message(user_id, "Messi")
print(response)

# User selects from results
response = agent.handle_message(user_id, "1")
print(response)

# User confirms
response = agent.handle_message(user_id, "yes")
print(response)
```

### Direct Data Access
```python
from src.data_collector import StatsBombDataCollector, WebScraper

# Get StatsBomb data
statsbomb = StatsBombDataCollector()
career_data = statsbomb.get_player_career_data("Lionel Messi")

# Get Transfermarkt data
scraper = WebScraper()
transfermarkt_data = scraper.get_transfermarkt_data("Lionel Messi")
```

## 📊 Main Features

### 1. Conversational Interface
- Interactive agent that handles multiple search results
- User selection from multiple players with same name
- Confirmation flow for accurate player selection
- Session management for multiple users

### 2. Player Performance Analysis
- Pass, shot, dribble metrics from StatsBomb
- Comparison with position standards
- Identification of strengths and weaknesses
- Improvement recommendations

### 3. Career Statistics
- Comprehensive career overview
- Aggregated metrics across competitions
- Performance trends and patterns
- Historical data analysis

### 4. Transfer Market Data
- Market values from Transfermarkt
- Transfer history and career moves
- Club performance data
- Contract and valuation information

## 🏗️ Architecture

```
soccer_agent/
├── src/
│   ├── data_collector.py      # Data collection and analysis
│   └── soccer_agent.py        # Conversational agent
├── examples/
│   ├── simple_agent_example.py # Basic usage example
│   └── telegram_bot_example.py # Telegram bot implementation
├── notebooks/
│   ├── first_notebook.ipynb   # Original notebook
│   └── soccer_agent_demo.ipynb # Complete demo
├── requirements.txt           # All dependencies
└── README.md                 # This file
```

## 🔧 Technical Components

### Data Collector
- **StatsBombDataCollector**: Access to StatsBomb open data and player analysis
- **WebScraper**: Transfermarkt data collection and player search
- **DataAggregator**: Combines data from multiple sources

### Soccer Agent
- **SoccerAgent**: Main conversational agent
- **PlayerSelectionFlow**: Handles user selection and conversation states
- **Session Management**: Multi-user conversation tracking
- **Comprehensive Analysis**: Complete player reports

## 📈 Data Sources

### StatsBomb Open Data
- Match events and player actions
- Detailed player metrics and statistics
- Tactical data and formations
- Competition and match information

### Transfermarkt
- Player market values
- Transfer history and career moves
- Club performance data
- Contract information and valuations

## 🤖 Conversational Features

- **Multi-result Search**: Handles multiple players with same name
- **User Selection**: Interactive selection from search results
- **Confirmation Flow**: Ensures accurate player selection
- **Session Management**: Maintains conversation state per user
- **Rich Reports**: Comprehensive player analysis with formatting

## 📝 Usage Examples

### Conversational Analysis
```python
# Search for a player
response = agent.handle_message("user123", "Messi")
print(response)

# Select from results
response = agent.handle_message("user123", "1")
print(response)

# Confirm selection
response = agent.handle_message("user123", "yes")
print(response)
```

### Direct Data Access
```python
# Get StatsBomb career data
statsbomb = StatsBombDataCollector()
career_data = statsbomb.get_player_career_data("Lionel Messi")

# Get Transfermarkt data
scraper = WebScraper()
transfermarkt_data = scraper.get_transfermarkt_data("Lionel Messi")
```

### Telegram Bot
```bash
# Set your bot token
export TELEGRAM_BOT_TOKEN="your_bot_token"

# Run the bot
python examples/telegram_bot_example.py
```

## 🔍 Data Exploration

### Available Competitions
```python
from src.data_collector import StatsBombDataCollector

statsbomb = StatsBombDataCollector()
competitions = statsbomb.get_competitions()
print(competitions)
```

### Player Search
```python
from src.data_collector import WebScraper

scraper = WebScraper()
results = scraper.search_players_with_selection("Messi")
for result in results:
    print(f"{result.player_name} - {result.club}")
```

## 📊 Analysis Features

The agent provides comprehensive analysis:
- Player performance statistics
- Career overview and trends
- Transfer market information
- Technical metrics and insights

## 🚀 Next Steps

- [ ] Web interface with Streamlit
- [ ] More data sources (FIFA, Opta, etc.)
- [ ] Advanced ML models for predictions
- [ ] Complete team analysis
- [ ] REST API
- [ ] Persistent database
- [ ] Player comparisons
- [ ] Real-time match analysis

## 🔧 Troubleshooting

### Environment Variables Not Found
If you get errors about missing environment variables:

1. **Create `.env` file**:
```bash
cp env_example.txt .env
```

2. **Add your API keys** to the `.env` file:
```bash
RAPID_API_KEY=your_actual_rapidapi_key_here
TELEGRAM_BOT_TOKEN=your_actual_telegram_bot_token_here
OPENAI_API_KEY=your_actual_openai_api_key_here
```

3. **Test environment loading**:
```bash
python test_env.py
```

### Telegram Bot Issues
If you get Telegram bot errors:

1. **Install correct version**:
```bash
pip install python-telegram-bot==20.7
```

2. **Check bot token**:
```bash
python test_env.py
```

3. **Test basic functionality**:
```bash
python examples/simple_telegram_bot.py
```

4. **For full bot functionality**:
```bash
python install_telegram.py
python examples/telegram_bot_example.py
```

### Basic Functionality Test
Test core functionality without API keys:
```bash
python test_basic.py
```

## 🤝 Contributions

1. Fork the project
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

This project is under the MIT License. See `LICENSE` for more details.

## 🆘 Support

For questions or issues:
1. Check the documentation
2. Run the demo notebook
3. Open an issue on GitHub

## 🙏 Acknowledgments

- StatsBomb for open data
- OpenAI for LLM APIs
- LangChain for conversation framework
- Football analysis community

---

**Enjoy analyzing football with AI! ⚽🤖**