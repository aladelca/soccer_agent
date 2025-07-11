# Soccer Agent - Intelligent Football Player Analysis

An intelligent agent for comprehensive football player analysis using StatsBomb data, machine learning, and LLMs.

## 🚀 Features

- **General Performance Analysis**: Complete evaluation of a player's performance
- **Match-Specific Analysis**: Detailed analysis of performance in specific matches
- **Potential Prediction**: ML models to predict future performance
- **Interactive Chat**: Natural conversations with the agent using LangChain
- **Multi-Source Data**: StatsBomb Open Data + web scraping
- **Comprehensive Reports**: Complete analysis with visualizations

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

Edit the `.env` file and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## 🎯 Quick Start

### Demo Notebook
```python
# Open the demo notebook
jupyter notebook notebooks/soccer_agent_demo.ipynb
```

### Programmatic Usage
```python
from src.soccer_agent import SoccerAgent

# Initialize the agent
agent = SoccerAgent()

# General player analysis
analysis = agent.analyze_player_general_performance("Dusan Tadic")

# Match-specific analysis
match_analysis = agent.analyze_match_performance("Dusan Tadic", match_id=7532)

# Potential prediction
potential = agent.predict_player_potential("Dusan Tadic", years_ahead=3)

# Interactive chat
response = agent.chat_with_agent("What are this player's strengths?")
```

## 📊 Main Features

### 1. General Performance Analysis
- Pass, shot, dribble metrics
- Comparison with position standards
- Identification of strengths and weaknesses
- Improvement recommendations

### 2. Match-Specific Analysis
- Performance in specific matches
- Key moments and contributions
- Comparison with career averages
- Team impact

### 3. Potential Prediction
- Machine learning models
- 1-5 year predictions
- Influencing factors
- Confidence levels

### 4. Interactive Chat
- Natural conversations
- Conversation memory
- Contextual responses
- Personalized analysis

## 🏗️ Architecture

```
soccer_agent/
├── src/
│   ├── data_collector.py      # Data collection
│   ├── ml_predictor.py        # ML models
│   └── soccer_agent.py        # Main agent
├── notebooks/
│   ├── first_notebook.ipynb   # Original notebook
│   └── soccer_agent_demo.ipynb # Complete demo
├── models/                    # Trained models
├── requirements.txt           # Dependencies
└── README.md                 # This file
```

## 🔧 Technical Components

### Data Collector
- **StatsBombDataCollector**: Access to StatsBomb open data
- **WebScraper**: Scraping additional metrics
- **DataAggregator**: Aggregation from multiple sources

### ML Predictor
- **PlayerPerformancePredictor**: Prediction models
- **Feature Engineering**: Feature preparation
- **Model Training**: Training and evaluation

### Soccer Agent
- **OpenAI Integration**: LLM for analysis
- **LangChain**: Conversation management
- **Comprehensive Analysis**: Complete reports

## 📈 Data Sources

### StatsBomb Open Data
- Match events
- Player metrics
- Tactical data
- Competition information

### Web Scraping
- Transfermarkt (market value)
- Additional statistics
- Contract information

## 🤖 Machine Learning Models

- **Random Forest**: Performance prediction
- **Gradient Boosting**: Advanced models
- **Linear Regression**: Trend analysis
- **Feature Importance**: Key factor identification

## 📝 Usage Examples

### Complete Analysis
```python
# Comprehensive report
report = agent.get_comprehensive_report("Dusan Tadic", match_id=7532)
print(report['summary'])
```

### Custom Chat
```python
# Specific questions
response = agent.chat_with_agent("How does this player compare to other forwards?")
```

### Predictions
```python
# 5-year prediction
potential = agent.predict_player_potential("Player", years_ahead=5)
```

## 🔍 Data Exploration

### Available Competitions
```python
competitions = agent.get_available_competitions()
print(pd.DataFrame(competitions))
```

### Matches by Competition
```python
matches = agent.get_available_matches(competition_id=43, season_id=3)
print(f"Total matches: {len(matches)}")
```

## 📊 Visualizations

The agent includes automatic visualizations:
- Pass accuracy charts
- Shot analysis
- Dribble success rates
- Potential predictions

## 🚀 Next Steps

- [ ] Web interface with Streamlit
- [ ] More data sources
- [ ] Advanced ML models
- [ ] Complete team analysis
- [ ] REST API
- [ ] Persistent database

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