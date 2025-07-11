"""
Main Soccer Agent Module
Uses OpenAI framework and LangChain for conversation management
"""

import os
import json
from typing import Dict, List, Optional, Any
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from dotenv import load_dotenv
import logging
import pandas as pd

from .data_collector import DataAggregator
from .ml_predictor import PlayerPerformancePredictor

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SoccerAgent:
    """Main Soccer Agent for player analysis and prediction"""
    
    def __init__(self):
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Initialize LangChain components
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.7,
            api_key=os.getenv('OPENAI_API_KEY')
        )
        
        # Initialize conversation memory
        self.memory = ConversationBufferMemory()
        self.conversation = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            verbose=True
        )
        
        # Initialize data and ML components
        self.data_aggregator = DataAggregator()
        self.ml_predictor = PlayerPerformancePredictor()
        
        # System prompt for the agent
        self.system_prompt = """You are a professional soccer analyst assistant. You have access to comprehensive player data from StatsBomb and other sources. 

Your capabilities include:
1. General player performance analysis using StatsBomb data
2. Match-specific analysis for particular games
3. Player potential prediction using machine learning models
4. Web-scraped data from various sources

Always provide detailed, professional analysis with specific metrics and insights. Use the data available to support your conclusions."""

    def analyze_player_general_performance(self, player_name: str) -> Dict:
        """Analyze general performance of a player"""
        try:
            logger.info(f"Analyzing general performance for {player_name}")
            
            # Get player data
            player_profile = self.data_aggregator.get_complete_player_profile(player_name)
            
            if not player_profile['statsbomb_data']:
                return {
                    'error': f'No data found for player {player_name}',
                    'suggestions': 'Try checking the spelling or use a different player name'
                }
            
            # Prepare analysis prompt
            analysis_prompt = f"""
            Analyze the general performance of {player_name} based on the following data:
            
            Player Profile: {json.dumps(player_profile, indent=2)}
            
            Please provide:
            1. Overall performance summary
            2. Key strengths and weaknesses
            3. Statistical highlights
            4. Comparison to typical players in their position
            5. Recommendations for improvement
            
            Format your response as a professional analysis report.
            """
            
            # Get LLM analysis
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": analysis_prompt}
                ],
                max_tokens=1000
            )
            
            analysis = response.choices[0].message.content
            
            return {
                'player_name': player_name,
                'analysis': analysis,
                'raw_data': player_profile,
                'metrics_summary': player_profile.get('statsbomb_data', {}).get('aggregated_metrics', {})
            }
            
        except Exception as e:
            logger.error(f"Error analyzing general performance: {e}")
            return {'error': f'Analysis failed: {str(e)}'}

    def analyze_match_performance(self, player_name: str, match_id: int) -> Dict:
        """Analyze player performance in a specific match"""
        try:
            logger.info(f"Analyzing match performance for {player_name} in match {match_id}")
            
            # Get match-specific data
            player_profile = self.data_aggregator.get_complete_player_profile(player_name, match_id)
            
            if not player_profile['match_specific_data']:
                return {
                    'error': f'No match data found for {player_name} in match {match_id}',
                    'suggestions': 'Check if the player participated in this match'
                }
            
            # Prepare analysis prompt
            analysis_prompt = f"""
            Analyze the match performance of {player_name} in match ID {match_id} based on the following data:
            
            Match Data: {json.dumps(player_profile['match_specific_data'], indent=2)}
            Career Context: {json.dumps(player_profile.get('statsbomb_data', {}).get('aggregated_metrics', {}), indent=2)}
            
            Please provide:
            1. Match performance summary
            2. Key moments and contributions
            3. Performance compared to career averages
            4. Impact on team performance
            5. Areas of excellence and improvement needed
            
            Format your response as a detailed match report.
            """
            
            # Get LLM analysis
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": analysis_prompt}
                ],
                max_tokens=1000
            )
            
            analysis = response.choices[0].message.content
            
            return {
                'player_name': player_name,
                'match_id': match_id,
                'analysis': analysis,
                'match_data': player_profile['match_specific_data'],
                'career_context': player_profile.get('statsbomb_data', {}).get('aggregated_metrics', {})
            }
            
        except Exception as e:
            logger.error(f"Error analyzing match performance: {e}")
            return {'error': f'Match analysis failed: {str(e)}'}

    def predict_player_potential(self, player_name: str, years_ahead: int = 3) -> Dict:
        """Predict player potential for future years"""
        try:
            logger.info(f"Predicting potential for {player_name} ({years_ahead} years ahead)")
            
            # Get player data
            player_profile = self.data_aggregator.get_complete_player_profile(player_name)
            
            if not player_profile['statsbomb_data']:
                return {
                    'error': f'No data found for player {player_name}',
                    'suggestions': 'Cannot make predictions without historical data'
                }
            
            # Get ML prediction
            prediction_result = self.ml_predictor.predict_player_potential(player_profile, years_ahead)
            
            if 'error' in prediction_result:
                return prediction_result
            
            # Prepare analysis prompt
            analysis_prompt = f"""
            Analyze the future potential of {player_name} based on the following prediction data:
            
            Current Performance: {prediction_result['current_performance']}
            Predictions: {json.dumps(prediction_result['predictions'], indent=2)}
            Confidence: {prediction_result['confidence']}
            Historical Data: {json.dumps(player_profile.get('statsbomb_data', {}).get('aggregated_metrics', {}), indent=2)}
            
            Please provide:
            1. Current performance assessment
            2. Future potential analysis for the next {years_ahead} years
            3. Key factors influencing the prediction
            4. Confidence level explanation
            5. Recommendations for maximizing potential
            6. Risk factors and considerations
            
            Format your response as a professional scouting report.
            """
            
            # Get LLM analysis
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": analysis_prompt}
                ],
                max_tokens=1200
            )
            
            analysis = response.choices[0].message.content
            
            return {
                'player_name': player_name,
                'analysis': analysis,
                'predictions': prediction_result['predictions'],
                'current_performance': prediction_result['current_performance'],
                'confidence': prediction_result['confidence'],
                'raw_data': player_profile
            }
            
        except Exception as e:
            logger.error(f"Error predicting player potential: {e}")
            return {'error': f'Prediction failed: {str(e)}'}

    def chat_with_agent(self, user_message: str) -> str:
        """Chat with the agent using LangChain conversation management"""
        try:
            # Add context about available capabilities
            enhanced_message = f"""
            {user_message}
            
            Available capabilities:
            - General player performance analysis
            - Match-specific analysis
            - Player potential prediction
            - Data from StatsBomb and web sources
            
            Please specify what type of analysis you need.
            """
            
            # Get response using LangChain
            response = self.conversation.predict(input=enhanced_message)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return f"Sorry, I encountered an error: {str(e)}"

    def get_available_competitions(self) -> List[Dict]:
        """Get list of available competitions"""
        try:
            competitions = self.data_aggregator.statsbomb.get_competitions()
            return competitions.to_dict('records') if not competitions.empty else []
        except Exception as e:
            logger.error(f"Error getting competitions: {e}")
            return []

    def get_available_matches(self, competition_id: int, season_id: int) -> List[Dict]:
        """Get list of available matches for a competition"""
        try:
            matches = self.data_aggregator.statsbomb.get_matches(competition_id, season_id)
            return matches.to_dict('records') if not matches.empty else []
        except Exception as e:
            logger.error(f"Error getting matches: {e}")
            return []

    def train_prediction_models(self, training_data: List[Dict]) -> Dict:
        """Train the machine learning models"""
        try:
            logger.info("Training prediction models...")
            result = self.ml_predictor.train_performance_model(training_data)
            
            if 'error' not in result:
                # Save trained models
                self.ml_predictor.save_models('models/player_prediction_models.pkl')
                logger.info("Models trained and saved successfully")
            
            return result
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            return {'error': f'Training failed: {str(e)}'}

    def load_trained_models(self, model_path: str = 'models/player_prediction_models.pkl') -> bool:
        """Load pre-trained models"""
        try:
            self.ml_predictor.load_models(model_path)
            logger.info("Models loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False

    def get_comprehensive_report(self, player_name: str, match_id: Optional[int] = None) -> Dict:
        """Get a comprehensive report combining all analyses"""
        try:
            report = {
                'player_name': player_name,
                'timestamp': str(pd.Timestamp.now()),
                'general_performance': {},
                'match_analysis': {},
                'potential_prediction': {},
                'summary': ''
            }
            
            # Get general performance
            general = self.analyze_player_general_performance(player_name)
            if 'error' not in general:
                report['general_performance'] = general
            
            # Get match analysis if match_id provided
            if match_id:
                match_analysis = self.analyze_match_performance(player_name, match_id)
                if 'error' not in match_analysis:
                    report['match_analysis'] = match_analysis
            
            # Get potential prediction
            prediction = self.predict_player_potential(player_name)
            if 'error' not in prediction:
                report['potential_prediction'] = prediction
            
            # Generate summary
            summary_prompt = f"""
            Create a comprehensive executive summary for {player_name} based on the following analyses:
            
            General Performance: {general.get('analysis', 'N/A')}
            Match Analysis: {report.get('match_analysis', {}).get('analysis', 'N/A')}
            Potential Prediction: {prediction.get('analysis', 'N/A')}
            
            Provide a concise summary highlighting the key insights and recommendations.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": summary_prompt}
                ],
                max_tokens=500
            )
            
            report['summary'] = response.choices[0].message.content
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating comprehensive report: {e}")
            return {'error': f'Report generation failed: {str(e)}'} 