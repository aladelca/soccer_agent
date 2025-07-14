"""
Soccer Agent - Main conversational agent for football player analysis
Handles user interactions and provides comprehensive player analysis
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.data_collector import StatsBombDataCollector, WebScraper, PlayerSearchResult

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class ConversationState(Enum):
    """States for the player selection conversation"""
    SEARCHING = "searching"
    SHOWING_RESULTS = "showing_results"
    CONFIRMING_SELECTION = "confirming_selection"
    COMPLETED = "completed"
    ERROR = "error"

class PlayerSelectionFlow:
    """Handles player selection flow when multiple results are found"""
    
    def __init__(self):
        self.conversation_state = ConversationState.SEARCHING
        self.search_results: List[PlayerSearchResult] = []
        self.selected_player: Optional[PlayerSearchResult] = None
        self.user_context: Dict[str, Any] = {}
    
    def start_search(self, player_name: str) -> str:
        """Start the player search process"""
        self.conversation_state = ConversationState.SEARCHING
        self.user_context['search_query'] = player_name
        
        return f"🔍 Searching for players with name '{player_name}'...\n\nPlease wait while I search the database."
    
    def process_search_results(self, results: List[PlayerSearchResult]) -> str:
        """Process search results and generate response"""
        self.search_results = results
        self.conversation_state = ConversationState.SHOWING_RESULTS
        
        if not results:
            self.conversation_state = ConversationState.ERROR
            return "❌ I couldn't find any players with that name. Please check the spelling or try a different name."
        
        if len(results) == 1:
            # Only one result, auto-select
            self.selected_player = results[0]
            self.conversation_state = ConversationState.CONFIRMING_SELECTION
            return self._generate_single_result_message(results[0])
        
        # Multiple results, show options
        return self._generate_multiple_results_message(results)
    
    def handle_user_selection(self, user_input: str) -> str:
        """Handle user selection from multiple results"""
        try:
            # Try to parse as number
            selection = int(user_input.strip())
            if 1 <= selection <= len(self.search_results):
                self.selected_player = self.search_results[selection - 1]
                self.conversation_state = ConversationState.CONFIRMING_SELECTION
                return self._generate_confirmation_message(self.selected_player)
            else:
                return f"❌ Please select a number between 1 and {len(self.search_results)}."
        except ValueError:
            # Try to match by name
            user_input_lower = user_input.lower().strip()
            for i, result in enumerate(self.search_results):
                if user_input_lower in result.player_name.lower():
                    self.selected_player = result
                    self.conversation_state = ConversationState.CONFIRMING_SELECTION
                    return self._generate_confirmation_message(self.selected_player)
            
            return f"❌ I couldn't find an exact match. Please select a number from 1 to {len(self.search_results)}."
    
    def handle_confirmation(self, user_input: str) -> Tuple[str, Optional[PlayerSearchResult]]:
        """Handle user confirmation of selected player"""
        user_input_lower = user_input.lower().strip()
        
        if user_input_lower in ['yes', 'y', 'confirm', 'sure', 'ok']:
            if self.selected_player is None:
                logger.error("Selected player is None during confirmation")
                return "❌ An error occurred. Please try your search again.", None
            self.conversation_state = ConversationState.COMPLETED
            return f"✅ Perfect! I've selected {self.selected_player.player_name} for analysis.", self.selected_player
        elif user_input_lower in ['no', 'n', 'cancel', 'other', 'different']:
            self.conversation_state = ConversationState.SHOWING_RESULTS
            return self._generate_multiple_results_message(self.search_results), None
        else:
            return "❓ Please respond with 'yes' to confirm or 'no' to select another player.", None
    
    def get_current_state(self) -> ConversationState:
        """Get current conversation state"""
        return self.conversation_state
    
    def reset(self):
        """Reset the conversation flow"""
        self.conversation_state = ConversationState.SEARCHING
        self.search_results = []
        self.selected_player = None
        self.user_context = {}
    
    def _generate_single_result_message(self, player: PlayerSearchResult) -> str:
        """Generate message for single search result"""
        return f"""✅ I found exactly one player:

👤 **{player.player_name}**
🏟️ Club: {player.club}

Is this the player you're looking for? Respond 'yes' to continue or 'no' to search for another."""
    
    def _generate_multiple_results_message(self, results: List[PlayerSearchResult]) -> str:
        """Generate message for multiple search results"""
        message = f"🔍 I found {len(results)} players with that name:\n\n"
        
        for i, player in enumerate(results, 1):
            message += f"{i}. **{player.player_name}**\n"
            message += f"   🏟️ {player.club}\n\n"
        
        message += "Please select the number of the player you want to analyze, or type part of the name for a more specific search."
        
        return message
    
    def _generate_confirmation_message(self, player: PlayerSearchResult) -> str:
        """Generate confirmation message for selected player"""
        return f"""✅ You have selected:

👤 **{player.player_name}**
🏟️ Club: {player.club}

Do you confirm that you want to analyze this player? Respond 'yes' to continue or 'no' to select another."""

class SoccerAgent:
    """Main conversational agent for football player analysis"""
    
    def __init__(self):
        self.statsbomb_collector = StatsBombDataCollector()
        self.web_scraper = WebScraper()
        self.player_selection_flow = PlayerSelectionFlow()
        self.user_sessions: Dict[str, PlayerSelectionFlow] = {}
    
    def handle_message(self, user_id: str, message: str) -> str:
        """
        Handle incoming message from user
        
        Args:
            user_id: User identifier
            message: User message
            
        Returns:
            Response message to send back
        """
        try:
            # Get or create user session
            if user_id not in self.user_sessions:
                logger.info(f"Creating new session for user {user_id}")
                self.user_sessions[user_id] = PlayerSelectionFlow()
            
            session = self.user_sessions[user_id]
            current_state = session.get_current_state()
            logger.info(f"Processing message for user {user_id} in state {current_state.value}")
            
            # Handle different conversation states
            if current_state == ConversationState.SEARCHING:
                logger.debug(f"User {user_id} initiating search with query: {message}")
                return self._handle_search_request(user_id, message)
            elif current_state == ConversationState.SHOWING_RESULTS:
                logger.debug(f"User {user_id} selecting from results: {message}")
                return session.handle_user_selection(message)
            elif current_state == ConversationState.CONFIRMING_SELECTION:
                logger.debug(f"User {user_id} confirming selection: {message}")
                response, selected_player = session.handle_confirmation(message)
                if selected_player:
                    logger.info(f"User {user_id} confirmed player selection: {selected_player.player_name}")
                    return self._analyze_selected_player(user_id, selected_player, response)
                return response
            elif current_state == ConversationState.COMPLETED:
                # Reset session for new search
                logger.info(f"Resetting completed session for user {user_id}")
                session.reset()
                return self._handle_search_request(user_id, message)
            else:
                logger.warning(f"Unexpected state {current_state} for user {user_id}, resetting session")
                session.reset()
                return self._handle_search_request(user_id, message)
                
        except Exception as e:
            logger.error(f"Error handling message for user {user_id}: {str(e)}", exc_info=True)
            return "❌ An unexpected error occurred. Please try again."
    
    def _handle_search_request(self, user_id: str, message: str) -> str:
        """Handle initial search request"""
        session = self.user_sessions[user_id]
        logger.info(f"Starting player search for user {user_id} with query: {message}")
        
        # Start search process
        search_message = session.start_search(message)
        
        try:
            # Perform actual search
            logger.debug(f"Calling web scraper search for query: {message}")
            search_results = self.web_scraper.search_players_with_selection(message)
            logger.info(f"Found {len(search_results)} players matching query: {message}")
            
            # Log search results for debugging
            for idx, result in enumerate(search_results, 1):
                logger.debug(f"Search result {idx}: Player ID: {result.player_id}, "
                           f"Name: {result.player_name}, Club: {result.club}, "
                           f"Confidence: {result.confidence_score}")
            
            # Process results
            return session.process_search_results(search_results)
            
        except Exception as e:
            logger.error(f"Error during player search for user {user_id}: {str(e)}", exc_info=True)
            session.conversation_state = ConversationState.ERROR
            return "❌ An error occurred while searching for players. Please try again."
    
    def _analyze_selected_player(self, user_id: str, selected_player: PlayerSearchResult, confirmation_message: str) -> str:
        """Analyze the selected player and return comprehensive report"""
        try:
            logger.info(f"Starting analysis for player {selected_player.player_name} (ID: {selected_player.player_id})")
            
            # Get comprehensive data for the selected player
            logger.debug(f"Fetching Transfermarkt data for player ID: {selected_player.player_id}")
            transfermarkt_data = self.web_scraper.get_transfermarkt_data_by_id(selected_player.player_id)
            
            logger.debug(f"Fetching StatsBomb data for player: {selected_player.player_name}")
            #TODO: Add StatsBomb data
            #career_data = self.statsbomb_collector.get_player_career_data(selected_player.player_name)
            
            # Log data retrieval results
            logger.debug(f"Transfermarkt data retrieved: {bool(transfermarkt_data)}")
            #logger.debug(f"Career data retrieved: {bool(career_data)}")
            
            # Generate analysis report
            #TODO: Add StatsBomb data
            report = self._generate_player_report(selected_player, transfermarkt_data, None)
            
            # Reset session for next search
            logger.info(f"Analysis completed for user {user_id}, resetting session")
            self.user_sessions[user_id].reset()
            
            return report
            
        except Exception as e:
            logger.error(f"Error analyzing player {selected_player.player_name}: {str(e)}", exc_info=True)
            return f"❌ Error analyzing {selected_player.player_name}. Please try again."
    
    def _generate_player_report(self, player: PlayerSearchResult, transfermarkt_data: Dict, career_data: Optional[Dict] = None) -> str:
        """Generate comprehensive player analysis report"""
        try:
            logger.info(f"Generating report for player: {player.player_name}")
            
            report = f"""📊 **COMPREHENSIVE ANALYSIS OF {player.player_name.upper()}**

👤 **BASIC INFORMATION**
🏟️ Club: {player.club}
"""
            
            # Add Transfermarkt data
            if 'error' not in transfermarkt_data:
                logger.debug("Processing Transfermarkt data for report")
                profile = transfermarkt_data.get('profile', {})
                performance = transfermarkt_data.get('performance_data', {})
                transfers = transfermarkt_data.get('transfer_history', {})
                
                report += f"""
📈 **TRANSFERMARKT DATA**
"""
                
                if performance.get('performance_records'):
                    logger.debug("Adding performance statistics to report")
                    total_matches = sum(int(record.get('matches', 0)) for record in performance['performance_records'])
                    total_goals = sum(int(record.get('goals', 0)) for record in performance['performance_records'])
                    total_assists = sum(int(record.get('assists', 0)) for record in performance['performance_records'])
                    
                    report += f"🎯 Total matches: {total_matches}\n"
                    report += f"⚽ Total goals: {total_goals}\n"
                    report += f"🎯 Total assists: {total_assists}\n"
                    
                    if total_matches > 0:
                        report += f"📊 Goals per match: {total_goals/total_matches:.2f}\n"
                        report += f"📊 Assists per match: {total_assists/total_matches:.2f}\n"
                
                if transfers.get('transfers'):
                    report += f"🔄 Transfers: {len(transfers['transfers'])}\n"
            else:
                logger.warning(f"No Transfermarkt data available for player {player.player_name}")
            
            '''
            # Add StatsBomb data
            if 'error' not in career_data and career_data.get('matches_analyzed', 0) > 0:
                logger.debug("Processing StatsBomb data for report")
                career_summary = career_data.get('career_summary', {})
                
                report += f"""
⚽ **TECHNICAL ANALYSIS (StatsBomb)**
📊 Matches analyzed: {career_data.get('matches_analyzed', 0)}
"""
                
                if career_summary.get('total_passes', 0) > 0:
                    pass_accuracy = career_summary.get('career_pass_accuracy', 0)
                    report += f"🎯 Pass accuracy: {pass_accuracy:.1%}\n"
                
                if career_summary.get('total_dribbles', 0) > 0:
                    dribble_success = career_summary.get('career_dribble_success_rate', 0)
                    report += f"🔄 Dribble success rate: {dribble_success:.1%}\n"
                
                if career_summary.get('total_progressive_actions', 0) > 0:
                    report += f"📈 Progressive actions: {career_summary.get('total_progressive_actions', 0)}\n"
            else:
                logger.warning(f"No StatsBomb data available for player {player.player_name}")
            '''
            report += f"""

🔍 To search for another player, simply type their name.
"""
            
            logger.info(f"Report generated successfully for player {player.player_name}")
            return report
            
        except Exception as e:
            logger.error(f"Error generating player report: {str(e)}", exc_info=True)
            return f"❌ Error generating report for {player.player_name}."
    
    def reset_user_session(self, user_id: str):
        """Reset user session"""
        if user_id in self.user_sessions:
            self.user_sessions[user_id].reset()
    
    def get_user_state(self, user_id: str) -> ConversationState:
        """Get current state for user"""
        if user_id in self.user_sessions:
            return self.user_sessions[user_id].get_current_state()
        return ConversationState.SEARCHING
    
    def get_welcome_message(self) -> str:
        """Get welcome message for new users"""
        return """⚽ **Welcome to Soccer Agent!** ⚽

I'm your intelligent football player analysis assistant. I can provide you with:

📊 Detailed player statistics
🎯 Technical performance analysis
📈 Career and transfer data
🏆 Player comparisons

**To get started:**
Simply type the name of the player you want to analyze.

**Examples:**
- "Lionel Messi"
- "Cristiano Ronaldo"
- "Erling Haaland"

Let's begin! 🚀"""
    
    def get_help_message(self) -> str:
        """Get help message"""
        return """📚 **Available Commands:**

/start - Start the bot
/help - Show this help
/reset - Reset your current session
/status - View current conversation status

**How to use the agent:**

1. **Search player:** Type the player's name
2. **Select:** If there are multiple results, choose the correct number
3. **Confirm:** Confirm your selection with "yes"
4. **View analysis:** Receive the complete player report

**Example conversation flow:**
```
User: "Messi"
Agent: "I found 3 players with that name:
1. Lionel Messi (Inter Miami)
2. Messi Riquelme (Boca Juniors)
3. Messi Silva (Santos)

Please select the number of the player..."

User: "1"
Agent: "You have selected: Lionel Messi
Do you confirm? Respond 'yes' to continue..."

User: "yes"
Agent: [Sends complete analysis]
```

Enjoy analyzing players! ⚽"""
    
    def get_status_message(self, user_id: str) -> str:
        """Get status message for user"""
        current_state = self.get_user_state(user_id)
        
        state_messages = {
            "searching": "🔍 Status: Searching for player",
            "showing_results": "📋 Status: Showing search results",
            "confirming_selection": "✅ Status: Confirming selection",
            "completed": "✅ Status: Analysis completed",
            "error": "❌ Status: Search error"
        }
        
        return f"📊 **Current session status:**\n\n{state_messages.get(current_state.value, '❓ Unknown status')}" 