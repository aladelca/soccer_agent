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

from langchain.prompts import PromptTemplate
from langchain.memory import ConversationTokenBufferMemory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.memory.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage

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

class LangChainConversationManager:
    """Manages conversations using LangChain"""
    
    def __init__(self):
        """Initialize the conversation manager with LangChain components"""
        # Initialize ChatOpenAI with appropriate temperature
        self.llm = ChatOpenAI(
            temperature=0.7,
            model="gpt-3.5-turbo",
            model_kwargs={"max_tokens": 1000}  # Limit response length
        )
        
        # Create chat history for memory
        self.chat_history = ChatMessageHistory()
        
        # Create conversation memory with token limit
        self.memory = ConversationTokenBufferMemory(
            llm=self.llm,
            chat_memory=self.chat_history,
            return_messages=True,
            memory_key="chat_history",
            max_token_limit=4000  # Keep last 4000 tokens
        )
        
        # Initialize state and data
        self.current_state: ConversationState = ConversationState.SEARCHING
        self.search_results: List[PlayerSearchResult] = []
        self.selected_player: Optional[PlayerSearchResult] = None
        self.player_data: Optional[Dict[str, Any]] = None  # Store the player's data here
        
        # Define the conversation prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a soccer analysis assistant that helps analyze players using StatsBomb and Transfermarkt data.

Instructions based on message type:
1. Welcome: Brief intro and examples
2. Help: List commands and basic flow
3. Search Results: 
   - Show each player with their number and club
   - Format as: "{{number}}. {{player_name}} (🏟️ {{club}})"
   - Ask user to select by number
4. Selection: Show selected player details and confirm
5. Analysis: Key stats and achievements in markdown
6. Errors: Clear explanation and next steps

Keep responses concise and use emojis sparingly."""),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])
        
        # Create the conversation chain using RunnableWithMessageHistory
        self.chain = self.prompt | self.llm
        self.conversation_chain = RunnableWithMessageHistory(
            self.chain,
            lambda session_id: self.chat_history,
            input_messages_key="input",
            history_messages_key="chat_history"
        )
        
        # Initialize state
        self.current_state = ConversationState.SEARCHING
        self.search_results = []
        self.selected_player = None
    
    def get_response(self, user_input: str, state: ConversationState, context: Optional[Dict] = None) -> str:
        """Get a response from the LangChain conversation manager"""
        try:
            # Format the input with state and context
            formatted_input = user_input
            if state:
                formatted_input += f"\nState: {state.value}"
            if context:
                formatted_input += "\n" + self._format_context(context)
            
            # Get response from LangChain
            response = self.conversation_chain.invoke(
                {"input": formatted_input},
                config={"configurable": {"session_id": "default"}}
            )
            
            # Extract the response text
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # If we're showing results, make sure the response includes the player list
            if state == ConversationState.SHOWING_RESULTS and context and "results" in context:
                if "1." not in response_text:
                    results_str = self._format_context(context)
                    response_text = f"{response_text}\n{results_str}"
            
            return response_text
            
        except Exception as e:
            logger.error(f"Error getting response: {str(e)}")
            # Clear memory if we hit token limits
            if "context_length_exceeded" in str(e):
                self.memory.clear()
                return "⚠️ Had to clear some conversation history. Please repeat your last message."
            return "❌ An error occurred. Please try again."
    
    def _format_context(self, context: Dict) -> str:
        """Format context data minimally"""
        if context.get("type") in ["welcome", "help", "status"]:
            return f"Type: {context['type']}"
        
        if "error" in context:
            return f"Error: {context['error']}"
        
        if "player" in context:
            player = context["player"]
            return f"Selected Player: {player['player_name']} (🏟️ {player['club']})"
        
        if "results" in context:
            results = context["results"]
            results_str = "\nSearch Results:\n"
            for result in results:
                results_str += f"{result['number']}. {result['player_name']} (🏟️ {result['club']})\n"
            if "message" in context:
                results_str += f"\n{context['message']}"
            return results_str
        
        return ""
    
    def handle_search_state(self, user_input: str, search_results: List[PlayerSearchResult]) -> str:
        """Handle the search state of the conversation"""
        self.search_results = search_results
        
        if not search_results:
            self.current_state = ConversationState.ERROR
            return self.get_response(
                user_input,
                self.current_state,
                {"error": "No players found"}
            )
        
        if len(search_results) == 1:
            # For single result, skip confirmation and go straight to analysis
            self.selected_player = search_results[0]
            self.current_state = ConversationState.COMPLETED
            return self.get_response(
                user_input,
                self.current_state,
                {"player": {
                    "player_name": self.selected_player.player_name,
                    "club": self.selected_player.club,
                    "player_id": self.selected_player.player_id
                }}
            )
        
        # Multiple results, show options
        self.current_state = ConversationState.SHOWING_RESULTS
        formatted_results = []
        for idx, player in enumerate(search_results, 1):
            formatted_results.append({
                "number": idx,
                "player_name": player.player_name,
                "club": player.club,
                "player_id": player.player_id
            })
            
        return self.get_response(
            user_input,
            self.current_state,
            {
                "results": formatted_results,
                "message": f"Found {len(search_results)} players matching '{user_input}'. Please select a number:"
            }
        )

    def handle_selection_state(self, user_input: str) -> str:
        """Handle the selection state of the conversation"""
        try:
            # Try to parse as number
            selection = int(user_input.strip())
            if 1 <= selection <= len(self.search_results):
                self.selected_player = self.search_results[selection - 1]
                # After selection is made, go straight to completed state
                self.current_state = ConversationState.COMPLETED
                return self.get_response(
                    user_input,
                    self.current_state,
                    {"player": {"player_name": self.selected_player.player_name, "club": self.selected_player.club}}
                )
        except ValueError:
            # Try to match by name
            user_input_lower = user_input.lower().strip()
            for result in self.search_results:
                if user_input_lower in result.player_name.lower():
                    self.selected_player = result
                    # After selection is made, go straight to completed state
                    self.current_state = ConversationState.COMPLETED
                    return self.get_response(
                        user_input,
                        self.current_state,
                        {"player": {"player_name": self.selected_player.player_name, "club": self.selected_player.club}}
                    )
        
        return self.get_response(
            user_input,
            self.current_state,
            {"error": "Invalid selection"}
        )
    
    def handle_confirmation_state(self, user_input: str) -> Tuple[str, Optional[PlayerSearchResult]]:
        """Handle the confirmation state of the conversation"""
        user_input_lower = user_input.lower().strip()
        
        if user_input_lower in ['yes', 'y', 'confirm', 'sure', 'ok']:
            if self.selected_player is None:
                self.current_state = ConversationState.ERROR
                return self.get_response(
                    user_input,
                    self.current_state,
                    {"error": "No player selected"}
                ), None
            
            self.current_state = ConversationState.COMPLETED
            response = self.get_response(
                user_input,
                self.current_state,
                {"player": {"player_name": self.selected_player.player_name, "club": self.selected_player.club}}
            )
            return response, self.selected_player
            
        elif user_input_lower in ['no', 'n', 'cancel', 'other', 'different']:
            self.current_state = ConversationState.SHOWING_RESULTS
            # Only send essential player data
            results = [{"player_name": r.player_name, "club": r.club} for r in self.search_results]
            return self.get_response(
                user_input,
                self.current_state,
                {"results": results}
            ), None
        
        return self.get_response(
            user_input,
            self.current_state,
            {"player": {"player_name": self.selected_player.player_name, "club": self.selected_player.club} if self.selected_player else None}
        ), None
    
    def reset(self):
        """Reset the conversation state"""
        self.current_state = ConversationState.SEARCHING
        self.search_results = []
        self.selected_player = None
        self.player_data = None  # Clear stored player data
        self.memory.clear()  # Clear conversation memory
    
    def get_current_state(self) -> ConversationState:
        """Get the current conversation state"""
        return self.current_state

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
        self.user_sessions: Dict[str, LangChainConversationManager] = {}
    
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
                self.user_sessions[user_id] = LangChainConversationManager()
            
            session = self.user_sessions[user_id]
            current_state = session.get_current_state()
            logger.info(f"Processing message for user {user_id} in state {current_state.value}")
            
            # Handle different conversation states
            if current_state == ConversationState.SEARCHING:
                logger.debug(f"User {user_id} initiating search with query: {message}")
                response = self._handle_search_request(user_id, message)
                # If we got a single result, analyze the player immediately
                if len(self.user_sessions[user_id].search_results) == 1:
                    logger.info("Single result found, analyzing player directly")
                    player = self.user_sessions[user_id].search_results[0]
                    self.user_sessions[user_id].selected_player = player
                    self.user_sessions[user_id].current_state = ConversationState.COMPLETED
                    return self._analyze_selected_player(user_id, player, response)
                return response
            elif current_state == ConversationState.SHOWING_RESULTS:
                logger.debug(f"User {user_id} selecting from results: {message}")
                response = session.handle_selection_state(message)
                # If selection was successful and we moved to COMPLETED state, analyze the player
                if session.get_current_state() == ConversationState.COMPLETED and session.selected_player:
                    return self._analyze_selected_player(user_id, session.selected_player, response)
                return response
            elif current_state == ConversationState.COMPLETED:
                # Check if this is a follow-up question about the current player
                if session.selected_player and self._is_follow_up_question(message):
                    logger.info(f"Follow-up question about player {session.selected_player.player_name}")
                    return self._handle_follow_up_question(user_id, session.selected_player, message)
                else:
                    # Assume it's a new player search
                    logger.info(f"Starting new player search, resetting session for user {user_id}")
                    session.reset()
                    return self._handle_search_request(user_id, message)
            else:
                logger.warning(f"Unexpected state {current_state} for user {user_id}, resetting session")
                session.reset()
                return self._handle_search_request(user_id, message)
                
        except Exception as e:
            logger.error(f"Error handling message for user {user_id}: {str(e)}", exc_info=True)
            return "❌ An unexpected error occurred. Please try again."

    def _is_follow_up_question(self, message: str) -> bool:
        """
        Use LLM to determine if a message is a follow-up question about the current player
        """
        try:
            # Create a temporary session for this decision
            temp_session = LangChainConversationManager()
            
            response = temp_session.get_response(
                f"""Determine if this message is a follow-up question about a player or a new player search request.
                
                Message: "{message}"
                
                Rules for classification:
                1. Follow-up questions ask for more details about the same player
                2. New search requests ask to find or look up a different player
                3. When in doubt, classify as a new search
                
                Respond with ONLY one word: 'follow-up' or 'new-search'
                """,
                ConversationState.COMPLETED,
                {}
            )
            
            # Clean up response and check result
            result = response.lower().strip().replace('"', '').replace("'", "")
            return 'follow-up' in result
            
        except Exception as e:
            logger.error(f"Error determining if message is follow-up: {str(e)}")
            return False  # On error, assume it's a new search

    def _handle_follow_up_question(self, user_id: str, player: PlayerSearchResult, question: str) -> str:
        """
        Handle a follow-up question about a player
        """
        try:
            logger.info(f"Handling follow-up question about {player.player_name}: {question}")
            
            session = self.user_sessions[user_id]
            
            # Use the stored data instead of fetching again
            if not session.player_data:
                raise Exception("No player data available")
            
            # Extract essential data from stored data
            essential_data = self._extract_essential_data(session.player_data)
            
            # Generate focused response for the specific question
            response = session.get_response(
                f"""Answer this specific question about {player.player_name}:
                "{question}"
                
                Base your answer on these current stats:
                {essential_data}
                
                Keep the response focused and under 1000 characters.
                """,
                ConversationState.COMPLETED,
                {}
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error handling follow-up question: {str(e)}")
            return f"❌ Sorry, I couldn't get that information about {player.player_name}. Please try asking in a different way or search for another player."
    
    def _extract_player_name(self, message: str) -> str:
        """
        Use LLM to extract player name from natural language input
        
        Args:
            message: Natural language input from user
            
        Returns:
            Extracted player name
        """
        try:
            # Create a temporary session for name extraction
            temp_session = LangChainConversationManager()
            
            response = temp_session.get_response(
                f"""Extract ONLY the football player name from this message. If multiple names are mentioned, extract the most relevant one based on context.
                
                Message: "{message}"
                
                Rules:
                1. Return ONLY the player name, nothing else
                2. Remove any extra words or context
                3. Keep the exact spelling as provided
                4. If no player name is found, return 'unknown'
                
                Example inputs and outputs:
                "Can you tell me about Messi?" -> "Messi"
                "I want to know Kylian Mbappé's stats" -> "Kylian Mbappé"
                "How good is Kevin De Bruyne?" -> "Kevin De Bruyne"
                "What are the stats for last season" -> "unknown"
                """,
                ConversationState.COMPLETED,
                {}
            )
            
            # Clean up response
            player_name = response.strip().replace('"', '').replace("'", "")
            return player_name if player_name.lower() != 'unknown' else message
            
        except Exception as e:
            logger.error(f"Error extracting player name: {str(e)}")
            return message  # Return original message if extraction fails
    
    def _handle_search_request(self, user_id: str, message: str) -> str:
        """Handle initial search request"""
        session = self.user_sessions[user_id]
        logger.info(f"Starting player search for user {user_id} with query: {message}")
        
        try:
            # Extract player name from natural language input
            player_name = self._extract_player_name(message)
            logger.info(f"Extracted player name: {player_name} from message: {message}")
            
            # Perform actual search
            logger.debug(f"Calling web scraper search for query: {player_name}")
            search_results = self.web_scraper.search_players_with_selection(player_name)
            logger.info(f"Found {len(search_results)} players matching query: {player_name}")
            
            # Log search results for debugging
            for idx, result in enumerate(search_results, 1):
                logger.debug(f"Search result {idx}: Player ID: {result.player_id}, "
                           f"Name: {result.player_name}, Club: {result.club}, "
                           f"Confidence: {result.confidence_score}")
            
            # Process results using LangChain
            response = session.handle_search_state(message, search_results)
            
            # If single result, set state to COMPLETED
            if len(search_results) == 1:
                session.current_state = ConversationState.COMPLETED
            
            return response
            
        except Exception as e:
            logger.error(f"Error during player search for user {user_id}: {str(e)}", exc_info=True)
            session.current_state = ConversationState.ERROR
            return session.get_response(
                message,
                ConversationState.ERROR,
                {"error": str(e)}
            )
    
    def _analyze_selected_player(self, user_id: str, selected_player: PlayerSearchResult, confirmation_message: str) -> str:
        """Analyze the selected player and return comprehensive report"""
        try:
            logger.info(f"Starting analysis for player {selected_player.player_name} (ID: {selected_player.player_id})")
            
            session = self.user_sessions[user_id]
            
            # Only fetch data if we don't have it or it's a different player
            if (not session.player_data or 
                not session.selected_player or 
                session.selected_player.player_id != selected_player.player_id):
                
                logger.debug(f"Fetching Transfermarkt data for player ID: {selected_player.player_id}")
                transfermarkt_data = self.web_scraper.get_transfermarkt_data_by_id(selected_player.player_id)
                
                if not transfermarkt_data:
                    raise Exception("Could not fetch Transfermarkt data for player")
                
                # Store the data in session
                session.player_data = transfermarkt_data
                session.selected_player = selected_player
            else:
                logger.debug(f"Using cached data for player {selected_player.player_name}")
                transfermarkt_data = session.player_data

            # Extract essential data
            essential_data = self._extract_essential_data(transfermarkt_data)
            
            # Ensure we're in COMPLETED state
            session.current_state = ConversationState.COMPLETED
            
            # Generate the analysis report
            report = session.get_response(
                f"""Generate a concise analysis report for {selected_player.player_name} based on these key stats:
                {essential_data}
                
                Format the response as:
                1. Current Status (2-3 lines)
                2. Key Statistics (bullet points)
                3. Recent Performance (2-3 lines)
                4. Notable Achievements (if any)
                
                Keep the response under 2000 characters.
                """,
                ConversationState.COMPLETED,
                {}
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Error analyzing player {selected_player.player_name}: {str(e)}", exc_info=True)
            session = self.user_sessions[user_id]
            error_response = session.get_response(
                "Error analyzing player",
                ConversationState.ERROR,
                {"error": str(e)}
            )
            return error_response

    def _extract_essential_data(self, data: dict) -> str:
        """Extract only the essential data points for analysis"""
        essential = {}
        
        # Extract profile data
        if 'profile' in data and isinstance(data['profile'], dict):
            profile = data['profile'].get('raw_profile', {})
            essential['Current Club'] = profile.get('club', 'Unknown')
            essential['Position'] = profile.get('playerMainPosition', 'Unknown')
            essential['Age'] = profile.get('age', 'Unknown')
            essential['Nationality'] = profile.get('birthplaceCountry', 'Unknown')
        
        # Extract latest performance data
        if 'performance_data' in data and isinstance(data['performance_data'], dict):
            records = data['performance_data'].get('performance_records', [])
            if records:
                latest = records[0]  # Get most recent record
                essential['Recent Stats'] = {
                    'Matches': latest.get('matches', 0),
                    'Goals': latest.get('goals', 0),
                    'Assists': latest.get('assists', 0),
                    "Minutes Played": latest.get('minutes_played', 0),
                    "Minutes per Goal": latest.get('minutes_per_goal', 0),
                    "Conceded Goals": latest.get('conceded_goals', 0),
                    "Yellow Cards": latest.get('yellow_cards', 0),
                    "Red Cards": latest.get('red_cards', 0),
                    "Yellow Red Cards": latest.get('yellow_red_cards', 0)
                }
        
        # Extract transfer history summary
        if 'transfer_history' in data and isinstance(data['transfer_history'], dict):
            transfers = data['transfer_history'].get('transfers', [])
            if transfers:
                latest_transfer = transfers[0]
                essential['Latest Transfer'] = {
                    'From': latest_transfer.get('old_club', 'Unknown'),
                    'To': latest_transfer.get('new_club', 'Unknown'),
                    'Date': latest_transfer.get('date', 'Unknown')
                }
        
        # Format as string
        result = []
        for key, value in essential.items():
            if isinstance(value, dict):
                result.append(f"{key}:")
                for k, v in value.items():
                    result.append(f"  - {k}: {v}")
            else:
                result.append(f"{key}: {value}")
        
        return "\n".join(result)

    def _format_personal_info(self, data: dict) -> str:
        """Format personal information from Transfermarkt data"""
        info = []
        for key, value in data.items():
            if key.lower() in ['name', 'age', 'nationality', 'position', 'current_club', 'birth_date', 'height', 'foot']:
                info.append(f"• {key.replace('_', ' ').title()}: {value}")
        return '\n'.join(info) if info else 'No personal information available'

    def _format_career_stats(self, data: dict) -> str:
        """Format career statistics from Transfermarkt data"""
        stats = []
        for key, value in data.items():
            if key.lower() in ['appearances', 'goals', 'assists', 'clean_sheets', 'minutes_played', 'yellow_cards', 'red_cards']:
                stats.append(f"• {key.replace('_', ' ').title()}: {value}")
        return '\n'.join(stats) if stats else 'No career statistics available'

    def _format_transfer_history(self, data: dict) -> str:
        """Format transfer history from Transfermarkt data"""
        history = []
        if 'transfer_history' in data and isinstance(data['transfer_history'], list):
            for transfer in data['transfer_history']:
                history.append(f"• {transfer}")
        return '\n'.join(history) if history else 'No transfer history available'

    def _format_performance_data(self, data: dict) -> str:
        """Format performance data from Transfermarkt data"""
        performance = []
        for key, value in data.items():
            if key.lower() in ['market_value', 'form', 'fitness', 'recent_performances', 'season_stats']:
                performance.append(f"• {key.replace('_', ' ').title()}: {value}")
        return '\n'.join(performance) if performance else 'No performance data available'

    def _format_additional_info(self, data: dict) -> str:
        """Format additional information from Transfermarkt data"""
        additional = []
        for key, value in data.items():
            if key not in ['name', 'age', 'nationality', 'position', 'current_club', 'birth_date', 'height', 'foot',
                          'appearances', 'goals', 'assists', 'clean_sheets', 'minutes_played', 'yellow_cards', 'red_cards',
                          'transfer_history', 'market_value', 'form', 'fitness', 'recent_performances', 'season_stats']:
                additional.append(f"• {key.replace('_', ' ').title()}: {value}")
        return '\n'.join(additional) if additional else 'No additional information available'
    
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
        # Create a temporary session for welcome message
        temp_session = LangChainConversationManager()
        return temp_session.get_response(
            "Generate welcome message",
            ConversationState.SEARCHING,
            {"type": "welcome"}
        )
    
    def get_help_message(self) -> str:
        """Get help message"""
        # Create a temporary session for help message
        temp_session = LangChainConversationManager()
        return temp_session.get_response(
            "Generate help message",
            ConversationState.SEARCHING,
            {"type": "help"}
        )
    
    def get_status_message(self, user_id: str) -> str:
        """Get status message for user"""
        if user_id not in self.user_sessions:
            return "❓ No active session. Start by searching for a player."
            
        session = self.user_sessions[user_id]
        return session.get_response(
            "Generate status message",
            session.get_current_state(),
            {"type": "status"}
        ) 