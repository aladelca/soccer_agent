"""
Simple example of using the Soccer Agent
Shows how to use the conversational agent for player analysis
"""

import os
import sys

# Add the parent directory to the path so we can import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.soccer_agent import SoccerAgent

def main():
    """Example usage of the Soccer Agent"""
    
    # Initialize the agent
    agent = SoccerAgent()
    
    print("⚽ Welcome to Soccer Agent!")
    print("=" * 50)
    
    # Example conversation flow
    user_id = "user123"
    
    # 1. User searches for a player
    print("\n👤 User: Messi")
    response = agent.handle_message(user_id, "Messi")
    print(f"🤖 Agent: {response}")
    
    # 2. User selects from multiple results (if any)
    print("\n👤 User: 1")
    response = agent.handle_message(user_id, "1")
    print(f"🤖 Agent: {response}")
    
    # 3. User confirms selection
    print("\n👤 User: yes")
    response = agent.handle_message(user_id, "yes")
    print(f"🤖 Agent: {response}")
    
    # 4. Check user state
    print(f"\n📊 User state: {agent.get_user_state(user_id)}")
    
    # 5. Reset session and start new search
    print("\n🔄 Resetting session...")
    agent.reset_user_session(user_id)
    
    print("\n👤 User: Ronaldo")
    response = agent.handle_message(user_id, "Ronaldo")
    print(f"🤖 Agent: {response}")

if __name__ == "__main__":
    main() 