"""
Example Telegram Bot Implementation for Soccer Agent
Shows how to use the PlayerSelectionFlow with Telegram interface
"""

import os
import sys
import logging
from typing import NoReturn

# Add the parent directory to the path so we can import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from src.soccer_agent import SoccerAgent

# Setup logging with line numbers
logging.basicConfig(
    format='%(asctime)s - [%(filename)s:%(lineno)d] - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Initialize the soccer agent
agent = SoccerAgent()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    if update.message is None:
        return
    welcome_message = agent.get_welcome_message()
    await update.message.reply_text(welcome_message, parse_mode='Markdown')

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command"""
    if update.message is None:
        return
    help_message = agent.get_help_message()
    await update.message.reply_text(help_message, parse_mode='Markdown')

async def reset_session(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /reset command"""
    if update.message is None or update.effective_user is None:
        return
    user_id = str(update.effective_user.id)
    agent.reset_user_session(user_id)
    await update.message.reply_text("🔄 Session reset. You can start a new search by typing a player's name.")

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /status command"""
    if update.message is None or update.effective_user is None:
        return
    user_id = str(update.effective_user.id)
    status_message = agent.get_status_message(user_id)
    await update.message.reply_text(status_message, parse_mode='Markdown')

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming messages"""
    if update.message is None or update.effective_user is None:
        return
    user_id = str(update.effective_user.id)
    message_text = update.message.text
    if message_text is None:
        return
    
    # Get response from agent
    response = agent.handle_message(user_id, message_text)
    await update.message.reply_text(response, parse_mode='Markdown')

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    """Handle errors"""
    logger.error(f"Update {update} caused error {context.error}")
    if isinstance(update, Update) and update.message:
        error_message = "❌ Ocurrió un error inesperado. Por favor intenta de nuevo o usa /reset para reiniciar tu sesión."
        await update.message.reply_text(error_message)

def main() -> NoReturn:
    """Main function to run the bot"""
    # Get bot token from environment variable
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    
    if not bot_token:
        print("❌ Error: TELEGRAM_BOT_TOKEN environment variable not set")
        print("Por favor configura tu token de bot de Telegram:")
        print("export TELEGRAM_BOT_TOKEN='tu_token_aqui'")
        sys.exit(1)
    
    # Create application with minimal configuration
    app = Application.builder().token(bot_token).build()
    
    # Add handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("reset", reset_session))
    app.add_handler(CommandHandler("status", status_command))
    
    # Add message handler (must be last)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Add error handler
    app.add_error_handler(error_handler)
    
    # Start the bot
    print("🤖 Bot iniciado. Presiona Ctrl+C para detener.")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main() 