import os
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, CallbackContext
import logging

from bot.analysis import (
    analyze_symbol, 
    generate_chart, 
    backtest_strategy,
    scheduled_analysis
)
from bot.exchange import initialize_exchange

logger = logging.getLogger(__name__)

# Command handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Welcome to the Advanced Trading Bot! Use /help to see available commands."
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Available commands:\n"
        "/start - Start the bot\n"
        "/help - Show this help message\n"
        "/analyze <symbol> - Analyze a trading pair (e.g., /analyze BTC/USDT)\n"
        "/settings - Show current bot settings\n"
        "/watchlist - Show your watchlist\n"
        "/add <symbol> - Add a symbol to watchlist\n"
        "/remove <symbol> - Remove a symbol from watchlist\n"
        "/balance - Show your account balance\n"
        "/backtest <symbol> <days> - Backtest strategy on a symbol\n"
        "/alerts - Setup automated alerts"
    )

async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Please provide a symbol, e.g., /analyze BTC/USDT")
        return
    
    symbol = context.args[0].upper()
    await update.message.reply_text(f"Analyzing {symbol}...")
    
    try:
        exchange = initialize_exchange()
        analysis_text, chart_path = await analyze_symbol(exchange, symbol)
        
        # Send analysis and chart
        await update.message.reply_text(analysis_text)
        if chart_path:
            await update.message.reply_photo(photo=open(chart_path, 'rb'))
        
    except Exception as e:
        logger.error(f"Error in analyze command: {e}")
        await update.message.reply_text(f"An error occurred: {str(e)}")

async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Get user settings from context.user_data or use defaults
    settings = context.user_data.get('settings', {
        'risk_percentage': 1.0,
        'timeframe': '1h',
        'auto_trade': False,
        'notification': True
    })
    
    settings_text = "Current Bot Settings:\n\n"
    settings_text += f"Risk Percentage: {settings['risk_percentage']}%\n"
    settings_text += f"Default Timeframe: {settings['timeframe']}\n"
    settings_text += f"Auto-Trading: {'Enabled' if settings['auto_trade'] else 'Disabled'}\n"
    settings_text += f"Notifications: {'Enabled' if settings['notification'] else 'Disabled'}\n\n"
    settings_text += "To change settings, use:\n/set <setting> <value>"
    
    await update.message.reply_text(settings_text)

async def watchlist_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Implementation of watchlist command
    # ... (include the full implementation from the artifact)
    pass

async def add_to_watchlist_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Implementation of add_to_watchlist command
    # ... (include the full implementation from the artifact)
    pass

async def remove_from_watchlist_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Implementation of remove_from_watchlist command
    # ... (include the full implementation from the artifact)
    pass

async def balance_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Implementation of balance command
    # ... (include the full implementation from the artifact)
    pass

async def backtest_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Implementation of backtest command
    # ... (include the full implementation from the artifact)
    pass

async def setup_scheduled_jobs(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Implementation of setup_scheduled_jobs command
    # ... (include the full implementation from the artifact)
    pass

def setup_bot_handlers() -> None:
    """Set up the bot handlers and start the bot."""
    # Create application and pass bot token
    application = Application.builder().token(os.getenv('7571151091:AAH8amRDpUrlEh0Sst5Gaovger-s-sWGUks')).build()
    
    # Add command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("analyze", analyze_command))
    application.add_handler(CommandHandler("settings", settings_command))
    application.add_handler(CommandHandler("watchlist", watchlist_command))
    application.add_handler(CommandHandler("add", add_to_watchlist_command))
    application.add_handler(CommandHandler("remove", remove_from_watchlist_command))
    application.add_handler(CommandHandler("balance", balance_command))
    application.add_handler(CommandHandler("backtest", backtest_command))
    application.add_handler(CommandHandler("alerts", setup_scheduled_jobs))
    
    # Start the Bot
    application.run_polling()