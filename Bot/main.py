import os
import logging
from dotenv import load_dotenv
import ccxt
import pandas as pd
import numpy as np
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackContext
import matplotlib.pyplot as plt
import time
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Initialize the exchange
def initialize_exchange():
    exchange = ccxt.bitget({
        'apiKey': os.getenv('bg_e83ddb4bf843a20f6627e844cd879f97'),
        'secret': os.getenv('10abc6d455b30c0dece31fce6c49642b55314200b67f7a845ba0f942c4137292'),
        'enableRateLimit': True,
    })
    return exchange

# Get historical data
async def get_historical_data(exchange, symbol, timeframe='1h', limit=100):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        logger.error(f"Error fetching historical data: {e}")
        return None

# Extended technical analysis function
def extended_technical_analysis(df):
    """
    Perform additional technical analysis calculations
    
    Args:
        df (pandas.DataFrame): DataFrame with historical price data
    
    Returns:
        pandas.DataFrame: DataFrame with additional technical indicators
    """
    # Average True Range (ATR)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean()
    
    # Standard Deviation of Price
    df['price_std'] = df['close'].rolling(window=20).std()
    
    # Momentum Indicator
    df['momentum'] = df['close'].diff(14)
    
    # Relative Vigor Index (RVI)
    df['open_close_diff'] = df['close'] - df['open']
    df['high_low_diff'] = df['high'] - df['low']
    
    rvi_numerator = df['open_close_diff'].rolling(window=10).mean()
    rvi_denominator = df['high_low_diff'].rolling(window=10).mean()
    
    df['rvi'] = rvi_numerator / rvi_denominator
    
    return df

# Calculate indicators
def calculate_indicators(df):
    # Original indicator calculations
    df['ema_short'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema_long'] = df['close'].ewm(span=21, adjust=False).mean()
    
    # Calculate RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['signal']
    
    # Calculate Bollinger Bands
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['std_20'] = df['close'].rolling(window=20).std()
    df['upper_band'] = df['sma_20'] + (df['std_20'] * 2)
    df['lower_band'] = df['sma_20'] - (df['std_20'] * 2)
    
    # Add extended technical analysis
    df = extended_technical_analysis(df)
    
    return df

# Generate trading signals
def generate_signals(df):
    signals = []
    
    # Latest data
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Check for buy signals
    if latest['ema_short'] > latest['ema_long'] and prev['ema_short'] <= prev['ema_long']:
        signals.append(('BUY', 'EMA Crossover'))
    
    if latest['rsi'] < 30:
        signals.append(('BUY', 'RSI Oversold'))
    
    if latest['close'] < latest['lower_band']:
        signals.append(('BUY', 'Price below Lower Bollinger Band'))
    
    if latest['macd'] > latest['signal'] and prev['macd'] <= prev['signal']:
        signals.append(('BUY', 'MACD Bullish Crossover'))
    
    # Add extended analysis signals
    if latest['atr'] < latest['atr'].rolling(window=10).mean():
        signals.append(('BUY', 'Low Volatility (ATR)'))
    
    if latest['momentum'] > 0:
        signals.append(('BUY', 'Positive Momentum'))
    
    # Check for sell signals
    if latest['ema_short'] < latest['ema_long'] and prev['ema_short'] >= prev['ema_long']:
        signals.append(('SELL', 'EMA Crossover'))
    
    if latest['rsi'] > 70:
        signals.append(('SELL', 'RSI Overbought'))
    
    if latest['close'] > latest['upper_band']:
        signals.append(('SELL', 'Price above Upper Bollinger Band'))
    
    if latest['macd'] < latest['signal'] and prev['macd'] >= prev['signal']:
        signals.append(('SELL', 'MACD Bearish Crossover'))
    
    # Add extended analysis sell signals
    if latest['atr'] > latest['atr'].rolling(window=10).mean():
        signals.append(('SELL', 'High Volatility (ATR)'))
    
    if latest['momentum'] < 0:
        signals.append(('SELL', 'Negative Momentum'))
    
    return signals

# Generate chart
async def generate_chart(df, symbol):
    plt.figure(figsize=(15, 12))
    
    # Plot price and EMAs
    plt.subplot(5, 1, 1)
    plt.plot(df['timestamp'], df['close'], label='Price')
    plt.plot(df['timestamp'], df['ema_short'], label='EMA 9')
    plt.plot(df['timestamp'], df['ema_long'], label='EMA 21')
    plt.plot(df['timestamp'], df['upper_band'], 'r--', label='Upper Band')
    plt.plot(df['timestamp'], df['sma_20'], 'g--', label='SMA 20')
    plt.plot(df['timestamp'], df['lower_band'], 'r--', label='Lower Band')
    plt.title(f'{symbol} Price Chart')
    plt.legend()
    
    # Plot RSI
    plt.subplot(5, 1, 2)
    plt.plot(df['timestamp'], df['rsi'])
    plt.axhline(y=70, color='r', linestyle='-')
    plt.axhline(y=30, color='g', linestyle='-')
    plt.title('RSI')
    
    # Plot MACD
    plt.subplot(5, 1, 3)
    plt.plot(df['timestamp'], df['macd'], label='MACD')
    plt.plot(df['timestamp'], df['signal'], label='Signal')
    plt.bar(df['timestamp'], df['macd_hist'], label='Histogram')
    plt.legend()
    plt.title('MACD')
    
    # Plot ATR and Volatility
    plt.subplot(5, 1, 4)
    plt.plot(df['timestamp'], df['atr'], label='ATR')
    plt.plot(df['timestamp'], df['atr'].rolling(window=10).mean(), label='ATR (10-day MA)')
    plt.title('Average True Range (ATR)')
    plt.legend()
    
    # Plot Volume
    plt.subplot(5, 1, 5)
    plt.bar(df['timestamp'], df['volume'])
    plt.title('Volume')
    
    plt.tight_layout()
    
    # Save chart
    chart_path = f"charts/{symbol.replace('/', '_')}_chart.png"
    os.makedirs(os.path.dirname(chart_path), exist_ok=True)
    plt.savefig(chart_path)
    plt.close()
    
    return chart_path

# Telegram bot commands
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
        "/backtest <symbol> <days> - Backtest strategy on a symbol"
    )

async def analyze(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Please provide a symbol, e.g., /analyze BTC/USDT")
        return
    
    symbol = context.args[0].upper()
    
    await update.message.reply_text(f"Analyzing {symbol}...")
    
    try:
        exchange = initialize_exchange()
        df = await get_historical_data(exchange, symbol)
        
        if df is None or df.empty:
            await update.message.reply_text(f"Could not fetch data for {symbol}. Please check the symbol is correct.")
            return
        
        df = calculate_indicators(df)
        signals = generate_signals(df)
        
        # Generate chart
        chart_path = await generate_chart(df, symbol)
        
        # Get current price
        ticker = exchange.fetch_ticker(symbol)
        current_price = ticker['last']
        
        # Prepare analysis message
        analysis = f"Analysis for {symbol}\n\n"
        analysis += f"Current Price: {current_price}\n"
        analysis += f"RSI: {df['rsi'].iloc[-1]:.2f}\n"
        analysis += f"MACD: {df['macd'].iloc[-1]:.2f}\n"
        analysis += f"Signal: {df['signal'].iloc[-1]:.2f}\n\n"
        
        if signals:
            analysis += "Trading Signals:\n"
            for signal, reason in signals:
                analysis += f"- {signal}: {reason}\n"
        else:
            analysis += "No trading signals detected at this time.\n"
        
        # Send analysis and chart
        await update.message.reply_text(analysis)
        await update.message.reply_photo(photo=open(chart_path, 'rb'))
        
    except Exception as e:
        logger.error(f"Error in analyze command: {e}")
        await update.message.reply_text(f"An error occurred: {str(e)}")

async def settings(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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

async def watchlist(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Get user watchlist from context.user_data or use empty list
    watchlist = context.user_data.get('watchlist', [])
    
    if not watchlist:
        await update.message.reply_text("Your watchlist is empty. Add symbols with /add command.")
        return
    
    watchlist_text = "Your Watchlist:\n\n"
    
    try:
        exchange = initialize_exchange()
        
        for symbol in watchlist:
            try:
                ticker = exchange.fetch_ticker(symbol)
                price = ticker['last']
                change = ticker['percentage'] if 'percentage' in ticker else None
                
                watchlist_text += f"{symbol}: ${price:.2f}"
                if change is not None:
                    watchlist_text += f" ({change:.2f}%)"
                watchlist_text += "\n"
                
            except Exception as e:
                watchlist_text += f"{symbol}: Error fetching data\n"
                logger.error(f"Error fetching data for {symbol}: {e}")
        
        await update.message.reply_text(watchlist_text)
        
    except Exception as e:
        logger.error(f"Error in watchlist command: {e}")
        await update.message.reply_text(f"An error occurred: {str(e)}")

async def add_to_watchlist(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Please provide a symbol, e.g., /add BTC/USDT")
        return
    
    symbol = context.args[0].upper()
    
    # Get user watchlist from context.user_data or create empty list
    if 'watchlist' not in context.user_data:
        context.user_data['watchlist'] = []
    
    watchlist = context.user_data['watchlist']
    
    if symbol in watchlist:
        await update.message.reply_text(f"{symbol} is already in your watchlist.")
        return
    
    try:
        exchange = initialize_exchange()
        ticker = exchange.fetch_ticker(symbol)
        
        watchlist.append(symbol)
        await update.message.reply_text(f"Added {symbol} to your watchlist.")
        
    except Exception as e:
        logger.error(f"Error adding {symbol} to watchlist: {e}")
        await update.message.reply_text(f"Could not add {symbol}. Please check the symbol is correct.")

async def remove_from_watchlist(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Please provide a symbol, e.g., /remove BTC/USDT")
        return
    
    symbol = context.args[0].upper()
    
    # Get user watchlist from context.user_data or create empty list
    if 'watchlist' not in context.user_data:
        context.user_data['watchlist'] = []
    
    watchlist = context.user_data['watchlist']
    
    if symbol not in watchlist:
        await update.message.reply_text(f"{symbol} is not in your watchlist.")
        return
    
    watchlist.remove(symbol)
    await update.message.reply_text(f"Removed {symbol} from your watchlist.")

async def balance(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        exchange = initialize_exchange()
        balance = exchange.fetch_balance()
        
        balance_text = "Your Account Balance:\n\n"
        
        # Get non-zero balances
        for currency, data in balance['total'].items():
            if data > 0:
                balance_text += f"{currency}: {data}\n"
        
        if balance_text == "Your Account Balance:\n\n":
            balance_text += "No balance found or all balances are zero."
        
        await update.message.reply_text(balance_text)
        
    except Exception as e:
        logger.error(f"Error in balance command: {e}")
        await update.message.reply_text(f"An error occurred: {str(e)}")

async def backtest(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if len(context.args) < 2:
        await update.message.reply_text("Please provide symbol and days, e.g., /backtest BTC/USDT 30")
        return
    
    symbol = context.args[0].upper()
    
    try:
        days = int(context.args[1])
        if days <= 0 or days > 365:
            await update.message.reply_text("Days must be between 1 and 365.")
            return
    except ValueError:
        await update.message.reply_text("Days must be a number.")
        return
    
    await update.message.reply_text(f"Backtesting {symbol} for {days} days...")
    
    try:
        exchange = initialize_exchange()
        # Calculate how many candles we need based on days
        limit = days * 24  # Assuming 1h timeframe
        
        df = await get_historical_data(exchange, symbol, timeframe='1h', limit=limit)
        
        if df is None or df.empty:
            await update.message.reply_text(f"Could not fetch data for {symbol}. Please check the symbol is correct.")
            return
        
        # Calculate indicators
        df = calculate_indicators(df)
        
        # Initial capital
        initial_capital = 1000.0
        capital = initial_capital
        position = None
        entry_price = 0
        trades = []
        
        # Run through the data
        for i in range(26, len(df) - 1):  # Start after indicators have enough data
            row = df.iloc[i]
            next_row = df.iloc[i+1]
            
            # Buy signal
            if position is None and row['ema_short'] > row['ema_long'] and row['macd'] > row['signal']:
                entry_price = next_row['open']
                position_size = capital * 0.95  # Use 95% of capital
                position = position_size / entry_price
                trades.append({
                    'type': 'BUY',
                    'date': next_row['timestamp'],
                    'price': entry_price,
                    'position': position,
                    'capital': capital
                })
            
            # Sell signal
            elif position is not None and (row['ema_short'] < row['ema_long'] or row['macd'] < row['signal']):
                exit_price = next_row['open']
                capital = position * exit_price
                trades.append({
                    'type': 'SELL',
                    'date': next_row['timestamp'],
                    'price': exit_price,
                    'position': position,
                    'capital': capital
                })
                position = None
        
        # Close any open position at the end
        if position is not None:
            exit_price = df.iloc[-1]['close']
            capital = position * exit_price
            trades.append({
                'type': 'SELL',
                'date': df.iloc[-1]['timestamp'],
                'price': exit_price,
                'position': position,
                'capital': capital
            })
        
        # Calculate stats
        profit = capital - initial_capital
        profit_percent = (profit / initial_capital) * 100
        num_trades = len(trades) // 2  # Each buy/sell pair is one complete trade
        
        # Create backtest report
        report = f"Backtest Results for {symbol} ({days} days):\n\n"
        report += f"Initial Capital: ${initial_capital:.2f}\n"
        report += f"Final Capital: ${capital:.2f}\n"
        report += f"Profit/Loss: ${profit:.2f} ({profit_percent:.2f}%)\n"
        report += f"Number of Trades: {num_trades}\n\n"
        
        if trades:
            report += "Last 5 Trades:\n"
            for trade in trades[-5:]:
                report += f"{trade['date'].strftime('%Y-%m-%d %H:%M')} - {trade['type']} at ${trade['price']:.2f}\n"
        
        await update.message.reply_text(report)
        
    except Exception as e:
        logger.error(f"Error in backtest command: {e}")
        await update.message.reply_text(f"An error occurred: {str(e)}")

# Set up scheduler for periodic tasks
async def scheduled_analysis(context: CallbackContext) -> None:
    """Scheduled task to analyze watchlisted symbols and send alerts"""
    job = context.job
    chat_id = job.chat_id
    exchange = initialize_exchange()
    
    watchlist = context.user_data.get('watchlist', [])
    
    if not watchlist:
        return
    
    for symbol in watchlist:
        try:
            df = await get_historical_data(exchange, symbol)
            
            if df is None or df.empty:
                continue
            
            df = calculate_indicators(df)
            signals = generate_signals(df)
            
            if signals:
                # Send alert for significant signals
                alert = f"🚨 Alert for {symbol} 🚨\n\n"
                alert += f"Current Price: ${df['close'].iloc[-1]:.2f}\n\n"
                alert += "Signals:\n"
                
                for signal, reason in signals:
                    alert += f"- {signal}: {reason}\n"
                
                await context.bot.send_message(chat_id=chat_id, text=alert)
                
        except Exception as e:
            logger.error(f"Error in scheduled analysis for {symbol}: {e}")

async def setup_scheduled_jobs(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Setup job to run every hour"""
    chat_id = update.effective_message.chat_id
    
    # Remove old job if exists
    current_jobs = context.job_queue.get_jobs_by_name(str(chat_id))
    for job in current_jobs:
        job.schedule_removal()
    
    # Add new hourly job
    context.job_queue.run_repeating(
        scheduled_analysis,
        interval=3600,  # 1 hour in seconds
        first=10,  # 10 seconds after command
        chat_id=chat_id,
        name=str(chat_id),
        data=context.user_data
    )
    
    await update.message.reply_text("Automated alerts have been setup. You'll receive notifications for trading signals.")

def main() -> None:
    """Start the bot."""
    # Create application and pass bot token
    application = Application.builder().token(os.getenv('TELEGRAM_BOT_TOKEN')).build()
    
    # Add command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("analyze", analyze))
    application.add_handler(CommandHandler("settings", settings))
    application.add_handler(CommandHandler("watchlist", watchlist))
    application.add_handler(CommandHandler("add", add_to_watchlist))
    application.add_handler(CommandHandler("remove", remove_from_watchlist))
    application.add_handler(CommandHandler("balance", balance))
    application.add_handler(CommandHandler("backtest", backtest))
    application.add_handler(CommandHandler("alerts", setup_scheduled_jobs))
    
    # Start the Bot
    application.run_polling()

if __name__ == '__main__':
    # Create charts directory if it doesn't exist
    os.makedirs('charts', exist_ok=True)
    main()