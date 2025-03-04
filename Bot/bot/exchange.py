import ccxt
import os
import logging

logger = logging.getLogger(__name__)

# Initialize the exchange
def initialize_exchange():
    exchange = ccxt.binance({
        'apiKey': os.getenv('bg_e83ddb4bf843a20f6627e844cd879f97'),
        'secret': os.getenv('10abc6d455b30c0dece31fce6c49642b55314200b67f7a845ba0f942c4137292'),
        'enableRateLimit': True,
    })
    return exchange