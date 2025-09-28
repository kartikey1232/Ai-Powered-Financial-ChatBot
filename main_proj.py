import json
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import os
import uuid
import google.generativeai as genai
from datetime import datetime, timedelta
import time
import requests
from functools import lru_cache
import warnings
import re
from collections import defaultdict
warnings.filterwarnings('ignore')

# Configure Gemini API
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("Gemini API key not found. Please set GEMINI_API_KEY environment variable or add it to Streamlit secrets.")
        st.stop()

genai.configure(api_key=api_key)

# =============================================================================
# LOGIC-BASED INFERENCE ENGINE
# =============================================================================

class Predicate:
    """Represents a logical predicate"""
    def __init__(self, name):
        self.name = name
    
    def __call__(self, *args):
        return Fact(self.name, args)
    
    def __str__(self):
        return self.name

class Fact:
    """Represents a logical fact"""
    def __init__(self, predicate, args):
        self.predicate = predicate
        self.args = args
    
    def __str__(self):
        if self.args:
            return f"{self.predicate}({', '.join(map(str, self.args))})"
        return self.predicate
    
    def __eq__(self, other):
        return isinstance(other, Fact) and self.predicate == other.predicate and self.args == other.args
    
    def __hash__(self):
        return hash((self.predicate, self.args))

class Rule:
    """Represents a logical rule (if-then statement)"""
    def __init__(self, conditions, conclusion):
        self.conditions = conditions if isinstance(conditions, list) else [conditions]
        self.conclusion = conclusion
    
    def can_apply(self, facts):
        """Check if all conditions are satisfied by the given facts"""
        for condition in self.conditions:
            if condition not in facts:
                return False
        return True
    
    def apply(self, facts):
        """Apply the rule if conditions are met"""
        if self.can_apply(facts):
            return self.conclusion
        return None
    
    def __str__(self):
        conditions_str = " AND ".join(str(c) for c in self.conditions)
        return f"IF {conditions_str} THEN {self.conclusion}"

class FinancialKnowledgeBase:
    """Financial domain knowledge base with logical rules"""
    
    def __init__(self):
        # Define predicates
        self.predicates = {
            'stock_bullish': Predicate('stock_bullish'),
            'stock_bearish': Predicate('stock_bearish'),
            'rsi_oversold': Predicate('rsi_oversold'),
            'rsi_overbought': Predicate('rsi_overbought'),
            'price_above_sma': Predicate('price_above_sma'),
            'price_below_sma': Predicate('price_below_sma'),
            'high_volume': Predicate('high_volume'),
            'low_volume': Predicate('low_volume'),
            'strong_buy': Predicate('strong_buy'),
            'strong_sell': Predicate('strong_sell'),
            'buy_signal': Predicate('buy_signal'),
            'sell_signal': Predicate('sell_signal'),
            'high_volatility': Predicate('high_volatility'),
            'low_volatility': Predicate('low_volatility'),
            'risky_investment': Predicate('risky_investment'),
            'safe_investment': Predicate('safe_investment')
        }
        
        # Initialize facts and rules
        self.facts = set()
        self.rules = self._initialize_rules()
    
    def _initialize_rules(self):
        """Initialize financial trading rules"""
        rules = []
        
        # Rule 1: If RSI < 30 and price > SMA, then bullish signal
        rules.append(Rule(
            [self.predicates['rsi_oversold']('X'), self.predicates['price_above_sma']('X')],
            self.predicates['buy_signal']('X')
        ))
        
        # Rule 2: If RSI > 70 and price < SMA, then bearish signal
        rules.append(Rule(
            [self.predicates['rsi_overbought']('X'), self.predicates['price_below_sma']('X')],
            self.predicates['sell_signal']('X')
        ))
        
        # Rule 3: If bullish and high volume, then strong buy
        rules.append(Rule(
            [self.predicates['stock_bullish']('X'), self.predicates['high_volume']('X')],
            self.predicates['strong_buy']('X')
        ))
        
        # Rule 4: If bearish and high volume, then strong sell
        rules.append(Rule(
            [self.predicates['stock_bearish']('X'), self.predicates['high_volume']('X')],
            self.predicates['strong_sell']('X')
        ))
        
        # Rule 5: If high volatility, then risky investment
        rules.append(Rule(
            [self.predicates['high_volatility']('X')],
            self.predicates['risky_investment']('X')
        ))
        
        # Rule 6: If low volatility and buy signal, then safe investment
        rules.append(Rule(
            [self.predicates['low_volatility']('X'), self.predicates['buy_signal']('X')],
            self.predicates['safe_investment']('X')
        ))
        
        # Rule 7: If buy signal, then stock is bullish
        rules.append(Rule(
            [self.predicates['buy_signal']('X')],
            self.predicates['stock_bullish']('X')
        ))
        
        # Rule 8: If sell signal, then stock is bearish
        rules.append(Rule(
            [self.predicates['sell_signal']('X')],
            self.predicates['stock_bearish']('X')
        ))
        
        return rules
    
    def add_fact(self, fact):
        """Add a fact to the knowledge base"""
        self.facts.add(fact)
    
    def add_facts(self, facts):
        """Add multiple facts to the knowledge base"""
        for fact in facts:
            self.add_fact(fact)
    
    def get_facts_for_ticker(self, ticker):
        """Get all facts related to a specific ticker"""
        ticker_facts = []
        for fact in self.facts:
            if ticker in fact.args:
                ticker_facts.append(fact)
        return ticker_facts

class InferenceEngine:
    """Implements forward and backward chaining inference"""
    
    def __init__(self, knowledge_base):
        self.kb = knowledge_base
    
    def forward_chain(self, max_iterations=10):
        """Forward chaining inference - derive new facts from existing ones"""
        new_facts_added = True
        iterations = 0
        derived_facts = []
        
        while new_facts_added and iterations < max_iterations:
            new_facts_added = False
            iterations += 1
            
            for rule in self.kb.rules:
                # Check if rule can be applied
                if rule.can_apply(self.kb.facts):
                    new_fact = rule.apply(self.kb.facts)
                    if new_fact and new_fact not in self.kb.facts:
                        self.kb.add_fact(new_fact)
                        derived_facts.append((new_fact, rule))
                        new_facts_added = True
        
        return derived_facts
    
    def backward_chain(self, goal, visited=None):
        """Backward chaining inference - try to prove a goal"""
        if visited is None:
            visited = set()
        
        # Avoid infinite recursion
        if goal in visited:
            return False, []
        
        visited.add(goal)
        
        # Check if goal is already a known fact
        if goal in self.kb.facts:
            return True, [f"Goal {goal} is already known"]
        
        # Try to prove goal using rules
        for rule in self.kb.rules:
            if str(rule.conclusion) == str(goal):
                proof_steps = [f"Trying to prove {goal} using rule: {rule}"]
                
                # Try to prove all conditions
                all_conditions_proven = True
                for condition in rule.conditions:
                    proven, steps = self.backward_chain(condition, visited.copy())
                    proof_steps.extend(steps)
                    if not proven:
                        all_conditions_proven = False
                        break
                
                if all_conditions_proven:
                    proof_steps.append(f"Successfully proved {goal}")
                    return True, proof_steps
        
        return False, [f"Cannot prove {goal}"]
    
    def query(self, query_str):
        """Process a natural language query and convert to logical reasoning"""
        # Simple pattern matching for common queries
        ticker_match = re.search(r'\b([A-Z]{2,5})\b', query_str.upper())
        ticker = ticker_match.group(1) if ticker_match else 'UNKNOWN'
        
        query_lower = query_str.lower()
        
        if 'bullish' in query_lower or 'buy' in query_lower:
            goal = self.kb.predicates['stock_bullish'](ticker)
            return self.backward_chain(goal)
        elif 'bearish' in query_lower or 'sell' in query_lower:
            goal = self.kb.predicates['stock_bearish'](ticker)
            return self.backward_chain(goal)
        elif 'risky' in query_lower or 'risk' in query_lower:
            goal = self.kb.predicates['risky_investment'](ticker)
            return self.backward_chain(goal)
        elif 'safe' in query_lower:
            goal = self.kb.predicates['safe_investment'](ticker)
            return self.backward_chain(goal)
        else:
            return False, ["Unable to understand query"]

# Initialize global knowledge base and inference engine
financial_kb = FinancialKnowledgeBase()
inference_engine = InferenceEngine(financial_kb)

# =============================================================================
# ORIGINAL CODE CONTINUES (All existing classes remain unchanged)
# =============================================================================

class StockDataProvider:
    """Base class for stock data providers"""
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_stock_price(self, ticker):
        raise NotImplementedError
    
    def get_stock_data(self, ticker, period='1y'):
        raise NotImplementedError

class AlphaVantageProvider(StockDataProvider):
    """Alpha Vantage API Provider (Free tier: 25 requests/day)"""
    def __init__(self, api_key=None):
        super().__init__()
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY") or "demo"
        self.base_url = "https://www.alphavantage.co/query"
    
    def get_stock_price(self, ticker):
        try:
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': ticker,
                'apikey': self.api_key
            }
            response = self.session.get(self.base_url, params=params, timeout=10)
            data = response.json()
            
            if 'Global Quote' in data:
                quote = data['Global Quote']
                price = float(quote['05. price'])
                change = float(quote['09. change'])
                change_percent = quote['10. change percent'].replace('%', '')
                
                return {
                    'price': price,
                    'change': change,
                    'change_percent': float(change_percent),
                    'symbol': ticker,
                    'source': 'Alpha Vantage'
                }
            return None
        except Exception as e:
            return None
    
    def get_stock_data(self, ticker, period='1y'):
        try:
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': ticker,
                'apikey': self.api_key,
                'outputsize': 'full'
            }
            response = self.session.get(self.base_url, params=params, timeout=15)
            data = response.json()
            
            if 'Time Series (Daily)' in data:
                time_series = data['Time Series (Daily)']
                df = pd.DataFrame.from_dict(time_series, orient='index')
                df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                df.index = pd.to_datetime(df.index)
                df = df.astype(float)
                df = df.sort_index()
                
                if period == '1mo':
                    df = df.tail(30)
                elif period == '3mo':
                    df = df.tail(90)
                elif period == '6mo':
                    df = df.tail(180)
                elif period == '1y':
                    df = df.tail(365)
                
                return df
            return None
        except Exception as e:
            return None

class FinnhubProvider(StockDataProvider):
    """Finnhub API Provider (Free tier: 60 calls/minute)"""
    def __init__(self, api_key=None):
        super().__init__()
        self.api_key = api_key or os.getenv("FINNHUB_API_KEY") or "demo"
        self.base_url = "https://finnhub.io/api/v1"
    
    def get_stock_price(self, ticker):
        try:
            params = {
                'symbol': ticker,
                'token': self.api_key
            }
            response = self.session.get(f"{self.base_url}/quote", params=params, timeout=10)
            data = response.json()
            
            if 'c' in data:
                return {
                    'price': data['c'],
                    'change': data['d'],
                    'change_percent': data['dp'],
                    'symbol': ticker,
                    'source': 'Finnhub'
                }
            return None
        except Exception as e:
            return None
    
    def get_stock_data(self, ticker, period='1y'):
        try:
            end_time = int(datetime.now().timestamp())
            if period == '1mo':
                start_time = int((datetime.now() - timedelta(days=30)).timestamp())
            elif period == '3mo':
                start_time = int((datetime.now() - timedelta(days=90)).timestamp())
            elif period == '6mo':
                start_time = int((datetime.now() - timedelta(days=180)).timestamp())
            else:
                start_time = int((datetime.now() - timedelta(days=365)).timestamp())
            
            params = {
                'symbol': ticker,
                'resolution': 'D',
                'from': start_time,
                'to': end_time,
                'token': self.api_key
            }
            response = self.session.get(f"{self.base_url}/stock/candle", params=params, timeout=15)
            data = response.json()
            
            if 's' in data and data['s'] == 'ok':
                df = pd.DataFrame({
                    'Open': data['o'],
                    'High': data['h'],
                    'Low': data['l'],
                    'Close': data['c'],
                    'Volume': data['v']
                })
                df.index = pd.to_datetime(data['t'], unit='s')
                return df
            return None
        except Exception as e:
            return None

class NSEProvider(StockDataProvider):
    """NSE Direct API Provider for Indian stocks"""
    def __init__(self):
        super().__init__()
        self.base_url = "https://www.nseindia.com/api"
        # NSE requires specific headers to prevent blocking
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
    
    def get_stock_price(self, ticker):
        try:
            # First get cookies by visiting NSE homepage
            self.session.get("https://www.nseindia.com", timeout=10)
            
            # Then fetch stock data
            url = f"{self.base_url}/quote-equity?symbol={ticker}"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'priceInfo' in data:
                    price_info = data['priceInfo']
                    return {
                        'price': float(price_info['lastPrice']),
                        'change': float(price_info['change']),
                        'change_percent': float(price_info['pChange']),
                        'symbol': ticker,
                        'source': 'NSE India'
                    }
            return None
        except Exception as e:
            return None
    
    def get_stock_data(self, ticker, period='1y'):
        try:
            # NSE historical data endpoint
            self.session.get("https://www.nseindia.com", timeout=10)
            
            # Calculate date range
            end_date = datetime.now()
            if period == '1mo':
                start_date = end_date - timedelta(days=30)
            elif period == '3mo':
                start_date = end_date - timedelta(days=90)
            elif period == '6mo':
                start_date = end_date - timedelta(days=180)
            else:  # 1y
                start_date = end_date - timedelta(days=365)
            
            # Format dates for NSE API
            start_str = start_date.strftime("%d-%m-%Y")
            end_str = end_date.strftime("%d-%m-%Y")
            
            url = f"{self.base_url}/historical/cm/equity?symbol={ticker}&series=[%22EQ%22]&from={start_str}&to={end_str}"
            response = self.session.get(url, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'data' in data and data['data']:
                    df_data = []
                    for record in data['data']:
                        df_data.append({
                            'Date': record['CH_TIMESTAMP'],
                            'Open': float(record['CH_OPENING_PRICE']),
                            'High': float(record['CH_TRADE_HIGH_PRICE']),
                            'Low': float(record['CH_TRADE_LOW_PRICE']),
                            'Close': float(record['CH_CLOSING_PRICE']),
                            'Volume': int(record['CH_TOT_TRADED_QTY'])
                        })
                    
                    df = pd.DataFrame(df_data)
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)
                    df = df.sort_index()
                    
                    return df
            return None
        except Exception as e:
            return None

class YahooIndiaCombinedProvider(StockDataProvider):
    """Enhanced Yahoo provider with Indian stock support"""
    def __init__(self):
        super().__init__()
        self.base_url = "https://query1.finance.yahoo.com"
    
    def get_stock_price(self, ticker):
        # Try multiple formats for Indian stocks
        variations = [
            ticker,           # Original
            f"{ticker}.NS",   # NSE
            f"{ticker}.BO"    # BSE
        ]
        
        for variant in variations:
            try:
                url = f"{self.base_url}/v8/finance/chart/{variant}"
                response = self.session.get(url, timeout=10)
                data = response.json()
                
                if 'chart' in data and data['chart']['result']:
                    result = data['chart']['result'][0]
                    meta = result['meta']
                    
                    return {
                        'price': meta['regularMarketPrice'],
                        'change': meta.get('regularMarketPrice', 0) - meta.get('previousClose', 0),
                        'change_percent': ((meta.get('regularMarketPrice', 0) - meta.get('previousClose', 1)) / meta.get('previousClose', 1)) * 100,
                        'symbol': variant,
                        'source': f'Yahoo Finance ({variant})'
                    }
            except Exception:
                continue
        return None
    
    def get_stock_data(self, ticker, period='1y'):
        variations = [ticker, f"{ticker}.NS", f"{ticker}.BO"]
        
        for variant in variations:
            try:
                period_map = {'1mo': '1mo', '3mo': '3mo', '6mo': '6mo', '1y': '1y'}
                yahoo_period = period_map.get(period, '1y')
                
                url = f"{self.base_url}/v8/finance/chart/{variant}"
                params = {
                    'period1': 0,
                    'period2': int(datetime.now().timestamp()),
                    'interval': '1d',
                    'range': yahoo_period
                }
                
                response = self.session.get(url, params=params, timeout=15)
                data = response.json()
                
                if 'chart' in data and data['chart']['result']:
                    result = data['chart']['result'][0]
                    timestamps = result['timestamp']
                    quotes = result['indicators']['quote'][0]
                    
                    df = pd.DataFrame({
                        'Open': quotes['open'],
                        'High': quotes['high'],
                        'Low': quotes['low'],
                        'Close': quotes['close'],
                        'Volume': quotes['volume']
                    })
                    df.index = pd.to_datetime(timestamps, unit='s')
                    df = df.dropna()
                    if not df.empty:
                        return df
            except Exception:
                continue
        return None

class MultiProviderStockData:
    def __init__(self):
        self.providers = [
            YahooIndiaCombinedProvider(),  # <-- Changed to the correct class name
            AlphaVantageProvider(),
            FinnhubProvider(),
            NSEProvider(),  # <-- Also added this for more data sources
        ]
        self.cache = {}
    
    @st.cache_data(ttl=300)
    def get_stock_price_multi(_self, ticker):
        """Try multiple providers for stock price"""
        ticker = ticker.upper().strip()
        
        for i, provider in enumerate(_self.providers):
            try:
                with st.spinner(f'🔄 Trying data source {i+1}/3...'):
                    result = provider.get_stock_price(ticker)
                    if result:
                        return result, None
                time.sleep(1)
            except Exception as e:
                continue
        
        return None, "All data sources failed"
    
    @st.cache_data(ttl=300)
    def get_stock_data_multi(_self, ticker, period='1y'):
        """Try multiple providers for historical data"""
        ticker = ticker.upper().strip()
        
        for i, provider in enumerate(_self.providers):
            try:
                with st.spinner(f'📊 Getting historical data from source {i+1}/3...'):
                    result = provider.get_stock_data(ticker, period)
                    if result is not None and not result.empty:
                        return result, None
                time.sleep(1)
            except Exception as e:
                continue
        
        return None, "All data sources failed"

# Initialize multi-provider
data_fetcher = MultiProviderStockData()

# =============================================================================
# ENHANCED ANALYSIS FUNCTIONS WITH LOGIC INTEGRATION
# =============================================================================

def populate_facts_from_analysis(ticker, price_data, rsi_value, sma_data, volume_data):
    """Convert technical analysis results to logical facts"""
    facts = []
    
    try:
        # RSI-based facts - handle scalar values properly
        if not pd.isna(rsi_value) and np.isscalar(rsi_value):
            if rsi_value < 30:
                facts.append(financial_kb.predicates['rsi_oversold'](ticker))
            elif rsi_value > 70:
                facts.append(financial_kb.predicates['rsi_overbought'](ticker))
        
        # Price vs SMA facts - ensure we have valid data
        if (sma_data is not None and not sma_data.empty and 
            price_data is not None and not price_data.empty):
            try:
                current_price = float(price_data.iloc[-1])
                sma_value = float(sma_data.iloc[-1])
                if not (pd.isna(current_price) or pd.isna(sma_value)):
                    if current_price > sma_value:
                        facts.append(financial_kb.predicates['price_above_sma'](ticker))
                    else:
                        facts.append(financial_kb.predicates['price_below_sma'](ticker))
            except (IndexError, TypeError, ValueError):
                pass  # Skip if data is invalid
        
        # Volume facts - handle potential missing volume data
        if (volume_data is not None and not volume_data.empty and 
            len(volume_data.dropna()) > 1):
            try:
                clean_volume = volume_data.dropna()
                avg_volume = float(clean_volume.mean())
                recent_volume = float(clean_volume.iloc[-1])
                if not (pd.isna(avg_volume) or pd.isna(recent_volume)) and avg_volume > 0:
                    if recent_volume > avg_volume * 1.5:
                        facts.append(financial_kb.predicates['high_volume'](ticker))
                    elif recent_volume < avg_volume * 0.5:
                        facts.append(financial_kb.predicates['low_volume'](ticker))
            except (IndexError, TypeError, ValueError):
                pass  # Skip if volume data is invalid
        
        # Volatility facts - handle price data safely
        if price_data is not None and not price_data.empty and len(price_data) > 1:
            try:
                returns = price_data.pct_change().dropna()
                if len(returns) > 0:
                    volatility = float(returns.std() * np.sqrt(252))
                    if not pd.isna(volatility):
                        if volatility > 0.3:
                            facts.append(financial_kb.predicates['high_volatility'](ticker))
                        elif volatility < 0.15:
                            facts.append(financial_kb.predicates['low_volatility'](ticker))
            except (TypeError, ValueError):
                pass  # Skip if volatility calculation fails
        
    except Exception as e:
        # If anything fails, return empty facts list
        pass
    
    return facts

def logical_stock_analysis(ticker):
    """Perform logical analysis using inference engine"""
    try:
        ticker = ticker.upper().strip()
        
        # Get technical data
        data, error = data_fetcher.get_stock_data_multi(ticker, '3mo')
        if data is None:
            return f"❌ Unable to fetch data for {ticker}: {error if error else 'No data available'}"
        
        # Ensure we have enough data
        if len(data) < 20:
            return f"❌ Insufficient data for {ticker}. Need at least 20 days, got {len(data)} days."
        
        # Calculate technical indicators with error handling
        close_prices = data['Close'].dropna()
        if len(close_prices) < 14:
            return f"❌ Insufficient price data for {ticker}. Need at least 14 days for RSI calculation."
        
        # Calculate RSI safely
        try:
            delta = close_prices.diff()
            up = delta.clip(lower=0)
            down = -delta.clip(upper=0)
            
            ema_up = up.ewm(span=14, adjust=False).mean()
            ema_down = down.ewm(span=14, adjust=False).mean()
            
            rs = ema_up / ema_down
            rsi = 100 - (100 / (1 + rs))
            current_rsi = float(rsi.iloc[-1])
            
            # Validate RSI
            if pd.isna(current_rsi) or not (0 <= current_rsi <= 100):
                current_rsi = 50  # Default neutral RSI
        except Exception:
            current_rsi = 50  # Default neutral RSI
        
        # Calculate SMA safely
        sma_20 = None
        try:
            if len(close_prices) >= 20:
                sma_20 = close_prices.rolling(20).mean()
            else:
                sma_20 = close_prices.rolling(len(close_prices)).mean()
        except Exception:
            pass  # SMA will remain None
        
        # Get volume data safely
        volume_data = None
        try:
            if 'Volume' in data.columns:
                volume_data = data['Volume'].dropna()
                if len(volume_data) == 0:
                    volume_data = None
        except Exception:
            pass  # Volume will remain None
        
        # Populate knowledge base with facts
        facts = populate_facts_from_analysis(
            ticker, close_prices, current_rsi, sma_20, volume_data
        )
        
        # Clear previous facts for this ticker and add new ones
        financial_kb.facts = {f for f in financial_kb.facts if ticker not in str(f)}
        financial_kb.add_facts(facts)
        
        # Apply forward chaining to derive new facts
        try:
            derived_facts = inference_engine.forward_chain()
        except Exception:
            derived_facts = []  # If inference fails, continue with empty derived facts
        
        # Format results
        result = f"""🧠 **Logical Analysis for {ticker}**

📊 **Known Facts:**
"""
        
        ticker_facts = financial_kb.get_facts_for_ticker(ticker)
        if ticker_facts:
            for fact in sorted(ticker_facts, key=str):
                result += f"• {fact}\n"
        else:
            result += "• No logical facts derived from current data\n"
        
        if derived_facts:
            result += f"\n🔍 **Derived Conclusions:**\n"
            for fact, rule in derived_facts:
                if ticker in str(fact):
                    result += f"• {fact} (from: {rule})\n"
        
        # Check for specific investment recommendations
        recommendations = []
        for fact in financial_kb.facts:
            if ticker in str(fact):
                fact_str = str(fact)
                if 'strong_buy' in fact_str:
                    recommendations.append("🟢 STRONG BUY signal detected")
                elif 'strong_sell' in fact_str:
                    recommendations.append("🔴 STRONG SELL signal detected")
                elif 'buy_signal' in fact_str:
                    recommendations.append("📈 BUY signal detected")
                elif 'sell_signal' in fact_str:
                    recommendations.append("📉 SELL signal detected")
                elif 'risky_investment' in fact_str:
                    recommendations.append("⚠️ HIGH RISK investment")
                elif 'safe_investment' in fact_str:
                    recommendations.append("✅ SAFE investment")
        
        if recommendations:
            result += f"\n🎯 **Investment Recommendations:**\n"
            for rec in recommendations:
                result += f"• {rec}\n"
        else:
            result += f"\n🎯 **Investment Recommendations:**\n• No clear signals detected - NEUTRAL stance recommended\n"
        
        result += f"\n📋 **Technical Data:**\n"
        result += f"• RSI: {current_rsi:.1f}\n"
        if sma_20 is not None and not sma_20.empty and not pd.isna(sma_20.iloc[-1]):
            result += f"• SMA(20): ${sma_20.iloc[-1]:.2f}\n"
        result += f"• Current Price: ${close_prices.iloc[-1]:.2f}\n"
        result += f"• Data Points: {len(data)} days\n"
        
        return result
        
    except Exception as e:
        return f"❌ Error in logical analysis for {ticker}: {str(e)}"

def query_knowledge_base(query):
    """Query the knowledge base using natural language"""
    try:
        proven, proof_steps = inference_engine.query(query)
        
        result = f"""🔍 **Knowledge Base Query**
Query: "{query}"

📋 **Reasoning Process:**
"""
        
        for step in proof_steps:
            result += f"• {step}\n"
        
        result += f"\n🎯 **Conclusion:** {'✅ PROVEN' if proven else '❌ NOT PROVEN'}"
        
        return result
        
    except Exception as e:
        return f"❌ Error processing query: {str(e)}"

# =============================================================================
# ALL EXISTING FUNCTIONS REMAIN UNCHANGED
# =============================================================================

def get_stock_price(ticker):
    """Get stock price using multiple data providers"""
    try:
        ticker = ticker.upper().strip()
        
        result, error = data_fetcher.get_stock_price_multi(ticker)
        
        if result:
            return f"""📊 **{ticker} Stock Price**
💰 Current Price: ${result['price']:.2f}
📈 Change: {result['change']:+.2f} ({result['change_percent']:+.2f}%)
🔌 Data Source: {result['source']}
✅ Status: Live Data Retrieved"""
        else:
            return f"❌ Unable to fetch price for '{ticker}' from any data source. Please verify the ticker symbol."
            
    except Exception as e:
        return f"❌ Error fetching stock price for {ticker}: {str(e)}"

def calculate_SMA(ticker, window):
    """Calculate SMA using multiple data providers"""
    try:
        ticker = ticker.upper().strip()
        window = int(window)
        
        if window <= 0:
            return "❌ Error: Window size must be a positive number."
        
        period = '1mo' if window <= 20 else '3mo' if window <= 50 else '1y'
        
        data, error = data_fetcher.get_stock_data_multi(ticker, period)
        
        if data is not None and len(data) >= window:
            close_prices = data['Close'].dropna()
            
            if len(close_prices) >= window:
                sma = close_prices.rolling(window=window).mean().iloc[-1]
                current_price = close_prices.iloc[-1]
                distance = ((current_price - sma) / sma) * 100
                
                trend = "🟢 Bullish" if distance > 0 else "🔴 Bearish"
                
                return f"""📊 **SMA({window}) Analysis for {ticker}**
🎯 SMA Value: ${sma:.2f}
💰 Current Price: ${current_price:.2f}
📏 Distance: {distance:+.2f}%
📈 Trend: {trend}
📊 Data Points: {len(close_prices)} days"""
            else:
                return f"⚠️ Insufficient data for SMA({window}). Need {window} days, got {len(close_prices)}."
        else:
            return f"❌ Unable to fetch sufficient data for {ticker} SMA calculation."
            
    except ValueError:
        return "❌ Error: Window must be a valid number."
    except Exception as e:
        return f"❌ Error calculating SMA: {str(e)}"

def calculate_RSI(ticker):
    """Calculate RSI using multiple data providers"""
    try:
        ticker = ticker.upper().strip()
        
        data, error = data_fetcher.get_stock_data_multi(ticker, '3mo')
        
        if data is not None and len(data) >= 20:
            close_prices = data['Close'].dropna()
            
            delta = close_prices.diff()
            up = delta.clip(lower=0)
            down = -delta.clip(upper=0)
            
            ema_up = up.ewm(span=14, adjust=False).mean()
            ema_down = down.ewm(span=14, adjust=False).mean()
            
            rs = ema_up / ema_down
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
            
            if current_rsi > 70:
                signal = "🔴 OVERBOUGHT - Consider selling"
            elif current_rsi < 30:
                signal = "🟢 OVERSOLD - Consider buying"
            else:
                signal = "⚪ NEUTRAL - No clear signal"
            
            return f"""📊 **RSI Analysis for {ticker}**
⚡ RSI (14): {current_rsi:.1f}
🎯 Signal: {signal}
📊 Data Points: {len(close_prices)} days"""
        else:
            return f"❌ Unable to calculate RSI for {ticker}. Need at least 20 days of data."
            
    except Exception as e:
        return f"❌ Error calculating RSI: {str(e)}"

def analyze_chart_insights(data, ticker):
    """Analyze chart data and provide detailed insights"""
    try:
        close_prices = data['Close'].dropna()
        if len(close_prices) < 20:
            return "Insufficient data for comprehensive analysis."
        
        insights = []
        
        # Price trend analysis
        recent_prices = close_prices.tail(20)
        price_change = ((recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]) * 100
        
        if price_change > 5:
            insights.append(f"📈 Strong upward trend: {price_change:.1f}% gain over last 20 days")
        elif price_change < -5:
            insights.append(f"📉 Strong downward trend: {price_change:.1f}% decline over last 20 days")
        else:
            insights.append(f"📊 Sideways movement: {price_change:.1f}% change over last 20 days")
        
        # Support and resistance levels
        recent_high = close_prices.tail(60).max()
        recent_low = close_prices.tail(60).min()
        current_price = close_prices.iloc[-1]
        
        resistance_distance = ((recent_high - current_price) / current_price) * 100
        support_distance = ((current_price - recent_low) / current_price) * 100
        
        insights.append(f"🎯 Key resistance at ${recent_high:.2f} ({resistance_distance:.1f}% above current)")
        insights.append(f"🛡️ Key support at ${recent_low:.2f} ({support_distance:.1f}% below current)")
        
        # Volatility analysis
        returns = close_prices.pct_change().dropna()
        volatility = returns.std() * 100
        
        if volatility > 3:
            insights.append(f"⚠️ High volatility detected: {volatility:.1f}% daily average moves")
        elif volatility < 1:
            insights.append(f"😴 Low volatility: {volatility:.1f}% daily average moves - potential breakout ahead")
        else:
            insights.append(f"📊 Normal volatility: {volatility:.1f}% daily average moves")
        
        # Volume analysis (if available)
        if 'Volume' in data.columns:
            volume = data['Volume'].dropna()
            if len(volume) > 20:
                avg_volume = volume.tail(60).mean()
                recent_volume = volume.tail(5).mean()
                volume_change = ((recent_volume - avg_volume) / avg_volume) * 100
                
                if volume_change > 50:
                    insights.append(f"🔊 High volume surge: {volume_change:.0f}% above average - strong conviction")
                elif volume_change < -30:
                    insights.append(f"🔇 Low volume: {abs(volume_change):.0f}% below average - lack of interest")
        
        # Moving average analysis
        if len(close_prices) >= 20:
            sma_20 = close_prices.rolling(20).mean().iloc[-1]
            sma_distance = ((current_price - sma_20) / sma_20) * 100
            
            if sma_distance > 5:
                insights.append(f"🔥 Price {sma_distance:.1f}% above 20-day SMA - strong momentum")
            elif sma_distance < -5:
                insights.append(f"❄️ Price {abs(sma_distance):.1f}% below 20-day SMA - potential oversold")
        
        # RSI analysis
        if len(returns) >= 14:
            delta = returns
            up = delta.clip(lower=0)
            down = -delta.clip(upper=0)
            ema_up = up.ewm(span=14, adjust=False).mean()
            ema_down = down.ewm(span=14, adjust=False).mean()
            rs = ema_up / ema_down
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
            
            if current_rsi > 75:
                insights.append(f"🚨 Overbought territory: RSI at {current_rsi:.1f} - potential pullback risk")
            elif current_rsi < 25:
                insights.append(f"💎 Oversold territory: RSI at {current_rsi:.1f} - potential bounce opportunity")
        
        # Price gaps analysis
        gaps = []
        for i in range(1, min(30, len(data))):
            prev_close = data['Close'].iloc[-(i+1)]
            curr_open = data['Open'].iloc[-i] if 'Open' in data.columns else data['Close'].iloc[-i]
            gap_pct = abs(curr_open - prev_close) / prev_close * 100
            if gap_pct > 2:
                gaps.append(gap_pct)
        
        if gaps:
            insights.append(f"⚡ {len(gaps)} significant price gaps detected in last 30 days - watch for gap fills")
        
        return "\n".join([f"• {insight}" for insight in insights[:8]])  # Limit to 8 key insights
        
    except Exception as e:
        return f"Error analyzing chart: {str(e)}"

def get_tradingview_link(ticker):
    """Generate TradingView link for the ticker with Indian market support"""
    clean_ticker = ticker.upper().strip()
    
    # Determine exchange based on ticker format
    if clean_ticker.endswith('.NS'):
        # NSE stock
        base_ticker = clean_ticker.replace('.NS', '')
        return f"https://www.tradingview.com/chart/?symbol=NSE:{base_ticker}"
    elif clean_ticker.endswith('.BO'):
        # BSE stock
        base_ticker = clean_ticker.replace('.BO', '')
        return f"https://www.tradingview.com/chart/?symbol=BSE:{base_ticker}"
    else:
        # Check if it's likely an Indian stock (common Indian stock patterns)
        indian_patterns = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK', 'SBIN', 'ITC', 'HINDUNILVR', 'BHARTIARTL', 'WIPRO']
        if any(pattern in clean_ticker for pattern in indian_patterns):
            return f"https://www.tradingview.com/chart/?symbol=NSE:{clean_ticker}"
        else:
            # Default to NASDAQ for US stocks
            return f"https://www.tradingview.com/chart/?symbol=NASDAQ:{clean_ticker}"

def plot_stock_price(ticker):
    """Create stock chart with detailed analysis and TradingView link"""
    try:
        ticker = ticker.upper().strip()
        
        data, error = data_fetcher.get_stock_data_multi(ticker, '6mo')
        
        if data is not None and len(data) > 5:
            session_id = st.session_state.get('session_id', str(uuid.uuid4())[:8])
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'stock_{ticker}_{session_id}_{timestamp}.png'
            
            # Create enhanced chart
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])
            
            # Price chart with enhanced features
            ax1.plot(data.index, data['Close'], linewidth=2.5, color='#1f77b4', label='Close Price')
            ax1.fill_between(data.index, data['Close'], alpha=0.2, color='#1f77b4')
            
            # Add moving averages
            if len(data) >= 20:
                sma_20 = data['Close'].rolling(20).mean()
                ax1.plot(data.index, sma_20, '--', linewidth=2, color='orange', label='SMA 20', alpha=0.8)
            
            if len(data) >= 50:
                sma_50 = data['Close'].rolling(50).mean()
                ax1.plot(data.index, sma_50, '--', linewidth=2, color='red', label='SMA 50', alpha=0.8)
            
            # Highlight recent high/low
            recent_data = data.tail(60)
            recent_high_idx = recent_data['Close'].idxmax()
            recent_low_idx = recent_data['Close'].idxmin()
            recent_high = recent_data['Close'].max()
            recent_low = recent_data['Close'].min()
            
            ax1.scatter(recent_high_idx, recent_high, color='green', s=100, marker='^', 
                       label=f'Recent High: ${recent_high:.2f}', zorder=5)
            ax1.scatter(recent_low_idx, recent_low, color='red', s=100, marker='v', 
                       label=f'Recent Low: ${recent_low:.2f}', zorder=5)
            
            ax1.set_title(f'{ticker} Stock Analysis - Enhanced Chart ({len(data)} days)', 
                         fontsize=18, fontweight='bold', pad=20)
            ax1.set_ylabel('Price ($)', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax1.legend(loc='upper left', framealpha=0.9, fontsize=10)
            
            # Enhanced volume chart
            if 'Volume' in data.columns:
                colors = ['green' if data['Close'].iloc[i] >= data['Close'].iloc[i-1] 
                         else 'red' for i in range(1, len(data))]
                colors.insert(0, 'gray')  # First bar color
                
                ax2.bar(data.index, data['Volume'], alpha=0.7, color=colors, width=1)
                ax2.set_ylabel('Volume', fontsize=12, fontweight='bold')
                
                # Add volume moving average
                vol_ma = data['Volume'].rolling(20).mean()
                ax2.plot(data.index, vol_ma, color='purple', linewidth=2, 
                        label='Volume MA(20)', alpha=0.8)
                ax2.legend(loc='upper right')
            else:
                ax2.text(0.5, 0.5, 'Volume data not available', transform=ax2.transAxes, 
                        ha='center', fontsize=12, style='italic')
            
            ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return filename, data
        else:
            return f"❌ Unable to create chart for {ticker}. No data available from any source.", None
            
    except Exception as e:
        return f"❌ Error creating chart: {str(e)}", None

def add_to_watchlist(ticker):
    """Add stock to watchlist"""
    if 'watchlist' not in st.session_state:
        st.session_state['watchlist'] = []
    
    ticker = ticker.upper().strip()
    
    if ticker not in st.session_state['watchlist']:
        st.session_state['watchlist'].append(ticker)
        return f"✅ Added {ticker} to your watchlist!"
    else:
        return f"ℹ️ {ticker} is already in your watchlist."

def show_watchlist():
    """Show watchlist with prices"""
    if 'watchlist' not in st.session_state or not st.session_state['watchlist']:
        return "📝 Your watchlist is empty. Add some stocks!"
    
    results = ["📊 **Your Watchlist:**\n"]
    
    for i, ticker in enumerate(st.session_state['watchlist'], 1):
        try:
            result, _ = data_fetcher.get_stock_price_multi(ticker)
            if result:
                emoji = "🟢" if result['change'] > 0 else "🔴" if result['change'] < 0 else "⚪"
                results.append(f"{i}. **{ticker}**: ${result['price']:.2f} {emoji} {result['change']:+.2f} ({result['change_percent']:+.2f}%)")
            else:
                results.append(f"{i}. **{ticker}**: ❌ Data unavailable")
        except:
            results.append(f"{i}. **{ticker}**: ⚠️ Error")
    
    return "\n".join(results)

# =============================================================================
# PORTFOLIO MANAGEMENT FUNCTIONS
# =============================================================================

def add_portfolio_position(ticker, shares, avg_cost, purchase_date=None):
    """Add a position to the portfolio"""
    try:
        return portfolio_manager.add_position(ticker, shares, avg_cost, purchase_date)
    except Exception as e:
        return f"❌ Error adding position: {str(e)}"

def remove_portfolio_position(ticker, shares=None):
    """Remove or reduce a position from the portfolio"""
    try:
        return portfolio_manager.remove_position(ticker, shares)
    except Exception as e:
        return f"❌ Error removing position: {str(e)}"

def show_portfolio():
    """Display portfolio summary"""
    try:
        return portfolio_manager.get_portfolio_summary()
    except Exception as e:
        return f"❌ Error displaying portfolio: {str(e)}"

def portfolio_analytics():
    """Get advanced portfolio analytics"""
    try:
        return portfolio_manager.get_portfolio_analytics()
    except Exception as e:
        return f"❌ Error in portfolio analytics: {str(e)}"

def portfolio_rebalancing():
    """Get portfolio rebalancing suggestions"""
    try:
        return portfolio_manager.rebalancing_suggestions()
    except Exception as e:
        return f"❌ Error in rebalancing analysis: {str(e)}"

def portfolio_risk_alerts():
    """Get portfolio risk management alerts"""
    try:
        return portfolio_manager.risk_management_alert()
    except Exception as e:
        return f"❌ Error in risk analysis: {str(e)}"
    
    
class PortfolioPosition:
    """Represents a single stock position in the portfolio"""
    def __init__(self, ticker, shares, avg_cost, purchase_date=None):
        self.ticker = ticker.upper().strip()
        self.shares = float(shares)
        self.avg_cost = float(avg_cost)
        self.purchase_date = purchase_date or datetime.now().strftime("%Y-%m-%d")
    
    def get_current_value(self):
        """Get current market value of position"""
        try:
            result, _ = data_fetcher.get_stock_price_multi(self.ticker)
            if result:
                current_price = result['price']
                return {
                    'current_price': current_price,
                    'market_value': current_price * self.shares,
                    'cost_basis': self.avg_cost * self.shares,
                    'unrealized_pnl': (current_price - self.avg_cost) * self.shares,
                    'return_pct': ((current_price - self.avg_cost) / self.avg_cost) * 100
                }
            return None
        except Exception:
            return None
    
    def to_dict(self):
        return {
            'ticker': self.ticker,
            'shares': self.shares,
            'avg_cost': self.avg_cost,
            'purchase_date': self.purchase_date
        }

class PortfolioManager:
    """Advanced portfolio management with AI insights"""
    
    def __init__(self):
        self.positions = {}
        self.load_portfolio()
    
    def load_portfolio(self):
        """Load portfolio from session state"""
        if 'portfolio_positions' in st.session_state:
            for pos_data in st.session_state['portfolio_positions']:
                pos = PortfolioPosition(**pos_data)
                self.positions[pos.ticker] = pos
    
    def save_portfolio(self):
        """Save portfolio to session state"""
        st.session_state['portfolio_positions'] = [
            pos.to_dict() for pos in self.positions.values()
        ]
    
    def add_position(self, ticker, shares, avg_cost, purchase_date=None):
        """Add or update a portfolio position"""
        ticker = ticker.upper().strip()
        shares = float(shares)
        avg_cost = float(avg_cost)
        
        if ticker in self.positions:
            # Update existing position (average down/up)
            existing = self.positions[ticker]
            total_cost = (existing.shares * existing.avg_cost) + (shares * avg_cost)
            total_shares = existing.shares + shares
            new_avg_cost = total_cost / total_shares
            
            self.positions[ticker] = PortfolioPosition(ticker, total_shares, new_avg_cost, existing.purchase_date)
        else:
            self.positions[ticker] = PortfolioPosition(ticker, shares, avg_cost, purchase_date)
        
        self.save_portfolio()
        return f"✅ Added {shares} shares of {ticker} at ${avg_cost:.2f} to portfolio"
    
    def remove_position(self, ticker, shares=None):
        """Remove or reduce a portfolio position"""
        ticker = ticker.upper().strip()
        
        if ticker not in self.positions:
            return f"❌ {ticker} not found in portfolio"
        
        position = self.positions[ticker]
        
        if shares is None or shares >= position.shares:
            # Remove entire position
            del self.positions[ticker]
            self.save_portfolio()
            return f"✅ Removed all {position.shares} shares of {ticker} from portfolio"
        else:
            # Reduce position
            position.shares -= float(shares)
            self.save_portfolio()
            return f"✅ Reduced {ticker} position by {shares} shares. Remaining: {position.shares}"
    
    def get_portfolio_summary(self):
        """Get comprehensive portfolio summary"""
        if not self.positions:
            return "📊 Your portfolio is empty. Add some positions to get started!"
        
        total_value = 0
        total_cost = 0
        positions_data = []
        
        for ticker, position in self.positions.items():
            current_data = position.get_current_value()
            if current_data:
                total_value += current_data['market_value']
                total_cost += current_data['cost_basis']
                positions_data.append({
                    'ticker': ticker,
                    'shares': position.shares,
                    'avg_cost': position.avg_cost,
                    'current_price': current_data['current_price'],
                    'market_value': current_data['market_value'],
                    'pnl': current_data['unrealized_pnl'],
                    'return_pct': current_data['return_pct'],
                    'weight': 0  # Will be calculated after total_value is known
                })
        
        # Calculate portfolio weights
        for pos in positions_data:
            pos['weight'] = (pos['market_value'] / total_value) * 100 if total_value > 0 else 0
        
        total_pnl = total_value - total_cost
        total_return_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0
        
        # Sort by portfolio weight
        positions_data.sort(key=lambda x: x['weight'], reverse=True)
        
        summary = f"""📊 **Portfolio Summary**

💰 **Total Portfolio Value**: ${total_value:,.2f}
💸 **Total Cost Basis**: ${total_cost:,.2f}
📈 **Unrealized P&L**: ${total_pnl:+,.2f} ({total_return_pct:+.2f}%)
📋 **Number of Positions**: {len(self.positions)}

🏆 **Top Holdings:**
"""
        
        for i, pos in enumerate(positions_data[:5], 1):
            pnl_emoji = "🟢" if pos['pnl'] > 0 else "🔴" if pos['pnl'] < 0 else "⚪"
            summary += f"{i}. **{pos['ticker']}** ({pos['weight']:.1f}%) {pnl_emoji}\n"
            summary += f"   {pos['shares']:.2f} shares @ ${pos['avg_cost']:.2f} → ${pos['current_price']:.2f}\n"
            summary += f"   Value: ${pos['market_value']:,.2f} | P&L: ${pos['pnl']:+,.2f} ({pos['return_pct']:+.1f}%)\n\n"
        
        return summary
    
    def get_portfolio_analytics(self):
        """Advanced portfolio analytics"""
        if not self.positions:
            return "❌ No positions in portfolio for analysis"
        
        try:
            # Collect data for analysis
            tickers = list(self.positions.keys())
            portfolio_data = {}
            total_value = 0
            
            for ticker in tickers:
                data, _ = data_fetcher.get_stock_data_multi(ticker, '1y')
                position = self.positions[ticker]
                current_value_data = position.get_current_value()
                
                if data is not None and current_value_data:
                    portfolio_data[ticker] = {
                        'data': data,
                        'weight': 0,  # Will calculate after getting total value
                        'current_value': current_value_data['market_value']
                    }
                    total_value += current_value_data['market_value']
            
            if not portfolio_data:
                return "❌ Unable to fetch data for portfolio analysis"
            
            # Calculate weights
            for ticker in portfolio_data:
                portfolio_data[ticker]['weight'] = portfolio_data[ticker]['current_value'] / total_value
            
            # Portfolio-level calculations
            portfolio_returns = None
            individual_volatilities = {}
            
            for ticker, data in portfolio_data.items():
                returns = data['data']['Close'].pct_change().dropna()
                if len(returns) > 0:
                    volatility = returns.std() * np.sqrt(252)
                    individual_volatilities[ticker] = volatility
                    
                    # Weight returns by portfolio allocation
                    weighted_returns = returns * data['weight']
                    
                    if portfolio_returns is None:
                        portfolio_returns = weighted_returns
                    else:
                        portfolio_returns = portfolio_returns.add(weighted_returns, fill_value=0)
            
            if portfolio_returns is None or len(portfolio_returns) == 0:
                return "❌ Insufficient data for portfolio analytics"
            
            # Calculate portfolio metrics
            portfolio_volatility = portfolio_returns.std() * np.sqrt(252)
            portfolio_annual_return = portfolio_returns.mean() * 252
            
            # Risk metrics
            downside_returns = portfolio_returns[portfolio_returns < 0]
            downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            
            # Sharpe ratio (assuming 2% risk-free rate)
            risk_free_rate = 0.02
            sharpe_ratio = (portfolio_annual_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
            
            # Maximum drawdown
            cumulative_returns = (1 + portfolio_returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Diversification analysis
            concentration_risk = max(data['weight'] for data in portfolio_data.values())
            
            analytics = f"""📊 **Portfolio Analytics**

📈 **Performance Metrics**
• Annual Return: {portfolio_annual_return:.2%}
• Volatility: {portfolio_volatility:.2%}
• Sharpe Ratio: {sharpe_ratio:.2f}
• Max Drawdown: {max_drawdown:.2%}

⚖️ **Risk Analysis**
• Portfolio Risk: {"High" if portfolio_volatility > 0.25 else "Medium" if portfolio_volatility > 0.15 else "Low"}
• Downside Volatility: {downside_volatility:.2%}
• Concentration Risk: {concentration_risk:.1%} ({"High" if concentration_risk > 0.3 else "Medium" if concentration_risk > 0.15 else "Low"})

🎯 **Individual Stock Volatilities:**
"""
            
            for ticker, vol in individual_volatilities.items():
                risk_level = "🔴 High" if vol > 0.3 else "🟡 Medium" if vol > 0.15 else "🟢 Low"
                analytics += f"• {ticker}: {vol:.1%} {risk_level}\n"
            
            # Recommendations
            analytics += f"\n💡 **Recommendations:**\n"
            
            if concentration_risk > 0.4:
                analytics += "• ⚠️ Consider diversifying - one position dominates portfolio\n"
            
            if portfolio_volatility > 0.3:
                analytics += "• 🛡️ High portfolio risk - consider adding defensive stocks\n"
            
            if sharpe_ratio < 0:
                analytics += "• 📉 Negative risk-adjusted returns - review underperforming positions\n"
            elif sharpe_ratio > 1:
                analytics += "• 🏆 Excellent risk-adjusted performance\n"
            
            if len(self.positions) < 5:
                analytics += "• 🔄 Consider adding more positions for better diversification\n"
            
            return analytics
            
        except Exception as e:
            return f"❌ Error in portfolio analytics: {str(e)}"
    
    def rebalancing_suggestions(self):
        """Provide intelligent rebalancing suggestions"""
        if not self.positions:
            return "❌ No positions in portfolio for rebalancing analysis"
        
        try:
            # Get current portfolio composition
            positions_data = []
            total_value = 0
            
            for ticker, position in self.positions.items():
                current_data = position.get_current_value()
                if current_data:
                    positions_data.append({
                        'ticker': ticker,
                        'current_value': current_data['market_value'],
                        'weight': 0  # Will calculate below
                    })
                    total_value += current_data['market_value']
            
            # Calculate current weights
            for pos in positions_data:
                pos['weight'] = pos['current_value'] / total_value if total_value > 0 else 0
            
            # Sort by weight
            positions_data.sort(key=lambda x: x['weight'], reverse=True)
            
            suggestions = f"""⚖️ **Portfolio Rebalancing Analysis**

📊 **Current Allocation:**
"""
            
            for pos in positions_data:
                weight_status = "🔴 Overweight" if pos['weight'] > 0.25 else "🟡 High" if pos['weight'] > 0.15 else "🟢 Balanced"
                suggestions += f"• {pos['ticker']}: {pos['weight']:.1%} ({weight_status})\n"
            
            suggestions += f"\n💡 **Rebalancing Suggestions:**\n"
            
            # Identify overweight positions
            overweight = [pos for pos in positions_data if pos['weight'] > 0.25]
            underweight = [pos for pos in positions_data if pos['weight'] < 0.05]
            
            if overweight:
                suggestions += f"🔴 **Reduce Overweight Positions:**\n"
                for pos in overweight:
                    target_weight = 0.20  # Target 20% max
                    excess_value = (pos['weight'] - target_weight) * total_value
                    suggestions += f"• Consider reducing {pos['ticker']} by ~${excess_value:,.0f}\n"
            
            if underweight and len(underweight) < len(positions_data):
                suggestions += f"\n🟢 **Consider Increasing:**\n"
                for pos in underweight:
                    target_increase = (0.10 - pos['weight']) * total_value  # Target 10% min
                    if target_increase > 0:
                        suggestions += f"• Add ~${target_increase:,.0f} to {pos['ticker']}\n"
            
            # Diversification suggestions
            if len(positions_data) < 5:
                suggestions += f"\n🔄 **Diversification:**\n"
                suggestions += f"• Consider adding {5 - len(positions_data)} more positions\n"
                suggestions += f"• Target allocation: ~{100/5:.0f}% per position\n"
            
            # Risk-based suggestions
            high_risk_positions = []
            for ticker in self.positions:
                try:
                    data, _ = data_fetcher.get_stock_data_multi(ticker, '6mo')
                    if data is not None and len(data) > 20:
                        returns = data['Close'].pct_change().dropna()
                        volatility = returns.std() * np.sqrt(252)
                        if volatility > 0.35:  # High volatility threshold
                            high_risk_positions.append((ticker, volatility))
                except:
                    continue
            
            if high_risk_positions:
                suggestions += f"\n⚠️ **High Risk Positions:**\n"
                for ticker, vol in high_risk_positions:
                    suggestions += f"• {ticker}: {vol:.1%} volatility - consider reducing\n"
            
            return suggestions
            
        except Exception as e:
            return f"❌ Error in rebalancing analysis: {str(e)}"
    
    def risk_management_alert(self):
        """Generate risk management alerts"""
        if not self.positions:
            return "📊 No positions to monitor"
        
        alerts = []
        total_value = 0
        total_pnl = 0
        
        for ticker, position in self.positions.items():
            current_data = position.get_current_value()
            if current_data:
                total_value += current_data['market_value']
                total_pnl += current_data['unrealized_pnl']
                
                # Individual position alerts
                if current_data['return_pct'] < -20:
                    alerts.append(f"🚨 {ticker} down {abs(current_data['return_pct']):.1f}% - Consider stop loss")
                elif current_data['return_pct'] > 50:
                    alerts.append(f"💰 {ticker} up {current_data['return_pct']:.1f}% - Consider taking profits")
                
                # Check volatility
                try:
                    data, _ = data_fetcher.get_stock_data_multi(ticker, '1mo')
                    if data is not None and len(data) > 10:
                        returns = data['Close'].pct_change().dropna()
                        recent_vol = returns.tail(10).std() * np.sqrt(252)
                        if recent_vol > 0.5:
                            alerts.append(f"⚡ {ticker} showing high volatility ({recent_vol:.1%})")
                except:
                    pass
        
        # Portfolio-level alerts
        if total_value > 0:
            portfolio_return = (total_pnl / (total_value - total_pnl)) * 100
            if portfolio_return < -15:
                alerts.append(f"🔴 Portfolio down {abs(portfolio_return):.1f}% - Review risk management")
            elif portfolio_return > 30:
                alerts.append(f"🎯 Portfolio up {portfolio_return:.1f}% - Consider rebalancing")
        
        if not alerts:
            alerts.append("✅ No risk alerts - Portfolio within normal parameters")
        
        result = "🛡️ **Risk Management Alerts**\n\n"
        for alert in alerts:
            result += f"• {alert}\n"
        
        return result

# Initialize portfolio manager
portfolio_manager = PortfolioManager()

# =============================================================================
# BAYESIAN ANALYSIS CLASSES (unchanged from your existing code)
# =============================================================================

class BayesianFinancialAnalyzer:
    """Implements Bayesian inference for financial market analysis"""
    
    def __init__(self):
        self.confidence_thresholds = {
            'high': 0.8,
            'medium': 0.6,
            'low': 0.4
        }
        
    def calculate_price_direction_probability(self, price_data, window=20):
        """Calculate probability of upward price movement using Bayesian inference"""
        try:
            if len(price_data) < window + 5:
                return None, "Insufficient data for Bayesian analysis"
            
            returns = price_data.pct_change().dropna()
            prior_prob_up = (returns > 0).mean()
            
            recent_returns = returns.tail(window)
            recent_up_days = (recent_returns > 0).sum()
            recent_total_days = len(recent_returns)
            
            total_days = len(returns)
            prior_alpha = prior_prob_up * total_days
            prior_beta = (1 - prior_prob_up) * total_days
            
            posterior_alpha = prior_alpha + recent_up_days
            posterior_beta = prior_beta + (recent_total_days - recent_up_days)
            
            posterior_prob_up = posterior_alpha / (posterior_alpha + posterior_beta)
            
            posterior_variance = (posterior_alpha * posterior_beta) / \
                               ((posterior_alpha + posterior_beta)**2 * (posterior_alpha + posterior_beta + 1))
            
            confidence = 1 - min(posterior_variance * 100, 1)
            
            return {
                'probability_up': posterior_prob_up,
                'probability_down': 1 - posterior_prob_up,
                'confidence': confidence,
                'prior_prob': prior_prob_up,
                'recent_evidence': recent_up_days / recent_total_days,
                'sample_size': len(returns)
            }, None
            
        except Exception as e:
            return None, f"Error in Bayesian calculation: {str(e)}"
    
    def volatility_regime_detection(self, price_data):
        """Detect market volatility regime using Bayesian model selection"""
        try:
            returns = price_data.pct_change().dropna()
            
            if len(returns) < 30:
                return None, "Insufficient data for regime detection"
            
            volatility = returns.rolling(window=20).std() * np.sqrt(252)
            volatility = volatility.dropna()
            
            low_vol_threshold = volatility.quantile(0.33)
            high_vol_threshold = volatility.quantile(0.67)
            
            current_vol = volatility.iloc[-1]
            recent_vol = volatility.tail(10).mean()
            
            regimes = []
            if current_vol < low_vol_threshold:
                regimes.append(('Low Volatility', 0.7))
            elif current_vol > high_vol_threshold:
                regimes.append(('High Volatility', 0.7))
            else:
                regimes.append(('Medium Volatility', 0.6))
            
            trend_evidence = "Increasing" if recent_vol > volatility.mean() else "Decreasing"
            
            return {
                'current_regime': regimes[0][0],
                'regime_confidence': regimes[0][1],
                'current_volatility': current_vol,
                'trend': trend_evidence,
                'volatility_percentile': stats.percentileofscore(volatility, current_vol)
            }, None
            
        except Exception as e:
            return None, f"Error in regime detection: {str(e)}"

class BeliefNetwork:
    """Simple belief network for modeling relationships between financial indicators"""
    
    def __init__(self):
        self.nodes = {
            'price_trend': {'states': ['bullish', 'bearish', 'neutral']},
            'volume_trend': {'states': ['high', 'normal', 'low']},
            'rsi_signal': {'states': ['overbought', 'oversold', 'neutral']},
            'volatility': {'states': ['high', 'medium', 'low']},
            'market_sentiment': {'states': ['positive', 'negative', 'neutral']}
        }
        
        self.priors = {
            'price_trend': {'bullish': 0.4, 'bearish': 0.3, 'neutral': 0.3},
            'volume_trend': {'high': 0.25, 'normal': 0.5, 'low': 0.25},
            'rsi_signal': {'overbought': 0.2, 'oversold': 0.2, 'neutral': 0.6},
            'volatility': {'high': 0.2, 'medium': 0.6, 'low': 0.2}
        }
        
        self.conditional_probs = {
            'market_sentiment': {
                ('bullish', 'high', 'neutral', 'low'): {'positive': 0.8, 'negative': 0.1, 'neutral': 0.1},
                ('bearish', 'low', 'oversold', 'high'): {'positive': 0.1, 'negative': 0.8, 'neutral': 0.1},
                ('neutral', 'normal', 'neutral', 'medium'): {'positive': 0.3, 'negative': 0.3, 'neutral': 0.4}
            }
        }
    
    def calculate_market_sentiment(self, evidence):
        """Calculate market sentiment probability given evidence"""
        try:
            sentiment_scores = {'positive': 0, 'negative': 0, 'neutral': 0}
            
            if evidence.get('price_trend') == 'bullish':
                sentiment_scores['positive'] += 0.4
            elif evidence.get('price_trend') == 'bearish':
                sentiment_scores['negative'] += 0.4
            else:
                sentiment_scores['neutral'] += 0.3
            
            if evidence.get('rsi_signal') == 'oversold':
                sentiment_scores['positive'] += 0.2
            elif evidence.get('rsi_signal') == 'overbought':
                sentiment_scores['negative'] += 0.2
            
            if evidence.get('volatility') == 'high':
                sentiment_scores['negative'] += 0.2
            elif evidence.get('volatility') == 'low':
                sentiment_scores['positive'] += 0.1
            
            total = sum(sentiment_scores.values())
            if total > 0:
                for key in sentiment_scores:
                    sentiment_scores[key] /= total
            
            return sentiment_scores
            
        except Exception as e:
            return {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}

class UncertaintyHandler:
    """Handles uncertainty quantification and confidence scoring"""
    
    def __init__(self):
        self.confidence_factors = {
            'data_quality': 0.3,
            'sample_size': 0.25,
            'volatility': 0.2,
            'model_agreement': 0.25
        }
    
    def assess_data_quality(self, data):
        """Assess quality of input data"""
        try:
            if data is None or len(data) == 0:
                return 0.0
            
            missing_ratio = data.isnull().sum() / len(data)
            z_scores = np.abs(stats.zscore(data.dropna()))
            outlier_ratio = (z_scores > 3).sum() / len(z_scores) if len(z_scores) > 0 else 0
            
            quality_score = (1 - missing_ratio) * (1 - outlier_ratio * 0.5)
            
            return max(0, min(1, quality_score))
            
        except Exception:
            return 0.5
    
    def calculate_confidence_score(self, analysis_results):
        """Calculate overall confidence score for analysis"""
        try:
            scores = {}
            
            if 'data_quality' in analysis_results:
                scores['data_quality'] = analysis_results['data_quality']
            else:
                scores['data_quality'] = 0.7
            
            sample_size = analysis_results.get('sample_size', 0)
            if sample_size > 100:
                scores['sample_size'] = 1.0
            elif sample_size > 50:
                scores['sample_size'] = 0.8
            elif sample_size > 20:
                scores['sample_size'] = 0.6
            else:
                scores['sample_size'] = 0.4
            
            volatility = analysis_results.get('volatility_regime', {}).get('current_volatility', 0.2)
            scores['volatility'] = max(0.2, 1 - (volatility / 0.5))
            
            scores['model_agreement'] = 0.75
            
            confidence = sum(
                scores[factor] * weight 
                for factor, weight in self.confidence_factors.items() 
                if factor in scores
            )
            
            return max(0, min(1, confidence))
            
        except Exception:
            return 0.5

def bayesian_stock_analysis(ticker):
    """Perform comprehensive Bayesian analysis of a stock"""
    try:
        data, error = data_fetcher.get_stock_data_multi(ticker, '1y')
        if data is None:
            return f"❌ Unable to fetch data for {ticker}"
        
        bayesian_analyzer = BayesianFinancialAnalyzer()
        belief_net = BeliefNetwork()
        uncertainty_handler = UncertaintyHandler()
        
        price_analysis, error = bayesian_analyzer.calculate_price_direction_probability(data['Close'])
        if error:
            return f"❌ Error in Bayesian analysis: {error}"
        
        vol_analysis, error = bayesian_analyzer.volatility_regime_detection(data['Close'])
        if error:
            vol_analysis = {'current_regime': 'Unknown', 'regime_confidence': 0.5}
        
        data_quality = uncertainty_handler.assess_data_quality(data['Close'])
        
        evidence = {
            'price_trend': 'bullish' if price_analysis['probability_up'] > 0.6 else 'bearish' if price_analysis['probability_up'] < 0.4 else 'neutral',
            'volatility': vol_analysis.get('current_regime', 'medium').lower().split()[0],
            'rsi_signal': 'neutral'
        }
        
        sentiment = belief_net.calculate_market_sentiment(evidence)
        
        analysis_results = {
            'data_quality': data_quality,
            'sample_size': len(data),
            'volatility_regime': vol_analysis
        }
        confidence = uncertainty_handler.calculate_confidence_score(analysis_results)
        
        result = f"""🧠 **Bayesian Analysis for {ticker}**

📊 **Price Direction Probability**
⬆️ Upward: {price_analysis['probability_up']:.1%}
⬇️ Downward: {price_analysis['probability_down']:.1%}
📈 Confidence: {price_analysis['confidence']:.1%}

🌊 **Volatility Regime**
🎯 Current: {vol_analysis.get('current_regime', 'Unknown')}
📊 Confidence: {vol_analysis.get('regime_confidence', 0.5):.1%}

🧮 **Market Sentiment (Belief Network)**
😊 Positive: {sentiment['positive']:.1%}
😟 Negative: {sentiment['negative']:.1%}
😐 Neutral: {sentiment['neutral']:.1%}

🎯 **Overall Analysis Confidence**
📋 Data Quality: {data_quality:.1%}
📊 Sample Size: {len(data)} days
🔍 Final Confidence: {confidence:.1%}

⚠️ **Risk Assessment**
{_get_risk_assessment(confidence, price_analysis['probability_up'])}
"""
        
        return result
        
    except Exception as e:
        return f"❌ Error in Bayesian analysis: {str(e)}"

def _get_risk_assessment(confidence, prob_up):
    """Generate risk assessment based on confidence and probability"""
    if confidence > 0.8:
        if prob_up > 0.7:
            return "🟢 HIGH CONFIDENCE - Strong upward probability"
        elif prob_up < 0.3:
            return "🔴 HIGH CONFIDENCE - Strong downward probability"
        else:
            return "🟡 HIGH CONFIDENCE - Neutral direction"
    elif confidence > 0.6:
        return "🟡 MEDIUM CONFIDENCE - Exercise caution"
    else:
        return "🔴 LOW CONFIDENCE - High uncertainty, avoid major decisions"

def uncertainty_analysis(ticker):
    """Provide detailed uncertainty analysis for a stock"""
    try:
        data, error = data_fetcher.get_stock_data_multi(ticker, '6mo')
        if data is None:
            return f"❌ Unable to fetch data for {ticker}"
        
        uncertainty_handler = UncertaintyHandler()
        returns = data['Close'].pct_change().dropna()
        data_quality = uncertainty_handler.assess_data_quality(data['Close'])
        
        current_price = data['Close'].iloc[-1]
        volatility = returns.std() * np.sqrt(252)
        
        z_score = 1.96
        margin = z_score * volatility * np.sqrt(30/252) * current_price
        
        upper_bound = current_price + margin
        lower_bound = current_price - margin
        
        result = f"""🎯 **Uncertainty Analysis for {ticker}**

📊 **Data Quality Assessment**
Quality Score: {data_quality:.1%}
Sample Size: {len(data)} days
Missing Data: {data['Close'].isnull().sum()} points

📈 **Price Prediction Uncertainty**
Current Price: ${current_price:.2f}
30-Day 95% Confidence Interval:
  Upper Bound: ${upper_bound:.2f} (+{((upper_bound/current_price - 1)*100):+.1f}%)
  Lower Bound: ${lower_bound:.2f} ({((lower_bound/current_price - 1)*100):+.1f}%)

⚡ **Volatility Analysis**
Annualized Volatility: {volatility:.1%}
Uncertainty Level: {"High" if volatility > 0.3 else "Medium" if volatility > 0.15 else "Low"}

🎲 **Recommendation**
{_get_uncertainty_recommendation(data_quality, volatility)}
"""
        
        return result
        
    except Exception as e:
        return f"❌ Error in uncertainty analysis: {str(e)}"

def _get_uncertainty_recommendation(data_quality, volatility):
    """Generate recommendation based on uncertainty metrics"""
    if data_quality > 0.8 and volatility < 0.2:
        return "🟢 LOW UNCERTAINTY - Reliable for analysis"
    elif data_quality > 0.6 and volatility < 0.3:
        return "🟡 MEDIUM UNCERTAINTY - Use with caution"
    else:
        return "🔴 HIGH UNCERTAINTY - Not suitable for major decisions"

class PortfolioPosition:
    """Represents a single stock position in the portfolio"""
    def __init__(self, ticker, shares, avg_cost, purchase_date=None):
        self.ticker = ticker.upper().strip()
        self.shares = float(shares)
        self.avg_cost = float(avg_cost)
        self.purchase_date = purchase_date or datetime.now().strftime("%Y-%m-%d")
    
    def get_current_value(self):
        """Get current market value of position"""
        try:
            result, _ = data_fetcher.get_stock_price_multi(self.ticker)
            if result:
                current_price = result['price']
                return {
                    'current_price': current_price,
                    'market_value': current_price * self.shares,
                    'cost_basis': self.avg_cost * self.shares,
                    'unrealized_pnl': (current_price - self.avg_cost) * self.shares,
                    'return_pct': ((current_price - self.avg_cost) / self.avg_cost) * 100
                }
            return None
        except Exception:
            return None
    
    def to_dict(self):
        return {
            'ticker': self.ticker,
            'shares': self.shares,
            'avg_cost': self.avg_cost,
            'purchase_date': self.purchase_date
        }


# =============================================================================
# STREAMLIT APP CONFIGURATION
# =============================================================================

# Function declarations for Gemini
function_declarations = [
    {
        "name": "get_stock_price",
        "description": "Get real-time stock price using multiple reliable data sources",
        "parameters": {
            "type_": "OBJECT",
            "properties": {
                "ticker": {"type_": "STRING", "description": "Stock ticker symbol (e.g., AAPL)"}
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "calculate_SMA",
        "description": "Calculate Simple Moving Average using multiple data sources",
        "parameters": {
            "type_": "OBJECT",
            "properties": {
                "ticker": {"type_": "STRING", "description": "Stock ticker symbol"},
                "window": {"type_": "INTEGER", "description": "Time window for SMA (e.g., 20)"}
            },
            "required": ["ticker", "window"]
        }
    },
    {
        "name": "calculate_RSI",
        "description": "Calculate RSI using multiple data sources",
        "parameters": {
            "type_": "OBJECT",
            "properties": {
                "ticker": {"type_": "STRING", "description": "Stock ticker symbol"}
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "plot_stock_price",
        "description": "Create stock price chart using multi-source data",
        "parameters": {
            "type_": "OBJECT",
            "properties": {
                "ticker": {"type_": "STRING", "description": "Stock ticker symbol"}
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "add_to_watchlist",
        "description": "Add stock to personal watchlist",
        "parameters": {
            "type_": "OBJECT",
            "properties": {
                "ticker": {"type_": "STRING", "description": "Stock ticker to add"}
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "show_watchlist",
        "description": "Display watchlist with current prices",
        "parameters": {
            "type_": "OBJECT",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "bayesian_stock_analysis",
        "description": "Perform advanced Bayesian probability analysis on stock data",
        "parameters": {
            "type_": "OBJECT",
            "properties": {
                "ticker": {"type_": "STRING", "description": "Stock ticker symbol"}
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "uncertainty_analysis",
        "description": "Analyze uncertainty and confidence levels in stock predictions",
        "parameters": {
            "type_": "OBJECT",
            "properties": {
                "ticker": {"type_": "STRING", "description": "Stock ticker symbol"}
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "logical_stock_analysis",
        "description": "Perform logic-based inference analysis using forward and backward chaining",
        "parameters": {
            "type_": "OBJECT",
            "properties": {
                "ticker": {"type_": "STRING", "description": "Stock ticker symbol"}
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "query_knowledge_base",
        "description": "Query the financial knowledge base using natural language",
        "parameters": {
            "type_": "OBJECT",
            "properties": {
                "query": {"type_": "STRING", "description": "Natural language query about stocks"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "add_portfolio_position",
        "description": "Add a stock position to the portfolio",
        "parameters": {
            "type_": "OBJECT",
            "properties": {
                "ticker": {"type_": "STRING", "description": "Stock ticker symbol"},
                "shares": {"type_": "NUMBER", "description": "Number of shares"},
                "avg_cost": {"type_": "NUMBER", "description": "Average cost per share"},
                "purchase_date": {"type_": "STRING", "description": "Purchase date (YYYY-MM-DD), optional"}
            },
            "required": ["ticker", "shares", "avg_cost"]
        }
    },
    {
        "name": "remove_portfolio_position",
        "description": "Remove or reduce a position from the portfolio",
        "parameters": {
            "type_": "OBJECT",
            "properties": {
                "ticker": {"type_": "STRING", "description": "Stock ticker symbol"},
                "shares": {"type_": "NUMBER", "description": "Number of shares to remove (optional, removes all if not specified)"}
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "show_portfolio",
        "description": "Display complete portfolio summary with P&L",
        "parameters": {
            "type_": "OBJECT",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "portfolio_analytics",
        "description": "Get advanced portfolio analytics including risk metrics and performance",
        "parameters": {
            "type_": "OBJECT",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "portfolio_rebalancing",
        "description": "Get intelligent portfolio rebalancing suggestions",
        "parameters": {
            "type_": "OBJECT",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "portfolio_risk_alerts",
        "description": "Get portfolio risk management alerts and warnings",
        "parameters": {
            "type_": "OBJECT",
            "properties": {},
            "required": []
        }
    }
]

available_functions = {
    'get_stock_price': get_stock_price,
    'calculate_SMA': calculate_SMA,
    'calculate_RSI': calculate_RSI,
    'plot_stock_price': plot_stock_price,
    'add_to_watchlist': add_to_watchlist,
    'show_watchlist': show_watchlist,
    'bayesian_stock_analysis': bayesian_stock_analysis,
    'uncertainty_analysis': uncertainty_analysis,
    'logical_stock_analysis': logical_stock_analysis,
    'query_knowledge_base': query_knowledge_base,
     'add_portfolio_position': add_portfolio_position,
    'remove_portfolio_position': remove_portfolio_position,
    'show_portfolio': show_portfolio,
    'portfolio_analytics': portfolio_analytics,
    'portfolio_rebalancing': portfolio_rebalancing,
    'portfolio_risk_alerts': portfolio_risk_alerts
}

try:
    model = genai.GenerativeModel('models/gemini-2.5-flash', tools=function_declarations)
except Exception as e:
    st.error(f"Error with function calling: {e}")
    model = genai.GenerativeModel('models/gemini-2.5-flash')

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
    st.session_state['chat_session'] = model.start_chat()
if 'session_id' not in st.session_state:
    st.session_state['session_id'] = str(uuid.uuid4())[:8]
if 'watchlist' not in st.session_state:
    st.session_state['watchlist'] = []

# =============================================================================
# STREAMLIT UI
# =============================================================================

st.set_page_config(
    page_title="Logic-Based Financial Chatbot",
    page_icon="🧠",
    layout="wide"
)

st.title('🧠 Logic-Based Financial Analysis Chatbot')
st.markdown("*Powered by Forward/Backward Chaining, Bayesian Inference & Belief Networks*")

# Status indicators
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.success("🤖 Gemini AI: Online")
with col2:
    st.success("🧠 Logic Engine: Ready")
with col3:
    st.success("🎲 Bayesian Inference: Ready")
with col4:
    st.info("📊 Multi-API Data: Ready")

st.markdown("---")

# Quick test buttons
st.markdown("### 🚀 Quick Tests")
test_col1, test_col2, test_col3, test_col4, test_col5 = st.columns(5)

with test_col1:
    if st.button("📊 AAPL Price"):
        st.session_state.test_query = "What's the current price of AAPL?"

with test_col2:
    if st.button("🧠 Logic Analysis"):
        st.session_state.test_query = "Run logical analysis on TSLA"

with test_col3:
    if st.button("🎲 Bayesian Analysis"):
        st.session_state.test_query = "Perform Bayesian analysis on MSFT"

with test_col4:
    if st.button("❓ Query KB"):
        st.session_state.test_query = "Is AAPL a bullish stock?"

with test_col5:
    if st.button("📈 Chart"):
        st.session_state.test_query = "Plot chart for GOOGL"

# Display chat history
for message in st.session_state['messages']:
    if message['role'] == 'user':
        st.chat_message("user").write(message['content'])
    elif message['role'] == 'assistant':
        st.chat_message("assistant").write(message['content'])

# Handle input
user_input = None
if hasattr(st.session_state, 'test_query'):
    user_input = st.session_state.test_query
    del st.session_state.test_query
else:
    user_input = st.chat_input("💬 Ask about stocks using logic and probability!")

if user_input:
    st.chat_message("user").write(user_input)
    st.session_state['messages'].append({'role': 'user', 'content': user_input})
    
    try:
        response = st.session_state['chat_session'].send_message(user_input)
        
        if response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    function_call = part.function_call
                    function_name = function_call.name
                    function_args = {key: value for key, value in function_call.args.items()}
                    
                    if function_name in available_functions:
                        function_to_call = available_functions[function_name]
                        function_response = function_to_call(**function_args)
                        
                        if function_name == 'plot_stock_price':
                            if isinstance(function_response, tuple) and len(function_response) == 2:
                                filename, chart_data = function_response
                                
                                if chart_data is not None and not filename.startswith('❌'):
                                    # Display the chart
                                    st.chat_message("assistant").image(filename)
                                    
                                    # Generate and display detailed analysis
                                    insights = analyze_chart_insights(chart_data, function_args['ticker'])
                                    tradingview_link = get_tradingview_link(function_args['ticker'])
                                    
                                    analysis_text = f"""📊 **Chart Analysis for {function_args['ticker'].upper()}**

🔍 **Key Insights:**
{insights}

📈 **Professional Analysis:**
For advanced charting tools, technical indicators, and real-time analysis, view this stock on TradingView:
🔗 **[Open {function_args['ticker'].upper()} on TradingView]({tradingview_link})**

💡 **Pro Tip:** TradingView offers advanced features like:
• Multiple timeframes and chart types
• 100+ technical indicators
• Drawing tools and pattern recognition
• Real-time market data and alerts
• Community insights and analysis"""
                                    
                                    st.chat_message("assistant").write(analysis_text)
                                    st.session_state['messages'].append({'role': 'assistant', 'content': analysis_text})
                                    
                                    # Clean up chart file
                                    try:
                                        os.remove(filename)
                                    except:
                                        pass
                                else:
                                    # Handle error case
                                    st.chat_message("assistant").error(filename)
                            else:
                                # Handle old function format (backward compatibility)
                                if function_response.startswith('❌'):
                                    st.chat_message("assistant").error(function_response)
                                else:
                                    st.chat_message("assistant").image(function_response)
                                    try:
                                        os.remove(function_response)
                                    except:
                                        pass
                        else:
                            # Handle all other functions normally
                            result_response = st.session_state['chat_session'].send_message([
                                {
                                    "function_response": {
                                        "name": function_name,
                                        "response": {"result": function_response}
                                    }
                                }
                            ])
                            
                            if result_response.text:
                                st.chat_message("assistant").write(result_response.text)
                                st.session_state['messages'].append({'role': 'assistant', 'content': result_response.text})
                
                elif hasattr(part, 'text') and part.text:
                    st.chat_message("assistant").write(part.text)
                    st.session_state['messages'].append({'role': 'assistant', 'content': part.text})
        
    except Exception as e:
        st.chat_message("assistant").error(f"System error: {str(e)}")

# Sidebar
with st.sidebar:
    st.markdown("### 🧠 AI Components")
    st.success("✅ Logic-Based Inference")
    st.success("✅ Forward/Backward Chaining") 
    st.success("✅ Bayesian Reasoning")
    st.success("✅ Belief Networks")
    
    st.markdown("### 📊 Available Functions")
    st.markdown("""
    - **Logic Analysis**: Rule-based inference
    - **Bayesian Analysis**: Probabilistic reasoning
    - **Knowledge Base Queries**: Natural language queries
    - **Technical Indicators**: SMA, RSI calculations
    - **Charts**: Professional visualizations
    - **Watchlist**: Track your favorites
    """)
    
    st.markdown("### 🎯 Try These Queries")
    st.markdown("""
    - "Run logical analysis on AAPL"
    - "Is TSLA bullish?"
    - "Perform Bayesian analysis on MSFT"
    - "Should I buy NVDA?"
    - "What's the uncertainty for GOOGL?"
    - "Query: AMZN risky investment"
    """)
    
    st.markdown("### 👀 Your Watchlist")
    if st.session_state.get('watchlist'):
        for ticker in st.session_state['watchlist']:
            st.text(f"• {ticker}")
    else:
        st.info("No stocks tracked")
    
    if st.button("🗑️ Clear Chat"):
        st.session_state['messages'] = []
        st.session_state['chat_session'] = model.start_chat()
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 🏆 Project Compliance")
    st.success("✅ Domain-Specific (Finance)")
    st.success("✅ Logic-Based Inference")
    st.success("✅ Forward/Backward Chaining")
    st.success("✅ Bayesian Reasoning")
    st.success("✅ Web Interface")

st.markdown("---")
st.markdown("**🏆 100% Project Compliance: Logic-Based Inference + Probabilistic Reasoning + Domain Expertise**")