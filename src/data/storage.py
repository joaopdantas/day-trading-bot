"""
Storage Module for Day Trading Bot.

This module handles the storage and retrieval of market data using MongoDB.
"""

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Union, Any

import pandas as pd
import pymongo
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class MarketDataStorage:
    """Class for storing and retrieving market data using MongoDB."""
    
    def __init__(self, mongo_uri: Optional[str] = None, db_name: str = "market_data"):
        """
        Initialize the data storage.
        
        Args:
            mongo_uri: MongoDB connection URI (optional if provided via env vars)
            db_name: Name of the database to use
        """
        self.mongo_uri = mongo_uri or os.getenv("MONGO_URI")
        if not self.mongo_uri:
            logger.warning("No MongoDB URI provided. Using in-memory storage instead.")
            self.client = None
            self.db = None
        else:
            try:
                # Connect to MongoDB
                self.client = MongoClient(self.mongo_uri)
                self.db = self.client[db_name]
                logger.info(f"Connected to MongoDB database: {db_name}")
                
                # Create indexes
                self._create_indexes()
            except PyMongoError as e:
                logger.error(f"Failed to connect to MongoDB: {e}")
                self.client = None
                self.db = None
    
    def _create_indexes(self):
        """Create indexes for efficient queries."""
        if self.db:
            try:
                # Index for historical data
                self.db.historical_data.create_index([
                    ("symbol", pymongo.ASCENDING),
                    ("interval", pymongo.ASCENDING),
                    ("timestamp", pymongo.DESCENDING)
                ])
                
                # Index for pattern recognition
                self.db.patterns.create_index([
                    ("symbol", pymongo.ASCENDING),
                    ("pattern_type", pymongo.ASCENDING),
                    ("timestamp", pymongo.DESCENDING)
                ])
                
                # Index for trading signals
                self.db.signals.create_index([
                    ("symbol", pymongo.ASCENDING),
                    ("timestamp", pymongo.DESCENDING)
                ])
                
                logger.info("Created MongoDB indexes")
            except PyMongoError as e:
                logger.error(f"Failed to create indexes: {e}")
    
    def store_historical_data(
        self, 
        symbol: str, 
        interval: str,
        data: pd.DataFrame,
        source: str = "unknown"
    ) -> bool:
        """
        Store historical market data for a symbol.
        
        Args:
            symbol: The stock/cryptocurrency symbol
            interval: Time interval of the data (e.g., '1d', '1h')
            data: DataFrame with market data
            source: Source of the data (e.g., 'alpha_vantage', 'yahoo_finance')
            
        Returns:
            True if storage was successful, False otherwise
        """
        if self.db is None:
            logger.warning("MongoDB not available. Data not stored.")
            return False
        
        if data.empty:
            logger.warning(f"Empty DataFrame provided for {symbol}. Nothing to store.")
            return False
        
        try:
            # Make sure index is datetime and convert to string for MongoDB
            if not isinstance(data.index, pd.DatetimeIndex):
                logger.warning("DataFrame index is not DatetimeIndex. Attempting to convert.")
                data.index = pd.to_datetime(data.index)
            
            # Reset index to make timestamp a column
            data_reset = data.reset_index()
            data_reset.rename(columns={'index': 'timestamp'}, inplace=True)
            
            # Convert DataFrame to list of dictionaries for MongoDB
            records = data_reset.to_dict('records')
            
            # Add metadata to each record
            for record in records:
                record['symbol'] = symbol
                record['interval'] = interval
                record['source'] = source
                record['stored_at'] = datetime.now()
            
            # Use bulk operations for efficiency
            bulk_ops = []
            for record in records:
                # Create filter for upsert (update if exists, insert if not)
                filter_doc = {
                    'symbol': symbol,
                    'interval': interval,
                    'timestamp': record['timestamp']
                }
                
                # Create update operation
                update_doc = {
                    '$set': record,
                    '$currentDate': {'last_modified': True}
                }
                
                # Add to bulk operations
                bulk_ops.append(
                    pymongo.UpdateOne(filter_doc, update_doc, upsert=True)
                )
            
            # Execute bulk operations
            if bulk_ops:
                result = self.db.historical_data.bulk_write(bulk_ops)
                logger.info(f"Stored {len(records)} records for {symbol} {interval}: "
                           f"{result.upserted_count} inserted, {result.modified_count} updated")
                return True
            else:
                logger.warning(f"No records to store for {symbol} {interval}")
                return False
                
        except Exception as e:
            logger.error(f"Error storing historical data for {symbol}: {e}")
            return False
    
    def retrieve_historical_data(
        self,
        symbol: str,
        interval: str = "1d",
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Retrieve historical market data for a symbol.
        
        Args:
            symbol: The stock/cryptocurrency symbol
            interval: Time interval of the data (e.g., '1d', '1h')
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            limit: Maximum number of records to retrieve
            
        Returns:
            DataFrame with historical price data
        """
        if self.db is None:
            logger.warning("MongoDB not available. Cannot retrieve data.")
            return pd.DataFrame()
        
        try:
            # Build query
            query = {
                'symbol': symbol,
                'interval': interval
            }
            
            # Add date range if provided
            if start_date or end_date:
                query['timestamp'] = {}
                if start_date:
                    if isinstance(start_date, str):
                        start_date = pd.to_datetime(start_date)
                    query['timestamp']['$gte'] = start_date
                
                if end_date:
                    if isinstance(end_date, str):
                        end_date = pd.to_datetime(end_date)
                    query['timestamp']['$lte'] = end_date
            
            # Execute query
            cursor = self.db.historical_data.find(
                query,
                {'_id': 0}  # Exclude MongoDB ID
            ).sort('timestamp', pymongo.DESCENDING)
            
            # Apply limit if provided
            if limit:
                cursor = cursor.limit(limit)
            
            # Convert to DataFrame
            data = list(cursor)
            if not data:
                logger.warning(f"No data found for {symbol} {interval}")
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def store_patterns(
        self,
        symbol: str,
        patterns: pd.DataFrame
    ) -> bool:
        """
        Store detected patterns for a symbol.
        
        Args:
            symbol: The stock/cryptocurrency symbol
            patterns: DataFrame with pattern data
            
        Returns:
            True if storage was successful, False otherwise
        """
        if self.db is None:
            logger.warning("MongoDB not available. Patterns not stored.")
            return False
        
        if patterns.empty:
            logger.warning(f"Empty patterns DataFrame provided for {symbol}. Nothing to store.")
            return False
        
        try:
            # Reset index to make timestamp a column
            patterns_reset = patterns.reset_index()
            
            # Ensure we have a timestamp column
            if 'timestamp' not in patterns_reset.columns and 'index' in patterns_reset.columns:
                patterns_reset.rename(columns={'index': 'timestamp'}, inplace=True)
            
            # Convert DataFrame to list of dictionaries for MongoDB
            records = patterns_reset.to_dict('records')
            
            # Add metadata to each record
            for record in records:
                record['symbol'] = symbol
                record['stored_at'] = datetime.now()
            
            # Use bulk operations for efficiency
            bulk_ops = []
            for record in records:
                # Create filter for upsert
                filter_doc = {
                    'symbol': symbol,
                    'timestamp': record['timestamp']
                }
                
                # Create update operation
                update_doc = {
                    '$set': record,
                    '$currentDate': {'last_modified': True}
                }
                
                # Add to bulk operations
                bulk_ops.append(
                    pymongo.UpdateOne(filter_doc, update_doc, upsert=True)
                )
            
            # Execute bulk operations
            if bulk_ops:
                result = self.db.patterns.bulk_write(bulk_ops)
                logger.info(f"Stored {len(records)} patterns for {symbol}: "
                           f"{result.upserted_count} inserted, {result.modified_count} updated")
                return True
            else:
                logger.warning(f"No patterns to store for {symbol}")
                return False
                
        except Exception as e:
            logger.error(f"Error storing patterns for {symbol}: {e}")
            return False
    
    def retrieve_patterns(
        self,
        symbol: str,
        pattern_type: Optional[str] = None,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Retrieve detected patterns for a symbol.
        
        Args:
            symbol: The stock/cryptocurrency symbol
            pattern_type: Type of pattern to retrieve (e.g., 'bullish_engulfing')
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            limit: Maximum number of records to retrieve
            
        Returns:
            DataFrame with pattern data
        """
        if self.db is None:
            logger.warning("MongoDB not available. Cannot retrieve patterns.")
            return pd.DataFrame()
        
        try:
            # Build query
            query = {'symbol': symbol}
            
            if pattern_type:
                query[f'pattern_{pattern_type}'] = 1
            
            # Add date range if provided
            if start_date or end_date:
                query['timestamp'] = {}
                if start_date:
                    if isinstance(start_date, str):
                        start_date = pd.to_datetime(start_date)
                    query['timestamp']['$gte'] = start_date
                
                if end_date:
                    if isinstance(end_date, str):
                        end_date = pd.to_datetime(end_date)
                    query['timestamp']['$lte'] = end_date
            
            # Execute query
            cursor = self.db.patterns.find(
                query,
                {'_id': 0}  # Exclude MongoDB ID
            ).sort('timestamp', pymongo.DESCENDING)
            
            # Apply limit if provided
            if limit:
                cursor = cursor.limit(limit)
            
            # Convert to DataFrame
            data = list(cursor)
            if not data:
                logger.warning(f"No patterns found for {symbol}")
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            
            # Set timestamp as index
            if 'timestamp' in df.columns:
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving patterns for {symbol}: {e}")
            return pd.DataFrame()
    
    def store_trading_signals(
        self,
        symbol: str,
        signals: pd.DataFrame
    ) -> bool:
        """
        Store trading signals for a symbol.
        
        Args:
            symbol: The stock/cryptocurrency symbol
            signals: DataFrame with signal data (should contain timestamp and signal type)
            
        Returns:
            True if storage was successful, False otherwise
        """
        if self.db is None:
            logger.warning("MongoDB not available. Signals not stored.")
            return False
        
        if signals.empty:
            logger.warning(f"Empty signals DataFrame provided for {symbol}. Nothing to store.")
            return False
        
        try:
            # Reset index to make timestamp a column
            signals_reset = signals.reset_index()
            
            # Ensure we have a timestamp column
            if 'timestamp' not in signals_reset.columns and 'index' in signals_reset.columns:
                signals_reset.rename(columns={'index': 'timestamp'}, inplace=True)
            
            # Convert DataFrame to list of dictionaries for MongoDB
            records = signals_reset.to_dict('records')
            
            # Add metadata to each record
            for record in records:
                record['symbol'] = symbol
                record['stored_at'] = datetime.now()
            
            # Use bulk operations for efficiency
            bulk_ops = []
            for record in records:
                # Create filter for upsert
                filter_doc = {
                    'symbol': symbol,
                    'timestamp': record['timestamp']
                }
                
                # Create update operation
                update_doc = {
                    '$set': record,
                    '$currentDate': {'last_modified': True}
                }
                
                # Add to bulk operations
                bulk_ops.append(
                    pymongo.UpdateOne(filter_doc, update_doc, upsert=True)
                )
            
            # Execute bulk operations
            if bulk_ops:
                result = self.db.signals.bulk_write(bulk_ops)
                logger.info(f"Stored {len(records)} signals for {symbol}: "
                           f"{result.upserted_count} inserted, {result.modified_count} updated")
                return True
            else:
                logger.warning(f"No signals to store for {symbol}")
                return False
                
        except Exception as e:
            logger.error(f"Error storing signals for {symbol}: {e}")
            return False
    
    def retrieve_trading_signals(
        self,
        symbol: str,
        signal_type: Optional[str] = None,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Retrieve trading signals for a symbol.
        
        Args:
            symbol: The stock/cryptocurrency symbol
            signal_type: Type of signal to retrieve (e.g., 'buy', 'sell')
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            limit: Maximum number of records to retrieve
            
        Returns:
            DataFrame with signal data
        """
        if self.db is None:
            logger.warning("MongoDB not available. Cannot retrieve signals.")
            return pd.DataFrame()
        
        try:
            # Build query
            query = {'symbol': symbol}
            
            if signal_type:
                query['signal_type'] = signal_type
            
            # Add date range if provided
            if start_date or end_date:
                query['timestamp'] = {}
                if start_date:
                    if isinstance(start_date, str):
                        start_date = pd.to_datetime(start_date)
                    query['timestamp']['$gte'] = start_date
                
                if end_date:
                    if isinstance(end_date, str):
                        end_date = pd.to_datetime(end_date)
                    query['timestamp']['$lte'] = end_date
            
            # Execute query
            cursor = self.db.signals.find(
                query,
                {'_id': 0}  # Exclude MongoDB ID
            ).sort('timestamp', pymongo.DESCENDING)
            
            # Apply limit if provided
            if limit:
                cursor = cursor.limit(limit)
            
            # Convert to DataFrame
            data = list(cursor)
            if not data:
                logger.warning(f"No signals found for {symbol}")
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            
            # Set timestamp as index
            if 'timestamp' in df.columns:
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving signals for {symbol}: {e}")
            return pd.DataFrame()
    
    def delete_historical_data(
        self,
        symbol: Optional[str] = None,
        interval: Optional[str] = None,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None
    ) -> int:
        """
        Delete historical market data.
        
        Args:
            symbol: The stock/cryptocurrency symbol (None for all symbols)
            interval: Time interval of the data (None for all intervals)
            start_date: Start date for data deletion
            end_date: End date for data deletion
            
        Returns:
            Number of records deleted
        """
        if self.db is None:
            logger.warning("MongoDB not available. Cannot delete data.")
            return 0
        
        try:
            # Build query
            query = {}
            
            if symbol:
                query['symbol'] = symbol
            
            if interval:
                query['interval'] = interval
            
            # Add date range if provided
            if start_date or end_date:
                query['timestamp'] = {}
                if start_date:
                    if isinstance(start_date, str):
                        start_date = pd.to_datetime(start_date)
                    query['timestamp']['$gte'] = start_date
                
                if end_date:
                    if isinstance(end_date, str):
                        end_date = pd.to_datetime(end_date)
                    query['timestamp']['$lte'] = end_date
            
            # Execute delete operation
            result = self.db.historical_data.delete_many(query)
            deleted_count = result.deleted_count
            
            logger.info(f"Deleted {deleted_count} historical data records")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error deleting historical data: {e}")
            return 0
    
    def get_available_symbols(self) -> List[str]:
        """
        Get list of all symbols available in the database.
        
        Returns:
            List of symbol strings
        """
        if self.db is None:
            logger.warning("MongoDB not available. Cannot retrieve symbols.")
            return []
        
        try:
            # Distinct query to get unique symbols
            symbols = self.db.historical_data.distinct('symbol')
            return symbols
            
        except Exception as e:
            logger.error(f"Error retrieving available symbols: {e}")
            return []
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about stored data.
        
        Returns:
            Dictionary with statistics about the data
        """
        if self.db is None:
            logger.warning("MongoDB not available. Cannot retrieve statistics.")
            return {}
        
        try:
            stats = {
                'total_records': self.db.historical_data.count_documents({}),
                'symbols_count': len(self.db.historical_data.distinct('symbol')),
                'patterns_count': self.db.patterns.count_documents({}),
                'signals_count': self.db.signals.count_documents({})
            }
            
            # Get count by symbol
            pipeline = [
                {"$group": {"_id": "$symbol", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}
            ]
            symbol_counts = list(self.db.historical_data.aggregate(pipeline))
            stats['records_by_symbol'] = {doc['_id']: doc['count'] for doc in symbol_counts}
            
            return stats
            
        except Exception as e:
            logger.error(f"Error retrieving data statistics: {e}")
            return {}
    
    def close(self):
        """Close the MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("Closed MongoDB connection")


# Factory function to get a storage instance
def get_storage(mongo_uri: Optional[str] = None, db_name: str = "market_data") -> MarketDataStorage:
    """
    Factory function to return a storage instance.
    
    Args:
        mongo_uri: MongoDB connection URI
        db_name: Name of the database to use
        
    Returns:
        Instance of MarketDataStorage
    """
    return MarketDataStorage(mongo_uri, db_name)