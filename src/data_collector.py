"""
Data Collector Module for Soccer Agent
Handles data collection from StatsBomb and other sources
"""

import pandas as pd
import numpy as np
from statsbombpy import sb
from mplsoccer import Sbopen
import requests
from bs4 import BeautifulSoup
import time
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StatsBombDataCollector:
    """Collects data from StatsBomb Open Data"""
    
    def __init__(self):
        self.parser = Sbopen(dataframe=True)
        
    def get_competitions(self) -> pd.DataFrame:
        """Get all available competitions"""
        try:
            return sb.competitions()
        except Exception as e:
            logger.error(f"Error fetching competitions: {e}")
            return pd.DataFrame()
    
    def get_matches(self, competition_id: int, season_id: int) -> pd.DataFrame:
        """Get matches for a specific competition and season"""
        try:
            return sb.matches(competition_id=competition_id, season_id=season_id)
        except Exception as e:
            logger.error(f"Error fetching matches: {e}")
            return pd.DataFrame()
    
    def get_events(self, match_id: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Get events for a specific match"""
        try:
            return self.parser.event(match_id)
        except Exception as e:
            logger.error(f"Error fetching events for match {match_id}: {e}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    def get_player_stats(self, match_id: int, player_name: str = None) -> pd.DataFrame:
        """Get player statistics for a specific match"""
        try:
            events, related, freeze, tactics = self.get_events(match_id)
            
            if player_name:
                player_events = events[events['player_name'] == player_name]
            else:
                player_events = events
                
            return player_events
        except Exception as e:
            logger.error(f"Error fetching player stats: {e}")
            return pd.DataFrame()
    
    def get_player_career_data(self, player_name: str, competitions: List[int] = None) -> pd.DataFrame:
        """Get career data for a specific player across multiple competitions"""
        try:
            all_events = []
            
            if not competitions:
                # Default to major competitions
                competitions = [43, 49, 72, 39]  # Eredivisie, Premier League, La Liga, Serie A
            
            for comp_id in competitions:
                comp_matches = self.get_matches(competition_id=comp_id, season_id=3)
                
                for _, match in comp_matches.iterrows():
                    events, _, _, _ = self.get_events(match['match_id'])
                    player_events = events[events['player_name'] == player_name]
                    
                    if not player_events.empty:
                        player_events['competition_id'] = comp_id
                        player_events['match_id'] = match['match_id']
                        all_events.append(player_events)
                        
                    time.sleep(0.1)  # Rate limiting
            
            if all_events:
                return pd.concat(all_events, ignore_index=True)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching career data for {player_name}: {e}")
            return pd.DataFrame()

class WebScraper:
    """Scrapes additional player data from various sources"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_transfermarkt_data(self, player_name: str) -> Dict:
        """Scrape player data from Transfermarkt"""
        try:
            # This is a simplified version - in production you'd need proper API access
            search_url = f"https://www.transfermarkt.com/schnellsuche/ergebnis/schnellsuche?query={player_name}"
            response = self.session.get(search_url)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                # Extract relevant data (simplified)
                return {
                    'market_value': 'N/A',
                    'contract_until': 'N/A',
                    'nationality': 'N/A'
                }
            else:
                return {}
                
        except Exception as e:
            logger.error(f"Error scraping Transfermarkt data: {e}")
            return {}
    
    def get_football_api_data(self, player_name: str, api_key: str = None) -> Dict:
        """Get data from football API (if available)"""
        if not api_key:
            return {}
            
        try:
            # Example API call (you'd need to implement based on specific API)
            url = f"https://api.football-data.org/v2/players?name={player_name}"
            headers = {'X-Auth-Token': api_key}
            
            response = self.session.get(url, headers=headers)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {}
                
        except Exception as e:
            logger.error(f"Error fetching football API data: {e}")
            return {}

class DataAggregator:
    """Aggregates data from multiple sources"""
    
    def __init__(self):
        self.statsbomb = StatsBombDataCollector()
        self.web_scraper = WebScraper()
    
    def get_complete_player_profile(self, player_name: str, match_id: int = None) -> Dict:
        """Get complete player profile from all available sources"""
        
        profile = {
            'player_name': player_name,
            'statsbomb_data': {},
            'web_data': {},
            'match_specific_data': {}
        }
        
        # Get StatsBomb career data
        career_data = self.statsbomb.get_player_career_data(player_name)
        if not career_data.empty:
            profile['statsbomb_data']['career'] = career_data.to_dict('records')
            
            # Calculate aggregated metrics
            profile['statsbomb_data']['aggregated_metrics'] = self._calculate_aggregated_metrics(career_data)
        
        # Get match-specific data if provided
        if match_id:
            match_data = self.statsbomb.get_player_stats(match_id, player_name)
            if not match_data.empty:
                profile['match_specific_data'] = match_data.to_dict('records')
        
        # Get web-scraped data
        profile['web_data'] = self.web_scraper.get_transfermarkt_data(player_name)
        
        return profile
    
    def _calculate_aggregated_metrics(self, career_data: pd.DataFrame) -> Dict:
        """Calculate aggregated metrics from career data"""
        metrics = {}
        
        try:
            # Pass metrics
            pass_events = career_data[career_data['type_name'] == 'Pass']
            if not pass_events.empty:
                metrics['total_passes'] = len(pass_events)
                metrics['completed_passes'] = len(pass_events[pass_events['outcome_name'].isna()])
                metrics['pass_accuracy'] = metrics['completed_passes'] / metrics['total_passes'] if metrics['total_passes'] > 0 else 0
            
            # Shot metrics
            shot_events = career_data[career_data['type_name'] == 'Shot']
            if not shot_events.empty:
                metrics['total_shots'] = len(shot_events)
                metrics['shots_on_target'] = len(shot_events[shot_events['outcome_name'] == 'Goal'])
                metrics['shot_accuracy'] = metrics['shots_on_target'] / metrics['total_shots'] if metrics['total_shots'] > 0 else 0
            
            # Dribble metrics
            dribble_events = career_data[career_data['type_name'] == 'Dribble']
            if not dribble_events.empty:
                metrics['total_dribbles'] = len(dribble_events)
                metrics['successful_dribbles'] = len(dribble_events[dribble_events['outcome_name'] == 'Complete'])
                metrics['dribble_success_rate'] = metrics['successful_dribbles'] / metrics['total_dribbles'] if metrics['total_dribbles'] > 0 else 0
            
            # Minutes played (approximate)
            metrics['total_minutes'] = len(career_data) * 0.1  # Rough estimate
            
        except Exception as e:
            logger.error(f"Error calculating aggregated metrics: {e}")
        
        return metrics 