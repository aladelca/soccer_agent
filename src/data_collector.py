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
import os
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass

# Setup logging with line numbers
logging.basicConfig(
    format='%(asctime)s - [%(filename)s:%(lineno)d] - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

@dataclass
class PlayerSearchResult:
    """Data class for player search results"""
    player_id: str
    player_name: str
    club: str
    confidence_score: float

class StatsBombDataCollector:
    """Collects data from StatsBomb Open Data"""
    
    def __init__(self):
        self.parser = Sbopen(dataframe=True)
        
    def get_competitions(self) -> pd.DataFrame:
        """Get all available competitions"""
        try:
            result = sb.competitions()
            if isinstance(result, pd.DataFrame):
                return result
            else:
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching competitions: {e}")
            return pd.DataFrame()
    
    def get_matches(self, competition_id: int, season_id: int) -> pd.DataFrame:
        """Get matches for a specific competition and season"""
        try:
            result = sb.matches(competition_id=competition_id, season_id=season_id)
            if isinstance(result, pd.DataFrame):
                return result
            else:
                return pd.DataFrame()
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
    
    def calculate_player_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Calculate comprehensive player statistics from StatsBomb events dataframe
        
        Args:
            df: DataFrame with player events from StatsBomb
            
        Returns:
            Dictionary with calculated statistics
        """
        stats = {}
        
        # Add player and team information
        if 'player_name' in df.columns and not df.empty:
            stats['player_name'] = df['player_name'].iloc[0]
        if 'team_name' in df.columns and not df.empty:
            stats['team_name'] = df['team_name'].iloc[0]
        
        # 1. Pass Statistics
        pass_events = df[df['type_name'] == 'Pass']
        if not pass_events.empty:
            stats['total_passes'] = len(pass_events)
            stats['successful_passes'] = len(pass_events[pass_events['outcome_name'].isna()])
            stats['intercepted_passes'] = len(pass_events[pass_events['outcome_name'] == 'Incomplete'])
            stats['pass_accuracy'] = stats['successful_passes'] / stats['total_passes'] if stats['total_passes'] > 0 else 0
            
            # Average pass distance
            if 'pass_length' in pass_events.columns:
                stats['avg_pass_distance'] = pass_events['pass_length'].mean()
            
            # Favorite pass recipient
            if 'pass_recipient_name' in pass_events.columns:
                recipient_counts = pass_events['pass_recipient_name'].value_counts()
                if not recipient_counts.empty:
                    stats['favorite_pass_recipient'] = recipient_counts.index[0]
                    stats['passes_to_favorite_recipient'] = recipient_counts.iloc[0]
        
        # 2. Shot Statistics
        shot_events = df[df['type_name'] == 'Shot']
        if not shot_events.empty:
            stats['total_shots'] = len(shot_events)
            stats['shots_on_target'] = len(shot_events[shot_events['outcome_name'] == 'Goal'])
            stats['shot_accuracy'] = stats['shots_on_target'] / stats['total_shots'] if stats['total_shots'] > 0 else 0
        
        # 3. Ball Carry Statistics
        carry_events = df[df['type_name'] == 'Carry']
        if not carry_events.empty:
            stats['total_carries'] = len(carry_events)
            
            # Calculate carry distance using coordinates
            if 'x' in carry_events.columns and 'y' in carry_events.columns and 'end_x' in carry_events.columns and 'end_y' in carry_events.columns:
                carry_distances = []
                for _, row in carry_events.iterrows():
                    if pd.notna(row['x']) and pd.notna(row['y']) and pd.notna(row['end_x']) and pd.notna(row['end_y']):
                        distance = np.sqrt((row['end_x'] - row['x'])**2 + (row['end_y'] - row['y'])**2)
                        carry_distances.append(distance)
                
                if carry_distances:
                    stats['avg_carry_distance'] = np.mean(carry_distances)
                    stats['total_carry_distance'] = np.sum(carry_distances)
            
            # Time carrying the ball (using duration field)
            if 'duration' in carry_events.columns:
                total_carry_time = carry_events['duration'].sum()
                stats['total_carry_time_seconds'] = total_carry_time
                stats['avg_carry_time_seconds'] = total_carry_time / len(carry_events) if len(carry_events) > 0 else 0
        
        # 4. Body Part Usage for Passes
        if not pass_events.empty and 'body_part_name' in pass_events.columns:
            body_part_counts = pass_events['body_part_name'].value_counts()
            if not body_part_counts.empty:
                stats['favorite_body_part_for_pass'] = body_part_counts.index[0]
                stats['passes_with_favorite_body_part'] = body_part_counts.iloc[0]
                stats['body_part_breakdown'] = body_part_counts.to_dict()
        
        # 5. Position Analysis
        if 'x' in df.columns and 'y' in df.columns:
            stats['avg_position_x'] = df['x'].mean()
            stats['avg_position_y'] = df['y'].mean()
            stats['position_range_x'] = df['x'].max() - df['x'].min()
            stats['position_range_y'] = df['y'].max() - df['y'].min()
        
        # 6. Time-based Statistics (FIXED for match timestamp)
        if 'timestamp' in df.columns:
            df_copy = df.copy()
            
            def timestamp_to_seconds(timestamp_str):
                try:
                    time_parts = str(timestamp_str).split(':')
                    if len(time_parts) >= 3:
                        hours = int(time_parts[0])
                        minutes = int(time_parts[1])
                        seconds_part = time_parts[2]
                        
                        if '.' in seconds_part:
                            seconds, microseconds = seconds_part.split('.')
                            seconds = int(seconds)
                            microseconds = int(microseconds)
                        else:
                            seconds = int(seconds_part)
                            microseconds = 0
                        
                        total_seconds = hours * 3600 + minutes * 60 + seconds + microseconds / 1000000
                        return total_seconds
                    return 0
                except:
                    return 0
            
            df_copy['time_seconds'] = df_copy['timestamp'].apply(timestamp_to_seconds)
            
            if len(df_copy) > 1:
                stats['first_action_time'] = df_copy['time_seconds'].min()
                stats['last_action_time'] = df_copy['time_seconds'].max()
                stats['total_playing_time_seconds'] = stats['last_action_time'] - stats['first_action_time']
                stats['actions_per_minute'] = len(df) / (stats['total_playing_time_seconds'] / 60) if stats['total_playing_time_seconds'] > 0 else 0
            else:
                stats['total_playing_time_seconds'] = 0
                stats['actions_per_minute'] = 0
        
        # 7. Pressure and Defensive Actions
        pressure_events = df[df['type_name'] == 'Pressure']
        if not pressure_events.empty:
            stats['total_pressures'] = len(pressure_events)
        
        # 8. Substitution Information
        substitution_events = df[df['type_name'] == 'Substitution']
        if not substitution_events.empty:
            stats['was_substituted'] = True
            if 'substitution_replacement_name' in substitution_events.columns:
                stats['replaced_by'] = substitution_events['substitution_replacement_name'].iloc[0]
        else:
            stats['was_substituted'] = False
        
        # 9. Formation and Tactical Information
        if 'tactics_formation' in df.columns:
            formation = df['tactics_formation'].iloc[0] if not df.empty else None
            stats['formation'] = formation
        
        # 10. Match Context
        if 'match_id' in df.columns:
            stats['match_id'] = df['match_id'].iloc[0]
        if 'position_name' in df.columns:
            stats['position'] = df['position_name'].iloc[0]
        
        return stats

    def calculate_period_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Calculate statistics per period (1st half = 45 min, 2nd half = 45 min, etc.)
        Each period has its own timestamp starting from 00:00:00
        """
        period_stats = {}
        
        # Add player and team information to period stats
        if 'player_name' in df.columns and not df.empty:
            period_stats['player_name'] = df['player_name'].iloc[0]
        if 'team_name' in df.columns and not df.empty:
            period_stats['team_name'] = df['team_name'].iloc[0]
        
        if 'period' not in df.columns:
            return period_stats
        
        for period in df['period'].unique():
            period_data = df[df['period'] == period]
            period_stats[f'period_{period}'] = self.calculate_player_statistics(period_data)
            
            # Add period-specific time analysis
            if 'timestamp' in period_data.columns and not period_data.empty:
                period_stats[f'period_{period}']['period_duration_minutes'] = 45  # Each period is 45 minutes
                
                # Convert timestamp to seconds within the period
                def period_timestamp_to_seconds(timestamp_str):
                    try:
                        time_parts = str(timestamp_str).split(':')
                        if len(time_parts) >= 3:
                            minutes = int(time_parts[1])
                            seconds_part = time_parts[2]
                            
                            if '.' in seconds_part:
                                seconds, microseconds = seconds_part.split('.')
                                seconds = int(seconds)
                                microseconds = int(microseconds)
                            else:
                                seconds = int(seconds_part)
                                microseconds = 0
                            
                            total_seconds = minutes * 60 + seconds + microseconds / 1000000
                            return total_seconds
                        return 0
                    except:
                        return 0
                
                period_data_copy = period_data.copy()
                period_data_copy['time_seconds'] = period_data_copy['timestamp'].apply(period_timestamp_to_seconds)
                
                if len(period_data_copy) > 1:
                    period_stats[f'period_{period}']['period_first_action_minute'] = period_data_copy['time_seconds'].min() / 60
                    period_stats[f'period_{period}']['period_last_action_minute'] = period_data_copy['time_seconds'].max() / 60
                    period_stats[f'period_{period}']['period_playing_time_minutes'] = (period_data_copy['time_seconds'].max() - period_data_copy['time_seconds'].min()) / 60
                    period_stats[f'period_{period}']['period_actions_per_minute'] = len(period_data) / period_stats[f'period_{period}']['period_playing_time_minutes'] if period_stats[f'period_{period}']['period_playing_time_minutes'] > 0 else 0
                else:
                    period_stats[f'period_{period}']['period_playing_time_minutes'] = 0
                    period_stats[f'period_{period}']['period_actions_per_minute'] = 0
        
        return period_stats

    def calculate_advanced_metrics(self, df: pd.DataFrame) -> Dict:
        """
        Calculate advanced metrics and insights for football analysis
        """
        advanced_stats = {}
        
        # Add player and team information
        if 'player_name' in df.columns and not df.empty:
            advanced_stats['player_name'] = df['player_name'].iloc[0]
        if 'team_name' in df.columns and not df.empty:
            advanced_stats['team_name'] = df['team_name'].iloc[0]
        
        # 1. Pass Analysis by Distance
        pass_events = df[df['type_name'] == 'Pass']
        if not pass_events.empty and 'pass_length' in pass_events.columns:
            # Short passes (0-15m)
            short_passes = pass_events[pass_events['pass_length'] <= 15]
            if not short_passes.empty:
                short_success = len(short_passes[short_passes['outcome_name'].isna()])
                advanced_stats['short_pass_accuracy'] = short_success / len(short_passes)
                advanced_stats['short_passes_count'] = len(short_passes)
            
            # Medium passes (15-30m)
            medium_passes = pass_events[(pass_events['pass_length'] > 15) & (pass_events['pass_length'] <= 30)]
            if not medium_passes.empty:
                medium_success = len(medium_passes[medium_passes['outcome_name'].isna()])
                advanced_stats['medium_pass_accuracy'] = medium_success / len(medium_passes)
                advanced_stats['medium_passes_count'] = len(medium_passes)
            
            # Long passes (>30m)
            long_passes = pass_events[pass_events['pass_length'] > 30]
            if not long_passes.empty:
                long_success = len(long_passes[long_passes['outcome_name'].isna()])
                advanced_stats['long_pass_accuracy'] = long_success / len(long_passes)
                advanced_stats['long_passes_count'] = len(long_passes)
        
        # 2. Progressive Actions Analysis
        if 'x' in df.columns and 'y' in df.columns and 'end_x' in df.columns and 'end_y' in df.columns:
            progressive_actions = 0
            total_distance_gained = 0
            
            for _, row in df.iterrows():
                if pd.notna(row['x']) and pd.notna(row['y']) and pd.notna(row['end_x']) and pd.notna(row['end_y']):
                    # Calculate distance gained towards opponent's goal
                    distance_gained = row['end_x'] - row['x']
                    if distance_gained > 5:  # Progressive if gained more than 5m forward
                        progressive_actions += 1
                        total_distance_gained += distance_gained
            
            advanced_stats['progressive_actions'] = progressive_actions
            advanced_stats['total_distance_gained'] = total_distance_gained
            advanced_stats['avg_distance_gained_per_action'] = total_distance_gained / len(df) if len(df) > 0 else 0
        
        # 3. Pressure Effectiveness
        pressure_events = df[df['type_name'] == 'Pressure']
        if not pressure_events.empty:
            # Count successful pressures (next event is turnover)
            successful_pressures = 0
            for i in range(len(pressure_events) - 1):
                if pressure_events.iloc[i+1]['type_name'] in ['Ball Recovery', 'Interception']:
                    successful_pressures += 1
            advanced_stats['pressure_success_rate'] = successful_pressures / len(pressure_events) if len(pressure_events) > 0 else 0
            advanced_stats['successful_pressures'] = successful_pressures
        
        # 4. Ball Recovery and Interceptions
        recovery_events = df[df['type_name'].isin(['Ball Recovery', 'Interception'])]
        advanced_stats['ball_recoveries'] = len(recovery_events)
        
        # 5. Dribble Analysis
        dribble_events = df[df['type_name'] == 'Dribble']
        if not dribble_events.empty:
            successful_dribbles = len(dribble_events[dribble_events['outcome_name'] == 'Complete'])
            advanced_stats['dribble_success_rate'] = successful_dribbles / len(dribble_events)
            advanced_stats['successful_dribbles'] = successful_dribbles
            advanced_stats['total_dribbles'] = len(dribble_events)
        
        # 6. Foul Analysis
        foul_events = df[df['type_name'] == 'Foul Committed']
        advanced_stats['fouls_committed'] = len(foul_events)
        
        foul_won_events = df[df['type_name'] == 'Foul Won']
        advanced_stats['fouls_won'] = len(foul_won_events)
        
        # 7. Aerial Duels
        aerial_events = df[df['type_name'] == 'Duel']
        if not aerial_events.empty and 'duel_type_name' in aerial_events.columns:
            aerial_duels = aerial_events[aerial_events['duel_type_name'] == 'Aerial']
            if not aerial_duels.empty:
                aerial_won = len(aerial_duels[aerial_duels['outcome_name'] == 'Won'])
                advanced_stats['aerial_duels_won'] = aerial_won
                advanced_stats['aerial_duels_total'] = len(aerial_duels)
                advanced_stats['aerial_duel_success_rate'] = aerial_won / len(aerial_duels)
        
        # 8. Cross Analysis
        if not pass_events.empty and 'pass_cross' in pass_events.columns:
            crosses = pass_events[pass_events['pass_cross'] == True]
            if not crosses.empty:
                successful_crosses = len(crosses[crosses['outcome_name'].isna()])
                advanced_stats['cross_accuracy'] = successful_crosses / len(crosses)
                advanced_stats['successful_crosses'] = successful_crosses
                advanced_stats['total_crosses'] = len(crosses)
        
        # 9. Key Passes and Assists
        if not pass_events.empty and 'pass_shot_assist' in pass_events.columns:
            key_passes = pass_events[pass_events['pass_shot_assist'] == True]
            advanced_stats['key_passes'] = len(key_passes)
        
        if not pass_events.empty and 'pass_goal_assist' in pass_events.columns:
            assists = pass_events[pass_events['pass_goal_assist'] == True]
            advanced_stats['assists'] = len(assists)
        
        # 10. Shot Analysis
        shot_events = df[df['type_name'] == 'Shot']
        if not shot_events.empty:
            # Shot locations
            if 'x' in shot_events.columns and 'y' in shot_events.columns:
                advanced_stats['avg_shot_distance'] = shot_events['x'].mean()
                advanced_stats['shots_from_inside_box'] = len(shot_events[shot_events['x'] > 83])
                advanced_stats['shots_from_outside_box'] = len(shot_events[shot_events['x'] <= 83])
            
            # Shot types
            if 'technique_name' in shot_events.columns:
                shot_techniques = shot_events['technique_name'].value_counts()
                advanced_stats['shot_techniques'] = shot_techniques.to_dict()
        
        # 11. Possession Analysis
        if 'possession' in df.columns:
            possession_counts = df['possession'].value_counts()
            advanced_stats['possession_sequences'] = len(possession_counts)
            advanced_stats['avg_actions_per_possession'] = len(df) / len(possession_counts) if len(possession_counts) > 0 else 0
        
        # 12. Time-based Performance (FIXED for 45-minute periods)
        if 'timestamp' in df.columns and 'period' in df.columns:
            df_copy = df.copy()
            
            def period_timestamp_to_seconds(row):
                try:
                    timestamp_str = str(row['timestamp'])
                    time_parts = timestamp_str.split(':')
                    if len(time_parts) >= 3:
                        minutes = int(time_parts[1])
                        seconds_part = time_parts[2]
                        
                        if '.' in seconds_part:
                            seconds, microseconds = seconds_part.split('.')
                            seconds = int(seconds)
                            microseconds = int(microseconds)
                        else:
                            seconds = int(seconds_part)
                            microseconds = 0
                        
                        total_seconds = minutes * 60 + seconds + microseconds / 1000000
                        return total_seconds
                    return 0
                except:
                    return 0
            
            df_copy['time_seconds'] = df_copy.apply(period_timestamp_to_seconds, axis=1)
            
            # Performance by match phase within each period
            for period in df_copy['period'].unique():
                period_data = df_copy[df_copy['period'] == period]
                
                # First 15 minutes of the period
                first_quarter = period_data[period_data['time_seconds'] <= 900]
                # Second 15 minutes of the period  
                second_quarter = period_data[(period_data['time_seconds'] > 900) & (period_data['time_seconds'] <= 1800)]
                # Third 15 minutes of the period
                third_quarter = period_data[(period_data['time_seconds'] > 1800) & (period_data['time_seconds'] <= 2700)]
                # Last 15 minutes of the period
                final_quarter = period_data[period_data['time_seconds'] > 2700]
                
                advanced_stats[f'period_{period}_actions_0_15min'] = len(first_quarter)
                advanced_stats[f'period_{period}_actions_15_30min'] = len(second_quarter)
                advanced_stats[f'period_{period}_actions_30_45min'] = len(third_quarter)
                advanced_stats[f'period_{period}_actions_45_60min'] = len(final_quarter)  # For extra time if any
        
        # 13. Defensive Actions
        defensive_events = df[df['type_name'].isin(['Tackle', 'Interception', 'Clearance', 'Block'])]
        advanced_stats['defensive_actions'] = len(defensive_events)
        
        # 14. Under Pressure Performance
        under_pressure_events = df[df['under_pressure'] == True]
        if not under_pressure_events.empty:
            advanced_stats['actions_under_pressure'] = len(under_pressure_events)
            advanced_stats['pressure_percentage'] = len(under_pressure_events) / len(df) * 100
            
            # Performance under pressure
            pressure_pass_events = under_pressure_events[under_pressure_events['type_name'] == 'Pass']
            if not pressure_pass_events.empty:
                pressure_pass_success = len(pressure_pass_events[pressure_pass_events['outcome_name'].isna()])
                advanced_stats['pass_accuracy_under_pressure'] = pressure_pass_success / len(pressure_pass_events)
        
        # 15. Set Piece Analysis
        set_piece_events = df[df['play_pattern_name'].isin(['From Free Kick', 'From Corner', 'From Throw In'])]
        advanced_stats['set_piece_actions'] = len(set_piece_events)
        
        return advanced_stats

    def calculate_comprehensive_player_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Comprehensive player analysis combining all statistics
        """
        analysis = {
            'basic_stats': self.calculate_player_statistics(df),
            'period_stats': self.calculate_period_statistics(df),
            'advanced_stats': self.calculate_advanced_metrics(df)
        }
        
        # Add player and team information to the main analysis
        if 'player_name' in df.columns and not df.empty:
            analysis['player_name'] = df['player_name'].iloc[0]
        if 'team_name' in df.columns and not df.empty:
            analysis['team_name'] = df['team_name'].iloc[0]
        
        # Add summary insights
        analysis['summary_insights'] = self.generate_summary_insights(analysis)
        
        return analysis

    def generate_summary_insights(self, analysis: Dict) -> Dict:
        """
        Generate summary insights from the analysis
        """
        insights = {}
        
        # Add player and team information to insights
        if 'player_name' in analysis:
            insights['player_name'] = analysis['player_name']
        if 'team_name' in analysis:
            insights['team_name'] = analysis['team_name']
        
        basic = analysis['basic_stats']
        advanced = analysis['advanced_stats']
        period_stats = analysis['period_stats']
        
        # Performance rating
        if 'pass_accuracy' in basic and 'actions_per_minute' in basic:
            performance_score = (basic['pass_accuracy'] * 0.4 + 
                            min(basic['actions_per_minute'] / 10, 1.0) * 0.3 +
                            (advanced.get('progressive_actions', 0) / max(len(analysis), 1)) * 0.3)
            insights['performance_score'] = min(performance_score, 1.0)
        
        # Playing style classification
        if 'pass_accuracy' in basic and 'avg_pass_distance' in basic:
            if basic['pass_accuracy'] > 0.85 and basic['avg_pass_distance'] < 20:
                insights['playing_style'] = 'Possession-based midfielder'
            elif basic['avg_pass_distance'] > 30:
                insights['playing_style'] = 'Long-range passer'
            elif advanced.get('dribble_success_rate', 0) > 0.7:
                insights['playing_style'] = 'Dribbling specialist'
            else:
                insights['playing_style'] = 'Balanced player'
        
        # Period performance comparison
        if len(period_stats) >= 2:
            period_1_actions = period_stats.get('period_1', {}).get('total_passes', 0)
            period_2_actions = period_stats.get('period_2', {}).get('total_passes', 0)
            
            if period_1_actions > 0 and period_2_actions > 0:
                performance_change = (period_2_actions - period_1_actions) / period_1_actions
                if performance_change > 0.2:
                    insights['performance_trend'] = 'Improved in second half'
                elif performance_change < -0.2:
                    insights['performance_trend'] = 'Declined in second half'
                else:
                    insights['performance_trend'] = 'Consistent performance'
        
        # Key strengths
        strengths = []
        if basic.get('pass_accuracy', 0) > 0.85:
            strengths.append('High pass accuracy')
        if advanced.get('progressive_actions', 0) > 10:
            strengths.append('Progressive play')
        if advanced.get('pressure_success_rate', 0) > 0.6:
            strengths.append('Effective pressing')
        
        insights['key_strengths'] = strengths
        
        return insights

    def get_player_stats(self, match_id: int, player_name: str) -> Dict:
        """
        Get comprehensive player statistics for a specific match
        
        Args:
            match_id: ID of the match to analyze
            player_name: Name of the player to analyze
            
        Returns:
            Dictionary containing comprehensive match analysis
        """
        try:
            events, related, freeze, tactics = self.get_events(match_id)
            
            if events.empty:
                return {
                    'player_name': player_name,
                    'match_id': match_id,
                    'error': 'No events found for this match',
                    'analysis': {}
                }
            
            player_events = events[events['player_name'] == player_name]
            
            if player_events.empty:
                return {
                    'player_name': player_name,
                    'match_id': match_id,
                    'error': f'Player {player_name} not found in match {match_id}',
                    'analysis': {}
                }
            
            # Calculate comprehensive analysis
            player_analysis = self.calculate_comprehensive_player_analysis(player_events)
            player_analysis['match_id'] = match_id
            
            # Add match context if available
            if not events.empty:
                player_analysis['match_teams'] = events['team_name'].unique().tolist()
                player_analysis['match_periods'] = events['period'].unique().tolist() if 'period' in events.columns else []
            
            return player_analysis
            
        except Exception as e:
            logger.error(f"Error fetching player stats for {player_name} in match {match_id}: {e}")
            return {
                'player_name': player_name,
                'match_id': match_id,
                'error': str(e),
                'analysis': {}
            }
    
    def get_player_career_data(self, player_name: str, competitions: Optional[List[int]] = None) -> Dict:
        """
        Get comprehensive career statistics for a specific player across multiple competitions
        
        Args:
            player_name: Name of the player to analyze
            competitions: List of competition IDs to analyze (default: major competitions)
            
        Returns:
            Dictionary containing comprehensive career analysis
        """
        try:
            all_events = []
            match_analyses = []
            
            if not competitions:
                # All competitions available
                competitions = list(int(i) for i in self.get_competitions()["competition_id"].unique() )
            
            logger.info(f"Fetching career data for {player_name} across {len(competitions)} competitions")
            
            seasons = list(int(i) for i in self.get_competitions()["season_id"].unique() )

            for comp_id in competitions:
                logger.info(f"Processing competition {comp_id}")
                for season in seasons:
                    comp_matches = self.get_matches(competition_id=comp_id, season_id=season)
                
                    for _, match in comp_matches.iterrows():
                        try:
                            events, _, _, _ = self.get_events(match['match_id'])
                            player_events = events[events['player_name'] == player_name]
                            
                            if not player_events.empty:
                                # Add competition and match context
                                player_events['competition_id'] = comp_id
                                player_events['match_id'] = match['match_id']
                                player_events['match_date'] = match.get('match_date', 'Unknown')
                                player_events['home_team'] = match.get('home_team_name', 'Unknown')
                                player_events['away_team'] = match.get('away_team_name', 'Unknown')
                                player_events['home_score'] = match.get('home_score', 0)
                                player_events['away_score'] = match.get('away_score', 0)
                                
                                all_events.append(player_events)
                                
                                # Calculate comprehensive analysis for this match
                                match_analysis = self.calculate_comprehensive_player_analysis(player_events)
                                match_analysis['match_id'] = match['match_id']
                                match_analysis['competition_id'] = comp_id
                                match_analysis['match_date'] = match.get('match_date', 'Unknown')
                                match_analysis['home_team'] = match.get('home_team_name', 'Unknown')
                                match_analysis['away_team'] = match.get('away_team_name', 'Unknown')
                                match_analysis['home_score'] = match.get('home_score', 0)
                                match_analysis['away_score'] = match.get('away_score', 0)
                                
                                match_analyses.append(match_analysis)
                                
                            time.sleep(0.1)  # Rate limiting
                            
                        except Exception as e:
                            logger.warning(f"Error processing match {match.get('match_id', 'Unknown')}: {e}")
                            continue
            
            if not all_events:
                return {
                    'player_name': player_name,
                    'error': f'No data found for player {player_name}',
                    'matches_analyzed': 0,
                    'career_summary': {},
                    'match_analyses': []
                }
            
            # Combine all events for overall career analysis
            combined_events = pd.concat(all_events, ignore_index=True)
            career_analysis = self.calculate_comprehensive_player_analysis(combined_events)
            
            # Calculate career aggregates
            career_summary = self._calculate_career_aggregates(match_analyses)
            
            # Add competition breakdown
            competition_breakdown = self._calculate_competition_breakdown(match_analyses)
            
            return {
                'player_name': player_name,
                'matches_analyzed': len(match_analyses),
                'total_events': len(combined_events),
                'career_analysis': career_analysis,
                'career_summary': career_summary,
                'competition_breakdown': competition_breakdown,
                'match_analyses': match_analyses,
                'raw_events': combined_events
            }
                
        except Exception as e:
            logger.error(f"Error fetching career data for {player_name}: {e}")
            return {
                'player_name': player_name,
                'error': str(e),
                'matches_analyzed': 0,
                'career_summary': {},
                'match_analyses': []
            }
    
    def _calculate_career_aggregates(self, match_analyses: List[Dict]) -> Dict:
        """
        Calculate aggregated career statistics from individual match analyses
        """
        if not match_analyses:
            return {}
        
        aggregates = {
            'total_matches': len(match_analyses),
            'total_minutes_played': 0,
            'total_passes': 0,
            'total_successful_passes': 0,
            'total_shots': 0,
            'total_goals': 0,
            'total_assists': 0,
            'total_key_passes': 0,
            'total_dribbles': 0,
            'total_successful_dribbles': 0,
            'total_pressures': 0,
            'total_defensive_actions': 0,
            'total_fouls_committed': 0,
            'total_fouls_won': 0,
            'total_aerial_duels': 0,
            'total_aerial_duels_won': 0,
            'total_crosses': 0,
            'total_successful_crosses': 0,
            'total_progressive_actions': 0,
            'total_distance_gained': 0,
            'total_carry_distance': 0,
            'total_carry_time': 0,
            'total_set_piece_actions': 0,
            'total_actions_under_pressure': 0,
            'performance_scores': [],
            'playing_styles': [],
            'performance_trends': []
        }
        
        for match_analysis in match_analyses:
            basic_stats = match_analysis.get('basic_stats', {})
            advanced_stats = match_analysis.get('advanced_stats', {})
            summary_insights = match_analysis.get('summary_insights', {})
            
            # Aggregate basic stats
            aggregates['total_minutes_played'] += basic_stats.get('total_playing_time_seconds', 0) / 60
            aggregates['total_passes'] += basic_stats.get('total_passes', 0)
            aggregates['total_successful_passes'] += basic_stats.get('successful_passes', 0)
            aggregates['total_shots'] += basic_stats.get('total_shots', 0)
            aggregates['total_goals'] += basic_stats.get('shots_on_target', 0)  # Assuming goals are shots on target
            
            # Aggregate advanced stats
            aggregates['total_assists'] += advanced_stats.get('assists', 0)
            aggregates['total_key_passes'] += advanced_stats.get('key_passes', 0)
            aggregates['total_dribbles'] += advanced_stats.get('total_dribbles', 0)
            aggregates['total_successful_dribbles'] += advanced_stats.get('successful_dribbles', 0)
            aggregates['total_pressures'] += basic_stats.get('total_pressures', 0)
            aggregates['total_defensive_actions'] += advanced_stats.get('defensive_actions', 0)
            aggregates['total_fouls_committed'] += advanced_stats.get('fouls_committed', 0)
            aggregates['total_fouls_won'] += advanced_stats.get('fouls_won', 0)
            aggregates['total_aerial_duels'] += advanced_stats.get('aerial_duels_total', 0)
            aggregates['total_aerial_duels_won'] += advanced_stats.get('aerial_duels_won', 0)
            aggregates['total_crosses'] += advanced_stats.get('total_crosses', 0)
            aggregates['total_successful_crosses'] += advanced_stats.get('successful_crosses', 0)
            aggregates['total_progressive_actions'] += advanced_stats.get('progressive_actions', 0)
            aggregates['total_distance_gained'] += advanced_stats.get('total_distance_gained', 0)
            aggregates['total_carry_distance'] += basic_stats.get('total_carry_distance', 0)
            aggregates['total_carry_time'] += basic_stats.get('total_carry_time_seconds', 0)
            aggregates['total_set_piece_actions'] += advanced_stats.get('set_piece_actions', 0)
            aggregates['total_actions_under_pressure'] += advanced_stats.get('actions_under_pressure', 0)
            
            # Collect performance metrics
            if 'performance_score' in summary_insights:
                aggregates['performance_scores'].append(summary_insights['performance_score'])
            if 'playing_style' in summary_insights:
                aggregates['playing_styles'].append(summary_insights['playing_style'])
            if 'performance_trend' in summary_insights:
                aggregates['performance_trends'].append(summary_insights['performance_trend'])
        
        # Calculate averages and rates
        if aggregates['total_matches'] > 0:
            aggregates['avg_minutes_per_match'] = aggregates['total_minutes_played'] / aggregates['total_matches']
            aggregates['avg_passes_per_match'] = aggregates['total_passes'] / aggregates['total_matches']
            aggregates['avg_shots_per_match'] = aggregates['total_shots'] / aggregates['total_matches']
            aggregates['avg_assists_per_match'] = aggregates['total_assists'] / aggregates['total_matches']
            aggregates['avg_key_passes_per_match'] = aggregates['total_key_passes'] / aggregates['total_matches']
            aggregates['avg_dribbles_per_match'] = aggregates['total_dribbles'] / aggregates['total_matches']
            aggregates['avg_pressures_per_match'] = aggregates['total_pressures'] / aggregates['total_matches']
            aggregates['avg_defensive_actions_per_match'] = aggregates['total_defensive_actions'] / aggregates['total_matches']
            aggregates['avg_progressive_actions_per_match'] = aggregates['total_progressive_actions'] / aggregates['total_matches']
        
        # Calculate success rates
        if aggregates['total_passes'] > 0:
            aggregates['career_pass_accuracy'] = aggregates['total_successful_passes'] / aggregates['total_passes']
        if aggregates['total_dribbles'] > 0:
            aggregates['career_dribble_success_rate'] = aggregates['total_successful_dribbles'] / aggregates['total_dribbles']
        if aggregates['total_crosses'] > 0:
            aggregates['career_cross_accuracy'] = aggregates['total_successful_crosses'] / aggregates['total_crosses']
        if aggregates['total_aerial_duels'] > 0:
            aggregates['career_aerial_duel_success_rate'] = aggregates['total_aerial_duels_won'] / aggregates['total_aerial_duels']
        
        # Calculate most common playing style
        if aggregates['playing_styles']:
            from collections import Counter
            style_counts = Counter(aggregates['playing_styles'])
            aggregates['most_common_playing_style'] = style_counts.most_common(1)[0][0]
        
        # Calculate average performance score
        if aggregates['performance_scores']:
            aggregates['avg_performance_score'] = sum(aggregates['performance_scores']) / len(aggregates['performance_scores'])
        
        return aggregates
    
    def _calculate_competition_breakdown(self, match_analyses: List[Dict]) -> Dict:
        """
        Calculate statistics breakdown by competition
        """
        if not match_analyses:
            return {}
        
        competition_stats = {}
        
        for match_analysis in match_analyses:
            comp_id = match_analysis.get('competition_id')
            if comp_id not in competition_stats:
                competition_stats[comp_id] = {
                    'matches': 0,
                    'total_minutes': 0,
                    'total_passes': 0,
                    'total_successful_passes': 0,
                    'total_shots': 0,
                    'total_goals': 0,
                    'total_assists': 0,
                    'total_key_passes': 0,
                    'total_dribbles': 0,
                    'total_successful_dribbles': 0,
                    'total_pressures': 0,
                    'total_defensive_actions': 0,
                    'performance_scores': []
                }
            
            basic_stats = match_analysis.get('basic_stats', {})
            advanced_stats = match_analysis.get('advanced_stats', {})
            summary_insights = match_analysis.get('summary_insights', {})
            
            competition_stats[comp_id]['matches'] += 1
            competition_stats[comp_id]['total_minutes'] += basic_stats.get('total_playing_time_seconds', 0) / 60
            competition_stats[comp_id]['total_passes'] += basic_stats.get('total_passes', 0)
            competition_stats[comp_id]['total_successful_passes'] += basic_stats.get('successful_passes', 0)
            competition_stats[comp_id]['total_shots'] += basic_stats.get('total_shots', 0)
            competition_stats[comp_id]['total_goals'] += basic_stats.get('shots_on_target', 0)
            competition_stats[comp_id]['total_assists'] += advanced_stats.get('assists', 0)
            competition_stats[comp_id]['total_key_passes'] += advanced_stats.get('key_passes', 0)
            competition_stats[comp_id]['total_dribbles'] += advanced_stats.get('total_dribbles', 0)
            competition_stats[comp_id]['total_successful_dribbles'] += advanced_stats.get('successful_dribbles', 0)
            competition_stats[comp_id]['total_pressures'] += basic_stats.get('total_pressures', 0)
            competition_stats[comp_id]['total_defensive_actions'] += advanced_stats.get('defensive_actions', 0)
            
            if 'performance_score' in summary_insights:
                competition_stats[comp_id]['performance_scores'].append(summary_insights['performance_score'])
        
        # Calculate averages and rates for each competition
        for comp_id, stats in competition_stats.items():
            if stats['matches'] > 0:
                stats['avg_minutes_per_match'] = stats['total_minutes'] / stats['matches']
                stats['avg_passes_per_match'] = stats['total_passes'] / stats['matches']
                stats['avg_shots_per_match'] = stats['total_shots'] / stats['matches']
                stats['avg_assists_per_match'] = stats['total_assists'] / stats['matches']
                stats['avg_key_passes_per_match'] = stats['total_key_passes'] / stats['matches']
                stats['avg_dribbles_per_match'] = stats['total_dribbles'] / stats['matches']
                stats['avg_pressures_per_match'] = stats['total_pressures'] / stats['matches']
                stats['avg_defensive_actions_per_match'] = stats['total_defensive_actions'] / stats['matches']
                
                if stats['total_passes'] > 0:
                    stats['pass_accuracy'] = stats['total_successful_passes'] / stats['total_passes']
                if stats['total_dribbles'] > 0:
                    stats['dribble_success_rate'] = stats['total_successful_dribbles'] / stats['total_dribbles']
                if stats['performance_scores']:
                    stats['avg_performance_score'] = sum(stats['performance_scores']) / len(stats['performance_scores'])
        
        return competition_stats

class WebScraper:
    """Scrapes additional player data from various sources"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.url_base = "https://transfermarket.p.rapidapi.com/"
        self.api_key = os.getenv('RAPID_API_KEY')
        
        if not self.api_key:
            logger.warning("RAPID_API_KEY not found in environment variables")
    
    def get_transfermarkt_data(self, player_name: str) -> Dict:
        """
        Get comprehensive player data from Transfermarkt API
        
        Args:
            player_name: Name of the player to search for
            
        Returns:
            Dictionary containing player profile, transfer history, and performance data
        """
        try:
            if not self.api_key:
                return {
                    'error': 'RAPID_API_KEY not configured',
                    'player_name': player_name
                }
            
            logger.info(f"Fetching Transfermarkt data for {player_name}")
            
            # Step 1: Search for player
            player_id = self._search_player(player_name)
            if not player_id:
                return {
                    'error': f'Player {player_name} not found on Transfermarkt',
                    'player_name': player_name
                }
            
            # Step 2: Get player profile
            profile_data = self._get_player_profile(player_id)
            
            # Step 3: Get transfer history
            transfer_data = self._get_transfer_history(player_id)
            
            # Step 4: Get performance data
            performance_data = self._get_performance_data(player_id, profile_data.get('seasons_played', []))
            
            return {
                'player_name': player_name,
                'player_id': player_id,
                'profile': profile_data,
                'transfer_history': transfer_data,
                'performance_data': performance_data,
                'clubs_played_for': self._extract_clubs_from_transfers(transfer_data)
            }
            
        except Exception as e:
            logger.error(f"Error fetching Transfermarkt data for {player_name}: {e}")
            return {
                'error': str(e),
                'player_name': player_name
            }
    
    def search_players_with_selection(self, player_name: str) -> List[PlayerSearchResult]:
        """
        Search for players and return multiple results for user selection
        
        Args:
            player_name: Name of the player to search for
            
        Returns:
            List of PlayerSearchResult objects
        """
        try:
            if not self.api_key:
                logger.warning("RAPID_API_KEY not configured")
                return []
            
            logger.info(f"Searching for players with name: {player_name}")
            return self._search_players(player_name)
            
        except Exception as e:
            logger.error(f"Error searching for players with name {player_name}: {e}")
            return []
    
    def get_transfermarkt_data_by_id(self, player_id: str) -> Dict:
        """
        Get comprehensive player data from Transfermarkt API using player ID
        
        Args:
            player_id: Transfermarkt player ID
            
        Returns:
            Dictionary containing player profile, transfer history, and performance data
        """
        try:
            if not self.api_key:
                return {
                    'error': 'RAPID_API_KEY not configured',
                    'player_id': player_id
                }
            
            logger.info(f"Fetching Transfermarkt data for player ID: {player_id}")
            
            # Step 1: Get player profile
            profile_data = self._get_player_profile(player_id)
            
            # Step 2: Get transfer history
            transfer_data = self._get_transfer_history(player_id)
            
            # Step 3: Get performance data
            performance_data = self._get_performance_data(player_id, profile_data.get('seasons_played', []))
            
            return {
                'player_id': player_id,
                'profile': profile_data,
                'transfer_history': transfer_data,
                'performance_data': performance_data,
                'clubs_played_for': self._extract_clubs_from_transfers(transfer_data)
            }
            
        except Exception as e:
            logger.error(f"Error fetching Transfermarkt data for player ID {player_id}: {e}")
            return {
                'error': str(e),
                'player_id': player_id
            }
    
    def _search_player(self, player_name: str) -> Optional[str]:
        """Search for player and return player ID (legacy method for single result)"""
        results = self._search_players(player_name)
        if results:
            return results[0].player_id
        return None
    
    def _search_players(self, player_name: str) -> List[PlayerSearchResult]:
        """Search for players and return multiple results"""
        try:
            url = f"{self.url_base}/search"
            querystring = {"query": player_name, "domain": "en"}
            
            headers = {
                "x-rapidapi-key": self.api_key,
                "x-rapidapi-host": "transfermarket.p.rapidapi.com"
            }
            
            response = requests.get(url, headers=headers, params=querystring)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            if isinstance(data, dict) and 'players' in data and isinstance(data['players'], list):
                for player in data['players']:
                    # Calculate confidence score based on name similarity
                    name_similarity = self._calculate_name_similarity(player_name, str(player.get('name', '')))
                    print(player)
                    print(type(player), "################")
                    result = PlayerSearchResult(
                        player_id=str(player.get('id', '')),
                        player_name=str(player.get('playerName', '')),
                        club=str(player.get('club', 'Unknown')),
                        confidence_score=name_similarity
                    )
                    results.append(result)
                
                # Sort by confidence score
                results.sort(key=lambda x: x.confidence_score, reverse=True)
            
                logger.info(f"Found {len(results)} players matching '{player_name}'")
            else:
                logger.warning(f"Unexpected API response format for player search: {data}")
        
            return results
                
        except Exception as e:
            logger.error(f"Error searching for player {player_name}: {e}", exc_info=True)
            return []
    
    def _calculate_name_similarity(self, search_name: str, player_name: str) -> float:
        """Calculate similarity between search name and player name"""
        search_lower = search_name.lower().strip()
        player_lower = player_name.lower().strip()
        
        # Exact match
        if search_lower == player_lower:
            return 1.0
        
        # Contains match
        if search_lower in player_lower or player_lower in search_lower:
            return 0.8
        
        # Word match
        search_words = set(search_lower.split())
        player_words = set(player_lower.split())
        
        if search_words & player_words:  # Intersection
            return 0.6
        
        return 0.0
    
    def _get_player_profile(self, player_id: str) -> Dict:
        """Get player profile information"""
        try:
            url = f"{self.url_base}/players/get-profile"
            querystring = {"id": player_id, "domain": "en"}
            
            headers = {
                "x-rapidapi-key": self.api_key,
                "x-rapidapi-host": "transfermarket.p.rapidapi.com"
            }
            
            response = self.session.get(url, headers=headers, params=querystring)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract seasons played
            seasons_played = [season["key"] for season in data.get("performanceSeasons", [])]
            
            return {
                'seasons_played': seasons_played,
                'raw_profile': data
            }
            
        except Exception as e:
            logger.error(f"Error getting player profile for ID {player_id}: {e}")
            return {'seasons_played': [], 'raw_profile': {}}
    
    def _get_transfer_history(self, player_id: str) -> Dict:
        """Get player transfer history"""
        try:
            url = f"{self.url_base}/players/get-transfer-history"
            querystring = {"id": player_id, "domain": "en"}
            
            headers = {
                "x-rapidapi-key": self.api_key,
                "x-rapidapi-host": "transfermarket.p.rapidapi.com"
            }
            
            response = self.session.get(url, headers=headers, params=querystring)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract transfer data
            transfers = []
            for transfer in data.get("transferHistory", []):
                transfers.append({
                    'old_club': transfer.get("oldClubName"),
                    'new_club': transfer.get("newClubName"),
                    'old_club_id': transfer.get("oldClubID"),
                    'new_club_id': transfer.get("newClubID"),
                    'date': transfer.get("date"),
                    'fee': transfer.get("fee"),
                    'type': transfer.get("type")
                })
            
            return {
                'transfers': transfers,
                'total_transfers': len(transfers),
                'raw_data': data
            }
            
        except Exception as e:
            logger.error(f"Error getting transfer history for ID {player_id}: {e}")
            return {'transfers': [], 'total_transfers': 0, 'raw_data': {}}
    
    def _get_performance_data(self, player_id: str, seasons_played: List[str]) -> Dict:
        """Get player performance data for all seasons"""
        try:
            url = f"{self.url_base}/players/get-performance-summary"
            
            all_performance = []
            
            for season in seasons_played:
                try:
                    querystring = {"id": player_id, "domain": "en", "seasonID": season}
                    
                    headers = {
                        "x-rapidapi-key": self.api_key,
                        "x-rapidapi-host": "transfermarket.p.rapidapi.com"
                    }
                    
                    response = self.session.get(url, headers=headers, params=querystring)
                    response.raise_for_status()
                    
                    data = response.json()
                    
                    # Extract performance data for each competition
                    for comp_performance in data.get("competitionPerformanceSummery", []):
                        performance_record = {
                            'season': season,
                            'competition': comp_performance.get("competition", {}).get("name"),
                            'competition_id': comp_performance.get("competition", {}).get("id"),
                            'club_id': comp_performance.get("clubs", [{}])[0].get("id"),
                            'club_name': comp_performance.get("clubs", [{}])[0].get("name"),
                            'is_national_team': comp_performance.get("clubs", [{}])[0].get("nationalTeam") == "x",
                            'matches': comp_performance.get("performance", {}).get("matches", 0),
                            'goals': comp_performance.get("performance", {}).get("goals", 0),
                            'assists': comp_performance.get("performance", {}).get("assists", 0),
                            'minutes_played': comp_performance.get("performance", {}).get("minutesPlayed", 0),
                            'minutes_per_goal': comp_performance.get("performance", {}).get("minutesPerGoal", 0),
                            'penalty_goals': comp_performance.get("performance", {}).get("penaltyGoals", 0),
                            'yellow_cards': comp_performance.get("performance", {}).get("yellowCards", 0),
                            'yellow_red_cards': comp_performance.get("performance", {}).get("yellowRedCards", 0),
                            'red_cards': comp_performance.get("performance", {}).get("redCards", 0),
                            'to_nil': comp_performance.get("performance", {}).get("toNil", 0),
                            'conceded_goals': comp_performance.get("performance", {}).get("concededGoals", 0)
                        }
                        all_performance.append(performance_record)
                    
                    # Rate limiting
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.warning(f"Error getting performance data for season {season}: {e}")
                    continue
            
            return {
                'performance_records': all_performance,
                'total_seasons': len(seasons_played),
                'total_competitions': len(all_performance)
            }
            
        except Exception as e:
            logger.error(f"Error getting performance data for ID {player_id}: {e}")
            return {'performance_records': [], 'total_seasons': 0, 'total_competitions': 0}
    
    def _extract_clubs_from_transfers(self, transfer_data: Dict) -> List[Dict]:
        """Extract unique clubs from transfer history"""
        try:
            clubs = set()
            clubs_data = []
            
            for transfer in transfer_data.get('transfers', []):
                if transfer.get('old_club_id'):
                    clubs.add(transfer['old_club_id'])
                if transfer.get('new_club_id'):
                    clubs.add(transfer['new_club_id'])
            
            # Convert to list of dictionaries
            for club_id in clubs:
                clubs_data.append({
                    'club_id': club_id,
                    'club_name': self._get_club_name_from_transfers(club_id, transfer_data.get('transfers', []))
                })
            
            return clubs_data
            
        except Exception as e:
            logger.error(f"Error extracting clubs from transfers: {e}")
            return []
    
    def _get_club_name_from_transfers(self, club_id: str, transfers: List[Dict]) -> str:
        """Get club name from transfer history"""
        for transfer in transfers:
            if transfer.get('old_club_id') == club_id:
                return transfer.get('old_club', 'Unknown')
            if transfer.get('new_club_id') == club_id:
                return transfer.get('new_club', 'Unknown')
        return 'Unknown'
    
    def get_football_api_data(self, player_name: str, api_key: Optional[str] = None) -> Dict:
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
    
    def get_player_career_summary(self, player_name: str) -> Dict:
        """
        Get a summary of player career from Transfermarkt data
        
        Args:
            player_name: Name of the player to analyze
            
        Returns:
            Dictionary with career summary statistics
        """
        try:
            transfermarkt_data = self.get_transfermarkt_data(player_name)
            
            if 'error' in transfermarkt_data:
                return transfermarkt_data
            
            # Calculate career summary
            performance_data = transfermarkt_data.get('performance_data', {})
            transfer_data = transfermarkt_data.get('transfer_history', {})
            
            career_summary = {
                'player_name': player_name,
                'total_seasons': performance_data.get('total_seasons', 0),
                'total_competitions': performance_data.get('total_competitions', 0),
                'total_transfers': transfer_data.get('total_transfers', 0),
                'total_clubs': len(transfermarkt_data.get('clubs_played_for', [])),
                'career_stats': self._calculate_career_stats(performance_data.get('performance_records', [])),
                'transfer_summary': self._calculate_transfer_summary(transfer_data.get('transfers', []))
            }
            
            return career_summary
            
        except Exception as e:
            logger.error(f"Error calculating career summary for {player_name}: {e}")
            return {
                'error': str(e),
                'player_name': player_name
            }
    
    def _calculate_career_stats(self, performance_records: List[Dict]) -> Dict:
        """Calculate career statistics from performance records"""
        try:
            total_matches = sum(record.get('matches', 0) for record in performance_records)
            total_goals = sum(record.get('goals', 0) for record in performance_records)
            total_assists = sum(record.get('assists', 0) for record in performance_records)
            total_minutes = sum(record.get('minutes_played', 0) for record in performance_records)
            total_yellow_cards = sum(record.get('yellow_cards', 0) for record in performance_records)
            total_red_cards = sum(record.get('red_cards', 0) for record in performance_records)
            
            return {
                'total_matches': total_matches,
                'total_goals': total_goals,
                'total_assists': total_assists,
                'total_minutes': total_minutes,
                'total_yellow_cards': total_yellow_cards,
                'total_red_cards': total_red_cards,
                'avg_goals_per_match': total_goals / total_matches if total_matches > 0 else 0,
                'avg_assists_per_match': total_assists / total_matches if total_matches > 0 else 0,
                'avg_minutes_per_match': total_minutes / total_matches if total_matches > 0 else 0,
                'minutes_per_goal': total_minutes / total_goals if total_goals > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating career stats: {e}")
            return {}
    
    def _calculate_transfer_summary(self, transfers: List[Dict]) -> Dict:
        """Calculate transfer history summary"""
        try:
            if not transfers:
                return {}
            
            # Calculate transfer fees
            total_fees = 0
            free_transfers = 0
            loan_transfers = 0
            
            for transfer in transfers:
                fee = transfer.get('fee', '')
                if fee and fee != 'free transfer':
                    # Try to extract numeric value from fee string
                    try:
                        if '€' in fee:
                            fee_value = fee.replace('€', '').replace(',', '').replace('m', '000000').replace('k', '000')
                            total_fees += float(fee_value)
                    except:
                        pass
                elif fee == 'free transfer':
                    free_transfers += 1
                elif 'loan' in fee.lower():
                    loan_transfers += 1
            
            return {
                'total_transfers': len(transfers),
                'free_transfers': free_transfers,
                'loan_transfers': loan_transfers,
                'estimated_total_fees': total_fees,
                'first_transfer_date': min(transfer.get('date', '9999') for transfer in transfers),
                'last_transfer_date': max(transfer.get('date', '0000') for transfer in transfers)
            }
            
        except Exception as e:
            logger.error(f"Error calculating transfer summary: {e}")
            return {}

class DataAggregator:
    """Aggregates data from multiple sources"""
    
    def __init__(self):
        self.statsbomb = StatsBombDataCollector()
        self.web_scraper = WebScraper()
    
    def get_complete_player_profile(self, player_name: str, match_id: Optional[int] = None) -> Dict:
        """Get complete player profile from all available sources"""
        
        profile = {
            'player_name': player_name,
            'statsbomb_data': {},
            'web_data': {},
            'match_specific_data': {}
        }
        
        # Get StatsBomb career data
        career_data = self.statsbomb.get_player_career_data(player_name)
        if career_data and 'error' not in career_data:
            profile['statsbomb_data']['career'] = career_data
            
            # Calculate aggregated metrics if we have raw events
            if 'raw_events' in career_data and not career_data['raw_events'].empty:
                profile['statsbomb_data']['aggregated_metrics'] = self._calculate_aggregated_metrics(career_data['raw_events'])
        
        # Get match-specific data if provided
        if match_id:
            match_data = self.statsbomb.get_player_stats(match_id, player_name)
            if match_data and 'error' not in match_data:
                profile['match_specific_data'] = match_data
        
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

 