"""
Machine Learning Module for Soccer Agent
Handles player performance prediction and analysis
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class PlayerPerformancePredictor:
    """Predicts player performance and potential"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_columns = []
        
    def prepare_features(self, player_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for machine learning models"""
        try:
            features = pd.DataFrame()
            
            # Basic player metrics
            features['total_passes'] = len(player_data[player_data['type_name'] == 'Pass'])
            features['completed_passes'] = len(player_data[
                (player_data['type_name'] == 'Pass') & 
                (player_data['outcome_name'].isna())
            ])
            features['pass_accuracy'] = features['completed_passes'] / features['total_passes'] if features['total_passes'] > 0 else 0
            
            # Shot metrics
            features['total_shots'] = len(player_data[player_data['type_name'] == 'Shot'])
            features['goals'] = len(player_data[
                (player_data['type_name'] == 'Shot') & 
                (player_data['outcome_name'] == 'Goal')
            ])
            features['shot_accuracy'] = features['goals'] / features['total_shots'] if features['total_shots'] > 0 else 0
            
            # Dribble metrics
            features['total_dribbles'] = len(player_data[player_data['type_name'] == 'Dribble'])
            features['successful_dribbles'] = len(player_data[
                (player_data['type_name'] == 'Dribble') & 
                (player_data['outcome_name'] == 'Complete')
            ])
            features['dribble_success_rate'] = features['successful_dribbles'] / features['total_dribbles'] if features['total_dribbles'] > 0 else 0
            
            # Defensive metrics
            features['tackles'] = len(player_data[player_data['type_name'] == 'Dribbled Past'])
            features['interceptions'] = len(player_data[player_data['type_name'] == 'Interception'])
            
            # Time-based features
            features['total_minutes'] = len(player_data) * 0.1  # Rough estimate
            features['actions_per_minute'] = len(player_data) / features['total_minutes'] if features['total_minutes'] > 0 else 0
            
            # Position-based features (if available)
            if 'position_name' in player_data.columns:
                position_encoder = LabelEncoder()
                features['position_encoded'] = position_encoder.fit_transform(
                    player_data['position_name'].fillna('Unknown')
                )
                self.label_encoders['position'] = position_encoder
            
            # Competition level (if available)
            if 'competition_id' in player_data.columns:
                features['competition_level'] = player_data['competition_id'].mean()
            
            return features
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return pd.DataFrame()
    
    def train_performance_model(self, training_data: List[Dict]) -> Dict:
        """Train a model to predict player performance"""
        try:
            # Prepare training data
            X_list = []
            y_list = []
            
            for player_record in training_data:
                player_df = pd.DataFrame(player_record['statsbomb_data']['career'])
                if not player_df.empty:
                    features = self.prepare_features(player_df)
                    if not features.empty:
                        X_list.append(features.iloc[0].to_dict())
                        
                        # Target variable: overall performance score
                        performance_score = self._calculate_performance_score(player_df)
                        y_list.append(performance_score)
            
            if not X_list or not y_list:
                return {'error': 'Insufficient training data'}
            
            X = pd.DataFrame(X_list)
            y = np.array(y_list)
            
            # Store feature columns
            self.feature_columns = X.columns.tolist()
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            self.scalers['performance'] = scaler
            
            # Train multiple models
            models = {
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'linear_regression': LinearRegression()
            }
            
            results = {}
            
            for name, model in models.items():
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Predictions
                y_pred = model.predict(X_test_scaled)
                
                # Metrics
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
                
                results[name] = {
                    'model': model,
                    'mse': mse,
                    'r2': r2,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }
            
            # Select best model
            best_model_name = max(results.keys(), key=lambda k: results[k]['r2'])
            self.models['performance'] = results[best_model_name]['model']
            
            return {
                'best_model': best_model_name,
                'results': results,
                'feature_importance': self._get_feature_importance(results[best_model_name]['model'])
            }
            
        except Exception as e:
            logger.error(f"Error training performance model: {e}")
            return {'error': str(e)}
    
    def predict_player_potential(self, player_data: Dict, years_ahead: int = 3) -> Dict:
        """Predict player potential for the next few years"""
        try:
            if 'statsbomb_data' not in player_data or 'career' not in player_data['statsbomb_data']:
                return {'error': 'No career data available'}
            
            player_df = pd.DataFrame(player_data['statsbomb_data']['career'])
            if player_df.empty:
                return {'error': 'Empty career data'}
            
            # Prepare features
            features = self.prepare_features(player_df)
            if features.empty:
                return {'error': 'Could not prepare features'}
            
            # Ensure all required features are present
            for col in self.feature_columns:
                if col not in features.columns:
                    features[col] = 0
            
            features = features[self.feature_columns]
            
            # Scale features
            if 'performance' in self.scalers:
                features_scaled = self.scalers['performance'].transform(features)
            else:
                return {'error': 'Model not trained yet'}
            
            # Make prediction
            if 'performance' in self.models:
                current_performance = self.models['performance'].predict(features_scaled)[0]
            else:
                return {'error': 'Model not trained yet'}
            
            # Predict future performance based on age and current performance
            predictions = self._predict_future_performance(
                current_performance, 
                player_data.get('age', 25), 
                years_ahead
            )
            
            return {
                'current_performance': current_performance,
                'predictions': predictions,
                'confidence': self._calculate_prediction_confidence(features)
            }
            
        except Exception as e:
            logger.error(f"Error predicting player potential: {e}")
            return {'error': str(e)}
    
    def _calculate_performance_score(self, player_data: pd.DataFrame) -> float:
        """Calculate overall performance score for a player"""
        try:
            score = 0.0
            
            # Pass contribution (30%)
            pass_events = player_data[player_data['type_name'] == 'Pass']
            if not pass_events.empty:
                pass_accuracy = len(pass_events[pass_events['outcome_name'].isna()]) / len(pass_events)
                score += pass_accuracy * 0.3
            
            # Goal contribution (40%)
            shot_events = player_data[player_data['type_name'] == 'Shot']
            if not shot_events.empty:
                goal_rate = len(shot_events[shot_events['outcome_name'] == 'Goal']) / len(shot_events)
                score += goal_rate * 0.4
            
            # Dribble contribution (20%)
            dribble_events = player_data[player_data['type_name'] == 'Dribble']
            if not dribble_events.empty:
                dribble_success = len(dribble_events[dribble_events['outcome_name'] == 'Complete']) / len(dribble_events)
                score += dribble_success * 0.2
            
            # Defensive contribution (10%)
            defensive_events = player_data[player_data['type_name'].isin(['Tackle', 'Interception', 'Clearance'])]
            if not defensive_events.empty:
                defensive_score = len(defensive_events) / len(player_data)
                score += defensive_score * 0.1
            
            return score
            
        except Exception as e:
            logger.error(f"Error calculating performance score: {e}")
            return 0.0
    
    def _predict_future_performance(self, current_performance: float, age: int, years_ahead: int) -> Dict:
        """Predict future performance based on age curve"""
        predictions = {}
        
        # Simple age-based performance curve
        # Peak performance typically around 27-28 years old
        peak_age = 27.5
        
        for year in range(1, years_ahead + 1):
            future_age = age + year
            
            # Performance decline after peak age
            if future_age <= peak_age:
                # Still improving or at peak
                performance_factor = 1.0 + (peak_age - future_age) * 0.02
            else:
                # Declining after peak
                performance_factor = 1.0 - (future_age - peak_age) * 0.03
            
            # Apply some randomness
            performance_factor *= np.random.normal(1.0, 0.05)
            performance_factor = max(0.1, min(1.5, performance_factor))  # Clamp between 0.1 and 1.5
            
            predicted_performance = current_performance * performance_factor
            predictions[f'year_{year}'] = {
                'age': future_age,
                'predicted_performance': predicted_performance,
                'performance_factor': performance_factor
            }
        
        return predictions
    
    def _get_feature_importance(self, model) -> Dict:
        """Get feature importance from the model"""
        try:
            if hasattr(model, 'feature_importances_'):
                importance_dict = dict(zip(self.feature_columns, model.feature_importances_))
                return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            else:
                return {}
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return {}
    
    def _calculate_prediction_confidence(self, features: pd.DataFrame) -> float:
        """Calculate confidence in the prediction based on data quality"""
        try:
            # Simple confidence calculation based on data completeness
            non_zero_features = (features != 0).sum().sum()
            total_features = features.size
            
            confidence = non_zero_features / total_features if total_features > 0 else 0
            return min(1.0, confidence * 1.2)  # Boost confidence slightly
            
        except Exception as e:
            logger.error(f"Error calculating prediction confidence: {e}")
            return 0.5
    
    def save_models(self, filepath: str):
        """Save trained models to disk"""
        try:
            model_data = {
                'models': self.models,
                'scalers': self.scalers,
                'label_encoders': self.label_encoders,
                'feature_columns': self.feature_columns
            }
            joblib.dump(model_data, filepath)
            logger.info(f"Models saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def load_models(self, filepath: str):
        """Load trained models from disk"""
        try:
            model_data = joblib.load(filepath)
            self.models = model_data['models']
            self.scalers = model_data['scalers']
            self.label_encoders = model_data['label_encoders']
            self.feature_columns = model_data['feature_columns']
            logger.info(f"Models loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading models: {e}") 