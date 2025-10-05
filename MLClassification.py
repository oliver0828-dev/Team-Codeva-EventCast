import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend - MUST be before pyplot import
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
from scipy import stats
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

CITIES = {
    'Seoul': 'power_met_south_korea_seoul_2019-01-01_2024-12-31.csv',
    'Delhi': 'power_met_india_delhi_2019-01-01_2024-12-31.csv',
    'Tokyo': 'power_met_japan_tokyo_2019-01-01_2024-12-31.csv',
    'Phoenix': 'power_met_united_states_phoenix_2019-01-01_2024-12-31.csv',
    'Singapore': 'power_met_singapore_singapore_2019-01-01_2024-12-31.csv'
}

class ProbabilisticWeatherPredictor:
    """
    Advanced ML system for predicting probability of extreme weather events
    Uses ensemble methods with temporal features and historical patterns
    """
    
    def __init__(self, filepath, city_name):
        self.city = city_name
        self.df = pd.read_csv(filepath)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.models = {}
        self.scalers = {}
        self.thresholds = {}
        self.feature_importance = {}
        
    def engineer_features(self):
        """Create sophisticated temporal and cyclical features"""
        # Temporal features
        self.df['year'] = self.df['date'].dt.year
        self.df['month'] = self.df['date'].dt.month
        self.df['day'] = self.df['date'].dt.day
        self.df['dayofyear'] = self.df['date'].dt.dayofyear
        self.df['week'] = self.df['date'].dt.isocalendar().week
        
        # Cyclical encoding (captures seasonality better than raw numbers)
        self.df['month_sin'] = np.sin(2 * np.pi * self.df['month'] / 12)
        self.df['month_cos'] = np.cos(2 * np.pi * self.df['month'] / 12)
        self.df['day_sin'] = np.sin(2 * np.pi * self.df['dayofyear'] / 365)
        self.df['day_cos'] = np.cos(2 * np.pi * self.df['dayofyear'] / 365)
        
        # Rolling statistics (capture recent trends)
        for window in [7, 14, 30]:
            self.df[f't2m_roll_mean_{window}'] = self.df['t2m_C_mean'].rolling(window=window, min_periods=1).mean()
            self.df[f't2m_roll_std_{window}'] = self.df['t2m_C_mean'].rolling(window=window, min_periods=1).std()
            self.df[f'rain_roll_sum_{window}'] = self.df['rain_mm_day'].rolling(window=window, min_periods=1).sum()
            self.df[f'rain_roll_max_{window}'] = self.df['rain_mm_day'].rolling(window=window, min_periods=1).max()
        
        # Lag features (yesterday's weather affects today)
        for lag in [1, 3, 7]:
            self.df[f't2m_lag_{lag}'] = self.df['t2m_C_mean'].shift(lag)
            self.df[f'rain_lag_{lag}'] = self.df['rain_mm_day'].shift(lag)
        
        # Historical averages for same day of year
        self.df['historical_temp_mean'] = self.df.groupby('dayofyear')['t2m_C_mean'].transform('mean')
        self.df['historical_temp_std'] = self.df.groupby('dayofyear')['t2m_C_mean'].transform('std')
        self.df['temp_anomaly'] = (self.df['t2m_C_mean'] - self.df['historical_temp_mean']) / (self.df['historical_temp_std'] + 0.01)
        
        # Interaction features
        self.df['temp_humidity_interaction'] = self.df['t2m_C_mean'] * self.df['rh2m_pct_mean']
        self.df['wind_rain_interaction'] = self.df['wind10m_ms_mean'] * self.df['rain_mm_day']
        
        # Drop NaN from rolling/lag features
        self.df = self.df.dropna().reset_index(drop=True)
        
        return self.df
    
    def define_extreme_events(self):
        """Define extreme weather event labels using multiple methods"""
        # Method 1: Percentile-based (adaptive to location)
        self.thresholds['temp_hot_p95'] = self.df['t2m_C_mean'].quantile(0.95)
        self.thresholds['temp_cold_p05'] = self.df['t2m_C_mean'].quantile(0.05)
        self.thresholds['rain_heavy_p90'] = self.df['rain_mm_day'].quantile(0.90)
        self.thresholds['wind_strong_p90'] = self.df['wind10m_ms_mean'].quantile(0.90)
        
        # Method 2: Heat index (combines temp + humidity)
        self.df['heat_index'] = self._calculate_heat_index(
            self.df['t2m_C_mean'], 
            self.df['rh2m_pct_mean']
        )
        self.thresholds['heat_index_p95'] = self.df['heat_index'].quantile(0.95)
        
        # Create binary labels for extreme events
        self.df['extreme_hot'] = (self.df['t2m_C_mean'] >= self.thresholds['temp_hot_p95']).astype(int)
        self.df['extreme_cold'] = (self.df['t2m_C_mean'] <= self.thresholds['temp_cold_p05']).astype(int)
        self.df['extreme_rain'] = (self.df['rain_mm_day'] >= self.thresholds['rain_heavy_p90']).astype(int)
        self.df['extreme_wind'] = (self.df['wind10m_ms_mean'] >= self.thresholds['wind_strong_p90']).astype(int)
        self.df['extreme_discomfort'] = (self.df['heat_index'] >= self.thresholds['heat_index_p95']).astype(int)
        
        print(f"\n{self.city} Extreme Event Thresholds:")
        print(f"  Very Hot: >{self.thresholds['temp_hot_p95']:.1f}°C ({self._c_to_f(self.thresholds['temp_hot_p95']):.1f}°F)")
        print(f"  Very Cold: <{self.thresholds['temp_cold_p05']:.1f}°C ({self._c_to_f(self.thresholds['temp_cold_p05']):.1f}°F)")
        print(f"  Heavy Rain: >{self.thresholds['rain_heavy_p90']:.1f}mm/day")
        print(f"  Strong Wind: >{self.thresholds['wind_strong_p90']:.1f}m/s")
        print(f"  Uncomfortable (Heat Index): >{self.thresholds['heat_index_p95']:.1f}°C")
        
    def _c_to_f(self, celsius):
        return celsius * 9/5 + 32
    
    def _calculate_heat_index(self, temp_c, rh):
        """Calculate heat index (apparent temperature)"""
        temp_f = temp_c * 9/5 + 32
        hi = 0.5 * (temp_f + 61.0 + ((temp_f-68.0)*1.2) + (rh*0.094))
        
        # Refined calculation for high temps
        mask = hi >= 80
        if mask.any():
            hi_adj = (-42.379 + 2.04901523*temp_f + 10.14333127*rh - 0.22475541*temp_f*rh - 
                      0.00683783*temp_f*temp_f - 0.05481717*rh*rh + 0.00122874*temp_f*temp_f*rh + 
                      0.00085282*temp_f*rh*rh - 0.00000199*temp_f*temp_f*rh*rh)
            hi = np.where(mask, hi_adj, hi)
        
        return (hi - 32) * 5/9  # Convert back to Celsius
    
    def prepare_features(self, target_col):
        """Prepare feature matrix and target variable"""
        feature_cols = [
            'month_sin', 'month_cos', 'day_sin', 'day_cos',
            't2m_roll_mean_7', 't2m_roll_mean_14', 't2m_roll_mean_30',
            't2m_roll_std_7', 't2m_roll_std_14', 't2m_roll_std_30',
            'rain_roll_sum_7', 'rain_roll_sum_14', 'rain_roll_sum_30',
            'rain_roll_max_7', 'rain_roll_max_14', 'rain_roll_max_30',
            't2m_lag_1', 't2m_lag_3', 't2m_lag_7',
            'rain_lag_1', 'rain_lag_3', 'rain_lag_7',
            'historical_temp_mean', 'historical_temp_std', 'temp_anomaly',
            'rh2m_pct_mean', 'wind10m_ms_mean',
            'temp_humidity_interaction', 'wind_rain_interaction'
        ]
        
        X = self.df[feature_cols].copy()
        y = self.df[target_col].copy()
        
        return X, y, feature_cols
    
    def train_ensemble_model(self, event_type):
        """Train ensemble of models for robust predictions"""
        print(f"\nTraining models for {event_type}...")
        
        X, y, feature_cols = self.prepare_features(event_type)
        
        # Check if enough positive samples exist
        positive_samples = y.sum()
        if positive_samples < 10:
            print(f"  Warning: Only {positive_samples} positive samples. Skipping this event type.")
            return None
        
        # Calculate class weights to handle imbalance
        class_weight = {0: 1.0, 1: len(y) / (2 * positive_samples)}
        print(f"  Positive samples: {positive_samples}/{len(y)} ({100*positive_samples/len(y):.1f}%)")
        print(f"  Using class weight: {class_weight[1]:.2f} for minority class")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers[event_type] = scaler
        
        # Time series cross-validation (respects temporal order)
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Train multiple models with class weights
        models = {
            'rf': RandomForestClassifier(
                n_estimators=200, max_depth=15, 
                min_samples_split=20, 
                class_weight=class_weight,
                random_state=42, n_jobs=-1
            ),
            'gb': GradientBoostingClassifier(
                n_estimators=200, max_depth=5, 
                learning_rate=0.05, 
                random_state=42
            )
        }
        
        # Add XGBoost if available
        if HAS_XGB:
            scale_pos_weight = class_weight[1]
            models['xgb'] = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            )
        
        # Add LightGBM if available
        if HAS_LGBM:
            models['lgbm'] = LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                class_weight=class_weight,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        
        results = {}
        for name, model in models.items():
            # Cross-validation scores
            scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='roc_auc', n_jobs=-1)
            results[name] = {
                'model': model,
                'cv_scores': scores,
                'mean_auc': scores.mean(),
                'std_auc': scores.std()
            }
            print(f"  {name.upper()}: AUC = {scores.mean():.3f} (+/- {scores.std():.3f})")
        
        # Select best model and train on full dataset
        best_model_name = max(results, key=lambda k: results[k]['mean_auc'])
        best_model = results[best_model_name]['model']
        best_model.fit(X_scaled, y)
        
        # Store model and feature importance
        self.models[event_type] = best_model
        self.feature_importance[event_type] = pd.DataFrame({
            'feature': feature_cols,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"  Best model: {best_model_name.upper()}")
        print(f"  Top 5 features:")
        for idx, row in self.feature_importance[event_type].head().iterrows():
            print(f"    {row['feature']}: {row['importance']:.4f}")
        
        return results[best_model_name]
    
    def predict_probability(self, target_date, event_type):
        """Predict probability of extreme event for a specific date"""
        if event_type not in self.models:
            return None
        
        # Find historical data for same day of year
        target_doy = target_date.timetuple().tm_yday
        historical = self.df[self.df['dayofyear'] == target_doy].copy()
        
        if len(historical) == 0:
            return None
        
        # Use most recent year's data as features
        recent = historical.sort_values('date').iloc[-1:].copy()
        
        # Prepare features
        X, _, feature_cols = self.prepare_features(event_type)
        X_pred = recent[feature_cols].values
        
        # Scale and predict
        X_pred_scaled = self.scalers[event_type].transform(X_pred)
        prob = self.models[event_type].predict_proba(X_pred_scaled)[0, 1]
        
        # Calculate confidence interval using historical variance
        historical_rate = historical[event_type].mean()
        n = len(historical)
        se = np.sqrt(historical_rate * (1 - historical_rate) / n)
        ci_lower = max(0, historical_rate - 1.96 * se)
        ci_upper = min(1, historical_rate + 1.96 * se)
        
        return {
            'probability': prob * 100,
            'historical_rate': historical_rate * 100,
            'confidence_interval': (ci_lower * 100, ci_upper * 100),
            'sample_size': n
        }
    
    def get_feeling_level(self, probability):
        """
        Convert probability to human-readable feeling level
        Returns: (level, description, emoji, color)
        """
        if probability >= 60:
            return ("Very Likely", "High certainty this will occur", "🔴", "red")
        elif probability >= 40:
            return ("Likely", "Good chance this will happen", "🟠", "orange")
        elif probability >= 20:
            return ("Possible", "Moderate likelihood", "🟡", "yellow")
        elif probability >= 10:
            return ("Unlikely", "Low probability", "🟢", "lightgreen")
        else:
            return ("Very Unlikely", "Very low chance", "🟢", "green")
    
    def query_date(self, month, day, verbose=True):
        """
        User-friendly query interface for a specific date
        Returns comprehensive prediction with feeling levels
        """
        try:
            target_date = datetime(2024, month, day)
        except ValueError:
            return {"error": "Invalid date"}
        
        result = {
            'city': self.city,
            'date': target_date.strftime('%B %d, %Y'),
            'month': month,
            'day': day,
            'predictions': {}
        }
        
        event_names = {
            'extreme_hot': 'Very Hot',
            'extreme_cold': 'Very Cold',
            'extreme_rain': 'Very Wet',
            'extreme_wind': 'Very Windy',
            'extreme_discomfort': 'Very Uncomfortable'
        }
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Weather Forecast: {self.city} - {target_date.strftime('%B %d, %Y')}")
            print('='*70)
        
        for event_type, display_name in event_names.items():
            if event_type in self.models:
                pred = self.predict_probability(target_date, event_type)
                if pred:
                    feeling_level, description, emoji, color = self.get_feeling_level(pred['probability'])
                    
                    result['predictions'][event_type] = {
                        'condition': display_name,
                        'probability': round(pred['probability'], 1),
                        'feeling_level': feeling_level,
                        'feeling_description': description,
                        'historical_rate': round(pred['historical_rate'], 1),
                        'confidence_interval': {
                            'lower': round(pred['confidence_interval'][0], 1),
                            'upper': round(pred['confidence_interval'][1], 1)
                        },
                        'years_of_data': pred['sample_size']
                    }
                    
                    if verbose:
                        print(f"\n{emoji} {display_name}: {feeling_level}")
                        print(f"   Probability: {pred['probability']:.1f}%")
                        print(f"   {description}")
                        print(f"   Historical average: {pred['historical_rate']:.1f}%")
        
        # Overall recommendation
        if result['predictions']:
            probs = [p['probability'] for p in result['predictions'].values()]
            max_prob = max(probs)
            
            if max_prob >= 60:
                result['overall_rating'] = "Poor conditions expected"
                result['recommendation'] = "High risk day - consider rescheduling"
            elif max_prob >= 40:
                result['overall_rating'] = "Challenging conditions likely"
                result['recommendation'] = "Be prepared for adverse weather"
            elif max_prob >= 20:
                result['overall_rating'] = "Generally good conditions"
                result['recommendation'] = "Moderate risk - have backup plans"
            else:
                result['overall_rating'] = "Excellent conditions expected"
                result['recommendation'] = "Great day for outdoor activities"
            
            if verbose:
                print(f"\n{'='*70}")
                print(f"Overall: {result['overall_rating']}")
                print(f"Recommendation: {result['recommendation']}")
                print('='*70)
        
        return result
        """
        Comprehensive analysis for planning outdoor events
        Returns probabilities and recommendations
        """
        try:
            target_date = datetime(2024, month, day)
        except ValueError:
            return "Invalid date"
        
        results = {
            'city': self.city,
            'date': target_date.strftime('%B %d'),
            'events': {}
        }
        
        print(f"\n{'='*70}")
        print(f"Event Planning Analysis: {self.city} - {target_date.strftime('%B %d')}")
        print('='*70)
        
        for event in events:
            if event not in self.models:
                continue
            pred = self.predict_probability(target_date, event)
            if pred:
                results['events'][event] = pred
                
                event_name = event.replace('extreme_', '').replace('_', ' ').title()
                print(f"\n{event_name}:")
                print(f"  ML Probability: {pred['probability']:.1f}%")
                print(f"  Historical Rate: {pred['historical_rate']:.1f}%")
                print(f"  95% Confidence: [{pred['confidence_interval'][0]:.1f}%, {pred['confidence_interval'][1]:.1f}%]")
                print(f"  Based on {pred['sample_size']} years of data")
                
                # Risk assessment
                if pred['probability'] > 40:
                    risk = "HIGH RISK"
                elif pred['probability'] > 20:
                    risk = "MODERATE RISK"
                else:
                    risk = "LOW RISK"
                print(f"  Assessment: {risk}")
        
        # Overall recommendation
        if results['events']:
            max_prob = max([p['probability'] for p in results['events'].values()])
            if max_prob > 40:
                recommendation = "Consider alternative dates or be well-prepared for adverse conditions"
            elif max_prob > 20:
                recommendation = "Moderate risk - monitor closer to event date and have backup plans"
            else:
                recommendation = "Good conditions likely - low risk of extreme weather"
        else:
            recommendation = "No models available for requested events"
        
        results['recommendation'] = recommendation
        print(f"\n{recommendation}")
        
        return results
    
    def plot_climate_trends(self):
        """Analyze and visualize trends in extreme events over time"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{self.city}: Climate Trends & Extreme Event Analysis', 
                     fontsize=16, fontweight='bold')
        
        # 1. Temperature time series with trend
        ax = axes[0, 0]
        yearly_temp = self.df.groupby('year')['t2m_C_mean'].agg(['mean', 'max', 'min'])
        ax.plot(yearly_temp.index, yearly_temp['mean'], 'o-', label='Annual Mean', linewidth=2)
        ax.fill_between(yearly_temp.index, yearly_temp['min'], yearly_temp['max'], 
                        alpha=0.3, label='Min-Max Range')
        
        # Add trend line
        z = np.polyfit(yearly_temp.index, yearly_temp['mean'], 1)
        p = np.poly1d(z)
        ax.plot(yearly_temp.index, p(yearly_temp.index), "r--", 
               label=f'Trend: {z[0]:.3f}°C/year', linewidth=2)
        
        ax.set_xlabel('Year', fontsize=11)
        ax.set_ylabel('Temperature (°C)', fontsize=11)
        ax.set_title('Temperature Trends Over Time', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Extreme event frequency trends
        ax = axes[0, 1]
        event_trends = self.df.groupby('year')[
            ['extreme_hot', 'extreme_cold', 'extreme_rain', 'extreme_wind']
        ].sum()
        
        for col in event_trends.columns:
            label = col.replace('extreme_', '').title()
            ax.plot(event_trends.index, event_trends[col], 'o-', label=label, linewidth=2)
        
        ax.set_xlabel('Year', fontsize=11)
        ax.set_ylabel('Number of Extreme Days', fontsize=11)
        ax.set_title('Frequency of Extreme Events by Year', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Distribution of temperature anomalies (bell curve)
        ax = axes[1, 0]
        ax.hist(self.df['temp_anomaly'], bins=50, density=True, alpha=0.7, 
               edgecolor='black', label='Observed')
        
        # Fit normal distribution
        mu, sigma = self.df['temp_anomaly'].mean(), self.df['temp_anomaly'].std()
        x = np.linspace(self.df['temp_anomaly'].min(), self.df['temp_anomaly'].max(), 100)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, 
               label=f'Normal(μ={mu:.2f}, σ={sigma:.2f})')
        
        # Mark extreme thresholds
        ax.axvline(x=2, color='orange', linestyle='--', label='2σ threshold')
        ax.axvline(x=-2, color='orange', linestyle='--')
        
        ax.set_xlabel('Temperature Anomaly (σ)', fontsize=11)
        ax.set_ylabel('Probability Density', fontsize=11)
        ax.set_title('Distribution of Temperature Anomalies', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Rainfall distribution
        ax = axes[1, 1]
        rain_nonzero = self.df[self.df['rain_mm_day'] > 0]['rain_mm_day']
        ax.hist(rain_nonzero, bins=50, density=True, alpha=0.7, 
               edgecolor='black', color='skyblue')
        
        # Mark extreme threshold
        ax.axvline(x=self.thresholds['rain_heavy_p90'], color='red', 
                  linestyle='--', linewidth=2, 
                  label=f'Heavy Rain (90th %ile): {self.thresholds["rain_heavy_p90"]:.1f}mm')
        
        ax.set_xlabel('Daily Rainfall (mm)', fontsize=11)
        ax.set_ylabel('Probability Density', fontsize=11)
        ax.set_title('Distribution of Rainfall (Rainy Days Only)', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f'{self.city.lower()}_climate_trends.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n✓ Saved climate trends visualization: {filename}")
        return filename
    
    def plot_seasonal_probability_curves(self, event_types=['extreme_hot', 'extreme_rain']):
        """Plot probability curves across the entire year for planning"""
        fig, axes = plt.subplots(len(event_types), 1, 
                                figsize=(14, 5*len(event_types)))
        if len(event_types) == 1:
            axes = [axes]
        
        fig.suptitle(f'{self.city}: Seasonal Probability Patterns', 
                    fontsize=16, fontweight='bold')
        
        for idx, event_type in enumerate(event_types):
            if event_type not in self.models:
                continue
                
            ax = axes[idx]
            
            # Calculate probabilities for each day of year
            days_of_year = []
            ml_probs = []
            hist_rates = []
            ci_lower = []
            ci_upper = []
            
            for month in range(1, 13):
                for day in [1, 8, 15, 22]:  # Sample 4 days per month
                    try:
                        target_date = datetime(2024, month, day)
                        pred = self.predict_probability(target_date, event_type)
                        if pred:
                            days_of_year.append(target_date.timetuple().tm_yday)
                            ml_probs.append(pred['probability'])
                            hist_rates.append(pred['historical_rate'])
                            ci_lower.append(pred['confidence_interval'][0])
                            ci_upper.append(pred['confidence_interval'][1])
                    except ValueError:
                        continue
            
            # Plot
            ax.plot(days_of_year, ml_probs, 'o-', linewidth=2, 
                   label='ML Prediction', color='darkblue')
            ax.plot(days_of_year, hist_rates, 's--', linewidth=1.5, 
                   label='Historical Rate', color='gray', alpha=0.7)
            ax.fill_between(days_of_year, ci_lower, ci_upper, 
                           alpha=0.2, color='blue', label='95% Confidence Interval')
            
            # Add risk zones
            ax.axhline(y=40, color='red', linestyle=':', alpha=0.5, label='High Risk (>40%)')
            ax.axhline(y=20, color='orange', linestyle=':', alpha=0.5, label='Moderate Risk (>20%)')
            
            # Month labels
            month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            ax.set_xticks(month_starts)
            ax.set_xticklabels(month_names)
            
            event_name = event_type.replace('extreme_', '').replace('_', ' ').title()
            ax.set_ylabel('Probability (%)', fontsize=11)
            ax.set_title(f'{event_name} Risk Throughout the Year', 
                        fontsize=12, fontweight='bold')
            ax.set_ylim([0, 100])
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Month', fontsize=11)
        
        plt.tight_layout()
        filename = f'{self.city.lower()}_seasonal_curves.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved seasonal probability curves: {filename}")
        return filename
    
    def plot_extreme_event_summary(self):
        """Create comprehensive summary visualization of all extreme events"""
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        fig.suptitle(f'{self.city}: Extreme Weather Event Summary', 
                    fontsize=16, fontweight='bold')
        
        # 1. Monthly heatmap of all events
        ax1 = fig.add_subplot(gs[0, :])
        monthly_events = self.df.groupby('month')[
            ['extreme_hot', 'extreme_cold', 'extreme_rain', 'extreme_wind', 'extreme_discomfort']
        ].sum()
        monthly_events.columns = [c.replace('extreme_', '').title() for c in monthly_events.columns]
        
        sns.heatmap(monthly_events.T, annot=True, fmt='d', cmap='YlOrRd', 
                   cbar_kws={'label': 'Number of Days'}, ax=ax1)
        ax1.set_xlabel('Month', fontsize=11)
        ax1.set_ylabel('Event Type', fontsize=11)
        ax1.set_title('Monthly Distribution of Extreme Events', fontsize=12, fontweight='bold')
        ax1.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        
        # 2-5. Individual event type plots
        event_types = [
            ('extreme_hot', 't2m_C_mean', 'Hot Days', 'Reds'),
            ('extreme_cold', 't2m_C_mean', 'Cold Days', 'Blues_r'),
            ('extreme_rain', 'rain_mm_day', 'Heavy Rain', 'Blues'),
            ('extreme_wind', 'wind10m_ms_mean', 'Strong Wind', 'Greens')
        ]
        
        for idx, (event, variable, title, cmap) in enumerate(event_types):
            row = (idx + 3) // 3
            col = (idx + 3) % 3
            ax = fig.add_subplot(gs[row, col])
            
            # Scatter plot: all days colored by extreme status
            extreme_days = self.df[self.df[event] == 1]
            normal_days = self.df[self.df[event] == 0]
            
            ax.scatter(normal_days['dayofyear'], normal_days[variable], 
                      alpha=0.3, s=10, c='lightgray', label='Normal')
            ax.scatter(extreme_days['dayofyear'], extreme_days[variable], 
                      alpha=0.7, s=20, c='red', label='Extreme', edgecolors='darkred')
            
            # Add threshold line
            if 'hot' in event:
                ax.axhline(y=self.thresholds['temp_hot_p95'], color='red', 
                          linestyle='--', linewidth=2, alpha=0.7)
            elif 'cold' in event:
                ax.axhline(y=self.thresholds['temp_cold_p05'], color='blue', 
                          linestyle='--', linewidth=2, alpha=0.7)
            elif 'rain' in event:
                ax.axhline(y=self.thresholds['rain_heavy_p90'], color='blue', 
                          linestyle='--', linewidth=2, alpha=0.7)
            elif 'wind' in event:
                ax.axhline(y=self.thresholds['wind_strong_p90'], color='green', 
                          linestyle='--', linewidth=2, alpha=0.7)
            
            ax.set_xlabel('Day of Year', fontsize=10)
            ax.set_ylabel(variable.split('_')[0].upper(), fontsize=10)
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # 6. Overall statistics
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.axis('off')
        
        stats_text = f"""
        Dataset Statistics:
        ─────────────────
        Total Days: {len(self.df):,}
        Date Range: {self.df['date'].min().strftime('%Y-%m-%d')} 
                   to {self.df['date'].max().strftime('%Y-%m-%d')}
        
        Extreme Event Frequencies:
        ─────────────────────────
        Hot Days: {self.df['extreme_hot'].sum()} ({100*self.df['extreme_hot'].mean():.1f}%)
        Cold Days: {self.df['extreme_cold'].sum()} ({100*self.df['extreme_cold'].mean():.1f}%)
        Heavy Rain: {self.df['extreme_rain'].sum()} ({100*self.df['extreme_rain'].mean():.1f}%)
        Strong Wind: {self.df['extreme_wind'].sum()} ({100*self.df['extreme_wind'].mean():.1f}%)
        Discomfort: {self.df['extreme_discomfort'].sum()} ({100*self.df['extreme_discomfort'].mean():.1f}%)
        """
        
        ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, 
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        filename = f'{self.city.lower()}_extreme_summary.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved extreme event summary: {filename}")
        return filename


def export_predictions_with_feeling_levels(predictors, output_prefix='weather_predictions'):
    """
    Export all predictions to CSV with feeling levels for easy interpretation
    Creates master file and city-specific files
    """
    print("\n" + "="*70)
    print("EXPORTING PREDICTIONS WITH FEELING LEVELS")
    print("="*70)
    
    all_predictions = []
    
    # Generate predictions for every day
    for month in range(1, 13):
        for day in range(1, 32):
            try:
                target_date = datetime(2024, month, day)
                
                for city, predictor in predictors.items():
                    # Use the new query_date method
                    result = predictor.query_date(month, day, verbose=False)
                    
                    if 'error' not in result and result['predictions']:
                        for event_type, pred_data in result['predictions'].items():
                            all_predictions.append({
                                'city': city,
                                'date': target_date.strftime('%Y-%m-%d'),
                                'month': month,
                                'month_name': target_date.strftime('%B'),
                                'day': day,
                                'condition': pred_data['condition'],
                                'probability_percent': pred_data['probability'],
                                'feeling_level': pred_data['feeling_level'],
                                'feeling_description': pred_data['feeling_description'],
                                'historical_rate_percent': pred_data['historical_rate'],
                                'confidence_lower_percent': pred_data['confidence_interval']['lower'],
                                'confidence_upper_percent': pred_data['confidence_interval']['upper'],
                                'years_of_data': pred_data['years_of_data']
                            })
                            
            except ValueError:
                continue
    
    # Convert to DataFrame
    df = pd.DataFrame(all_predictions)
    
    # Export master file
    master_file = f'{output_prefix}_all_with_feelings.csv'
    df.to_csv(master_file, index=False)
    print(f"\n✓ Master file: {master_file}")
    print(f"  Total records: {len(df):,}")
    
    # Export city-specific files
    for city in predictors.keys():
        city_df = df[df['city'] == city].copy()
        city_file = f'{output_prefix}_{city.lower()}_with_feelings.csv'
        city_df.to_csv(city_file, index=False)
        print(f"  {city}: {city_file} ({len(city_df):,} records)")
    
    # Export condition-specific files for easy filtering
    print("\n✓ Condition-specific files:")
    for condition in df['condition'].unique():
        condition_df = df[df['condition'] == condition].copy()
        condition_file = f'{output_prefix}_{condition.lower().replace(" ", "_")}.csv'
        condition_df.to_csv(condition_file, index=False)
        print(f"  {condition}: {condition_file} ({len(condition_df):,} records)")
    
    # Create summary statistics
    print("\n✓ Summary statistics:")
    summary_stats = df.groupby(['city', 'condition']).agg({
        'probability_percent': ['mean', 'std', 'min', 'max'],
        'feeling_level': lambda x: x.value_counts().to_dict()
    }).round(2)
    
    summary_file = f'{output_prefix}_summary_statistics.csv'
    summary_stats.to_csv(summary_file)
    print(f"  {summary_file}")
    
    # Export thresholds with feeling level interpretation
    print("\n✓ Thresholds guide:")
    thresholds_data = []
    for city, predictor in predictors.items():
        thresholds_data.append({
            'city': city,
            'very_hot_celsius': round(predictor.thresholds['temp_hot_p95'], 2),
            'very_hot_fahrenheit': round(predictor._c_to_f(predictor.thresholds['temp_hot_p95']), 2),
            'very_cold_celsius': round(predictor.thresholds['temp_cold_p05'], 2),
            'very_cold_fahrenheit': round(predictor._c_to_f(predictor.thresholds['temp_cold_p05']), 2),
            'heavy_rain_mm': round(predictor.thresholds['rain_heavy_p90'], 2),
            'strong_wind_ms': round(predictor.thresholds['wind_strong_p90'], 2),
            'heat_index_threshold': round(predictor.thresholds['heat_index_p95'], 2)
        })
    
    thresholds_df = pd.DataFrame(thresholds_data)
    thresholds_file = f'{output_prefix}_thresholds_guide.csv'
    thresholds_df.to_csv(thresholds_file, index=False)
    print(f"  {thresholds_file}")
    
    # Create feeling level guide
    feeling_guide = pd.DataFrame([
        {'probability_range': '60-100%', 'feeling_level': 'Very Likely', 'description': 'High certainty this will occur', 'icon': '🔴'},
        {'probability_range': '40-59%', 'feeling_level': 'Likely', 'description': 'Good chance this will happen', 'icon': '🟠'},
        {'probability_range': '20-39%', 'feeling_level': 'Possible', 'description': 'Moderate likelihood', 'icon': '🟡'},
        {'probability_range': '10-19%', 'feeling_level': 'Unlikely', 'description': 'Low probability', 'icon': '🟢'},
        {'probability_range': '0-9%', 'feeling_level': 'Very Unlikely', 'description': 'Very low chance', 'icon': '🟢'}
    ])
    
    guide_file = f'{output_prefix}_feeling_level_guide.csv'
    feeling_guide.to_csv(guide_file, index=False)
    print(f"  {guide_file}")
    
    print("\n" + "="*70)
    print(f"Export complete! Generated {len(predictors) + 5 + len(df['condition'].unique())} CSV files")
    print("="*70)
    
    return master_file


def create_interactive_query_interface(predictors):
    """
    Simple command-line interface for querying predictions
    """
    print("\n" + "="*70)
    print("INTERACTIVE WEATHER PREDICTION QUERY")
    print("="*70)
    print("\nAvailable cities:")
    for i, city in enumerate(predictors.keys(), 1):
        print(f"  {i}. {city}")
    
    print("\nExample queries you can make:")
    print("  • predictors['Tokyo'].query_date(7, 15)")
    print("  • predictors['Singapore'].query_date(12, 25)")
    print("  • predictors['Phoenix'].query_date(8, 1)")
    
    # Demo queries
    print("\n" + "="*70)
    print("DEMO QUERIES - Popular Travel Dates")
    print("="*70)
    
    demo_dates = [
        (7, 4, "Independence Day (US)"),
        (12, 25, "Christmas"),
        (1, 1, "New Year's Day")
    ]
    
    for month, day, holiday in demo_dates:
        print(f"\n{'─'*70}")
        print(f"{holiday} - {datetime(2024, month, day).strftime('%B %d')}")
        print('─'*70)
        
        for city, predictor in list(predictors.items())[:3]:  # Show first 3 cities
            result = predictor.query_date(month, day, verbose=False)
            if result.get('predictions'):
                print(f"\n{city}:")
                for event_type, pred in result['predictions'].items():
                    level_icon = {"Very Likely": "🔴", "Likely": "🟠", "Possible": "🟡", 
                                 "Unlikely": "🟢", "Very Unlikely": "🟢"}
                    icon = level_icon.get(pred['feeling_level'], '')
                    print(f"  {icon} {pred['condition']}: {pred['feeling_level']} ({pred['probability']:.0f}%)")
    
    print("\n" + "="*70)
    
    return predictors
    """Run analysis for all cities and create comparative visualizations"""
    print("\n" + "#"*70)
    print("MULTI-CITY PROBABILISTIC WEATHER PREDICTION SYSTEM")
    print("#"*70)
    
    predictors = {}
    
    # Train models for each city
    for city, filepath in CITIES.items():
        print(f"\n{'='*70}")
        print(f"Processing: {city}")
        print('='*70)
        
        try:
            predictor = ProbabilisticWeatherPredictor(filepath, city)
            
            # Feature engineering
            print("\n[1/4] Engineering features...")
            predictor.engineer_features()
            print(f"  Created {len(predictor.df.columns)} features")
            
            # Define extreme events
            print("\n[2/4] Defining extreme weather events...")
            predictor.define_extreme_events()
            
            # Train models
            print("\n[3/4] Training ML models...")
            events = ['extreme_hot', 'extreme_cold', 'extreme_rain', 'extreme_wind', 'extreme_discomfort']
            for event in events:
                predictor.train_ensemble_model(event)
            
            # Generate visualizations
            print("\n[4/4] Creating visualizations...")
            predictor.plot_climate_trends()
            predictor.plot_seasonal_probability_curves(['extreme_hot', 'extreme_rain'])
            predictor.plot_extreme_event_summary()
            
            predictors[city] = predictor
            print(f"\n✓ {city} analysis complete")
            
        except Exception as e:
            print(f"\n✗ Error processing {city}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n\n" + "="*70)
    print("EXAMPLE: Planning International Summer Vacation (July 15)")
    print("="*70)
    
    for city, predictor in predictors.items():
        predictor.analyze_future_event(7, 15, ['extreme_hot', 'extreme_rain', 'extreme_discomfort'])
    
    print("\n\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nGenerated Visualizations per City:")
    print("  - [city]_climate_trends.png (temperature trends & event frequency)")
    print("  - [city]_seasonal_curves.png (probability curves throughout year)")
    print("  - [city]_extreme_summary.png (comprehensive event summary)")
    print("\nTo query specific dates:")
    print("  predictors['Singapore'].analyze_future_event(12, 25, ['extreme_rain'])")
    
    return predictors


# Only run if executed directly
if __name__ == "__main__":
    print("\nOptional performance enhancements:")
    if not HAS_XGB:
        print("  ⚠ XGBoost not found. Install with: pip install xgboost")
    else:
        print("  ✓ XGBoost available")
    if not HAS_LGBM:
        print("  ⚠ LightGBM not found. Install with: pip install lightgbm")
    else:
        print("  ✓ LightGBM available")
    print()
    
    predictors = run_complete_multi_city_analysis()
    
    print("\n\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nTo use the predictors, try:")
    print("  predictors['Singapore'].analyze_future_event(7, 15, ['extreme_hot', 'extreme_rain'])")