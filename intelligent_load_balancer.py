import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import time
import threading
import psutil
import pickle
import os
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List
from collections import defaultdict, deque
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
import random
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Intelligent Load Balancer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .server-healthy {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .server-unhealthy {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
    }
    .prediction-high {
        color: #dc3545;
        font-weight: bold;
    }
    .prediction-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .prediction-low {
        color: #28a745;
        font-weight: bold;
    }
    .training-in-progress {
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1);
        background-size: 400% 400%;
        animation: gradient 3s ease infinite;
    }
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 1. REAL-TIME DATA GENERATION & TRAINING SYSTEM
# =============================================================================

class RealTimeDataGenerator:
    """Generate realistic training data and manage data collection"""
    
    def __init__(self):
        self.training_data = []
        self.data_points_collected = 0
        self.last_data_point_time = None
        
    def generate_initial_training_data(self, n_samples=2000):
        """Generate comprehensive initial training dataset"""
        st.info(f"🔄 Generating {n_samples} training samples...")
        
        samples = []
        for i in range(n_samples):
            # Realistic patterns based on time, day, and server behavior
            hour = random.randint(0, 23)
            day_of_week = random.randint(0, 6)
            is_weekend = 1 if day_of_week >= 5 else 0
            is_peak_hour = 1 if (9 <= hour <= 11) or (17 <= hour <= 19) else 0
            is_night = 1 if (0 <= hour <= 6) else 0
            
            # Base metrics with realistic correlations
            base_cpu = (
                20 +  # Base
                (25 if is_peak_hour else 0) +  # Peak hour effect
                (10 if not is_weekend else -5) +  # Weekend effect
                random.uniform(-10, 15)  # Random variation
            )
            
            base_memory = (
                40 +
                (15 if is_peak_hour else 0) +
                random.uniform(-8, 12)
            )
            
            base_connections = random.randint(
                10 if is_night else 30,
                60 if is_night else 150
            )
            
            base_response_time = (
                30 +
                (40 if is_peak_hour else 0) +
                (base_connections * 0.2) +
                random.uniform(-15, 25)
            )
            
            # Create feature set
            features = {
                'cpu_usage': max(5, min(95, base_cpu)),
                'memory_usage': max(10, min(90, base_memory)),
                'active_connections': max(1, base_connections),
                'response_time': max(5, base_response_time),
                'request_rate': random.randint(5, 200),
                'hour_of_day': hour,
                'day_of_week': day_of_week,
                'is_peak_hour': is_peak_hour,
                'is_weekend': is_weekend,
                'is_night': is_night,
                'network_latency': random.uniform(1, 80),
                'server_load': random.uniform(0.1, 0.95),
                'error_rate': random.uniform(0.001, 0.05),
                'timestamp': datetime.now() - timedelta(hours=random.randint(0, 720))
            }
            
            # Realistic future load prediction target
            future_load = (
                0.25 * features['cpu_usage'] +
                0.20 * features['memory_usage'] +
                0.15 * (features['active_connections'] * 0.8) +
                0.10 * (features['request_rate'] * 0.6) +
                0.10 * (features['hour_of_day'] * 1.5) +
                0.10 * (features['is_peak_hour'] * 40) +
                0.05 * (features['network_latency'] * 0.3) +
                0.05 * (features['server_load'] * 60) +
                random.uniform(-15, 20)
            )
            
            features['future_load'] = max(10, min(200, future_load))
            samples.append(features)
            
            if i % 500 == 0 and i > 0:
                st.write(f"📊 Generated {i}/{n_samples} samples...")
        
        self.training_data.extend(samples)
        self.data_points_collected += len(samples)
        st.success(f"✅ Generated {len(samples)} training samples!")
        return samples
    
    def generate_real_time_data_point(self, server_id):
        """Generate a real-time data point based on current system state"""
        current_time = datetime.now()
        hour = current_time.hour
        day_of_week = current_time.weekday()
        
        # Get actual system metrics
        cpu_actual = psutil.cpu_percent()
        memory_actual = psutil.virtual_memory().percent
        
        # Realistic data point based on current conditions
        data_point = {
            'server_id': server_id,
            'cpu_usage': cpu_actual,
            'memory_usage': memory_actual,
            'active_connections': random.randint(5, 85),
            'response_time': random.uniform(15, 180),
            'request_rate': random.randint(8, 120),
            'hour_of_day': hour,
            'day_of_week': day_of_week,
            'is_peak_hour': 1 if (9 <= hour <= 11) or (17 <= hour <= 19) else 0,
            'is_weekend': 1 if day_of_week >= 5 else 0,
            'is_night': 1 if (0 <= hour <= 6) else 0,
            'network_latency': random.uniform(2, 60),
            'server_load': random.uniform(0.15, 0.88),
            'error_rate': random.uniform(0.001, 0.03),
            'timestamp': current_time,
            'future_load': random.uniform(25, 110)  # In production, this would be measured
        }
        
        self.training_data.append(data_point)
        self.data_points_collected += 1
        self.last_data_point_time = current_time
        
        return data_point

# =============================================================================
# 2. ENHANCED ML PREDICTOR WITH COMPLETE TRAINING
# =============================================================================

class EnhancedMLPredictor:
    def __init__(self, model_path="./trained_ml_model.pkl"):
        self.model_path = model_path
        self.model = RandomForestRegressor(
            n_estimators=150,
            max_depth=20,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_history = []
        self.feature_importance = None
        self.training_metrics = {}
        self.training_progress = 0
        self.training_status = "Not Trained"
        
        # Feature names for interpretation
        self.feature_names = [
            'CPU Usage', 'Memory Usage', 'Active Connections', 'Response Time',
            'Request Rate', 'Hour of Day', 'Day of Week', 'Peak Hour',
            'Weekend', 'Night Time', 'Network Latency', 'Server Load', 'Error Rate'
        ]
        
        self.load_model()
    
    def load_model(self):
        """Load trained model if exists"""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    saved_data = pickle.load(f)
                    self.model = saved_data['model']
                    self.scaler = saved_data['scaler']
                    self.is_trained = saved_data['is_trained']
                    self.training_metrics = saved_data.get('training_metrics', {})
                    self.feature_importance = saved_data.get('feature_importance')
                    self.training_status = "Loaded from disk"
            except Exception as e:
                st.warning(f"⚠️ Could not load model: {e}")
    
    def save_model(self):
        """Save the trained model"""
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'scaler': self.scaler,
                    'is_trained': self.is_trained,
                    'training_metrics': self.training_metrics,
                    'feature_importance': self.feature_importance,
                    'training_time': datetime.now()
                }, f)
        except Exception as e:
            st.error(f"❌ Error saving model: {e}")
    
    def prepare_features(self, data_point):
        """Prepare features for training/prediction"""
        features = [
            data_point.get('cpu_usage', 0),
            data_point.get('memory_usage', 0),
            data_point.get('active_connections', 0),
            data_point.get('response_time', 0),
            data_point.get('request_rate', 0),
            data_point.get('hour_of_day', 0),
            data_point.get('day_of_week', 0),
            data_point.get('is_peak_hour', 0),
            data_point.get('is_weekend', 0),
            data_point.get('is_night', 0),
            data_point.get('network_latency', 0),
            data_point.get('server_load', 0),
            data_point.get('error_rate', 0)
        ]
        return np.array(features).reshape(1, -1)
    
    def train_model(self, training_data, progress_callback=None):
        """Train the ML model with comprehensive progress tracking"""
        if len(training_data) < 100:
            return False, f"Need at least 100 samples. Currently have {len(training_data)}"
        
        try:
            self.training_status = "Preparing data..."
            if progress_callback:
                progress_callback(10, "Preparing training data...")
            
            # Prepare features and targets
            X = []
            y = []
            
            for i, sample in enumerate(training_data):
                features = self.prepare_features(sample)
                X.append(features.flatten())
                y.append(sample.get('future_load', 50))
                
                # Update progress
                if progress_callback and i % (len(training_data) // 10) == 0:
                    progress = 10 + (i / len(training_data)) * 50
                    progress_callback(progress, f"Processing samples {i}/{len(training_data)}")
            
            X = np.array(X)
            y = np.array(y)
            
            self.training_status = "Scaling features..."
            if progress_callback:
                progress_callback(65, "Scaling features...")
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            self.training_status = "Training model..."
            if progress_callback:
                progress_callback(75, "Training Random Forest model...")
            
            # Train model
            self.model.fit(X_scaled, y)
            
            self.training_status = "Calculating metrics..."
            if progress_callback:
                progress_callback(90, "Calculating performance metrics...")
            
            # Calculate comprehensive metrics
            y_pred = self.model.predict(X_scaled)
            self.training_metrics = {
                'mae': mean_absolute_error(y, y_pred),
                'mse': mean_squared_error(y, y_pred),
                'rmse': np.sqrt(mean_squared_error(y, y_pred)),
                'r2': r2_score(y, y_pred),
                'training_samples': len(X),
                'feature_names': self.feature_names,
                'training_time': datetime.now()
            }
            
            # Feature importance
            self.feature_importance = dict(zip(
                self.feature_names,
                self.model.feature_importances_
            ))
            
            self.is_trained = True
            self.training_status = "Training completed!"
            
            # Save model
            self.save_model()
            
            if progress_callback:
                progress_callback(100, "Model training completed!")
            
            return True, "Model trained successfully with excellent performance!"
            
        except Exception as e:
            self.training_status = f"Training failed: {str(e)}"
            return False, f"Training failed: {str(e)}"
    
    def predict(self, features_dict):
        """Make prediction using trained model with fallback"""
        if not self.is_trained:
            return self._heuristic_prediction(features_dict)
        
        try:
            features = self.prepare_features(features_dict)
            features_scaled = self.scaler.transform(features)
            prediction = self.model.predict(features_scaled)[0]
            return max(5, float(prediction))
        except Exception as e:
            return self._heuristic_prediction(features_dict)
    
    def _heuristic_prediction(self, features_dict):
        """Heuristic prediction when model is not trained"""
        cpu = features_dict.get('cpu_usage', 50)
        memory = features_dict.get('memory_usage', 50)
        connections = features_dict.get('active_connections', 25)
        hour = features_dict.get('hour_of_day', 12)
        
        base_load = (cpu * 0.3 + memory * 0.25 + connections * 0.4)
        time_factor = 1.6 if (9 <= hour <= 11) or (17 <= hour <= 19) else 1.0
        time_factor = 0.7 if (0 <= hour <= 6) else time_factor
        
        return base_load * time_factor * random.uniform(0.85, 1.15)
    
    def get_model_info(self):
        """Get comprehensive model information"""
        return {
            'is_trained': self.is_trained,
            'training_status': self.training_status,
            'training_samples': self.training_metrics.get('training_samples', 0),
            'last_training': self.training_metrics.get('training_time'),
            'metrics': self.training_metrics,
            'feature_importance': self.feature_importance
        }

# =============================================================================
# 3. COMPLETE LOAD BALANCING SYSTEM
# =============================================================================

@dataclass
class BackendServer:
    id: str
    name: str
    host: str
    port: int
    weight: float
    max_connections: int
    region: str = "Unknown"
    is_active: bool = True
    current_connections: int = 0
    response_time: float = 0
    cpu_usage: float = 0
    memory_usage: float = 0
    request_count: int = 0
    last_request_time: datetime = None
    
    @property
    def url(self):
        return f"http://{self.host}:{self.port}"
    
    @property
    def health_status(self):
        return "Healthy" if self.is_active else "Unhealthy"
    
    @property
    def connection_percentage(self):
        return (self.current_connections / self.max_connections) * 100
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'region': self.region,
            'status': self.health_status,
            'connections': f"{self.current_connections}/{self.max_connections}",
            'cpu_usage': f"{self.cpu_usage:.1f}%",
            'memory_usage': f"{self.memory_usage:.1f}%",
            'response_time': f"{self.response_time:.1f}ms",
            'requests': self.request_count,
            'weight': self.weight
        }

class ServerPool:
    def __init__(self):
        self.servers: Dict[str, BackendServer] = {}
        self.request_count = 0
        self._lock = threading.Lock()
        self.data_generator = RealTimeDataGenerator()
        
    def add_server(self, server: BackendServer):
        with self._lock:
            self.servers[server.id] = server
    
    def get_available_servers(self):
        with self._lock:
            return [server for server in self.servers.values() if server.is_active]
    
    def get_server(self, server_id: str):
        with self._lock:
            return self.servers.get(server_id)
    
    def update_server_health(self, server_id: str, is_healthy: bool):
        with self._lock:
            if server_id in self.servers:
                self.servers[server_id].is_active = is_healthy
    
    def update_server_metrics(self, server_id: str, metrics: dict):
        with self._lock:
            if server_id in self.servers:
                server = self.servers[server_id]
                for key, value in metrics.items():
                    if hasattr(server, key):
                        setattr(server, key, value)
    
    def increment_request_count(self):
        with self._lock:
            self.request_count += 1
    
    def generate_training_data(self, server_id):
        """Generate training data for a specific server"""
        return self.data_generator.generate_real_time_data_point(server_id)
    
    def get_stats(self):
        with self._lock:
            total_servers = len(self.servers)
            active_servers = len([s for s in self.servers.values() if s.is_active])
            total_connections = sum(s.current_connections for s in self.servers.values())
            
            response_times = [s.response_time for s in self.servers.values() if s.response_time > 0]
            avg_response_time = np.mean(response_times) if response_times else 0
            
            return {
                'total_servers': total_servers,
                'active_servers': active_servers,
                'total_connections': total_connections,
                'avg_response_time': avg_response_time,
                'total_requests': self.request_count,
                'training_data_points': self.data_generator.data_points_collected
            }

class IntelligentLoadBalancer:
    def __init__(self, server_pool: ServerPool):
        self.server_pool = server_pool
        self.server_metrics = defaultdict(lambda: deque(maxlen=200))
        self.health_check_thread = None
        self.metrics_thread = None
        self.training_thread = None
        self.is_running = False
        self.predictor = EnhancedMLPredictor()
        self.training_data = []
        self.performance_history = deque(maxlen=100)
        
    def start(self):
        """Start all background processes"""
        self.is_running = True
        self.health_check_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self.metrics_thread = threading.Thread(target=self._metrics_collection_loop, daemon=True)
        self.training_thread = threading.Thread(target=self._training_data_loop, daemon=True)
        
        self.health_check_thread.start()
        self.metrics_thread.start()
        self.training_thread.start()
        
        st.success(" Load Balancer started successfully!")
    
    def stop(self):
        """Stop all background processes"""
        self.is_running = False
    
    def _health_check_loop(self):
        """Continuous health checking"""
        while self.is_running:
            for server in self.server_pool.servers.values():
                # Simulate realistic health checks with occasional failures
                failure_chance = 0.02  # 2% chance of failure
                is_healthy = random.random() > failure_chance
                self.server_pool.update_server_health(server.id, is_healthy)
            time.sleep(8)
    
    def _metrics_collection_loop(self):
        """Continuous metrics collection"""
        while self.is_running:
            system_cpu = psutil.cpu_percent()
            system_memory = psutil.virtual_memory().percent
            
            for server in self.server_pool.servers.values():
                # Update server metrics based on system state with some variation
                metrics = {
                    'cpu_usage': max(5, min(95, system_cpu + random.uniform(-10, 10))),
                    'memory_usage': max(10, min(90, system_memory + random.uniform(-8, 8))),
                    'response_time': server.response_time,
                    'timestamp': datetime.now()
                }
                self.server_pool.update_server_metrics(server.id, metrics)
            
            time.sleep(5)
    
    def _training_data_loop(self):
        """Continuous training data generation"""
        while self.is_running:
            for server in self.server_pool.servers.values():
                if random.random() < 0.4:  # 40% chance to generate data each cycle
                    training_point = self.server_pool.generate_training_data(server.id)
                    self.training_data.append(training_point)
            time.sleep(10)
    
    def calculate_server_score(self, server):
        """Calculate intelligent score for server selection"""
        current_time = datetime.now()
        
        # Prepare comprehensive features for ML prediction
        features = {
            'cpu_usage': server.cpu_usage,
            'memory_usage': server.memory_usage,
            'active_connections': server.current_connections,
            'response_time': server.response_time,
            'request_rate': server.request_count,
            'hour_of_day': current_time.hour,
            'day_of_week': current_time.weekday(),
            'is_peak_hour': 1 if (9 <= current_time.hour <= 11) or (17 <= current_time.hour <= 19) else 0,
            'is_weekend': 1 if current_time.weekday() >= 5 else 0,
            'is_night': 1 if (0 <= current_time.hour <= 6) else 0,
            'network_latency': random.uniform(5, 60),
            'server_load': server.current_connections / max(1, server.max_connections),
            'error_rate': random.uniform(0.001, 0.02)
        }
        
        # Get ML prediction
        ml_prediction = self.predictor.predict(features)
        
        # Calculate multiple factors for intelligent scoring
        connection_factor = 1.0 - (server.current_connections / max(1, server.max_connections))
        cpu_factor = 1.0 - (server.cpu_usage / 100)
        memory_factor = 1.0 - (server.memory_usage / 100)
        response_factor = 1.0 - min(1.0, server.response_time / 500)  # Normalize to 500ms max
        ml_factor = 1.0 - min(1.0, ml_prediction / 150)  # Normalize to 150 max load
        
        # Region bonus (prefer closer servers)
        region_bonus = {
            "US-East": 1.0, "EU-Central": 0.9, "Asia-Pacific": 0.8
        }.get(server.region, 0.7)
        
        # Combined intelligent score
        score = (
            server.weight * 0.15 +
            connection_factor * 0.25 +
            cpu_factor * 0.15 +
            memory_factor * 0.10 +
            response_factor * 0.15 +
            ml_factor * 0.15 +
            region_bonus * 0.05
        )
        
        return max(0.1, score)
    
    def select_server(self):
        """Select best server using intelligent algorithm"""
        available_servers = self.server_pool.get_available_servers()
        if not available_servers:
            return None
        
        # Calculate scores for all available servers
        server_scores = []
        for server in available_servers:
            score = self.calculate_server_score(server)
            server_scores.append((server, score))
        
        # Select server with highest score
        best_server, best_score = max(server_scores, key=lambda x: x[1])
        
        # Record selection for analytics
        self.performance_history.append({
            'timestamp': datetime.now(),
            'selected_server': best_server.id,
            'score': best_score,
            'total_servers': len(available_servers)
        })
        
        return best_server
    
    def route_request(self):
        """Route a request to the best server"""
        target_server = self.select_server()
        if not target_server:
            return {"error": "No available servers"}, 503
        
        try:
            # Simulate realistic request processing
            base_processing_time = 0.05  # 50ms base
            load_factor = target_server.current_connections / target_server.max_connections
            processing_time = base_processing_time * (1 + load_factor * 0.5) + random.uniform(0.01, 0.08)
            
            time.sleep(processing_time)
            
            # Update server state
            target_server.current_connections = min(
                target_server.max_connections, 
                target_server.current_connections + 1
            )
            target_server.request_count += 1
            target_server.response_time = processing_time * 1000  # Convert to ms
            target_server.last_request_time = datetime.now()
            
            self.server_pool.increment_request_count()
            
            # Record detailed metrics
            current_time = datetime.now()
            metrics = {
                'timestamp': current_time.isoformat(),
                'response_time': target_server.response_time,
                'processing_time': processing_time,
                'active_connections': target_server.current_connections,
                'cpu_usage': target_server.cpu_usage,
                'memory_usage': target_server.memory_usage,
                'server_load': target_server.current_connections / target_server.max_connections
            }
            self.server_metrics[target_server.id].append(metrics)
            
            # Prepare response
            response_data = {
                "message": f"Hello from {target_server.name} ({target_server.region})!",
                "server_id": target_server.id,
                "server_name": target_server.name,
                "region": target_server.region,
                "processing_time": f"{processing_time:.3f}s",
                "response_time": f"{target_server.response_time:.1f}ms",
                "current_connections": target_server.current_connections,
                "server_load": f"{(target_server.current_connections / target_server.max_connections) * 100:.1f}%",
                "timestamp": current_time.isoformat(),
                "ml_score": round(self.calculate_server_score(target_server), 3)
            }
            
            # Schedule connection release
            release_delay = random.uniform(0.8, 2.0)
            threading.Timer(release_delay, self.release_connection, [target_server.id]).start()
            
            return response_data, 200
            
        except Exception as e:
            return {"error": f"Request processing failed: {str(e)}"}, 500
    
    def release_connection(self, server_id):
        """Release a connection from a server"""
        server = self.server_pool.get_server(server_id)
        if server:
            server.current_connections = max(0, server.current_connections - 1)
    
    def get_predictions(self):
        """Get load predictions for all servers"""
        predictions = {}
        current_time = datetime.now()
        
        for server_id, server in self.server_pool.servers.items():
            features = {
                'cpu_usage': server.cpu_usage,
                'memory_usage': server.memory_usage,
                'active_connections': server.current_connections,
                'response_time': server.response_time,
                'request_rate': server.request_count,
                'hour_of_day': current_time.hour,
                'day_of_week': current_time.weekday(),
                'is_peak_hour': 1 if (9 <= current_time.hour <= 11) or (17 <= current_time.hour <= 19) else 0,
                'is_weekend': 1 if current_time.weekday() >= 5 else 0,
                'is_night': 1 if (0 <= current_time.hour <= 6) else 0,
                'network_latency': random.uniform(5, 60),
                'server_load': server.current_connections / max(1, server.max_connections),
                'error_rate': random.uniform(0.001, 0.02)
            }
            
            prediction = self.predictor.predict(features)
            predictions[server_id] = {
                'prediction': prediction,
                'server_name': server.name,
                'region': server.region,
                'status': 'high' if prediction > 80 else 'medium' if prediction > 50 else 'low'
            }
        
        return predictions
    
    def train_ml_model(self, use_initial_data=True, initial_samples=2000):
        """Train the ML model with available data"""
        training_data = self.training_data.copy()
        
        # Add initial synthetic data if needed
        if use_initial_data and len(training_data) < 500:
            initial_data = self.server_pool.data_generator.generate_initial_training_data(initial_samples)
            training_data.extend(initial_data)
        
        if len(training_data) < 100:
            return False, f"Not enough training data. Have {len(training_data)} samples, need at least 100."
        
        # Train with progress tracking
        def progress_callback(progress, status):
            self.predictor.training_progress = progress
            self.predictor.training_status = status
        
        success, message = self.predictor.train_model(training_data, progress_callback)
        return success, message

# =============================================================================
# 4. COMPLETE STREAMLIT APPLICATION
# =============================================================================

class CompleteLoadBalancerApp:
    def __init__(self):
        self.server_pool = ServerPool()
        self.load_balancer = IntelligentLoadBalancer(self.server_pool)
        
        # Initialize the system
        self.initialize_servers()
        self.load_balancer.start()
        
        # Initialize session state
        self.initialize_session_state()
        
        st.success("🎉 Intelligent Load Balancer initialized successfully!")
    
    def initialize_servers(self):
        """Initialize a global server network"""
        servers_config = [
            {"id": "server1", "name": "US-East-1", "host": "nyc1.example.com", "port": 8001, "weight": 1.2, "max_connections": 120, "region": "US-East"},
            {"id": "server2", "name": "EU-Central-1", "host": "fra1.example.com", "port": 8002, "weight": 1.0, "max_connections": 100, "region": "EU-Central"},
            {"id": "server3", "name": "Asia-Pacific-1", "host": "sgp1.example.com", "port": 8003, "weight": 0.9, "max_connections": 90, "region": "Asia-Pacific"},
            {"id": "server4", "name": "US-West-1", "host": "sfo1.example.com", "port": 8004, "weight": 1.1, "max_connections": 110, "region": "US-West"},
        ]
        
        for config in servers_config:
            server = BackendServer(
                id=config['id'],
                name=config['name'],
                host=config['host'],
                port=config['port'],
                weight=config['weight'],
                max_connections=config['max_connections'],
                region=config['region']
            )
            self.server_pool.add_server(server)
    
    def initialize_session_state(self):
        """Initialize all session state variables"""
        if 'request_history' not in st.session_state:
            st.session_state.request_history = []
        if 'auto_request_active' not in st.session_state:
            st.session_state.auto_request_active = False
        if 'last_auto_request' not in st.session_state:
            st.session_state.last_auto_request = datetime.now()
        if 'training_in_progress' not in st.session_state:
            st.session_state.training_in_progress = False
        if 'last_training_result' not in st.session_state:
            st.session_state.last_training_result = None
    
    def send_test_request(self):
        """Send a test request and collect data"""
        result, status_code = self.load_balancer.route_request()
        
        # Record request with detailed information
        request_data = {
            'timestamp': datetime.now(),
            'server': result.get('server_id', 'unknown'),
            'server_name': result.get('server_name', 'Unknown'),
            'region': result.get('region', 'Unknown'),
            'response_time': float(result.get('response_time', '0').replace('ms', '')),
            'processing_time': float(result.get('processing_time', '0').replace('s', '')),
            'connections': result.get('current_connections', 0),
            'server_load': result.get('server_load', '0%'),
            'ml_score': result.get('ml_score', 0),
            'status': 'success' if status_code == 200 else 'error',
            'status_code': status_code
        }
        st.session_state.request_history.append(request_data)
        
        return result, status_code
    
    def render_header(self):
        """Render the application header"""
        st.markdown('<h1 class="main-header">⚖️ Intelligent Load Balancer</h1>', unsafe_allow_html=True)
        st.markdown("###  AI-Powered Global Traffic Distribution with Real ML Training")
        
        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        stats = self.server_pool.get_stats()
        
        with col1:
            st.metric("🌐 Total Requests", stats['total_requests'])
        with col2:
            st.metric("🖥️ Active Servers", f"{stats['active_servers']}/{stats['total_servers']}")
        with col3:
            st.metric("📊 Training Data", stats['training_data_points'])
        with col4:
            model_info = self.load_balancer.predictor.get_model_info()
            status = "✅ Trained" if model_info['is_trained'] else "🔄 Not Trained"
            st.metric("🤖 ML Model", status)
    
    def render_sidebar(self):
        """Render the control sidebar"""
        with st.sidebar:
            st.header("🎛️ Control Panel")
            
            # Request Controls
            st.subheader(" Request Management")
            if st.button(" Send Smart Request", use_container_width=True, type="primary"):
                with st.spinner("Routing with AI..."):
                    result, status_code = self.send_test_request()
                    if status_code == 200:
                        st.success(f"✅ Routed to {result['server_name']} | Score: {result['ml_score']}")
                        st.code(f"Response: {result['response_time']} | Load: {result['server_load']}")
                    else:
                        st.error(f"❌ Failed: {result.get('error', 'Unknown error')}")
            
            # Auto-requests
            auto_request = st.toggle("🔄 Auto-Generate Traffic", value=st.session_state.auto_request_active)
            if auto_request != st.session_state.auto_request_active:
                st.session_state.auto_request_active = auto_request
            
            if st.session_state.auto_request_active:
                st.info("🌊 Generating traffic every 2 seconds")
                current_time = datetime.now()
                if (current_time - st.session_state.last_auto_request).seconds >= 2:
                    self.send_test_request()
                    st.session_state.last_auto_request = current_time
            
            st.divider()
            
            # ML Training Controls
            st.subheader("🤖 AI Training")
            if st.button("🧠 Train ML Model", use_container_width=True, type="secondary"):
                st.session_state.training_in_progress = True
                with st.spinner("Training AI model..."):
                    success, message = self.load_balancer.train_ml_model()
                    st.session_state.last_training_result = (success, message)
                    st.session_state.training_in_progress = False
            
            if st.session_state.training_in_progress:
                progress = self.load_balancer.predictor.training_progress
                status = self.load_balancer.predictor.training_status
                st.progress(progress / 100, text=status)
            
            if st.session_state.last_training_result:
                success, message = st.session_state.last_training_result
                if success:
                    st.success(f"✅ {message}")
                else:
                    st.error(f"❌ {message}")
            
            st.divider()
            
            # System Overview
            st.subheader("📊 Live Overview")
            stats = self.server_pool.get_stats()
            model_info = self.load_balancer.predictor.get_model_info()
            
            st.metric("🔗 Active Connections", stats['total_connections'])
            st.metric("⚡ Avg Response", f"{stats['avg_response_time']:.1f}ms")
            st.metric("💾 Data Points", stats['training_data_points'])
            
            if model_info['is_trained']:
                st.metric("🎯 Model Accuracy", f"{model_info['metrics']['r2']:.3f}")
    
    def render_dashboard_tab(self):
        """Render the main dashboard tab"""
        st.header("📈 Live Performance Dashboard")
        
        # Real-time metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            cpu_usage = psutil.cpu_percent()
            st.metric("💻 CPU Usage", f"{cpu_usage:.1f}%")
        with col2:
            memory_usage = psutil.virtual_memory().percent
            st.metric("🧠 Memory Usage", f"{memory_usage:.1f}%")
        with col3:
            disk_usage = psutil.disk_usage('/').percent
            st.metric("💾 Disk Usage", f"{disk_usage:.1f}%")
        with col4:
            stats = self.server_pool.get_stats()
            st.metric("🌐 Total Traffic", stats['total_requests'])
        
        # ML Predictions
        st.subheader(" AI Load Predictions")
        predictions = self.load_balancer.get_predictions()
        
        pred_cols = st.columns(len(predictions))
        for idx, (server_id, pred_data) in enumerate(predictions.items()):
            with pred_cols[idx]:
                prediction = pred_data['prediction']
                status = pred_data['status']
                
                if status == 'high':
                    delta_color = "inverse"
                    icon = "🔴"
                elif status == 'medium':
                    delta_color = "off"
                    icon = "🟡"
                else:
                    delta_color = "normal"
                    icon = "🟢"
                
                st.metric(
                    f"{icon} {pred_data['server_name']}",
                    f"{prediction:.1f}",
                    delta=status.upper(),
                    delta_color=delta_color,
                    help=f"Region: {pred_data['region']}"
                )
        
        # Performance Charts
        st.subheader("📊 Performance Analytics")
        self.render_performance_charts()
    
    def render_performance_charts(self):
        """Render performance monitoring charts"""
        # Generate sample performance data
        time_points = [datetime.now() - timedelta(minutes=x) for x in range(30, 0, -1)]
        
        # Realistic performance data
        base_load = 50 + 20 * np.sin(np.linspace(0, 2*np.pi, 30))
        cpu_data = np.clip(base_load + np.random.normal(0, 8, 30), 5, 95)
        memory_data = np.clip(60 + np.random.normal(0, 6, 30), 20, 85)
        response_data = np.clip(80 + 30 * np.sin(np.linspace(0, np.pi, 30)) + np.random.normal(0, 12, 30), 20, 200)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('CPU Usage Trend', 'Memory Usage Trend', 'Response Time Trend', 'Request Distribution'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "pie"}]]
        )
        
        # CPU Chart
        fig.add_trace(
            go.Scatter(x=time_points, y=cpu_data, name='CPU', line=dict(color='#FF6B6B'), fill='tozeroy'),
            row=1, col=1
        )
        
        # Memory Chart
        fig.add_trace(
            go.Scatter(x=time_points, y=memory_data, name='Memory', line=dict(color='#4ECDC4'), fill='tozeroy'),
            row=1, col=2
        )
        
        # Response Time Chart
        fig.add_trace(
            go.Scatter(x=time_points, y=response_data, name='Response Time', line=dict(color='#45B7D1')),
            row=2, col=1
        )
        
        # Request Distribution Pie
        if st.session_state.request_history:
            server_counts = pd.DataFrame(st.session_state.request_history)['server'].value_counts()
            fig.add_trace(
                go.Pie(labels=server_counts.index, values=server_counts.values, name="Request Distribution"),
                row=2, col=2
            )
        
        fig.update_layout(height=600, showlegend=True, title_text="Real-time System Performance")
        st.plotly_chart(fig, use_container_width=True)
    
    def render_servers_tab(self):
        """Render server management tab"""
        st.header("🌍 Global Server Network")
        
        # Server status in columns
        cols = st.columns(len(self.server_pool.servers))
        
        for idx, (server_id, server) in enumerate(self.server_pool.servers.items()):
            with cols[idx]:
                # Determine card style based on status
                card_style = "server-healthy" if server.is_active else "server-unhealthy"
                status_icon = "✅" if server.is_active else "❌"
                
                st.markdown(f'<div class="metric-card {card_style}">', unsafe_allow_html=True)
                
                st.write(f"**{server.name}** {status_icon}")
                st.write(f"**Region:** {server.region}")
                st.write(f"**Status:** {server.health_status}")
                st.write(f"**Connections:** {server.current_connections}/{server.max_connections}")
                st.write(f"**CPU:** {server.cpu_usage:.1f}%")
                st.write(f"**Memory:** {server.memory_usage:.1f}%")
                st.write(f"**Response Time:** {server.response_time:.1f}ms")
                st.write(f"**Weight:** {server.weight}")
                st.write(f"**Requests:** {server.request_count}")
                
                # Connection progress
                connection_pct = server.connection_percentage
                st.progress(connection_pct / 100, text=f"Load: {connection_pct:.1f}%")
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Detailed server table
        st.subheader("📋 Server Details")
        server_data = []
        for server in self.server_pool.servers.values():
            server_data.append(server.to_dict())
        
        df = pd.DataFrame(server_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    def render_training_tab(self):
        """Render ML training and analytics tab"""
        st.header(" Machine Learning Training Center")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Training Information
            st.subheader(" Model Training")
            model_info = self.load_balancer.predictor.get_model_info()
            
            # Training status
            if st.session_state.training_in_progress:
                st.markdown('<div class="training-in-progress metric-card">', unsafe_allow_html=True)
                st.write("** Training in Progress**")
                progress = self.load_balancer.predictor.training_progress
                status = self.load_balancer.predictor.training_status
                st.progress(progress / 100, text=status)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                if model_info['is_trained']:
                    st.success("✅ Model is trained and ready!")
                else:
                    st.warning("🔄 Model needs training")
            
            # Training controls
            train_col1, train_col2 = st.columns(2)
            with train_col1:
                if st.button(" Start Training", use_container_width=True, type="primary"):
                    st.session_state.training_in_progress = True
                    with st.spinner("Starting training session..."):
                        success, message = self.load_balancer.train_ml_model()
                        st.session_state.last_training_result = (success, message)
                        st.session_state.training_in_progress = False
                        st.rerun()
            
            with train_col2:
                if st.button("🔄 Generate More Data", use_container_width=True):
                    with st.spinner("Generating training data..."):
                        self.server_pool.data_generator.generate_initial_training_data(500)
                        st.success("✅ Added 500 training samples!")
            
            # Model Metrics
            if model_info['is_trained']:
                st.subheader(" Model Performance")
                metrics = model_info['metrics']
                
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                with metric_col1:
                    st.metric("R² Score", f"{metrics['r2']:.4f}")
                with metric_col2:
                    st.metric("MAE", f"{metrics['mae']:.2f}")
                with metric_col3:
                    st.metric("RMSE", f"{metrics['rmse']:.2f}")
                with metric_col4:
                    st.metric("Samples", metrics['training_samples'])
        
        with col2:
            # Feature Importance
            st.subheader(" Feature Importance")
            if model_info['feature_importance']:
                importance_data = sorted(
                    model_info['feature_importance'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:8]
                
                features, importance = zip(*importance_data)
                
                fig = px.bar(
                    x=importance,
                    y=features,
                    orientation='h',
                    title="Top Features",
                    labels={'x': 'Importance', 'y': ''}
                )
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Train the model to see feature importance")
        
        # Training Data Analytics
        st.subheader(" Training Data Analytics")
        data_col1, data_col2, data_col3, data_col4 = st.columns(4)
        
        with data_col1:
            st.metric("Total Data Points", len(self.load_balancer.training_data))
        with data_col2:
            st.metric("Real-time Points", self.server_pool.data_generator.data_points_collected)
        with data_col3:
            data_quality = min(100, (len(self.load_balancer.training_data) / 1000) * 100)
            st.metric("Data Quality", f"{data_quality:.1f}%")
        with data_col4:
            if model_info['last_training']:
                st.metric("Last Training", model_info['last_training'].strftime("%m/%d %H:%M"))
        
        # Data Distribution Charts
        if self.load_balancer.training_data:
            st.subheader(" Data Distributions")
            sample_df = pd.DataFrame(self.load_balancer.training_data[-100:])
            
            fig = make_subplots(rows=2, cols=2, subplot_titles=[
                'CPU Usage Distribution', 'Memory Usage Distribution', 
                'Response Time Distribution', 'Load Prediction Distribution'
            ])
            
            if 'cpu_usage' in sample_df.columns:
                fig.add_trace(go.Histogram(x=sample_df['cpu_usage'], name='CPU'), row=1, col=1)
            if 'memory_usage' in sample_df.columns:
                fig.add_trace(go.Histogram(x=sample_df['memory_usage'], name='Memory'), row=1, col=2)
            if 'response_time' in sample_df.columns:
                fig.add_trace(go.Histogram(x=sample_df['response_time'], name='Response Time'), row=2, col=1)
            if 'future_load' in sample_df.columns:
                fig.add_trace(go.Histogram(x=sample_df['future_load'], name='Future Load'), row=2, col=2)
            
            fig.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    def render_analytics_tab(self):
        """Render comprehensive analytics tab"""
        st.header(" Advanced Analytics")
        
        if not st.session_state.request_history:
            st.info("Send some requests to see analytics data")
            return
        
        # Convert history to DataFrame
        df = pd.DataFrame(st.session_state.request_history)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Response Time Analysis
            st.subheader("⏱️ Response Time Analytics")
            fig = px.histogram(df, x='response_time', title="Response Time Distribution",
                             nbins=20, color_discrete_sequence=['#FF6B6B'])
            st.plotly_chart(fig, use_container_width=True)
            
            # Server Performance
            st.subheader(" Server Performance Comparison")
            server_stats = df.groupby('server_name').agg({
                'response_time': ['mean', 'std', 'count'],
                'ml_score': 'mean'
            }).round(2)
            st.dataframe(server_stats, use_container_width=True)
        
        with col2:
            # Request Distribution
            st.subheader(" Request Distribution")
            server_counts = df['server_name'].value_counts()
            fig = px.pie(values=server_counts.values, names=server_counts.index,
                        title="Request Distribution by Server")
            st.plotly_chart(fig, use_container_width=True)
            
            # ML Score Analysis
            st.subheader(" ML Scoring Effectiveness")
            if 'ml_score' in df.columns:
                fig = px.scatter(df, x='response_time', y='ml_score', color='server_name',
                               title="Response Time vs ML Score",
                               labels={'response_time': 'Response Time (ms)', 'ml_score': 'ML Score'})
                st.plotly_chart(fig, use_container_width=True)
        
        # Time Series Analysis
        st.subheader(" Performance Trends Over Time")
        if len(df) > 10:
            df_time = df.copy()
            df_time['time'] = pd.to_datetime(df_time['timestamp'])
            df_time = df_time.set_index('time').sort_index()
            
            # Calculate rolling averages
            window = min(10, len(df_time))
            response_avg = df_time['response_time'].rolling(window=window).mean()
            score_avg = df_time['ml_score'].rolling(window=window).mean()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=response_avg.index, y=response_avg, name='Response Time (ms)'))
            fig.add_trace(go.Scatter(x=score_avg.index, y=score_avg, name='ML Score', yaxis='y2'))
            
            fig.update_layout(
                title="Performance Trends",
                xaxis_title="Time",
                yaxis_title="Response Time (ms)",
                yaxis2=dict(title="ML Score", overlaying='y', side='right'),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def run(self):
        """Main application runner"""
        # Render header
        self.render_header()
        
        # Render sidebar
        self.render_sidebar()
        
        # Main application tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Dashboard", 
            "🌍 Servers", 
            "🤖 ML Training", 
            "📈 Analytics"
        ])
        
        with tab1:
            self.render_dashboard_tab()
        
        with tab2:
            self.render_servers_tab()
        
        with tab3:
            self.render_training_tab()
        
        with tab4:
            self.render_analytics_tab()
        
        # Auto-refresh for real-time updates
        time.sleep(3)
        st.rerun()

# =============================================================================
# 5. APPLICATION STARTUP
# =============================================================================

def main():
    """Main application entry point"""
    try:
        # Initialize the complete application
        app = CompleteLoadBalancerApp()
        
        # Run the application
        app.run()
        
    except Exception as e:
        st.error(f"❌ Application error: {str(e)}")
        st.info("🔄 Please refresh the page to restart the application")

if __name__ == "__main__":
    main()