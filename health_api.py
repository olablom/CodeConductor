#!/usr/bin/env python3
"""
Health API for CodeConductor - Provides health endpoints for external monitoring
"""

from flask import Flask, jsonify, request
import asyncio
import time
import json
from datetime import datetime
import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

from ensemble.model_manager import ModelManager
from ensemble.ensemble_engine import EnsembleEngine

app = Flask(__name__)

# Global monitoring data
monitoring_data = {
    'start_time': time.time(),
    'total_requests': 0,
    'successful_requests': 0,
    'failed_requests': 0,
    'model_metrics': {},
    'last_health_check': 0
}

class HealthMonitor:
    """Simple health monitor for the API"""
    
    def __init__(self):
        self.model_manager = ModelManager()
        self.ensemble_engine = None
        self.last_check = 0
        self.check_interval = 30  # Check every 30 seconds
    
    async def check_model_health(self):
        """Check health of all models"""
        try:
            models = await self.model_manager.list_models()
            health_data = {}
            
            for model in models:
                try:
                    # Simple health check - try to get model info
                    is_healthy = await self.model_manager.check_health(model)
                    health_data[model.id] = {
                        'status': 'healthy' if is_healthy else 'unhealthy',
                        'provider': model.provider,
                        'endpoint': model.endpoint,
                        'last_check': time.time()
                    }
                except Exception as e:
                    health_data[model.id] = {
                        'status': 'error',
                        'error': str(e),
                        'last_check': time.time()
                    }
            
            return health_data
        except Exception as e:
            return {'error': str(e)}
    
    async def check_ensemble_health(self):
        """Check if ensemble engine is working"""
        try:
            if not self.ensemble_engine:
                self.ensemble_engine = EnsembleEngine(min_confidence=0.6)
            
            # Try to initialize ensemble engine
            success = await self.ensemble_engine.initialize()
            
            return {
                'status': 'healthy' if success else 'unhealthy',
                'models_available': len(await self.model_manager.list_models()) if success else 0,
                'last_check': time.time()
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'last_check': time.time()
            }

# Global health monitor
health_monitor = HealthMonitor()

@app.route('/health', methods=['GET'])
def health_check():
    """Main health endpoint"""
    global monitoring_data
    
    # Update request count
    monitoring_data['total_requests'] += 1
    
    try:
        # Get current time
        current_time = time.time()
        
        # Check if we need to refresh health data
        if current_time - monitoring_data['last_health_check'] > 30:
            # Run async health checks
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                model_health = loop.run_until_complete(health_monitor.check_model_health())
                ensemble_health = loop.run_until_complete(health_monitor.check_ensemble_health())
                
                monitoring_data['model_health'] = model_health
                monitoring_data['ensemble_health'] = ensemble_health
                monitoring_data['last_health_check'] = current_time
            finally:
                loop.close()
        
        # Calculate overall health
        model_health = monitoring_data.get('model_health', {})
        ensemble_health = monitoring_data.get('ensemble_health', {})
        
        # Count healthy models
        healthy_models = sum(1 for model in model_health.values() 
                           if isinstance(model, dict) and model.get('status') == 'healthy')
        total_models = len(model_health)
        
        # Determine overall status
        if ensemble_health.get('status') == 'healthy' and healthy_models > 0:
            overall_status = 'healthy'
            status_code = 200
        elif healthy_models > 0:
            overall_status = 'degraded'
            status_code = 200
        else:
            overall_status = 'unhealthy'
            status_code = 503
        
        # Build response
        response = {
            'status': overall_status,
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': current_time - monitoring_data['start_time'],
            'system': {
                'total_requests': monitoring_data['total_requests'],
                'successful_requests': monitoring_data['successful_requests'],
                'failed_requests': monitoring_data['failed_requests'],
                'success_rate': monitoring_data['successful_requests'] / monitoring_data['total_requests'] if monitoring_data['total_requests'] > 0 else 0
            },
            'ensemble': ensemble_health,
            'models': {
                'total': total_models,
                'healthy': healthy_models,
                'details': model_health
            }
        }
        
        monitoring_data['successful_requests'] += 1
        return jsonify(response), status_code
        
    except Exception as e:
        monitoring_data['failed_requests'] += 1
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/health/models', methods=['GET'])
def models_health():
    """Detailed model health endpoint"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            model_health = loop.run_until_complete(health_monitor.check_model_health())
            return jsonify({
                'status': 'success',
                'models': model_health,
                'timestamp': datetime.now().isoformat()
            }), 200
        finally:
            loop.close()
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/health/ensemble', methods=['GET'])
def ensemble_health():
    """Ensemble engine health endpoint"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            ensemble_health = loop.run_until_complete(health_monitor.check_ensemble_health())
            return jsonify({
                'status': 'success',
                'ensemble': ensemble_health,
                'timestamp': datetime.now().isoformat()
            }), 200
        finally:
            loop.close()
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus-style metrics endpoint"""
    global monitoring_data
    
    current_time = time.time()
    uptime = current_time - monitoring_data['start_time']
    
    metrics = f"""# CodeConductor Metrics
codeconductor_uptime_seconds {uptime}
codeconductor_total_requests {monitoring_data['total_requests']}
codeconductor_successful_requests {monitoring_data['successful_requests']}
codeconductor_failed_requests {monitoring_data['failed_requests']}
codeconductor_success_rate {monitoring_data['successful_requests'] / monitoring_data['total_requests'] if monitoring_data['total_requests'] > 0 else 0}
"""
    
    # Add model-specific metrics
    model_health = monitoring_data.get('model_health', {})
    for model_id, health in model_health.items():
        if isinstance(health, dict):
            status_value = 1 if health.get('status') == 'healthy' else 0
            metrics += f'codeconductor_model_healthy{{model="{model_id}"}} {status_value}\n'
    
    return metrics, 200, {'Content-Type': 'text/plain'}

@app.route('/ready', methods=['GET'])
def readiness_check():
    """Kubernetes readiness probe endpoint"""
    try:
        # Simple check - just verify we can import our modules
        from ensemble.model_manager import ModelManager
        from ensemble.ensemble_engine import EnsembleEngine
        
        return jsonify({
            'status': 'ready',
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'not_ready',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 503

@app.route('/live', methods=['GET'])
def liveness_check():
    """Kubernetes liveness probe endpoint"""
    return jsonify({
        'status': 'alive',
        'timestamp': datetime.now().isoformat()
    }), 200

if __name__ == '__main__':
    print("🚀 Starting CodeConductor Health API...")
    print("📊 Available endpoints:")
    print("  - GET /health - Main health check")
    print("  - GET /health/models - Model health details")
    print("  - GET /health/ensemble - Ensemble engine health")
    print("  - GET /metrics - Prometheus metrics")
    print("  - GET /ready - Kubernetes readiness probe")
    print("  - GET /live - Kubernetes liveness probe")
    print()
    
    app.run(host='0.0.0.0', port=8080, debug=False) 