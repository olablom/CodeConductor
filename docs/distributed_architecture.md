# 🏗️ Fas 12: Distributed Architecture - CodeConductor

## 📋 Översikt

Skala ut CodeConductor över flera noder med Celery + Redis för verklig enterprise-skalning.

---

## 🏛️ Arkitekturöversikt

### **System Components**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client/API    │    │   Load Balancer │    │   Redis Broker  │
│                 │───▶│                 │───▶│                 │
│ (Streamlit UI)  │    │ (Nginx/Traefik) │    │ (Message Queue) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                                                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Result Store  │    │   Flower        │    │   Celery        │
│                 │◀───│                 │◀───│                 │
│ (Redis Backend) │    │ (Monitoring)    │    │ (Task Queue)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                                                       ▼
                    ┌─────────────────────────────────────────────────┐
                    │              Worker Nodes                       │
                    │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
                    │  │ CodeGen     │ │ Architect   │ │ Review      │ │
                    │  │ Worker      │ │ Worker      │ │ Worker      │ │
                    │  │ (GPU-opt)   │ │ (CPU-opt)   │ │ (CPU-opt)   │ │
                    │  └─────────────┘ └─────────────┘ └─────────────┘ │
                    │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
                    │  │ Policy      │ │ Reward      │ │ Q-Learning  │ │
                    │  │ Worker      │ │ Worker      │ │ Worker      │ │
                    │  │ (CPU-opt)   │ │ (CPU-opt)   │ │ (CPU-opt)   │ │
                    │  └─────────────┘ └─────────────┘ └─────────────┘ │
                    └─────────────────────────────────────────────────┘
```

### **Data Flow**

1. **Client Request** → Load Balancer
2. **Load Balancer** → Redis Broker (task queue)
3. **Redis Broker** → Available Worker Node
4. **Worker Processing** → Agent-specific task execution
5. **Result** → Redis Result Backend
6. **Client Polling** → Result retrieval

---

## 🔧 Celery Configuration

### **`integrations/celery_config.py`**

```python
from celery import Celery
from kombu import Queue
import os

# Redis Configuration
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
REDIS_RESULT_BACKEND = os.getenv('REDIS_RESULT_BACKEND', 'redis://localhost:6379/1')

# Celery App Configuration
celery_app = Celery(
    'codeconductor',
    broker=REDIS_URL,
    backend=REDIS_RESULT_BACKEND,
    include=[
        'agents.celery_tasks',
        'integrations.celery_tasks'
    ]
)

# Task Routing
celery_app.conf.task_routes = {
    'agents.celery_tasks.run_analyze': {'queue': 'analysis'},
    'agents.celery_tasks.run_propose': {'queue': 'proposal'},
    'agents.celery_tasks.run_review': {'queue': 'review'},
    'agents.celery_tasks.run_policy_check': {'queue': 'policy'},
    'agents.celery_tasks.run_reward_calculation': {'queue': 'reward'},
    'agents.celery_tasks.run_q_learning_update': {'queue': 'qlearning'},
}

# Queue Configuration
celery_app.conf.task_queues = (
    Queue('analysis', routing_key='analysis'),
    Queue('proposal', routing_key='proposal'),
    Queue('review', routing_key='review'),
    Queue('policy', routing_key='policy'),
    Queue('reward', routing_key='reward'),
    Queue('qlearning', routing_key='qlearning'),
    Queue('default', routing_key='default'),
)

# Worker Configuration
celery_app.conf.worker_prefetch_multiplier = 1
celery_app.conf.task_acks_late = True
celery_app.conf.worker_max_tasks_per_child = 1000

# Retry Configuration
celery_app.conf.task_default_retry_delay = 60
celery_app.conf.task_max_retries = 3
celery_app.conf.task_soft_time_limit = 300  # 5 minutes
celery_app.conf.task_time_limit = 600       # 10 minutes

# Result Configuration
celery_app.conf.result_expires = 3600  # 1 hour
celery_app.conf.result_persistent = True

# Monitoring
celery_app.conf.worker_send_task_events = True
celery_app.conf.task_send_sent_event = True
```

---

## 🤖 Agent Tasks

### **`agents/celery_tasks.py`**

```python
from celery import current_task
from integrations.celery_config import celery_app
import time
import logging

logger = logging.getLogger(__name__)

# Agent Registry
AGENT_REGISTRY = {
    'codegen': None,      # Lazy loaded
    'architect': None,    # Lazy loaded
    'reviewer': None,     # Lazy loaded
    'policy': None,       # Lazy loaded
    'reward': None,       # Lazy loaded
    'qlearning': None,    # Lazy loaded
}

def get_agent(agent_name):
    """Lazy load agent instances"""
    if AGENT_REGISTRY[agent_name] is None:
        if agent_name == 'codegen':
            from agents.codegen_agent import CodeGenAgent
            AGENT_REGISTRY[agent_name] = CodeGenAgent()
        elif agent_name == 'architect':
            from agents.architect_agent import ArchitectAgent
            AGENT_REGISTRY[agent_name] = ArchitectAgent()
        # ... other agents
    return AGENT_REGISTRY[agent_name]

@celery_app.task(bind=True, name='agents.celery_tasks.run_analyze')
def run_analyze(self, agent_name: str, context: dict):
    """Run agent analysis phase"""
    try:
        logger.info(f"Starting analysis for {agent_name}")
        current_task.update_state(
            state='PROGRESS',
            meta={'agent': agent_name, 'phase': 'analyze', 'status': 'running'}
        )

        agent = get_agent(agent_name)
        result = agent.analyze(context)

        logger.info(f"Analysis completed for {agent_name}")
        return {
            'agent': agent_name,
            'phase': 'analyze',
            'result': result,
            'status': 'completed'
        }

    except Exception as e:
        logger.error(f"Analysis failed for {agent_name}: {e}")
        current_task.update_state(
            state='FAILURE',
            meta={'agent': agent_name, 'phase': 'analyze', 'error': str(e)}
        )
        raise

@celery_app.task(bind=True, name='agents.celery_tasks.run_propose')
def run_propose(self, agent_name: str, context: dict, analysis_results: dict):
    """Run agent proposal phase"""
    try:
        logger.info(f"Starting proposal for {agent_name}")
        current_task.update_state(
            state='PROGRESS',
            meta={'agent': agent_name, 'phase': 'propose', 'status': 'running'}
        )

        agent = get_agent(agent_name)
        result = agent.propose(context, analysis_results)

        logger.info(f"Proposal completed for {agent_name}")
        return {
            'agent': agent_name,
            'phase': 'propose',
            'result': result,
            'status': 'completed'
        }

    except Exception as e:
        logger.error(f"Proposal failed for {agent_name}: {e}")
        current_task.update_state(
            state='FAILURE',
            meta={'agent': agent_name, 'phase': 'propose', 'error': str(e)}
        )
        raise

@celery_app.task(bind=True, name='agents.celery_tasks.run_review')
def run_review(self, agent_name: str, context: dict, code: str):
    """Run agent review phase"""
    try:
        logger.info(f"Starting review for {agent_name}")
        current_task.update_state(
            state='PROGRESS',
            meta={'agent': agent_name, 'phase': 'review', 'status': 'running'}
        )

        agent = get_agent(agent_name)
        result = agent.review(context, code)

        logger.info(f"Review completed for {agent_name}")
        return {
            'agent': agent_name,
            'phase': 'review',
            'result': result,
            'status': 'completed'
        }

    except Exception as e:
        logger.error(f"Review failed for {agent_name}: {e}")
        current_task.update_state(
            state='FAILURE',
            meta={'agent': agent_name, 'phase': 'review', 'error': str(e)}
        )
        raise

@celery_app.task(bind=True, name='agents.celery_tasks.run_pipeline')
def run_pipeline(self, context: dict):
    """Run complete pipeline with distributed agents"""
    try:
        logger.info("Starting distributed pipeline")
        current_task.update_state(
            state='PROGRESS',
            meta={'phase': 'pipeline', 'status': 'initializing'}
        )

        # Phase 1: Parallel Analysis
        analysis_tasks = []
        for agent_name in ['codegen', 'architect', 'reviewer', 'policy']:
            task = run_analyze.delay(agent_name, context)
            analysis_tasks.append(task)

        # Wait for all analysis tasks
        analysis_results = {}
        for i, task in enumerate(analysis_tasks):
            result = task.get(timeout=300)  # 5 minute timeout
            analysis_results[result['agent']] = result['result']

        current_task.update_state(
            state='PROGRESS',
            meta={'phase': 'pipeline', 'status': 'analysis_completed'}
        )

        # Phase 2: Parallel Proposal
        proposal_tasks = []
        for agent_name in ['codegen', 'architect']:
            task = run_propose.delay(agent_name, context, analysis_results)
            proposal_tasks.append(task)

        # Wait for all proposal tasks
        proposal_results = {}
        for task in proposal_tasks:
            result = task.get(timeout=300)
            proposal_results[result['agent']] = result['result']

        current_task.update_state(
            state='PROGRESS',
            meta={'phase': 'pipeline', 'status': 'proposal_completed'}
        )

        # Phase 3: Parallel Review
        review_tasks = []
        for agent_name in ['reviewer', 'policy']:
            for code in proposal_results.get('codegen', {}).get('code_samples', []):
                task = run_review.delay(agent_name, context, code)
                review_tasks.append(task)

        # Wait for all review tasks
        review_results = []
        for task in review_tasks:
            result = task.get(timeout=300)
            review_results.append(result)

        # Compile final result
        final_result = {
            'analysis': analysis_results,
            'proposal': proposal_results,
            'review': review_results,
            'status': 'completed'
        }

        logger.info("Distributed pipeline completed")
        return final_result

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        current_task.update_state(
            state='FAILURE',
            meta={'phase': 'pipeline', 'error': str(e)}
        )
        raise
```

---

## 🐳 Docker Configuration

### **`docker-compose.yml`**

```yaml
version: "3.8"

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

  flower:
    image: mher/flower:1.0
    ports:
      - "5555:5555"
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0
      - CELERY_RESULT_BACKEND=redis://redis:6379/1
    depends_on:
      - redis

  worker-codegen:
    build: .
    command: celery -A integrations.celery_config worker --loglevel=info --queues=analysis,proposal --concurrency=2
    environment:
      - REDIS_URL=redis://redis:6379/0
      - REDIS_RESULT_BACKEND=redis://redis:6379/1
      - GPU_ENABLED=true
    volumes:
      - ./data:/app/data
    depends_on:
      - redis
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  worker-architect:
    build: .
    command: celery -A integrations.celery_config worker --loglevel=info --queues=analysis,proposal --concurrency=2
    environment:
      - REDIS_URL=redis://redis:6379/0
      - REDIS_RESULT_BACKEND=redis://redis:6379/1
    volumes:
      - ./data:/app/data
    depends_on:
      - redis

  worker-review:
    build: .
    command: celery -A integrations.celery_config worker --loglevel=info --queues=review,policy --concurrency=4
    environment:
      - REDIS_URL=redis://redis:6379/0
      - REDIS_RESULT_BACKEND=redis://redis:6379/1
    volumes:
      - ./data:/app/data
    depends_on:
      - redis

  worker-reward:
    build: .
    command: celery -A integrations.celery_config worker --loglevel=info --queues=reward,qlearning --concurrency=2
    environment:
      - REDIS_URL=redis://redis:6379/0
      - REDIS_RESULT_BACKEND=redis://redis:6379/1
    volumes:
      - ./data:/app/data
    depends_on:
      - redis

  api:
    build: .
    command: python -m uvicorn main:app --host 0.0.0.0 --port 8000
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379/0
      - REDIS_RESULT_BACKEND=redis://redis:6379/1
    volumes:
      - ./data:/app/data
    depends_on:
      - redis

  dashboard:
    build: .
    command: streamlit run ui/simple_dashboard.py --server.port 8501 --server.address 0.0.0.0
    ports:
      - "8501:8501"
    environment:
      - REDIS_URL=redis://redis:6379/0
      - REDIS_RESULT_BACKEND=redis://redis:6379/1
    volumes:
      - ./data:/app/data
    depends_on:
      - redis

volumes:
  redis_data:
```

---

## 🚀 Deployment & Scaling

### **Local Development**

```bash
# Start Redis
docker-compose up -d redis

# Start Flower monitoring
docker-compose up -d flower

# Start workers (scale as needed)
docker-compose up -d worker-codegen worker-architect worker-review worker-reward

# Start API and Dashboard
docker-compose up -d api dashboard

# Scale workers
docker-compose up -d --scale worker-codegen=2 --scale worker-review=3
```

### **Production Deployment**

```bash
# Kubernetes deployment
kubectl apply -f k8s/redis.yaml
kubectl apply -f k8s/celery-workers.yaml
kubectl apply -f k8s/api.yaml
kubectl apply -f k8s/dashboard.yaml

# Auto-scaling
kubectl autoscale deployment codeconductor-workers --cpu-percent=70 --min=2 --max=10
```

---

## 📊 Monitoring & Observability

### **Flower Dashboard**

- **URL**: http://localhost:5555
- **Features**: Task monitoring, worker status, queue metrics
- **Alerts**: Failed tasks, worker offline, queue backlog

### **Metrics Collection**

- **Task completion rates**
- **Worker utilization**
- **Queue depths**
- **Response times**
- **Error rates**

### **Health Checks**

```python
@celery_app.task
def health_check():
    """Periodic health check"""
    return {
        'status': 'healthy',
        'timestamp': time.time(),
        'workers': len(celery_app.control.inspect().active()),
        'queues': celery_app.control.inspect().active_queues()
    }
```

---

## 🔄 Fault Tolerance & Recovery

### **Retry Policies**

- **Exponential backoff**: 60s, 120s, 240s
- **Max retries**: 3 attempts
- **Dead letter queue**: Failed tasks after max retries

### **Worker Recovery**

- **Auto-restart**: Failed workers restart automatically
- **Graceful shutdown**: Workers finish current tasks before stopping
- **Load redistribution**: Tasks redistributed when workers go offline

### **Data Persistence**

- **Redis persistence**: AOF (Append Only File)
- **Result storage**: 1-hour TTL with persistence
- **Backup strategy**: Daily Redis snapshots

---

## 🎯 Performance Targets

### **Throughput Goals**

- **Single node**: 10-20 tasks/sec
- **Multi-node**: 50-100 tasks/sec
- **GPU-accelerated**: 100-200 tasks/sec

### **Latency Targets**

- **Analysis phase**: < 5 seconds
- **Proposal phase**: < 10 seconds
- **Review phase**: < 15 seconds
- **Full pipeline**: < 30 seconds

### **Scalability**

- **Horizontal scaling**: Add worker nodes as needed
- **Vertical scaling**: GPU acceleration for code generation
- **Load balancing**: Automatic task distribution

---

## 📋 Implementation Checklist

- [ ] **Celery Configuration** (`integrations/celery_config.py`)
- [ ] **Agent Tasks** (`agents/celery_tasks.py`)
- [ ] **Docker Setup** (`docker-compose.yml`)
- [ ] **Monitoring** (Flower dashboard)
- [ ] **Testing** (Local multi-worker setup)
- [ ] **Documentation** (Deployment guide)
- [ ] **Performance Validation** (Load testing)
- [ ] **Production Deployment** (Kubernetes)

---

## 🚀 Next Steps

1. **Implement Celery configuration**
2. **Create agent tasks**
3. **Set up Docker environment**
4. **Test local scaling**
5. **Deploy to production**
6. **Monitor and optimize**

**Estimated Time**: 4-6 hours for complete implementation
