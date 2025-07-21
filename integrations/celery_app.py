"""
Celery application for distributed task execution in CodeConductor
"""

from celery import Celery
from omegaconf import OmegaConf
import logging

# Load configuration
cfg = OmegaConf.load("config/base.yaml")

# Celery configuration
broker_url = cfg.distributed.broker_url
result_backend = cfg.distributed.result_backend

# Create Celery app
celery_app = Celery(
    "codeconductor",
    broker=broker_url,
    backend=result_backend,
    include=["integrations.celery_tasks", "agents.celery_agents"],
)

# Configure Celery
celery_app.conf.update(
    task_serializer=cfg.celery.task_serializer,
    result_serializer=cfg.celery.result_serializer,
    accept_content=cfg.celery.accept_content,
    timezone=cfg.celery.timezone,
    enable_utc=cfg.celery.enable_utc,
    task_track_started=cfg.celery.task_track_started,
    task_time_limit=cfg.celery.task_time_limit,
    task_soft_time_limit=cfg.celery.task_soft_time_limit,
    worker_concurrency=cfg.distributed.worker_concurrency,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_always_eager=False,  # Set to True for testing without workers
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@celery_app.task(bind=True)
def debug_task(self):
    """Debug task to test Celery setup"""
    logger.info(f"Request: {self.request!r}")
    return "Celery is working!"


def get_celery_stats():
    """Get Celery statistics"""
    try:
        inspect = celery_app.control.inspect()
        stats = inspect.stats()
        active_tasks = inspect.active()
        reserved_tasks = inspect.reserved()

        return {
            "stats": stats,
            "active_tasks": active_tasks,
            "reserved_tasks": reserved_tasks,
            "broker_url": broker_url,
            "result_backend": result_backend,
        }
    except Exception as e:
        logger.error(f"Failed to get Celery stats: {e}")
        return {"error": str(e)}


def is_celery_available():
    """Check if Celery broker is available"""
    try:
        from redis import Redis

        redis_client = Redis.from_url(broker_url)
        redis_client.ping()
        return True
    except Exception as e:
        logger.error(f"Celery broker not available: {e}")
        return False


if __name__ == "__main__":
    # Test Celery setup
    if is_celery_available():
        print("✅ Celery broker is available")
        result = debug_task.delay()
        print(f"Debug task result: {result.get()}")
    else:
        print("❌ Celery broker is not available")
        print("Please start Redis: redis-server")
