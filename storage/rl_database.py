"""
RL Database - SQLite Storage for Reinforcement Learning History

Stores:
- Prompt performance history
- RL episode data
- Learning curves
- Agent interactions
- Human feedback
"""

import sqlite3
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import threading


class RLDatabase:
    """SQLite database for RL history and learning data"""

    def __init__(self, db_path: str = "data/rl_history.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self.lock = threading.Lock()
        self._init_database()

    def _init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Episodes table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS episodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    episode_id TEXT UNIQUE NOT NULL,
                    timestamp TEXT NOT NULL,
                    project_description TEXT,
                    initial_prompt TEXT,
                    optimized_prompt TEXT,
                    final_code TEXT,
                    total_reward REAL,
                    iteration_count INTEGER,
                    execution_time REAL,
                    status TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Reward components table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS reward_components (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    episode_id TEXT NOT NULL,
                    component_name TEXT NOT NULL,
                    reward_value REAL NOT NULL,
                    metadata TEXT,
                    FOREIGN KEY (episode_id) REFERENCES episodes (episode_id)
                )
            """)

            # Test results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS test_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    episode_id TEXT NOT NULL,
                    tests_run INTEGER,
                    tests_passed INTEGER,
                    tests_failed INTEGER,
                    coverage REAL,
                    errors TEXT,
                    warnings TEXT,
                    overall_score REAL,
                    FOREIGN KEY (episode_id) REFERENCES episodes (episode_id)
                )
            """)

            # Human feedback table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS human_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    episode_id TEXT NOT NULL,
                    approved BOOLEAN,
                    reason TEXT,
                    feedback_text TEXT,
                    edited_code TEXT,
                    timestamp TEXT,
                    FOREIGN KEY (episode_id) REFERENCES episodes (episode_id)
                )
            """)

            # Agent interactions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS agent_interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    episode_id TEXT NOT NULL,
                    agent_name TEXT NOT NULL,
                    interaction_type TEXT,
                    confidence REAL,
                    analysis_result TEXT,
                    timestamp TEXT,
                    FOREIGN KEY (episode_id) REFERENCES episodes (episode_id)
                )
            """)

            # Learning metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS learning_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    episode_id TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL,
                    metadata TEXT,
                    timestamp TEXT,
                    FOREIGN KEY (episode_id) REFERENCES episodes (episode_id)
                )
            """)

            # Prompt optimization history
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS prompt_optimizations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    episode_id TEXT NOT NULL,
                    optimization_type TEXT,
                    original_prompt TEXT,
                    optimized_prompt TEXT,
                    confidence REAL,
                    improvement_score REAL,
                    timestamp TEXT,
                    FOREIGN KEY (episode_id) REFERENCES episodes (episode_id)
                )
            """)

            conn.commit()

    def store_episode(self, episode_data: Dict[str, Any]) -> str:
        """Store a complete episode"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Insert episode
                cursor.execute(
                    """
                    INSERT INTO episodes (
                        episode_id, timestamp, project_description, initial_prompt,
                        optimized_prompt, final_code, total_reward, iteration_count,
                        execution_time, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        episode_data["episode_id"],
                        episode_data["timestamp"],
                        episode_data.get("project_description", ""),
                        episode_data.get("initial_prompt", ""),
                        episode_data.get("optimized_prompt", ""),
                        episode_data.get("final_code", ""),
                        episode_data.get("total_reward", 0.0),
                        episode_data.get("iteration_count", 1),
                        episode_data.get("execution_time", 0.0),
                        episode_data.get("status", "completed"),
                    ),
                )

                # Store reward components
                reward_components = episode_data.get("reward_components", {})
                for component_name, reward_value in reward_components.items():
                    cursor.execute(
                        """
                        INSERT INTO reward_components (
                            episode_id, component_name, reward_value, metadata
                        ) VALUES (?, ?, ?, ?)
                    """,
                        (
                            episode_data["episode_id"],
                            component_name,
                            reward_value,
                            json.dumps(episode_data.get("metadata", {})),
                        ),
                    )

                # Store test results
                test_results = episode_data.get("test_results")
                if test_results:
                    cursor.execute(
                        """
                        INSERT INTO test_results (
                            episode_id, tests_run, tests_passed, tests_failed,
                            coverage, errors, warnings, overall_score
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            episode_data["episode_id"],
                            test_results.get("tests_run", 0),
                            test_results.get("tests_passed", 0),
                            test_results.get("tests_failed", 0),
                            test_results.get("coverage", 0.0),
                            json.dumps(test_results.get("errors", [])),
                            json.dumps(test_results.get("warnings", [])),
                            test_results.get("overall_score", 0.0),
                        ),
                    )

                # Store human feedback
                human_feedback = episode_data.get("human_feedback")
                if human_feedback:
                    cursor.execute(
                        """
                        INSERT INTO human_feedback (
                            episode_id, approved, reason, feedback_text, edited_code, timestamp
                        ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                        (
                            episode_data["episode_id"],
                            human_feedback.get("approved", False),
                            human_feedback.get("reason", ""),
                            human_feedback.get("feedback", ""),
                            human_feedback.get("edited_code", ""),
                            human_feedback.get("timestamp", ""),
                        ),
                    )

                # Store agent interactions
                agent_interactions = episode_data.get("agent_interactions", [])
                for interaction in agent_interactions:
                    cursor.execute(
                        """
                        INSERT INTO agent_interactions (
                            episode_id, agent_name, interaction_type, confidence,
                            analysis_result, timestamp
                        ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                        (
                            episode_data["episode_id"],
                            interaction.get("agent_name", ""),
                            interaction.get("interaction_type", ""),
                            interaction.get("confidence", 0.0),
                            json.dumps(interaction.get("analysis_result", {})),
                            interaction.get("timestamp", ""),
                        ),
                    )

                # Store learning metrics
                learning_metrics = episode_data.get("learning_metrics", {})
                for metric_name, metric_value in learning_metrics.items():
                    cursor.execute(
                        """
                        INSERT INTO learning_metrics (
                            episode_id, metric_name, metric_value, metadata, timestamp
                        ) VALUES (?, ?, ?, ?, ?)
                    """,
                        (
                            episode_data["episode_id"],
                            metric_name,
                            metric_value,
                            json.dumps(episode_data.get("metadata", {})),
                            episode_data.get("timestamp", ""),
                        ),
                    )

                # Store prompt optimizations
                prompt_optimizations = episode_data.get("prompt_optimizations", [])
                for optimization in prompt_optimizations:
                    cursor.execute(
                        """
                        INSERT INTO prompt_optimizations (
                            episode_id, optimization_type, original_prompt,
                            optimized_prompt, confidence, improvement_score, timestamp
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            episode_data["episode_id"],
                            optimization.get("type", ""),
                            optimization.get("original_prompt", ""),
                            optimization.get("optimized_prompt", ""),
                            optimization.get("confidence", 0.0),
                            optimization.get("improvement_score", 0.0),
                            optimization.get("timestamp", ""),
                        ),
                    )

                conn.commit()
                return episode_data["episode_id"]

    def store_human_feedback(
        self,
        episode_id: str,
        approved: bool,
        feedback_text: str = "",
        feedback_score: int = 0,
        comment: str = "",
    ) -> bool:
        """Store detailed human feedback with thumbs up/down"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT INTO human_feedback (
                        episode_id, approved, reason, feedback_text, edited_code, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        episode_id,
                        approved,
                        f"Thumbs {'up' if approved else 'down'} - Score: {feedback_score}",
                        feedback_text,
                        comment,  # Store comment in edited_code field for now
                        datetime.now().isoformat(),
                    ),
                )
                conn.commit()
                return True

    def get_feedback_statistics(self) -> Dict[str, Any]:
        """Get feedback statistics for dashboard"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Total feedback count
            cursor.execute("SELECT COUNT(*) FROM human_feedback")
            total_feedback = cursor.fetchone()[0]

            # Positive vs negative
            cursor.execute("SELECT COUNT(*) FROM human_feedback WHERE approved = 1")
            positive_feedback = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM human_feedback WHERE approved = 0")
            negative_feedback = cursor.fetchone()[0]

            # Average comment length
            cursor.execute("""
                SELECT AVG(LENGTH(feedback_text)) 
                FROM human_feedback 
                WHERE feedback_text IS NOT NULL AND feedback_text != ''
            """)
            avg_comment_length = cursor.fetchone()[0] or 0

            # Recent feedback (last 10)
            cursor.execute("""
                SELECT approved, feedback_text, timestamp 
                FROM human_feedback 
                ORDER BY timestamp DESC 
                LIMIT 10
            """)
            recent_feedback = [
                {"approved": bool(row[0]), "feedback_text": row[1], "timestamp": row[2]}
                for row in cursor.fetchall()
            ]

            return {
                "total_feedback": total_feedback,
                "positive_feedback": positive_feedback,
                "negative_feedback": negative_feedback,
                "approval_rate": (positive_feedback / total_feedback * 100)
                if total_feedback > 0
                else 0,
                "avg_comment_length": round(avg_comment_length, 1),
                "recent_feedback": recent_feedback,
            }

    def get_episode(self, episode_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a complete episode"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Get episode data
            cursor.execute("SELECT * FROM episodes WHERE episode_id = ?", (episode_id,))
            episode_row = cursor.fetchone()

            if not episode_row:
                return None

            # Get column names
            columns = [description[0] for description in cursor.description]
            episode_data = dict(zip(columns, episode_row))

            # Get reward components
            cursor.execute(
                "SELECT * FROM reward_components WHERE episode_id = ?", (episode_id,)
            )
            reward_components = {}
            for row in cursor.fetchall():
                reward_components[row[2]] = row[3]  # component_name -> reward_value

            episode_data["reward_components"] = reward_components

            # Get test results
            cursor.execute(
                "SELECT * FROM test_results WHERE episode_id = ?", (episode_id,)
            )
            test_row = cursor.fetchone()
            if test_row:
                test_columns = [description[0] for description in cursor.description]
                episode_data["test_results"] = dict(zip(test_columns, test_row))

            # Get human feedback
            cursor.execute(
                "SELECT * FROM human_feedback WHERE episode_id = ?", (episode_id,)
            )
            feedback_row = cursor.fetchone()
            if feedback_row:
                feedback_columns = [
                    description[0] for description in cursor.description
                ]
                episode_data["human_feedback"] = dict(
                    zip(feedback_columns, feedback_row)
                )

            return episode_data

    def get_learning_curve(self, window_size: int = 10) -> List[Dict[str, Any]]:
        """Get learning curve data"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT episode_id, total_reward, timestamp
                FROM episodes
                ORDER BY timestamp
            """)

            episodes = cursor.fetchall()

            if len(episodes) < window_size:
                return []

            learning_curve = []
            for i in range(window_size, len(episodes)):
                window_rewards = [ep[1] for ep in episodes[i - window_size : i]]
                avg_reward = sum(window_rewards) / len(window_rewards)

                learning_curve.append(
                    {
                        "episode": i,
                        "episode_id": episodes[i][0],
                        "average_reward": avg_reward,
                        "window_size": window_size,
                        "timestamp": episodes[i][2],
                    }
                )

            return learning_curve

    def get_reward_statistics(self) -> Dict[str, Any]:
        """Get comprehensive reward statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT 
                    COUNT(*) as total_episodes,
                    AVG(total_reward) as average_reward,
                    MAX(total_reward) as best_reward,
                    MIN(total_reward) as worst_reward,
                    AVG(iteration_count) as avg_iterations,
                    AVG(execution_time) as avg_execution_time
                FROM episodes
            """)

            row = cursor.fetchone()
            if not row:
                return {
                    "total_episodes": 0,
                    "average_reward": 0.0,
                    "best_reward": 0.0,
                    "worst_reward": 0.0,
                    "avg_iterations": 0.0,
                    "avg_execution_time": 0.0,
                }

            return {
                "total_episodes": row[0],
                "average_reward": row[1] or 0.0,
                "best_reward": row[2] or 0.0,
                "worst_reward": row[3] or 0.0,
                "avg_iterations": row[4] or 0.0,
                "avg_execution_time": row[5] or 0.0,
            }

    def get_component_analysis(self) -> Dict[str, Any]:
        """Analyze reward components"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT 
                    component_name,
                    AVG(reward_value) as avg_reward,
                    COUNT(*) as count
                FROM reward_components
                GROUP BY component_name
            """)

            components = {}
            for row in cursor.fetchall():
                components[row[0]] = {"average_reward": row[1] or 0.0, "count": row[2]}

            return components

    def get_recent_episodes(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent episodes"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT episode_id, total_reward, iteration_count, status, timestamp
                FROM episodes
                ORDER BY timestamp DESC
                LIMIT ?
            """,
                (limit,),
            )

            episodes = []
            for row in cursor.fetchall():
                episodes.append(
                    {
                        "episode_id": row[0],
                        "total_reward": row[1],
                        "iteration_count": row[2],
                        "status": row[3],
                        "timestamp": row[4],
                    }
                )

            return episodes

    def get_episode_count(self) -> int:
        """Get total number of episodes"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM episodes")
            return cursor.fetchone()[0]

    def export_data(self, file_path: str):
        """Export all data to JSON"""
        data = {
            "episodes": [],
            "statistics": self.get_reward_statistics(),
            "component_analysis": self.get_component_analysis(),
            "learning_curve": self.get_learning_curve(),
            "export_timestamp": datetime.now().isoformat(),
        }

        # Get all episodes
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT episode_id FROM episodes ORDER BY timestamp")

            for row in cursor.fetchall():
                episode = self.get_episode(row[0])
                if episode:
                    data["episodes"].append(episode)

        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

    def clear_data(self):
        """Clear all data (for testing)"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM reward_components")
                cursor.execute("DELETE FROM test_results")
                cursor.execute("DELETE FROM human_feedback")
                cursor.execute("DELETE FROM agent_interactions")
                cursor.execute("DELETE FROM learning_metrics")
                cursor.execute("DELETE FROM prompt_optimizations")
                cursor.execute("DELETE FROM episodes")
                conn.commit()

    def get_database_info(self) -> Dict[str, Any]:
        """Get database information"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Get table sizes
            tables = [
                "episodes",
                "reward_components",
                "test_results",
                "human_feedback",
                "agent_interactions",
                "learning_metrics",
                "prompt_optimizations",
            ]

            table_sizes = {}
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                table_sizes[table] = cursor.fetchone()[0]

            # Get database size
            db_size = self.db_path.stat().st_size if self.db_path.exists() else 0

            return {
                "database_path": str(self.db_path),
                "database_size_bytes": db_size,
                "table_sizes": table_sizes,
                "total_episodes": table_sizes.get("episodes", 0),
            }
