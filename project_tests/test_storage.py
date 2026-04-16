"""Tests for storage.py — schema, migrations, save/get round-trips."""

from config import ServerConfig
from storage import get_db, save_run, save_result, get_runs, get_results, update_quality


def test_schema_creates_all_columns(tmp_db):
    conn = get_db(tmp_db)
    runs_cols = {row[1] for row in conn.execute("PRAGMA table_info(runs)").fetchall()}
    results_cols = {row[1] for row in conn.execute("PRAGMA table_info(results)").fetchall()}
    conn.close()

    assert "run_id" in runs_cols
    assert "backend" in runs_cols
    assert "hf_repo" in runs_cols
    assert "judge_score" in results_cols
    assert "judge_reason" in results_cols
    assert "finish_reason" in results_cols
    assert "truncated" in results_cols


def test_migrations_are_idempotent(tmp_db):
    conn1 = get_db(tmp_db)
    conn1.close()
    # Second call should not raise
    conn2 = get_db(tmp_db)
    cols = {row[1] for row in conn2.execute("PRAGMA table_info(runs)").fetchall()}
    conn2.close()
    assert "backend" in cols


def test_save_run_and_get_runs(db_conn, sample_config):
    config = sample_config(label="my-run", hf_repo="org/repo", backend="mlx")
    save_run(db_conn, "run1", "2025-01-01T00:00:00Z", config)

    runs = get_runs(db_conn)
    assert len(runs) == 1
    assert runs[0]["run_id"] == "run1"
    assert runs[0]["model_label"] == "my-run"
    assert runs[0]["hf_repo"] == "org/repo"
    assert runs[0]["backend"] == "mlx"


def test_save_result_and_get_results(db_conn, sample_config):
    config = sample_config()
    save_run(db_conn, "run1", "2025-01-01T00:00:00Z", config)

    result = {
        "run_id": "run1",
        "category": "coding",
        "prompt_name": "fizzbuzz",
        "prompt_text": "Write fizzbuzz",
        "response_text": "def fizzbuzz()...",
        "prompt_tokens": 10,
        "completion_tokens": 50,
        "generation_tps": 45.5,
        "prompt_eval_tps": 100.0,
        "time_to_first_token_ms": 200.0,
        "total_time_sec": 1.5,
        "peak_rss_mb": 4096.0,
        "quality_score": 8.5,
        "quality_notes": "good",
        "iteration": 1,
        "finish_reason": "stop",
        "truncated": 0,
    }
    save_result(db_conn, result)

    results = get_results(db_conn, run_ids=["run1"])
    assert len(results) == 1
    assert results[0]["prompt_name"] == "fizzbuzz"
    assert results[0]["generation_tps"] == 45.5
    assert results[0]["finish_reason"] == "stop"


def test_get_results_filtering(db_conn, sample_config):
    config = sample_config()
    save_run(db_conn, "r1", "2025-01-01T00:00:00Z", config)
    save_run(db_conn, "r2", "2025-01-02T00:00:00Z", sample_config(label="other"))

    base = {
        "prompt_text": "test", "response_text": "ok",
        "prompt_tokens": 0, "completion_tokens": 0,
        "generation_tps": 0, "prompt_eval_tps": 0,
        "time_to_first_token_ms": 0, "total_time_sec": 0,
        "peak_rss_mb": 0, "quality_score": 5, "quality_notes": "",
        "iteration": 1, "finish_reason": "stop", "truncated": 0,
    }
    save_result(db_conn, {**base, "run_id": "r1", "category": "coding", "prompt_name": "a"})
    save_result(db_conn, {**base, "run_id": "r1", "category": "math", "prompt_name": "b"})
    save_result(db_conn, {**base, "run_id": "r2", "category": "coding", "prompt_name": "c"})

    assert len(get_results(db_conn, run_ids=["r1"])) == 2
    assert len(get_results(db_conn, run_ids=["r2"])) == 1
    assert len(get_results(db_conn, categories=["coding"])) == 2
    assert len(get_results(db_conn, run_ids=["r1"], categories=["math"])) == 1


def test_update_quality(db_conn, sample_config):
    config = sample_config()
    save_run(db_conn, "r1", "2025-01-01T00:00:00Z", config)
    save_result(db_conn, {
        "run_id": "r1", "category": "coding", "prompt_name": "test",
        "prompt_text": "t", "response_text": "r",
        "prompt_tokens": 0, "completion_tokens": 0,
        "generation_tps": 0, "prompt_eval_tps": 0,
        "time_to_first_token_ms": 0, "total_time_sec": 0,
        "peak_rss_mb": 0, "quality_score": 5, "quality_notes": "",
        "iteration": 1, "finish_reason": "stop", "truncated": 0,
    })

    results = get_results(db_conn)
    update_quality(db_conn, results[0]["id"], 9.0, "great")

    updated = get_results(db_conn)
    assert updated[0]["quality_score"] == 9.0
    assert updated[0]["quality_notes"] == "great"


def test_save_result_with_missing_optional_keys(db_conn, sample_config):
    config = sample_config()
    save_run(db_conn, "r1", "2025-01-01T00:00:00Z", config)

    # Missing finish_reason and truncated — should save as None
    result = {
        "run_id": "r1", "category": "coding", "prompt_name": "test",
        "prompt_text": "t", "response_text": "r",
        "prompt_tokens": 0, "completion_tokens": 0,
        "generation_tps": 0, "prompt_eval_tps": 0,
        "time_to_first_token_ms": 0, "total_time_sec": 0,
        "peak_rss_mb": 0, "quality_score": 5, "quality_notes": "",
        "iteration": 1,
    }
    save_result(db_conn, result)
    results = get_results(db_conn)
    assert len(results) == 1
    assert results[0]["finish_reason"] is None
