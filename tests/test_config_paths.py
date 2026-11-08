from pathlib import Path

from fintech_adoption.config import PipelineConfig, resolve_project_root


def test_resolve_project_root_points_to_workspace():
    root = resolve_project_root()

    assert root.is_absolute()
    assert (root / "src").is_dir()
    assert (root / "data").exists()
    assert (root / "results").exists()


def test_default_output_paths_follow_research_workspace():
    cfg = PipelineConfig()
    root = resolve_project_root()

    assert Path(cfg.output_dir) == root / "data" / "analysis_output"
    assert Path(cfg.log_dir) == root / "results" / "logs"
    assert Path(cfg.embedding.cache_dir) == root / "results" / "models" / "embeddings"
