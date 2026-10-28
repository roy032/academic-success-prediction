import pytest

from academic_success.data import make_synthetic
from academic_success.train import TrainConfig, run


@pytest.fixture(scope="session")
def synthetic_df():
    return make_synthetic(n=1200, seed=0)


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    """One small end-to-end training run shared by the slower tests."""
    out = tmp_path_factory.mktemp("artifacts")
    cfg = TrainConfig(
        out=str(out),
        synthetic_rows=1200,
        cv_folds=3,
        n_boot=200,
        models=["logreg", "logreg_balanced", "hist_gb_balanced"],
    )
    return out, run(cfg)
