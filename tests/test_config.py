from pose_controller.config import AppConfig


def test_dev_laptop_config_loads():
    config = AppConfig.from_yaml("configs/dev_laptop.yaml")
    assert config.capture.source == "0"
    assert config.inference.backend == "cpu"
    assert config.overlay.show_window is True


def test_dragon_q6a_config_loads():
    config = AppConfig.from_yaml("configs/dragon_q6a.yaml")
    assert config.inference.backend == "qnn"
    assert config.inference.model_dir == "models/qcs6490"


def test_defaults_when_sections_missing(tmp_path):
    config_path = tmp_path / "minimal.yaml"
    config_path.write_text("capture:\n  source: '1'\n")

    config = AppConfig.from_yaml(config_path)

    assert config.capture.source == "1"
    assert config.inference.backend == "cpu"
    assert config.overlay.window_name == "Pose-Controller"
