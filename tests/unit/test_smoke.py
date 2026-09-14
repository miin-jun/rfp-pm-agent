from rfp_pm_agent import main


def test_main_is_callable() -> None:
    assert callable(main)


def test_intentional_fail() -> None:
    assert False
