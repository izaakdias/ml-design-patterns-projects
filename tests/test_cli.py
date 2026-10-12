import json

import pytest

from ml_patterns_lab.__main__ import main


def test_cli_lists_real_versions(trained_artifacts, capsys):
    main(["list", "churn"])
    result = json.loads(capsys.readouterr().out)
    assert len(result) == 1
    assert result[0]["current"]
    assert result[0]["run_id"] == trained_artifacts["churn"][1]["run_id"]


@pytest.mark.parametrize("args", [["train", "missing"], ["serve", "churn", "--port", "0"]])
def test_cli_rejects_invalid_commands(args):
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2
