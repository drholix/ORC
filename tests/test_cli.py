from app import cli


def test_cli_run(tmp_path, sample_image):
    output_path = tmp_path / "result.json"
    cli.main(["run", "--input", str(sample_image), "--output", str(output_path)])
    assert output_path.exists()


def test_cli_has_snip_command():
    parser = cli.build_parser()
    subparsers = None
    for action in parser._subparsers._group_actions:  # type: ignore[attr-defined]
        subparsers = action.choices
        break
    assert subparsers is not None
    assert "snip" in subparsers
