from __future__ import annotations

from fastxc.operators.sourcepack.builder import AsyncSourcePackMaterializer


def test_finish_marks_global_success_by_default(tmp_path):
    xcpack_dir = tmp_path / "ncf" / "xcpack"
    xcpack_dir.mkdir(parents=True)
    (xcpack_dir / "2020.001.0000.done").write_text("", encoding="utf-8")

    sourcepack_dir = tmp_path / "sourcepack"
    progress_file = tmp_path / "progress" / "sourcepack_progress.tsv"
    materializer = AsyncSourcePackMaterializer(
        xcpack_dir,
        sourcepack_dir,
        progress_file=progress_file,
    )

    result = materializer.finish()

    assert len(result.index_paths) == 1
    assert result.index_paths[0].is_file()
    assert (sourcepack_dir / "2020.001.0000" / "_SUCCESS").is_file()
    assert (sourcepack_dir / "_SUCCESS").is_file()
    assert "\tDONE\t" in progress_file.read_text(encoding="utf-8")


def test_finish_without_success_keeps_timestamp_indexes_but_not_global_success(tmp_path):
    xcpack_dir = tmp_path / "ncf" / "xcpack"
    xcpack_dir.mkdir(parents=True)
    (xcpack_dir / "2020.001.0000.done").write_text("", encoding="utf-8")

    sourcepack_dir = tmp_path / "sourcepack"
    sourcepack_dir.mkdir()
    (sourcepack_dir / "_SUCCESS").write_text("stale\n", encoding="utf-8")
    progress_file = tmp_path / "progress" / "sourcepack_progress.tsv"
    materializer = AsyncSourcePackMaterializer(
        xcpack_dir,
        sourcepack_dir,
        progress_file=progress_file,
    )

    result = materializer.finish(mark_success=False)

    assert len(result.index_paths) == 1
    assert result.index_paths[0].is_file()
    assert (sourcepack_dir / "2020.001.0000" / "_SUCCESS").is_file()
    assert not (sourcepack_dir / "_SUCCESS").exists()
    assert "\tFAILED\t" in progress_file.read_text(encoding="utf-8")
