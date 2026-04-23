"""Artifact Protocol 测试.

运行: uv run pytest tests/test_artifacts.py -xvs
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mini_code_agent.artifacts import (
    ArtifactBuilder,
    ArtifactBuilderError,
    ArtifactDecision,
    ArtifactMeta,
    ArtifactStatus,
    ArtifactStore,
    CheckResult,
    Confidence,
    EditOperation,
    FileEdit,
    Patch,
    ResourceUsage,
    ScopeCheck,
    ScopeChecker,
    SelfVerification,
    SubtaskArtifact,
    generate_unified_diff,
)


# ============================================================
# 辅助工厂函数
# ============================================================

def _make_check_result(
    name: str = "syntax",
    passed: bool = True,
    skipped: bool = False,
    items_checked: int = 1,
    items_failed: int = 0,
) -> CheckResult:
    return CheckResult(
        check_name=name,
        passed=passed,
        skipped=skipped,
        skip_reason="no files" if skipped else None,
        duration_seconds=0.1,
        details="ok" if passed else "error: something went wrong",
        items_checked=items_checked,
        items_failed=items_failed,
    )


def _make_verification(overall_passed: bool = True) -> SelfVerification:
    return SelfVerification(
        syntax_check=_make_check_result("syntax", passed=overall_passed),
        lint_check=None,
        type_check=None,
        unit_test=_make_check_result("unit_test", passed=overall_passed, items_checked=3),
        import_check=_make_check_result("import", passed=True),
        overall_passed=overall_passed,
    )


def _make_resource() -> ResourceUsage:
    return ResourceUsage(
        tokens_input=1000,
        tokens_output=500,
        tokens_total=1500,
        llm_calls=2,
        tool_calls=5,
        wall_time_seconds=10.5,
        model_used="deepseek-chat",
    )


def _make_patch(
    edits: list[FileEdit] | None = None,
    base_hash: str = "abc123",
) -> Patch:
    if edits is None:
        edits = [
            FileEdit(
                path="src/foo.py",
                operation=EditOperation.CREATE,
                old_content=None,
                new_content="print('hello')\n",
                old_path=None,
                unified_diff="--- /dev/null\n+++ b/src/foo.py\n@@ -0,0 +1 @@\n+print('hello')\n",
                lines_added=1,
                lines_removed=0,
            )
        ]
    total_added = sum(e.lines_added for e in edits)
    total_removed = sum(e.lines_removed for e in edits)
    return Patch(
        edits=edits,
        total_files_changed=len(edits),
        total_lines_added=total_added,
        total_lines_removed=total_removed,
        base_git_hash=base_hash,
    )


def _make_artifact(
    confidence: Confidence = Confidence.DONE,
    status: ArtifactStatus = ArtifactStatus.DRAFT,
    patch: Patch | None = None,
    verification: SelfVerification | None = None,
    open_questions: list[str] | None = None,
) -> SubtaskArtifact:
    return SubtaskArtifact(
        artifact_id="artifact-001",
        task_id="task-001",
        created_at=datetime(2025, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
        producer="worker-001",
        task_description="实现 foo 模块",
        allowed_paths=["src/**"],
        verification_spec="syntax + unit_test",
        patch=patch or _make_patch(),
        self_verification=verification or _make_verification(),
        scope_check=ScopeCheck(
            allowed_paths=["src/**"],
            touched_paths=["src/foo.py"],
            out_of_scope_paths=[],
            is_clean=True,
        ),
        decisions=[
            ArtifactDecision(
                description="使用 dataclass",
                reason="项目规范要求",
                alternatives_considered=["Pydantic"],
                reversible=True,
                step_number=1,
            )
        ],
        resource_usage=_make_resource(),
        confidence=confidence,
        self_summary="创建了 foo 模块，包含基本功能",
        open_questions=open_questions or [],
        status=status,
    )


# ============================================================
# Patch 测试
# ============================================================

class TestEditOperation:
    def test_enum_values(self):
        assert EditOperation.CREATE.value == "CREATE"
        assert EditOperation.MODIFY.value == "MODIFY"
        assert EditOperation.DELETE.value == "DELETE"
        assert EditOperation.RENAME.value == "RENAME"

    def test_str_serialization(self):
        """枚举序列化为字符串，不是整数."""
        assert str(EditOperation.CREATE) == "EditOperation.CREATE"
        assert EditOperation("CREATE") == EditOperation.CREATE


class TestPatch:
    def test_empty_patch(self):
        patch = _make_patch(edits=[])
        assert patch.is_empty()
        assert patch.to_unified_diff() == ""

    def test_non_empty_patch(self):
        patch = _make_patch()
        assert not patch.is_empty()
        assert "src/foo.py" in patch.to_unified_diff()

    def test_multiple_edits_combined_diff(self):
        edits = [
            FileEdit(
                path="a.py",
                operation=EditOperation.CREATE,
                old_content=None,
                new_content="# a\n",
                old_path=None,
                unified_diff="--- /dev/null\n+++ b/a.py\n",
                lines_added=1,
                lines_removed=0,
            ),
            FileEdit(
                path="b.py",
                operation=EditOperation.CREATE,
                old_content=None,
                new_content="# b\n",
                old_path=None,
                unified_diff="--- /dev/null\n+++ b/b.py\n",
                lines_added=1,
                lines_removed=0,
            ),
        ]
        patch = _make_patch(edits=edits)
        diff = patch.to_unified_diff()
        assert "a.py" in diff
        assert "b.py" in diff

    def test_frozen(self):
        """Patch 是 frozen dataclass."""
        patch = _make_patch()
        with pytest.raises(AttributeError):
            patch.base_git_hash = "new_hash"  # type: ignore[misc]


class TestPatchApply:
    def test_create_file(self, tmp_path: Path):
        edit = FileEdit(
            path="new_file.py",
            operation=EditOperation.CREATE,
            old_content=None,
            new_content="print('new')\n",
            old_path=None,
            unified_diff="",
            lines_added=1,
            lines_removed=0,
        )
        patch = _make_patch(edits=[edit])
        result = patch.apply_to(str(tmp_path))

        assert result.success
        assert "new_file.py" in result.applied_files
        assert (tmp_path / "new_file.py").read_text() == "print('new')\n"

    def test_create_with_nested_dirs(self, tmp_path: Path):
        edit = FileEdit(
            path="src/deep/new_file.py",
            operation=EditOperation.CREATE,
            old_content=None,
            new_content="# deep\n",
            old_path=None,
            unified_diff="",
            lines_added=1,
            lines_removed=0,
        )
        patch = _make_patch(edits=[edit])
        result = patch.apply_to(str(tmp_path))
        assert result.success
        assert (tmp_path / "src" / "deep" / "new_file.py").exists()

    def test_create_existing_file_fails(self, tmp_path: Path):
        (tmp_path / "existing.py").write_text("old")
        edit = FileEdit(
            path="existing.py",
            operation=EditOperation.CREATE,
            old_content=None,
            new_content="new",
            old_path=None,
            unified_diff="",
            lines_added=1,
            lines_removed=0,
        )
        patch = _make_patch(edits=[edit])
        result = patch.apply_to(str(tmp_path))
        assert not result.success
        assert "existing.py" in result.failed_files

    def test_modify_file(self, tmp_path: Path):
        (tmp_path / "mod.py").write_text("old content")
        edit = FileEdit(
            path="mod.py",
            operation=EditOperation.MODIFY,
            old_content="old content",
            new_content="new content",
            old_path=None,
            unified_diff="",
            lines_added=1,
            lines_removed=1,
        )
        patch = _make_patch(edits=[edit])
        result = patch.apply_to(str(tmp_path))
        assert result.success
        assert (tmp_path / "mod.py").read_text() == "new content"

    def test_modify_conflict(self, tmp_path: Path):
        (tmp_path / "mod.py").write_text("current content")
        edit = FileEdit(
            path="mod.py",
            operation=EditOperation.MODIFY,
            old_content="expected old content",
            new_content="new content",
            old_path=None,
            unified_diff="",
            lines_added=1,
            lines_removed=1,
        )
        patch = _make_patch(edits=[edit])
        result = patch.apply_to(str(tmp_path))
        assert not result.success
        assert "冲突" in result.errors[0]

    def test_modify_nonexistent_fails(self, tmp_path: Path):
        edit = FileEdit(
            path="missing.py",
            operation=EditOperation.MODIFY,
            old_content="old",
            new_content="new",
            old_path=None,
            unified_diff="",
            lines_added=1,
            lines_removed=1,
        )
        patch = _make_patch(edits=[edit])
        result = patch.apply_to(str(tmp_path))
        assert not result.success

    def test_delete_file(self, tmp_path: Path):
        (tmp_path / "to_delete.py").write_text("bye")
        edit = FileEdit(
            path="to_delete.py",
            operation=EditOperation.DELETE,
            old_content="bye",
            new_content=None,
            old_path=None,
            unified_diff="",
            lines_added=0,
            lines_removed=1,
        )
        patch = _make_patch(edits=[edit])
        result = patch.apply_to(str(tmp_path))
        assert result.success
        assert not (tmp_path / "to_delete.py").exists()

    def test_delete_nonexistent_fails(self, tmp_path: Path):
        edit = FileEdit(
            path="no_such.py",
            operation=EditOperation.DELETE,
            old_content="content",
            new_content=None,
            old_path=None,
            unified_diff="",
            lines_added=0,
            lines_removed=1,
        )
        patch = _make_patch(edits=[edit])
        result = patch.apply_to(str(tmp_path))
        assert not result.success

    def test_rename_file(self, tmp_path: Path):
        (tmp_path / "old_name.py").write_text("content")
        edit = FileEdit(
            path="new_name.py",
            operation=EditOperation.RENAME,
            old_content="content",
            new_content="content",
            old_path="old_name.py",
            unified_diff="",
            lines_added=0,
            lines_removed=0,
        )
        patch = _make_patch(edits=[edit])
        result = patch.apply_to(str(tmp_path))
        assert result.success
        assert not (tmp_path / "old_name.py").exists()
        assert (tmp_path / "new_name.py").exists()

    def test_delete_only_patch(self, tmp_path: Path):
        """只有 DELETE 操作的 Patch."""
        (tmp_path / "a.py").write_text("a")
        (tmp_path / "b.py").write_text("b")
        edits = [
            FileEdit(
                path="a.py",
                operation=EditOperation.DELETE,
                old_content="a",
                new_content=None,
                old_path=None,
                unified_diff="",
                lines_added=0,
                lines_removed=1,
            ),
            FileEdit(
                path="b.py",
                operation=EditOperation.DELETE,
                old_content="b",
                new_content=None,
                old_path=None,
                unified_diff="",
                lines_added=0,
                lines_removed=1,
            ),
        ]
        patch = _make_patch(edits=edits)
        result = patch.apply_to(str(tmp_path))
        assert result.success
        assert len(result.applied_files) == 2


class TestGenerateUnifiedDiff:
    def test_create_diff(self):
        diff = generate_unified_diff("new.py", EditOperation.CREATE, None, "line1\nline2\n")
        assert "+++ b/new.py" in diff
        assert "--- /dev/null" in diff

    def test_delete_diff(self):
        diff = generate_unified_diff("old.py", EditOperation.DELETE, "line1\n", None)
        assert "--- a/old.py" in diff
        assert "+++ /dev/null" in diff

    def test_modify_diff(self):
        diff = generate_unified_diff("mod.py", EditOperation.MODIFY, "old\n", "new\n")
        assert "--- a/mod.py" in diff
        assert "+++ b/mod.py" in diff

    def test_rename_diff(self):
        diff = generate_unified_diff("new.py", EditOperation.RENAME, "content\n", "content\n", old_path="old.py")
        assert "--- a/old.py" in diff
        assert "+++ b/new.py" in diff


# ============================================================
# Verification 测试
# ============================================================

class TestCheckResult:
    def test_basic_fields(self):
        cr = _make_check_result("syntax", passed=True, items_checked=5)
        assert cr.check_name == "syntax"
        assert cr.passed
        assert cr.items_checked == 5


class TestSelfVerification:
    def test_summary_all_passed(self):
        sv = _make_verification(overall_passed=True)
        s = sv.summary()
        assert "✅" in s
        assert "syntax" in s
        assert "3 test" in s
        assert "import" in s

    def test_summary_with_failures(self):
        sv = SelfVerification(
            syntax_check=_make_check_result("syntax", passed=False, items_failed=1),
            lint_check=_make_check_result("lint", passed=False, items_failed=2),
            type_check=None,
            unit_test=_make_check_result("unit_test", skipped=True, passed=False),
            import_check=_make_check_result("import", passed=True),
            overall_passed=False,
        )
        s = sv.summary()
        assert "❌" in s
        assert "⏭️" in s

    def test_summary_all_none_optional(self):
        sv = SelfVerification(
            syntax_check=_make_check_result("syntax"),
            lint_check=None,
            type_check=None,
            unit_test=None,
            import_check=_make_check_result("import"),
            overall_passed=True,
        )
        s = sv.summary()
        # Should not mention lint/type/test
        assert "lint" not in s
        assert "type" not in s


# ============================================================
# ScopeChecker 测试
# ============================================================

class TestScopeChecker:
    def test_exact_match(self):
        result = ScopeChecker.check(["src/foo.py"], ["src/foo.py"])
        assert result.is_clean
        assert result.out_of_scope_paths == []

    def test_glob_star_star(self):
        """src/auth/** 匹配 src/auth/ 下任意深度的文件."""
        result = ScopeChecker.check(
            ["src/auth/**"],
            ["src/auth/login.py", "src/auth/utils/hash.py"],
        )
        assert result.is_clean

    def test_glob_star_star_does_not_match_sibling(self):
        result = ScopeChecker.check(
            ["src/auth/**"],
            ["src/models/user.py"],
        )
        assert not result.is_clean
        assert "src/models/user.py" in result.out_of_scope_paths

    def test_glob_star_py(self):
        """*.py 匹配当前目录下的 .py 文件."""
        result = ScopeChecker.check(
            ["*.py"],
            ["main.py", "setup.py"],
        )
        assert result.is_clean

    def test_glob_star_py_matches_nested(self):
        """*.py 通过 fnmatch 也能匹配嵌套目录（fnmatch 不把 / 当特殊字符）."""
        result = ScopeChecker.check(
            ["*.py"],
            ["src/foo.py"],
        )
        assert result.is_clean

    def test_mixed_patterns(self):
        result = ScopeChecker.check(
            ["src/auth/**", "README.md", "*.txt"],
            ["src/auth/login.py", "README.md", "notes.txt", "src/other.py"],
        )
        assert not result.is_clean
        assert result.out_of_scope_paths == ["src/other.py"]

    def test_empty_touched(self):
        result = ScopeChecker.check(["src/**"], [])
        assert result.is_clean

    def test_empty_allowed(self):
        result = ScopeChecker.check([], ["any_file.py"])
        assert not result.is_clean

    def test_prefix_directory_match(self):
        """src/auth/** 应该匹配 src/auth 本身."""
        result = ScopeChecker.check(["src/auth/**"], ["src/auth"])
        assert result.is_clean


# ============================================================
# Decision 测试
# ============================================================

class TestArtifactDecision:
    def test_basic_construction(self):
        d = ArtifactDecision(
            description="使用 dataclass",
            reason="项目规范",
            alternatives_considered=["Pydantic", "TypedDict"],
            reversible=True,
            step_number=1,
        )
        assert d.description == "使用 dataclass"
        assert len(d.alternatives_considered) == 2

    def test_defaults(self):
        d = ArtifactDecision(description="foo", reason="bar")
        assert d.alternatives_considered == []
        assert d.reversible is True
        assert d.step_number == 0


# ============================================================
# ResourceUsage 测试
# ============================================================

class TestResourceUsage:
    def test_basic_fields(self):
        r = _make_resource()
        assert r.tokens_total == 1500
        assert r.model_used == "deepseek-chat"


# ============================================================
# SubtaskArtifact 序列化 / 反序列化 roundtrip
# ============================================================

class TestSubtaskArtifactSerialization:
    def test_roundtrip(self):
        """序列化后反序列化，内容一致."""
        original = _make_artifact()
        data = original.to_json()
        restored = SubtaskArtifact.from_json(data)

        assert restored.artifact_id == original.artifact_id
        assert restored.task_id == original.task_id
        assert restored.created_at == original.created_at
        assert restored.producer == original.producer
        assert restored.confidence == original.confidence
        assert restored.status == original.status
        assert restored.self_summary == original.self_summary
        assert restored.open_questions == original.open_questions
        assert restored.patch.total_files_changed == original.patch.total_files_changed
        assert restored.patch.base_git_hash == original.patch.base_git_hash
        assert len(restored.decisions) == len(original.decisions)
        assert restored.decisions[0].description == original.decisions[0].description
        assert restored.resource_usage.tokens_total == original.resource_usage.tokens_total

    def test_datetime_iso_format(self):
        """datetime 序列化为 ISO 8601 字符串."""
        artifact = _make_artifact()
        data = artifact.to_json()
        assert isinstance(data["created_at"], str)
        assert "2025-01-15T10:30:00" in data["created_at"]

    def test_enums_as_strings(self):
        """enum 序列化为字符串值，不是整数."""
        artifact = _make_artifact()
        data = artifact.to_json()
        assert data["confidence"] == "DONE"
        assert data["status"] == "DRAFT"
        for edit in data["patch"]["edits"]:
            assert isinstance(edit["operation"], str)
            assert edit["operation"] in ["CREATE", "MODIFY", "DELETE", "RENAME"]

    def test_json_roundtrip_via_string(self):
        """经过 json.dumps/loads 后仍可反序列化."""
        original = _make_artifact()
        json_str = json.dumps(original.to_json(), ensure_ascii=False)
        data = json.loads(json_str)
        restored = SubtaskArtifact.from_json(data)
        assert restored.artifact_id == original.artifact_id

    def test_save_and_load(self, tmp_path: Path):
        """save 到磁盘后 load 回来."""
        original = _make_artifact()
        path = str(tmp_path / "test_artifact.json")
        original.save(path)

        loaded = SubtaskArtifact.load(path)
        assert loaded.artifact_id == original.artifact_id
        assert loaded.created_at == original.created_at
        assert loaded.patch.total_files_changed == original.patch.total_files_changed

    def test_frozen(self):
        """SubtaskArtifact 是 frozen dataclass."""
        artifact = _make_artifact()
        with pytest.raises(AttributeError):
            artifact.status = ArtifactStatus.APPROVED  # type: ignore[misc]


# ============================================================
# SubtaskArtifact 方法测试
# ============================================================

class TestSubtaskArtifactMethods:
    def test_summary_for_reviewer(self):
        artifact = _make_artifact()
        summary = artifact.summary_for_reviewer()
        assert "artifact-001" in summary
        assert "task-001" in summary
        assert "DRAFT" in summary
        assert "DONE" in summary

    def test_summary_for_reviewer_with_open_questions(self):
        artifact = _make_artifact(open_questions=["要不要加缓存？"])
        summary = artifact.summary_for_reviewer()
        assert "要不要加缓存" in summary

    def test_summary_for_ledger(self):
        artifact = _make_artifact()
        ledger = artifact.summary_for_ledger()
        assert ledger["artifact_id"] == "artifact-001"
        assert ledger["task_id"] == "task-001"
        assert ledger["status"] == "DRAFT"
        assert ledger["confidence"] == "DONE"
        assert ledger["scope_clean"] is True

    def test_diff_preview_short(self):
        artifact = _make_artifact()
        preview = artifact.diff_preview(max_lines=100)
        # 短 diff 不会被截断
        assert "more lines" not in preview

    def test_diff_preview_truncated(self):
        # 构造一个有很多行的 diff
        content = "\n".join(f"line {i}" for i in range(100))
        edit = FileEdit(
            path="big.py",
            operation=EditOperation.CREATE,
            old_content=None,
            new_content=content,
            old_path=None,
            unified_diff="\n".join(f"+line {i}" for i in range(100)),
            lines_added=100,
            lines_removed=0,
        )
        patch = _make_patch(edits=[edit])
        artifact = _make_artifact(patch=patch)
        preview = artifact.diff_preview(max_lines=10)
        assert "more lines" in preview

    def test_diff_preview_empty_patch(self):
        patch = _make_patch(edits=[])
        artifact = _make_artifact(patch=patch)
        assert artifact.diff_preview() == "(no changes)"

    def test_summary_for_reviewer_out_of_scope(self):
        artifact = SubtaskArtifact(
            artifact_id="a-002",
            task_id="t-002",
            created_at=datetime(2025, 1, 15, tzinfo=timezone.utc),
            producer="worker-001",
            task_description="test",
            allowed_paths=["src/**"],
            verification_spec="syntax",
            patch=_make_patch(),
            self_verification=_make_verification(),
            scope_check=ScopeCheck(
                allowed_paths=["src/**"],
                touched_paths=["src/foo.py", "config.yaml"],
                out_of_scope_paths=["config.yaml"],
                is_clean=False,
            ),
            decisions=[],
            resource_usage=_make_resource(),
            confidence=Confidence.DONE,
            self_summary="test",
            open_questions=[],
            status=ArtifactStatus.DRAFT,
        )
        summary = artifact.summary_for_reviewer()
        assert "❌" in summary
        assert "config.yaml" in summary


# ============================================================
# Confidence / Status 约束测试
# ============================================================

class TestConfidenceConstraints:
    def test_stuck_must_have_open_questions(self):
        """Confidence = STUCK 时应有 open_questions."""
        artifact = _make_artifact(
            confidence=Confidence.STUCK,
            open_questions=["怎么解决循环依赖？"],
            verification=_make_verification(overall_passed=False),
        )
        assert artifact.confidence == Confidence.STUCK
        assert len(artifact.open_questions) > 0

    def test_stuck_can_have_failed_verification(self):
        """STUCK 时 self_verification.overall_passed 可以是 False."""
        artifact = _make_artifact(
            confidence=Confidence.STUCK,
            open_questions=["无法继续"],
            verification=_make_verification(overall_passed=False),
        )
        assert not artifact.self_verification.overall_passed


# ============================================================
# ArtifactBuilder 测试
# ============================================================

class TestArtifactBuilder:
    def test_full_build_flow(self):
        """逐步构建后 finalize 的完整性."""
        builder = ArtifactBuilder(
            task_id="task-100",
            task_description="创建 utils 模块",
            allowed_paths=["src/utils/**"],
            verification_spec="syntax + import",
            producer="worker-test",
        )
        builder.start(base_git_hash="deadbeef")

        builder.record_file_edit(
            path="src/utils/helpers.py",
            operation=EditOperation.CREATE,
            old=None,
            new="def add(a, b):\n    return a + b\n",
        )
        builder.record_decision(
            description="使用纯函数",
            reason="简单直接",
            alternatives=["类方法"],
        )
        builder.record_tool_call("write_file", "...", "ok")
        builder.record_llm_call(tokens_in=100, tokens_out=50, model="test-model")

        verification = _make_verification()
        builder.attach_self_verification(verification)
        builder.set_confidence(Confidence.DONE, "创建了 helpers.py")
        builder.add_open_question("要不要加 type hints？")

        artifact = builder.finalize()

        assert artifact.task_id == "task-100"
        assert artifact.producer == "worker-test"
        assert artifact.patch.total_files_changed == 1
        assert artifact.patch.base_git_hash == "deadbeef"
        assert artifact.confidence == Confidence.DONE
        assert artifact.self_summary == "创建了 helpers.py"
        assert len(artifact.open_questions) == 1
        assert len(artifact.decisions) == 1
        assert artifact.resource_usage.llm_calls == 1
        assert artifact.resource_usage.tool_calls == 1
        assert artifact.resource_usage.tokens_total == 150
        assert artifact.status == ArtifactStatus.DRAFT
        assert artifact.scope_check.is_clean

    def test_finalize_without_verification_fails(self):
        builder = ArtifactBuilder(
            task_id="t",
            task_description="d",
            allowed_paths=[],
            verification_spec="",
            producer="p",
        )
        builder.start("abc")
        builder.set_confidence(Confidence.DONE, "done")
        with pytest.raises(ArtifactBuilderError, match="attach_self_verification"):
            builder.finalize()

    def test_finalize_without_confidence_fails(self):
        builder = ArtifactBuilder(
            task_id="t",
            task_description="d",
            allowed_paths=[],
            verification_spec="",
            producer="p",
        )
        builder.start("abc")
        builder.attach_self_verification(_make_verification())
        with pytest.raises(ArtifactBuilderError, match="set_confidence"):
            builder.finalize()

    def test_double_finalize_fails(self):
        builder = ArtifactBuilder(
            task_id="t",
            task_description="d",
            allowed_paths=[],
            verification_spec="",
            producer="p",
        )
        builder.start("abc")
        builder.attach_self_verification(_make_verification())
        builder.set_confidence(Confidence.DONE, "done")
        builder.finalize()

        with pytest.raises(ArtifactBuilderError, match="已经 finalize"):
            builder.finalize()

    def test_modify_after_finalize_fails(self):
        builder = ArtifactBuilder(
            task_id="t",
            task_description="d",
            allowed_paths=[],
            verification_spec="",
            producer="p",
        )
        builder.start("abc")
        builder.attach_self_verification(_make_verification())
        builder.set_confidence(Confidence.DONE, "done")
        builder.finalize()

        with pytest.raises(ArtifactBuilderError, match="已经 finalize"):
            builder.record_file_edit("x.py", EditOperation.CREATE, None, "x")

    def test_empty_patch_build(self):
        """空 Patch（只做了读操作、没改文件）的处理."""
        builder = ArtifactBuilder(
            task_id="t",
            task_description="只读任务",
            allowed_paths=["src/**"],
            verification_spec="syntax",
            producer="worker",
        )
        builder.start("abc")
        builder.attach_self_verification(_make_verification())
        builder.set_confidence(Confidence.DONE, "只是读了文件")

        artifact = builder.finalize()
        assert artifact.patch.is_empty()
        assert artifact.patch.total_files_changed == 0
        assert artifact.patch.total_lines_added == 0

    def test_scope_check_auto_computed(self):
        """finalize 时自动计算越界检查."""
        builder = ArtifactBuilder(
            task_id="t",
            task_description="d",
            allowed_paths=["src/auth/**"],
            verification_spec="",
            producer="p",
        )
        builder.start("abc")
        builder.record_file_edit("src/auth/login.py", EditOperation.CREATE, None, "ok")
        builder.record_file_edit("config.yaml", EditOperation.MODIFY, "old", "new")
        builder.attach_self_verification(_make_verification())
        builder.set_confidence(Confidence.DONE, "done")

        artifact = builder.finalize()
        assert not artifact.scope_check.is_clean
        assert "config.yaml" in artifact.scope_check.out_of_scope_paths

    def test_unified_diff_auto_generated(self):
        """record_file_edit 自动生成 unified_diff."""
        builder = ArtifactBuilder(
            task_id="t",
            task_description="d",
            allowed_paths=["**"],
            verification_spec="",
            producer="p",
        )
        builder.start("abc")
        builder.record_file_edit(
            "foo.py",
            EditOperation.CREATE,
            None,
            "print('hello')\n",
        )
        builder.attach_self_verification(_make_verification())
        builder.set_confidence(Confidence.DONE, "done")

        artifact = builder.finalize()
        diff = artifact.patch.edits[0].unified_diff
        assert "+++ b/foo.py" in diff
        assert "--- /dev/null" in diff

    def test_lines_count_auto_computed(self):
        """record_file_edit 自动计算 lines_added / lines_removed."""
        builder = ArtifactBuilder(
            task_id="t",
            task_description="d",
            allowed_paths=["**"],
            verification_spec="",
            producer="p",
        )
        builder.start("abc")
        builder.record_file_edit(
            "foo.py",
            EditOperation.MODIFY,
            "line1\nline2\nline3\n",
            "line1\nnew_line2\nline3\nextra\n",
        )
        builder.attach_self_verification(_make_verification())
        builder.set_confidence(Confidence.DONE, "done")

        artifact = builder.finalize()
        edit = artifact.patch.edits[0]
        assert edit.lines_added > 0 or edit.lines_removed > 0  # 有变化
        assert artifact.patch.total_lines_added == edit.lines_added
        assert artifact.patch.total_lines_removed == edit.lines_removed


# ============================================================
# ArtifactStore 测试
# ============================================================

class TestArtifactStore:
    def test_save_and_load(self, tmp_path: Path):
        store = ArtifactStore(storage_dir=str(tmp_path / "artifacts"))
        artifact = _make_artifact()
        saved_path = store.save(artifact)

        assert Path(saved_path).exists()

        loaded = store.load(artifact.artifact_id)
        assert loaded.artifact_id == artifact.artifact_id
        assert loaded.task_id == artifact.task_id

    def test_patch_file_saved(self, tmp_path: Path):
        """保存时同时生成 .patch 文件."""
        store = ArtifactStore(storage_dir=str(tmp_path / "artifacts"))
        artifact = _make_artifact()
        store.save(artifact)

        patch_file = tmp_path / "artifacts" / artifact.task_id / f"{artifact.artifact_id}.patch"
        assert patch_file.exists()

    def test_list_for_task(self, tmp_path: Path):
        store = ArtifactStore(storage_dir=str(tmp_path / "artifacts"))
        artifact = _make_artifact()
        store.save(artifact)

        metas = store.list_for_task("task-001")
        assert len(metas) == 1
        assert metas[0].artifact_id == "artifact-001"
        assert metas[0].status == "DRAFT"

    def test_list_for_task_empty(self, tmp_path: Path):
        store = ArtifactStore(storage_dir=str(tmp_path / "artifacts"))
        assert store.list_for_task("nonexistent") == []

    def test_get_latest_for_task(self, tmp_path: Path):
        store = ArtifactStore(storage_dir=str(tmp_path / "artifacts"))

        a1 = SubtaskArtifact(
            artifact_id="a-001",
            task_id="task-x",
            created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            producer="w",
            task_description="t",
            allowed_paths=[],
            verification_spec="",
            patch=_make_patch(edits=[]),
            self_verification=_make_verification(),
            scope_check=ScopeCheck([], [], [], True),
            decisions=[],
            resource_usage=_make_resource(),
            confidence=Confidence.DONE,
            self_summary="first",
            open_questions=[],
            status=ArtifactStatus.REJECTED,
        )
        a2 = SubtaskArtifact(
            artifact_id="a-002",
            task_id="task-x",
            created_at=datetime(2025, 1, 2, tzinfo=timezone.utc),
            producer="w",
            task_description="t",
            allowed_paths=[],
            verification_spec="",
            patch=_make_patch(edits=[]),
            self_verification=_make_verification(),
            scope_check=ScopeCheck([], [], [], True),
            decisions=[],
            resource_usage=_make_resource(),
            confidence=Confidence.DONE,
            self_summary="second",
            open_questions=[],
            status=ArtifactStatus.DRAFT,
        )
        store.save(a1)
        store.save(a2)

        latest = store.get_latest_for_task("task-x")
        assert latest is not None
        assert latest.artifact_id == "a-002"
        assert latest.self_summary == "second"

    def test_get_latest_for_empty_task(self, tmp_path: Path):
        store = ArtifactStore(storage_dir=str(tmp_path / "artifacts"))
        assert store.get_latest_for_task("nope") is None

    def test_load_nonexistent_raises(self, tmp_path: Path):
        store = ArtifactStore(storage_dir=str(tmp_path / "artifacts"))
        (tmp_path / "artifacts").mkdir(parents=True)
        with pytest.raises(FileNotFoundError):
            store.load("no-such-id")

    def test_load_storage_dir_missing_raises(self, tmp_path: Path):
        store = ArtifactStore(storage_dir=str(tmp_path / "nonexistent"))
        with pytest.raises(FileNotFoundError):
            store.load("any-id")
