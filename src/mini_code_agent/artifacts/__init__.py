"""artifacts 模块 — Artifact Protocol 数据结构，Worker/Agent 产出的标准契约."""

from .artifact import ArtifactStatus, Confidence, SubtaskArtifact
from .builder import ArtifactBuilder, ArtifactBuilderError
from .decision import ArtifactDecision
from .patch import ApplyResult, EditOperation, FileEdit, Patch, generate_unified_diff
from .resource import ResourceUsage
from .scope import ScopeCheck, ScopeChecker
from .storage import ArtifactMeta, ArtifactStore
from .verification import CheckResult, SelfVerification

__all__ = [
    "ApplyResult",
    "ArtifactBuilder",
    "ArtifactBuilderError",
    "ArtifactDecision",
    "ArtifactMeta",
    "ArtifactStatus",
    "ArtifactStore",
    "CheckResult",
    "Confidence",
    "EditOperation",
    "FileEdit",
    "Patch",
    "ResourceUsage",
    "ScopeCheck",
    "ScopeChecker",
    "SelfVerification",
    "SubtaskArtifact",
    "generate_unified_diff",
]
