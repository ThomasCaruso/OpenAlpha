class ResearchCoreError(Exception):
    """Base class for stable research-core failures."""


class ArtifactPathError(ResearchCoreError):
    """An artifact path escaped or violated the portable path policy."""


class ArtifactConflictError(ResearchCoreError):
    """An immutable artifact path already contains different bytes."""


class ArtifactIntegrityError(ResearchCoreError):
    """Stored artifact bytes do not match their declared identity."""


class ManifestIntegrityError(ResearchCoreError):
    """A run manifest cannot be published as verified evidence."""


class RunStateTransitionError(ResearchCoreError):
    """A run-state event violates the append-only lifecycle."""
