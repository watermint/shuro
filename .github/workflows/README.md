# GitHub Actions Workflows

## Build macOS Binary (`build-macos.yml`)

This workflow automatically builds macOS binaries for the shuro project with semantic versioning.

### Versioning Strategy

The workflow uses a semantic versioning scheme where:
- **Major.Minor**: Extracted from `Cargo.toml` version field
- **Patch**: GitHub Actions run number (`${{ github.run_number }}`)

**Example**: If `Cargo.toml` has `version = "0.1.0"` and this is the 42nd workflow run, the built binary will have version `0.1.42`.

### Build Outputs

The workflow produces a single binary artifact:
- **aarch64**: For Apple Silicon Macs

### Triggers

- Push to `main` branch
- Pull request to `main` branch
- Manual workflow dispatch from GitHub UI

### Artifacts

Built binaries are uploaded as GitHub Actions artifacts with names like:
- `shuro-aarch64-apple-darwin-v0.1.42` 