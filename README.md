# Markets

Derivatives pricing and risk infrastructure. Python 3.12 over QuantLib.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-fe5196.svg)](https://www.conventionalcommits.org/en/v1.0.0/)
[![SemVer](https://img.shields.io/badge/SemVer-2.0.0-blue.svg)](https://semver.org/spec/v2.0.0.html)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

<!-- CI, CodeQL and last-commit badges return on public visibility. -->

## Scope

- FX forwards, vanilla options and swaps
- IRS, FRAs, caps, floors and collars
- Curve bootstrap and log-DF monotone-Hermite interpolation
- Greeks, parametric and historical VaR, ICR stress, minimum-variance hedge ratios

## Install

```bash
uv sync --frozen --all-extras
```

Requires `uv >= 0.5`. Python 3.12 auto-selected via `.python-version`.

## Usage

```python
import markets
```

Case archives at `case-comp/<year>/`. Template at `case-comp/_template/`.

## Maintainers

See [CODEOWNERS](.github/CODEOWNERS).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Conventional Commits 1.0.0 and DCO sign-off required.

## License

[MIT](LICENSE).

## Related

[SECURITY](SECURITY.md) | [SUPPORT](SUPPORT.md) | [GOVERNANCE](GOVERNANCE.md) | [CHANGELOG](CHANGELOG.md) | [ROADMAP](ROADMAP.md) | [CITATION](CITATION.cff) | [ADRs](docs/adr/)
