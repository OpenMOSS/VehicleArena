# Cabin rules

[中文](../extensions/cabin-rules.md) · [English index](README.md)

YAML rules declare cabin triggers and acceptable outcomes. They are evaluation specifications, not a driving controller; road signals, collisions, and traffic risks are evaluated from outdoor trajectories.

An extension registers a rule directory during initialization:

```python
from pathlib import Path
from rules.rule_loader import register_rule_directory

register_rule_directory(Path(__file__).parent / "rules")
```

The framework builds one index from built-in and extension directories and rejects duplicate `(domain, id)` pairs. A minimal rule is:

```yaml
rules:
  - id: eco_mode
    description: Passenger requests energy-saving mode
    domain: user_intent
    requires_modules: [energyMeter]
    trigger:
      user_intent:
        messages: ["enable eco mode"]
        vague: ["save some energy"]
    expect:
      actions:
        - vw.energyMeter.set_mode('eco')
    priority: 20
```

Supported domains are `weather`, `daynight`, `map_event`, and `user_intent`; each has a domain-specific trigger structure. `requires_modules` determines whether the rule applies to a vehicle. If omitted, it can be inferred from the primary `vw.<module>.<method>` action. Passenger-language input is not filtered by installed equipment: the Driving Agent should query capabilities and either execute a supported request or explain why it cannot.

Rules may use `tolerance.any_of` for equivalent actions, `tolerance.skip` for irrelevant fields, `trend` or `ceiling` for directional/upper-bound checks, and `negative_checks` for continuously forbidden states. Rule actions allow a single `vw` method call with literal arguments; imports, assignments, private attributes, and arbitrary Python are not accepted.

```bash
python scripts/validate_extension.py \
  --extension your_package.vehiclearena_extension \
  --rules your_package/rules
```

See [the full rule HOWTO](../../vehiclearena/rules/HOWTO.md) for every field and example.
