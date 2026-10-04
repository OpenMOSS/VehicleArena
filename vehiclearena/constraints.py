"""
Layer 2: ConstraintEngine - Pre/Post Condition Enforcement
Checks constraints before and after API execution.
- HARD: Blocks execution, returns error
- SOFT: Allows execution, adds warning to result
- ADVISORY: Allows execution, logs info
"""

import threading
import logging
from enum import Enum
from typing import Callable, Dict, List, Any, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


class ConstraintLevel(Enum):
    """Severity level of a constraint."""
    HARD = "hard"           # Blocks execution entirely
    SOFT = "soft"           # Warns but allows execution
    ADVISORY = "advisory"   # Informational only


class ConstraintResult:
    """Result of a constraint check."""

    def __init__(self, passed: bool, level: ConstraintLevel,
                 message: str = "", constraint_name: str = ""):
        self.passed = passed
        self.level = level
        self.message = message
        self.constraint_name = constraint_name

    def __repr__(self):
        status = "PASS" if self.passed else "FAIL"
        return f"ConstraintResult({status}, {self.level.value}, {self.message})"


class Constraint:
    """
    A single constraint rule.

    The check function receives (module_instance, method_name, args, kwargs, vw)
    and returns a ConstraintResult.
    """

    def __init__(self, name: str, level: ConstraintLevel,
                 check_fn: Callable, phase: str = "pre",
                 target_modules: Optional[List[str]] = None,
                 target_methods: Optional[List[str]] = None):
        """
        Args:
            name: Human-readable constraint name
            level: HARD / SOFT / ADVISORY
            check_fn: Function(module_instance, method_name, args, kwargs, vw) -> ConstraintResult
            phase: "pre" (before execution) or "post" (after execution)
            target_modules: List of module names this constraint applies to (None = all)
            target_methods: List of method names this constraint applies to (None = all)
        """
        self.name = name
        self.level = level
        self.check_fn = check_fn
        self.phase = phase
        self.target_modules = target_modules
        self.target_methods = target_methods

    def applies_to(self, module_name: str, method_name: str) -> bool:
        """Check if this constraint applies to the given module/method."""
        if self.target_modules and module_name not in self.target_modules:
            return False
        if self.target_methods and method_name not in self.target_methods:
            return False
        return True


class ConstraintEngine:
    """
    Central constraint engine that manages and evaluates constraints.
    Thread-safe.
    """

    def __init__(self):
        self._pre_constraints: List[Constraint] = []
        self._post_constraints: List[Constraint] = []
        self._lock = threading.Lock()
        self._enabled = True
        self._vw = None  # Reference to VehicleWorld instance

    def __deepcopy__(self, memo):
        """Support deep-copying by creating a fresh ConstraintEngine (new lock).

        Constraints are NOT copied — the copy has an empty rule set and
        constraint checking disabled.  This is intentional: GT evaluation
        VWs only execute GT code_lines and don't need coupling rules.
        """
        import copy
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result._pre_constraints = []
        result._post_constraints = []
        result._lock = threading.Lock()
        result._enabled = False   # disabled — no coupling checks on GT copy
        result._vw = None
        return result

    def set_vehicle_world(self, vw):
        """Set the VehicleWorld reference for cross-module constraint checks."""
        self._vw = vw

    def register(self, constraint: Constraint):
        """Register a constraint."""
        with self._lock:
            if constraint.phase == "pre":
                self._pre_constraints.append(constraint)
            elif constraint.phase == "post":
                self._post_constraints.append(constraint)
            else:
                raise ValueError(f"Invalid constraint phase: {constraint.phase}")

    def check_pre(self, module_name: str, method_name: str,
                  module_instance: Any, args: tuple, kwargs: dict) -> Tuple[bool, List[ConstraintResult]]:
        """
        Run pre-execution constraints.

        Returns:
            (can_execute, results): can_execute is False if any HARD constraint fails
        """
        if not self._enabled:
            return True, []

        results = []
        can_execute = True

        with self._lock:
            constraints = list(self._pre_constraints)

        for constraint in constraints:
            if not constraint.applies_to(module_name, method_name):
                continue

            try:
                result = constraint.check_fn(
                    module_instance, method_name, args, kwargs, self._vw
                )
                if result and not result.passed:
                    result.constraint_name = constraint.name
                    results.append(result)

                    if constraint.level == ConstraintLevel.HARD:
                        can_execute = False
                        logger.warning(
                            f"HARD constraint '{constraint.name}' blocked "
                            f"{module_name}.{method_name}: {result.message}"
                        )
                    elif constraint.level == ConstraintLevel.SOFT:
                        logger.info(
                            f"SOFT constraint '{constraint.name}' warning for "
                            f"{module_name}.{method_name}: {result.message}"
                        )
                    else:
                        logger.debug(
                            f"ADVISORY '{constraint.name}' for "
                            f"{module_name}.{method_name}: {result.message}"
                        )
            except Exception as e:
                logger.error(f"Constraint '{constraint.name}' check error: {e}")

        return can_execute, results

    def check_post(self, module_name: str, method_name: str,
                   module_instance: Any, args: tuple, kwargs: dict,
                   result: Any) -> List[ConstraintResult]:
        """
        Run post-execution constraints.

        Returns:
            List of constraint results (informational)
        """
        if not self._enabled:
            return []

        constraint_results = []

        with self._lock:
            constraints = list(self._post_constraints)

        for constraint in constraints:
            if not constraint.applies_to(module_name, method_name):
                continue

            try:
                cr = constraint.check_fn(
                    module_instance, method_name, args, kwargs, self._vw
                )
                if cr and not cr.passed:
                    cr.constraint_name = constraint.name
                    constraint_results.append(cr)
            except Exception as e:
                logger.error(f"Post-constraint '{constraint.name}' error: {e}")

        return constraint_results

    def enable(self):
        """Enable constraint checking."""
        self._enabled = True

    def disable(self):
        """Disable constraint checking."""
        self._enabled = False

    def clear(self):
        """Remove all constraints."""
        with self._lock:
            self._pre_constraints.clear()
            self._post_constraints.clear()
