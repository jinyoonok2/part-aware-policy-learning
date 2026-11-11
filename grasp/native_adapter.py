"""Pure native-action bridge for bounded local interventions around MolmoBot.

The pinned DoorPlusOpen policy emits relative base/arm joint commands, absolute
gripper commands and one torso-height scalar. Local geometry controllers must
tag their absolute targets explicitly before converting them at the current
measured qpos. Nothing here changes controller modes, simulator state or weights.
"""
from dataclasses import dataclass
from typing import Mapping

import numpy as np


_DIMS = {'base': 3, 'left_arm': 7, 'right_arm': 7,
         'left_gripper': 1, 'right_gripper': 1, 'torso': 1}
_RELATIVE = ('base', 'left_arm', 'right_arm')


def _vector(value, length, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f'{name} must be a finite vector of length {length}, got {result.shape}')
    return result.copy()


def wrap_angle(angle):
    """Shortest signed angular displacement, including the +/-pi boundary."""
    return (float(angle) + np.pi) % (2 * np.pi) - np.pi


@dataclass(frozen=True)
class AbsoluteJointTargets:
    """Explicit command-space tag; do not wrap already-relative policy actions."""
    values: Mapping[str, np.ndarray]

    def __post_init__(self):
        copied = {}
        for name, value in self.values.items():
            if name not in _DIMS:
                raise ValueError(f'Unsupported local move group: {name}')
            if name == 'torso':
                # No 6-joint torso vector may cross into the native height API.
                # A generic local controller may carry such a hold; discard it
                # below and obtain the scalar from policy/current robot state.
                array = np.asarray(value, dtype=np.float64)
                if array.shape not in ((1,), (6,)) or not np.isfinite(array).all():
                    raise ValueError('Local torso hold must have one or six finite values')
                copied[name] = array.copy()
            else:
                copied[name] = _vector(value, _DIMS[name], f'absolute {name}')
        object.__setattr__(self, 'values', copied)


@dataclass(frozen=True)
class NativeDeltaLimits:
    arm_rad: float = .04
    base_xy_m: float = .015
    base_yaw_rad: float = .03

    def __post_init__(self):
        if min(self.arm_rad, self.base_xy_m, self.base_yaw_rad) <= 0:
            raise ValueError('Native delta limits must be positive')


def current_torso_height(qpos):
    """Pinned TorsoHeightJointPosController.joints_to_height uses joints[1]."""
    joints = _vector(qpos['torso'], 6, 'measured torso qpos')
    return float(joints[1])


def validate_native_action(action):
    """Return a copied six-group native action; reject torso dimensional leaks."""
    copied = {}
    for name, value in action.items():
        if name not in _DIMS:
            raise ValueError(f'Unsupported native move group: {name}')
        copied[name] = _vector(value, _DIMS[name], f'native {name}')
    return copied


def absolute_to_native_action(targets, current_qpos, policy_action=None,
                              torso_mode='policy', limits=None):
    """Merge absolute local targets into a copied native MolmoBot action.

    Relative base and arm commands are computed from measured qpos once. Base
    yaw uses the shortest wrap. Absolute gripper scalars are copied unchanged.
    Any local torso entry is deliberately ignored: ``policy`` retains the
    policy's one-value torso command, falling back to current height; ``hold``
    always uses the measured height. Optional clipping bounds local deltas only.
    Existing policy commands on untouched groups pass through unchanged.

    Requiring AbsoluteJointTargets prevents accidentally feeding a native dict
    back through the conversion a second time.
    """
    if not isinstance(targets, AbsoluteJointTargets):
        raise TypeError('Local commands must be tagged AbsoluteJointTargets; native actions are already relative')
    if torso_mode not in ('policy', 'hold'):
        raise ValueError('torso_mode must be policy or hold')
    if limits is not None and not isinstance(limits, NativeDeltaLimits):
        raise TypeError('limits must be NativeDeltaLimits or None')
    action = validate_native_action(policy_action or {})
    for name, absolute in targets.values.items():
        if name == 'torso':
            continue
        if name in _RELATIVE:
            measured = _vector(current_qpos[name], _DIMS[name], f'measured {name} qpos')
            delta = absolute - measured
            if name == 'base':
                delta[2] = wrap_angle(delta[2])
            if limits is not None:
                if name == 'base':
                    norm = np.linalg.norm(delta[:2])
                    if norm > limits.base_xy_m:
                        delta[:2] *= limits.base_xy_m / norm
                    delta[2] = np.clip(delta[2], -limits.base_yaw_rad, limits.base_yaw_rad)
                else:
                    delta = np.clip(delta, -limits.arm_rad, limits.arm_rad)
            action[name] = delta
        else:
            action[name] = absolute.copy()
    if torso_mode == 'hold' or 'torso' not in action:
        action['torso'] = np.array([current_torso_height(current_qpos)], dtype=np.float64)
    return action


def hold_native_action(current_qpos, policy_action=None, close_arm=None):
    """Zero native base/arm deltas and hold current torso during local control.

    Preserve policy gripper commands if supplied; otherwise hold measured finger
    position through the single gripper actuator. ``close_arm`` optionally
    forces that gripper closed with the same +100 scalar used by MolmoBot.
    """
    if close_arm not in (None, 'left', 'right'):
        raise ValueError('close_arm must be left, right or None')
    action = validate_native_action(policy_action or {})
    for name in _RELATIVE:
        _vector(current_qpos[name], _DIMS[name], f'measured {name} qpos')
        action[name] = np.zeros(_DIMS[name], dtype=np.float64)
    action['torso'] = np.array([current_torso_height(current_qpos)], dtype=np.float64)
    for arm in ('left', 'right'):
        name = arm + '_gripper'
        if name not in action:
            fingers = np.asarray(current_qpos[name], dtype=np.float64)
            if fingers.shape not in ((1,), (2,)) or not np.isfinite(fingers).all():
                raise ValueError(f'Invalid measured {name} qpos')
            action[name] = fingers[:1].copy()
    if close_arm:
        action[close_arm + '_gripper'] = np.array([100.])
    return action


def invalidate_action_buffer(policy):
    """Drop stale unexecuted commands without resetting any policy history.

    Do not use get_state/set_state: the pinned RBY1 subclass does not include
    obs_history in its get_state result. Initial image/point conditioning, RNG,
    executed-step counter and history remain the original objects/values.
    """
    discarded = max(len(policy.action_buffer) - int(policy.buffer_index), 0)
    policy.action_buffer = []
    policy.buffer_index = 0
    return discarded


def account_local_step(policy, observation, model_was_queried):
    """Account one actual local-control step if get_action was skipped.

    Native get_action/inference_model already append history and increment the
    counter before returning. If it was queried and its action overridden, do
    not account again. When local control skipped inference, record the current
    pre-action observation once so the next inference sees consistent history.
    """
    if model_was_queried:
        return
    obs = observation[0] if isinstance(observation, (tuple, list)) else observation
    policy.obs_history.append(obs)
    policy.step_count += 1


@dataclass(frozen=True)
class GateConfig:
    trigger_distance_m: float = .18
    close_threshold: float = 0.
    verification_samples: int = 13
    max_local_steps: int = 24
    max_events: int = 2
    cooldown_steps: int = 8

    def __post_init__(self):
        if self.trigger_distance_m <= 0 or min(self.verification_samples, self.max_local_steps, self.max_events) < 1 or self.cooldown_steps < 0:
            raise ValueError('Invalid local grasp gate bounds')


@dataclass(frozen=True)
class GateDecision:
    use_local: bool
    arm: str | None
    reason: str
    started: bool = False
    finished: bool = False
    force_samples: int = 0
    event_index: int = 0


class LocalGraspGate:
    """Pure bounded trigger shared by guard-only and oracle-pose interventions.

    Call ``decide`` exactly once per observed control frame. When inactive,
    provide the policy's proposed native action. When active, pass None and
    decide whether to keep controlling locally before querying MolmoBot again.
    A finished decision returns use_local=False; invalidate stale buffers and
    obtain a fresh policy action on the same observation before resuming.

    Triggering uses only policy closing intent and current TCP/handle distance.
    It never reads expert phases, future trajectories or benchmark outcomes.
    ``force_grasp`` contains per-arm instantaneous force/contact checks supplied
    by the runner. The gate requires consecutive observed samples to release.
    """
    def __init__(self, cfg=GateConfig()):
        self.cfg = cfg
        self.arm = None
        self.local_steps = 0
        self.cooldown = 0
        self.force_streak = {'left': 0, 'right': 0}
        self.events = []
        self.frames_seen = 0

    @property
    def active(self):
        return self.arm is not None

    def _finish(self, reason):
        arm = self.arm
        self.events[-1].update(end_frame=self.frames_seen - 1, local_steps=self.local_steps, reason=reason)
        self.arm = None
        self.cooldown = self.cfg.cooldown_steps
        return GateDecision(False, arm, reason, finished=True,
                            force_samples=self.force_streak[arm], event_index=len(self.events))

    def decide(self, policy_action, distances, force_grasp):
        self.frames_seen += 1
        for arm in ('left', 'right'):
            self.force_streak[arm] = self.force_streak[arm] + 1 if force_grasp.get(arm, False) else 0
        if self.active:
            if self.force_streak[self.arm] >= self.cfg.verification_samples:
                return self._finish('force_verified')
            if self.local_steps >= self.cfg.max_local_steps:
                return self._finish('intervention_timeout')
            self.local_steps += 1
            return GateDecision(True, self.arm, 'local_verification', force_samples=self.force_streak[self.arm], event_index=len(self.events))
        if self.cooldown:
            self.cooldown -= 1
            return GateDecision(False, None, 'cooldown', event_index=len(self.events))
        if len(self.events) >= self.cfg.max_events:
            return GateDecision(False, None, 'intervention_budget_exhausted', event_index=len(self.events))
        if policy_action is None:
            return GateDecision(False, None, 'policy_action_required', event_index=len(self.events))
        candidates = []
        for arm in ('left', 'right'):
            key = arm + '_gripper'
            if key not in policy_action or arm not in distances:
                continue
            distance = float(distances[arm])
            if distance < 0:
                raise ValueError('TCP-to-handle distances cannot be negative')
            closing = _vector(policy_action[key], 1, f'policy {key}')[0] >= self.cfg.close_threshold
            if (closing and np.isfinite(distance) and distance <= self.cfg.trigger_distance_m
                    and self.force_streak[arm] < self.cfg.verification_samples):
                candidates.append((distance, arm))
        if not candidates:
            return GateDecision(False, None, 'no_local_trigger', event_index=len(self.events))
        _, self.arm = min(candidates)
        self.local_steps = 1
        self.events.append(dict(start_frame=self.frames_seen - 1, arm=self.arm, trigger='near_handle_closing'))
        return GateDecision(True, self.arm, 'near_handle_closing', started=True,
                            force_samples=self.force_streak[self.arm], event_index=len(self.events))
