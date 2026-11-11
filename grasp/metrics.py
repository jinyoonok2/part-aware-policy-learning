"""Explicit sampled grasp-retention criterion, separate from benchmark success.

Contact forces and relative slip are simulation proxies for a firm grasp. Saved
10 Hz traces cannot establish continuous contact or real-hardware robustness.
"""
import numpy as np
from scipy.spatial.transform import Rotation


POLICY = dict(minimum_finger_force_N=1., minimum_aperture_m=.002,
              minimum_closed_ctrl=-.005, grasp_samples=13, max_opening_at_grasp=.10,
              minimum_opening_gain=.50, retention_fraction=.95,
              max_missing_contact_samples=1, max_translation_slip_m=.02,
              max_rotation_slip_deg=10., success_hold_samples=10, sample_period_s=.1)

# The historical pilot's POLICY and default call remain unchanged. In final
# evaluation the first force-qualified sample anchors the grasp. The thirteen
# confirming samples may include opening motion; no stationary wait is needed.
FINAL_POLICY = dict(POLICY, metric_version='firm_grasp_opening_v3_onset',
                    require_no_prior_major_opening=True,
                    anchor_at_force_onset=True)


def force_grasp(entry, policy=POLICY):
    forces = [0., 0.]
    for contact in entry['finger_contacts']:
        if contact['other_body'] == entry['selected_handle'] and contact['active_constraint']:
            forces[contact['finger'] - 1] += max(float(contact['normal_force_N']), 0.)
    closed = float(np.asarray(entry['gripper_ctrl']).reshape(-1)[0]) >= policy['minimum_closed_ctrl']
    valid = (entry['active_arm_exact_both_fingers'] and closed
             and entry['gripper_aperture_m'] >= policy['minimum_aperture_m']
             and min(forces) >= policy['minimum_finger_force_N'])
    return bool(valid), forces


def _retained_grasp_from_onset(trace, threshold, policy):
    """Anchor before opening, then verify without imposing a stationary wait."""
    anchor = None
    attempts = []
    prior_max_opening = None
    samples = contact_samples = missing = maximum_missing = 0
    streak = joint_streak = 0
    verified_step = None
    opening_at_verification = None
    max_translation = max_rotation = 0.

    def outcome(success, reason, step=None, blocked=False):
        value = dict(retained_grasp_opening_success=success, reason=reason,
                     metric_version=policy['metric_version'],
                     grasp_onset_step=anchor['step'] if anchor else None,
                     grasp_verified_step=verified_step,
                     opening_at_grasp=anchor['opening'] if anchor else None,
                     opening_at_verification=opening_at_verification,
                     max_translation_slip_m=max_translation,
                     max_rotation_slip_deg=max_rotation,
                     prior_failed_attempts=attempts,
                     maximum_prior_opening_fraction=prior_max_opening,
                     regrasp_blocked_after_major_opening=blocked)
        if step is not None:
            value['success_step' if success else 'disqualifying_step'] = step
        if success:
            value.update(opening_gain=current_opening - anchor['opening'],
                         contact_fraction=contact_samples / samples,
                         max_missing_contact_samples=maximum_missing)
        return value

    for entry in trace:
        current_opening = float(entry['opening_fraction'])
        prior_max_opening = (current_opening if prior_max_opening is None
                             else max(prior_max_opening, current_opening))
        contact, _ = force_grasp(entry, policy)
        relative = np.linalg.inv(np.asarray(entry['selected_handle_world'])) @ np.asarray(entry['tcp_world'])
        if anchor is None:
            if (policy.get('require_no_prior_major_opening', False)
                    and prior_max_opening > policy['max_opening_at_grasp']):
                reason = ('grasp_cannot_be_reestablished_after_major_opening' if attempts
                          else 'major_opening_before_force_grasp_onset')
                return outcome(False, reason, entry['step'], blocked=True)
            if not contact or current_opening > policy['max_opening_at_grasp']:
                continue
            anchor = dict(step=entry['step'], opening=current_opening, relative=relative.copy())
            samples = contact_samples = missing = maximum_missing = 0
            streak = joint_streak = 0
            verified_step = opening_at_verification = None
            max_translation = max_rotation = 0.

        samples += 1
        contact_samples += int(contact)
        streak = streak + 1 if contact else 0
        missing = 0 if contact else missing + 1
        maximum_missing = max(maximum_missing, missing)
        translation = float(np.linalg.norm(relative[:3, 3] - anchor['relative'][:3, 3]))
        rotation = float(np.degrees(Rotation.from_matrix(relative[:3, :3] @ anchor['relative'][:3, :3].T).magnitude()))
        max_translation, max_rotation = max(max_translation, translation), max(max_rotation, rotation)
        problem = None
        if verified_step is None and not contact:
            problem = 'force_grasp_verification_interrupted'
        elif maximum_missing > policy['max_missing_contact_samples']:
            problem = 'grasp_contact_or_force_lost'
        elif translation > policy['max_translation_slip_m'] or rotation > policy['max_rotation_slip_deg']:
            problem = 'handle_relative_slip_limit_exceeded'
        if problem:
            attempts.append(dict(grasp_step=anchor['step'], grasp_verified_step=verified_step,
                                 end_step=entry['step'], reason=problem,
                                 max_translation_slip_m=max_translation, max_rotation_slip_deg=max_rotation))
            if (policy.get('require_no_prior_major_opening', False)
                    and prior_max_opening > policy['max_opening_at_grasp']):
                return outcome(False, problem, entry['step'], blocked=True)
            anchor, verified_step, streak = None, None, 0
            opening_at_verification = None
            continue
        if verified_step is None and streak >= policy['grasp_samples']:
            verified_step = entry['step']
            opening_at_verification = current_opening
        simultaneous = contact and current_opening >= threshold
        joint_streak = joint_streak + 1 if simultaneous else 0
        if (verified_step is not None and joint_streak >= policy['success_hold_samples']
                and current_opening - anchor['opening'] >= policy['minimum_opening_gain']
                and contact_samples / samples >= policy['retention_fraction']):
            return outcome(True, 'sampled_force_and_retention_checks_passed', entry['step'])

    reason = ('force_grasp_verification_not_completed' if anchor and verified_step is None
              else 'opening_or_retention_requirement_not_completed' if anchor
              else attempts[-1]['reason'] if attempts
              else 'no_force_grasp_onset_before_major_opening')
    return outcome(False, reason)


def retained_grasp_opening(trace, threshold, policy=POLICY):
    """Evaluate one arm, preserving the historical default metric verbatim.

    FINAL_POLICY anchors the first qualifying force/contact sample before
    major opening, verifies thirteen consecutive samples while allowing door
    motion, and checks slip from that first sample. Include initial step zero.
    """
    if policy.get('anchor_at_force_onset', False):
        return _retained_grasp_from_onset(trace, threshold, policy)
    anchor = None
    streak, joint_streak, missing, maximum_missing = 0, 0, 0, 0
    samples, contact_samples = 0, 0
    attempts = []
    all_forces = []
    max_translation, max_rotation = 0., 0.
    for i, entry in enumerate(trace):
        contact, forces = force_grasp(entry, policy)
        all_forces.append(forces)
        streak = streak + 1 if contact else 0
        relative = np.linalg.inv(np.asarray(entry['selected_handle_world'])) @ np.asarray(entry['tcp_world'])
        if anchor is None and streak >= policy['grasp_samples'] and entry['opening_fraction'] <= policy['max_opening_at_grasp']:
            anchor = dict(index=i, step=entry['step'], opening=entry['opening_fraction'], relative=relative.copy())
            samples, contact_samples, missing, maximum_missing, joint_streak = 0, 0, 0, 0, 0
            max_translation, max_rotation = 0., 0.
        if anchor is None:
            continue
        samples += 1
        contact_samples += int(contact)
        missing = 0 if contact else missing + 1
        maximum_missing = max(maximum_missing, missing)
        translation = float(np.linalg.norm(relative[:3, 3] - anchor['relative'][:3, 3]))
        rotation = float(np.degrees(Rotation.from_matrix(relative[:3, :3] @ anchor['relative'][:3, :3].T).magnitude()))
        max_translation, max_rotation = max(max_translation, translation), max(max_rotation, rotation)
        problem = None
        if maximum_missing > policy['max_missing_contact_samples']:
            problem = 'grasp_contact_or_force_lost'
        elif translation > policy['max_translation_slip_m'] or rotation > policy['max_rotation_slip_deg']:
            problem = 'handle_relative_slip_limit_exceeded'
        if problem:
            attempts.append(dict(grasp_step=anchor['step'], end_step=entry['step'], reason=problem,
                                 max_translation_slip_m=max_translation, max_rotation_slip_deg=max_rotation))
            anchor, streak = None, 0
            continue
        simultaneous = contact and entry['opening_fraction'] >= threshold
        joint_streak = joint_streak + 1 if simultaneous else 0
        gain = entry['opening_fraction'] - anchor['opening']
        if (joint_streak >= policy['success_hold_samples'] and gain >= policy['minimum_opening_gain']
                and contact_samples / samples >= policy['retention_fraction']):
            return dict(retained_grasp_opening_success=True, grasp_verified_step=anchor['step'],
                        success_step=entry['step'], opening_at_grasp=anchor['opening'], opening_gain=gain,
                        contact_fraction=contact_samples / samples, max_missing_contact_samples=maximum_missing,
                        max_translation_slip_m=max_translation, max_rotation_slip_deg=max_rotation,
                        prior_failed_attempts=attempts, reason='sampled_force_and_retention_checks_passed')
    return dict(retained_grasp_opening_success=False,
                reason=(attempts[-1]['reason'] if attempts else
                        'no_force_verified_grasp_before_major_opening' if anchor is None else
                        'opening_or_retention_requirement_not_completed'),
                grasp_verified_step=anchor['step'] if anchor else None,
                opening_at_grasp=anchor['opening'] if anchor else None,
                max_translation_slip_m=max_translation, max_rotation_slip_deg=max_rotation,
                prior_failed_attempts=attempts)
