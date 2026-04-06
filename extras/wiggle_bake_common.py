"""
Shared post-bake steps for Wiggle-based jiggle (boobs, hair, etc.).
Single place to tune loop closure and non-loop start cleanup for Blender 4.x.
"""

import math
from mathutils import Quaternion


# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------

def _parse_bone_name(data_path):
    """Extract bone name from a pose bone data path. Returns '' on failure."""
    try:
        return data_path.split('pose.bones["')[1].split('"]')[0]
    except Exception:
        return ''


def _eval_quat_normalized(channels, frame):
    """Evaluate and normalize a quaternion from 4 fcurve channels at a frame."""
    q = Quaternion(tuple(ch.evaluate(frame) for ch in channels))
    if q.magnitude > 1e-8:
        q.normalize()
    return q


# ---------------------------------------------------------------------------
#  Loop detection
# ---------------------------------------------------------------------------

# Keywords that identify stable body bones for loop detection.
# Only these bones are checked — physics-driven bones (cape, hair, tail, etc.)
# can have different first/last frame values even in a true loop, which would
# cause false negatives if included.  Buffbones (VFX helpers) are excluded.
_LOOP_DETECT_KEYWORDS = (
    'spine', 'pelvis',
    'l_hip', 'r_hip',
    'l_clavicle', 'r_clavicle',
    'l_shoulder', 'r_shoulder',
    'head', 'neck',
)


def _is_loop_detect_bone(bone_name):
    """Return True if this bone is a stable body bone used for loop detection."""
    bl = bone_name.lower()
    if 'buffbone' in bl:
        return False
    return any(kw in bl for kw in _LOOP_DETECT_KEYWORDS)


def _detect_animation_loops(action, frame_start, frame_end):
    """Return True if the animation loops (first frame ≈ last frame).

    Only checks stable body bones (spine, pelvis, hips, shoulders, head, neck).
    Physics-driven bones (cape, hair, tail, etc.) are excluded because they
    can differ at the loop boundary even in a true loop animation.

    Resolves quaternion alias bugs where q and -q are mathematically different
    in F-curves but visually identical (handles them as equal).
    """
    bone_states_start = {}
    bone_states_end   = {}
    has_data = False

    for fcurve in action.fcurves:
        dp = fcurve.data_path
        if 'pose.bones["' not in dp:
            continue

        bname = _parse_bone_name(dp)
        if not _is_loop_detect_bone(bname):
            continue

        val_s = fcurve.evaluate(frame_start)
        val_e = fcurve.evaluate(frame_end)
        idx   = fcurve.array_index

        if dp not in bone_states_start:
            bone_states_start[dp] = [0.0, 0.0, 0.0, 1.0]
            bone_states_end[dp]   = [0.0, 0.0, 0.0, 1.0]

        if idx < 4:
            bone_states_start[dp][idx] = val_s
            bone_states_end[dp][idx]   = val_e
            has_data = True

    if not has_data:
        return False

    max_diff = 0.0
    for dp, vs in bone_states_start.items():
        ve = bone_states_end[dp]

        if 'rotation_quaternion' in dp:
            mag_s = math.sqrt(sum(v * v for v in vs))
            mag_e = math.sqrt(sum(v * v for v in ve))
            if mag_s > 0.001 and mag_e > 0.001:
                dot  = sum((vs[i] / mag_s) * (ve[i] / mag_e) for i in range(4))
                diff = 1.0 - abs(dot)
            else:
                diff = max(abs(vs[i] - ve[i]) for i in range(4))
        else:
            diff = max(abs(vs[i] - ve[i]) for i in range(3))

        if diff > max_diff:
            max_diff = diff

    return max_diff < 0.1


# ---------------------------------------------------------------------------
#  Non-loop start cleanup
# ---------------------------------------------------------------------------

def _restore_nonloop_start_to_tpose(action, frame_start, bone_names):
    """Non-looping animations: replace baked frame_start with identity pose."""
    bone_set = set(b for b in bone_names if b)

    for fcurve in action.fcurves:
        dp = fcurve.data_path
        if 'pose.bones["' not in dp:
            continue
        bname = _parse_bone_name(dp)
        if bname not in bone_set:
            continue

        prop = dp.split('].')[-1] if '].' in dp else ''
        idx  = fcurve.array_index

        if 'location' in prop:
            identity = 0.0
        elif 'rotation_quaternion' in prop:
            identity = 1.0 if idx == 0 else 0.0
        elif 'rotation_euler' in prop:
            identity = 0.0
        elif 'scale' in prop:
            identity = 1.0
        else:
            continue

        kp = fcurve.keyframe_points.insert(frame_start, identity, options={'FAST', 'REPLACE'})
        kp.interpolation = 'BEZIER'
        fcurve.update()


def _smooth_boundary_frames(action, frame_start, frame_end, bone_names,
                             smooth_range=3, smooth_ends='both'):
    """Lerp boundary frames toward the anchor keyframe for smooth ease-in/out."""
    bone_set = set(b for b in bone_names if b)

    for fcurve in action.fcurves:
        dp = fcurve.data_path
        if 'pose.bones["' not in dp:
            continue
        if _parse_bone_name(dp) not in bone_set:
            continue

        kf_map = {int(kp.co[0]): kp for kp in fcurve.keyframe_points}

        anchor_kp = kf_map.get(frame_start)
        if anchor_kp:
            anchor_val = anchor_kp.co[1]
            for i in range(1, smooth_range + 1):
                kp = kf_map.get(frame_start + i)
                if kp is None:
                    continue
                t = i / (smooth_range + 1)
                kp.co[1] = anchor_val * (1.0 - t) + kp.co[1] * t

        if smooth_ends == 'both':
            anchor_kp = kf_map.get(frame_end)
            if anchor_kp:
                anchor_val = anchor_kp.co[1]
                for i in range(1, smooth_range + 1):
                    kp = kf_map.get(frame_end - i)
                    if kp is None:
                        continue
                    t = i / (smooth_range + 1)
                    kp.co[1] = anchor_val * (1.0 - t) + kp.co[1] * t

        fcurve.update()


# ---------------------------------------------------------------------------
#  Loop seam helpers
# ---------------------------------------------------------------------------

def _force_loop_perfect_match(action, frame_start, frame_end, bone_names):
    """Snap last-frame keys to first-frame keys for looping clips.

    Minimal operation: only overwrites the last frame's value.
    Use after post-bake collision correction to re-seal the loop
    without touching the surrounding frames (which are already smooth
    from the initial _velocity_match_loop pass).
    """
    bone_set = set(b for b in bone_names if b)

    for fcurve in action.fcurves:
        dp = fcurve.data_path
        if 'pose.bones["' not in dp:
            continue
        if _parse_bone_name(dp) not in bone_set:
            continue

        kf_map   = {int(kp.co[0]): kp for kp in fcurve.keyframe_points}
        first_kp = kf_map.get(frame_start)
        if first_kp is None:
            continue

        last_kp = kf_map.get(frame_end)
        if last_kp is None:
            last_kp = fcurve.keyframe_points.insert(
                frame_end, first_kp.co[1], options={'FAST', 'REPLACE'}
            )

        last_kp.co[1] = first_kp.co[1]
        fcurve.update()


def _smooth_loop_closure_frames(action, frame_start, frame_end, bone_names,
                                smooth_range=2):
    """Ease the last few frames toward the loop anchor after last == first.

    Intentionally small (2 frames default): a wide blend zone overcorrects the
    physics, creating unnatural motion in the tail. With enough preroll cycles
    the physics is already near-periodic, so only 1-2 frames need a nudge.
    Uses quaternion SLERP on the short arc (negates anchor if dot < 0).
    """
    bone_set = set(b for b in bone_names if b)
    span = frame_end - frame_start
    if span < 2:
        return
    n = min(smooth_range, span - 1)
    if n < 1:
        return

    quat_groups  = {}
    other_curves = []

    for fcurve in action.fcurves:
        dp = fcurve.data_path
        if 'pose.bones["' not in dp:
            continue
        bname = _parse_bone_name(dp)
        if bname not in bone_set:
            continue
        prop = dp.split('].')[-1] if '].' in dp else ''
        if 'rotation_quaternion' in prop:
            key = (bname, 'rotation_quaternion')
            if key not in quat_groups:
                quat_groups[key] = [None, None, None, None]
            idx = fcurve.array_index
            if 0 <= idx < 4:
                quat_groups[key][idx] = fcurve
        else:
            other_curves.append(fcurve)

    for key, channels in quat_groups.items():
        if any(c is None for c in channels):
            continue

        q_anchor = _eval_quat_normalized(channels, frame_end)
        kf_maps  = [{int(kp.co[0]): kp for kp in ch.keyframe_points} for ch in channels]

        for i in range(1, n + 1):
            f        = frame_end - i
            t        = i / (n + 1)
            slerp_w  = 1.0 - t
            q_f      = _eval_quat_normalized(channels, f)
            qa       = q_anchor.copy()
            if q_f.dot(qa) < 0.0:
                qa.negate()
            q_new = q_f.slerp(qa, slerp_w)
            if q_new.magnitude > 1e-8:
                q_new.normalize()
            for j, ch in enumerate(channels):
                kp = kf_maps[j].get(f)
                if kp is not None:
                    kp.co[1] = q_new[j]
                else:
                    ch.keyframe_points.insert(f, q_new[j], options={'FAST', 'REPLACE'})

        # Update once per channel after all frames are written (not per-frame).
        for ch in channels:
            ch.update()

    for fcurve in other_curves:
        dp    = fcurve.data_path
        bname = _parse_bone_name(dp)
        if bname not in bone_set:
            continue

        kf_map    = {int(kp.co[0]): kp for kp in fcurve.keyframe_points}
        anchor_kp = kf_map.get(frame_end)
        if anchor_kp is None:
            continue
        anchor_val = anchor_kp.co[1]

        for i in range(1, n + 1):
            kp = kf_map.get(frame_end - i)
            if kp is None:
                continue
            t = i / (n + 1)
            kp.co[1] = anchor_val * (1.0 - t) + kp.co[1] * t
        fcurve.update()


def _velocity_match_loop(action, frame_start, frame_end, bone_names):
    """Fix the loop seam for velocity continuity with minimal modification.

    For a seamless loop with LINEAR interpolation, we need:
      val[frame_end]   = val[frame_start]           (value match)
      val[frame_end-1] ≈ 2*val[start] - val[start+1] (velocity match)

    The mirror formula (2*anchor - val[start+k]) gives the exact position
    that makes the velocity entering the seam equal the velocity leaving it.

    Only 3 frames are modified:
      frame_end:   100% mirror (= anchor, exact snap)
      frame_end-1:  60% blend toward mirror target
      frame_end-2:  20% blend toward mirror target

    This preserves almost all natural physics while eliminating the seam pop.
    With well-converged preroll the baked values are already near the mirror
    targets, so the blend barely changes anything.
    """
    bone_set = set(b for b in bone_names if b)
    # (frame_offset, blend_weight) — how much to push toward mirror target
    _BLEND = ((1, 0.6), (2, 0.2))

    for fcurve in action.fcurves:
        dp = fcurve.data_path
        if 'pose.bones["' not in dp:
            continue
        if _parse_bone_name(dp) not in bone_set:
            continue

        kf_map = {int(kp.co[0]): kp for kp in fcurve.keyframe_points}
        first_kp = kf_map.get(frame_start)
        last_kp  = kf_map.get(frame_end)
        if first_kp is None or last_kp is None:
            continue

        anchor = first_kp.co[1]

        # 1. Snap last frame to first frame value (exact value match).
        last_kp.co[1] = anchor

        # 2. Blend nearby end frames toward their mirror targets.
        for k, w in _BLEND:
            kp_end   = kf_map.get(frame_end - k)
            kp_start = kf_map.get(frame_start + k)
            if kp_end is None or kp_start is None:
                continue
            mirror = 2.0 * anchor - kp_start.co[1]
            kp_end.co[1] = kp_end.co[1] * (1.0 - w) + mirror * w

        fcurve.update()


# ---------------------------------------------------------------------------
#  Keyframe interpolation / cleanup
# ---------------------------------------------------------------------------

def _set_linear_interpolation(action, bone_names):
    """Set all baked physics keyframes to LINEAR interpolation and CONSTANT extrapolation.

    LINEAR: nla.bake produces BEZIER keyframes. With one keyframe per frame,
    BEZIER creates sub-frame micro-overshoots invisible frame-by-frame but
    visible as wobble during real-time playback. LINEAR evaluates to exactly
    the straight line between adjacent samples.

    CONSTANT extrapolation: during loop playback the playhead briefly
    evaluates past the last keyframe before wrapping. Without CONSTANT
    the curve continues on the last segment's slope and shows a wrong pose
    for one sub-frame at every loop iteration.
    """
    bone_set = set(b for b in bone_names if b)

    for fcurve in action.fcurves:
        dp = fcurve.data_path
        if 'pose.bones["' not in dp:
            continue
        if _parse_bone_name(dp) not in bone_set:
            continue

        fcurve.extrapolation = 'CONSTANT'
        for kp in fcurve.keyframe_points:
            kp.interpolation = 'LINEAR'
        fcurve.update()


def _clean_tpose_keyframes(action, bone_names):
    """Force frame 0 to identity pose on wiggle bones (LoL bind pose).
    Ensures physics values never bleed into the T-pose bind frame.
    """
    bone_set = set(b for b in bone_names if b)

    for fcurve in action.fcurves:
        dp = fcurve.data_path
        if 'pose.bones["' not in dp:
            continue
        if _parse_bone_name(dp) not in bone_set:
            continue

        prop = dp.split('].')[-1] if '].' in dp else ''
        idx  = fcurve.array_index

        if 'location' in prop:
            identity = 0.0
        elif 'rotation_quaternion' in prop:
            identity = 1.0 if idx == 0 else 0.0
        elif 'rotation_euler' in prop:
            identity = 0.0
        elif 'scale' in prop:
            identity = 1.0
        else:
            continue

        kp = fcurve.keyframe_points.insert(0, identity, options={'FAST', 'REPLACE'})
        kp.interpolation = 'CONSTANT'
        fcurve.update()
