"""
Auto Boobs Physics - Automated breast jiggle physics for LoL custom skins.
Uses the Wiggle 2 physics engine to apply jiggle to breast bones,
then bakes the result into animation keyframes for in-game use.
"""

import bpy
import os
import math
from bpy.types import Panel, Operator, PropertyGroup, UIList
from bpy.props import (
    StringProperty, CollectionProperty, IntProperty,
    PointerProperty, FloatProperty, BoolProperty, EnumProperty
)
from ..ui import icons
from ..io import import_anm
from ..io import export_anm


# ---------------------------------------------------------------------------
#  Jiggle preset mapping: slider 1-20 → Wiggle 2 parameters
#  1  = barely any movement (very stiff, high damping)
#  10 = natural looking jiggle
#  20 = unrealistic, exaggerated bounce
# ---------------------------------------------------------------------------

def get_jiggle_params(intensity):
    """Map a 1-20 intensity slider to Wiggle 2 bone physics parameters.

    Returns dict with: stiff, damp, gravity, mass, stretch, chain

    The slider primarily controls bounce/oscillation via stiffness and damping.
    Gravity is kept mild across the full range so it doesn't dominate the look.
    """
    t = max(0.0, min(1.0, (intensity - 1) / 19.0))  # 0..1

    # Stiffness: high at low intensity (bones snap back fast), low at high
    # This is the main "how far do they move" knob
    stiff = lerp(800.0, 20.0, t)

    # Damping: high at low intensity (kills oscillation fast), low at high
    # Low damping = more sustained bouncing/oscillation — THE jiggle knob
    damp = lerp(15.0, 0.02, t)

    # Gravity: near-ZERO across the full range. Breast jiggle comes from
    # the body moving and bones oscillating, NOT from gravity pulling down.
    # Any noticeable gravity just makes them droop which looks bad in-game.
    gravity = lerp(0.02, 0.12, t)

    # Mass: affects momentum/inertia — heavier = more follow-through
    mass = lerp(0.4, 1.8, t)

    # Stretch: allows bone to extend past rest length
    stretch = lerp(0.0, 0.15, t)

    return {
        'stiff': stiff,
        'damp': damp,
        'gravity': gravity,
        'mass': mass,
        'stretch': stretch,
        'chain': True,
    }


def lerp(a, b, t):
    return a + (b - a) * t


# ---------------------------------------------------------------------------
#  Property group for animation list items (reused pattern from anim_loader)
# ---------------------------------------------------------------------------

class BoobsAnimListItem(PropertyGroup):
    """Single animation file entry"""
    name: StringProperty(name="Animation Name")
    filepath: StringProperty(name="File Path")


def update_search_filter(self, context):
    for area in context.screen.areas:
        if area.type == 'VIEW_3D':
            area.tag_redraw()


# ---------------------------------------------------------------------------
#  Main properties stored on the Scene
# ---------------------------------------------------------------------------

class BoobsPhysicsProperties(PropertyGroup):
    # Bone names (user picks from armature)
    breast_bone_L: StringProperty(
        name="Left Breast Bone",
        description="Name of the left breast bone in the armature",
        default=""
    )
    breast_bone_R: StringProperty(
        name="Right Breast Bone",
        description="Name of the right breast bone in the armature",
        default=""
    )

    jiggle_intensity: IntProperty(
        name="Jiggle Intensity",
        description="1 = barely any jiggle, 10 = natural, 20 = exaggerated bounce",
        default=10,
        min=1,
        max=20
    )

    # Animation folder browser
    animations: CollectionProperty(type=BoobsAnimListItem)
    active_index: IntProperty(default=0)
    animations_folder: StringProperty(name="Animations Folder", default="")
    custom_folder: StringProperty(
        name="Custom Folder",
        description="Manually selected animations folder",
        default="",
        subtype='DIR_PATH'
    )
    search_filter: StringProperty(
        name="Search",
        description="Filter animations by name",
        default="",
        update=update_search_filter,
        options={'TEXTEDIT_UPDATE'}
    )
    status_text: StringProperty(default="Ready")
    current_loaded: StringProperty(name="Currently Loaded", default="")

    # Export folder
    export_folder: StringProperty(
        name="Export Folder",
        description="Folder to export baked animations",
        default="",
        subtype='DIR_PATH'
    )

    # Processing flags
    is_processing: BoolProperty(default=False)


# ---------------------------------------------------------------------------
#  Utility helpers
# ---------------------------------------------------------------------------

def find_armature(context):
    """Find the best armature in the scene."""
    if context.active_object and context.active_object.type == 'ARMATURE':
        arm = context.active_object
        if arm.get("lol_skl_filepath") or arm.get("lol_skn_filepath"):
            return arm

    for obj in context.scene.objects:
        if obj.type == 'ARMATURE':
            if obj.get("lol_skl_filepath") or obj.get("lol_skn_filepath"):
                return obj

    for obj in context.scene.objects:
        if obj.type == 'ARMATURE':
            return obj

    return None


def get_animations_folder(armature_obj):
    """Get the animations folder path based on the armature's SKL filepath."""
    if not armature_obj:
        return None
    skl_path = armature_obj.get("lol_skl_filepath")
    if not skl_path:
        skn_path = armature_obj.get("lol_skn_filepath")
        if skn_path:
            skl_path = skn_path
    if not skl_path:
        return None
    parent_folder = os.path.dirname(skl_path)
    animations_folder = os.path.join(parent_folder, "animations")
    if os.path.isdir(animations_folder):
        return animations_folder
    return None


def ensure_object_mode(context):
    """Safely switch to object mode."""
    try:
        if context.active_object and context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
    except RuntimeError:
        pass


def select_armature(context, armature_obj):
    """Select and activate an armature."""
    ensure_object_mode(context)
    bpy.ops.object.select_all(action='DESELECT')
    armature_obj.select_set(True)
    context.view_layer.objects.active = armature_obj


def ensure_physics_registered():
    """Make sure the Wiggle 2 physics engine is registered.
    If not, register it automatically so wiggle properties exist.
    """
    if not hasattr(bpy.types.Scene, 'wiggle_enable'):
        try:
            from . import physics
            physics.register()
            print("Auto-registered Wiggle 2 physics for boobs_physics")
        except Exception as e:
            raise RuntimeError(
                f"Cannot enable jiggle physics — Wiggle 2 failed to load: {e}\n"
                "Make sure 'League Physics' is enabled in addon preferences."
            )


def apply_wiggle_to_bones(context, armature_obj, bone_names, intensity):
    """Configure Wiggle 2 physics on the given bone names.
    MUST be called with the armature active. Switches to pose mode internally
    so that wiggle property update callbacks can access selected_pose_bones.
    """
    ensure_physics_registered()
    params = get_jiggle_params(intensity)

    # Switch to pose mode — wiggle update callbacks need selected_pose_bones
    select_armature(context, armature_obj)
    bpy.ops.object.mode_set(mode='POSE')

    # Enable wiggle on scene and armature
    context.scene.wiggle_enable = True
    armature_obj.wiggle_enable = True
    armature_obj.wiggle_mute = False
    armature_obj.wiggle_freeze = False

    configured = []
    for bname in bone_names:
        if not bname:
            continue
        pb = armature_obj.pose.bones.get(bname)
        if not pb:
            continue

        # Select ONLY this bone so update callbacks iterate safely
        bpy.ops.pose.select_all(action='DESELECT')
        pb.bone.select = True
        armature_obj.data.bones.active = pb.bone

        # Enable tail wiggle on the breast bone
        pb.wiggle_tail = True
        pb.wiggle_head = False
        pb.wiggle_mute = False
        pb.wiggle_enable = True

        # Apply jiggle parameters
        pb.wiggle_stiff = params['stiff']
        pb.wiggle_damp = params['damp']
        pb.wiggle_gravity = params['gravity']
        pb.wiggle_mass = params['mass']
        pb.wiggle_stretch = params['stretch']
        pb.wiggle_chain = params['chain']

        configured.append(bname)

    # Rebuild wiggle bone list so the engine picks them up
    from . import physics
    try:
        physics.build_list()
    except Exception:
        pass

    return configured


def clear_wiggle_from_bones(context, armature_obj, bone_names):
    """Remove wiggle settings from specified bones."""
    ensure_physics_registered()

    # Need pose mode for the update callbacks
    select_armature(context, armature_obj)
    bpy.ops.object.mode_set(mode='POSE')

    for bname in bone_names:
        if not bname:
            continue
        pb = armature_obj.pose.bones.get(bname)
        if not pb:
            continue

        # Select the bone so update callbacks work
        bpy.ops.pose.select_all(action='DESELECT')
        pb.bone.select = True
        armature_obj.data.bones.active = pb.bone

        pb.wiggle_tail = False
        pb.wiggle_head = False
        pb.wiggle_enable = False

    from . import physics
    try:
        physics.build_list()
    except Exception:
        pass


def _detect_animation_loops(action, frame_start, frame_end):
    """Detect whether an animation loops by comparing f-curve values at
    the first and last frames. Non-looping animations (death, spawn, etc.)
    have drastically different poses at start vs end.
    """
    diffs = []
    for fcurve in action.fcurves:
        dp = fcurve.data_path
        if 'pose.bones["' not in dp:
            continue
        # Only look at location and rotation channels for detection
        if not any(p in dp for p in ['location', 'rotation']):
            continue
        val_s = fcurve.evaluate(frame_start)
        val_e = fcurve.evaluate(frame_end)
        diffs.append(abs(val_e - val_s))

    if not diffs:
        return True  # assume looping if we can't tell

    avg = sum(diffs) / len(diffs)
    return avg < 0.05


def bake_wiggle_for_current_action(context, armature_obj, bone_names, intensity):
    """Bake wiggle physics into the current action.

    Auto-detects whether the animation loops:
    - Looping anims (idle, walk, dance): 3-cycle preroll + seamless blend
    - Non-looping anims (death, spawn): 1-pass preroll, no blend

    Always cleans up frame 0 (T-pose) so physics values never bleed into it.
    """
    scene = context.scene

    if not armature_obj.animation_data or not armature_obj.animation_data.action:
        return False, "No animation loaded on armature"

    action = armature_obj.animation_data.action
    frame_start = max(1, int(action.frame_range[0]))
    frame_end = int(action.frame_range[1])

    if frame_end <= frame_start:
        return False, "Animation has no frames"

    # Frame 0 = bind/T-pose, frame 1 = first real animation frame
    scene.frame_start = 1
    scene.frame_end = frame_end

    # Detect loop BEFORE physics bake (uses raw animation keyframes)
    is_loop = _detect_animation_loops(action, frame_start, frame_end)

    # Must be in pose mode with the armature active
    select_armature(context, armature_obj)
    bpy.ops.object.mode_set(mode='POSE')

    # Unfreeze (needed if previous bake froze it)
    armature_obj.wiggle_freeze = False

    # loop=True prevents the wiggle handler from resetting physics at frame 1
    scene.wiggle.loop = True
    scene.wiggle.bake_overwrite = True

    # --- Step 1: Reset physics to clean state ---
    bpy.ops.wiggle.reset()

    # --- Step 2: Forward preroll ---
    # Looping anims: cycle 3× so physics reaches steady-state for seamless loop
    # Non-looping anims: single forward pass — physics starts from rest which
    # is correct (character starts in a neutral standing pose)
    preroll_cycles = 3 if is_loop else 1
    scene.wiggle.is_preroll = True
    for _cycle in range(preroll_cycles):
        for f in range(frame_start, frame_end + 1):
            scene.frame_set(f)
    scene.wiggle.is_preroll = False

    # --- Step 3: Select wiggle bones for baking ---
    bpy.ops.wiggle.select()

    # --- Step 4: Bake one full pass (keyframes are recorded) ---
    try:
        bpy.ops.nla.bake(
            frame_start=frame_start,
            frame_end=frame_end,
            only_selected=True,
            visual_keying=True,
            use_current_action=True,
            bake_types={'POSE'}
        )
    except Exception as e:
        return False, f"NLA bake failed: {str(e)}"

    # --- Step 5: Post-bake cleanup ---
    action = armature_obj.animation_data.action
    if action:
        # Smooth boundary frames to dampen any remaining first/last frame spikes
        _smooth_boundary_frames(action, frame_start, frame_end, bone_names)

        # Seamless blend only for looping animations
        if is_loop:
            _seamless_blend_physics_loop(action, frame_start, frame_end, bone_names)

        # ALWAYS clean the T-pose (frame 0) so baked boob positions don't bleed in
        _clean_tpose_keyframes(action, bone_names)

    armature_obj.wiggle_freeze = True
    return True, "Baked successfully"


def _smooth_boundary_frames(action, frame_start, frame_end, bone_names,
                             smooth_range=4):
    """Smooth the first and last few baked frames to remove physics spikes.

    At animation boundaries the physics can produce momentary overshoots
    (e.g. the spring snapping from preroll state). This function detects
    keyframes that deviate wildly from their neighbours and pulls them
    toward the local average, keeping the middle of the animation untouched.
    """
    bone_set = set(b for b in bone_names if b)

    for fcurve in action.fcurves:
        dp = fcurve.data_path
        if 'pose.bones["' not in dp:
            continue
        try:
            bname = dp.split('pose.bones["')[1].split('"]')[0]
        except Exception:
            continue
        if bname not in bone_set:
            continue

        kf_map = {int(kp.co[0]): kp for kp in fcurve.keyframe_points}

        # --- Smooth the FIRST smooth_range frames ---
        # Use the value at (frame_start + smooth_range) as the "settled" target
        settled_kp = kf_map.get(frame_start + smooth_range)
        if settled_kp:
            settled_val = settled_kp.co[1]
            for i in range(smooth_range):
                f = frame_start + i
                kp = kf_map.get(f)
                if kp is None:
                    continue
                # Blend factor: 0 at frame_start (full smoothing) → 1 at boundary
                t = (i + 1) / (smooth_range + 1)
                kp.co[1] = settled_val * (1.0 - t) + kp.co[1] * t

        # --- Smooth the LAST smooth_range frames ---
        # Use the value at (frame_end - smooth_range) as the "settled" target
        settled_kp = kf_map.get(frame_end - smooth_range)
        if settled_kp:
            settled_val = settled_kp.co[1]
            for i in range(smooth_range):
                f = frame_end - i
                kp = kf_map.get(f)
                if kp is None:
                    continue
                t = (i + 1) / (smooth_range + 1)
                kp.co[1] = settled_val * (1.0 - t) + kp.co[1] * t

        fcurve.update()


def _seamless_blend_physics_loop(action, frame_start, frame_end, bone_names):
    """Smoothly blend the last few baked frames toward frame 1 values for a
    perfect animation loop. Uses a gradual blend instead of a hard copy so
    there's no visible pop at the loop boundary.
    """
    bone_set = set(b for b in bone_names if b)
    duration = frame_end - frame_start
    # Blend zone = ~8% of the animation, minimum 2 frames, maximum 6
    blend_count = max(2, min(6, int(duration * 0.08)))

    for fcurve in action.fcurves:
        dp = fcurve.data_path
        if 'pose.bones["' not in dp:
            continue
        try:
            bname = dp.split('pose.bones["')[1].split('"]')[0]
        except Exception:
            continue
        if bname not in bone_set:
            continue

        # Collect keyframe values by frame number
        kf_map = {int(kp.co[0]): kp for kp in fcurve.keyframe_points}

        first_kp = kf_map.get(frame_start)
        if first_kp is None:
            continue
        target_val = first_kp.co[1]

        # Blend the last blend_count frames toward the target value
        for i in range(1, blend_count + 1):
            f = frame_end - blend_count + i
            kp = kf_map.get(f)
            if kp is None:
                continue
            t = i / float(blend_count)
            kp.co[1] = kp.co[1] * (1.0 - t) + target_val * t

        # Ensure exact match on the very last frame
        last_kp = kf_map.get(frame_end)
        if last_kp:
            last_kp.co[1] = target_val
        else:
            fcurve.keyframe_points.insert(frame_end, target_val,
                                          options={'FAST', 'REPLACE'})
        fcurve.update()


def _clean_tpose_keyframes(action, bone_names):
    """Insert identity-transform keyframes at frame 0 for breast bones.
    This prevents baked physics values from bleeding into the T-pose when
    scrubbing to frame 0 in the timeline or during export.
    Uses CONSTANT interpolation so there's a hard cut between T-pose and
    the first animation frame — no weird blending.
    """
    bone_set = set(b for b in bone_names if b)

    for fcurve in action.fcurves:
        dp = fcurve.data_path
        if 'pose.bones["' not in dp:
            continue
        try:
            bname = dp.split('pose.bones["')[1].split('"]')[0]
        except Exception:
            continue
        if bname not in bone_set:
            continue

        # Figure out the identity value for this channel
        prop = dp.split('].')[-1] if '].' in dp else ''
        idx = fcurve.array_index

        if 'location' in prop:
            identity = 0.0
        elif 'rotation_quaternion' in prop:
            identity = 1.0 if idx == 0 else 0.0  # (w=1, x=0, y=0, z=0)
        elif 'rotation_euler' in prop:
            identity = 0.0
        elif 'scale' in prop:
            identity = 1.0
        else:
            continue

        # Insert at frame 0 with CONSTANT interpolation
        kp = fcurve.keyframe_points.insert(0, identity, options={'FAST', 'REPLACE'})
        kp.interpolation = 'CONSTANT'
        fcurve.update()


# ---------------------------------------------------------------------------
#  Operators
# ---------------------------------------------------------------------------

class BOOBS_OT_AutoDetectBones(Operator):
    """Try to auto-detect breast bones from common naming conventions"""
    bl_idname = "boobs_physics.auto_detect"
    bl_label = "Auto-Detect Breast Bones"
    bl_description = "Scan the armature for common breast bone names"
    bl_options = {'REGISTER'}

    # Common naming patterns for breast bones in LoL models
    PATTERNS_LEFT = [
        'L_Breast', 'Breast_L', 'breast_l', 'l_breast',
        'L_Boob', 'Boob_L', 'boob_l', 'l_boob',
        'Breast1_L', 'L_Breast1', 'Breast_01_L',
        'L_Bust', 'Bust_L', 'bust_l',
        'C_Buffbone_Glb_Chest_Loc',  # some LoL models
        'Buffbone_Glb_Chest_Loc',
    ]
    PATTERNS_RIGHT = [
        'R_Breast', 'Breast_R', 'breast_r', 'r_breast',
        'R_Boob', 'Boob_R', 'boob_r', 'r_boob',
        'Breast1_R', 'R_Breast1', 'Breast_01_R',
        'R_Bust', 'Bust_R', 'bust_r',
    ]

    def execute(self, context):
        props = context.scene.boobs_physics
        armature_obj = find_armature(context)

        if not armature_obj:
            self.report({'ERROR'}, "No armature found in scene")
            return {'CANCELLED'}

        bone_names = [b.name for b in armature_obj.pose.bones]

        # Try exact matches first
        found_l = ""
        found_r = ""

        for pattern in self.PATTERNS_LEFT:
            if pattern in bone_names:
                found_l = pattern
                break

        for pattern in self.PATTERNS_RIGHT:
            if pattern in bone_names:
                found_r = pattern
                break

        # If exact match failed, try case-insensitive substring match
        if not found_l:
            for bname in bone_names:
                bl = bname.lower()
                if ('breast' in bl or 'boob' in bl or 'bust' in bl) and ('_l' in bl or 'l_' in bl or bl.endswith('_l') or bl.startswith('l_') or '.l' in bl):
                    found_l = bname
                    break

        if not found_r:
            for bname in bone_names:
                bl = bname.lower()
                if ('breast' in bl or 'boob' in bl or 'bust' in bl) and ('_r' in bl or 'r_' in bl or bl.endswith('_r') or bl.startswith('r_') or '.r' in bl):
                    found_r = bname
                    break

        if found_l:
            props.breast_bone_L = found_l
        if found_r:
            props.breast_bone_R = found_r

        if found_l or found_r:
            msg = f"Found: {found_l or '(none)'} + {found_r or '(none)'}"
            self.report({'INFO'}, msg)
        else:
            self.report({'WARNING'}, "Could not auto-detect breast bones. Please select them manually.")

        return {'FINISHED'}


class BOOBS_OT_PreviewPhysics(Operator):
    """Apply jiggle physics to the current animation for preview"""
    bl_idname = "boobs_physics.preview"
    bl_label = "Preview Jiggle"
    bl_description = "Apply jiggle physics to the current animation and bake it for preview"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.boobs_physics
        armature_obj = find_armature(context)

        if not armature_obj:
            self.report({'ERROR'}, "No armature found in scene")
            return {'CANCELLED'}

        bone_names = [props.breast_bone_L, props.breast_bone_R]
        bone_names = [b for b in bone_names if b]

        if not bone_names:
            self.report({'ERROR'}, "No breast bones selected. Use Auto-Detect or pick them manually.")
            return {'CANCELLED'}

        if not armature_obj.animation_data or not armature_obj.animation_data.action:
            self.report({'ERROR'}, "No animation loaded. Load an animation first.")
            return {'CANCELLED'}

        # Step 1: Unfreeze if previously baked
        armature_obj.wiggle_freeze = False

        # Step 2: Configure wiggle on the breast bones (switches to pose mode)
        configured = apply_wiggle_to_bones(context, armature_obj, bone_names, props.jiggle_intensity)
        if not configured:
            self.report({'ERROR'}, "Could not find the specified bones in the armature")
            return {'CANCELLED'}

        # Step 3: Bake the physics (uses wiggle.bake with preroll)
        props.status_text = "Baking physics preview..."
        success, message = bake_wiggle_for_current_action(
            context, armature_obj, bone_names, props.jiggle_intensity
        )

        if not success:
            props.status_text = f"Preview failed: {message}"
            self.report({'ERROR'}, message)
            return {'CANCELLED'}

        # Step 4: Clean up - disable wiggle on bones (keyframes are already baked)
        clear_wiggle_from_bones(context, armature_obj, bone_names)

        props.status_text = "Preview ready — play the animation!"
        self.report({'INFO'}, f"Jiggle physics baked on: {', '.join(configured)}")

        # Go back to object mode for playback
        ensure_object_mode(context)

        return {'FINISHED'}


class BOOBS_OT_UndoPreview(Operator):
    """Undo the jiggle preview by removing baked physics keyframes"""
    bl_idname = "boobs_physics.undo_preview"
    bl_label = "Undo Preview"
    bl_description = "Remove baked jiggle keyframes from the current animation"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.boobs_physics
        armature_obj = find_armature(context)

        if not armature_obj:
            self.report({'ERROR'}, "No armature found")
            return {'CANCELLED'}

        if not armature_obj.animation_data or not armature_obj.animation_data.action:
            self.report({'WARNING'}, "No animation loaded")
            return {'CANCELLED'}

        # Unfreeze
        armature_obj.wiggle_freeze = False

        # Clear wiggle from bones (needs pose mode internally)
        bone_names = [props.breast_bone_L, props.breast_bone_R]
        clear_wiggle_from_bones(context, armature_obj, bone_names)

        props.status_text = "Ready"
        self.report({'INFO'}, "Wiggle state cleared. Use Ctrl+Z to undo baked keyframes.")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
#  Animation folder browsing operators (pattern from anim_loader)
# ---------------------------------------------------------------------------

class BOOBS_OT_BrowseFolder(Operator):
    """Browse for animation folder"""
    bl_idname = "boobs_physics.browse_folder"
    bl_label = "Browse Animations Folder"
    bl_description = "Choose a folder containing .anm animation files"
    bl_options = {'REGISTER'}

    directory: StringProperty(subtype='DIR_PATH')

    def execute(self, context):
        props = context.scene.boobs_physics
        if self.directory:
            props.custom_folder = self.directory.rstrip('/\\')
            bpy.ops.boobs_physics.refresh_anims()
        return {'FINISHED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


class BOOBS_OT_BrowseExportFolder(Operator):
    """Browse for export folder"""
    bl_idname = "boobs_physics.browse_export_folder"
    bl_label = "Browse Export Folder"
    bl_description = "Choose a folder to export baked animations"
    bl_options = {'REGISTER'}

    directory: StringProperty(subtype='DIR_PATH')

    def execute(self, context):
        props = context.scene.boobs_physics
        if self.directory:
            props.export_folder = self.directory.rstrip('/\\')
        return {'FINISHED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


class BOOBS_OT_ClearFolder(Operator):
    """Clear custom folder"""
    bl_idname = "boobs_physics.clear_folder"
    bl_label = "Clear Custom Folder"
    bl_options = {'REGISTER'}

    def execute(self, context):
        props = context.scene.boobs_physics
        props.custom_folder = ""
        bpy.ops.boobs_physics.refresh_anims()
        return {'FINISHED'}


class BOOBS_OT_RefreshAnims(Operator):
    """Scan the animations folder and refresh the list"""
    bl_idname = "boobs_physics.refresh_anims"
    bl_label = "Refresh Animations"
    bl_options = {'REGISTER'}

    def execute(self, context):
        props = context.scene.boobs_physics
        props.animations.clear()
        props.animations_folder = ""
        props.search_filter = ""

        if props.custom_folder and os.path.isdir(props.custom_folder):
            anim_folder = props.custom_folder
        else:
            armature_obj = find_armature(context)
            if not armature_obj:
                self.report({'WARNING'}, "No armature found. Use folder button to select manually.")
                return {'CANCELLED'}
            anim_folder = get_animations_folder(armature_obj)
            if not anim_folder:
                self.report({'WARNING'}, "No 'animations' folder found. Use folder button.")
                return {'CANCELLED'}

        props.animations_folder = anim_folder

        anm_files = sorted([f for f in os.listdir(anim_folder) if f.lower().endswith('.anm')])

        for filename in anm_files:
            item = props.animations.add()
            item.name = os.path.splitext(filename)[0]
            item.filepath = os.path.join(anim_folder, filename)

        if anm_files:
            self.report({'INFO'}, f"Found {len(anm_files)} animations")
        else:
            self.report({'WARNING'}, f"No .anm files in: {anim_folder}")

        return {'FINISHED'}


class BOOBS_OT_LoadAnimation(Operator):
    """Load a single animation for preview"""
    bl_idname = "boobs_physics.load_anim"
    bl_label = "Load Animation"
    bl_options = {'REGISTER', 'UNDO'}

    filepath: StringProperty()
    anim_name: StringProperty()
    index: IntProperty(default=-1)

    def execute(self, context):
        props = context.scene.boobs_physics

        if not self.filepath or not os.path.exists(self.filepath):
            self.report({'ERROR'}, "Animation file not found")
            return {'CANCELLED'}

        armature_obj = find_armature(context)
        if not armature_obj:
            self.report({'ERROR'}, "No armature found")
            return {'CANCELLED'}

        select_armature(context, armature_obj)

        # Unfreeze wiggle
        armature_obj.wiggle_freeze = False

        try:
            action_name = self.anim_name or os.path.splitext(os.path.basename(self.filepath))[0]
            anm = import_anm.read_anm(self.filepath)

            if not armature_obj.animation_data:
                armature_obj.animation_data_create()

            new_action = bpy.data.actions.new(name=action_name)
            armature_obj.animation_data.action = new_action

            import_anm.apply_anm(anm, armature_obj, frame_offset=0)

            new_action["lol_anm_filepath"] = self.filepath
            new_action["lol_anm_filename"] = os.path.basename(self.filepath)

            props.current_loaded = action_name

            if self.index >= 0:
                props.active_index = self.index

            props.status_text = "Ready"
            self.report({'INFO'}, f"Loaded: {action_name}")
            return {'FINISHED'}

        except Exception as e:
            props.status_text = "Import failed"
            self.report({'ERROR'}, f"Failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}


class BOOBS_OT_ApplyToAll(Operator):
    """Apply jiggle physics to all animations in the folder, bake, and export"""
    bl_idname = "boobs_physics.apply_all"
    bl_label = "Apply to All Animations"
    bl_description = "Import each animation, apply jiggle, bake, and export to the output folder"
    bl_options = {'REGISTER'}

    def execute(self, context):
        props = context.scene.boobs_physics
        armature_obj = find_armature(context)

        if not armature_obj:
            self.report({'ERROR'}, "No armature found in scene")
            return {'CANCELLED'}

        bone_names = [props.breast_bone_L, props.breast_bone_R]
        bone_names = [b for b in bone_names if b]

        if not bone_names:
            self.report({'ERROR'}, "No breast bones selected")
            return {'CANCELLED'}

        if len(props.animations) == 0:
            self.report({'ERROR'}, "No animations loaded. Click Refresh first.")
            return {'CANCELLED'}

        # Determine export folder
        export_dir = props.export_folder
        if not export_dir:
            # Default: same as source folder
            export_dir = props.animations_folder
        if not export_dir or not os.path.isdir(export_dir):
            self.report({'ERROR'}, "No valid export folder. Set one or ensure animations folder exists.")
            return {'CANCELLED'}

        select_armature(context, armature_obj)

        if not armature_obj.animation_data:
            armature_obj.animation_data_create()

        # Store original action
        original_action = armature_obj.animation_data.action

        total = len(props.animations)
        success_count = 0
        fail_count = 0
        fps = context.scene.render.fps

        # Configure wiggle on breast bones ONCE (properties persist on PoseBone)
        apply_wiggle_to_bones(context, armature_obj, bone_names, props.jiggle_intensity)

        for idx, anim_item in enumerate(props.animations):
            props.status_text = f"Processing {idx + 1}/{total}: {anim_item.name}..."
            bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)

            try:
                # 1. Unfreeze (wiggle.bake freezes after each bake)
                armature_obj.wiggle_freeze = False

                # 2. Import animation — must go to object mode first
                ensure_object_mode(context)
                select_armature(context, armature_obj)

                anm = import_anm.read_anm(anim_item.filepath)
                action_name = anim_item.name
                new_action = bpy.data.actions.new(name=action_name)
                armature_obj.animation_data.action = new_action
                import_anm.apply_anm(anm, armature_obj, frame_offset=0)
                new_action["lol_anm_filepath"] = anim_item.filepath

                # 3. Bake physics (uses wiggle.bake with preroll)
                ok, msg = bake_wiggle_for_current_action(
                    context, armature_obj, bone_names, props.jiggle_intensity
                )
                if not ok:
                    print(f"Bake failed for {anim_item.name}: {msg}")
                    fail_count += 1
                    continue

                # 4. Export to ANM (baked keyframes are in the action)
                out_path = os.path.join(export_dir, f"{anim_item.name}.anm")
                export_anm.write_anm(out_path, armature_obj, fps)

                success_count += 1

            except Exception as e:
                print(f"Failed on {anim_item.name}: {e}")
                import traceback
                traceback.print_exc()
                fail_count += 1

        # Clean up wiggle from bones after all done
        clear_wiggle_from_bones(context, armature_obj, bone_names)

        # Restore original action
        try:
            armature_obj.animation_data.action = original_action
        except Exception:
            pass

        armature_obj.wiggle_freeze = False
        ensure_object_mode(context)
        props.status_text = "Ready"

        if fail_count > 0:
            self.report({'WARNING'}, f"Done: {success_count}/{total} exported ({fail_count} failed) to {export_dir}")
        else:
            self.report({'INFO'}, f"Done: {success_count} animations exported to {export_dir}")

        return {'FINISHED'}


# ---------------------------------------------------------------------------
#  UI List for animations
# ---------------------------------------------------------------------------

class BOOBS_UL_AnimList(UIList):
    """Scrollable animation list with filtering"""

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        props = context.scene.boobs_physics

        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            row = layout.row(align=True)
            if item.name == props.current_loaded:
                row.label(text="", icon='PLAY')
            else:
                row.label(text="", icon='ACTION')

            op = row.operator("boobs_physics.load_anim", text=item.name, emboss=False)
            op.filepath = item.filepath
            op.anim_name = item.name
            op.index = index

        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text=item.name, icon='ACTION')

    def filter_items(self, context, data, propname):
        props = context.scene.boobs_physics
        items = getattr(data, propname)
        filter_name = props.search_filter.lower()

        flt_flags = [self.bitflag_filter_item] * len(items)
        flt_neworder = []

        if filter_name:
            for i, item in enumerate(items):
                if filter_name not in item.name.lower():
                    flt_flags[i] = 0

        return flt_flags, flt_neworder


# ---------------------------------------------------------------------------
#  Panel
# ---------------------------------------------------------------------------

class BOOBS_PT_MainPanel(Panel):
    """Auto Physics panel in the 3D viewport sidebar"""
    bl_label = "Auto Physics"
    bl_idname = "VIEW3D_PT_auto_physics"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Auto Physics'

    def draw_header(self, context):
        layout = self.layout
        layout.label(text="", icon_value=icons.get_icon("icon_52"))

    def draw(self, context):
        layout = self.layout
        layout.label(text="Select a sub-panel below.", icon='INFO')


class BOOBS_PT_BoobsPhysics(Panel):
    """Boobs Physics sub-panel"""
    bl_label = "Boobs Physics"
    bl_idname = "VIEW3D_PT_boobs_physics"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Auto Physics'
    bl_parent_id = "VIEW3D_PT_auto_physics"

    def draw(self, context):
        layout = self.layout
        props = context.scene.boobs_physics
        armature_obj = find_armature(context)

        # Status
        box = layout.box()
        box.label(text=props.status_text, icon='INFO')

        # --- Bone selection ---
        box = layout.box()
        box.label(text="Breast Bones", icon='BONE_DATA')

        if armature_obj:
            row = box.row(align=True)
            row.prop_search(props, "breast_bone_L", armature_obj.pose, "bones", text="Left")

            row = box.row(align=True)
            row.prop_search(props, "breast_bone_R", armature_obj.pose, "bones", text="Right")

            box.operator("boobs_physics.auto_detect", text="Auto-Detect", icon='VIEWZOOM')
        else:
            box.label(text="No armature in scene", icon='ERROR')

        # --- Jiggle intensity slider ---
        box = layout.box()
        box.label(text="Jiggle Settings", icon='MOD_WAVE')

        col = box.column(align=True)
        col.prop(props, "jiggle_intensity", slider=True)

        # Labels indicating the scale
        row = col.row(align=True)
        row.alignment = 'CENTER'
        intensity = props.jiggle_intensity
        if intensity <= 3:
            row.label(text="▸ Subtle", icon='NONE')
        elif intensity <= 7:
            row.label(text="▸ Mild", icon='NONE')
        elif intensity <= 13:
            row.label(text="▸ Natural", icon='NONE')
        elif intensity <= 17:
            row.label(text="▸ Bouncy", icon='NONE')
        else:
            row.label(text="▸ Extreme!", icon='NONE')

        # --- Preview controls ---
        box = layout.box()
        box.label(text="Preview", icon='PLAY')

        has_anim = bool(armature_obj and armature_obj.animation_data and armature_obj.animation_data.action)
        has_bones = bool(props.breast_bone_L or props.breast_bone_R)

        col = box.column(align=True)
        col.scale_y = 1.3

        row = col.row(align=True)
        row.enabled = has_anim and has_bones
        row.operator("boobs_physics.preview", text="Preview Jiggle", icon='MOD_WAVE')

        row = col.row(align=True)
        row.enabled = has_anim
        row.operator("boobs_physics.undo_preview", text="Undo Preview", icon='LOOP_BACK')

        if props.current_loaded:
            box.label(text=f"Animation: {props.current_loaded}", icon='ANIM')

        # --- Animation folder & batch processing ---
        box = layout.box()
        box.label(text="Batch Processing", icon='FILE_FOLDER')

        # Folder controls
        row = box.row(align=True)
        row.operator("boobs_physics.refresh_anims", text="Refresh", icon='FILE_REFRESH')
        row.operator("boobs_physics.browse_folder", text="", icon='FILEBROWSER')

        # Show current folder
        row = box.row(align=True)
        row.scale_y = 0.7
        if props.animations_folder:
            folder_name = os.path.basename(props.animations_folder)
            parent_name = os.path.basename(os.path.dirname(props.animations_folder))
            if props.custom_folder:
                row.label(text=f".../{parent_name}/{folder_name}", icon='FILE_FOLDER')
                row.operator("boobs_physics.clear_folder", text="", icon='X')
            else:
                row.label(text=f".../{parent_name}/{folder_name}", icon='FILE_FOLDER')
        else:
            row.label(text="No folder selected", icon='FILE_FOLDER')

        # Animation list
        if len(props.animations) > 0:
            filter_text = props.search_filter.lower()
            if filter_text:
                visible = sum(1 for item in props.animations if filter_text in item.name.lower())
                label_text = f"Animations ({visible}/{len(props.animations)})"
            else:
                label_text = f"Animations ({len(props.animations)})"

            row = box.row()
            row.label(text=label_text, icon='ANIM')

            row = box.row()
            row.template_list(
                "BOOBS_UL_AnimList", "",
                props, "animations",
                props, "active_index",
                rows=8
            )

            row = box.row(align=True)
            row.prop(props, "search_filter", text="", icon='VIEWZOOM')

        # Export folder
        layout.separator()
        box = layout.box()
        box.label(text="Export", icon='EXPORT')

        row = box.row(align=True)
        row.label(text="Output Folder:", icon='FILE_FOLDER')
        row.operator("boobs_physics.browse_export_folder", text="", icon='FILEBROWSER')

        if props.export_folder:
            row = box.row(align=True)
            row.scale_y = 0.7
            folder_name = os.path.basename(props.export_folder)
            parent_name = os.path.basename(os.path.dirname(props.export_folder))
            row.label(text=f".../{parent_name}/{folder_name}")
        else:
            row = box.row(align=True)
            row.scale_y = 0.7
            row.label(text="(defaults to source folder)")

        # Apply to all button
        layout.separator()
        col = layout.column(align=True)
        col.scale_y = 1.5
        col.enabled = has_bones and len(props.animations) > 0
        col.operator("boobs_physics.apply_all", text="⚡ Apply to All Animations", icon='MOD_WAVE')


# ---------------------------------------------------------------------------
#  Registration
# ---------------------------------------------------------------------------

classes = [
    BoobsAnimListItem,
    BoobsPhysicsProperties,
    BOOBS_OT_AutoDetectBones,
    BOOBS_OT_PreviewPhysics,
    BOOBS_OT_UndoPreview,
    BOOBS_OT_BrowseFolder,
    BOOBS_OT_BrowseExportFolder,
    BOOBS_OT_ClearFolder,
    BOOBS_OT_RefreshAnims,
    BOOBS_OT_LoadAnimation,
    BOOBS_OT_ApplyToAll,
    BOOBS_UL_AnimList,
    BOOBS_PT_MainPanel,
    BOOBS_PT_BoobsPhysics,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.boobs_physics = PointerProperty(type=BoobsPhysicsProperties)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    if hasattr(bpy.types.Scene, 'boobs_physics'):
        del bpy.types.Scene.boobs_physics
