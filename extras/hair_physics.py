"""
Hair Physics - Automated hair jiggle physics for LoL custom skins.
Same Wiggle 2 engine as boobs_physics but tuned for hair chains:
higher gravity, lighter mass, stronger chain coupling.
"""

import bpy
import os
import math
from bpy.types import Panel, Operator, PropertyGroup, UIList
from bpy.props import (
    StringProperty, CollectionProperty, IntProperty,
    PointerProperty, FloatProperty, BoolProperty
)
from ..ui import icons
from ..io import import_anm
from ..io import export_anm

def update_hair_search_filter(self, context):
    pass


# ---------------------------------------------------------------------------
#  Hair physics parameter mapping  (intensity 1-20)
# ---------------------------------------------------------------------------

def _lerp(a, b, t):
    return a + (b - a) * t

def _lerp_exp(a, b, t):
    return math.exp(math.log(a) + (math.log(b) - math.log(a)) * t)

def get_hair_params(intensity):
    """Map 1-20 slider to Wiggle 2 params tuned for hair bones.

    Hair differences vs breast bones:
      - gravity is noticeable (hair hangs/swings visibly)
      - stiffness is lower overall (hair is more flexible)
      - chain coupling is critical (bones in a chain pull each other)
      - mass is lighter (hair strands aren't heavy)
      - damping floor is slightly higher to prevent wild fly-away
    """
    t = max(0.0, min(1.0, (intensity - 1) / 19.0))
    t = t * t  # quadratic bias: natural feel in the lower half

    return {
        'stiff':   _lerp_exp(400.0, 40.0, t),   # lower than boobs; hair is flexible
        'damp':    _lerp_exp(8.0, 0.35, t),      # floor 0.35; prevents fly-away
        'gravity': _lerp(0.05, 0.35, t),         # visible gravity — hair hangs
        'mass':    _lerp_exp(0.15, 1.2, t),      # light strands
        'stretch': _lerp(0.0, 0.04, t),          # minimal stretch
        'chain':   True,
    }


# ---------------------------------------------------------------------------
#  Properties
# ---------------------------------------------------------------------------

class HairBoneItem(PropertyGroup):
    """One entry in the hair bone list."""
    bone_name: StringProperty(name="Bone", default="")


class HairAnimListItem(PropertyGroup):
    name: StringProperty()
    filepath: StringProperty()


class HairPhysicsProperties(PropertyGroup):
    bones: CollectionProperty(type=HairBoneItem)
    active_bone_index: IntProperty(default=0)

    jiggle_intensity: IntProperty(
        name="Hair Intensity",
        description="1 = barely moves, 10 = natural sway, 20 = exaggerated",
        default=8, min=1, max=20
    )
    status_text: StringProperty(default="Ready")
    backup_action_name: StringProperty(default="")

    # Animation folder browser
    animations: CollectionProperty(type=HairAnimListItem)
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
        update=update_hair_search_filter,
        options={'TEXTEDIT_UPDATE'}
    )
    current_loaded: StringProperty(name="Currently Loaded", default="")

    # Export folder
    export_folder: StringProperty(
        name="Export Folder",
        description="Folder to export baked animations",
        default="",
        subtype='DIR_PATH'
    )


# ---------------------------------------------------------------------------
#  Helpers  (re-use boobs_physics utilities where possible)
# ---------------------------------------------------------------------------

def _find_armature(context):
    from .boobs_physics import find_armature
    return find_armature(context)

def _ensure_object_mode(context):
    from .boobs_physics import ensure_object_mode
    ensure_object_mode(context)

def _select_armature(context, arm):
    from .boobs_physics import select_armature
    select_armature(context, arm)

def _get_bone_names(props):
    return [item.bone_name for item in props.bones if item.bone_name.strip()]

def _apply_wiggle(context, arm, bone_names, intensity):
    from .boobs_physics import ensure_physics_registered, apply_wiggle_to_bones
    ensure_physics_registered()
    params = get_hair_params(intensity)

    _select_armature(context, arm)
    bpy.ops.object.mode_set(mode='POSE')

    context.scene.wiggle_enable = True
    arm.wiggle_enable = True
    arm.wiggle_mute = False
    arm.wiggle_freeze = False

    configured = []
    for bname in bone_names:
        pb = arm.pose.bones.get(bname)
        if not pb:
            continue
        bpy.ops.pose.select_all(action='DESELECT')
        pb.bone.select = True
        arm.data.bones.active = pb.bone

        pb.wiggle_tail    = True
        pb.wiggle_head    = False
        pb.wiggle_mute    = False
        pb.wiggle_enable  = True
        pb.wiggle_stiff   = params['stiff']
        pb.wiggle_damp    = params['damp']
        pb.wiggle_gravity = params['gravity']
        pb.wiggle_mass    = params['mass']
        pb.wiggle_stretch = params['stretch']
        pb.wiggle_chain   = params['chain']
        configured.append(bname)

    from . import physics
    try:
        physics.build_list()
    except Exception:
        pass

    return configured

def _clear_wiggle(context, arm, bone_names):
    from .boobs_physics import clear_wiggle_from_bones
    clear_wiggle_from_bones(context, arm, bone_names)

def _strip_fcurves(action, bone_names):
    bone_set = set(b for b in bone_names if b)
    remove = []
    for fc in action.fcurves:
        if 'pose.bones["' not in fc.data_path:
            continue
        try:
            bname = fc.data_path.split('pose.bones["')[1].split('"]')[0]
        except Exception:
            continue
        if bname in bone_set:
            remove.append(fc)
    for fc in remove:
        action.fcurves.remove(fc)

def _bake_hair(context, arm, bone_names, intensity):
    """Detect loop, preroll, bake — same pipeline as boobs_physics."""
    from .boobs_physics import (
        _detect_animation_loops,
        _smooth_boundary_frames,
        _seamless_blend_physics_loop,
        _restore_nonloop_start_to_tpose,
        _clean_tpose_keyframes,
    )
    scene = context.scene
    if not arm.animation_data or not arm.animation_data.action:
        return False, "No animation loaded"

    action = arm.animation_data.action
    frame_start = max(1, int(action.frame_range[0]))
    frame_end   = int(action.frame_range[1])
    if frame_end <= frame_start:
        return False, "Animation has no frames"

    scene.frame_start = 1
    scene.frame_end   = frame_end

    is_loop = _detect_animation_loops(action, frame_start, frame_end)

    # Same non-loop cap pattern as boobs
    eff = intensity if is_loop else min(intensity, 13)
    if eff != intensity:
        _apply_wiggle(context, arm, bone_names, eff)

    _select_armature(context, arm)
    bpy.ops.object.mode_set(mode='POSE')
    arm.wiggle_freeze = False
    scene.wiggle.loop = True
    scene.wiggle.bake_overwrite = True
    bpy.ops.wiggle.reset()

    preroll_cycles = 3 if is_loop else 1
    scene.wiggle.is_preroll = True
    for _ in range(preroll_cycles):
        for f in range(frame_start, frame_end + 1):
            scene.frame_set(f)
    scene.wiggle.is_preroll = False

    bpy.ops.wiggle.select()
    try:
        bpy.ops.nla.bake(
            frame_start=frame_start, frame_end=frame_end,
            only_selected=True, visual_keying=True,
            use_current_action=True, bake_types={'POSE'}
        )
    except Exception as e:
        return False, f"Bake failed: {e}"

    action = arm.animation_data.action
    if action:
        if is_loop:
            _smooth_boundary_frames(action, frame_start, frame_end, bone_names, smooth_ends='both')
            _seamless_blend_physics_loop(action, frame_start, frame_end, bone_names)
        else:
            _restore_nonloop_start_to_tpose(action, frame_start, bone_names)
            _smooth_boundary_frames(action, frame_start, frame_end, bone_names, smooth_ends='start')
        _clean_tpose_keyframes(action, bone_names)

    arm.wiggle_freeze = True
    return True, "Baked successfully"


# ---------------------------------------------------------------------------
#  Operators
# ---------------------------------------------------------------------------

class HAIR_OT_AddBone(Operator):
    bl_idname = "hair_physics.add_bone"
    bl_label  = "Add Bone"
    bl_options = {'REGISTER'}

    def execute(self, context):
        context.scene.hair_physics.bones.add()
        return {'FINISHED'}

class HAIR_OT_RemoveBone(Operator):
    bl_idname  = "hair_physics.remove_bone"
    bl_label   = "Remove Bone"
    bl_options = {'REGISTER'}

    def execute(self, context):
        props = context.scene.hair_physics
        if props.bones:
            idx = min(props.active_bone_index, len(props.bones) - 1)
            props.bones.remove(idx)
            props.active_bone_index = max(0, idx - 1)
        return {'FINISHED'}

class HAIR_OT_Preview(Operator):
    bl_idname  = "hair_physics.preview"
    bl_label   = "Preview Hair Jiggle"
    bl_options = {'REGISTER'}

    def execute(self, context):
        props = context.scene.hair_physics
        arm   = _find_armature(context)
        if not arm:
            self.report({'ERROR'}, "No armature found"); return {'CANCELLED'}

        bone_names = _get_bone_names(props)
        if not bone_names:
            self.report({'ERROR'}, "Add at least one hair bone"); return {'CANCELLED'}

        if not arm.animation_data or not arm.animation_data.action:
            self.report({'ERROR'}, "Load an animation first"); return {'CANCELLED'}

        action = arm.animation_data.action

        # Snapshot for undo
        if props.backup_action_name:
            old = bpy.data.actions.get(props.backup_action_name)
            if old:
                bpy.data.actions.remove(old)
        bkp = action.copy()
        bkp.name = f"__hair_bkp_{action.name}"
        bkp.use_fake_user = True
        props.backup_action_name = bkp.name

        _strip_fcurves(action, bone_names)

        arm.wiggle_freeze = False
        configured = _apply_wiggle(context, arm, bone_names, props.jiggle_intensity)
        if not configured:
            self.report({'ERROR'}, "None of the listed bones found in armature")
            return {'CANCELLED'}

        props.status_text = "Baking hair physics..."
        ok, msg = _bake_hair(context, arm, bone_names, props.jiggle_intensity)
        if not ok:
            props.status_text = f"Failed: {msg}"
            self.report({'ERROR'}, msg); return {'CANCELLED'}

        _clear_wiggle(context, arm, bone_names)
        props.status_text = "Hair preview ready!"
        self.report({'INFO'}, f"Hair baked: {', '.join(configured)}")
        _ensure_object_mode(context)
        return {'FINISHED'}

class HAIR_OT_Undo(Operator):
    bl_idname  = "hair_physics.undo"
    bl_label   = "Undo Hair Preview"
    bl_options = {'REGISTER'}

    def execute(self, context):
        props = context.scene.hair_physics
        arm   = _find_armature(context)
        if not arm:
            self.report({'ERROR'}, "No armature found"); return {'CANCELLED'}

        arm.wiggle_freeze = False
        _clear_wiggle(context, arm, _get_bone_names(props))

        if props.backup_action_name:
            bkp = bpy.data.actions.get(props.backup_action_name)
            if bkp and arm.animation_data:
                baked = arm.animation_data.action
                arm.animation_data.action = bkp
                bkp.name = bkp.name.replace("__hair_bkp_", "", 1)
                bkp.use_fake_user = False
                if baked and baked != bkp:
                    bpy.data.actions.remove(baked)
                props.backup_action_name = ""
                props.status_text = "Ready"
                self.report({'INFO'}, "Restored pre-preview state")
                _ensure_object_mode(context)
                return {'FINISHED'}

        props.status_text = "Ready"
        self.report({'WARNING'}, "No backup found")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
#  Animation folder browsing operators
# ---------------------------------------------------------------------------

class HAIR_OT_BrowseFolder(Operator):
    """Browse for animation folder"""
    bl_idname = "hair_physics.browse_folder"
    bl_label = "Browse Animations Folder"
    bl_description = "Choose a folder containing .anm animation files"
    bl_options = {'REGISTER'}

    directory: StringProperty(subtype='DIR_PATH')

    def execute(self, context):
        props = context.scene.hair_physics
        if self.directory:
            props.custom_folder = self.directory.rstrip('/\\')
            bpy.ops.hair_physics.refresh_anims()
        return {'FINISHED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


class HAIR_OT_BrowseExportFolder(Operator):
    """Browse for export folder"""
    bl_idname = "hair_physics.browse_export_folder"
    bl_label = "Browse Export Folder"
    bl_description = "Choose a folder to export baked animations"
    bl_options = {'REGISTER'}

    directory: StringProperty(subtype='DIR_PATH')

    def execute(self, context):
        props = context.scene.hair_physics
        if self.directory:
            props.export_folder = self.directory.rstrip('/\\')
        return {'FINISHED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


class HAIR_OT_ClearFolder(Operator):
    """Clear custom folder"""
    bl_idname = "hair_physics.clear_folder"
    bl_label = "Clear Custom Folder"
    bl_options = {'REGISTER'}

    def execute(self, context):
        props = context.scene.hair_physics
        props.custom_folder = ""
        bpy.ops.hair_physics.refresh_anims()
        return {'FINISHED'}


class HAIR_OT_RefreshAnims(Operator):
    """Scan the animations folder and refresh the list"""
    bl_idname = "hair_physics.refresh_anims"
    bl_label = "Refresh Animations"
    bl_options = {'REGISTER'}

    def execute(self, context):
        props = context.scene.hair_physics
        from .boobs_physics import get_animations_folder
        props.animations.clear()
        props.animations_folder = ""
        props.search_filter = ""

        if props.custom_folder and os.path.isdir(props.custom_folder):
            anim_folder = props.custom_folder
        else:
            armature_obj = _find_armature(context)
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


class HAIR_OT_LoadAnimation(Operator):
    """Load a single animation for preview"""
    bl_idname = "hair_physics.load_anim"
    bl_label = "Load Animation"
    bl_options = {'REGISTER', 'UNDO'}

    filepath: StringProperty()
    anim_name: StringProperty()
    index: IntProperty(default=-1)

    def execute(self, context):
        props = context.scene.hair_physics

        if not self.filepath or not os.path.exists(self.filepath):
            self.report({'ERROR'}, "Animation file not found")
            return {'CANCELLED'}

        armature_obj = _find_armature(context)
        if not armature_obj:
            self.report({'ERROR'}, "No armature found")
            return {'CANCELLED'}

        _select_armature(context, armature_obj)

        armature_obj.wiggle_freeze = False

        bpy.ops.object.mode_set(mode='POSE')
        bone_names = _get_bone_names(props)
        for bname in bone_names:
            pb = armature_obj.pose.bones.get(bname)
            if pb:
                pb.location = (0.0, 0.0, 0.0)
                pb.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
                pb.rotation_euler = (0.0, 0.0, 0.0)
                pb.scale = (1.0, 1.0, 1.0)
        bpy.ops.object.mode_set(mode='OBJECT')

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


class HAIR_OT_ApplyToAll(Operator):
    """Apply hair physics to all animations in the folder, bake, and export"""
    bl_idname = "hair_physics.apply_all"
    bl_label = "Apply to All Animations"
    bl_description = "Import each animation, apply hair physics, bake, and export to the output folder"
    bl_options = {'REGISTER'}

    def execute(self, context):
        props = context.scene.hair_physics
        armature_obj = _find_armature(context)

        if not armature_obj:
            self.report({'ERROR'}, "No armature found in scene")
            return {'CANCELLED'}

        bone_names = _get_bone_names(props)
        if not bone_names:
            self.report({'ERROR'}, "No hair bones selected")
            return {'CANCELLED'}

        if len(props.animations) == 0:
            self.report({'ERROR'}, "No animations loaded. Click Refresh first.")
            return {'CANCELLED'}

        export_dir = props.export_folder
        if not export_dir:
            export_dir = props.animations_folder
        if not export_dir or not os.path.isdir(export_dir):
            self.report({'ERROR'}, "No valid export folder. Set one or ensure animations folder exists.")
            return {'CANCELLED'}

        _select_armature(context, armature_obj)

        if not armature_obj.animation_data:
            armature_obj.animation_data_create()

        original_action = armature_obj.animation_data.action

        total = len(props.animations)
        success_count = 0
        fail_count = 0
        fps = context.scene.render.fps

        _apply_wiggle(context, armature_obj, bone_names, props.jiggle_intensity)

        for idx, anim_item in enumerate(props.animations):
            props.status_text = f"Processing {idx + 1}/{total}: {anim_item.name}..."
            bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)

            try:
                armature_obj.wiggle_freeze = False
                try:
                    bpy.ops.wiggle.reset()
                except Exception:
                    pass

                _ensure_object_mode(context)
                _select_armature(context, armature_obj)

                anm = import_anm.read_anm(anim_item.filepath)
                action_name = anim_item.name
                new_action = bpy.data.actions.new(name=action_name)
                armature_obj.animation_data.action = new_action
                import_anm.apply_anm(anm, armature_obj, frame_offset=0)
                new_action["lol_anm_filepath"] = anim_item.filepath

                _apply_wiggle(context, armature_obj, bone_names, props.jiggle_intensity)

                ok, msg = _bake_hair(context, armature_obj, bone_names, props.jiggle_intensity)
                if not ok:
                    print(f"Bake failed for {anim_item.name}: {msg}")
                    fail_count += 1
                    continue

                out_path = os.path.join(export_dir, f"{anim_item.name}.anm")
                export_anm.write_anm(out_path, armature_obj, fps)

                success_count += 1

            except Exception as e:
                print(f"Failed on {anim_item.name}: {e}")
                fail_count += 1

        _clear_wiggle(context, armature_obj, bone_names)

        try:
            armature_obj.animation_data.action = original_action
        except Exception:
            pass

        armature_obj.wiggle_freeze = False
        _ensure_object_mode(context)
        props.status_text = "Ready"

        if fail_count > 0:
            self.report({'WARNING'}, f"Done: {success_count}/{total} exported ({fail_count} failed) to {export_dir}")
        else:
            self.report({'INFO'}, f"Done: {success_count} animations exported to {export_dir}")

        return {'FINISHED'}


# ---------------------------------------------------------------------------
#  UI List for animations
# ---------------------------------------------------------------------------

class HAIR_UL_AnimList(UIList):
    """Scrollable animation list with filtering"""

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        props = context.scene.hair_physics

        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            row = layout.row(align=True)
            if item.name == props.current_loaded:
                row.label(text="", icon='PLAY')
            else:
                row.label(text="", icon='ACTION')

            op = row.operator("hair_physics.load_anim", text=item.name, emboss=False)
            op.filepath = item.filepath
            op.anim_name = item.name
            op.index = index

        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text=item.name, icon='ACTION')

    def filter_items(self, context, data, propname):
        props = context.scene.hair_physics
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

class HAIR_PT_Physics(Panel):
    bl_label      = "Hair Physics"
    bl_idname     = "VIEW3D_PT_hair_physics"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category   = 'Auto Physics'
    bl_parent_id  = "VIEW3D_PT_auto_physics"

    def draw(self, context):
        layout = self.layout
        props  = context.scene.hair_physics
        arm    = _find_armature(context)

        box = layout.box()
        box.label(text=props.status_text, icon='INFO')

        # --- Bone list ---
        box = layout.box()
        box.label(text="Hair Bones", icon='BONE_DATA')

        if arm:
            for i, item in enumerate(props.bones):
                row = box.row(align=True)
                row.prop_search(item, "bone_name", arm.pose, "bones", text=f"Bone {i+1}")

            row = box.row(align=True)
            row.operator("hair_physics.add_bone",    text="Add Bone",    icon='ADD')
            row.operator("hair_physics.remove_bone", text="Remove",      icon='REMOVE')
        else:
            box.label(text="No armature in scene", icon='ERROR')

        # --- Intensity ---
        box = layout.box()
        box.label(text="Hair Settings", icon='MOD_WAVE')
        col = box.column(align=True)
        col.prop(props, "jiggle_intensity", slider=True)
        row = col.row()
        row.alignment = 'CENTER'
        i = props.jiggle_intensity
        label = "▸ Stiff" if i <= 3 else "▸ Subtle" if i <= 7 else "▸ Natural" if i <= 13 else "▸ Flowing" if i <= 17 else "▸ Wild!"
        row.label(text=label)

        # --- Preview controls ---
        box = layout.box()
        box.label(text="Preview", icon='PLAY')
        has_anim  = bool(arm and arm.animation_data and arm.animation_data.action)
        has_bones = bool(_get_bone_names(props))
        col = box.column(align=True)
        col.scale_y = 1.3
        row = col.row(align=True)
        row.enabled = has_anim and has_bones
        row.operator("hair_physics.preview", text="Preview Hair Jiggle", icon='MOD_WAVE')
        row = col.row(align=True)
        row.operator("hair_physics.undo",    text="Undo Preview",        icon='LOOP_BACK')

        if props.current_loaded:
            box.label(text=f"Animation: {props.current_loaded}", icon='ANIM')

        # --- Animation folder & batch processing ---
        box = layout.box()
        box.label(text="Batch Processing", icon='FILE_FOLDER')

        # Folder controls
        row = box.row(align=True)
        row.operator("hair_physics.refresh_anims", text="Refresh", icon='FILE_REFRESH')
        row.operator("hair_physics.browse_folder", text="", icon='FILEBROWSER')

        # Show current folder
        row = box.row(align=True)
        row.scale_y = 0.7
        if props.animations_folder:
            folder_name = os.path.basename(props.animations_folder)
            parent_name = os.path.basename(os.path.dirname(props.animations_folder))
            if props.custom_folder:
                row.label(text=f".../{parent_name}/{folder_name}", icon='FILE_FOLDER')
                row.operator("hair_physics.clear_folder", text="", icon='X')
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
                "HAIR_UL_AnimList", "",
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
        row.operator("hair_physics.browse_export_folder", text="", icon='FILEBROWSER')

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
        col.operator("hair_physics.apply_all", text="⚡ Apply to All Animations", icon='MOD_WAVE')


# ---------------------------------------------------------------------------
#  Registration
# ---------------------------------------------------------------------------

classes = [
    HairBoneItem,
    HairAnimListItem,
    HairPhysicsProperties,
    HAIR_OT_AddBone,
    HAIR_OT_RemoveBone,
    HAIR_OT_Preview,
    HAIR_OT_Undo,
    HAIR_OT_BrowseFolder,
    HAIR_OT_BrowseExportFolder,
    HAIR_OT_ClearFolder,
    HAIR_OT_RefreshAnims,
    HAIR_OT_LoadAnimation,
    HAIR_OT_ApplyToAll,
    HAIR_UL_AnimList,
    HAIR_PT_Physics,
]

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.hair_physics = PointerProperty(type=HairPhysicsProperties)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    if hasattr(bpy.types.Scene, 'hair_physics'):
        del bpy.types.Scene.hair_physics
