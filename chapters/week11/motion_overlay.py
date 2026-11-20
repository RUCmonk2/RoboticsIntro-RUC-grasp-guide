"""在 MuJoCo 场景中常驻显示平移键和起始姿态的关节示例。"""

import mujoco
import numpy as np


ORIGIN = np.zeros(3)
IDENTITY = np.eye(3).ravel()
INACTIVE_ARROW_SCALE = np.array([.62, .62, .62, 1.], dtype=np.float32)
GUIDE_LINK_COLOR = np.array([.43, .48, .52, 1.], dtype=np.float32)
# 每组标出默认姿态下对该轴末端平移贡献最大的关节。
# 标签偏移量沿镜头的右、上方向，单位为米。
TRANSLATION_GUIDES = (
    ("WS", 2, (-.17, -.025), np.array([.12, .88, 1., 1.], dtype=np.float32)),
    ("AD", 3, (-.17, -.01), np.array([1., .72, .16, 1.], dtype=np.float32)),
    ("QE", 4, (-.11, .085), np.array([.78, .56, 1., 1.], dtype=np.float32)),
)


def add_geom(scene, geom_type, color, position=ORIGIN):
    geom = scene.geoms[scene.ngeom]
    scene.ngeom += 1
    mujoco.mjv_initGeom(geom, geom_type, ORIGIN, position, IDENTITY, color)
    geom.category = mujoco.mjtCatBit.mjCAT_DECOR
    geom.emission = .7
    return geom


def add_segment(scene, start, end, width, color):
    geom_type = mujoco.mjtGeom.mjGEOM_CAPSULE
    geom = add_geom(scene, geom_type, color)
    mujoco.mjv_connector(geom, geom_type, width, start, end)


def add_label(scene, position, text, color):
    geom = add_geom(scene, mujoco.mjtGeom.mjGEOM_LABEL, color, position)
    geom.label = text


def add_arrowhead(scene, tip, direction, size, width, color, camera):
    # 箭头两翼尽量面向相机。
    side = np.cross(camera.pos - tip, direction)
    if np.linalg.norm(side) < 1e-6:
        side = np.cross(camera.up, direction)
    side *= .5 * size / np.linalg.norm(side)
    start = tip - size * direction
    add_segment(scene, start - side, tip, width, color)
    add_segment(scene, start + side, tip, width, color)


def axis_label(axis):
    index = np.flatnonzero(axis)[0]
    sign = "+" if axis[index] > 0 else "-"
    return f"{sign}{'XYZ'[index]}"


def draw_translation_guide(scene, env, key_directions, next_key):
    camera = scene.camera[0]
    right = np.cross(camera.forward, camera.up)
    toward_camera = -camera.forward
    tip_marker = env.tcp_xyz + .09 * toward_camera
    # 方向指示放到夹爪右侧，给方块和两指之间的接触区域留空。
    start = tip_marker + .14 * right
    length = .135
    start[2] = max(start[2], length + .025)
    add_segment(scene, tip_marker, start, .0014, GUIDE_LINK_COLOR)

    for keys, joint_number, (offset_right, offset_up), color in TRANSLATION_GUIDES:
        joint_id = env.joint_ids[joint_number - 1]
        anchor = env.data.xanchor[joint_id]
        view_ray = camera.pos - anchor
        # 沿观线把标记移到外壳前方，保持它在画面中对准关节中心。
        marker = anchor + .19 * view_ray / np.linalg.norm(view_ray)
        dot = add_geom(scene, mujoco.mjtGeom.mjGEOM_SPHERE, color, marker)
        dot.size[:] = .009
        callout = marker + offset_right * right + offset_up * camera.up
        add_segment(scene, marker, callout, .002, color)
        add_label(scene, callout + .018 * camera.up, f"J{joint_number}  {'/'.join(keys)}", color)

        for key in keys:
            axis = np.asarray(key_directions[key], dtype=float)
            end = start + length * axis
            active = key == next_key
            width = .0055 if active else .0027
            head_size = .037 if active else .026
            key_label = f"[{key}]" if active else key
            arrow_color = color if active else color * INACTIVE_ARROW_SCALE
            add_segment(scene, start, end, width, arrow_color)
            add_arrowhead(scene, end, axis, head_size, width, arrow_color, camera)
            add_label(scene, end + .013 * axis + .013 * camera.up,
                      f"{key_label} {axis_label(axis)}", arrow_color)


def draw_grasp_cue(viewport, context, action, key):
    font_height = context.charHeightBig
    width = min(viewport.width, 10 * font_height)
    height = min(viewport.height, 2 * font_height)
    margin = min(font_height // 2, viewport.height - height)
    rect = mujoco.MjrRect((viewport.width - width) // 2,
                          viewport.height - height - margin, width, height)
    if action == "RELEASE":
        background = (1., .58, .30, 1.)
        text_color = (.10, .08, .04)
    elif key:
        background = (1., .86, .24, 1.)
        text_color = (.10, .08, .04)
    else:
        background = (.13, .24, .32, 1.)
        text_color = (1., 1., 1.)
    label = f"{action}  {key}" if key else action
    mujoco.mjr_label(rect, mujoco.mjtFont.mjFONT_BIG, label, *background, *text_color, context)
