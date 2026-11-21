"""The flybody fly as used by the walking project: the option-5 torque model, the joint sign probe and
the constants the controllers share.

Collected (code unchanged) from the earlier project modules it was developed in:
  fly_motor.py        JOINTS, MN_JOINT, LEG_SEGMENT, SIDE
  fly_mujoco_proc.py  GAIN, build_tethered (was build_model), probe_signs
  walk_mujoco_proc.py ADH_REF, TAU_ADH, build_model, heading_deg
  walk_gait.py        LEGS, TRIPOD_A
  walk_torque.py      FLOOR_Z, fly_spec, build
"""
import os

import mujoco
import numpy as np

from paths import flybody_file

# flybody model files (walk_torque.FLY_XML, fly_mujoco_proc.FLY_XML); looked up when first used, so
# modules that only need the constants below (e.g. walk_circuit_build.py) work without flybody
FLY_XML_NAME, FLOOR_XML_NAME = "fruitfly.xml", "floor.xml"

# ---- connectome motor neurons -> joints (fly_motor.py)
JOINTS = ["coxa", "femur", "tibia", "tarsus"]
# motor neuron type -> (joint, anatomical direction)
MN_JOINT = {
    "Tergopleural/Pleural promotor MN": ("coxa", 1),
    "Sternal anterior rotator MN":      ("coxa", 1),
    "Pleural remotor/abductor MN":      ("coxa", -1),
    "Sternal posterior rotator MN":     ("coxa", -1),
    "Tr extensor MN":                   ("femur", 1),
    "Sternotrochanter MN":              ("femur", 1),
    "Tergotr. MN":                      ("femur", 1),
    "Tr flexor MN":                     ("femur", -1),
    "Acc. tr flexor MN":                ("femur", -1),
    "Ti extensor MN":                   ("tibia", 1),
    "Ti flexor MN":                     ("tibia", -1),
    "Acc. ti flexor MN":                ("tibia", -1),
    "Ta levator MN":                    ("tarsus", 1),
    "Ta depressor MN":                  ("tarsus", -1),
    "ltm MN":                           ("tarsus", -1),
    "ltm1-tibia MN":                    ("tarsus", -1),
    "ltm2-femur MN":                    ("tarsus", -1),
}
LEG_SEGMENT = {"fl": "T1", "ml": "T2", "hl": "T3"}
SIDE = {"L": "left", "R": "right"}

# ---- legs (walk_gait.py)
LEGS = ["T1_left", "T2_left", "T3_left", "T1_right", "T2_right", "T3_right"]
TRIPOD_A = {"T1_left", "T2_right", "T3_left"}

# ---- actuators and adhesion (fly_mujoco_proc.py, walk_mujoco_proc.py)
GAIN = 0.8            # fraction of each joint's range used at full activation
ADH_REF = 0.3         # = abs(walk_gait.STANCE["tarsus"]): claw-group activation giving full grip
TAU_ADH = 2.0         # ms, claw grip time constant (grip must switch fast)
TETHER_HEIGHT = 0.3   # cm above the floor

# ---- scene (walk_torque.py)
FLOOR_Z = -0.132
FLIES = [("F_", 0.0, "fly", "torque", [0.95, 0.30, 0.20, 1])]


def build_model():
    """The single free fly of floor.xml (walk_mujoco_proc.build_model)."""
    model = mujoco.MjModel.from_xml_path(str(flybody_file(FLOOR_XML_NAME)))
    return model, mujoco.MjData(model)


def heading_deg(data, body):
    x = data.xmat[body].reshape(3, 3)[:, 0]
    return float(np.degrees(np.arctan2(x[1], x[0])))


def probe_signs(model, data, legs):
    """Sign mapping anatomical direction -> actuator control, measured from the
    kinematics: promotion moves the claw forward (+x), femur/tibia extension move it
    away from the coxa/femur, tarsal levation raises it. Measured from the model's
    default pose (qpos0), so it also works for a free-moving fly."""
    signs = {}
    claw_pos = lambda leg: data.xpos[model.body(f"claw_{leg}").id].copy()  # noqa: E731
    for leg in legs:
        for joint in JOINTS:
            def measure():
                mujoco.mj_forward(model, data)
                c = claw_pos(leg)
                return {"coxa": c[0],
                        "femur": np.linalg.norm(c - data.xpos[model.body(f"coxa_{leg}").id]),
                        "tibia": np.linalg.norm(c - data.xpos[model.body(f"femur_{leg}").id]),
                        "tarsus": c[2]}[joint]
            adr = model.joint(f"{joint}_{leg}").qposadr[0]
            data.qpos[:] = model.qpos0
            before = measure()
            data.qpos[adr] += 0.2
            signs[(leg, joint)] = 1 if measure() > before else -1
    data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)
    return signs


def fly_spec():
    fly_xml = flybody_file(FLY_XML_NAME)
    s = mujoco.MjSpec.from_file(str(fly_xml))
    mdir = os.path.join(os.path.dirname(fly_xml), s.meshdir)
    for mesh in s.meshes:
        mesh.file = os.path.normpath(os.path.join(mdir, mesh.file))
    s.meshdir = ""
    for k in list(s.keys):
        s.delete(k)
    s.njmax, s.nconmax, s.nkey = -1, -1, 0      # sizes are set by the parent scene
    return s


def build(torque_scale, flies=None):
    """Floor + flies (prefix, lane y, label, kind, colour). Every non-'servo' fly gets option-5 torque
    motors on its 24 leg joints: Tmax = kp * GAIN * max|ctrlrange| * torque_scale."""
    flies = FLIES if flies is None else flies
    src = mujoco.MjSpec.from_file(str(flybody_file(FLY_XML_NAME)))
    spec = mujoco.MjSpec()
    spec.modelname = "walk_torque"
    o, so = spec.option, src.option
    o.timestep, o.gravity, o.density, o.viscosity = so.timestep, so.gravity, so.density, so.viscosity
    o.cone, o.noslip_iterations = so.cone, so.noslip_iterations
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1280, 720
    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[.1, .2, .3], rgb2=[.2, .3, .4], width=300, height=300,
                     mark=mujoco.mjtMark.mjMARK_EDGE, markrgb=[.2, .3, .4])
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT, rgb1=[.4, .6, .8], rgb2=[0, 0, 0],
                     width=100, height=100)
    mat = spec.add_material(name="grid", texrepeat=[1, 1], texuniform=True, reflectance=0.2)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[20, 20, .1],
                            material="grid", pos=[0, 0, FLOOR_Z], solref=[0.0002, 1])
    spec.worldbody.add_light(pos=[0, 0, 5], dir=[0, 0, -1], diffuse=[.7, .7, .7], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    for prefix, y, *_ in flies:
        spec.worldbody.add_frame(pos=[0, y, 0]).attach_body(fly_spec().body("thorax"), prefix, "")

    tmax = {}
    for prefix, _, _, kind, *_ in flies:
        if kind == "servo":   # every other kind uses the option-5 torque motors
            continue
        for leg in LEGS:
            for j in JOINTS:
                a = spec.actuator(f"{prefix}{j}_{leg}")
                lo, hi = a.ctrlrange
                t = a.gainprm[0] * GAIN * max(abs(lo), abs(hi)) * torque_scale
                a.biastype = mujoco.mjtBias.mjBIAS_NONE
                for k in range(3):
                    a.biasprm[k] = 0.0
                a.gainprm[0] = 1.0
                a.ctrlrange = [-t, t]
                a.forcerange = [-t, t]
                tmax[(leg, j)] = t
    m = spec.compile()
    return m, mujoco.MjData(m), tmax
