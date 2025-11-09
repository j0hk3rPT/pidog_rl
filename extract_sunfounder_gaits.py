#!/usr/bin/env python3
"""
Extract Sunfounder PiDog gaits with correct action space conversion.

Key insights:
1. Sunfounder IK outputs angles in degrees
2. Our environment expects actions in [-1, 1] scaled to [-π/2, π/2]
3. Conversion: action = angle_degrees * (π/180) / (π/2) = angle_degrees / 90

Sunfounder leg order: [LF_hip, LF_knee, RF_hip, RF_knee, LH_hip, LH_knee, RH_hip, RH_knee]
MuJoCo order: [BR_hip, BR_knee, FR_hip, FR_knee, BL_hip, BL_knee, FL_hip, FL_knee]
  where BR=back_right, FR=front_right, BL=back_left, FL=front_left

Mapping:
  Sunfounder[0,1] = LF (left front) -> MuJoCo[6,7] = FL (front_left)
  Sunfounder[2,3] = RF (right front) -> MuJoCo[2,3] = FR (front_right)
  Sunfounder[4,5] = LH (left hind/back) -> MuJoCo[4,5] = BL (back_left)
  Sunfounder[6,7] = RH (right hind/back) -> MuJoCo[0,1] = BR (back_right)
"""

import argparse
import pickle
import numpy as np
from pathlib import Path
from math import sqrt, acos, atan2, pi, sin, cos
import sys

sys.path.insert(0, str(Path(__file__).parent))
from pidog_env import PiDogEnv

try:
    from imitation.data.types import Transitions
    IMITATION_AVAILABLE = True
except ImportError:
    IMITATION_AVAILABLE = False


# Sunfounder constants
LEG = 42  # Upper leg segment length (mm)
FOOT = 76  # Lower leg segment length (mm)


def coord2polar(coord):
    """
    Sunfounder inverse kinematics function.

    Args:
        coord: [y, z] coordinates in mm

    Returns:
        (alpha, beta) angles in degrees
        - alpha: hip/shoulder angle
        - beta: knee angle
    """
    y, z = coord
    u = sqrt(pow(y, 2) + pow(z, 2))

    # Knee angle (beta)
    cos_angle1 = (FOOT**2 + LEG**2 - u**2) / (2 * FOOT * LEG)
    cos_angle1 = min(max(cos_angle1, -1), 1)  # Clamp
    beta = acos(cos_angle1)

    # Hip angle (alpha)
    angle1 = atan2(y, z)
    cos_angle2 = (LEG**2 + u**2 - FOOT**2) / (2 * LEG * u)
    cos_angle2 = min(max(cos_angle2, -1), 1)  # Clamp
    angle2 = acos(cos_angle2)
    alpha = angle2 + angle1

    # Convert to degrees
    alpha = alpha / pi * 180
    beta = beta / pi * 180

    return alpha, beta


def legs_angle_calculation(coords):
    """
    Calculate leg angles from coordinates using Sunfounder IK.

    Args:
        coords: List of 4 [y, z] coordinates, one per leg
               Order: LF, RF, LH, RH (left front, right front, left hind, right hind)

    Returns:
        List of 8 angles in Sunfounder order [LF_hip, LF_knee, RF_hip, RF_knee, LH_hip, LH_knee, RH_hip, RH_knee]
    """
    angles = []
    for i, coord in enumerate(coords):
        leg_angle, foot_angle = coord2polar(coord)

        # Sunfounder adjustment
        foot_angle = foot_angle - 90

        # Mirror for right legs (odd indices)
        if i % 2 != 0:
            leg_angle = -leg_angle
            foot_angle = -foot_angle

        angles += [leg_angle, foot_angle]

    return angles


def sunfounder_to_mujoco_order(sunfounder_angles):
    """
    Convert from Sunfounder leg order to MuJoCo order.

    Sunfounder: [LF_hip, LF_knee, RF_hip, RF_knee, LH_hip, LH_knee, RH_hip, RH_knee]
    MuJoCo:     [BR_hip, BR_knee, FR_hip, FR_knee, BL_hip, BL_knee, FL_hip, FL_knee]
    """
    LF_hip, LF_knee = sunfounder_angles[0], sunfounder_angles[1]
    RF_hip, RF_knee = sunfounder_angles[2], sunfounder_angles[3]
    LH_hip, LH_knee = sunfounder_angles[4], sunfounder_angles[5]
    RH_hip, RH_knee = sunfounder_angles[6], sunfounder_angles[7]

    # Map to MuJoCo order
    mujoco_angles = [
        RH_hip, RH_knee,  # BR (back_right) <- RH (right hind)
        RF_hip, RF_knee,  # FR (front_right) <- RF (right front)
        LH_hip, LH_knee,  # BL (back_left) <- LH (left hind)
        LF_hip, LF_knee,  # FL (front_left) <- LF (left front)
    ]

    return mujoco_angles


def degrees_to_action(angle_degrees):
    """
    Convert angle in degrees to action space [-1, 1].

    Our environment scales actions: joint_angle = action * (π/2)
    So: action = joint_angle / (π/2) = angle_degrees * (π/180) / (π/2) = angle_degrees / 90
    """
    return angle_degrees / 90.0


def angles_to_actions(angles_degrees):
    """Convert list of angles in degrees to action space."""
    return [degrees_to_action(angle) for angle in angles_degrees]


def get_stand_coords():
    """Get standing position coordinates in Sunfounder format."""
    barycenter = -15
    height = 95

    # Stand coords: [LF, RF, LH, RH]
    coords = [
        [barycenter, height],          # LF
        [barycenter, height],          # RF
        [barycenter + 20, height - 5], # LH (back legs slightly forward and lower)
        [barycenter + 20, height - 5], # RH
    ]

    return coords


def get_walk_coords(num_cycles=1):
    """Generate walking gait coordinates."""
    # Walk parameters from Sunfounder
    LEG_STEP_HEIGHT = 20
    LEG_STEP_WIDTH = 80
    CENTER_OF_GRAVITY = -15
    LEG_POSITION_OFFSETS = [-10, -10, 20, 20]
    Z_ORIGIN = 80

    SECTION_COUNT = 8
    STEP_COUNT = 6
    LEG_ORDER = [1, 0, 4, 0, 2, 0, 3, 0]  # Order: leg 1, wait, leg 4, wait, leg 2, wait, leg 3, wait

    # Calculate parameters
    y_offset = CENTER_OF_GRAVITY
    leg_step_width = [LEG_STEP_WIDTH for _ in range(4)]
    section_length = [leg_step_width[i] / (SECTION_COUNT - 1) for i in range(4)]
    step_down_length = [section_length[i] / STEP_COUNT for i in range(4)]
    leg_origin = [leg_step_width[i] / 2 + y_offset + LEG_POSITION_OFFSETS[i] for i in range(4)]

    LEG_ORIGINAL_Y_TABLE = [0, 2, 3, 1]
    origin_leg_coord = [
        [leg_origin[i] - LEG_ORIGINAL_Y_TABLE[i] * 2 * section_length[i], Z_ORIGIN]
        for i in range(4)
    ]

    all_coords = []

    for _ in range(num_cycles):
        leg_coord = [list(coord) for coord in origin_leg_coord]

        for section in range(SECTION_COUNT):
            for step in range(STEP_COUNT):
                raise_leg = LEG_ORDER[section]

                for i in range(4):
                    if raise_leg != 0 and i == raise_leg - 1:
                        # Stepping leg - cosine motion
                        theta = step * pi / (STEP_COUNT - 1)
                        temp = leg_step_width[i] * (cos(theta) - 1) / 2
                        y = leg_origin[i] + temp
                        z = Z_ORIGIN - (LEG_STEP_HEIGHT * step / (STEP_COUNT - 1))
                    else:
                        # Support legs - linear motion
                        y = leg_coord[i][0] + step_down_length[i]
                        z = Z_ORIGIN

                    leg_coord[i] = [y, z]

                all_coords.append([list(coord) for coord in leg_coord])

        # Add origin position at end
        all_coords.append([list(coord) for coord in origin_leg_coord])

    return all_coords


def extract_gait(coords_list, gait_name="gait"):
    """Extract gait from coordinate list."""
    print(f"\n{'='*70}")
    print(f" EXTRACTING {gait_name.upper()}")
    print(f"{'='*70}")
    print(f"Frames: {len(coords_list)}")
    print(f"{'='*70}\n")

    all_actions = []

    for i, coords in enumerate(coords_list):
        # Calculate angles using Sunfounder IK
        angles_sunfounder = legs_angle_calculation(coords)

        # Convert to MuJoCo order
        angles_mujoco = sunfounder_to_mujoco_order(angles_sunfounder)

        # Convert to action space
        actions = angles_to_actions(angles_mujoco)

        # Clip to valid range
        actions = [np.clip(a, -1, 1) for a in actions]

        all_actions.append(np.array(actions, dtype=np.float32))

        if i == 0:
            print(f"First frame coordinates (LF, RF, LH, RH):")
            for leg_idx, coord in enumerate(coords):
                leg_names = ["LF", "RF", "LH", "RH"]
                print(f"  {leg_names[leg_idx]}: y={coord[0]:6.1f}mm, z={coord[1]:6.1f}mm")
            print(f"\nFirst frame angles (Sunfounder order):")
            for leg_idx in range(4):
                leg_names = ["LF", "RF", "LH", "RH"]
                hip = angles_sunfounder[leg_idx * 2]
                knee = angles_sunfounder[leg_idx * 2 + 1]
                print(f"  {leg_names[leg_idx]}: hip={hip:6.1f}°, knee={knee:6.1f}°")
            print(f"\nFirst frame actions (MuJoCo order):")
            mujoco_names = ["BR", "FR", "BL", "FL"]
            for leg_idx in range(4):
                hip = actions[leg_idx * 2]
                knee = actions[leg_idx * 2 + 1]
                print(f"  {mujoco_names[leg_idx]}: hip={hip:6.3f}, knee={knee:6.3f}")
            print()

    return all_actions


def collect_demonstrations(actions_list, gait_name="gait", use_camera=False):
    """Collect demonstrations by executing actions in environment."""
    env = PiDogEnv(use_camera=use_camera)

    obs_list = []
    acts_list = []
    next_obs_list = []
    dones_list = []

    obs, _ = env.reset()

    for action in actions_list:
        if use_camera and isinstance(obs, dict):
            obs_vec = obs["vector"]
        else:
            obs_vec = obs

        next_obs, reward, terminated, truncated, info = env.step(action)

        if use_camera and isinstance(next_obs, dict):
            next_obs_vec = next_obs["vector"]
        else:
            next_obs_vec = next_obs

        obs_list.append(obs_vec)
        acts_list.append(action)
        next_obs_list.append(next_obs_vec)
        dones_list.append(terminated or truncated)

        obs = next_obs

        if terminated or truncated:
            obs, _ = env.reset()
            break

    env.close()

    if not IMITATION_AVAILABLE:
        return {
            "obs": np.array(obs_list),
            "acts": np.array(acts_list),
            "next_obs": np.array(next_obs_list),
            "dones": np.array(dones_list),
        }

    transitions = Transitions(
        obs=np.array(obs_list),
        acts=np.array(acts_list),
        infos=np.array([{}] * len(obs_list)),
        next_obs=np.array(next_obs_list),
        dones=np.array(dones_list),
    )

    print(f"\n✓ Collected {len(transitions)} transitions")
    print(f"{'='*70}\n")
    return transitions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gait", type=str, default="stand",
                       choices=["stand", "walk"],
                       help="Which gait to extract")
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument("--n-cycles", type=int, default=5,
                       help="Number of cycles for walk gait")
    parser.add_argument("--use-camera", action="store_true")
    args = parser.parse_args()

    # Get coordinates for requested gait
    if args.gait == "stand":
        coords_list = [get_stand_coords()]
        output_file = args.output_file or "demonstrations/sunfounder_stand.pkl"
    elif args.gait == "walk":
        coords_list = get_walk_coords(num_cycles=args.n_cycles)
        output_file = args.output_file or "demonstrations/sunfounder_walk.pkl"

    # Extract actions
    actions = extract_gait(coords_list, gait_name=args.gait)

    # Collect demonstrations
    demos = collect_demonstrations(actions, gait_name=args.gait, use_camera=args.use_camera)

    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(demos, f)

    print(f"✓ Saved to: {output_path}")
    print(f"\nVisualize with:")
    print(f"  python visualize_demonstrations.py --demo-file {output_path}\n")


if __name__ == "__main__":
    main()
