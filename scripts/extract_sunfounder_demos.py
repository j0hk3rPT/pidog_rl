#!/usr/bin/env python3
"""
Extract expert demonstrations from Sunfounder PiDog gaits and convert to MuJoCo action space.

This script:
1. Extracts the trot and walk gaits from Sunfounder's pidog library
2. Converts leg coordinates to angles using inverse kinematics
3. Maps angles from degrees to normalized MuJoCo action space [-1, 1]
   where -1 maps to -π/2 and +1 maps to π
"""

import sys
import pickle
import numpy as np
from pathlib import Path
from math import sqrt, acos, atan2, pi, cos

# Sunfounder robot structure constants (in mm)
LEG_LENGTH = 42  # Upper leg length
FOOT_LENGTH = 76  # Lower leg length


# ============================================================================
# Extracted Walk and Trot classes from Sunfounder (no hardware dependencies)
# ============================================================================

class Walk:
    """Sunfounder Walk gait generator."""
    FORWARD = 1
    BACKWARD = -1
    LEFT = -1
    STRAIGHT = 0
    RIGHT = 1

    SECTION_COUNT = 8
    STEP_COUNT = 6
    LEG_ORDER = [1, 0, 4, 0, 2, 0, 3, 0]
    LEG_STEP_HEIGHT = 20
    LEG_STEP_WIDTH = 80
    CENTER_OF_GRAVIRTY = -15
    LEG_POSITION_OFFSETS = [-10, -10, 20, 20]
    Z_ORIGIN = 80

    TURNING_RATE = 0.3
    LEG_STEP_SCALES_LEFT = [TURNING_RATE, 1, TURNING_RATE, 1]
    LEG_STEP_SCALES_MIDDLE = [1, 1, 1, 1]
    LEG_STEP_SCALES_RIGHT = [1, TURNING_RATE, 1, TURNING_RATE]
    LEG_ORIGINAL_Y_TABLE = [0, 2, 3, 1]
    LEG_STEP_SCALES = [LEG_STEP_SCALES_LEFT, LEG_STEP_SCALES_MIDDLE, LEG_STEP_SCALES_RIGHT]

    def __init__(self, fb, lr):
        self.fb = fb
        self.lr = lr

        if self.fb == self.FORWARD:
            if self.lr == self.STRAIGHT:
                self.y_offset = 0 + self.CENTER_OF_GRAVIRTY
            else:
                self.y_offset = 0 + self.CENTER_OF_GRAVIRTY
        elif self.fb == self.BACKWARD:
            if self.lr == self.STRAIGHT:
                self.y_offset = 0 + self.CENTER_OF_GRAVIRTY
            else:
                self.y_offset = 0 + self.CENTER_OF_GRAVIRTY
        else:
            self.y_offset = self.CENTER_OF_GRAVIRTY

        self.leg_step_width = [
            self.LEG_STEP_WIDTH * self.LEG_STEP_SCALES[self.lr+1][i] for i in range(4)]
        self.section_length = [self.leg_step_width[i] / (self.SECTION_COUNT-1) for i in range(4)]
        self.step_down_length = [self.section_length[i] / self.STEP_COUNT for i in range(4)]
        self.leg_origin = [self.leg_step_width[i] / 2 + self.y_offset + (
            self.LEG_POSITION_OFFSETS[i] * self.LEG_STEP_SCALES[self.lr+1][i]) for i in range(4)]

    def step_y_func(self, leg, step):
        theta = step * pi / (self.STEP_COUNT-1)
        temp = (self.leg_step_width[leg] * (cos(theta) - self.fb) / 2 * self.fb)
        y = self.leg_origin[leg] + temp
        return y

    def step_z_func(self, step):
        return self.Z_ORIGIN - (self.LEG_STEP_HEIGHT * step / (self.STEP_COUNT-1))

    def get_coords(self):
        origin_leg_coord = [[self.leg_origin[i] - self.LEG_ORIGINAL_Y_TABLE[i]
                             * 2 * self.section_length[i], self.Z_ORIGIN] for i in range(4)]
        leg_coord = list.copy(origin_leg_coord)
        leg_coords = []
        for section in range(self.SECTION_COUNT):
            for step in range(self.STEP_COUNT):
                if self.fb == 1:
                    raise_leg = self.LEG_ORDER[section]
                else:
                    raise_leg = self.LEG_ORDER[self.SECTION_COUNT - section - 1]

                for i in range(4):
                    if raise_leg != 0 and i == raise_leg-1:
                        y = self.step_y_func(i, step)
                        z = self.step_z_func(step)
                    else:
                        y = leg_coord[i][0] + self.step_down_length[i] * self.fb
                        z = self.Z_ORIGIN
                    leg_coord[i] = [y, z]
                leg_coords.append(list.copy(leg_coord))
        leg_coords.append(origin_leg_coord)
        return leg_coords


class Trot:
    """Sunfounder Trot gait generator."""
    FORWARD = 1
    BACKWARD = -1
    LEFT = -1
    STRAIGHT = 0
    RIGHT = 1

    SECTION_COUNT = 2
    STEP_COUNT = 3
    LEG_RAISE_ORDER = [[1, 4], [2, 3]]
    LEG_STEP_HEIGHT = 20
    LEG_STEP_WIDTH = 100
    CENTER_OF_GRAVITY = -17
    LEG_STAND_OFFSET = 5
    Z_ORIGIN = 80

    TURNING_RATE = 0.5
    LEG_STAND_OFFSET_DIRS = [-1, -1, 1, 1]
    LEG_STEP_SCALES_LEFT = [TURNING_RATE, 1, TURNING_RATE, 1]
    LEG_STEP_SCALES_MIDDLE = [1, 1, 1, 1]
    LEG_STEP_SCALES_RIGHT = [1, TURNING_RATE, 1, TURNING_RATE]
    LEG_ORIGINAL_Y_TABLE = [0, 1, 1, 0]
    LEG_STEP_SCALES = [LEG_STEP_SCALES_LEFT, LEG_STEP_SCALES_MIDDLE, LEG_STEP_SCALES_RIGHT]

    def __init__(self, fb, lr):
        self.fb = fb
        self.lr = lr

        if self.fb == self.FORWARD:
            if self.lr == self.STRAIGHT:
                self.y_offset = 0 + self.CENTER_OF_GRAVITY
            else:
                self.y_offset = -2 + self.CENTER_OF_GRAVITY
        elif self.fb == self.BACKWARD:
            if self.lr == self.STRAIGHT:
                self.y_offset = 8 + self.CENTER_OF_GRAVITY
            else:
                self.y_offset = 1 + self.CENTER_OF_GRAVITY
        else:
            self.y_offset = self.CENTER_OF_GRAVITY

        self.leg_step_width = [
            self.LEG_STEP_WIDTH * self.LEG_STEP_SCALES[self.lr+1][i] for i in range(4)]
        self.section_length = [self.leg_step_width[i] / (self.SECTION_COUNT-1) for i in range(4)]
        self.step_down_length = [self.section_length[i] / self.STEP_COUNT for i in range(4)]
        self.leg_offset = [self.LEG_STAND_OFFSET * self.LEG_STAND_OFFSET_DIRS[i] for i in range(4)]
        self.leg_origin = [self.leg_step_width[i] / 2 + self.y_offset + (
            self.leg_offset[i] * self.LEG_STEP_SCALES[self.lr+1][i]) for i in range(4)]

    def step_y_func(self, leg, step):
        theta = step * pi / (self.STEP_COUNT-1)
        temp = (self.leg_step_width[leg] * (cos(theta) - self.fb) / 2 * self.fb)
        y = self.leg_origin[leg] + temp
        return y

    def step_z_func(self, step):
        return self.Z_ORIGIN - (self.LEG_STEP_HEIGHT * step / (self.STEP_COUNT-1))

    def get_coords(self):
        origin_leg_coord = [[self.leg_origin[i] - self.LEG_ORIGINAL_Y_TABLE[i]
                             * self.section_length[i], self.Z_ORIGIN] for i in range(4)]
        leg_coords = []
        for section in range(self.SECTION_COUNT):
            for step in range(self.STEP_COUNT):
                if self.fb == 1:
                    raise_legs = self.LEG_RAISE_ORDER[section]
                else:
                    raise_legs = self.LEG_RAISE_ORDER[self.SECTION_COUNT - section - 1]
                leg_coord = []

                for i in range(4):
                    if i + 1 in raise_legs:
                        y = self.step_y_func(i, step)
                        z = self.step_z_func(step)
                    else:
                        y = origin_leg_coord[i][0] + self.step_down_length[i] * self.fb
                        z = self.Z_ORIGIN
                    leg_coord.append([y, z])
                origin_leg_coord = leg_coord
                leg_coords.append(leg_coord)
        return leg_coords


def coord2polar(coord, leg_length=LEG_LENGTH, foot_length=FOOT_LENGTH):
    """
    Convert leg coordinate [y, z] to joint angles using inverse kinematics.

    This replicates Sunfounder's coord2polar method.
    Returns angles in degrees.
    """
    y, z = coord
    u = sqrt(y**2 + z**2)

    # Knee angle (beta)
    cos_angle1 = (foot_length**2 + leg_length**2 - u**2) / (2 * foot_length * leg_length)
    cos_angle1 = max(min(cos_angle1, 1), -1)  # Clamp to [-1, 1]
    beta = acos(cos_angle1)

    # Shoulder angle (alpha)
    angle1 = atan2(y, z)
    cos_angle2 = (leg_length**2 + u**2 - foot_length**2) / (2 * leg_length * u)
    cos_angle2 = max(min(cos_angle2, 1), -1)
    angle2 = acos(cos_angle2)
    alpha = angle2 + angle1

    # Convert to degrees
    alpha_deg = alpha * 180 / pi
    beta_deg = beta * 180 / pi

    return alpha_deg, beta_deg


def legs_angle_calculation(coords):
    """
    Calculate joint angles for all 4 legs from coordinates.

    Args:
        coords: List of 4 [y, z] coordinates for legs in Sunfounder order [LF, RF, LH, RH]

    Returns:
        List of 8 angles in degrees, reordered to MuJoCo order [RH, RF, LH, LF]
    """
    # First calculate angles for all legs in Sunfounder order
    sunfounder_angles = []
    for i, coord in enumerate(coords):
        leg_angle, foot_angle = coord2polar(coord)
        foot_angle = foot_angle - 90  # Offset adjustment

        # Right legs (odd indices) have negated angles
        if i % 2 != 0:
            leg_angle = -leg_angle
            foot_angle = -foot_angle

        sunfounder_angles.extend([leg_angle, foot_angle])

    # Reorder from Sunfounder [LF, RF, LH, RH] to MuJoCo [RH, RF, LH, LF]
    # Sunfounder indices: LF=0,1  RF=2,3  LH=4,5  RH=6,7
    # MuJoCo indices:     RH=0,1  RF=2,3  LH=4,5  LF=6,7
    mujoco_angles = [
        sunfounder_angles[6], sunfounder_angles[7],  # RH (was index 6,7)
        sunfounder_angles[2], sunfounder_angles[3],  # RF (was index 2,3)
        sunfounder_angles[4], sunfounder_angles[5],  # LH (was index 4,5)
        sunfounder_angles[0], sunfounder_angles[1],  # LF (was index 0,1)
    ]

    return mujoco_angles


def degrees_to_normalized(angle_deg):
    """
    Convert angle from degrees to normalized MuJoCo action space [-1, 1].

    MuJoCo servo range: -π/2 to π radians (-90° to 180°)
    Action space: -1 to 1

    Mapping:
        -1 → -π/2 (-90°)
        +1 → π (180°)
    """
    # Convert degrees to radians
    angle_rad = angle_deg * pi / 180

    # MuJoCo range
    low = -pi / 2   # -90 degrees
    high = pi       # 180 degrees

    # Map to [-1, 1]
    normalized = 2 * (angle_rad - low) / (high - low) - 1

    return normalized


def extract_gait_actions(gait_name, fb, lr):
    """
    Extract action sequence for a specific gait.

    Args:
        gait_name: 'trot' or 'walk'
        fb: forward/backward direction
        lr: left/right/straight turning

    Returns:
        numpy array of shape (n_steps, 8) with normalized actions
    """
    # Create gait object
    if gait_name == 'trot':
        gait = Trot(fb=fb, lr=lr)
    elif gait_name == 'walk':
        gait = Walk(fb=fb, lr=lr)
    else:
        raise ValueError(f"Unknown gait: {gait_name}")

    # Get leg coordinates
    coords_sequence = gait.get_coords()

    # Convert to angles and normalize
    actions = []
    for coords in coords_sequence:
        # Convert coords to angles (degrees)
        angles_deg = legs_angle_calculation(coords)

        # Normalize to [-1, 1]
        normalized = [degrees_to_normalized(a) for a in angles_deg]

        # Clip to valid action space
        normalized = [max(-1.0, min(1.0, a)) for a in normalized]
        actions.append(normalized)

    return np.array(actions)


def create_demonstration_dataset(gaits_to_extract, n_repeats=5):
    """
    Create a demonstration dataset by extracting and repeating gait sequences.

    Args:
        gaits_to_extract: List of (gait_name, fb, lr, label) tuples
        n_repeats: Number of times to repeat each gait sequence

    Returns:
        Dictionary with 'actions' and 'labels' arrays
    """
    all_actions = []
    all_labels = []

    for gait_name, fb, lr, label in gaits_to_extract:
        print(f"Extracting {label}...")
        actions = extract_gait_actions(gait_name, fb, lr)
        print(f"  {len(actions)} steps")

        # Repeat the sequence
        for _ in range(n_repeats):
            all_actions.append(actions)
            all_labels.extend([label] * len(actions))

    # Concatenate all sequences
    actions_array = np.vstack(all_actions)

    print(f"\nTotal dataset: {len(actions_array)} steps")

    return {
        'actions': actions_array,
        'labels': all_labels,
        'n_actions': len(actions_array),
    }


def main():
    """Main extraction function."""
    print("=" * 70)
    print("Extracting Sunfounder PiDog Demonstration Data")
    print("=" * 70)

    # Define gaits to extract
    gaits_to_extract = [
        # (gait_name, fb, lr, label)
        ('trot', Trot.FORWARD, Trot.STRAIGHT, 'trot_forward'),
        ('trot', Trot.BACKWARD, Trot.STRAIGHT, 'trot_backward'),
        ('trot', Trot.FORWARD, Trot.LEFT, 'trot_left'),
        ('trot', Trot.FORWARD, Trot.RIGHT, 'trot_right'),
        ('walk', Walk.FORWARD, Walk.STRAIGHT, 'walk_forward'),
        ('walk', Walk.BACKWARD, Walk.STRAIGHT, 'walk_backward'),
        ('walk', Walk.FORWARD, Walk.LEFT, 'walk_left'),
        ('walk', Walk.FORWARD, Walk.RIGHT, 'walk_right'),
    ]

    # Create dataset
    dataset = create_demonstration_dataset(gaits_to_extract, n_repeats=10)

    # Create output directory
    output_dir = Path(__file__).parent.parent / "datasets"
    output_dir.mkdir(exist_ok=True)

    # Save dataset
    output_path = output_dir / "sunfounder_demos.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)

    print(f"\n{'=' * 70}")
    print(f"Dataset saved to: {output_path}")
    print(f"Size: {output_path.stat().st_size / 1024:.1f} KB")
    print(f"{'=' * 70}")

    # Print statistics
    print("\nDataset Statistics:")
    print(f"  Total actions: {dataset['n_actions']}")
    print(f"  Action shape: {dataset['actions'].shape}")
    print(f"  Action range: [{dataset['actions'].min():.3f}, {dataset['actions'].max():.3f}]")

    # Print sample
    print("\nSample actions (first 3 steps of trot_forward):")
    for i in range(min(3, len(dataset['actions']))):
        print(f"  Step {i}: {dataset['actions'][i]}")


if __name__ == "__main__":
    main()
