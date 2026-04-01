import glob
import os
from huggingface_hub import hf_hub_download, list_repo_files
import pandas as pd
import json
import numpy as np
from tqdm import tqdm
import math
from pathlib import Path

arena_center = np.array([0.24, 1.97])
arena_radius = 4.73485
bot_rotate_speed = 200
base_tolerate = 0.01
tolerate_turn = 0.20

def fmt(num: float) -> str:
    return f"{num:.2f}" if num != 0 else "0"

GOAL_ACTION_MAP = {
    "Attack in optimal angle to enemy": [
        "Skill",
        "Dash",
        "Accelerate",
    ],
    "Attack the enemy directly at close range": [
        "Dash",
        "Accelerate",
    ],
    "Charge while turning": [
        "Accelerate",
        "Turn",
    ],
    "Turn to face the enemy before attacking": [
        "Turn",
        "Accelerate",
    ],
    "Charge towards enemy to close the gap": [
        "Skill",
        "Accelerate",
    ],
    "Adjust angle then approach enemy": [
        "Turn",
    ],
    "Reposition towards center before pursuing enemy": [
        "Skill",
        "Turn",
    ],
    "Circle the arena": [
        "TurnTolerate",
    ],
    "Hold position and observe enemy": [
        "Idle",
    ]
}

def normalize_360(angle_series):
    angle = angle_series % 360
    angle = np.where(angle < 0, angle + 360, angle)
    return angle

def signed_angle_unity(a, b):
    dot = max(min(a[0]*b[0] + a[1]*b[1], 1.0), -1.0)
    det = a[0] * b[1] - a[1] * b[0]
    return math.degrees(math.atan2(det, dot)), (det, dot)

def angle(ori_pos, ori_rot, target_pos):
    # Normalize rotation to [0, 360)
    z_rot = ori_rot % 360
    if z_rot < 0:
        z_rot += 360

    # Facing direction (Unity's Vector2.up rotated clockwise)
    rad = math.radians(-z_rot)  # <-- negative to match Unity’s clockwise rotation
    facing_dir = (math.sin(rad), math.cos(rad))

    # Same as Unity: target - origin
    to_target = (target_pos[0] - ori_pos[0], target_pos[1] - ori_pos[1])
    length = math.hypot(to_target[0], to_target[1])
    if length != 0:
        to_target = (to_target[0] / length, to_target[1] / length)

    signed_angle, _ = signed_angle_unity(facing_dir, to_target)

    return signed_angle, math.cos(np.deg2rad(signed_angle))

def distance_to_enemy(bot_x, bot_y, target_x, target_y):
    
    to_target = np.array([target_x, target_y]) - np.array([bot_x,bot_y])
    magnitude = np.sqrt(to_target[0]**2 + to_target[1]**2)
    to_target_norm = magnitude / (arena_radius * 2)
    
    return 1 - to_target_norm

def near_arena(bot_x, bot_y):
    to_target = np.array([bot_x, bot_y]) - arena_center
    magnitute = np.sqrt(to_target[0]**2 + to_target[1]**2)
    return (magnitute / arena_radius)

def facing_to_outside(bot_x, bot_y, bot_rot):

    # Vector from arena center to bot
    to_center = np.array([bot_x, bot_y]) - arena_center
    to_center_norm = to_center / np.linalg.norm(to_center)  # normalized direction

    # Forward vector from bot rotation
    rad = math.radians(-bot_rot)  # <-- negative to match Unity’s clockwise rotation
    facing_dir = (math.sin(rad), math.cos(rad))

    # Dot product between normalized vectors
    dot = np.dot(facing_dir, to_center_norm)
    return dot
    
def determine_goal(angle_score, distance_score, near_border, facing_outside, actedAt, angleToEnemy):
    # Case 1: excellent alignment
    if angle_score > 0.90:
        if distance_score < 0.5:
            return (
                "Charge towards enemy to close the gap",
                f"Since the AngleToEnemyScore is above 0.90 ({angle_score:.2f}) meaning it's perfectly aligned, "
                f"and the DistanceToEnemyScore is below 0.5 ({distance_score:.2f}) meaning the enemy is still far, "
                "it's best to charge forward to close the gap."
            )
        return (
            "Attack in optimal angle to enemy",
            f"Since the AngleToEnemyScore is above 0.90 ({angle_score:.2f}) meaning it's perfectly aligned, "
            f"and the DistanceToEnemyScore is more or equal than 0.5 ({distance_score:.2f}) meaning the enemy is within an effective range, "
            "it's best to attack directly at the optimal angle."
        )

    # Case 2: too close to border and facing out
    if near_border >= 0.75:
        if facing_outside > 0.5:
            return (
                "Circle the arena",
                f"Since the NearBorderArenaScore value is more or equal than 0.75 ({near_border:.2f}) meaning the bot is very close to the edge, "
                f"and FacingToArena is more than 0.5 ({facing_outside:.2f}) meaning the bot is oriented outward, "
                "circling the arena prevents going out of bounds."
            )

    # Case 3: moderately close to border
    if 0.62 <= near_border < 0.75:
        if facing_outside > -0.30:
            return (
                "Reposition towards center before pursuing enemy",
                f"Since the NearBorderArenaScore value is between 0.62 and 0.75 ({near_border:.2f}) meaning the bot is moderately close to the edge, "
                f"and FacingToArena more or equal than -0.30 ({facing_outside:.2f}) meaning the bot is somewhat outward, "
                "it's safer to reposition towards the center before engaging the enemy."
            )

    # Case 4: good angle and close
    if angle_score > 0.75:
        if distance_score < 0.5:
            return (
                "Charge while turning",
                f"Since the AngleToEnemyScore is more than 0.75 ({angle_score:.2f}) meaning the bot has good alignment, "
                f"and the DistanceToEnemyScore is less than 0.5 ({distance_score:.2f}) meaning the enemy is close, "
                "charging while turning maintains alignment during approach."
            )

    # Case 5: somewhat near border
    if near_border > 0.3:
        return (
            "Adjust angle then approach enemy",
            f"Since NearBorderArenaScore is more than 0.3 ({near_border:.2f}) meaning the bot is not fully safe from the edge, "
            f"and the alignment of AngleToEnemyScore is not perfect ({angle_score:.2f}), "
            "adjusting angle before approaching the enemy is the best choice."
        )

    # Default fallback
    return (
        "Turn to face the enemy before attacking",
        f"Since the AngleToEnemyScore is low ({angle_score:.2f}) meaning poor alignment, "
        f"and the DistanceToEnemyScore is {distance_score:.2f}, "
        "the bot should first turn to face the enemy before making an attack."
    )

def tweak_actions(actions, distance_score, angle_to_enemy, bot_pos_x, bot_pos_y, bot_rot, completion_mode):
    result = []

    for action in actions:
        dur = 0.1
        if action in ["Turn", "TurnTolerate"]:
            angle_to_center, _ = angle([bot_pos_x, bot_pos_y], bot_rot, arena_center)  # Angle to arena center
            if abs(angle_to_enemy) < abs(angle_to_center):
                if completion_mode=="short":
                    action = "TL" if angle_to_enemy > 0 else "TR"
                else:
                    action = "TurnLeft" if angle_to_enemy > 0 else "TurnRight"
                dur = max(0.1,np.abs(angle_to_enemy) / bot_rotate_speed)
            else:
                if completion_mode=="short":
                    action = "TL" if angle_to_center > 0 else "TR"
                else:
                    action = "TurnLeft" if angle_to_center > 0 else "TurnRight"
                dur = max(0.1, abs(angle_to_center) / bot_rotate_speed)
            
            if action == "TurnTolerate":
                dur = max(0.1, abs(dur - tolerate_turn))
            else:
                dur = max(0.1, abs(dur - base_tolerate))
        if action in ["Accelerate"]:
            if completion_mode=="short":
                action = "FWD"
            dur = max(0.1, 0.1 / distance_score)
        if action in ["Skill"]:
            if completion_mode=="short":
                action = "SK"
            dur = 0
        if action in ["Dash"]:
            if completion_mode=="short":
                action = "DS"
            dur = 0
        result.append((action, dur))
    return result


def generate_goal_and_actions(
        angle_score,
        distance_score, 
        near_border, 
        facing_outside, 
        signed_angle, 
        bot_pos_x,
        bot_pos_y,
        bot_rot, 
        actedAt, completion_mode = "normal"):
    goal, reason = determine_goal(angle_score, distance_score, near_border, facing_outside, actedAt, signed_angle)

    actions = GOAL_ACTION_MAP.get(goal)

    actions = tweak_actions(actions, distance_score, signed_angle,bot_pos_x,bot_pos_y,bot_rot,  completion_mode)

    final_actions = []
    for action, dur in actions:
        # Remove duration when it's Dash / Skill
        action_without_dur = ["Dash","Skill"] if completion_mode == "normal" else ["DS","SK"]
        if action in action_without_dur:
            final_actions.append(action)
        else:
            if completion_mode == "short":
                final_actions.append(f"{action}{dur:.2f}")
            else:
                final_actions.append(f"{action} with {dur:.2f} seconds")
   
    return reason, final_actions


def export_dataset(df, output_path, format="txt", completion_mode="normal", include_pos_rot=True):
    """
    Export dataset in either txt or jsonl format.

    Args:
    """

    with open(output_path, "w", encoding="utf-8") as f:
        for i, row in tqdm(df.iterrows(), total=len(df)):
            try:
                bot_pos = [row["BotPosX"], row["BotPosY"]]
                bot_rot = row["BotRot"]
                enemy_pos = [row["EnemyBotPosX"], row["EnemyBotPosY"]]
                actedAt = f"{row['StartedAt']:.2f}"

                signed_angle_result, signed_angle_result_norm = angle(
                    bot_pos, bot_rot, enemy_pos
                )
                distance_to_enemy_result = distance_to_enemy(
                    bot_pos[0], bot_pos[1], enemy_pos[0], enemy_pos[1]
                )
                near_arena_result = near_arena(bot_pos[0], bot_pos[1])
                facing_to_outside_result = facing_to_outside(
                    bot_pos[0], bot_pos[1], bot_rot
                )

                _, actions = generate_goal_and_actions(
                    signed_angle_result_norm,
                    distance_to_enemy_result,
                    near_arena_result,
                    facing_to_outside_result,
                    signed_angle_result,
                    bot_pos[0],
                    bot_pos[1],
                    bot_rot,
                    actedAt,
                    completion_mode
                )

                # Build prompt text
                prompt_str = (
                    f"AngleToEnemy={signed_angle_result:.2f}, "
                    f"AngleToEnemyScore={signed_angle_result_norm:.2f}, "
                    f"DistanceToEnemyScore={distance_to_enemy_result:.2f}, "
                    f"NearBorderArenaScore={near_arena_result:.2f}, "
                    f"FacingToArena={facing_to_outside_result:.2f}."
                )

                if include_pos_rot:
                    extra = (
                        f"BotPos=[{row['BotPosX']:.2f},{row['BotPosY']:.2f}], "
                        f"BotRot={int(row['BotRot'])}, "
                        f"EnemyPos=[{row['EnemyBotPosX']:.2f},{row['EnemyBotPosY']:.2f}], "
                        f"EnemyRot={int(row['EnemyBotRot'])}, "
                    )
                    prompt_str = extra + prompt_str

                if format == "txt":
                    line = f"{prompt_str} Result: {', '.join(actions)}"
                    f.write(line + "\n")
                elif format == "jsonl_prompt_completion":
                    line = f"You are a Sumobot assistant. Given this state: {prompt_str} Suggested Action:"
                    record = {
                        "prompt": line,
                        "completion": ', '.join(actions)
                    }
                    f.write(json.dumps(record) + "\n")
                elif format == "jsonl_message":
                    record = {
                        "messages": [
                            {"role": "system", "content": "You are a Sumobot assistant that decides actions based on game state."},
                            {"role": "user", "content": f"Given this game state: {prompt_str}"},
                            {"role": "assistant", "content": ', '.join(actions)}
                        ]
                    }
                    f.write(json.dumps(record) + "\n")
                elif format == "jsonl_text":
                    line = f"You are a Sumobot assistant. Given this state: {prompt_str} Suggested Action: {', '.join(actions)}"
                    record = {
                        "text": line,
                    }
                    f.write(json.dumps(record) + "\n")
                elif format == "jsonl_state_action":
                    record = {
                        "state": line,
                        "action": ', '.join(actions),
                    }
                    f.write(json.dumps(record) + "\n")
                

            except Exception as e:
                print(f"Error at row {i}: {e}")

def filter_inside_arena(df, margin=0.95):
    bot_dist = np.sqrt(df["BotPosX"]**2 + df["BotPosY"]**2)
    df["IsOutOfArena"] = bot_dist > (arena_radius * margin)
    return df[~df["IsOutOfArena"]].copy()


def find_project_root(target_name="sumobot"):
    path = Path(__file__).resolve()
    for parent in path.parents:
        if parent.name == target_name:
            return parent
    raise RuntimeError(f"Project root '{target_name}' not found")


def get_dataset_dir():
    root_dir = os.getcwd().split("/")[-1].lower()

    # For online notebook
    if root_dir == "content": 
        path = "dataset"
        os.makedirs(path, exist_ok=True)
        return "dataset"
    
    if root_dir=="slm" or root_dir=="llm" or "classification":
        os.makedirs("../dataset", exist_ok=True)
        return f"../dataset"
    else:
        os.makedirs("dataset", exist_ok=True)
        return "dataset"
    
def get_slm_dir():
    root_dir = os.getcwd().split("/")[-1]
    if root_dir=="slm":
        return f"./"
    else:
        return f"slm/"
    

def get_dataset(
    prefer_local: bool = True,
    inside_arena: bool = False,
    save_downloaded_dataset :bool = True,
    dataset_dir: str = None
):
    # Use provided dataset directory or get default one
    if dataset_dir:
        local_dataset_path = dataset_dir
        # Create directory if it doesn't exist
        os.makedirs(local_dataset_path, exist_ok=True)
        print(f"Using predefined dataset directory: {local_dataset_path}")
    else:
        local_dataset_path = get_dataset_dir()

    dfs = []

    if not prefer_local:  # Use HuggingFace
        dfs = get_dataset_from_hf(save_downloaded_dataset=save_downloaded_dataset, dataset_dir=local_dataset_path)
    else:
        print(f"Scanning dataset directory: {local_dataset_path}")
        csv_files = glob.glob(os.path.join(local_dataset_path, "*.csv"))

        if csv_files:
            print(f"Found {len(csv_files)} CSV file(s) in directory")
            for fname in csv_files:
                df = pd.read_csv(fname)
                df["source_file"] = os.path.basename(fname)
                dfs.append(df)
        else:
            print(f"No CSV files found in {local_dataset_path}")

    if not dfs:
        print("No dataset found locally, downloading from HuggingFace...")
        dfs = get_dataset_from_hf(save_downloaded_dataset=save_downloaded_dataset, dataset_dir=local_dataset_path)

    # Merge into one DataFrame
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df = merged_df.dropna(subset=["Name", "Duration"])
    if inside_arena:
        merged_df = filter_inside_arena(merged_df)
    print("Merged shape:", merged_df.shape)
    return merged_df, local_dataset_path

def get_dataset_from_hf(
        repo_id: str = "ardiawanbagus/sumobot-simulation-data",
        repo_dataset_path: str = "training-mlp-slm-llm",
        save_downloaded_dataset :bool = True,
        dataset_dir: str = None):

    print(f"Fetching dataset from HuggingFace repo: {repo_id}")
    dfs = []

    # Use provided dataset directory or get default one
    target_dir = dataset_dir if dataset_dir else get_dataset_dir()

    all_files = list_repo_files(repo_id=repo_id, repo_type="dataset")
    hf_csv_files = [f for f in all_files if f.startswith(repo_dataset_path) and f.endswith(".csv")]

    print(f"Auto-detected {len(hf_csv_files)} CSV files from HuggingFace.")
    print("\n".join(hf_csv_files))

    for fname in hf_csv_files:
        file_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=f"{repo_dataset_path}/{fname}" if not fname.startswith(repo_dataset_path) else fname
        )
        df = pd.read_csv(file_path)
        df["source_file"] = os.path.basename(fname)
        dfs.append(df)

        if save_downloaded_dataset:
            save_path = os.path.join(target_dir, os.path.basename(fname))
            df.to_csv(save_path, index=False)
            print(f"Saved: {save_path}")
    return dfs
