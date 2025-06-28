import csv
import random

# Define turn actions with angles
turn_left_actions = [f"turn_left_{a}" for a in [15, 45, 90]]
turn_right_actions = [f"turn_right_{a}" for a in [15, 45, 90]]

# List of valid single actions (including turn angles)
single_actions = [
    "accelerate",
    "dash",
    "boost",
    "stay"
] + turn_left_actions + turn_right_actions

# Combo actions for turning while moving/complex actions
combo_actions = []
for turn in turn_left_actions + turn_right_actions:
    combo_actions.append(f"{turn}+accelerate")
    combo_actions.append(f"dash+{turn}+accelerate")
    combo_actions.append(f"{turn}+dash+accelerate")
    combo_actions.append(f"{turn}+accelerate+boost")
combo_actions += ["accelerate+boost"]

num_rows = 500

fieldnames = [
    # Numeric features (same as old)
    "enemy_distance", "enemy_angle", "edge_distance", "center_distance",
    "enemy_stuck", "enemy_behind", "skill_ready", "dash_ready",
    # Log-like fields:
    "situation", "category", "round",
    # Strategy label
    "strategy"
]

def random_bool():
    return random.choice([0, 1])

def random_strategy():
    if random.random() < 0.65:
        return random.choice(single_actions)
    else:
        return random.choice(combo_actions)

def sort_strategy_combo(strategy):
    actions = [a.strip() for a in strategy.split('+')]
    def action_priority(a):
        if a.startswith("turn"): return 0
        if a.startswith("dash"): return 1
        if a.startswith("accelerate"): return 2
        if a.startswith("boost"): return 3
        return 4
    actions.sort(key=action_priority)
    return '+'.join(actions)

def features_to_situation_text(enemy_distance, enemy_angle, edge_distance, center_distance, enemy_stuck, enemy_behind, skill_ready, dash_ready):
    return (
        f"Enemy is {enemy_distance} meters ahead, "
        f"angle {enemy_angle} degrees, "
        f"edge distance {edge_distance}, "
        f"center distance {center_distance}, "
        f"enemy stuck: {bool(enemy_stuck)}, "
        f"enemy behind: {bool(enemy_behind)}, "
        f"skill ready: {bool(skill_ready)}, "
        f"dash ready: {bool(dash_ready)}"
    )

with open('sumobot_auto_dataset.csv', 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for i in range(num_rows):
        enemy_distance = round(random.uniform(0.1, 2.5), 2)
        enemy_angle = round(random.uniform(-180, 180), 1)
        edge_distance = round(random.uniform(0.1, 5.0), 2)
        center_distance = round(random.uniform(0.0, 5.0), 2)
        enemy_stuck = random_bool()
        enemy_behind = random_bool()
        skill_ready = random_bool()
        dash_ready = random_bool()

        # Log-like categories
        category = random.choice(["simulation", "practice", "tournament"])
        round_number = random.randint(1, 5)

        # Smart action selection
        if edge_distance < 0.7 and dash_ready:
            strategy = random.choice([
                f"dash+turn_left_{random.choice([45,90])}+accelerate",
                f"dash+turn_right_{random.choice([45,90])}+accelerate"
            ])
        elif abs(enemy_angle) > 40 and abs(enemy_angle) < 160:
            strategy = random.choice([
                f"turn_left_{random.choice([15,45,90])}+accelerate",
                f"turn_right_{random.choice([15,45,90])}+accelerate"
            ])
        elif enemy_stuck and skill_ready:
            strategy = "boost"
        elif enemy_distance < 0.4 and abs(enemy_angle) > 140:
            strategy = random.choice([
                f"turn_left_{random.choice([45,90])}+accelerate",
                f"turn_right_{random.choice([45,90])}+accelerate"
            ])
        else:
            strategy = random_strategy()
        strategy = sort_strategy_combo(strategy)

        # Situation (log-like, English)
        situation = features_to_situation_text(
            enemy_distance, enemy_angle, edge_distance, center_distance,
            enemy_stuck, enemy_behind, skill_ready, dash_ready
        )

        writer.writerow({
            "enemy_distance": enemy_distance,
            "enemy_angle": enemy_angle,
            "edge_distance": edge_distance,
            "center_distance": center_distance,
            "enemy_stuck": enemy_stuck,
            "enemy_behind": enemy_behind,
            "skill_ready": skill_ready,
            "dash_ready": dash_ready,
            "situation": situation,
            "category": category,
            "round": round_number,
            "strategy": strategy
        })

print("Hybrid dataset generated: sumobot_auto_dataset.csv")
