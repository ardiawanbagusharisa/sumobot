import json
import csv
import glob

def situation_to_text(before):
    # Create a simple situation text from the 'before' state
    pos = before.get('position', {})
    lin_v = before.get('linear_velocity', {})
    rot = before.get('rotation', {})
    ang_v = before.get('angular_velocity', 0)
    # Free format, can be changed to be more "human language" later
    return f"pos:{pos.get('x',0):.2f},{pos.get('y',0):.2f}; rot:{rot.get('z',0):.0f}; vel:{lin_v.get('x',0):.2f},{lin_v.get('y',0):.2f}; ang_vel:{ang_v:.2f}"

def event_to_action(event):
    act_type = event['data'].get('type')
    param = event['data'].get('parameter')
    # Translate to strategy labels (turn_left_15, turn_right_45, etc.)
    if act_type is None:
        return None
    act_type = act_type.lower()
    if act_type.startswith("turnleft") and param:
        return f"turn_left_{abs(int(float(param)))}"
    if act_type.startswith("turnright") and param:
        return f"turn_right_{abs(int(float(param)))}"
    if act_type in ["accelerate", "dash", "boost"]:
        return act_type
    return None

def parse_battle_log(json_path, writer):
    with open(json_path) as f:
        log = json.load(f)

    for game in log.get("games", []):
        for rnd in game.get("rounds", []):
            for event in rnd.get("events", []):
                actor = event.get("actor")
                
                if actor not in ["LeftPlayer", "RightPlayer"]:
                    continue
                action = event_to_action(event)
                before = event['data'].get('before')
                if not action or not before:
                    continue
                situation = situation_to_text(before)
                writer.writerow({"situation": situation, "strategy": action})

def main():
    output_csv = "sumobot_slm_dataset.csv"
    with open(output_csv, "w", newline="") as csvfile:
        fieldnames = ["situation", "strategy"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for path in glob.glob("battle_*.json"):
            print(f"Parsing {path} ...")
            parse_battle_log(path, writer)
    print(f"Selesai! Dataset tersimpan di {output_csv}")

if __name__ == "__main__":
    main()
