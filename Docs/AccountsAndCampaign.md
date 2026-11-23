# Local account and campaign testing

## Unity Authentication setup

Use the Unity Cloud project already linked in `Edit > Project Settings > Services`. In the Unity Dashboard, open **Authentication > Player Authentication**, add the **Username & Password** identity provider, and save it. The same project must have Lobby and Relay enabled for online matches. No service key or password belongs in this repository.

At launch, choose **Sign In**, **Create Account**, or **Continue as Guest**. Two local test windows use separate `-ugs-profile` values; sign in to **two different player accounts** when testing an online match. Creating a player account happens in the game UI. The account ID shown after sign-in is the Unity Authentication player ID, which is also used by inventory and the online match identity. Display names and inventory are stored locally under that ID for this development phase; they do not yet synchronize across computers. Passwords are handled by Unity Authentication and are never written to the local save.

To keep an existing guest's Unity ID, choose **Link Guest Account** on the ready screen and supply a new username/password. Existing `local_<guid>` dummy-account data is copied once into the first Unity player ID used by that launch profile, provided that target has no save yet; the old file remains as a backup. Use the link path after continuing as a guest if that guest has inventory you want to keep. This project does not yet offer password reset; Unity's username/password reset requires a trusted server flow. The Unity provider must be enabled before Create Account, Sign In, or Link succeeds.

## Fifteen campaign levels

The Campaign menu exposes all 15 levels for local testing. They are grouped into five **Buttons**, five **Live Commands**, and five **FSM Bot** challenges. Each has authored instructions, an opponent algorithm and action interval, a round target, and a time limit. The existing three campaign scenes are templates selected by control mode; `CampaignCatalog` supplies the level configuration at runtime. Wins are recorded per player ID and shown in the menu. The first win in each level grants its listed coin reward; replaying it does not grant that reward again.

| Levels | Player input | Opponents | Final challenge |
| --- | --- | --- | --- |
| 1-1 to 1-5 | On-screen buttons | Primitive, FSM, behavior tree | Best of 3 against the behavior tree |
| 2-1 to 2-5 | Live commands | Primitive, FSM, behavior tree, utility | Best of 3 against utility |
| 3-1 to 3-5 | Predefined FSM bot | Primitive, FSM, behavior tree, utility | Best of 3 against the behavior tree |

Open Campaign from the main menu, scroll right through the cards, and choose any level. Each match displays its own instructions before the start button. Victory records completion and the first-win reward in `Application.persistentDataPath/PlayerData/<UnityPlayerId>.json`.
