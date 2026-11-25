# Local account and campaign testing

## Unity Authentication setup

Use the Unity Cloud project already linked in `Edit > Project Settings > Services`. In the Unity Dashboard, open **Authentication > Player Authentication**, add the **Username & Password** identity provider, and save it. The same project must have Lobby and Relay enabled for online matches. No service key or password belongs in this repository.

At launch, choose **Sign In**, **Create Account**, or **Continue as Guest**. Two local test windows use separate `-ugs-profile` values; sign in to **two different player accounts** when testing an online match. Creating a player account happens in the game UI. The account ID shown after sign-in is the Unity Authentication player ID, which is also used by inventory and the online match identity. Display names and inventory are stored locally under that ID for this development phase; they do not yet synchronize across computers. Passwords are handled by Unity Authentication and are never written to the local save.

To keep an existing guest's Unity ID, choose **Link Guest Account** on the ready screen and supply a new username/password. Existing `local_<guid>` dummy-account data is copied once into the first Unity player ID used by that launch profile, provided that target has no save yet; the old file remains as a backup. Use the link path after continuing as a guest if that guest has inventory you want to keep. This project does not yet offer password reset; Unity's username/password reset requires a trusted server flow. The Unity provider must be enabled before Create Account, Sign In, or Link succeeds.

## Fifteen campaign levels

The Campaign menu exposes all 15 levels for local testing. They are grouped into five **Buttons**, five **Live Commands**, and five **FSM Bot** challenges. Each level has its own ScriptableObject in `Assets/Resources/Campaign/`; the three campaign scenes remain reusable templates. Each asset sets the opponent, time limit, arena radius, spawn positions, task, instructions, and first-completion reward. When a task is fulfilled, a Mission Complete sticker appears immediately. The battle stays visible for five seconds before the result panel opens. Only levels 1-5 and 2-5 require a ring-out; the other levels do not require winning the match. A failed task can be retried. Existing saved level IDs and first-completion rewards remain valid.

| Levels | Player input | Tasks |
| --- | --- | --- |
| 1-1 to 1-5 | On-screen buttons or W/A/D/E/Q keyboard controls | Move, turn and approach, aligned dash, edge recovery, ring-out |
| 2-1 to 2-5 | Live commands | Command movement, sequence, aligned dash, edge recovery, ring-out |
| 3-1 to 3-5 | Predefined FSM bot | Observe Approaching, Attacking, Dodging, Recovering, then the engagement sequence; answer a question after each observation |

Open Campaign from the main menu, scroll through the 15 authored cards, and choose any level. The mission panel shows instructions before battle and progress during play. Completing a task records progress and the one-time coin reward in `Application.persistentDataPath/PlayerData/<UnityPlayerId>.json`. Circular arena radii are 8, 10, or 12 world units; both spawn positions follow the chosen radius. FSM questions appear only after their target state or state sequence has been observed.
