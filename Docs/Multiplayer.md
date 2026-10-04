# Sumobot online multiplayer

The online prototype is a two-player, client-hosted match built with Unity Authentication, Multiplayer Services sessions/lobby queries, Relay using DTLS, Unity Transport, and Netcode for GameObjects.

## Required account and Unity Dashboard setup

Create or use one Unity ID with Owner or Manager access to a Unity Cloud organization. In the Unity Editor, sign in with that Unity ID and open `Edit > Project Settings > Services`. Link this checkout to a Unity Cloud project, then enable Authentication, Lobby, and Relay in its Unity Dashboard.

This repository contains the previous cloud project ID `b9ca5334-4f50-46eb-8fd3-14bcbc1933f9`, but a command-line build reported the Editor connection as unbound. If that project does not appear in Project Settings, either ask its owner to add your Unity ID or create and link a new Unity Cloud project. Do not paste service keys into the repository.

Anonymous Authentication is sufficient for this phase, so you do not need to create two player email accounts. Each launch profile creates and caches a distinct anonymous Unity player ID. For production-grade recoverable inventory and leaderboard identity, configure Unity Player Accounts (or another supported identity provider) later and let players link their anonymous account.

The game uses the default UGS environment unless `-ugs-environment=<name>` is supplied. Create a `development` environment before launching with `-ugs-environment=development`.

## Build and run two clients

1. Open `Sumobot > Multiplayer > Build Windows Test Client` in the Unity Editor. The development build is written to `Builds/Windows/Sumobot.exe`.
2. Run `Tools/LaunchMultiplayerTest.ps1`. It opens the same build twice with isolated UGS profiles `sumobot_p1` and `sumobot_p2`, separate log files, and a shared one-run matchmaking pool. The pool prevents either client from joining a stale session left by an earlier local test.
3. Complete the local dummy login in each window.
4. In the first window, open Multiplayer > Online, choose **Buttons / Keyboard** or **Live Commands**, then select **Create room**.
5. In the second window, choose its input mode and join the first player's room. The list refreshes automatically every five seconds and also has a manual Refresh button.
6. Once both players are connected, a host-authoritative 30-second ready timer appears in the room panel. Each player may click **Ready now**; both ready clicks load Battle immediately, or the room loads Battle when the timer expires. The selected input modes are locked on arrival. Both scene peers then see the synchronized 10-second preparation countdown before the host starts the battle. Battle will not start before the joining client's scene and controllers are ready.

The room browser uses UGS session queries filtered to the launcher's one-run match pool. Full rooms are removed from the list. Closing a waiting host room leaves/deletes it; leaving during battle immediately awards the match to the remaining player.

For a named UGS environment:

```powershell
.\Tools\LaunchMultiplayerTest.ps1 -EnvironmentName development
```

For an unattended connection smoke test, use `-AutoOnline`. The launcher creates and joins the room automatically, and both test clients mark themselves ready so the 30-second room timer can finish early:

```powershell
.\Tools\LaunchMultiplayerTest.ps1 -AutoOnline
```

`Force Single Instance` must remain disabled and `Run In Background` enabled. The multiplayer build command applies those settings to the build without permanently changing the project settings.

Each online window uses the same local keyboard map regardless of arena side: `W` moves forward relative to the bot, `A`/`D` rotate, `E` dashes, and `Q` uses the selected skill. The sumo controls intentionally have no reverse action, so `S` is unused. Only the focused window receives keyboard events; the unfocused process continues networking because Run In Background is enabled. Offline same-keyboard play keeps separate left/right key maps.

You may also launch `Sumobot.exe` twice manually. Without command-line profile arguments, each live build reserves the first available persistent profile slot (`sumobot_local_1`, `sumobot_local_2`, and so on), preventing both windows from authenticating as the same UGS player. If using a terminal, you can still explicitly choose profiles such as `-ugs-profile=sumobot_p1` and `-ugs-profile=sumobot_p2`. Do not open the same Unity project in two standard Editor processes; use one Editor plus a build, two builds, or Multiplayer Play Mode additional instances.

## Multiplayer Play Mode

The installed Multiplayer Play Mode package can start an additional local instance. Configure distinct additional-instance arguments such as `-ugs-profile=sumobot_p2`; use `-ugs-profile=sumobot_p1` for the main player. Both instances must use different profiles or Unity Authentication treats them as the same anonymous player.

## Current authority model

- The session host is the left player and owns battle state and 2D physics.
- The joining client is the right player and sends validated input commands to the host.
- The host sends battle metadata, transforms, action-trail flags, skill selection, and dash/skill HUD state at 20 Hz. Dash and collision bursts use reliable visual-event messages, so both windows render those effects without running client-side physics. Individual particle patterns may differ.
- A host departure ends the match; host migration is not part of this prototype.
- If either player leaves or disconnects after the Battle scene loads, the connected opponent is immediately shown as the winner and the result is recorded. A network failure can only prove that the remote peer disappeared; production ranked play should validate forfeits on a trusted server.
- Online matches are human-vs-human. The scene's `BattleSimulator` and both predefined AI bots are disabled while an online session is active so they cannot compete with network player input.
- At the result screen, either player can request a rematch during a host-authoritative 15-second window. The other player's button changes to **Accept Rematch**; a new match starts only after both players accept before the timer expires.
- Finished online matches are written to the local Elo leaderboard on both peers using the synchronized dummy-account identities. This makes the complete flow testable without another service account. It is not a global or cheat-resistant leaderboard; a production deployment should replace the local store with a trusted server/Cloud Code backend.

## Local-first Market chat

The Market chat placeholder now sends and displays messages. It stores JSON-lines history under Unity's `Application.persistentDataPath`, and polls it with file sharing enabled. New messages move the authored scroll view to the bottom. The left tab lists the local game processes currently online, using short-lived presence heartbeats; a crashed client disappears after about eight seconds. That means the two local build instances can chat and see each other immediately without another account or paid backend. The item detail **Ask** button opens chat and pre-fills an item/creator mention. This is intentionally same-machine only; cross-device production chat and presence should be replaced with a moderated service such as Unity Vivox text chat or an authenticated application backend.
