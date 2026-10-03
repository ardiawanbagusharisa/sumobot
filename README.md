# Sumobot
Sumobot is a 2D, top-down sumo robot battle game intended to encourage player to learn the very basic of programming and artificial intelligence (AI). Inspired by real [Sumobot competition](https://www.sumobot.ca/competition) and [Battlesnake](https://play.battlesnake.com/leaderboards), we want to make this game (also AI competition platform) to be as simple and as open as possible. 

<p align="center">
<img width="600" alt="Sumobot" src="https://github.com/user-attachments/assets/6ad93b14-fcd7-4cb0-af7f-20888053b1c0" />  
</p>

<p align="center">
<img width="600" alt="Sumobot" src="https://github.com/user-attachments/assets/a4205870-d0c0-4f86-80c4-36a8996fa675" />  
  
Youtube: https://youtu.be/i5KUPYFWv8s 
</p>
 
# Our purposes 
1. Enhance the players' motivation and encourage players to learn at least the concept of computational thinking, and eventually programming basics and AI. It is predicted that in the future there would be more jobs related to STEM including AI. 
2. Create a sense of experimentation for the players to develop their logical thinking by providing live command terminal and AI script submission. 
3. Players may use the AI scripts they have created as portfolio. Imagine, you are a top player in one of the leaderboards, it proves that you have technical skill. Therefore, it increase your chance to find a software or AI engineering job, hopefully. 

## For Players
Go to [players page](https://github.com/ardiawanbagusharisa/sumobot/wiki/Players-Page) if you are a Sumobot player and want to learn how to play the game, and how to create your own AI script or want join for the AI competition. 

## For Developers
Go to [developers page](https://github.com/ardiawanbagusharisa/sumobot/wiki/Developers-Page) if you are a developer who want to contribute in building the platform. 

## Test online multiplayer on one machine

The Windows test build supports multiple simultaneous instances. In Unity, use `Sumobot > Multiplayer > Build Windows Test Client`; this creates `Builds/Windows/Sumobot.exe` with single-instance locking disabled and background execution enabled. Then run:

```powershell
.\Tools\LaunchMultiplayerTest.ps1
```

The launcher opens the same build twice at 960×540 with different Unity Authentication profiles (`sumobot_p1` and `sumobot_p2`) and separate logs. Complete the dummy login in both windows. In player 1, choose **Multiplayer > Online**, choose Buttons/Keyboard or Live Commands, and create a room. In player 2, open the same Online panel, choose an input mode, refresh if necessary, and click player 1's room. Both clients then enter the same battle.

For an unattended connection test:

```powershell
.\Tools\LaunchMultiplayerTest.ps1 -AutoOnline
```

You can also double-click `Sumobot.exe` twice, but pass different `-ugs-profile` values if you launch it from a terminal; two clients using the same profile are treated as the same anonymous UGS player. Unity cannot safely open the exact same project in two normal Editor processes because the project is locked. Use one Editor plus a build, two builds, or Unity Multiplayer Play Mode/additional instances instead. See [Docs/Multiplayer.md](Docs/Multiplayer.md) for service setup, protocol details, and troubleshooting.

---

## Micro-Competition 

<html>
<body>
<!--StartFragment-->

Rank | Bot | Win-rate | Action Duration | Actions | Collisions
-- | -- | -- | -- | -- | --
1 | **DAPPO_Cimin** :crown: | **0.81 (0.27)** | 13.29 (17.98) | 117.98 (114.58) | 9.18 (5.64)
2 | NN (baseline) | 0.77 (0.36) | 5.94 (5.65) | 76.5 (68.43) | 8.15 (4.03)
3 | FSM_Anandan | 0.7 (0.35) | 16.13 (19.87) | 224.11 (265.83) | 7.97 (5.61)
8 | ... | ... | ... | ... | ...
<!--EndFragment-->
</body>
</html>

Go to [Micro-Competition May 2026](https://github.com/ardiawanbagusharisa/sumobot/wiki/Micro%E2%80%90Competition-May2026) to see the detailed of our first micro-competition. 
