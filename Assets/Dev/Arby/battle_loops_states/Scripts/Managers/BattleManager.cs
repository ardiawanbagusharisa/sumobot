using System;
using System.Collections;
using System.Collections.Generic;
using BattleLoop;
using CoreSumoRobot;
using UnityEngine;

namespace BattleLoop
{
    public enum BattleState
    {
        PreBatle_Preparing,     // Initial state in scene, used only once. 
        Battle_Preparing,       // Initializes a new battle or a rematch, players are setup.
        Battle_Countdown,       // Countdown before the round starts.
        Battle_Ongoing,         // Main gameplay state.
        Battle_End,             // Battle ends, players are disabled.
        Battle_Reset,           // Prepares next round or ends match.
        PostBattle_ShowResult,  // Final state to show results. 
    }

    public enum BattleWinner
    {
        Left,   // Left player wins
        Right,  // Right player wins
        Draw,   // Draw
        None,   // No winner yet
    }

    public enum RoundSystem
    {
        BestOf1 = 1,    // Need 1 winning round
        BestOf3 = 3,    // Need 2 winning rounds
        BestOf5 = 5,    // Need 3 winning rounds
    }

    public class BattleManager : MonoBehaviour
    {
        // Singleton 
        public static BattleManager Instance { get; private set; }

        // Configuration 
        public BattleInputType RobotInputType = BattleInputType.Keyboard;
        public RoundSystem RoundSystem = RoundSystem.BestOf3;
        public float BattleTime = 60f;
        public float CountdownTime = 3f;
        public List<Transform> StartPositions = new List<Transform>();
        public GameObject SumoPrefab;

        // State & Internal 
        public BattleState CurrentState = BattleState.PreBatle_Preparing;
        public float ElapsedTime = 0;
        public Battle Battle;
        public Round CurrentRound;
        private List<GameObject> players = new List<GameObject>();

        // Events 
        public event Action<float> OnCountdownChanged;
        public event Action<Battle> OnBattleChanged;
        private Coroutine battleTimerCoroutine;
        private Coroutine countdownCoroutine;

        #region Unity
        private void Awake()
        {
            if (Instance != null)
            {
                Destroy(gameObject);
                return;
            }
            Instance = this;

        }

        void Start()
        {
            TransitionToState(BattleState.PreBatle_Preparing);
        }

        void Update()
        {
            if (CurrentRound != null && CurrentRound.BattleState == BattleState.Battle_Ongoing)
            {
                ElapsedTime += Time.deltaTime;
            }
        }

        // Start a battle. Can also be called from UI.
        public void Battle_Start()
        {
            if (CurrentState == BattleState.Battle_Preparing || 
                CurrentState == BattleState.Battle_Countdown || 
                CurrentState == BattleState.Battle_Ongoing)
            {
                return;
            }
            TransitionToState(BattleState.Battle_Preparing);
        }
        #endregion

        #region Core Logic 
        private void InitializePlayerByPosition(Transform playerPosition)
        {
            // [Todo] Can this be merged with the bottom part?
            if (players.Count == 2)
            {
                players.ForEach(p =>
                {
                    p.GetComponent<SumoRobotController>().ResetForNewBattle();
                    GetComponent<InputManager>().RegisterInput(p, RobotInputType);
                });
                StartCoroutine(AllPlayerAreReadyToPlay());
                return;
            }

            GameObject player = Instantiate(SumoPrefab, playerPosition.position, playerPosition.rotation);
            // [Todo] If contained? 
            if (!players.Contains(player))
            {
                int playerIdx = players.Count;
                bool isLeftSide = playerIdx == 0;

                // Initialize player components
                SumoRobotController controller = player.GetComponent<SumoRobotController>();
                controller.InitializeForBattle(playerIdx, playerPosition);
                controller.OnOutOfArena += OnPlayerOutOfArena;

                // Check whether player left or right
                if (controller.Side == PlayerSide.Left)
                {
                    Battle.LeftPlayer = controller;
                }
                else
                {
                    Battle.RightPlayer = controller;
                }

                players.Add(player);

                GetComponent<InputManager>().RegisterInput(player, RobotInputType);

                Debug.Log($"Player registered: {playerIdx}");
                
                // Auto-start when 2 players registered
                if (players.Count == 2 && CurrentState == BattleState.Battle_Preparing)
                {
                    StartCoroutine(AllPlayerAreReadyToPlay());
                }
            }
        }

        IEnumerator AllPlayerAreReadyToPlay()
        {
            // Delay state transition reaction
            yield return new WaitForSeconds(0.5f);
            TransitionToState(BattleState.Battle_Countdown);
        }

        private void Deinitialize()
        {
            players.ForEach(p =>
            {
                GetComponent<InputManager>().UnregisterInput(p, RobotInputType);
            });
        }

        private IEnumerator StartCountdown()
        {
            Debug.Log("Battle starting in...");
            yield return new WaitForSeconds(1f);

            float timer = CountdownTime;
            while (timer > 0 && CurrentState == BattleState.Battle_Countdown)
            {
                Debug.Log(Mathf.Ceil(timer));
                OnCountdownChanged?.Invoke(timer);
                yield return new WaitForSeconds(1f);
                timer -= 1f;
            }
            TransitionToState(BattleState.Battle_Ongoing);
        }

        private IEnumerator StartBattleTimer()
        {
            float timer = BattleTime;
            while (timer > 0 && CurrentState == BattleState.Battle_Ongoing)
            {
                int time = Mathf.CeilToInt(timer);
                Debug.Log(time);
                CurrentRound.TimeLeft = time;

                // In order to update UI for time left in realtime, we need to call [ChangeBattleInfo]
                ChangeBattleInfo();

                if (time <= 1)
                {
                    // Draw isn't tested yet
                    TransitionToState(BattleState.Battle_End);

                    yield return new WaitForSeconds(1f);
                }

                yield return new WaitForSeconds(1f);
                timer -= 1f;
            }
        }


        private IEnumerator ResetBattle()
        {
            yield return new WaitForSeconds(3f);
            foreach (GameObject player in players)
            {
                player.GetComponent<SumoRobotController>().ResetForNewBattle();
            }
            TransitionToState(BattleState.Battle_Reset);
            yield return new WaitForSeconds(1f);
        }

        private void OnPlayerOutOfArena(PlayerSide side)
        {
            Debug.Log("OnPlayerOutOfArena");

            // Find player who's winner
            //SumoRobotController winner = players.Find(p => p.GetComponent<SumoRobotController>().Side != side).GetComponent<SumoRobotController>();
            // Find the surviving player safely
            SumoRobotController winner = null;
            foreach (var player in players)
            {
                if (player == null) continue; // object might be destroyed
                var controller = player.GetComponent<SumoRobotController>();
                if (controller != null && controller.Side != side)
                {
                    winner = controller;
                    break;
                }
            }

            if (winner == null)
            {
                Debug.LogWarning("Winner not found!");
                return;
            }

            // check whether [winner] is LeftSide
            if (winner.Side == PlayerSide.Left)
            {
                Battle.SetRoundWinner(BattleWinner.Left);
            }
            else
            {
                Battle.SetRoundWinner(BattleWinner.Right);
            }
            ChangeBattleInfo();

            TransitionToState(BattleState.Battle_End);
        }

        private void TransitionToState(BattleState newState)
        {
            Debug.Log($"State Transition: {CurrentState} → {newState}");
            CurrentState = newState;

            switch (CurrentState)
            {
                // Prebatle
                case BattleState.PreBatle_Preparing:
                    Battle = new Battle(Guid.NewGuid().ToString(), this.RoundSystem);
                    break;

                // Battle
                case BattleState.Battle_Preparing:
                    Battle.Clear();
                    CurrentRound = new Round(1, Mathf.CeilToInt(BattleTime));
                    StartPositions.ForEach(x => InitializePlayerByPosition(x));
                    break;
                case BattleState.Battle_Countdown:
                    ElapsedTime = 0;
                    if (countdownCoroutine != null)
                    {
                        StopCoroutine(countdownCoroutine);
                    }
                    countdownCoroutine = StartCoroutine(StartCountdown());
                    break;
                case BattleState.Battle_Ongoing:
                    battleTimerCoroutine = StartCoroutine(StartBattleTimer());
                    players.ForEach((p) => p.GetComponent<SumoRobotController>().SetSkillEnabled(true));
                    players.ForEach((p) => p.GetComponent<SumoRobotController>().SetMovementEnabled(true));
                    break;
                case BattleState.Battle_End:
                    StopCoroutine(battleTimerCoroutine);
                    players.ForEach((p) => p.GetComponent<SumoRobotController>().SetSkillEnabled(false));
                    players.ForEach((p) => p.GetComponent<SumoRobotController>().SetMovementEnabled(false));
                    StartCoroutine(ResetBattle());
                    break;
                case BattleState.Battle_Reset:
                    if (Battle.GetBattleWinner() != BattleWinner.None)
                    {
                        TransitionToState(BattleState.PostBattle_ShowResult);
                    }
                    else
                    {
                        int previousRound = Battle.CurrentRound.RoundNumber;

                        // Create n+1 round
                        CurrentRound = new Round(previousRound + 1, Mathf.CeilToInt(BattleTime));
                        
                        Debug.Log($"CurrentRound.RoundNumber {CurrentRound.RoundNumber}");
                        //Start a round again
                        TransitionToState(BattleState.Battle_Countdown);
                    }

                    break;
                // Battle


                // Post Battle
                case BattleState.PostBattle_ShowResult:
                    Deinitialize();
                    break;
                    // Post Battle
            }

            ChangeBattleInfo();
        }

        // Call this when we need to trigger OnBattleChanged immediately
        private void ChangeBattleInfo()
        {

            if (CurrentRound != null)
            {
                CurrentRound.BattleState = CurrentState;

                Battle.CurrentRound = CurrentRound;
                Battle.Rounds[CurrentRound.RoundNumber] = CurrentRound;
            }
            Debug.Log($"OnBattleChanged {OnBattleChanged == null}");
            OnBattleChanged?.Invoke(Battle);
        }
        #endregion
    }

}

[Serializable]
public class Battle
{
    public string BattleID;
    public RoundSystem RoundSystem;
    public SumoRobotController LeftPlayer;
    public SumoRobotController RightPlayer;
    public Round CurrentRound;

    public Dictionary<int, BattleWinner> WinnerEachRound
    {
        get;
        private set;
    } = new Dictionary<int, BattleWinner>();
    public Dictionary<int, Round> Rounds = new Dictionary<int, Round>();

    public int LeftWinCount;
    public int RightWinCount;

    // [Todo]: Utilize of handling log for state & loop changes
    public Dictionary<float, string> BattleLog;

    public Battle(string battleID, RoundSystem roundSystem)
    {
        BattleID = battleID;
        RoundSystem = roundSystem;
    }

    public void SetRoundWinner(BattleWinner winner)
    {
        switch (winner)
        {
            case BattleWinner.Left:
                LeftWinCount += 1;
                break;
            case BattleWinner.Right:
                RightWinCount += 1;
                break;
            case BattleWinner.Draw:
                break;
        }

        CurrentRound.RoundWinner = winner;
        WinnerEachRound[CurrentRound.RoundNumber] = winner;
    }

    public BattleWinner GetBattleWinner()
    {
        Debug.Log($"[Battle][GetBattleWinner] leftWinCount: {LeftWinCount}, rightWinCount: {RightWinCount}");

        // Calculate winner based on Best Of N rules.
        switch (RoundSystem)
        {
            case RoundSystem.BestOf1:
                return CheckWinnerByDifference(includeDraw: true);
            case RoundSystem.BestOf3:
                BattleWinner winner3 = CheckWinnerByTreshold(2);
                if (winner3 != BattleWinner.None)
                {
                    return winner3;
                }

                if (CurrentRound.RoundNumber == 3)
                {
                    return CheckWinnerByDifference(includeDraw: true);
                }
                break;
            case RoundSystem.BestOf5:
                BattleWinner winner5 = CheckWinnerByTreshold(3);
                if (winner5 != BattleWinner.None)
                {
                    return winner5;
                }

                if (CurrentRound.RoundNumber > 3)
                {
                    return CheckWinnerByDifference(includeDraw: false);
                }
                break;
        }
        return BattleWinner.None;
    }

    private BattleWinner CheckWinnerByDifference(bool includeDraw)
    {
        if (LeftWinCount == RightWinCount)
        {
            if (includeDraw)
            {
                Debug.Log($"[Battle][GetBattleWinner] Draw");
                return BattleWinner.Draw;
            }
            else
            {
                Debug.Log($"[Battle][GetBattleWinner] None");
                return BattleWinner.None;
            }

        }
        else if (LeftWinCount > RightWinCount)
        {
            Debug.Log($"[Battle][GetBattleWinner] Left");
            return BattleWinner.Left;
        }
        else
        {
            Debug.Log($"[Battle][GetBattleWinner] Right");
            return BattleWinner.Right;
        }
    }

    // Detect if one of sides has [treshold] more scores. 
    // e.g. in BestOf3 the treshold is 2, when the difference score is 2 - 0, Left is the Winner.
    private BattleWinner CheckWinnerByTreshold(int treshold)
    {
        if (Math.Abs(LeftWinCount - RightWinCount) == treshold)
        {
            if (LeftWinCount > RightWinCount)
            {
                Debug.Log($"[Battle][GetBattleWinner] Left Win");
                return BattleWinner.Left;
            }
            else
            {
                Debug.Log($"[Battle][GetBattleWinner] Right Win");
                return BattleWinner.Right;
            }
        }
        return BattleWinner.None;
    }

    public void Clear()
    {
        WinnerEachRound.Clear();
        LeftWinCount = 0;
        RightWinCount = 0;
    }
}

[Serializable]
public class Round
{
    public int Id;
    public int TimeLeft;
    public int RoundNumber = 0;
    public BattleState BattleState;
    public BattleWinner RoundWinner;

    public Round(int roundNumber, int time)
    {
        RoundNumber = roundNumber;
        TimeLeft = time;
    }
}
