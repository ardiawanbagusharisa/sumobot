using System;
using System.Collections;
using System.Collections.Generic;
using BattleLoop;
using CoreSumoRobot;
using Unity.VisualScripting;
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
        public InputType BattleInputType = InputType.UI;
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
        // private List<GameObject> players = new List<GameObject>();

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

        void OnDisable()
        {
            Battle.LeftPlayer.OnOutOfArena -= OnPlayerOutOfArena;
            Battle.RightPlayer.OnOutOfArena -= OnPlayerOutOfArena;
        }

        void Update()
        {
            if (CurrentRound != null && CurrentState == BattleState.Battle_Ongoing)
            {
                ElapsedTime += Time.deltaTime;
            }
        }

        public void SetLeftDefaultSkill(int type)
        {
            Battle.LeftPlayer.Skill.Type = type == 0 ? ERobotSkillType.Boost : ERobotSkillType.Stone;
        }

        public void SetRightDefaultSkill(int type)
        {
            Battle.RightPlayer.Skill.Type = type == 0 ? ERobotSkillType.Boost : ERobotSkillType.Stone;
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

            if (Battle.LeftPlayer == null && Battle.RightPlayer == null)
            {
                return;
            }
            TransitionToState(BattleState.Battle_Preparing);
        }
        #endregion

        #region Core Logic 
        private void InitializePlayerByPosition(Transform playerPosition)
        {
            GameObject player = Instantiate(SumoPrefab, playerPosition.position, playerPosition.rotation);

            // Detect the position of position.x < 0: meaning LeftSide (0), otherwise it's RightSide (1)
            PlayerSide side = playerPosition.position.x < 0 ? PlayerSide.Left : PlayerSide.Right;

            // Initialize player components
            SumoRobotController controller = player.GetComponent<SumoRobotController>();
            controller.InitializeForBattle(side, playerPosition);
            controller.OnOutOfArena += OnPlayerOutOfArena;

            // Check whether player left or right, assign to Battle data
            if (controller.Side == PlayerSide.Left)
            {
                Battle.LeftPlayer = controller;
            }
            else
            {
                Battle.RightPlayer = controller;
            }

            Debug.Log($"Player registered: {side}");
        }

        IEnumerator AllPlayersReady()
        {
            // Delay state transition reaction
            yield return new WaitForSeconds(0.5f);
            TransitionToState(BattleState.Battle_Countdown);
        }

        private void Deinitialize()
        {
            Battle.LeftPlayer.InputProvider = null;
            Battle.RightPlayer.InputProvider = null;
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
            while (timer >= 0 && CurrentState == BattleState.Battle_Ongoing)
            {
                int time = Mathf.CeilToInt(timer);
                Debug.Log(time);
                CurrentRound.TimeLeft = time;

                // In order to update UI for time left in realtime, we need to call [ChangeBattleInfo]
                UpdateBattleData();

                yield return new WaitForSeconds(1f);
                timer -= 1f;
            }

            TransitionToState(BattleState.Battle_End);
        }


        private IEnumerator ResetBattle()
        {
            yield return new WaitForSeconds(3f);
            Battle.LeftPlayer.GetComponent<SumoRobotController>().ResetForNewBattle();
            Battle.RightPlayer.GetComponent<SumoRobotController>().ResetForNewBattle();
            TransitionToState(BattleState.Battle_Reset);
            yield return new WaitForSeconds(1f);
        }

        private void OnPlayerOutOfArena(PlayerSide side)
        {
            if (CurrentState != BattleState.Battle_Ongoing) return;

            Debug.Log("OnPlayerOutOfArena");

            // Find player who's winner
            SumoRobotController winner;

            if (side == PlayerSide.Left)
            {
                winner = Battle.RightPlayer;
            }
            else
            {
                winner = Battle.LeftPlayer;
            }

            if (winner == null)
            {
                Debug.LogWarning("Winner not found!");
                return;
            }

            Battle.SetRoundWinner(winner);

            TransitionToState(BattleState.Battle_End);
        }

        private void TransitionToState(BattleState newState)
        {
            Debug.Log($"State Transition: {CurrentState} → {newState}");
            CurrentState = newState;

            switch (CurrentState)
            {
                // Prebattle
                case BattleState.PreBatle_Preparing:
                    Battle = new Battle(Guid.NewGuid().ToString(), RoundSystem);
                    StartPositions.ForEach(x => InitializePlayerByPosition(x));
                    break;

                // Battle
                case BattleState.Battle_Preparing:
                    Battle.ClearWinner();
                    CurrentRound = new Round(1, Mathf.CeilToInt(BattleTime));

                    Battle.LeftPlayer.ResetForNewBattle();
                    Battle.RightPlayer.ResetForNewBattle();
                    InputManager.Instance.PrepareInput(Battle.LeftPlayer);
                    InputManager.Instance.PrepareInput(Battle.RightPlayer);
                    StartCoroutine(AllPlayersReady());
                    break;
                case BattleState.Battle_Countdown:
                    ElapsedTime = 0;
                    if (!gameObject.IsDestroyed() && countdownCoroutine != null)
                    {
                        StopCoroutine(countdownCoroutine);
                    }
                    countdownCoroutine = StartCoroutine(StartCountdown());
                    break;
                case BattleState.Battle_Ongoing:
                    battleTimerCoroutine = StartCoroutine(StartBattleTimer());

                    Battle.LeftPlayer.SetSkillEnabled(true);
                    Battle.LeftPlayer.SetMovementEnabled(true);
                    Battle.RightPlayer.SetSkillEnabled(true);
                    Battle.RightPlayer.SetMovementEnabled(true);
                    break;
                case BattleState.Battle_End:
                    if (!gameObject.IsDestroyed())
                    {
                        StopCoroutine(battleTimerCoroutine);
                    }

                    Battle.LeftPlayer.SetSkillEnabled(false);
                    Battle.LeftPlayer.SetMovementEnabled(false);
                    Battle.RightPlayer.SetSkillEnabled(false);
                    Battle.RightPlayer.SetMovementEnabled(false);
                    StartCoroutine(ResetBattle());
                    break;
                case BattleState.Battle_Reset:
                    if (Battle.GetBattleWinner() != null)
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

            UpdateBattleData();
        }

        // Call this when we need to trigger OnBattleChanged immediately
        private void UpdateBattleData()
        {

            if (CurrentRound != null)
            {
                Battle.CurrentRound = CurrentRound;
                Battle.Rounds[CurrentRound.RoundNumber] = CurrentRound;
            }

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

    public Dictionary<int, SumoRobotController> Winners
    {
        get;
        private set;
    } = new Dictionary<int, SumoRobotController>();

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

    public void SetRoundWinner(SumoRobotController winner)
    {
        if (winner.Side == PlayerSide.Left)
        {
            LeftWinCount += 1;
        }
        else
        {
            RightWinCount += 1;
        }

        CurrentRound.RoundWinner = winner;
        Winners[CurrentRound.RoundNumber] = winner;
    }

    public SumoRobotController GetBattleWinner()
    {
        Debug.Log($"[Battle][GetBattleWinner] leftWinCount: {LeftWinCount}, rightWinCount: {RightWinCount}");

        // winningTreshold is a treshold to help of deciding who has more different score based on BestOfN
        int winningTreshold = 0;

        switch (RoundSystem)
        {
            case RoundSystem.BestOf1:
                winningTreshold = 1;
                break;
            case RoundSystem.BestOf3:
                winningTreshold = 2;
                break;
            case RoundSystem.BestOf5:
                winningTreshold = 3;
                break;
        }

        int scoreDifference = Math.Abs(LeftWinCount - RightWinCount);
        if (scoreDifference >= winningTreshold)
        {
            if (LeftWinCount > RightWinCount)
            {
                Debug.Log($"[Battle][GetBattleWinner] Left!");
                return LeftPlayer;
            }
            else
            {
                Debug.Log($"[Battle][GetBattleWinner] Right!");
                return RightPlayer;
            }
        }

        // Check whether current round reaches max round
        if (CurrentRound.RoundNumber == (int)RoundSystem)
        {
            if (LeftWinCount == RightWinCount)
            {
                Debug.Log($"[Battle][GetBattleWinner] Draw!");
            }
            else if (LeftWinCount > RightWinCount)
            {
                Debug.Log($"[Battle][GetBattleWinner] Left!");
                return LeftPlayer;
            }
            else
            {
                Debug.Log($"[Battle][GetBattleWinner] Right!");
                return RightPlayer;
            }
        }

        return null;
    }


    public void ClearWinner()
    {
        Winners.Clear();
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
    public SumoRobotController RoundWinner;
    public Round(int roundNumber, int time)
    {
        RoundNumber = roundNumber;
        TimeLeft = time;
    }
}
