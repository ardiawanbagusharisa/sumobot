using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using BattleLoop;
using CoreSumoRobot;
using JetBrains.Annotations;
using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.PlayerLoop;
using UnityEngine.UI;

namespace BattleLoop
{
    public enum BattleState
    {
        PreBatle_Preparing,


        // [BattleState.Preparing] is very initial state, 
        // A battle state will not back at [BattleState.Preparing] unless a Rematch created
        Battle_Preparing,
        Battle_Countdown,
        Battle_Ongoing,
        Battle_End,
        // The next state of [BattleState.Reset] is [BattleState.Countdown] 
        Battle_Reset,
        // End of Sub-states of Battle


        PostBattle_ShowResult,
    }

    public class BattleManager : MonoBehaviour
    {
        public static BattleManager Instance { get; private set; }

        public BattleInputType RobotInputType = BattleInputType.Keyboard;
        public BattleState CurrentState { get; private set; } = BattleState.PreBatle_Preparing;
        public float BattleTime = 60f;
        public RoundSystem RoundSystem = RoundSystem.BestOf3;
        public Battle Battle;
        public Round CurrentRound;
        public GameObject SumoPrefab;
        public List<Transform> StartPositions = new List<Transform>();
        public event Action<float> OnCountdownChanged;
        public event Action<Battle> OnBattleChanged;
        public float ElapsedTime = 0;


        private List<GameObject> players = new List<GameObject>();
        private float countdownTime = 3f;
        private Coroutine battleTimerCoroutine;


        private void Awake()
        {
            if (Instance != null)
            {
                Destroy(gameObject);
                return;
            }
            Instance = this;
        }
        void OnDestroy()
        {
            // players.ForEach(p => p.gameObject.GetComponent<SumoRobotController>().OnOutOfArena -= OnPlayerOutOfArena);
        }
        private void Start()
        {
            PreBattle_Prepare();
        }
        void Update()
        {
            if (CurrentRound != null && CurrentRound.BattleState == BattleState.Battle_Ongoing)
            {
                ElapsedTime += Time.deltaTime;
            }
        }

        #region PreBattle

        // For now, only initializing. Battle wont be played until someone Call [Battle_Start()]
        private void PreBattle_Prepare()
        {
            TransitionToState(BattleState.PreBatle_Preparing);
        }
        #endregion

        #region Battle

        // In order to start a battle, this function can be called from outside, e.g. "Start" or "Rematch" button
        public void Battle_Start()
        {
            Battle_Prepare();
        }

        private void Battle_Prepare()
        {
            TransitionToState(BattleState.Battle_Preparing);
        }

        private void Battle_Countdown()
        {
            TransitionToState(BattleState.Battle_Countdown);
        }

        private void Battle_Ongoing()
        {
            TransitionToState(BattleState.Battle_Ongoing);
        }

        private void Battle_End()
        {
            TransitionToState(BattleState.Battle_End);
        }
        #endregion

        #region PostBattle
        private void PostBattle_ShowResult()
        {
            TransitionToState(BattleState.PostBattle_ShowResult);
        }
        #endregion

        // Mostly private
        #region Core Logic 
        private void InitializePlayerByPosition(Transform playerPosition)
        {
            // THis is used when rematch button pressed. 
            // We don't need to reinitialize players, reset the players instead 
            if (players.Count == 2)
            {
                players.ForEach(p => p.GetComponent<SumoRobotController>().ResetForNewBattle());
                StartCoroutine(AllPlayerAreReadyToPlay());
                return;
            }

            GameObject player = Instantiate(SumoPrefab, playerPosition.position, playerPosition.rotation);
            if (!players.Contains(player))
            {
                int playerIdx = players.Count;
                bool isLeftSide = playerIdx == 0;

                // Initialize player components
                player.GetComponent<SumoRobot>().IdInt = playerIdx;
                player.GetComponent<SumoRobotController>().UpdateFaceColor();
                player.GetComponent<SumoRobotController>().StartPosition = playerPosition;
                player.GetComponent<SumoRobotController>().OnOutOfArena += OnPlayerOutOfArena;


                // Initialize battleplayer participants
                BattlePlayer battlePlayerInfo = new BattlePlayer();
                battlePlayerInfo.Id = playerIdx;
                battlePlayerInfo.Sumo = player.GetComponent<SumoRobot>();
                battlePlayerInfo.SumoRobotController = player.GetComponent<SumoRobotController>();
                battlePlayerInfo.Score = 0;
                battlePlayerInfo.SumoRobotController.SetMovementEnabled(false);

                // Check whether player left or right
                if (battlePlayerInfo.Sumo.IsLeftSide)
                {
                    Battle.LeftPlayer = battlePlayerInfo;
                }
                else
                {
                    Battle.RightPlayer = battlePlayerInfo;
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

        // Delay state transition reaction
        IEnumerator AllPlayerAreReadyToPlay()
        {
            yield return new WaitForSeconds(0.5f);
            Battle_Countdown();
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

            float timer = countdownTime;
            while (timer > 0)
            {
                Debug.Log(Mathf.Ceil(timer));
                OnCountdownChanged?.Invoke(timer);
                yield return new WaitForSeconds(1f);
                timer -= 1f;
            }
            Battle_Ongoing();
        }

        private IEnumerator StartBattleTimer()
        {
            float timer = BattleTime;
            while (timer > 0)
            {
                int time = Mathf.CeilToInt(timer);
                Debug.Log(time);
                CurrentRound.TimeLeft = time;

                // In order to update UI for time left in realtime, we need to call [ChangeBattleInfo]
                ChangeBattleInfo();

                if (time <= 1)
                {
                    // Draw isn't tested yet
                    Battle_End();

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

        private void OnPlayerOutOfArena(int playerIdx)
        {
            Debug.Log("OnPlayerOutOfArena");

            // Find player who's winner
            GameObject winner = players.Find(p => p.GetComponent<SumoRobot>().IdInt != playerIdx);

            // check whether [winner] is LeftSide
            if (winner.GetComponent<SumoRobot>().IsLeftSide)
            {
                Battle.SetRoundWinner(BattleWinner.Left);
            }
            else
            {
                Battle.SetRoundWinner(BattleWinner.Right);
            }
            Battle_End();
        }

        private void TransitionToState(BattleState newState)
        {
            Debug.Log($"State Transition: {CurrentState} → {newState}");
            CurrentState = newState;

            switch (newState)
            {
                // Prebatle
                case BattleState.PreBatle_Preparing:
                    Battle = new Battle();
                    Battle.RoundSystem = RoundSystem;

                    break;
                // Prebatle

                // Battle
                case BattleState.Battle_Preparing:
                    Battle.Clear();

                    CurrentRound = new Round();
                    CurrentRound.RoundNumber = 1;
                    CurrentRound.TimeLeft = Mathf.CeilToInt(BattleTime);

                    StartPositions.ForEach(x => InitializePlayerByPosition(x));
                    break;
                case BattleState.Battle_Countdown:
                    ElapsedTime = 0;
                    StartCoroutine(StartCountdown());
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
                        PostBattle_ShowResult();
                    }
                    else
                    {
                        int previousRound = Battle.CurrentRound.RoundNumber;

                        CurrentRound = new Round();
                        CurrentRound.TimeLeft = Mathf.CeilToInt(BattleTime);
                        CurrentRound.RoundNumber = previousRound + 1;

                        Debug.Log($"CurrentRound.RoundNumber {CurrentRound.RoundNumber}");
                        //Start a round again
                        Battle_Countdown();
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


        private void ChangeBattleInfo()
        {

            if (CurrentRound != null)
            {
                CurrentRound.BattleState = CurrentState;

                Battle.CurrentRound = CurrentRound;
                Battle.Rounds[CurrentRound.RoundNumber] = CurrentRound;
            }
            OnBattleChanged?.Invoke(Battle);
        }
        #endregion
    }


}

public enum BattleWinner
{
    Left,
    Right,
    Draw,
    None,
}

public enum RoundSystem
{
    BestOf1 = 1,
    BestOf3 = 3,
    BestOf5 = 5,
}

public class Battle
{
    public string UUID;
    public RoundSystem RoundSystem;
    public BattlePlayer LeftPlayer;
    public BattlePlayer RightPlayer;
    public Round CurrentRound;

    public Dictionary<int, BattleWinner> WinnerEachRound
    {
        get;
        private set;
    } = new Dictionary<int, BattleWinner>();
    public Dictionary<int, Round> Rounds = new Dictionary<int, Round>();

    public int LeftWinCount;
    public int RightWinCount;

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

public class Round
{
    public int Id;
    public int TimeLeft;
    public int RoundNumber = 0;
    public BattleState BattleState;
    public BattleWinner RoundWinner;

    // [Todo]: add log for player actions
}

public class BattlePlayer
{
    public int Id;
    public int Score = 0;
    public SumoRobot Sumo;
    public SumoRobotController SumoRobotController;
}