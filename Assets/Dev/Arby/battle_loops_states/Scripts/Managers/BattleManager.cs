using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using BattleLoop;
using CoreSumoRobot;
using Unity.VisualScripting;
using UnityEngine;
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
        public BattleInfo BattleInfo;
        public Dictionary<int, BattleInfo> RoundInfo;
        public event Action<float> OnCountdownChanged;
        public event Action<BattleInfo> OnBattleInfoChanged;
        public event Action<Dictionary<int, BattleInfo>> OnRoundInfoChanged;
        public GameObject SumoPrefab;
        public List<Transform> StartPositions = new List<Transform>();


        private List<GameObject> players = new List<GameObject>();
        private float countdownTime = 3f;
        private event Action<int> onPlayerAdded;
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
            onPlayerAdded -= OnPlayerAdded;
            // players.ForEach(p => p.gameObject.GetComponent<SumoRobotController>().OnOutOfArena -= OnPlayerOutOfArena);
        }
        private void Start()
        {
            RoundInfo = new Dictionary<int, BattleInfo>();
            BattleInfo = new BattleInfo();
            PreBattle_Prepare();
        }

        #region Core API (public)
        // usecase exmaple for start button or rematch buttonf or now
        public void Battle_Start()
        {
            Battle_Prepare();
        }
        #endregion


        #region PreBattle (private)

        // For now only initializing, Battle wont be played until someone Call [Battle_Start()]
        private void PreBattle_Prepare()
        {
            TransitionToState(BattleState.PreBatle_Preparing);
        }
        #endregion

        #region Battle
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
        #region Logic 
        private void OnPlayerAdded(int index)
        {
            Debug.Log($"Player registered: {index}");

            var battlePlayerInfo = new BattlePlayerInfo();
            battlePlayerInfo.Sumo = players[index].GetComponent<SumoRobot>();
            battlePlayerInfo.SumoRobotController = players[index].GetComponent<SumoRobotController>();
            battlePlayerInfo.Id = index;
            battlePlayerInfo.Score = 0;

            if (index == 0)
            {

                BattleInfo.LeftPlayer = battlePlayerInfo;
            }
            else
            {
                BattleInfo.RightPlayer = battlePlayerInfo;
            }

            ChangeBattleInfo(BattleInfo);

            players[index].GetComponent<SumoRobotController>().SetMovementEnabled(false);

            // Auto-start when 2 players registered
            if (players.Count == 2 && CurrentState == BattleState.Battle_Preparing)
            {
                Battle_Countdown();
            }
        }

        private void InitializePlayerByPosition(Transform playerPosition)
        {

            var player = Instantiate(SumoPrefab, playerPosition.position, playerPosition.rotation);
            if (!players.Contains(player))
            {
                var playerIdx = players.Count;
                var isLeftSide = playerIdx == 0;

                // [Todo]: Wrap with a function
                player.GetComponent<SumoRobot>().IdInt = playerIdx;
                player.GetComponent<SumoRobotController>().StartPosition = playerPosition;
                player.GetComponent<SumoRobotController>().ChangeFaceColor(isLeftSide);
                player.GetComponent<SumoRobotController>().OnOutOfArena += OnPlayerOutOfArena;

                GetComponent<InputManager>().RegisterInput(player, RobotInputType);

                players.Add(player);
                onPlayerAdded.Invoke(playerIdx);
            }
        }
        private void Deinitialize()
        {
            players.ForEach(p =>
            {
                GetComponent<InputManager>().UnregisterInput(p, RobotInputType);
                Destroy(p);
            });
            players = new List<GameObject>();
            RoundInfo = new Dictionary<int, BattleInfo>();
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
                var time = Mathf.CeilToInt(timer);
                Debug.Log(time);
                BattleInfo.Time = time;

                ChangeBattleInfo(BattleInfo);

                if (time <= 1)
                {
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
            foreach (var player in players)
            {
                player.GetComponent<SumoRobotController>().ResetForNewBattle();
            }
            TransitionToState(BattleState.Battle_Reset);
            yield return new WaitForSeconds(1f);
        }

        private void OnPlayerOutOfArena(int playerIdx)
        {
            Debug.Log("OnPlayerOutOfArena");

            // Find player whos winner
            var winner = players.Find(p => p.GetComponent<SumoRobot>().IdInt != playerIdx);

            // check whether [winner] is LeftSide
            if (winner.GetComponent<SumoRobot>().IsLeftSide)
            {
                BattleInfo.LeftPlayer.Score += 1;
                BattleInfo.WinnerEachRound[BattleInfo.Rounds] = BattleWinner.Left;
            }
            else
            {
                BattleInfo.RightPlayer.Score += 1;
                BattleInfo.WinnerEachRound[BattleInfo.Rounds] = BattleWinner.Right;
            }
            Battle_End();
        }

        private void TransitionToState(BattleState newState)
        {
            Debug.Log($"State Transition: {CurrentState} → {newState}");
            CurrentState = newState;

            BattleInfo.battleState = newState;


            switch (newState)
            {
                // Prebatle
                case BattleState.PreBatle_Preparing:
                    RoundInfo = new Dictionary<int, BattleInfo>();
                    BattleInfo = new BattleInfo();
                    break;
                // Prebatle

                // Battle
                case BattleState.Battle_Preparing:
                    onPlayerAdded += OnPlayerAdded;

                    StartPositions.ForEach(x => InitializePlayerByPosition(x));
                    BattleInfo.Rounds = 1;
                    BattleInfo.Time = Mathf.CeilToInt(BattleTime);
                    break;
                case BattleState.Battle_Countdown:
                    StartCoroutine(StartCountdown());
                    break;
                case BattleState.Battle_Ongoing:
                    battleTimerCoroutine = StartCoroutine(StartBattleTimer());
                    players.ForEach((p) => p.GetComponent<SumoRobotController>().SetMovementEnabled(true));
                    break;
                case BattleState.Battle_End:
                    StopCoroutine(battleTimerCoroutine);
                    players.ForEach((p) => p.GetComponent<SumoRobotController>().SetMovementEnabled(false));
                    StartCoroutine(ResetBattle());
                    break;
                case BattleState.Battle_Reset:
                    if (BattleInfo.Rounds == 3)
                    {
                        PostBattle_ShowResult();
                    }
                    else
                    {
                        BattleInfo.Rounds += 1;
                        //Start a round again
                        Battle_Countdown();
                    }

                    break;
                // Battle


                // Post Battle
                case BattleState.PostBattle_ShowResult:
                    Deinitialize();
                    BattleInfo.WinnerEachRound.Clear();
                    RoundInfo.Clear();
                    break;

                    // Post Battle
            }

            ChangeBattleInfo(BattleInfo);
        }


        private void ChangeBattleInfo(BattleInfo info)
        {
            BattleInfo = info;
            RoundInfo[BattleInfo.Rounds] = BattleInfo;
            foreach (var round in RoundInfo)
            {
                Debug.Log($"ChangeBattleInfo Round: {round.Key}, Winner: ${round.Value.GetWinner()}");
            }
            OnBattleInfoChanged?.Invoke(info);
            OnRoundInfoChanged?.Invoke(RoundInfo);
        }
        #endregion

        #region Battle Info Getters
        public Dictionary<int, BattleInfo> GetLog()
        {
            return RoundInfo;
        }
        #endregion
    }


}

public enum BattleWinner
{
    Left,
    Right,
    Draw,
    Ongoing,
}

public class BattleInfo
{
    public int Id;
    public int Time;
    public int Rounds = 0;
    public BattleState battleState;
    public BattlePlayerInfo LeftPlayer;
    public BattlePlayerInfo RightPlayer;

    // Not good, should create SetWinner to easily handle current round winner and total winner
    public Dictionary<int, BattleWinner> WinnerEachRound = new Dictionary<int, BattleWinner>();
    public BattleWinner GetWinner()
    {
        if (battleState < BattleState.Battle_End)
        {
            return BattleWinner.Ongoing;
        }

        var leftScore = LeftPlayer.Score;
        var rightScore = RightPlayer.Score;
        if (leftScore == rightScore)
        {
            return BattleWinner.Draw;
        }
        else if (rightScore > LeftPlayer.Score)
        {
            return BattleWinner.Right;
        }
        else
        {
            return BattleWinner.Left;
        }
    }


}

public class BattlePlayerInfo
{
    public int Id;
    public int Score = 0;
    public SumoRobot Sumo;
    public SumoRobotController SumoRobotController;
}