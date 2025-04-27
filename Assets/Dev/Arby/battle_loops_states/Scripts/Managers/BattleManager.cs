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
        Preparing,
        Countdown,
        Battle,
        BattleEnded,
        Reset
    }

    public class BattleManager : MonoBehaviour
    {
        public static BattleManager Instance { get; private set; }

        public BattleInputType RobotInputType = BattleInputType.Keyboard;
        public BattleState CurrentState { get; private set; } = BattleState.Preparing;
        public float BattleTime = 60f;
        public BattleInfo BattleInfo;

        public event Action<BattleState> OnPostStateChanged;
        public event Action<float> OnCountdownChanged;
        public event Action<BattleInfo> OnBattleInfoChanged;
        public GameObject SumoPrefab;
        public List<Transform> StartPositions = new List<Transform>();

        private List<GameObject> players = new List<GameObject>();
        private float countdownTime = 3f;
        private event Action<int> onPlayerAdded;
        private Coroutine battleTimerCoroutine;
        private Dictionary<int, BattleInfo> battleInfos;

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
            battleInfos = new Dictionary<int, BattleInfo>();
            BattleInfo = new BattleInfo();

            onPlayerAdded += OnPlayerAdded;
            Prepare();
        }

        #region Core Battle Manager
        private void Prepare()
        {
            TransitionToState(BattleState.Preparing);

            StartPositions.ForEach(x => InitializePlayerByPosition(x));
        }

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
            if (players.Count == 2 && CurrentState == BattleState.Preparing)
            {
                StartCoroutine(StartCountdown());
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

                GetComponent<InputManager>().RegisterInput(playerIdx, player.GetComponent<SumoRobotController>(), RobotInputType);

                players.Add(player);
                onPlayerAdded.Invoke(playerIdx);
            }
        }

        private IEnumerator StartCountdown()
        {
            BattleInfo.Time = Mathf.CeilToInt(BattleTime);
            ChangeBattleInfo(BattleInfo);

            TransitionToState(BattleState.Countdown);
            Debug.Log("Battle starting in...");

            float timer = countdownTime;
            while (timer > 0)
            {
                Debug.Log(Mathf.Ceil(timer));
                OnCountdownChanged?.Invoke(timer);
                yield return new WaitForSeconds(1f);
                timer -= 1f;
            }

            StartBattle();
        }

        private IEnumerator StartBattleTimer()
        {
            float timer = BattleTime;
            while (timer > 0)
            {
                var time = Mathf.CeilToInt(timer);
                Debug.Log(time);
                BattleInfo.Time = time;

                if (time <= 1)
                {
                    ChangeBattleInfo(BattleInfo);
                    EndBattle(null);
                    yield return new WaitForSeconds(1f);
                }
                else
                {
                    ChangeBattleInfo(BattleInfo);
                }

                yield return new WaitForSeconds(1f);
                timer -= 1f;
            }
        }

        private void StartBattle()
        {
            Debug.Log("Battle Start!");
            TransitionToState(BattleState.Battle);
        }

        private void EndBattle(GameObject winner)
        {
            BattleInfo.Rounds += 1;

            // Draw
            if (winner == null)
            {
                ChangeBattleInfo(BattleInfo);
                TransitionToState(BattleState.BattleEnded);
                StartCoroutine(ResetBattle());
                return;
            }

            var winnerId = winner.GetComponent<SumoRobot>().IdInt;
            Debug.Log($"Winner: {winnerId}");

            if (winnerId == 0)
            {
                BattleInfo.LeftPlayer.Score += 1;
            }
            else
            {
                BattleInfo.RightPlayer.Score += 1;
            }
            ChangeBattleInfo(BattleInfo);

            TransitionToState(BattleState.BattleEnded);
            StartCoroutine(ResetBattle());
        }

        private IEnumerator ResetBattle()
        {
            yield return new WaitForSeconds(3f);
            foreach (var player in players)
            {
                player.GetComponent<SumoRobotController>().Reset();
            }
            TransitionToState(BattleState.Reset);
            yield return new WaitForSeconds(1f);
            StartCoroutine(StartCountdown());
        }

        private void OnPlayerOutOfArena(int playerIdx)
        {
            // Find player whos winner
            EndBattle(players.Find(p => p.GetComponent<SumoRobot>().IdInt != playerIdx));
        }

        private void TransitionToState(BattleState newState)
        {
            Debug.Log($"State Transition: {CurrentState} → {newState}");
            CurrentState = newState;

            BattleInfo.battleState = newState;
            ChangeBattleInfo(BattleInfo);

            switch (newState)
            {
                case BattleState.Battle:
                    battleTimerCoroutine = StartCoroutine(StartBattleTimer());
                    players.ForEach((p) => p.GetComponent<SumoRobotController>().SetMovementEnabled(true));
                    break;
                case BattleState.BattleEnded:
                    battleInfos[BattleInfo.Rounds] = BattleInfo;
                    StopCoroutine(battleTimerCoroutine);
                    players.ForEach((p) => p.GetComponent<SumoRobotController>().SetMovementEnabled(false));
                    break;
            }


            OnPostStateChanged?.Invoke(newState);
        }


        private void ChangeBattleInfo(BattleInfo info)
        {
            BattleInfo = info;

            OnBattleInfoChanged?.Invoke(info);
        }
        #endregion

        #region Battle Info Getters
        public Dictionary<int, BattleInfo> GetLog()
        {
            return battleInfos;
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
    public int Rounds;
    public BattleState battleState;
    public BattlePlayerInfo LeftPlayer;
    public BattlePlayerInfo RightPlayer;
    public BattleWinner GetWinner()
    {
        if (battleState != BattleState.BattleEnded)
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
    public SumoRobot Sumo;
    public SumoRobotController SumoRobotController;
    public int Score = 0;
}