using System;
using System.Collections;
using System.Collections.Generic;
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
        public event Action<BattleState> OnPostStateChanged;
        public event Action<float> OnCountdownChanged;
        public GameObject SumoPrefab;
        public List<Transform> StartPositions = new List<Transform>();


        private List<GameObject> players = new List<GameObject>();
        private float countdownTime = 3f;
        private event Action<int> onPlayerAdded;

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
            onPlayerAdded += OnPlayerAdded;
            Prepare();
        }

        private void Prepare()
        {
            TransitionToState(BattleState.Preparing);

            StartPositions.ForEach(x => InitializePlayerByPosition(x));
        }

        private void OnPlayerAdded(int index)
        {
            Debug.Log($"Player registered: {index}");

            players[index].GetComponent<SumoRobotController>().SetMovementEnabled(false);
            // Auto-start when 2 players registered
            if (players.Count == 2 && CurrentState == BattleState.Preparing)
            {
                StartCoroutine(StartCountdown());
            }
        }

        private void Update()
        {
            switch (CurrentState)
            {
                case BattleState.Battle:
                    break;
            }
        }

        private void InitializePlayerByPosition(Transform playerPosition)
        {

            var player = Instantiate(SumoPrefab, playerPosition.position, playerPosition.rotation);
            if (!players.Contains(player))
            {
                var playerIdx = players.Count;
                var isLeftSide = playerIdx % 2 == 0;

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

        private void StartBattle()
        {
            Debug.Log("Battle Start!");
            TransitionToState(BattleState.Battle);
        }

        private void EndBattle(GameObject winner)
        {
            Debug.Log($"Winner: {winner.GetComponent<SumoRobot>().IdInt}");
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
            EndBattle(players.Find(p => p.GetComponent<SumoRobot>().IdInt != playerIdx));
        }

        private void TransitionToState(BattleState newState)
        {
            Debug.Log($"State Transition: {CurrentState} → {newState}");
            CurrentState = newState;

            switch (newState)
            {
                case BattleState.Battle:
                    players.ForEach((p) => p.GetComponent<SumoRobotController>().SetMovementEnabled(true));
                    break;
                case BattleState.BattleEnded:
                    players.ForEach((p) => p.GetComponent<SumoRobotController>().SetMovementEnabled(false));
                    break;
            }


            OnPostStateChanged?.Invoke(newState);
        }
    }


}