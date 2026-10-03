using System;
using System.Collections;
using System.Collections.Generic;
using SumoBot;
using SumoCore;
using SumoHelper;
using SumoInput;
using SumoLeaderboard;
using SumoMultiplayer;
using Unity.VisualScripting;
using UnityEngine;

namespace SumoManager
{
    #region Battle enums
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

    public enum BattleWinner
    {
        Left,
        Right,
        Draw,
    }
    #endregion

    public class BattleManager : MonoBehaviour
    {
        public static BattleManager Instance { get; private set; }

        #region Battle Configuration properties
        [Header("Battle Configuration")]
        public InputType LeftInputType = InputType.UI;
        public InputType RightInputType = InputType.UI;
        public RoundSystem RoundSystem = RoundSystem.BestOf3;
        public float BattleTime = 60f;
        public float CountdownTime = 3f;
        public float ActionInterval = 0.1f;
        public List<Transform> StartPositions = new();
        // public GameObject SumoPrefab;
        public GameObject LeftPlayerObject;
        public GameObject RightPlayerObject;
        public GameObject Arena;
        public float ArenaRadius;
        [HideInInspector] public bool RequireExternalStartConfirmation;
        [HideInInspector] public bool CampaignStartConfirmed;
        #endregion

        #region Runtime (readonly) properties 
        public BattleState CurrentState = BattleState.PreBatle_Preparing;
        public float ElapsedTime = 0;
        public float CountdownRemaining { get; private set; }
        public float TimeLeft => BattleTime - ElapsedTime;

        public Battle Battle;
        public BotManager BotManager;
        public PacingManager PacingManager;
        private BattleSimulator simulator;
        #endregion

        #region Events properties 
        public EventRegistry Events = new();
        public const string OnCountdownChanged = "OnCountdownChanged";  // [float]
        public const string OnBattleChanged = "OnBattleChanged"; // [Battle]
        public const string OnActionUpdate = "OnActionUpdate"; // [Battle]

        private Coroutine battleTimerCoroutine;
        private Coroutine countdownCoroutine;
        private float elapsedActionTime = 0f;

        // Guards against double-recording the same match (e.g. repeated state
        // broadcasts); reset when a new battle/rematch is prepared.
        private bool leaderboardRecorded = false;
        private bool remoteInputsInitialized = false;
        private bool remoteBattlePrepared = false;

        /// <summary>Rating changes of the match just recorded (null when nothing was recorded).</summary>
        public LeaderboardOutcome? LastLeaderboardOutcome { get; private set; }
        #endregion

        #region Unity methods 
        private void Awake()
        {
            if (Instance != null && Instance != this)
            {
                Destroy(gameObject);
                return;
            }
            Instance = this;
        }

        void OnEnable()
        {
            // [Todo]: Move game manager to the first scene
            _ = GameManager.Instance;

            simulator = GetComponent<BattleSimulator>();
            BotManager = GetComponent<BotManager>();
            PacingManager = GetComponent<PacingManager>();

            // The scene simulator is an offline AI-vs-AI test harness. Online
            // PvP is driven by one human on each peer and simulated by the host.
            if (OnlineBattleSession.IsActive && simulator != null)
                simulator.enabled = false;

            if (simulator.enabled)
            {
                if (simulator.Mode == SimulatorMode.Simple)
                    Init();
                simulator.PrepareSimulation();
            }
            else
                Init();
        }

        private void Init()
        {
            LogManager.UnregisterAction();
            LogManager.InitLog(false);
            LogManager.InitBattle();
            Battle = new Battle(Guid.NewGuid().ToString(), RoundSystem);
        }

        void Start()
        {
            var scale = Arena.transform.lossyScale;
            ArenaRadius = Arena.GetComponent<CircleCollider2D>().radius * ((scale.x + scale.y) / 2f);
            TransitionToState(BattleState.PreBatle_Preparing);
        }

        void OnDisable()
        {
            Battle.LeftPlayer.Events[SumoController.OnOutOfArena].Unsubscribe(OnPlayerOutOfArena);
            Battle.RightPlayer.Events[SumoController.OnOutOfArena].Unsubscribe(OnPlayerOutOfArena);
        }

        void Update()
        {
            // In an online client the host owns simulation and battle state. The
            // client only renders snapshots applied by OnlineBattleSession.
            if (OnlineBattleSession.IsClient)
                return;

            if (Battle.CurrentRound != null && CurrentState == BattleState.Battle_Ongoing)
            {
                ElapsedTime += Time.deltaTime;
                elapsedActionTime += Time.deltaTime;

                if (elapsedActionTime >= ActionInterval)
                {
                    elapsedActionTime = 0;

                    SumoController left = Battle.LeftPlayer;
                    SumoController right = Battle.RightPlayer;

                    BotManager.OnUpdate();

                    left.FlushInput();
                    right.FlushInput();

                    left.OnUpdate();
                    right.OnUpdate();

                    // Tick pacing handlers
                    if (PacingManager != null && PacingManager.isActiveAndEnabled)
                        PacingManager.Tick();
                }
            }
        }

        #endregion

        #region API methods
        public void Battle_Start()
        {
            // Scene buttons must not bypass the online peer-ready gate. Only
            // OnlineBattleSession may authorize the actual host transition.
            if (OnlineBattleSession.IsActive &&
                !OnlineBattleSession.IsHostStartAuthorized)
            {
                OnlineBattleSession.RequestStartFromSceneButton();
                return;
            }

            if (RequireExternalStartConfirmation && !CampaignStartConfirmed)
                return;

            if (CurrentState == BattleState.Battle_Preparing ||
                CurrentState == BattleState.Battle_Countdown ||
                CurrentState == BattleState.Battle_Ongoing)
            {
                return;
            }

            if (Battle.LeftPlayer == null && Battle.RightPlayer == null)
                return;
            TransitionToState(BattleState.Battle_Preparing);
        }

        /// <summary>Used by campaign instruction popups to release the start gate.</summary>
        public void ConfirmCampaignStart()
        {
            CampaignStartConfirmed = true;
            Battle_Start();
        }
        #endregion

        #region Core Logic methods 
        private void InitializeController(SumoController controller)
        {
            PlayerSide side = controller.transform.position.x < 0 ? PlayerSide.Left : PlayerSide.Right;

            if (controller.Side == PlayerSide.Left)
            {
                controller.Initialize(
                    side,
                    controller.transform,
                    GameManager.Instance.Left);

                Battle.LeftPlayer = controller;
            }
            else
            {
                controller.Initialize(
                    side,
                    controller.transform,
                    GameManager.Instance.Right);

                Battle.RightPlayer = controller;
            }

            controller.Events[SumoController.OnOutOfArena].Subscribe(OnPlayerOutOfArena);

            LogManager.LogBattleState(
                    data: new Dictionary<string, object>()
                    {
                        {"type", "Player"},
                        {"outPlayerSide", controller.Side},
                        {"skill", controller.Skill.Type},
                    });
            Logger.Info($"Player registered: {side}");
        }

        IEnumerator AllPlayersReady()
        {
            yield return new WaitForSeconds(0.5f);
            TransitionToState(BattleState.Battle_Countdown);
        }

        private IEnumerator StartCountdown()
        {
            yield return new WaitForSeconds(1f);

            float timer = CountdownTime;
            CountdownRemaining = timer;
            while (timer > 0 && CurrentState == BattleState.Battle_Countdown)
            {
                CountdownRemaining = timer;
                SFXManager.Instance.Play2D("ui_accept_small");
                Events[OnCountdownChanged].Invoke(new EventParameter(floatParam: timer));
                yield return new WaitForSeconds(1f);
                timer -= 1f;
            }
            CountdownRemaining = 0f;
            TransitionToState(BattleState.Battle_Ongoing);
        }

        private IEnumerator StartBattleTimer()
        {
            float timer = BattleTime;
            while (timer > 0 && CurrentState == BattleState.Battle_Ongoing)
            {
                timer -= Time.deltaTime;
                yield return null;
            }

            LogManager.FlushActionLog();
            PlayerSide? side = LogManager.GetWinnerByContactMade();
            if (side == null)
            {
                LogManager.SetRoundWinner(side.ToString());
                Battle.SetRoundWinner(side == PlayerSide.Left ? Battle.LeftPlayer : Battle.RightPlayer);
            }
            else
            {
                LogManager.SetRoundWinner("Draw");
                Battle.CurrentRound.RoundWinner = null;
                Battle.Winners[Battle.CurrentRound.RoundNumber] = null;
            }

            TransitionToState(BattleState.Battle_End);
        }

        private IEnumerator ResetBattle()
        {
            yield return new WaitForSeconds(2.5f);
            LogManager.LogLastPosition();
            yield return new WaitForSeconds(0.5f);

            Battle.LeftPlayer.Reset();
            Battle.RightPlayer.Reset();

            TransitionToState(BattleState.Battle_Reset);
            yield return new WaitForSeconds(1f);
        }

        private void OnPlayerOutOfArena(EventParameter param)
        {
            if (CurrentState != BattleState.Battle_Ongoing)
                return;
            Logger.Info("OnPlayerOutOfArena");
            PlayerSide Side = param.Side;
            SumoController winner = Side == PlayerSide.Left ? Battle.RightPlayer : Battle.LeftPlayer;

            if (winner == null)
            {
                Debug.LogWarning("Winner not found!");
                return;
            }

            Battle.SetRoundWinner(winner);
            LogManager.FlushActionLog();
            LogManager.SetRoundWinner(winner.Side.ToString());
            TransitionToState(BattleState.Battle_End);
        }

        private void TransitionToState(BattleState newState)
        {
            Logger.Info($"State Transition: {CurrentState} → {newState}");
            CurrentState = newState;

            if (Battle.CurrentRound == null || Battle.CurrentRound.RoundNumber == 0)
            {
                LogManager.LogBattleState(
                    data: new Dictionary<string, object>()
                    {
                    {"type", "battle_state"},
                    {"state", CurrentState.ToString()},
                    });
            }
            else
            {
                LogManager.LogBattleState(
                    includeInCurrentRound: true,
                    data: new Dictionary<string, object>()
                    {
                    {"type", "battle_state"},
                    { "battle_state", CurrentState.ToString()}
                    });
            }

            // Post-state
            switch (CurrentState)
            {
                // Prebattle
                case BattleState.PreBatle_Preparing:
                    if (LeftPlayerObject != null && RightPlayerObject != null)
                    {
                        InitializeController(LeftPlayerObject.GetComponent<SumoController>());
                        InitializeController(RightPlayerObject.GetComponent<SumoController>());
                    }
                    break;

                // Battle
                case BattleState.Battle_Preparing:
                    leaderboardRecorded = false;
                    remoteInputsInitialized = false;
                    LastLeaderboardOutcome = null;
                    SFXManager.Instance.Play2D("ui_accept");
                    LogManager.SetPlayerBots(BotManager.Left, BotManager.Right);
                    LogManager.UpdateMetadata(logTakenAction: false);
                    LogManager.StartGameLog();

                    Battle.ClearWinner();
                    Battle.CurrentRound = new Round(1, Mathf.CeilToInt(BattleTime));
                    LogManager.StartRound(Battle.CurrentRound.RoundNumber);

                    Battle.LeftPlayer.Reset();
                    Battle.RightPlayer.Reset();
                    InputManager.Instance.InitializeInput(Battle.LeftPlayer, LeftInputType);
                    InputManager.Instance.InitializeInput(Battle.RightPlayer, RightInputType);

                    LogManager.RegisterAction();
                    StartCoroutine(AllPlayersReady());
                    break;
                case BattleState.Battle_Countdown:
                    ElapsedTime = 0;

                    if (!gameObject.IsDestroyed() && countdownCoroutine != null)
                        StopCoroutine(countdownCoroutine);

                    countdownCoroutine = StartCoroutine(StartCountdown());
                    break;
                case BattleState.Battle_Ongoing:
                    battleTimerCoroutine = StartCoroutine(StartBattleTimer());

                    Battle.LeftPlayer.SetSkillEnabled(true);
                    Battle.RightPlayer.SetSkillEnabled(true);
                    break;
                case BattleState.Battle_End:
                    Battle.CurrentRound.FinishTime = ElapsedTime;

                    if (!gameObject.IsDestroyed())
                        StopCoroutine(battleTimerCoroutine);

                    Battle.LeftPlayer.SetSkillEnabled(false);
                    Battle.RightPlayer.SetSkillEnabled(false);
                    Battle.LeftPlayer.ClearInput();
                    Battle.RightPlayer.ClearInput();
                    StartCoroutine(ResetBattle());
                    break;
                case BattleState.Battle_Reset:
                    BattleWinner? winner = Battle.GetBattleWinner();
                    if (winner != null)
                    {
                        LogManager.SetGameWinner((BattleWinner)winner!);
                        LogManager.UpdateMetadata();
                        TransitionToState(BattleState.PostBattle_ShowResult);
                    }
                    else
                    {
                        LogManager.UpdateMetadata();
                        LogManager.SortAndSave();

                        int previousRound = Battle.CurrentRound.RoundNumber;
                        Battle.CurrentRound = new Round(previousRound + 1, Mathf.CeilToInt(BattleTime));
                        LogManager.StartRound(Battle.CurrentRound.RoundNumber);

                        Logger.Info($"CurrentRound.RoundNumber {Battle.CurrentRound.RoundNumber}");

                        TransitionToState(BattleState.Battle_Countdown);
                    }
                    break;

                // Post Battle
                case BattleState.PostBattle_ShowResult:
                    LogManager.SortAndSave();
                    RecordLeaderboardResult();
                    break;
            }

            BroadcastBattleData();
        }

        // Commits the finished match to the local leaderboards exactly once.
        // Batch simulations are excluded: they are analysis runs, not ladder play.
        private void RecordLeaderboardResult()
        {
            if (leaderboardRecorded)
                return;
            if (simulator != null && simulator.enabled)
                return;
            leaderboardRecorded = true;

            try
            {
                LastLeaderboardOutcome =
                    LeaderboardService.Instance.RecordBattle(LeaderboardRecordFactory.FromBattle(this));
            }
            catch (Exception ex)
            {
                Logger.Error($"[BattleManager] Failed to record leaderboard result: {ex.Message}");
            }
        }

        // Call this when we need to trigger OnBattleChanged immediately
        private void BroadcastBattleData()
        {
            EventParameter stateParam = new(
                battleStateParam: CurrentState);

            if (CurrentState == BattleState.Battle_End)
                stateParam.Winner = Battle.GetRoundWinner();

            Events[OnBattleChanged].Invoke(stateParam);
        }

        /// <summary>
        /// Applies host-authored battle metadata on a joining client without
        /// executing the local state machine or physics side effects.
        /// </summary>
        public void ApplyRemoteSnapshot(
            BattleState state,
            float elapsedTime,
            float countdownRemaining,
            int roundNumber,
            int leftWins,
            int rightWins,
            BattleWinner roundWinner)
        {
            if (!OnlineBattleSession.IsClient || Battle == null)
                return;

            // Scene messages can arrive before Start() has initialized the two
            // local controller objects. Do not notify UI listeners until both
            // controllers and their input providers are ready.
            if (!PrepareRemoteClient())
                return;

            bool stateChanged = CurrentState != state;
            // Snapshots are intentionally unreliable. If the one brief
            // Battle_Preparing snapshot is missed, the next countdown/ongoing
            // snapshot must still open and initialize the battle UI.
            bool synthesizePreparation = stateChanged && !remoteBattlePrepared &&
                state > BattleState.Battle_Preparing &&
                state <= BattleState.PostBattle_ShowResult &&
                (CurrentState == BattleState.PreBatle_Preparing ||
                 CurrentState == BattleState.PostBattle_ShowResult);
            ElapsedTime = Mathf.Max(0f, elapsedTime);
            CountdownRemaining = Mathf.Max(0f, countdownRemaining);

            if ((state == BattleState.Battle_Preparing && stateChanged) ||
                synthesizePreparation)
            {
                leaderboardRecorded = false;
                LastLeaderboardOutcome = null;
                Battle.ClearWinner();
                Battle.CurrentRound = new Round(Mathf.Max(1, roundNumber), Mathf.CeilToInt(BattleTime));
                remoteBattlePrepared = true;
            }
            else if (roundNumber > 0 &&
                     (Battle.CurrentRound == null || Battle.CurrentRound.RoundNumber != roundNumber))
            {
                Battle.CurrentRound = new Round(roundNumber, Mathf.CeilToInt(BattleTime));
            }

            Battle.LeftWinCount = Mathf.Max(0, leftWins);
            Battle.RightWinCount = Mathf.Max(0, rightWins);

            if (synthesizePreparation)
            {
                Logger.Info($"[Online] Client recovered missed Battle_Preparing snapshot before {state}.");
                CurrentState = BattleState.Battle_Preparing;
                Events[OnBattleChanged].Invoke(
                    new EventParameter(battleStateParam: BattleState.Battle_Preparing));
            }

            if (Battle.CurrentRound != null &&
                state >= BattleState.Battle_End &&
                state <= BattleState.PostBattle_ShowResult)
            {
                SumoController winnerController = roundWinner switch
                {
                    BattleWinner.Left => Battle.LeftPlayer,
                    BattleWinner.Right => Battle.RightPlayer,
                    _ => null
                };

                Battle.CurrentRound.RoundWinner = winnerController;
                Battle.Winners[Battle.CurrentRound.RoundNumber] = winnerController;
            }

            CurrentState = state;

            if (state == BattleState.PostBattle_ShowResult && stateChanged)
                RecordLeaderboardResult();

            if (state == BattleState.Battle_Countdown)
            {
                Events[OnCountdownChanged].Invoke(
                    new EventParameter(floatParam: Mathf.Ceil(CountdownRemaining)));
            }

            if (stateChanged)
            {
                if (OnlineBattleSession.IsClient)
                    Logger.Info($"[Online] Client battle state: {state}.");
                EventParameter stateParameter = new(battleStateParam: state);
                if (state == BattleState.Battle_End)
                    stateParameter.Winner = roundWinner;
                Events[OnBattleChanged].Invoke(stateParameter);
                OnlineBattleSession.RefreshBattleInputVisibility();
            }

            if (state == BattleState.PostBattle_ShowResult)
                remoteBattlePrepared = false;
        }

        public bool PrepareRemoteClient()
        {
            if (!OnlineBattleSession.IsClient || Battle == null ||
                Battle.LeftPlayer == null || Battle.RightPlayer == null ||
                InputManager.Instance == null)
            {
                return false;
            }

            if (!remoteInputsInitialized ||
                Battle.LeftPlayer.InputProvider == null || Battle.RightPlayer.InputProvider == null)
            {
                InputManager.Instance.InitializeInput(
                    Battle.LeftPlayer,
                    LeftInputType,
                    initializeSimulationSystems: false);
                InputManager.Instance.InitializeInput(
                    Battle.RightPlayer,
                    RightInputType,
                    initializeSimulationSystems: false);
                remoteInputsInitialized = true;
            }

            return Battle.LeftPlayer.InputProvider != null && Battle.RightPlayer.InputProvider != null;
        }

        /// <summary>
        /// Ends an online match immediately when the remote player leaves. This
        /// bypasses the normal round-reset delay and records the remaining player
        /// as the match winner on both peers.
        /// </summary>
        public bool FinishOnlineByForfeit(PlayerSide winnerSide)
        {
            if (!OnlineBattleSession.IsActive || Battle == null ||
                Battle.LeftPlayer == null || Battle.RightPlayer == null ||
                CurrentState == BattleState.PostBattle_ShowResult)
            {
                return false;
            }

            StopAllCoroutines();
            Battle.LeftPlayer.SetSkillEnabled(false);
            Battle.RightPlayer.SetSkillEnabled(false);
            Battle.LeftPlayer.ClearInput();
            Battle.RightPlayer.ClearInput();

            if (Battle.CurrentRound == null || Battle.CurrentRound.RoundNumber <= 0)
                Battle.CurrentRound = new Round(1, Mathf.CeilToInt(BattleTime));

            int winningThreshold = ((int)Battle.RoundSystem / 2) + 1;
            SumoController winner = winnerSide == PlayerSide.Left
                ? Battle.LeftPlayer
                : Battle.RightPlayer;
            Battle.ClearWinner();
            Battle.LeftWinCount = winnerSide == PlayerSide.Left ? winningThreshold : 0;
            Battle.RightWinCount = winnerSide == PlayerSide.Right ? winningThreshold : 0;
            Battle.CurrentRound.RoundWinner = winner;
            Battle.Winners[Battle.CurrentRound.RoundNumber] = winner;

            leaderboardRecorded = false;
            LastLeaderboardOutcome = null;
            CurrentState = BattleState.PostBattle_ShowResult;
            RecordLeaderboardResult();
            Events[OnBattleChanged].Invoke(
                new EventParameter(battleStateParam: BattleState.PostBattle_ShowResult));
            OnlineBattleSession.RefreshBattleInputVisibility();
            Logger.Info($"[Online] {winnerSide} wins by opponent forfeit.");
            return true;
        }
        #endregion
    }

    #region Battle and Round class
    [Serializable]
    public record Battle
    {
        public string BattleID;
        public RoundSystem RoundSystem;
        public SumoController LeftPlayer;
        public SumoController RightPlayer;
        public Round CurrentRound;

        public Dictionary<int, SumoController> Winners
        {
            get;
            private set;
        } = new Dictionary<int, SumoController>();

        public Dictionary<int, Round> Rounds = new();

        public int LeftWinCount;
        public int RightWinCount;

        public Battle(string battleID, RoundSystem roundSystem)
        {
            BattleID = battleID;
            RoundSystem = roundSystem;
        }

        public void SetRoundWinner(SumoController winner)
        {
            if (winner.Side == PlayerSide.Left)
                LeftWinCount += 1;
            else
                RightWinCount += 1;

            CurrentRound.RoundWinner = winner;
            Winners[CurrentRound.RoundNumber] = winner;
        }

        public BattleWinner GetRoundWinner(int? roundNumber = null)
        {
            if (Winners.TryGetValue(roundNumber ?? CurrentRound.RoundNumber, out SumoController winner) && winner != null)
            {
                if (winner.Side == PlayerSide.Left)
                    return BattleWinner.Left;
                else
                    return BattleWinner.Right;
            }
            return BattleWinner.Draw;
        }

        public BattleWinner? GetBattleWinner()
        {
            Logger.Info($"[Battle][GetBattleWinner] leftWinCount: {LeftWinCount}, rightWinCount: {RightWinCount}");

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
                    return BattleWinner.Left;
                else
                    return BattleWinner.Right;
            }

            // Check whether current round reaches max round
            if (CurrentRound.RoundNumber == (int)RoundSystem)
            {
                if (LeftWinCount == RightWinCount)
                    return BattleWinner.Draw;
                else if (LeftWinCount > RightWinCount)
                    return BattleWinner.Left;
                else
                    return BattleWinner.Right;
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
        public float FinishTime;
        public int RoundNumber = 0;
        public SumoController RoundWinner;
        public Round(int roundNumber, int time)
        {
            RoundNumber = roundNumber;
            FinishTime = time;
        }
    }

    public static class BattleExt
    {
        public static SumoController ToController(this BattleWinner? battleWinner, Battle battle)
        {
            switch (battleWinner)
            {
                case BattleWinner.Left:
                    return battle.LeftPlayer;
                case BattleWinner.Right:
                    return battle.RightPlayer;
                default:
                    return null;
            }
        }
    }
    #endregion
}
