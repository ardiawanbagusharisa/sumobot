using System;
using System.Collections;
using System.Collections.Generic;
using CoreSumo;
using Unity.VisualScripting;
using UnityEngine;

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
    public InputType BattleInputType = InputType.UI;
    public RoundSystem RoundSystem = RoundSystem.BestOf3;
    public float BattleTime = 60f;
    public float CountdownTime = 3f;
    public List<Transform> StartPositions = new List<Transform>();
    public GameObject SumoPrefab;
    public GameObject LeftPlayerObject;
    public GameObject RightPlayerObject;
    public GameObject Arena;
    #endregion

    #region Runtime (readonly) properties 
    public BattleState CurrentState = BattleState.PreBatle_Preparing;
    public float ElapsedTime = 0;
    public float TimeLeft => BattleTime - ElapsedTime;
    public Battle Battle;
    public Round CurrentRound = null;
    #endregion
    
    #region Events properties 
    public event Action<float> OnCountdownChanged;
    public event Action<Battle> OnBattleChanged;
    private Coroutine battleTimerCoroutine;
    private Coroutine countdownCoroutine;
    #endregion

    #region Unity methods 
    private void Awake()
    {
        if (Instance != null)
        {
            Destroy(gameObject);
            return;
        }
        Instance = this;
    }

    void OnEnable()
    {
        LogManager.InitLog();
        Battle = new Battle(Guid.NewGuid().ToString(), RoundSystem);
        LogManager.InitBattle();
    }

    void Start()
    {
        TransitionToState(BattleState.PreBatle_Preparing);
    }

    void OnDisable()
    {
        Battle.LeftPlayer.OnPlayerOutOfArena -= OnPlayerOutOfArena;
        Battle.RightPlayer.OnPlayerOutOfArena -= OnPlayerOutOfArena;
    }

    void Update()
    {
        if (CurrentRound != null && CurrentState == BattleState.Battle_Ongoing)
        {
            ElapsedTime += Time.deltaTime;
        }
    }
    #endregion

    #region API methods
    public void SetLeftDefaultSkill(int type)
    {
        Battle.LeftPlayer.Skill.Type = type == 0 ? SkillType.Boost : SkillType.Stone;
    }

    public void SetRightDefaultSkill(int type)
    {
        Battle.RightPlayer.Skill.Type = type == 0 ? SkillType.Boost : SkillType.Stone;
    }

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

    #region Core Logic methods 
    private void InitializePlayer(SumoController controller)
    {
        PlayerSide side = controller.transform.position.x < 0 ? PlayerSide.Left : PlayerSide.Right;
        controller.Initialize(side, controller.transform);
        controller.OnPlayerOutOfArena += OnPlayerOutOfArena;

        if (controller.Side == PlayerSide.Left)
            Battle.LeftPlayer = controller;
        else
            Battle.RightPlayer = controller;

        LogManager.LogBattleState(
                actor: LogActorType.System,
                data: new Dictionary<string, object>()
                {
                        {"type", "Player"},
                        {"outPlayerSide", controller.Side.ToLogActorType()},
                        {"skill", controller.Skill.Type.ToString()},
                });
        Debug.Log($"Player registered: {side}");
    }

    IEnumerator AllPlayersReady()
    {
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
        //Debug.Log("Battle starting in...");
        yield return new WaitForSeconds(1f);

        float timer = CountdownTime;
        while (timer > 0 && CurrentState == BattleState.Battle_Countdown)
        {
            //Debug.Log(Mathf.Ceil(timer));
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
            yield return new WaitForSeconds(1f);
            timer -= 1f;
        }

        LogManager.SetRoundWinner("Draw");
        CurrentRound.RoundWinner = null;
        Battle.Winners[CurrentRound.RoundNumber] = null;
        LogManager.CleanIncompletePlayerAction();
        TransitionToState(BattleState.Battle_End);
    }

    private IEnumerator ResetBattle()
    {
        yield return new WaitForSeconds(3f);
        Battle.LeftPlayer.Reset();
        Battle.RightPlayer.Reset();
        TransitionToState(BattleState.Battle_Reset);
        yield return new WaitForSeconds(1f);
    }

    private void OnPlayerOutOfArena(PlayerSide outPlayerSide)
    {
        if (CurrentState != BattleState.Battle_Ongoing) 
            return;
        Debug.Log("OnPlayerOutOfArena");

        SumoController winner = outPlayerSide == PlayerSide.Left ? Battle.RightPlayer : Battle.LeftPlayer;

        if (winner == null)
        {
            Debug.LogWarning("Winner not found!");
            return;
        }

        Battle.SetRoundWinner(winner);
        LogManager.CleanIncompletePlayerAction();
        TransitionToState(BattleState.Battle_End);
    }

    private void TransitionToState(BattleState newState)
    {
        Debug.Log($"State Transition: {CurrentState} → {newState}");
        CurrentState = newState;

        if (CurrentRound.RoundNumber == 0)
        {
            LogManager.LogBattleState(
                actor: LogActorType.System,
                data: new Dictionary<string, object>()
                {
                    {"type", "battle_state"},
                    {"state", CurrentState.ToString()},
                });
        }
        else
        {
            LogManager.LogBattleState(
                actor: LogActorType.System,
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
                    InitializePlayer(LeftPlayerObject.GetComponent<SumoController>());
                    InitializePlayer(RightPlayerObject.GetComponent<SumoController>());
                }
                else if (SumoPrefab != null && StartPositions.Count > 0)
                {
                    StartPositions.ForEach(pos =>
                    {
                        GameObject player = Instantiate(SumoPrefab, pos.position, pos.rotation);
                        InitializePlayer(player.GetComponent<SumoController>());
                    });
                }
                break;

            // Battle
            case BattleState.Battle_Preparing:
                LogManager.UpdateMetadata();
                LogManager.StartGameLog();

                Battle.ClearWinner();
                CurrentRound = new Round(1, Mathf.CeilToInt(BattleTime));
                LogManager.StartRound(CurrentRound.RoundNumber);

                Battle.LeftPlayer.Reset();
                Battle.RightPlayer.Reset();
                InputManager.Instance.InitializeInput(Battle.LeftPlayer);
                InputManager.Instance.InitializeInput(Battle.RightPlayer);

                LogManager.SetPlayerAction();
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
                Battle.LeftPlayer.SetMovementEnabled(true);
                Battle.RightPlayer.SetSkillEnabled(true);
                Battle.RightPlayer.SetMovementEnabled(true);
                break;
            case BattleState.Battle_End:
                CurrentRound.FinishTime = ElapsedTime;

                if (!gameObject.IsDestroyed())
                    StopCoroutine(battleTimerCoroutine);

                Battle.LeftPlayer.SetSkillEnabled(false);
                Battle.LeftPlayer.SetMovementEnabled(false);
                Battle.RightPlayer.SetSkillEnabled(false);
                Battle.RightPlayer.SetMovementEnabled(false);
                StartCoroutine(ResetBattle());
                break;
            case BattleState.Battle_Reset:
                var winner = Battle.GetBattleWinner();
                if (winner != null)
                {
                    LogManager.SetGameWinner((BattleWinner)winner!);
                    LogManager.UpdateMetadata();
                    TransitionToState(BattleState.PostBattle_ShowResult);
                }
                else
                {
                    LogManager.UpdateMetadata();

                    int previousRound = Battle.CurrentRound.RoundNumber;
                    CurrentRound = new Round(previousRound + 1, Mathf.CeilToInt(BattleTime));
                    LogManager.StartRound(CurrentRound.RoundNumber);

                    Debug.Log($"CurrentRound.RoundNumber {CurrentRound.RoundNumber}");

                    TransitionToState(BattleState.Battle_Countdown);
                }
                LogManager.SortAndSave();
                break;

            // Post Battle
            case BattleState.PostBattle_ShowResult:
                LogManager.SortAndSave();

                Deinitialize();
                break;
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

    public void SetRoundWinner(SumoController winner)
    {
        if (winner.Side == PlayerSide.Left)
            LeftWinCount += 1;
        else
            RightWinCount += 1;

        LogManager.SetRoundWinner(winner.Side.ToString());

        CurrentRound.RoundWinner = winner;
        Winners[CurrentRound.RoundNumber] = winner;
    }

    public BattleWinner? GetBattleWinner()
    {
        Debug.Log($"[Battle][GetBattleWinner] leftWinCount: {LeftWinCount}, rightWinCount: {RightWinCount}");

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
    public int Id;
    public float FinishTime;
    public int RoundNumber = 0;
    public SumoController RoundWinner;
    public Round(int roundNumber, int time)
    {
        RoundNumber = roundNumber;
        FinishTime = time;
    }
}
#endregion