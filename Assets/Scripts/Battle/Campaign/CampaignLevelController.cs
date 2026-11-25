using System;
using System.Collections;
using System.Linq;
using SumoBot;
using SumoCore;
using SumoInput;
using SumoManager;
using SumoServices;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace SumoCampaign
{
    public sealed class CampaignLevelController : MonoBehaviour
    {
        [SerializeField] private string defaultLevelId = "1-1";
        [SerializeField] private CampaignMissionView missionView;
        public CampaignLevelDefinition Level { get; private set; }

        private float delayQuiz = 3f;
		private BattleManager battle;
        private BotManager bots;
        private AlBot_FSM fsm;
        private bool finished;
        private bool successPending;
        private bool started;
        private bool actionMove;
        private bool actionTurn;
        private bool commandTurn;
        private bool commandAccelerate;
        private bool enteredEdge;
        private bool observedState;
        private int sequenceStep;
        private float turnDegrees;
        private float lastRotation;
        private Vector3 startPosition;

        private void Awake()
        {
            Level = CampaignCatalog.Selected ??
                CampaignCatalog.Levels.FirstOrDefault(item => item.Id == defaultLevelId);
            battle = GetComponent<BattleManager>();
            if (Level == null || battle == null)
            {
                Debug.LogError("Campaign level definition or BattleManager missing.", this);
                enabled = false;
                return;
            }
            battle.RequireExternalStartConfirmation = true;
        }

        private IEnumerator Start()
        {
            yield return null;
            bots = GetComponent<BotManager>();
            if (bots == null || missionView == null || battle.Battle.LeftPlayer == null ||
                battle.Battle.RightPlayer == null)
            {
                Debug.LogError("Campaign scene is missing its bot manager, mission view, or players.", this);
                yield break;
            }

            ConfigureBattle();
            missionView.Bind(this);
            missionView.HideCompletionSticker();
            battle.Events[BattleManager.OnBattleChanged].Subscribe(OnBattleChanged);
            battle.Battle.LeftPlayer.Events[SumoController.OnAction].Subscribe(OnPlayerAction);
            if (bots.Left is AlBot_FSM leftFsm)
            {
                fsm = leftFsm;
                fsm.DodgeOnAnyImpact = Level.Task == CampaignTask.FsmDodging ||
                    Level.Task == CampaignTask.FsmRecovering;
                fsm.StateChanged += OnFsmStateChanged;
            }
            missionView.Progress.text = Level.Instructions;
            missionView.Show($"CAMPAIGN {Level.Id}: {Level.Title}", Level.Instructions, true, false, false);
        }

        private void OnDisable()
        {
            Time.timeScale = 1f;
            if (battle != null)
            {
                battle.Events[BattleManager.OnBattleChanged].Unsubscribe(OnBattleChanged);
                if (battle.Battle?.LeftPlayer != null)
                    battle.Battle.LeftPlayer.Events[SumoController.OnAction].Unsubscribe(OnPlayerAction);
            }
            if (fsm != null) fsm.StateChanged -= OnFsmStateChanged;
        }

        private void ConfigureBattle()
        {
            battle.LeftInputType = Level.Input;
            battle.RightInputType = InputType.Script;
            battle.RoundSystem = Level.Rounds;
            battle.Battle.RoundSystem = Level.Rounds;
            battle.BattleTime = Level.BattleSeconds;
            battle.CountdownTime = 3f;

            var arenaCollider = battle.Arena.GetComponent<CircleCollider2D>();
            float currentRadius = arenaCollider.radius *
                Mathf.Abs(battle.Arena.transform.lossyScale.x);
            float ratio = Level.ArenaRadius / currentRadius;
            battle.Arena.transform.localScale *= ratio;
            battle.ArenaRadius = Level.ArenaRadius;
            Vector3 center = battle.Arena.transform.TransformPoint(arenaCollider.offset);
            var left = battle.Battle.LeftPlayer;
            var right = battle.Battle.RightPlayer;
            left.transform.position = center + Vector3.left * Level.ArenaRadius * Level.SpawnFraction;
            right.transform.position = center + Vector3.right * Level.ArenaRadius * Level.OpponentSpawnFraction;
            left.StartPosition = left.transform.position;
            right.StartPosition = right.transform.position;
            startPosition = left.transform.position;

            bots.enabled = true;
            bots.LeftEnabled = Level.Input == InputType.Script;
            bots.RightEnabled = true;
            bots.LeftActionInterval = 0f;
            bots.RightActionInterval = Level.OpponentInterval;
            foreach (InputProvider provider in FindObjectsByType<InputProvider>(
                         FindObjectsInactive.Include, FindObjectsSortMode.None))
                provider.IncludeKeyboard = Level.Input == InputType.UI &&
                    provider.PlayerSide == PlayerSide.Left;
            AssignBot(Level.PlayerBot, PlayerSide.Left, Level.Input == InputType.Script);
            AssignBot(Level.OpponentBot, PlayerSide.Right, true);
            HideOpponentUi();
            var pre = FindSceneObject("Pre-Battle");
            if (pre != null) pre.SetActive(false);
            var ongoing = FindSceneObject("Ongoing Battle");
            if (ongoing != null) ongoing.SetActive(true);
        }

        private void AssignBot(string id, PlayerSide side, bool required)
        {
            if (!required) return;
            Bot bot = BotUtility.GetAllBotInstances().FirstOrDefault(candidate =>
                string.Equals(candidate.ID, id, StringComparison.OrdinalIgnoreCase) ||
                string.Equals(candidate.GetType().Name, id, StringComparison.OrdinalIgnoreCase));
            if (bot == null)
            {
                Debug.LogError($"Campaign {Level.Id}: bot '{id}' not found.", this);
                return;
            }
            bots.Assign(bot, side);
        }

        public void StartMission()
        {
            if (started || finished) return;
            started = true;
            if (Level.Task == CampaignTask.FsmSequence) sequenceStep = 1;
            missionView.Hide();
            battle.ConfirmCampaignStart();
        }

        public void RetryMission()
        {
            Time.timeScale = 1f;
            CampaignCatalog.Launch(Level);
        }

        public void ContinueFromMissionPanel() => StartMission();

        public void ReturnToMenu()
        {
            Time.timeScale = 1f;
            CampaignCatalog.ClearSelection();
            SceneManager.LoadScene("MainMenu");
        }

        public void AnswerQuestion(int answer)
        {
            if (!started || finished || successPending || !observedState) return;
            if (answer == Level.CorrectAnswer)
            {
                missionView.Hide();
                Time.timeScale = 1f;
                Complete();
            }
            else Fail("That answer is incorrect. Review the bot's states and try again.");
        }

        private void Update()
        {
            if (!started || finished || successPending || battle.CurrentState != BattleState.Battle_Ongoing)
                return;
            var player = battle.Battle.LeftPlayer;
            var enemy = battle.Battle.RightPlayer;
            if (player == null || enemy == null) return;

            float radius = Level.ArenaRadius;
            float fromCenter = Vector2.Distance(player.transform.position, battle.Arena.transform.position);
            float distance = Vector2.Distance(player.transform.position, enemy.transform.position);
            float rotation = player.RigidBody.rotation;
            if (actionTurn || commandTurn)
                turnDegrees += Mathf.Abs(Mathf.DeltaAngle(lastRotation, rotation));
            lastRotation = rotation;
            if (fromCenter >= radius * .8f) enteredEdge = true;

            switch (Level.Task)
            {
                case CampaignTask.Move:
                case CampaignTask.CommandMove:
                    if (actionMove && Vector2.Distance(startPosition, player.transform.position) >= 2f)
                        Complete();
                    break;
                case CampaignTask.TurnAndApproach:
                    if (turnDegrees >= 90f && distance <= 3f) Complete();
                    break;
                case CampaignTask.CommandSequence:
                    if (commandTurn && commandAccelerate && distance <= 3f) Complete();
                    break;
                case CampaignTask.EdgeRecovery:
                    if (enteredEdge && fromCenter <= radius * .5f && !player.IsOutOfArena)
                        Complete();
                    break;
                case CampaignTask.CommandRecovery:
                    if (enteredEdge && commandTurn && commandAccelerate &&
                        fromCenter <= radius * .5f && !player.IsOutOfArena)
                        Complete();
                    break;
            }
            if (missionView.Progress != null)
            {
                string detail = Level.Task switch
                {
                    CampaignTask.Move or CampaignTask.CommandMove =>
                        $"Distance {Mathf.Min(2f, Vector2.Distance(startPosition, player.transform.position)):0.0}/2.0",
                    CampaignTask.TurnAndApproach => $"Turn {Mathf.Min(90f, turnDegrees):0}/90°; opponent {distance:0.0}/3.0 units",
                    CampaignTask.CommandSequence => $"Turn {(commandTurn ? "✓" : "○")} → Accelerate {(commandAccelerate ? "✓" : "○")}; opponent {distance:0.0}/3.0",
                    CampaignTask.EdgeRecovery or CampaignTask.CommandRecovery =>
                        enteredEdge ? $"Return to inner half: {fromCenter / radius:P0} of ring radius" : "Reach the outer 20% of the ring",
                    CampaignTask.FsmApproaching or CampaignTask.FsmAttacking or CampaignTask.FsmDodging or
                        CampaignTask.FsmRecovering or CampaignTask.FsmSequence =>
                        $"FSM state: {(fsm == null ? "Waiting" : fsm.CurrentState.ToString())}",
                    _ => Level.Instructions
                };
                missionView.Progress.text = $"{Level.Title}: {detail}";
            }
        }

        private void OnPlayerAction(EventParameter parameter)
        {
            if (!started || finished || successPending || !parameter.Bool || parameter.ActionList == null ||
                battle.CurrentState != BattleState.Battle_Ongoing)
                return;
            foreach (var action in parameter.ActionList)
            {
                bool command = action.InputUsed == InputType.LiveCommand;
                bool button = action.InputUsed == InputType.UI ||
                    action.InputUsed == InputType.Keyboard;
                if (action.Type == ActionType.Accelerate)
                {
                    if (button && Level.Task == CampaignTask.Move) actionMove = true;
                    if (command)
                    {
                        if (Level.Task == CampaignTask.CommandMove) actionMove = true;
                        if (commandTurn && (Level.Task != CampaignTask.CommandRecovery || enteredEdge))
                            commandAccelerate = true;
                    }
                }
                if (action.Type == ActionType.TurnLeft || action.Type == ActionType.TurnRight)
                {
                    if (button) actionTurn = true;
                    if (command && (Level.Task != CampaignTask.CommandRecovery || enteredEdge))
                        commandTurn = true;
                }
                if (action.Type == ActionType.Dash &&
                    ((button && Level.Task == CampaignTask.AlignedDash) ||
                     (command && Level.Task == CampaignTask.CommandDash)))
                {
                    var player = battle.Battle.LeftPlayer;
                    var enemy = battle.Battle.RightPlayer;
                    Vector2 toward = enemy.transform.position - player.transform.position;
                    if (toward.magnitude <= 4f &&
                        Vector2.Angle(player.transform.up, toward) <= 30f)
                        Complete();
                }
            }
        }

        private void OnFsmStateChanged(BotState state)
        {
            if (!started || finished || successPending) return;
            if (Level.Task == CampaignTask.FsmSequence)
            {
                if (state == BotState.Searching) sequenceStep = 1;
                else if (state == BotState.Approaching && sequenceStep >= 1) sequenceStep = 2;
                else if (state == BotState.Attacking && sequenceStep >= 2) ShowQuiz();
                return;
            }
            bool target = (Level.Task == CampaignTask.FsmApproaching && state == BotState.Approaching) ||
                (Level.Task == CampaignTask.FsmAttacking && state == BotState.Attacking) ||
                (Level.Task == CampaignTask.FsmDodging && state == BotState.Dodging) ||
                (Level.Task == CampaignTask.FsmRecovering && state == BotState.Recovering);
            if (target) ShowQuiz();
        }

        private void ShowQuiz()
        {
            if (observedState) return;
            observedState = true;
            StartCoroutine(ShowQuizAfterObservation());
        }

        private IEnumerator ShowQuizAfterObservation()
        {
            yield return new WaitForSeconds(delayQuiz);
            if (finished || successPending) yield break;
            Time.timeScale = 0f;
            missionView.Show($"CAMPAIGN {Level.Id}: OBSERVE", Level.Question, false, false, true);
        }

        private void OnBattleChanged(EventParameter parameter)
        {
            if (!started || finished || successPending || observedState) return;
            if (parameter.BattleState == BattleState.Battle_Ongoing &&
                Level.Task == CampaignTask.FsmRecovering && fsm != null)
            {
                StartCoroutine(BeginEdgeRecoveryDemo());
                return;
            }
            if (parameter.BattleState != BattleState.Battle_End &&
                parameter.BattleState != BattleState.PostBattle_ShowResult) return;
            bool ringOut = Level.Task == CampaignTask.RingOut ||
                Level.Task == CampaignTask.CommandRingOut;
            if (ringOut && battle.Battle.GetRoundWinner() == BattleWinner.Left)
                Complete();
            else
                Fail("The task was not completed before the round ended.");
        }

        private IEnumerator BeginEdgeRecoveryDemo()
        {
            yield return null;
            if (battle.CurrentState == BattleState.Battle_Ongoing && !finished)
                fsm.BeginEdgeRecoveryDemonstration();
        }

        private void Complete()
        {
            if (finished || successPending) return;
            successPending = true;
            missionView.ShowCompletionSticker();
            _ = SaveCompletionAsync();
            StartCoroutine(FinishSuccessAfterDelay());
        }

        private IEnumerator FinishSuccessAfterDelay()
        {
            yield return new WaitForSecondsRealtime(5f);
            if (finished) yield break;
            finished = true;
            missionView.HideCompletionSticker();
            PauseBattle();
            missionView.Show($"CAMPAIGN {Level.Id} COMPLETE", "Task complete! Your progress has been saved.", false, false, false);
        }

        private void Fail(string reason)
        {
            if (finished) return;
            finished = true;
            PauseBattle();
            missionView.Show($"CAMPAIGN {Level.Id}: TRY AGAIN", reason, false, true, false);
        }

        private void PauseBattle()
        {
            battle.StopAllCoroutines();
            battle.enabled = false;
            Time.timeScale = 0f;
        }

        private async System.Threading.Tasks.Task SaveCompletionAsync()
        {
            var data = GameServices.PlayerData?.Current;
            if (data == null) return;
            if (data.CompletedCampaignLevels.Contains(Level.Id)) return;
            data.CompletedCampaignLevels.Add(Level.Id);
            ServiceResult result = await GameServices.PlayerData.SaveAsync();
            if (!result.Success)
            {
                data.CompletedCampaignLevels.Remove(Level.Id);
                Debug.LogError($"Campaign {Level.Id} save failed: {result.Error}", this);
                return;
            }
            result = await GameServices.PlayerData.AddCoinsAsync(Level.FirstWinCoins);
            if (!result.Success)
                Debug.LogError($"Campaign {Level.Id} reward failed: {result.Error}", this);
        }

        private static GameObject FindSceneObject(string name)
        {
            foreach (Transform candidate in FindObjectsByType<Transform>(
                         FindObjectsInactive.Include, FindObjectsSortMode.None))
                if (candidate.name == name) return candidate.gameObject;
            return null;
        }

        private static void HideOpponentUi()
        {
            GameObject ui = FindSceneObject("UI");
            if (ui == null) return;
            string[] paths = {
                "Pre-Battle/Players/PlayerHUD 2",
                "Ongoing Battle/Footer/RightSide",
                "Ongoing Battle/Body/Player_Indicators/Right",
                "Post Battle/Menu/Players/PlayerHUD_Right",
                "Post Battle/Menu/PlayersFinalScores/Right"
            };
            foreach (string path in paths)
            {
                Transform child = ui.transform.Find(path);
                if (child != null) child.gameObject.SetActive(false);
            }
        }
    }
}
