using System;
using System.Collections;
using System.Collections.Generic;
using SumoCore;
using SumoInput;
using SumoManager;
using UnityEngine;

namespace SumoBot
{
    public class BotManager : MonoBehaviour
    {
        public bool LeftEnabled = true;

        [HideInInspector]
        public Bot Left;

        public bool RightEnabled = true;

        [HideInInspector]
        public Bot Right;

        [HideInInspector] public int leftBotIndex = 0;
        [HideInInspector] public int rightBotIndex = 0;

        public bool BotEnabled => LeftEnabled || RightEnabled;

        private BotConfig leftConfig;
        private BotConfig rightConfig;


        private void OnEnable()
        {
            BattleManager.Instance.Events[BattleManager.OnBattleChanged].Subscribe(OnBattleStateChanged);
        }
        void Start()
        {
            if (!enabled)
                return;

            var allTypes = BotUtility.GetAllBotTypes();

            if (LeftEnabled && leftBotIndex >= 0 && leftBotIndex < allTypes.Count)
                Left = ScriptableObject.CreateInstance(allTypes[leftBotIndex]) as Bot;

            if (RightEnabled && rightBotIndex >= 0 && rightBotIndex < allTypes.Count)
                Right = ScriptableObject.CreateInstance(allTypes[rightBotIndex]) as Bot;
        }

        public void OnUpdate()
        {
            if (!BotEnabled || !enabled)
                return;

            if (LeftEnabled && Left != null)
            {
                leftConfig.IsOnUpdate = true;

                try
                {
                    Left.OnBotUpdate();
                }
                finally
                {
                    leftConfig.IsOnUpdate = false;
                }

                leftConfig.RunCoroutine(this);
            }

            if (RightEnabled && Right != null)
            {
                rightConfig.IsOnUpdate = true;

                try
                {
                    Right.OnBotUpdate();
                }
                finally
                {
                    rightConfig.IsOnUpdate = false;
                }

                rightConfig.RunCoroutine(this);
            }
        }

        public void OnBattleStateChanged(EventParameter param)
        {
            if (!BotEnabled || !enabled)
                return;

            if (LeftEnabled && Left != null)
            {
                Left.OnBattleStateChanged(param.BattleState, param.Winner);

                if (param.BattleState == BattleState.Battle_End)
                {
                    leftConfig.CleanCoroutine(this);
                }
            }

            if (RightEnabled && Right != null)
            {
                Right.OnBattleStateChanged(param.BattleState, param.Winner);

                if (param.BattleState == BattleState.Battle_End)
                {
                    rightConfig.CleanCoroutine(this);
                }
            }
        }

        public void Assign(Bot leftParam, Bot rightParam)
        {
            BattleManager instance = BattleManager.Instance;
            SumoController leftPlayer = instance.Battle.LeftPlayer;
            SumoController rightPlayer = instance.Battle.RightPlayer;

            UnInit(leftPlayer);
            UnInit(rightPlayer);

            Left = leftParam;
            Right = rightParam;

            if (instance.CurrentState >= BattleState.Battle_Preparing)
            {
                Init(leftPlayer);
                Init(rightPlayer);
            }
        }

        public void Init(SumoController controller)
        {
            if (!BotEnabled || !enabled)
                return;

            if (LeftEnabled && Left != null && controller.Side == PlayerSide.Left)
            {
                leftConfig = new()
                {
                    InputProvider = controller.InputProvider,
                    Actions = new(),
                };
                controller.AssignSkill(Left.SkillType);
                controller.Events[SumoController.OnBounce].Subscribe(OnLeftBounce);
                Left.Init(leftConfig);
                Left.OnBotInit(leftConfig.InputProvider.API);
            }

            if (RightEnabled && Right != null && controller.Side == PlayerSide.Right)
            {
                rightConfig = new()
                {
                    InputProvider = controller.InputProvider,
                    Actions = new(),
                };
                controller.AssignSkill(Right.SkillType);
                controller.Events[SumoController.OnBounce].Subscribe(OnRightBounce);
                Right.Init(rightConfig);
                Right.OnBotInit(rightConfig.InputProvider.API);
            }
        }

        public void UnInit(SumoController controller)
        {
            if (!BotEnabled || !enabled)
                return;

            if (LeftEnabled && Left != null && controller.Side == PlayerSide.Left)
            {
                controller.Events[SumoController.OnBounce].Unsubscribe(OnLeftBounce);
                Left.OnBotDestroy();
                leftConfig?.CleanCoroutine(this);
            }

            if (RightEnabled && Right != null && controller.Side == PlayerSide.Right)
            {
                controller.Events[SumoController.OnBounce].Unsubscribe(OnRightBounce);
                Right.OnBotDestroy();
                rightConfig?.CleanCoroutine(this);
            }
        }

        public void Swap()
        {
            if (!BotEnabled || !enabled)
                return;

            if (Left == null || Right == null)
            {
                return;
            }

            Assign(Right, Left);
        }

        void OnDestroy()
        {
            if (LeftEnabled && Left != null)
            {
                Left.OnBotDestroy();
            }

            if (RightEnabled && Right != null)
            {
                Right.OnBotDestroy();
            }
        }

        public void OnLeftBounce(EventParameter parameter)
        {
            Left.OnBotCollision(parameter.BounceEvent);
        }

        public void OnRightBounce(EventParameter parameter)
        {
            Right.OnBotCollision(parameter.BounceEvent);
        }
    }

    public class BotConfig
    {
        public InputProvider InputProvider;
        public Queue<ISumoAction> Actions;
        public IEnumerator RoutineFunc;
        public Coroutine RunningCoroutine;
        public bool CoroutineCrash;
        public bool IsOnUpdate = false;

        public void Submit()
        {
            InputProvider.EnqueueCommand(Actions);
        }
        public void Enqueue(ISumoAction action)
        {
            Actions.Enqueue(action);
        }

        public void RunCoroutine(MonoBehaviour runner)
        {
            if (RunningCoroutine != null)
            {
                runner.StopCoroutine(RunningCoroutine);
            }

            if (RoutineFunc != null)
            {
                RunningCoroutine = runner.StartCoroutine(SafeCoroutine(RoutineFunc));
            }
        }

        public void CleanCoroutine(MonoBehaviour runner)
        {
            CoroutineCrash = false;
            if (RunningCoroutine != null)
            {
                runner.StopCoroutine(RunningCoroutine);
            }
            RunningCoroutine = null;
            RoutineFunc = null;
        }

        private IEnumerator SafeCoroutine(IEnumerator coroutineFactory)
        {
            while (true)
            {
                object current;
                try
                {
                    if (!coroutineFactory.MoveNext())
                        break;

                    current = coroutineFactory.Current;
                }
                catch (Exception ex)
                {
                    Debug.LogError($"Exception during coroutine execution: {ex}");
                    CoroutineCrash = true;
                    yield break;
                }

                yield return current;
            }
        }
    }
}