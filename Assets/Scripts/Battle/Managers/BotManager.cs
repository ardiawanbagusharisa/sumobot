using System.Collections;
using System.Collections.Generic;
using SumoCore;
using SumoInput;
using SumoManager;
using Unity.VisualScripting;
using UnityEditor.SceneManagement;
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
                Left.OnBotUpdate();

                if (leftConfig?.RunningCoroutine != null)
                {
                    StopCoroutine(leftConfig.RoutineFunc);
                }

                if (leftConfig.RoutineFunc != null)
                {
                    leftConfig.RunningCoroutine = StartCoroutine(leftConfig.RoutineFunc);
                }
            }

            if (RightEnabled && Right != null)
            {
                Right.OnBotUpdate();

            }
        }

        public void OnBattleStateChanged(EventParameter param)
        {
            if (!BotEnabled || !enabled)
                return;

            if (LeftEnabled && Left != null)
            {
                Left.OnBattleStateChanged(param.BattleState, param.Winner);
            }
            if (RightEnabled && Right != null)
            {
                Right.OnBattleStateChanged(param.BattleState, param.Winner);
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
            }

            if (RightEnabled && Right != null && controller.Side == PlayerSide.Right)
            {
                controller.Events[SumoController.OnBounce].Unsubscribe(OnRightBounce);
                Right.OnBotDestroy();
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

        public void Submit()
        {
            InputProvider.EnqueueCommand(Actions);
        }
    }
}