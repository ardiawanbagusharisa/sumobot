using SumoCore;
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
                Left.OnBattleStateChanged(param.BattleState);
            }
            if (RightEnabled && Right != null)
            {
                Right.OnBattleStateChanged(param.BattleState);
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
                controller.AssignSkill(Left.SkillType);
                controller.Events[SumoController.OnBounce].Subscribe(Left.OnBotCollision);
                Left.SetProvider(controller.InputProvider);
                Left.OnBotInit(controller.InputProvider.API);
            }

            if (RightEnabled && Right != null && controller.Side == PlayerSide.Right)
            {
                controller.AssignSkill(Right.SkillType);
                controller.Events[SumoController.OnBounce].Subscribe(Right.OnBotCollision);
                Right.SetProvider(controller.InputProvider);
                Right.OnBotInit(controller.InputProvider.API);
            }
        }

        public void UnInit(SumoController controller)
        {
            if (!BotEnabled || !enabled)
                return;

            if (LeftEnabled && Left != null && controller.Side == PlayerSide.Left)
            {
                controller.Events[SumoController.OnBounce].Unsubscribe(Left.OnBotCollision);
            }

            if (RightEnabled && Right != null && controller.Side == PlayerSide.Right)
            {
                controller.Events[SumoController.OnBounce].Unsubscribe(Right.OnBotCollision);
            }
        }
    }
}