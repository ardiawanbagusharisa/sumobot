using Mono.Cecil.Cil;
using SumoCore;
using SumoManager;
using Unity.Collections;
using UnityEngine;

namespace SumoBot
{
    public class BotManager : MonoBehaviour
    {
        public bool LeftEnabled = true;
        [ReadOnly] public Bot Left;

        public bool RightEnabled = true;
        [ReadOnly] public Bot Right;

        [HideInInspector] public int leftBotIndex = 0;
        [HideInInspector] public int rightBotIndex = 0;

        public bool IsEnable => LeftEnabled || RightEnabled;

        void Start()
        {
            var allTypes = BotUtility.GetAllBotTypes();

            if (LeftEnabled && leftBotIndex >= 0 && leftBotIndex < allTypes.Count)
                Left = ScriptableObject.CreateInstance(allTypes[leftBotIndex]) as Bot;

            if (RightEnabled && rightBotIndex >= 0 && rightBotIndex < allTypes.Count)
                Right = ScriptableObject.CreateInstance(allTypes[rightBotIndex]) as Bot;
        }

        public void OnUpdate(float ElapsedTime)
        {
            if (!IsEnable)
                return;

            if (LeftEnabled && Left != null)
            {
                Left.ElapsedTime = ElapsedTime;
                if (Left.ElapsedTime >= Left.Interval)
                {
                    Left.ElapsedTime = 0;
                    Left.OnBotUpdate();
                }
            }

            if (RightEnabled && Right != null)
            {
                Right.ElapsedTime = ElapsedTime;
                if (Right.ElapsedTime >= Right.Interval)
                {
                    Right.ElapsedTime = 0;
                    Right.OnBotUpdate();
                }
            }
        }


        public void OnBattleStateChanged(BattleState currState)
        {
            if (!IsEnable)
                return;

            if (LeftEnabled && Left != null)
            {
                Left.OnBattleStateChanged(currState);
            }
            if (LeftEnabled && Right != null)
            {
                Right.OnBattleStateChanged(currState);
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
            if (!IsEnable)
                return;

            if (LeftEnabled && Left != null && controller.Side == PlayerSide.Left)
            {
                controller.AssignSkill(Left.SkillType);
                controller.Actions[SumoController.OnBounce].Subscribe(Left.OnBotCollision);
                Left.SetProvider(controller.InputProvider);
                Left.OnBotInit(controller.Side, controller.InputProvider.API);
            }

            if (RightEnabled && Right != null && controller.Side == PlayerSide.Right)
            {
                controller.AssignSkill(Right.SkillType);
                controller.Actions[SumoController.OnBounce].Subscribe(Right.OnBotCollision);
                Right.SetProvider(controller.InputProvider);
                Right.OnBotInit(controller.Side, controller.InputProvider.API);
            }
        }

        public void UnInit(SumoController controller)
        {
            if (!IsEnable)
                return;

            if (LeftEnabled && Left != null && controller.Side == PlayerSide.Left)
            {
                controller.Actions[SumoController.OnBounce].Unsubscribe(Left.OnBotCollision);
            }

            if (RightEnabled && Right != null && controller.Side == PlayerSide.Right)
            {
                controller.Actions[SumoController.OnBounce].Unsubscribe(Right.OnBotCollision);
            }
        }
    }
}