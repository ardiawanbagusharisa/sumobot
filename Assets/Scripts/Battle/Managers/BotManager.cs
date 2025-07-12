using System;
using SumoCore;
using SumoManager;
using UnityEngine;

namespace SumoBot
{
    public class BotManager : MonoBehaviour
    {
        public bool IsEnable;
        public bool IsScriptable = true;

        public Bot Left;
        public Bot Right;

        [HideInInspector] public int leftBotIndex = 0;
        [HideInInspector] public int rightBotIndex = 0;

        void Start()
        {
            if (!IsScriptable) return;

            var allTypes = BotUtility.GetAllBotTypes();

            if (leftBotIndex >= 0 && leftBotIndex < allTypes.Count)
                Left = ScriptableObject.CreateInstance(allTypes[leftBotIndex]) as Bot;

            if (rightBotIndex >= 0 && rightBotIndex < allTypes.Count)
                Right = ScriptableObject.CreateInstance(allTypes[rightBotIndex]) as Bot;
        }

        public void OnUpdate(float ElapsedTime)
        {
            if (!IsEnable) return;

            if (Left != null)
            {
                Left.ElapsedTime = ElapsedTime;
                if (Left.ElapsedTime >= Left.Interval)
                {
                    Left.ElapsedTime = 0;
                    Left.OnBotUpdate();
                }
            }

            if (Right != null)
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
            if (!IsEnable) return;

            if (Left != null)
            {
                Left.OnBattleStateChanged(currState);
            }
            if (Right != null)
            {
                Right.OnBattleStateChanged(currState);
            }
        }
    }
}